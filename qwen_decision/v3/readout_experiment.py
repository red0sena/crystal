"""Compare LM-code, independent linear/MLP and shared candidate readouts.

Stages: prepare -> screen -> tune -> confirm. Only train/selection are read by
optimisation. A fresh document holdback is scored after models are frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from qwen_decision.v3 import architecture_search as search
from qwen_decision.v3 import pointer, pointer_experiment as pilot, readout, train

RATES = {'code': (1e-5, 1e-4), 'linear': (.001, .003), 'mlp': (.0001, .001),
         'warm_mlp': (1e-5, .0001), 'option_linear': (.001, .01), 'option_mlp': (.0001, .001)}
SEEDS = (41, 73)
BATCH_ARGS = SimpleNamespace(pack=False, micro_batch=4, token_budget=4096)


def read_json(path):
    return json.loads(Path(path).read_text())


def group_keys(row):
    state = row['state'] if row['state'] not in (None, '', {}) else row['question']
    normalized = ' '.join(json.dumps(state, sort_keys=True, ensure_ascii=False).split())
    return search.keys(row) + [('normalized', search.digest(normalized))]


def document_groups(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in group_keys(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    names = {}
    for i, row in enumerate(rows):
        g = find(i)
        names[g] = min(names.get(g, row['id']), row['id'])
    return {r['id']: names[find(i)] for i, r in enumerate(rows)}


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def save_head(head, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(path))


@torch.no_grad()
def make_cache(model, encoded, pad_id, path):
    n = len(encoded)
    hidden = model.config.text_config.hidden_size
    lengths = torch.tensor([len(r['target']) for r in encoded], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    width = int(lengths.max())
    cache = {'base': torch.empty(n, hidden, dtype=torch.bfloat16),
             'options': torch.empty(int(lengths.sum()), hidden, dtype=torch.bfloat16),
             'lengths': lengths, 'offsets': offsets, 'columns': torch.zeros(n, width, dtype=torch.long),
             'targets': torch.zeros(n, width)}
    lookup = {r['id']: i for i, r in enumerate(encoded)}
    started, done = time.monotonic(), 0
    for group in train.microbatches(encoded, BATCH_ARGS):
        batch = pointer.collate(group, pad_id, 'cuda')
        base, options = readout.vectors(model.model.language_model, batch)
        base, options = base.cpu(), options.cpu()
        for j, row in enumerate(group):
            i, k = lookup[row['id']], len(row['target'])
            lo = int(offsets[i])
            cache['base'][i] = base[j]
            cache['options'][lo:lo+k] = options[j, :k]
            cache['columns'][i, :k] = torch.tensor(row['code_columns'])
            cache['targets'][i, :k] = torch.tensor(row['target'])
        done += len(group)
        if done % 1024 < len(group):
            print(f'cache {path.stem}: {done}/{n}, {time.monotonic()-started:.0f}s', flush=True)
    torch.save(cache, path)
    return cache


def prepare(args):
    out = args.output
    if out.exists():
        raise SystemExit('prepare needs a new output directory')
    old = args.previous
    old_manifest = read_json(old/'manifest.json')
    renderer, config = search.renderer_for(Path(old_manifest['args']['base']))
    raw_dev = pilot.read_rows(Path('data/mix/v3-gold3.dev.jsonl'))
    lookup = {r['id']: r for r in raw_dev}
    groups = document_groups(raw_dev)
    selection_previous = search.read_rows(old, 'selection')
    exposed_ids = {r['id'] for split in ('selection', 'confirm') for r in search.read_rows(old, split)}
    exposed_ids.update(r['id'] for r in rows('runs/pointer/g4b-gold3-head100-tune20-s41/code/frozen/dev-step-0.jsonl'))
    exposed_ids.update(r['id'] for r in rows('runs/inputs/search-v2/confirmation-ids.jsonl'))
    exposed_groups = {groups[i] for i in exposed_ids if i in groups}
    selected_raw = [lookup[r['id']] for r in selection_previous]
    def encode(example):
        return {**renderer(example), 'group': groups.get(example['id'], example['id'])}
    selection = [encode(r) for r in selected_raw]
    if any(a['input_ids'] != b['input_ids'] or a['target'] != b['target'] for a,b in zip(selection,selection_previous)):
        raise ValueError('fixed selection rendering changed')
    by_source = defaultdict(list)
    for r in raw_dev:
        if groups[r['id']] not in exposed_groups:
            by_source[r['source']].append(r)
    confirm_raw, confirm = [], []
    for source, examples in sorted(by_source.items()):
        chosen = 0
        for r in sorted(examples, key=lambda r: search.digest(['readout-v1', r['id']])):
            encoded = encode(r)
            if len(encoded['input_ids']) <= args.max_tokens:
                confirm_raw.append(r)
                confirm.append(encoded)
                chosen += 1
            if chosen == args.confirm_per_source:
                break
    if not confirm:
        raise ValueError('no unexposed document holdback remains')
    reserved_keys = {k for r in selected_raw + confirm_raw for k in group_keys(r)}
    raw_train = pilot.read_rows(Path('data/mix/v3-gold3.jsonl'), args.raw_train_rows)
    training, training_raw, dropped = [], [], []
    for r in raw_train:
        if reserved_keys.intersection(group_keys(r)):
            dropped.append(r['id'])
            continue
        encoded = encode(r)
        if len(encoded['input_ids']) <= args.max_tokens:
            training.append(encoded)
            training_raw.append(r)
    pilot.check_disjoint({'train':training_raw, 'selection':selected_raw, 'confirmation':confirm_raw})
    exposed_keys = {k for i in exposed_ids if i in lookup for k in group_keys(lookup[i])}
    confirmation_keys = {k for r in confirm_raw for k in group_keys(r)}
    if exposed_keys.intersection(confirmation_keys):
        raise ValueError('confirmation overlaps a previous experiment')
    out.mkdir(parents=True)
    for split, encoded in [('train',training),('selection',selection),('confirmation',confirm)]:
        search.save_records(out/f'{split}.jsonl', encoded)
    manifest = {'base':old_manifest['args']['base'], 'base_manifest':old_manifest, 'config':config,
                'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'counts':{'train':len(training),'selection':len(selection),'confirmation':len(confirm)},
                'sources':{'train':len({r['source'] for r in training}),
                           'confirmation':len({r['source'] for r in confirm})},
                'confirmation_groups':len({r['group'] for r in confirm}),
                'excluded_train_overlap':dropped, 'prior_exposed_ids':len(exposed_ids),
                'shared_confirmation_keys':0, 'rates':RATES, 'seeds':SEEDS,
                'holdback':'new documents for this head experiment; base historically used full gold3 dev',
                'training':'same fixed gold3 rendered examples, no augmentation/teacher; cached head stage then last-two-layer tuning',
                'encoded_sha256':{split:train.sha256_file(out/f'{split}.jsonl')
                                  for split in ('train','selection','confirmation')},
                'sha256':{str(p):train.sha256_file(p) for p in [Path('data/mix/v3-gold3.jsonl'),Path('data/mix/v3-gold3.dev.jsonl')]}}
    train.atomic_json(out/'manifest.json', manifest)
    print('PREPARED',json.dumps(manifest['counts']),flush=True)
    model = search.load_model(old_manifest).requires_grad_(False)
    weight = model.get_output_embeddings().weight[renderer.code_ids].detach().float().cpu()
    torch.save(weight,out/'code-weight.pt')
    make_cache(model, training, renderer.tokenizer.pad_token_id, out/'train.pt')
    make_cache(model, selection, renderer.tokenizer.pad_token_id, out/'selection.pt')
    train.atomic_json(out/'prepared.json', {'complete':True})


def gpu_cache(path):
    return {k:v.cuda() for k,v in torch.load(path,map_location='cpu',weights_only=True).items()}


@torch.no_grad()
def evaluate_cache(head, cache, encoded):
    head.eval()
    result = []
    for start in range(0,len(encoded),128):
        batch = readout.cache_batch(cache,range(start,min(start+128,len(encoded))),head.uses_options)
        z = readout.cached_logits(head,batch)
        pred = z.argmax(-1)
        correct = batch['targets'].gather(1,pred[:,None])[:,0] >= batch['targets'].max(-1).values-1e-6
        for row,p,c in zip(encoded[start:start+128],pred.tolist(),correct.tolist()):
            result.append({k:row[k] for k in ('id','source','group')} | {'prediction':p,'correct':c})
    return search.metrics(result),result


def build(spec, weight):
    head = readout.DecisionReadout(spec['kind'],weight,inner=512).cuda()
    if 'directory' in spec:
        head.load_state_dict(load_file(str(Path(spec['directory'])/'head.safetensors')))
    return head


def screen(args,manifest):
    out=args.output
    if not (out/'prepared.json').exists():
        raise SystemExit('prepare incomplete')
    if (out/'screen-frozen.json').exists():
        raise SystemExit('screen already frozen')
    hashes={name:train.sha256_file(Path(__file__).with_name(name)) for name in ('readout.py','readout_experiment.py')}
    if (out/'implementation.json').exists() and read_json(out/'implementation.json')!=hashes:
        raise ValueError('implementation changed during screening')
    train.atomic_json(out/'implementation.json',hashes)
    training,selection = gpu_cache(out/'train.pt'),gpu_cache(out/'selection.pt')
    selected_rows=rows(out/'selection.jsonl')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    baseline,_=evaluate_cache(build({'kind':'code'},weight),selection,selected_rows)
    results=[]
    for kind, rates in RATES.items():
        for lr in rates:
            for seed in SEEDS:
                destination=out/'heads'/f'{kind}-lr{lr:g}-s{seed}'
                if (destination/'result.json').exists():
                    results.append(read_json(destination/'result.json'))
                    continue
                destination.mkdir(parents=True,exist_ok=True)
                train.seed_all(seed)
                head=build({'kind':kind},weight)
                optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.)
                score,_=evaluate_cache(head,selection,selected_rows)
                best,best_step=score['macro'],0
                history={'0':score}
                save_head(head,destination/'head.safetensors')
                started=time.monotonic()
                block_loss=torch.zeros((),device='cuda')
                block_count=0
                schedule=train.batches(len(training['base']),128,args.head_steps,random.Random(seed))
                for step,(indices,_) in enumerate(schedule,1):
                    head.train()
                    batch=readout.cache_batch(training,indices,head.uses_options)
                    optimizer.zero_grad()
                    loss=pointer.soft_ce(readout.cached_logits(head,batch),batch['targets']).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite head loss')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite head gradient')
                    optimizer.step()
                    block_loss+=loss.detach()
                    block_count+=1
                    if step%128==0 or step==args.head_steps:
                        score,_=evaluate_cache(head,selection,selected_rows)
                        history[str(step)]={**score,'mean_train_loss':float(block_loss/block_count),
                                            'last_grad_norm':float(norm)}
                        block_loss.zero_();block_count=0
                        if score['macro']>best:
                            best,best_step=score['macro'],step
                            save_head(head,destination/'head.safetensors')
                head.load_state_dict(load_file(str(destination/'head.safetensors')))
                score,records=evaluate_cache(head,selection,selected_rows)
                result={'kind':kind,'lr':lr,'seed':seed,'directory':str(destination),'best_step':best_step,
                        'selection':score,'steps':args.head_steps,'batch':128,
                        'trainable_parameters':sum(p.numel() for p in head.parameters()),
                        'seconds':time.monotonic()-started}
                train.atomic_json(destination/'result.json',result)
                train.atomic_json(destination/'history.json',history)
                search.save_records(destination/'selection.jsonl',records)
                results.append(result)
                print(f'HEAD {kind} lr={lr:g} seed={seed}: {score["macro"]:.5f}, step={best_step}',flush=True)
    grouped=defaultdict(list)
    for result in results:
        grouped[(result['kind'],result['lr'])].append(result)
    ranking=sorted([{'kind':kind,'lr':lr,'mean_macro':sum(r['selection']['macro'] for r in runs)/len(runs),'runs':runs}
                    for (kind,lr),runs in grouped.items()],key=lambda r:-r['mean_macro'])
    winner=next(r for r in ranking if r['kind']!='code')
    linear=next(r for r in ranking if r['kind'] in ('linear','option_linear'))
    mlp=next(r for r in ranking if r['kind'] in ('mlp','warm_mlp','option_mlp'))
    control=next(r for r in ranking if r['kind']=='code')
    frozen={'baseline':baseline,'ranking':ranking,'candidate':winner,'readout':linear,'mlp':mlp,'control':control,
            'tail_layers':2,'tail_steps':args.tune_steps,'tail_batch':16,'tail_lr':1e-6,
            'rule':'freeze best linear readout and best MLP separately by two-seed mean selection; tune both and matched code control; confirm every seed and both stages'}
    train.atomic_json(out/'screen-frozen.json',frozen)
    print('SCREEN FROZEN',json.dumps({'candidate':winner['kind'],'lr':winner['lr'],'mean_macro':winner['mean_macro']}),flush=True)


@torch.no_grad()
def evaluate_live(model,head,encoded,pad_id):
    model.eval(); head.eval()
    result=[]
    for group in train.microbatches(encoded,BATCH_ARGS):
        batch=pointer.collate(group,pad_id,'cuda')
        z=readout.live_logits(model,head,batch)
        pred=z.argmax(-1)
        correct=batch['targets'].gather(1,pred[:,None])[:,0]>=batch['targets'].max(-1).values-1e-6
        for row,p,c in zip(group,pred.tolist(),correct.tolist()):
            result.append({k:row[k] for k in ('id','source','group')}|{'prediction':p,'correct':c})
    return search.metrics(result),result


def tune(args,manifest):
    out=args.output
    if (out/'models-frozen.json').exists():
        raise SystemExit('models already frozen')
    frozen=read_json(out/'screen-frozen.json')
    training,selection=rows(out/'train.jsonl'),rows(out/'selection.jsonl')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    pad_id=manifest['base_manifest']['pad_id']
    models=[]
    for family in ('readout','mlp','control'):
        for spec in frozen[family]['runs']:
            destination=out/'tuned'/Path(spec['directory']).name
            if (destination/'result.json').exists():
                models.append(read_json(destination/'result.json'));continue
            destination.mkdir(parents=True,exist_ok=True)
            train.seed_all(spec['seed'])
            model=search.load_model(manifest['base_manifest']).requires_grad_(False)
            head=build(spec,weight)
            tail=model.model.language_model.layers[-2:]
            tail.requires_grad_(True)
            # Earlier layers are frozen; no embedding gradients or full-backbone graph.
            optimizer=train.MasterAdamW(nn.ModuleDict({'tail':tail,'head':head}),1e-6)
            master={id(p):m for p,m in optimizer.pairs}
            head_ids={id(master[id(p)]) for p in head.parameters()}
            for group in list(optimizer.param_groups):
                hp=[p for p in group['params'] if id(p) in head_ids]
                group['params']=[p for p in group['params'] if id(p) not in head_ids]
                if hp:
                    optimizer.optimizer.add_param_group({**group,'params':hp,'lr':spec['lr']})
            score,initial=evaluate_live(model,head,selection,pad_id)
            expected={r['id']:r['prediction'] for r in rows(Path(spec['directory'])/'selection.jsonl')}
            mismatch=sum(expected[r['id']]!=r['prediction'] for r in initial)
            if mismatch:
                raise ValueError(f'cached/live head predictions differ: {mismatch}')
            history={'0':score};best,best_step=score['macro'],0
            def save():
                save_head(head,destination/'head.safetensors')
                save_file({k:v.detach().cpu().contiguous() for k,v in tail.state_dict().items()},str(destination/'tail.safetensors'))
            save()
            started=time.monotonic()
            schedule=train.batches(len(training),16,args.tune_steps,random.Random(spec['seed']))
            with (destination/'train.jsonl').open('w') as log:
                for step,(indices,_) in enumerate(schedule,1):
                    model.train();head.train();optimizer.zero_grad()
                    loss_sum=0.
                    for group in train.microbatches([training[i] for i in indices],BATCH_ARGS):
                        batch=pointer.collate(group,pad_id,'cuda')
                        loss=pointer.soft_ce(readout.live_logits(model,head,batch),batch['targets']).sum()/len(indices)
                        if not torch.isfinite(loss):raise RuntimeError('nonfinite tail loss')
                        loss.backward();loss_sum+=loss.item()
                    norm=optimizer.clip_grad_norm(1.)
                    if not math.isfinite(norm):raise RuntimeError('nonfinite tail gradient')
                    optimizer.step()
                    log.write(json.dumps({'step':step,'loss':loss_sum,'grad_norm':norm})+'\n')
                    if step%32==0 or step==args.tune_steps:
                        score,_=evaluate_live(model,head,selection,pad_id)
                        history[str(step)]=score
                        if score['macro']>best:
                            best,best_step=score['macro'],step;save()
                        print(f'TAIL {family} seed={spec["seed"]} step={step}: {score["macro"]:.5f}',flush=True)
            head.load_state_dict(load_file(str(destination/'head.safetensors')))
            tail.load_state_dict(load_file(str(destination/'tail.safetensors')))
            score,records=evaluate_live(model,head,selection,pad_id)
            result={**spec,'family':family,'stage':'tuned','directory':str(destination),'best_step':best_step,
                    'steps':args.tune_steps,'batch':16,'selection':score,'cache_live_mismatches':mismatch,
                    'seconds':time.monotonic()-started,
                    'tail_trainable_parameters':sum(p.numel() for p in tail.parameters())}
            train.atomic_json(destination/'history.json',history)
            train.atomic_json(destination/'result.json',result)
            search.save_records(destination/'selection.jsonl',records)
            models.append(result)
            del optimizer,head,tail,model
            torch.cuda.empty_cache()
    checksums={str(Path(spec['directory'])/name):train.sha256_file(Path(spec['directory'])/name)
               for spec in models+frozen['readout']['runs']+frozen['mlp']['runs']+frozen['control']['runs']
               for name in ('head.safetensors','tail.safetensors') if (Path(spec['directory'])/name).exists()}
    train.atomic_json(out/'models-frozen.json',{'head_readout':frozen['readout']['runs'],
                      'head_mlp':frozen['mlp']['runs'],
                      'head_control':frozen['control']['runs'],'tuned':models,'sha256':checksums,
                      'rule':'No model/seed changes after confirmation. Report both seeds at both stages.'})


def confirm(args,manifest):
    out=args.output
    if (out/'confirmation-results.json').exists():
        raise SystemExit('confirmation already evaluated')
    frozen=read_json(out/'models-frozen.json')
    for file,sha in frozen['sha256'].items():
        if train.sha256_file(Path(file))!=sha:raise ValueError('frozen model checksum changed')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    encoded=rows(out/'confirmation.jsonl')
    model=search.load_model(manifest['base_manifest']).requires_grad_(False)
    cache=make_cache(model,encoded,manifest['base_manifest']['pad_id'],out/'confirmation.pt')
    cache={k:v.cuda() for k,v in cache.items()}
    base,base_records=evaluate_cache(build({'kind':'code'},weight),cache,encoded)
    search.save_records(out/'confirmation-base.jsonl',base_records)
    results=[];control_records={}
    for family in ('head_control','head_readout','head_mlp'):
        for spec in frozen[family]:
            head=build(spec,weight)
            score,rec=evaluate_cache(head,cache,encoded)
            search.save_records(out/f'confirmation-{family}-s{spec["seed"]}.jsonl',rec)
            result={'family':family,'seed':spec['seed'],'kind':spec['kind'],'metrics':score,
                    'vs_base':search.paired_delta(base_records,rec)}
            if family=='head_control':control_records[spec['seed']]=rec
            else:result['vs_control']=search.paired_delta(control_records[spec['seed']],rec)
            results.append(result)
    del cache,model,head
    torch.cuda.empty_cache()
    # Control first so each candidate can be compared with its seed-matched control.
    for spec in sorted(frozen['tuned'],key=lambda r:r['family']!='control'):
        model=search.load_model(manifest['base_manifest']).requires_grad_(False)
        head=build(spec,weight)
        model.model.language_model.layers[-2:].load_state_dict(load_file(str(Path(spec['directory'])/'tail.safetensors')))
        # Reproduce selected predictions after reconstructing the real model.
        _,reloaded=evaluate_live(model,head,rows(out/'selection.jsonl'),manifest['base_manifest']['pad_id'])
        expected={r['id']:r['prediction'] for r in rows(Path(spec['directory'])/'selection.jsonl')}
        mismatch=sum(expected[r['id']]!=r['prediction'] for r in reloaded)
        if mismatch:raise ValueError('saved/reloaded model predictions differ')
        score,rec=evaluate_live(model,head,encoded,manifest['base_manifest']['pad_id'])
        family='tuned_'+spec['family']
        search.save_records(out/f'confirmation-{family}-s{spec["seed"]}.jsonl',rec)
        result={'family':family,'seed':spec['seed'],'kind':spec['kind'],'metrics':score,
                'vs_base':search.paired_delta(base_records,rec),'reload_mismatches':mismatch}
        if spec['family']=='control':control_records[spec['seed']]=rec
        else:result['vs_control']=search.paired_delta(control_records[spec['seed']],rec)
        results.append(result)
        del model,head
        torch.cuda.empty_cache()
        print('CONFIRM',json.dumps(result),flush=True)
    for result in results:
        result['evidence_of_improvement']=('vs_control' in result and all(
            result[key]['paired_document_bootstrap_95ci'][0]>0 for key in ('vs_base','vs_control')))
    train.atomic_json(out/'confirmation-results.json',{'base':base,'models':results,
                      'scope':manifest['holdback'],'rule':'positive lower 95% CI against base AND matched trained control'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','screen','tune','confirm'))
    p.add_argument('--output',type=Path,default=Path('runs/readout/search-v1'))
    p.add_argument('--previous',type=Path,default=Path('runs/architecture/search-v1'))
    p.add_argument('--raw-train-rows',type=int,default=32768)
    p.add_argument('--max-tokens',type=int,default=1536)
    p.add_argument('--confirm-per-source',type=int,default=64)
    p.add_argument('--head-steps',type=int,default=1024)
    p.add_argument('--tune-steps',type=int,default=128)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    else:
        manifest=read_json(args.output/'manifest.json')
        if args.head_steps!=manifest['args']['head_steps'] or args.tune_steps!=manifest['args']['tune_steps']:
            raise ValueError('training budgets differ from the prepared plan')
        for split,sha in manifest['encoded_sha256'].items():
            if train.sha256_file(args.output/f'{split}.jsonl')!=sha:
                raise ValueError(f'{split} examples changed after preparation')
        if args.stage in ('tune','confirm'):
            hashes={name:train.sha256_file(Path(__file__).with_name(name))
                    for name in ('readout.py','readout_experiment.py')}
            if hashes!=read_json(args.output/'implementation.json'):
                raise ValueError('implementation changed after head screening')
        globals()[args.stage](args,manifest)


if __name__=='__main__':
    main()
