"""Staged, source-balanced architecture search with a sealed confirmation partition.

python -m qwen_decision.v3.architecture_search prepare --output runs/architecture/search1
python -m qwen_decision.v3.architecture_search readouts --output runs/architecture/search1
python -m qwen_decision.v3.architecture_search surgery --output runs/architecture/search1
python -m qwen_decision.v3.architecture_search gates --output runs/architecture/search1
python -m qwen_decision.v3.architecture_search confirm --output runs/architecture/search1

Selection sees only the selection partition. Confirm freezes the winning specification
before evaluating held-back labels. The base model had previously used gold3 dev for
early stopping: this is a new architecture holdback, not a pristine new benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from qwen_decision.v3 import architecture as arch, modeling, pointer, train, format as fmt
from qwen_decision.v3 import pointer_experiment as pilot

BASE = Path('runs/v3/g4b-gold3-pad0/best')
MIX = Path('data/mix/v3-gold3.jsonl')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def keys(row):
    content = row['state'] if row['state'] not in (None, '', {}) else row['question']
    result = [('id', row.get('upstream_id', row['id'])), ('text', digest(content))]
    if row.get('item'):
        result.append(('item', row['item']))
    result += [('document', d) for d in row.get('documents') or []]
    return result


def partition_dev(rows):
    """Keep exact document/upstream variants together; never split individual questions at random."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in keys(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    names = {}
    for i, row in enumerate(rows):
        g = find(i)
        names[g] = min(names.get(g, row['id']), row['id'])
    return [('selection' if int(digest(['architecture-v1', names[find(i)]])[:8], 16) % 2 == 0 else 'confirm',
             names[find(i)]) for i in range(len(rows))]


def renderer_for(base):
    config = modeling.read_config(base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    args = SimpleNamespace(layout=config['layout'], pointer_positions=True, pad_probability=0.,
                           noul_choice_probability=0., code_shuffle_probability=0., code_sample_probability=0.)
    return pilot.Renderer(tokenizer, args, []), config


def tensor_cache(model, rows, pad_id, args, destination):
    collector = arch.SummaryCollector(model.model.language_model)
    n, layers, hidden = len(rows), len(model.model.language_model.layers), model.config.text_config.hidden_size
    features = torch.empty(n, layers, 3, hidden, dtype=torch.bfloat16)
    base = torch.empty(n, hidden, dtype=torch.bfloat16)
    width = max(len(r['target']) for r in rows)
    targets = torch.zeros(n, width)
    columns = torch.zeros(n, width, dtype=torch.long)
    valid = torch.zeros(n, width, dtype=torch.bool)
    index = {r['id']: i for i, r in enumerate(rows)}
    done, started = 0, time.monotonic()
    try:
        for group in train.microbatches(rows, args):
            batch = pointer.collate(group, pad_id, 'cuda')
            b, f = collector(batch)
            positions = torch.tensor([index[r['id']] for r in group])
            features[positions], base[positions] = f.cpu(), b.cpu()
            for row in group:
                i, k = index[row['id']], len(row['target'])
                targets[i, :k] = torch.tensor(row['target'])
                columns[i, :k] = torch.tensor(row['code_columns'])
                valid[i, :k] = True
            done += len(group)
            if done % 512 < len(group):
                print(f'cache {destination.stem}: {done}/{n}, {time.monotonic()-started:.0f}s', flush=True)
    finally:
        collector.close()
    torch.save({'features': features, 'base': base, 'targets': targets, 'columns': columns, 'valid': valid}, destination)


def load_cache(out, split):
    return torch.load(out / f'{split}.pt', map_location='cpu', weights_only=True)


def cuda_slice(cache, index):
    return {k: v[index].to('cuda') for k, v in cache.items()}


def metrics(records):
    by = defaultdict(list)
    for row in records:
        by[row['source']].append(bool(row['correct']))
    sources = {s: {'n': len(v), 'accuracy': sum(v)/len(v)} for s, v in sorted(by.items())}
    return {'n': len(records), 'accuracy': sum(r['correct'] for r in records)/len(records),
            'macro': sum(v['accuracy'] for v in sources.values())/len(sources), 'sources': sources}


@torch.no_grad()
def eval_cached(head, cache, rows):
    head.eval()
    result = []
    for start in range(0, len(rows), 128):
        batch = cuda_slice(cache, slice(start, start+128))
        logits = head(batch['base'], batch['features'], batch['columns'], batch['valid'])
        pred = logits.argmax(-1)
        gold = batch['targets'].gather(1, pred[:, None])[:, 0] >= batch['targets'].max(-1).values - 1e-6
        for row, p, c in zip(rows[start:start+128], pred.tolist(), gold.tolist()):
            result.append({'id': row['id'], 'group': row['group'], 'source': row['source'], 'prediction': p, 'correct': c})
    return metrics(result), result


def read_rows(out, split):
    return [json.loads(line) for line in (out / f'{split}.jsonl').open()]


def save_records(file, records):
    file.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))


def prepare(args):
    if args.output.exists():
        raise SystemExit('Preserve runs: prepare needs a new --output directory')
    renderer, config = renderer_for(args.base)
    if config['attention'] != 'noncausal':
        raise ValueError('use the same noncausal v3 baseline')
    raw_dev = pilot.read_rows(args.mix.with_suffix('.dev.jsonl'))
    groups = partition_dev(raw_dev)
    split = defaultdict(list)
    # Filter before sampling so all arms have the same source coverage and length limit.
    for e, (name, group) in zip(raw_dev, groups):
        split[name].append({**e, 'group': group})
    retained = {}
    for name, limit in [('selection', args.selection_per_source), ('confirm', args.confirm_per_source)]:
        counts, selected = Counter(), []
        for e in sorted(split[name], key=lambda r: digest(r['id'])):
            if counts[e['source']] >= limit:
                continue
            row = renderer(e)
            if len(row['input_ids']) <= args.max_tokens:
                selected.append({**row, 'group': e['group']})
                counts[e['source']] += 1
        retained[name] = selected
    eval_keys = {key for e in raw_dev for key in keys(e)}
    retained['train'], dropped = [], []
    for e in pilot.read_rows(args.mix, args.train_rows):
        if any(key in eval_keys for key in keys(e)):
            dropped.append(e['id'])
            continue
        row = renderer(e)
        if len(row['input_ids']) <= args.max_tokens:
            retained['train'].append({**row, 'group': e.get('item', e['id'])})
    if any(not rows for rows in retained.values()):
        raise ValueError('empty split')
    args.output.mkdir(parents=True)
    for split_name, rows in retained.items():
        save_records(args.output / f'{split_name}.jsonl', rows)
    manifest = {'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'config': config, 'code_ids': renderer.code_ids, 'pad_id': renderer.tokenizer.pad_token_id,
                'counts': {s: len(v) for s,v in retained.items()}, 'train_overlap_dropped': dropped,
                'sha256': {str(p): train.sha256_file(p) for p in [args.mix, args.mix.with_suffix('.dev.jsonl')]},
                'baseline_sha256': {p.name: train.sha256_file(p) for p in sorted(args.base.iterdir())
                                    if p.is_file() and p.suffix in ('.json', '.safetensors')},
                'implementation_sha256': {Path(p).name: train.sha256_file(Path(p)) for p in [__file__, arch.__file__]},
                'holdback': 'confirmation not used for architecture selection; g4b had historically used full gold3 dev',
                'training': 'same fixed rendered examples, no augmentation or teacher labels; only train split gradients'}
    train.atomic_json(args.output / 'manifest.json', manifest)
    model = modeling.load(args.base, attention=config['attention'], deltanet=config['deltanet'], kernel='sdpa').eval()
    model.requires_grad_(False)
    weight = model.get_output_embeddings().weight[renderer.code_ids].detach().float().cpu()
    torch.save(weight, args.output / 'code-weight.pt')
    for split_name in ['train', 'selection', 'confirm']:
        tensor_cache(model, retained[split_name], manifest['pad_id'], args, args.output / f'{split_name}.pt')
    print(json.dumps(manifest['counts']), flush=True)


def readouts(args, manifest):
    out = args.output
    train_cache, dev = load_cache(out, 'train'), load_cache(out, 'selection')
    rows = read_rows(out, 'selection')
    weight = torch.load(out / 'code-weight.pt', weights_only=True)
    layers = train_cache['features'].shape[1]
    baseline, _ = eval_cached(arch.Readout('base', weight, layers).cuda(), dev, rows)
    results = {'baseline': baseline, 'candidates': []}
    candidates = [('code', 1e-5), ('code', 1e-4), ('depth', .003), ('depth', .03),
                  ('pool', .003), ('pool', .03), ('scoring', .001), ('scoring', .01),
                  ('class_attention', .0001), ('class_attention', .001),
                  ('ffn',.00001),('ffn',.0001),('ffn',.001)]
    for kind, lr in candidates:
        for seed in (41, 73):
            name = f'{kind}-lr{lr:g}-s{seed}'
            destination = out / 'readouts' / name
            if (destination / 'result.json').exists():
                result = json.loads((destination / 'result.json').read_text())
                results['candidates'].append(result)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            train.seed_all(seed)
            head = arch.Readout(kind, weight, layers).cuda()
            parameters = [p for p in head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
            best, best_step, history = baseline['macro'], 0, {'0': baseline}
            save_file({k: v.detach().cpu().contiguous() for k,v in head.state_dict().items()}, str(destination/'head.safetensors'))
            for step, (indices, _) in enumerate(train.batches(len(train_cache['base']), 128, args.readout_steps, random.Random(seed)), 1):
                head.train()
                batch = cuda_slice(train_cache, torch.tensor(indices))
                optimizer.zero_grad()
                logits = head(batch['base'], batch['features'], batch['columns'], batch['valid'])
                loss = pointer.soft_ce(logits, batch['targets']).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                if step % 64 == 0 or step == args.readout_steps:
                    score, _ = eval_cached(head, dev, rows)
                    history[str(step)] = score
                    if score['macro'] > best:
                        best, best_step = score['macro'], step
                        save_file({k: v.detach().cpu().contiguous() for k,v in head.state_dict().items()}, str(destination/'head.safetensors'))
            head.load_state_dict(load_file(str(destination/'head.safetensors')))
            score, records = eval_cached(head, dev, rows)
            result = {'family': 'readout', 'kind': kind, 'lr': lr, 'seed': seed, 'best_step': best_step,
                      'selection': score, 'directory': str(destination), 'steps': args.readout_steps,
                      'trainable_parameters': sum(p.numel() for p in parameters)}
            train.atomic_json(destination / 'result.json', result)
            train.atomic_json(destination / 'history.json', history)
            save_records(destination / 'selection-predictions.jsonl', records)
            results['candidates'].append(result)
            train.atomic_json(out / 'readouts.json', results)
            print(f'{name}: selection {score["macro"]:.4f} (base {baseline["macro"]:.4f}), step {best_step}', flush=True)
    train.atomic_json(out / 'readouts.json', results)


def controls(args, manifest):
    """Original-head controls with exactly the gate arms' examples, batches and update count."""
    out=args.output
    training_cache,dev=load_cache(out,'train'),load_cache(out,'selection')
    rows=read_rows(out,'selection')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    layers=training_cache['features'].shape[1]
    baseline,_=eval_cached(arch.Readout('base',weight,layers).cuda(),dev,rows)
    results={'candidates':[]}
    for lr in [1e-5,1e-4]:
        for seed in [41,73]:
            train.seed_all(seed)
            destination=out/'readouts'/f'code-gate-budget-lr{lr:g}-s{seed}'
            if (destination/'result.json').exists():
                results['candidates'].append(json.loads((destination/'result.json').read_text()))
                continue
            destination.mkdir(parents=True,exist_ok=True)
            head=arch.Readout('code',weight,layers).cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.)
            best,best_step=baseline['macro'],0

            def save():
                save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(destination/'head.safetensors'))

            save()
            for step,(indices,_) in enumerate(train.batches(len(training_cache['base']),16,args.gate_steps,random.Random(seed)),1):
                batch=cuda_slice(training_cache,torch.tensor(indices))
                optimizer.zero_grad()
                loss=pointer.soft_ce(head(batch['base'],batch['features'],batch['columns'],batch['valid']),batch['targets']).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                optimizer.step()
                if step%64==0 or step==args.gate_steps:
                    score,_=eval_cached(head,dev,rows)
                    if score['macro']>best:
                        best,best_step=score['macro'],step
                        save()
            head.load_state_dict(load_file(str(destination/'head.safetensors')))
            score,records=eval_cached(head,dev,rows)
            result={'family':'readout','kind':'code','budget':'gates','lr':lr,'seed':seed,'best_step':best_step,
                    'selection':score,'directory':str(destination),'steps':args.gate_steps,'batch':16}
            save_records(destination/'selection-predictions.jsonl',records)
            train.atomic_json(destination/'result.json',result)
            results['candidates'].append(result)
            print(f'gate-budget code {lr:g} s{seed}: {score["macro"]:.4f}',flush=True)
    train.atomic_json(out/'controls.json',results)


def contrast(args,manifest):
    out=args.output
    cache,rows=load_cache(out,'selection'),read_rows(out,'selection')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    results={'candidates':[]}
    for mode in ['mean','js']:
        for window in [4,8,12]:
            for alpha in [.05,.1,.2,.4]:
                kind=f'contrast-{mode}-last{window}-a{alpha:g}'
                head=arch.ContrastReadout(weight,window,alpha,mode).cuda()
                score,records=eval_cached(head,cache,rows)
                results['candidates'].append({'family':'contrast','kind':kind,'mode':mode,'window':window,
                    'alpha':alpha,'selection':score,'trainable_parameters':0})
                save_records(out/f'{kind}-selection.jsonl',records)
                print(f'{kind}: {score["macro"]:.4f}',flush=True)
    train.atomic_json(out/'contrast.json',results)


def option_batch(cache,indices):
    if isinstance(indices,slice):
        indices=list(range(len(cache['base'])))[indices]
    elif isinstance(indices,torch.Tensor):
        indices=indices.tolist()
    lengths=[int(cache['starts'][i+1]-cache['starts'][i]) for i in indices]
    width=max(lengths)
    options=torch.zeros(len(indices),width,cache['values'].shape[1],dtype=torch.bfloat16)
    for b,i in enumerate(indices):
        lo,hi=int(cache['starts'][i]),int(cache['starts'][i+1])
        options[b,:hi-lo]=cache['values'][lo:hi]
    index=torch.tensor(indices)
    result={k:cache[k][index].to('cuda') for k in ['base','targets','columns','valid']}
    for k in ['targets','columns','valid']:
        result[k]=result[k][:,:width]
    result['features']=options.to('cuda')
    return result


@torch.no_grad()
def eval_options(head,cache,rows):
    head.eval()
    records=[]
    for start in range(0,len(rows),128):
        batch=option_batch(cache,slice(start,start+128))
        logits=head(batch['base'],batch['features'],batch['columns'],batch['valid'])
        pred=logits.argmax(-1)
        correct=batch['targets'].gather(1,pred[:,None])[:,0]>=batch['targets'].max(-1).values-1e-6
        for r,p,c in zip(rows[start:start+128],pred.tolist(),correct.tolist()):
            records.append({'id':r['id'],'group':r['group'],'source':r['source'],'prediction':p,'correct':c})
    return metrics(records),records


def options(args,manifest):
    out=args.output
    missing=[s for s in ['train','selection','confirm'] if not (out/f'options-{s}.pt').exists()]
    if missing:
        model=load_model(manifest).requires_grad_(False)
        for split in missing:
            rows=read_rows(out,split)
            cache=load_cache(out,split)
            del cache['features']
            index={r['id']:i for i,r in enumerate(rows)}
            values=[None]*len(rows)
            done,difference=0,0.
            with torch.no_grad():
                for group in train.microbatches(rows,args):
                    batch=pointer.collate(group,manifest['pad_id'],'cuda')
                    h=model.model.language_model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                                                 use_cache=False).last_hidden_state
                    for b,r in enumerate(group):
                        i=index[r['id']]
                        values[i]=h[b,batch['positions'][b,:len(r['target'])]].cpu()
                        actual=h[b,int(batch['attention_mask'][b].sum())-1].cpu()
                        difference=max(difference,(actual-cache['base'][i]).abs().max().item())
                    done+=len(group)
                    if done%1024<len(group):
                        print(f'option cache {split} {done}/{len(rows)}',flush=True)
            if difference>.05:
                raise RuntimeError(f'option cache differs from original backbone: {difference}')
            lengths=[len(v) for v in values]
            cache['values']=torch.cat(values)
            cache['starts']=torch.tensor([0]+list(np.cumsum(lengths)))
            torch.save(cache,out/f'options-{split}.pt')
        del model,cache,values
        gc.collect()
        torch.cuda.empty_cache()
    training_cache=torch.load(out/'options-train.pt',weights_only=True)
    dev=torch.load(out/'options-selection.pt',weights_only=True)
    rows=read_rows(out,'selection')
    weight=torch.load(out/'code-weight.pt',weights_only=True)
    base=eval_options(arch.OptionReadout('pointer-residual',weight).cuda(),dev,rows)[0]
    results={'baseline':base,'candidates':[]}
    for kind in ['pointer-residual','set-residual']:
        for lr in [.0001,.001]:
            for seed in [41,73]:
                destination=out/'readouts'/f'{kind}-lr{lr:g}-s{seed}'
                if (destination/'result.json').exists():
                    results['candidates'].append(json.loads((destination/'result.json').read_text()))
                    continue
                destination.mkdir(parents=True,exist_ok=True)
                train.seed_all(seed)
                head=arch.OptionReadout(kind,weight).cuda()
                optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.)
                best,best_step=base['macro'],0

                def snapshot():
                    save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(destination/'head.safetensors'))

                snapshot()
                history={'0':base}
                for step,(indices,_) in enumerate(train.batches(len(training_cache['base']),128,args.readout_steps,random.Random(seed)),1):
                    head.train()
                    batch=option_batch(training_cache,indices)
                    optimizer.zero_grad()
                    loss=pointer.soft_ce(head(batch['base'],batch['features'],batch['columns'],batch['valid']),batch['targets']).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite option loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                    optimizer.step()
                    if step%64==0 or step==args.readout_steps:
                        score,_=eval_options(head,dev,rows)
                        history[str(step)]=score
                        if score['macro']>best:
                            best,best_step=score['macro'],step
                            snapshot()
                head.load_state_dict(load_file(str(destination/'head.safetensors')))
                score,records=eval_options(head,dev,rows)
                result={'family':'options','kind':kind,'lr':lr,'seed':seed,'best_step':best_step,
                        'selection':score,'directory':str(destination),'steps':args.readout_steps,
                        'trainable_parameters':sum(p.numel() for p in head.parameters())}
                train.atomic_json(destination/'result.json',result)
                train.atomic_json(destination/'history.json',history)
                save_records(destination/'selection-predictions.jsonl',records)
                results['candidates'].append(result)
                train.atomic_json(out/'options.json',results)
                print(f'{kind} lr{lr:g} s{seed}: selection {score["macro"]:.4f}, step {best_step}',flush=True)
    train.atomic_json(out/'options.json',results)


def external(args,manifest):
    """One post-selection JEVal restricted short-context check; never a training source."""
    from qwen_decision.v3 import proxy_eval
    from qwen_decision.v3.architecture_inference import Scorer
    out=args.output
    if (out/'external-jeval.json').exists():
        raise SystemExit('External evaluation already exists; preserve the result.')
    frozen=json.loads((out/'frozen-selection.json').read_text())
    # Canonical seed 41, fixed before viewing external scores. No seed selection on this set.
    candidate=next((c for c in frozen['chosen'] if c.get('seed')==41),frozen['chosen'][0])
    renderer,_=renderer_for(Path(manifest['args']['base']))
    rows,skipped=[],[]
    for r in proxy_eval.load_rows(40,seed=0,split='restricted'):
        key,q=next(iter(json.loads(r['questions_json']).items()))
        label=json.loads(r['label_json'])
        label=label[key] if isinstance(label,dict) else label
        kit,_=proxy_eval.as_kit(q)
        option_keys,_=fmt.options(kit)
        correct=(1 if bool(label) else 0) if q['type']=='noul' else option_keys.index(str(int(label)) if q['type']=='score' else label)
        target=[float(i==correct) for i in range(len(option_keys))]
        e={'id':r['id'],'kind':q['type'],'source':r['source'],'state':r['state'],'question':kit,
           'target':target,'ordered':q['type']=='score'}
        row=renderer(e)
        if len(row['input_ids'])>args.max_tokens:
            skipped.append(r['id'])
        else:
            rows.append({**row,'group':r['id'],'chance':1/len(option_keys)})
    results={'scope':f'JEVal v2 restricted, at most 40/source, prompts <= {args.max_tokens} tokens; not the full benchmark',
             'candidate':candidate,'n':len(rows),'skipped_over_limit':skipped,
             'protocol_sha256':train.sha256_file(proxy_eval.PROTOCOL),
             'data_sha256':train.sha256_file(proxy_eval.JEVAL),'arms':{}}
    record_map={}
    for name,spec in [('base',None),('candidate',candidate)]:
        scorer=Scorer(manifest,spec)
        records=[]
        for group in train.microbatches(rows,args):
            batch=pointer.collate(group,manifest['pad_id'],'cuda')
            logits=scorer.logits(batch)
            pred=logits.argmax(-1).tolist()
            for r,p in zip(group,pred):
                records.append({'id':r['id'],'group':r['group'],'source':r['source'],'prediction':p,
                    'correct':r['target'][p]>0,'chance':r['chance'],'error':None})
        results['arms'][name]={'accuracy':metrics(records),'proxy':proxy_eval.summarize(records)}
        record_map[name]=records
        save_records(out/f'external-jeval-{name}.jsonl',records)
        scorer.close()
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    results['paired']=paired_delta(record_map['base'],record_map['candidate'])
    train.atomic_json(out/'external-jeval.json',results)
    print(json.dumps({'n':len(rows),'base_macro':results['arms']['base']['accuracy']['macro'],
        'candidate_macro':results['arms']['candidate']['accuracy']['macro'],'paired':results['paired']},indent=2),flush=True)


@torch.no_grad()
def eval_model(model, rows, args, pad_id, code_ids):
    model.eval()
    records = []
    for group in train.microbatches(rows, args):
        batch = pointer.collate(group, pad_id, 'cuda')
        codes = torch.tensor(code_ids, device='cuda')[batch['columns']]
        logits = modeling.code_logits(model, batch['input_ids'], batch['attention_mask'], codes, batch['valid'])
        pred = logits.argmax(-1)
        correct = batch['targets'].gather(1, pred[:, None])[:, 0] >= batch['targets'].max(-1).values - 1e-6
        for row, p, c in zip(group, pred.tolist(), correct.tolist()):
            records.append({'id': row['id'], 'group': row['group'], 'source': row['source'], 'prediction': p, 'correct': c})
    return metrics(records), records


def load_model(manifest):
    return modeling.load(Path(manifest['args']['base']), attention=manifest['config']['attention'],
                         deltanet=manifest['config']['deltanet'], kernel='sdpa').eval()


def surgery(args, manifest):
    out = args.output
    model = load_model(manifest)
    model.requires_grad_(False)
    original = list(model.model.language_model.layers)
    rows = read_rows(out, 'selection')
    # Rank initial surgeries on 8 rows/source, then evaluate only the best two on all selection rows.
    counts, small = Counter(), []
    for r in rows:
        if counts[r['source']] < 8:
            small.append(r)
            counts[r['source']] += 1
    candidates = []
    for kind, indexes in [('base', [None]), ('skip', range(2, len(original)-2)), ('swap', range(2, len(original)-3, 2))]:
        for index in indexes:
            order = arch.layer_order(len(original), kind, index)
            arch.apply_layer_order(model.model.language_model, original, order)
            score, _ = eval_model(model, small, args, manifest['pad_id'], manifest['code_ids'])
            result = {'family': 'surgery', 'kind': kind, 'index': index, 'order': order, 'screen': score}
            candidates.append(result)
            train.atomic_json(out/'surgery-screen.json', candidates)
            print(f'{kind} {index}: screen {score["macro"]:.4f}', flush=True)
    candidates = sorted([r for r in candidates if r['kind'] != 'base'], key=lambda r:r['screen']['macro'], reverse=True)[:2]
    for candidate in candidates:
        arch.apply_layer_order(model.model.language_model, original, candidate['order'])
        score, records = eval_model(model, rows, args, manifest['pad_id'], manifest['code_ids'])
        candidate['selection'] = score
        save_records(out / f'surgery-{candidate["kind"]}-{candidate["index"]}-selection.jsonl', records)
    arch.apply_layer_order(model.model.language_model, original, list(range(len(original))))
    train.atomic_json(out/'surgery.json', {'candidates': candidates})


def gates(args, manifest, modes=None, result_file='gates.json'):
    out = args.output
    training_rows, dev = read_rows(out, 'train'), read_rows(out, 'selection')
    results = {'candidates': []}
    # Frozen original code head, no readout retraining. Both arms start EXACTLY at the baseline.
    for mode, lr in (modes or [('scalar', .003), ('channel', .003)]):
        channels = mode == 'channel'
        for seed in (41, 73):
            name = f'{mode}-s{seed}'
            destination = out/'gates'/name
            if (destination/'result.json').exists():
                results['candidates'].append(json.loads((destination/'result.json').read_text()))
                continue
            train.seed_all(seed)
            model = load_model(manifest)
            model.requires_grad_(False)
            parameters, handles = (arch.install_inner_scales(model.model.language_model) if mode == 'ia3'
                else arch.install_branch_scales(model.model.language_model, channels=channels))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            optimizer = torch.optim.AdamW(list(parameters.values()), lr=lr, weight_decay=0.)
            initial, records = eval_model(model, dev, args, manifest['pad_id'], manifest['code_ids'])
            best, best_step, history = initial['macro'], 0, {'0': initial}

            def snapshot():
                save_file({k:v.detach().cpu().contiguous() for k,v in parameters.items()}, str(destination/'scales.safetensors'))

            destination.mkdir(parents=True, exist_ok=True)
            snapshot()
            with (destination/'train.jsonl').open('w', buffering=1) as log:
                for step, (indices, _) in enumerate(train.batches(len(training_rows), 16, args.gate_steps, random.Random(seed)), 1):
                    model.train()
                    optimizer.zero_grad()
                    drawn = [training_rows[i] for i in indices]
                    loss_sum = 0.
                    for group in train.microbatches(drawn, args):
                        batch = pointer.collate(group, manifest['pad_id'], 'cuda')
                        codes = torch.tensor(manifest['code_ids'], device='cuda')[batch['columns']]
                        logits = modeling.code_logits(model, batch['input_ids'], batch['attention_mask'], codes, batch['valid'])
                        loss = pointer.soft_ce(logits, batch['targets']).sum()/len(drawn)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'{name}: nonfinite loss')
                        loss.backward()
                        loss_sum += loss.item()
                    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError(f'{name}: nonfinite gradient')
                    optimizer.step()
                    log.write(json.dumps({'step':step, 'loss':loss_sum, 'grad_norm':float(norm)})+'\n')
                    if step % 64 == 0 or step == args.gate_steps:
                        score, records = eval_model(model, dev, args, manifest['pad_id'], manifest['code_ids'])
                        history[str(step)] = score
                        if score['macro'] > best:
                            best, best_step = score['macro'], step
                            snapshot()
                        print(f'{name} {step}: selection {score["macro"]:.4f} (base {initial["macro"]:.4f})', flush=True)
            state = load_file(str(destination/'scales.safetensors'))
            with torch.no_grad():
                for k,p in parameters.items():
                    p.copy_(state[k].to(p.device))
            score, records = eval_model(model, dev, args, manifest['pad_id'], manifest['code_ids'])
            result = {'family':'gates', 'kind':mode, 'channels':channels,
                      'lr':lr, 'seed':seed, 'best_step':best_step, 'steps':args.gate_steps,
                      'selection':score, 'directory':str(destination),
                      'implementation_sha256':{Path(p).name:train.sha256_file(Path(p)) for p in [__file__,arch.__file__]},
                      'trainable_parameters':sum(p.numel() for p in parameters.values())}
            save_records(destination/'selection-predictions.jsonl', records)
            train.atomic_json(destination/'history.json', history)
            train.atomic_json(destination/'result.json', result)
            results['candidates'].append(result)
            train.atomic_json(out/result_file, results)
            arch.remove_branch_scales(model.model.language_model, handles)
            del optimizer, model, parameters, handles
            gc.collect()
            torch.cuda.empty_cache()
    train.atomic_json(out/result_file, results)


def inner(args,manifest):
    gates(args,manifest,modes=[('ia3',.003)],result_file='inner.json')


def paired_delta(base, candidate, iterations=2000):
    a, b = {r['id']:r for r in base}, {r['id']:r for r in candidate}
    if a.keys() != b.keys():
        raise ValueError('paired evaluation ids differ')
    by = defaultdict(lambda:defaultdict(list))
    for key, r in a.items():
        by[r['source']][r['group']].append(int(b[key]['correct'])-int(r['correct']))
    groups = [[np.asarray(v, dtype=float) for v in per.values()] for per in by.values()]
    rng = np.random.default_rng(20261019)
    draws = []
    for _ in range(iterations):
        scores = []
        for source in groups:
            sampled = rng.integers(len(source), size=len(source))
            scores.append(np.concatenate([source[i] for i in sampled]).mean())
        draws.append(float(np.mean(scores)))
    return {'macro_delta':metrics(candidate)['macro']-metrics(base)['macro'],
            'paired_document_bootstrap_95ci':np.quantile(draws, [.025,.975]).tolist(),
            'wins':sum(b[k]['correct'] and not a[k]['correct'] for k in a),
            'losses':sum(a[k]['correct'] and not b[k]['correct'] for k in a)}


def confirm(args, manifest):
    out = args.output
    if (out/'confirm-results.json').exists():
        raise SystemExit('Confirmation already evaluated. Preserve it; do not select again on its results.')
    candidates = []
    for name in ['readouts', 'surgery', 'gates', 'inner', 'controls','contrast','options']:
        candidates += json.loads((out/f'{name}.json').read_text())['candidates']
    # Choose architecture/hyperparameters by mean over selection seeds, NEVER by confirmation.
    grouped = defaultdict(list)
    for r in candidates:
        grouped[(r['family'], r['kind'], r.get('lr'), r.get('index'),r.get('budget','readouts'))].append(r)
    ranked = sorted(grouped.values(), key=lambda group:np.mean([r['selection']['macro'] for r in group]), reverse=True)
    structural = [g for g in ranked if g[0]['kind'] != 'code']
    controls = [g for g in ranked if g[0]['kind'] == 'code']
    chosen = structural[0]
    matched_budget='gates' if chosen[0]['family']=='gates' else 'readouts'
    matched_controls=[g for g in controls if g[0].get('budget','readouts')==matched_budget]
    selected_controls=matched_controls[0]
    frozen = {'chosen':chosen, 'code_control':selected_controls, 'ranking':[
        {'family':g[0]['family'],'kind':g[0]['kind'],'lr':g[0].get('lr'),
         'mean_selection_macro':float(np.mean([r['selection']['macro'] for r in g]))} for g in ranked],
        'policy':'specifications frozen before confirmation evaluation; all seeds evaluated, no cherry-picking'}
    train.atomic_json(out/'frozen-selection.json', frozen)
    cache, rows = load_cache(out, 'confirm'), read_rows(out, 'confirm')
    weight = torch.load(out/'code-weight.pt', weights_only=True)
    baseline, baseline_records = eval_cached(arch.Readout('base',weight,cache['features'].shape[1]).cuda(), cache, rows)
    save_records(out/'confirm-base.jsonl', baseline_records)
    results = {'baseline':baseline, 'architecture':[], 'code_control':[], 'independence':manifest['holdback']}
    for label, selected in [('architecture',chosen),('code_control',selected_controls)]:
        for candidate in selected:
            if candidate['family']=='options':
                option_cache=torch.load(out/'options-confirm.pt',weights_only=True)
                head=arch.OptionReadout(candidate['kind'],weight).cuda()
                head.load_state_dict(load_file(str(Path(candidate['directory'])/'head.safetensors')))
                score,records=eval_options(head,option_cache,rows)
                del head,option_cache
            elif candidate['family'] in ('readout','contrast'):
                if candidate['family']=='contrast':
                    head=arch.ContrastReadout(weight,candidate['window'],candidate['alpha'],candidate['mode']).cuda()
                else:
                    head = arch.Readout(candidate['kind'],weight,cache['features'].shape[1]).cuda()
                    head.load_state_dict(load_file(str(Path(candidate['directory'])/'head.safetensors')))
                score, records = eval_cached(head, cache, rows)
                del head
            else:
                model = load_model(manifest)
                handles = None
                if candidate['family'] == 'surgery':
                    original = list(model.model.language_model.layers)
                    arch.apply_layer_order(model.model.language_model, original, candidate['order'])
                else:
                    parameters, handles = (arch.install_inner_scales(model.model.language_model) if candidate['kind']=='ia3'
                        else arch.install_branch_scales(model.model.language_model, channels=candidate['channels']))
                    state = load_file(str(Path(candidate['directory'])/'scales.safetensors'))
                    with torch.no_grad():
                        for k,p in parameters.items():
                            p.copy_(state[k].to(p.device))
                score, records = eval_model(model, rows, args, manifest['pad_id'], manifest['code_ids'])
                if handles:
                    arch.remove_branch_scales(model.model.language_model, handles)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            comparison = paired_delta(baseline_records, records)
            result = {'candidate':candidate, 'confirm':score, 'against_base':comparison}
            results[label].append(result)
            save_records(out/f'confirm-{label}-s{candidate.get("seed",0)}.jsonl', records)
    # Also compare with the best equally trained original-head control, not only frozen base.
    for result in results['architecture']:
        seed = result['candidate'].get('seed', 0)
        records = [json.loads(l) for l in (out/f'confirm-architecture-s{seed}.jsonl').open()]
        result['against_trained_code'] = []
        for control in results['code_control']:
            control_seed = control['candidate']['seed']
            control_records = [json.loads(l) for l in (out/f'confirm-code_control-s{control_seed}.jsonl').open()]
            result['against_trained_code'].append({'seed':control_seed,**paired_delta(control_records,records)})
    results['evidence_of_improvement'] = all(r['against_base']['paired_document_bootstrap_95ci'][0] > 0
        and all(c['paired_document_bootstrap_95ci'][0] > 0 for c in r['against_trained_code'])
        for r in results['architecture'])
    train.atomic_json(out/'confirm-results.json', results)
    print(json.dumps({'baseline_macro':baseline['macro'], 'evidence_of_improvement':results['evidence_of_improvement'],
        'architecture':[{k:r[k] for k in ['confirm','against_base']} for r in results['architecture']]}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage',choices=['prepare','readouts','surgery','gates','inner','controls','contrast','options','confirm','external'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--mix',type=Path,default=MIX)
    p.add_argument('--train-rows',type=int,default=8192)
    p.add_argument('--selection-per-source',type=int,default=32)
    p.add_argument('--confirm-per-source',type=int,default=64)
    p.add_argument('--max-tokens',type=int,default=1536)
    p.add_argument('--micro-batch',type=int,default=4)
    p.add_argument('--token-budget',type=int,default=4096)
    p.add_argument('--readout-steps',type=int,default=256)
    p.add_argument('--gate-steps',type=int,default=128)
    args = p.parse_args()
    args.pack = False
    if min(args.train_rows,args.selection_per_source,args.confirm_per_source,args.max_tokens,args.micro_batch,
           args.token_budget,args.readout_steps,args.gate_steps) < 1:
        p.error('row counts, steps and token limits must be positive')
    if args.stage == 'prepare':
        prepare(args)
    else:
        manifest = json.loads((args.output/'manifest.json').read_text())
        for field in ['base','mix','max_tokens','micro_batch','token_budget','readout_steps','gate_steps']:
            value = manifest['args'][field]
            setattr(args,field,Path(value) if field in ('base','mix') else value)
        globals()[args.stage](args,manifest)


if __name__ == '__main__':
    main()
