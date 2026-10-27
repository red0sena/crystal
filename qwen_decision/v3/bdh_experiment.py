"""BDH replacement and controlled data-quantity trial on exposed selection.

Three arms: native-large, BDH-large, BDH-small. All use 1024 updates and
the same per-step source trace. Small repeats 4096 rows four times; large
contains those rows plus 12288 distinct same-source rows, each seen once.
This estimates the data effect within BDH and the architecture effect on
large data; it is not a complete architecture-by-data factorial experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from qwen_decision.v3 import architecture_search as search,bdh_memory as bdh,modeling,train
from qwen_decision.v3 import pointer_experiment as pilot,readout_experiment as records
from qwen_decision.v3 import recurrent_experiment as old,robust_adaptation as views
from qwen_decision.v3.fresh_holdback import state_key

OUT = Path('runs/bdh-memory/data4x-v1')
PUBLIC = Path('runs/bdh-memory/public-source-v1')
BATCH = old.BATCH


def read(path):return json.loads(Path(path).read_text())


def implementation():
    modules = [bdh,modeling,train,search,pilot,records,old,views,pilot.fmt]
    return {str(p):train.sha256_file(p) for p in [Path(m.__file__) for m in modules]+[Path(__file__)]}


def prepare(out):
    if out.exists():raise ValueError('preserve existing trials')
    prior = old.verify(old.OUT)
    mix = Path('data/mix/v3-gold6.jsonl')
    expected = prior['input_sha256'].get(str(mix))
    if expected is None or train.sha256_file(mix)!=expected:raise ValueError('upstream training corpus changed')
    renderer,_ = search.renderer_for(search.BASE)
    before = records.rows(old.OUT/'train.jsonl')
    small_by = defaultdict(list)
    for r in before:small_by[r['source']].append(r)
    full = pilot.read_rows(mix)
    by = defaultdict(list)
    for r in full:
        if r['source'] in small_by:by[r['source']].append(r)
    pools = {};small_candidates = {}
    for source,base in sorted(small_by.items()):
        original_ids = {r['id'] for r in base}
        eligible = list(base)
        ordered = sorted((r for r in by[source] if r['id'] not in original_ids),
                         key=lambda r:search.digest(['bdh-data4x',r['id']]))
        for r in ordered:
            if len(eligible)>=256:break
            if len(views.view(renderer,r)['input_ids'])<=1536:eligible.append(r)
        cap = min(len(base),len(eligible)//4)
        if cap:
            small_candidates[source] = sorted(base,key=lambda r:search.digest(['bdh-small',r['id']]))[:cap]
            pools[source] = eligible
        print('POOL',source,'eligible',len(eligible),'small_cap',cap,flush=True)
    del full,by;gc.collect()
    quota = {s:0 for s in small_candidates};sources = sorted(quota)
    while sum(quota.values())<4096:
        changed = False
        for s in sources:
            if quota[s]<len(small_candidates[s]) and sum(quota.values())<4096:
                quota[s]+=1;changed=True
        if not changed:raise ValueError('insufficient paired small/large cases')
    small = [];large = []
    for source in sources:
        n = quota[source]
        chosen = small_candidates[source][:n];ids = {r['id'] for r in chosen}
        added = sorted((r for r in pools[source] if r['id'] not in ids),
                       key=lambda r:search.digest(['bdh-large',r['id']]))[:3*n]
        if len(added)!=3*n:raise ValueError('not enough unique large rows')
        small += chosen;large += chosen+added
    if len(small)!=4096 or len(large)!=16384 or len({r['id'] for r in large})!=16384:
        raise ValueError('data cardinality mismatch')
    selection = records.rows(old.OUT/'selection.jsonl')
    pilot.check_disjoint({'train':large,'selection':selection})
    small = sorted(small,key=lambda r:r['id']);large = sorted(large,key=lambda r:r['id'])
    large_index = {r['id']:i for i,r in enumerate(large)}
    large_by = defaultdict(list)
    for r in large:large_by[r['source']].append(large_index[r['id']])
    for s in large_by:random.Random(int(search.digest(['bdh-source-order',s])[:16],16)).shuffle(large_by[s])
    at = Counter();schedule = []
    for step,(indices,_) in enumerate(train.batches(len(small),16,1024,random.Random(41)),1):
        mapped = []
        for i in indices:
            s = small[i]['source'];mapped.append(large_by[s][at[s]]);at[s]+=1
        schedule.append({'step':step,'small':indices,'large':mapped})
    if len({i for e in schedule for i in e['large']})!=16384:raise ValueError('large data repeats')
    train_probe = sorted(small,key=lambda r:search.digest(['bdh-train-probe',r['id']]))[:512]
    plan = {'base':str(search.BASE),'arms':['bdh_large','native_large','bdh_small'],
            'steps':1024,'batch_size':16,'seed':41,'body_lr':3e-6,'memory_lr':1e-4,'warmup':64,
            'schedule':'linear warmup then cosine to 10%','snapshots':[256,512,1024],
            'architecture':{'start':16,'heads':4,'cycles':3,'parameters':11010048,
                'method':'positive features with paired BDH RoPE; no-self bidirectional associative kernel; multiplicative coactivation; shared refinements',
                'initialization':'copy removed native layer16 up/gate/down; no new zero gate',
                'native_executed_blocks':16,'unused_upper_blocks_frozen':8,'readout':'original normalized last-state code rows'},
            'data_small':4096,'data_large':16384,'draws_per_arm':16384,'small_repetitions':4,
            'paired_sources_per_update':True,'large_contains_small':True,
            'training':'all used language parameters and tied embedding/head; vision and skipped native blocks frozen',
            'optimizer':'AdamW FP32 master betas(.9,.95), WD.01, global clip1',
            'selection':'old exposed 4147 cases; primary common-update 1024 comparisons; trained-best snapshots secondary',
            'confirmation':'none unless architecture exceeds native-large selection by >=0.5pp; a new audited holdback would then be required',
            'train_probe':'fixed512 small-pool cases, seen in both data conditions; diagnostic only',
            'no_early_stop_on_accuracy':True,'no_new_teacher_labels_or_rationales':True,
            'fixed_update_budget':True,'matched_flops_or_parameters':False,
            'limits':['three arms, not a complete 2x2 factorial','one seed; repeated selection, not independent confirmation',
                      'retrofit of a trained decision model, not pretraining a new architecture from scratch']}
    out.mkdir(parents=True);train.atomic_json(out/'plan.json',plan)
    data = {'small':small,'large':large,'selection':selection,'train-probe':train_probe,'schedule':schedule}
    for name,rows in data.items():search.save_records(out/(name+'.jsonl'),rows)
    source = implementation();(out/'source').mkdir()
    for path in source:shutil.copyfile(path,out/'source'/Path(path).name)
    public = read(PUBLIC/'manifest.json')
    for name,sha in public['files'].items():
        if train.sha256_file(PUBLIC/name)!=sha:raise ValueError('public source changed')
    shutil.copytree(PUBLIC,out/'public-source')
    m = {'plan':plan,'implementation':source,'files':{k+'.jsonl':train.sha256_file(out/(k+'.jsonl')) for k in data},
         'base_sha256':prior['base_sha256'],'full_mix_sha256':expected,'full_mix_rows':699960,
         'upstream_manifest_sha256':train.sha256_file(old.OUT/'manifest.json'),'upstream_train_audit':prior.get('audit',{}),
         'public_source':public,'counts':{k:len(v) for k,v in data.items()},
         'sources':{k:dict(Counter(r['source'] for r in data[k])) for k in ['small','large','selection']},
         'distinct_states':{k:len({state_key(r['state']) for r in data[k]}) for k in ['small','large']},
         'scope':'large is sampled from full previously audited gold6 training; inherited selection exclusion of all gold2/3/6 training plus direct disjoint check; not a semantic paraphrase audit',
         'new_confirmation_cases_scored':0}
    train.atomic_json(out/'manifest.json',m)
    print('PREPARED',m['counts'],m['distinct_states'],flush=True)


def verify(out):
    m = read(out/'manifest.json')
    if implementation()!=m['implementation']:raise ValueError('source changed')
    for name,sha in m['files'].items():
        if train.sha256_file(out/name)!=sha:raise ValueError('input changed')
    if train.sha256_file(search.BASE/'model.safetensors')!=m['base_sha256']:raise ValueError('base changed')
    return m


def load_arm(kind,path=search.BASE):
    model = bdh.load(path);text = model.model.language_model
    model.requires_grad_(False)
    if kind=='native':
        text.requires_grad_(True);model.lm_head.requires_grad_(True)
    else:
        if not hasattr(text,'bdh_memory'):bdh.install(text,16,4,3)
        text.embed_tokens.requires_grad_(True);text.norm.requires_grad_(True);text.bdh_memory.requires_grad_(True)
        for layer in text.layers[:16]:layer.requires_grad_(True)
        for layer in text.layers[16:]:layer.to('cpu')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model


def logits_for(model,rows,pad,kind,**kwargs):
    ids,mask,codes,valid,target = train.collate(rows,pad)
    z = (modeling.code_logits(model,ids,mask,codes,valid) if kind=='native' else
         bdh.logits(model,ids,mask,codes,valid,**kwargs))
    return z,target


@torch.no_grad()
def evaluate(model,renderer,rows,kind,**kwargs):
    model.eval();train.set_checkpointing(model,False)
    encoded = [views.view(renderer,r) for r in rows];probabilities = {}
    torch.cuda.synchronize();started = time.monotonic()
    for group in train.microbatches(encoded,BATCH):
        z,_ = logits_for(model,group,renderer.tokenizer.pad_token_id,kind,**kwargs)
        for r,p in zip(group,z.softmax(-1).cpu()):probabilities[r['id']] = p[:len(r['target'])].tolist()
    torch.cuda.synchronize()
    result = []
    for r in rows:
        p = probabilities[r['id']];pred = max(range(len(p)),key=p.__getitem__)
        result.append({'id':r['id'],'source':r['source'],'group':r.get('group',r['id']),
                       'prediction':pred,'correct':r['target'][pred]>=max(r['target'])-1e-6,'probabilities':p,'views':1})
    return search.metrics(result)|{'seconds':time.monotonic()-started},result


def optimizer(model,kind,p):
    opt = train.MasterAdamW(model,p['body_lr'])
    if kind=='bdh':
        ids = {id(q) for q in model.model.language_model.bdh_memory.parameters()}
        masters = [m for q,m in opt.pairs if id(q) in ids];mids = {id(m) for m in masters}
        for group in opt.param_groups:group['params'] = [m for m in group['params'] if id(m) not in mids]
        opt.optimizer.add_param_group({'params':masters,'weight_decay':.01,'lr_scale':p['memory_lr']/p['body_lr']})
    return opt


def fit(out,arm,smoke=False):
    m = verify(out);p = m['plan'];kind,condition = arm.split('_');dest = out/('smoke' if smoke else arm)
    if dest.exists():raise ValueError('preserve existing fit')
    dest.mkdir();renderer,cfg = search.renderer_for(search.BASE)
    data = records.rows(out/(condition+'.jsonl'));selection = records.rows(out/'selection.jsonl')
    probe = records.rows(out/'train-probe.jsonl');schedule = records.rows(out/'schedule.jsonl')
    train.seed_all(p['seed']);model = load_arm(kind);opt = optimizer(model,kind,p)
    torch.cuda.reset_peak_memory_stats();started = time.monotonic();steps = 3 if smoke else p['steps']
    if not smoke:
        metric,rr = evaluate(model,renderer,selection,kind)
        train.atomic_json(dest/'selection-0.json',metric);search.save_records(dest/'selection-0.jsonl',rr)
        print('SELECTION',arm,0,metric['macro'],flush=True)
    snapshots = []
    with (dest/'train.jsonl').open('w') as log:
        for item in schedule[:steps]:
            step = item['step'];indices = item[condition]
            model.train();train.set_checkpointing(model,True);opt.zero_grad()
            encoded = [views.random_view(renderer,data[i],random.Random(int(search.digest([41,step,j])[:16],16)))
                       for j,i in enumerate(indices)]
            prompt_hash = search.digest([r['input_ids'] for r in encoded]);loss_sum = 0.
            reference = out/'bdh_large/train.jsonl'
            source_hash = search.digest([data[i]['source'] for i in indices])
            if arm!='bdh_large' and not smoke:
                if step==1:cached_trace = records.rows(reference)
                if source_hash!=cached_trace[step-1]['source_trace_sha256']:
                    raise ValueError('paired source trace differs')
                if condition=='large' and prompt_hash!=cached_trace[step-1]['prompt_sha256']:
                    raise ValueError('architecture comparison prompts differ')
            for group in train.microbatches(encoded,BATCH):
                z,target = logits_for(model,group,renderer.tokenizer.pad_token_id,kind)
                loss = train.soft_ce(z,target).sum()/len(indices)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();loss_sum+=loss.item()
            if step==1:
                if any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters() if q.requires_grad):
                    raise ValueError('missing/nonfinite trainable gradient')
                if kind=='bdh':
                    grad = {k:q.grad.norm().item() for k,q in model.model.language_model.bdh_memory.named_parameters()}
                    if min(grad.values())<=0:raise ValueError('inactive new core gradient')
                    train.atomic_json(dest/'initial-core-gradient.json',grad)
            norm = opt.clip_grad_norm(1.)
            if not math.isfinite(norm):raise ValueError('nonfinite gradients')
            scale = min(step/p['warmup'],1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-p['warmup'])/(p['steps']-p['warmup']))))
            for group in opt.param_groups:group['lr'] = p['body_lr']*scale*group['lr_scale']
            opt.step()
            entry = {'step':step,'loss':loss_sum,'grad_norm':norm,'seconds':time.monotonic()-started,
                     'prompt_sha256':prompt_hash,'source_trace_sha256':source_hash}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%32==0 or smoke:print('FIT',arm,entry,flush=True)
            if not smoke and step in p['snapshots']:
                path = dest/f'step-{step}'
                config = cfg|{'steps':step,'trained_from':str(search.BASE),'experiment':'bdh-data4x-v1','arm':arm}
                if kind=='bdh':bdh.save(model,renderer.tokenizer,path,config)
                else:modeling.save(model,renderer.tokenizer,path,config)
                metric,rr = evaluate(model,renderer,selection,kind)
                train.atomic_json(dest/f'selection-{step}.json',metric);search.save_records(dest/f'selection-{step}.jsonl',rr)
                training,tt = evaluate(model,renderer,probe,kind)
                train.atomic_json(dest/f'train-probe-{step}.json',training);search.save_records(dest/f'train-probe-{step}.jsonl',tt)
                snapshots.append({'step':step,'selection_macro':metric['macro'],'train_probe_macro':training['macro']})
                print('SELECTION',arm,step,metric['macro'],'TRAIN_PROBE',training['macro'],flush=True)
    result = {'arm':arm,'steps':steps,'seconds':time.monotonic()-started,'snapshots':snapshots,
              'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad),
              'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'new_confirmation_cases_scored':0}
    if smoke:
        opt.zero_grad();_,before = evaluate(model,renderer,data[:12],kind)
        bdh.save(model,renderer.tokenizer,dest/'checkpoint',cfg)
        del opt,model;gc.collect();torch.cuda.empty_cache()
        model = load_arm(kind,dest/'checkpoint').requires_grad_(False)
        _,after = evaluate(model,renderer,data[:12],kind)
        mismatch = sum(a['prediction']!=b['prediction'] for a,b in zip(before,after))
        delta = max(abs(a-b) for x,y in zip(before,after) for a,b in zip(x['probabilities'],y['probabilities']))
        if mismatch or delta:raise ValueError('reload changed computation')
        result.update(reload_prediction_mismatches=mismatch,reload_max_probability_delta=delta)
    train.atomic_json(dest/'result.json',result);print('RESULT',result,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','fit','run'])
    p.add_argument('--arm',choices=['bdh_large','native_large','bdh_small'],default='bdh_large')
    p.add_argument('--output',type=Path,default=OUT);a = p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage=='smoke':fit(a.output,'bdh_large',True)
    elif a.stage=='fit':fit(a.output,a.arm)
    else:
        for arm in read(a.output/'plan.json')['arms']:fit(a.output,arm)


if __name__=='__main__':main()
