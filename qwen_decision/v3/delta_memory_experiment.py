"""Matched whole-backbone block-delta routing trial with a fresh confirmation.

Reuse the verified native 512-update control; train one preregistered router.
Select only on the old selection set, freeze both checkpoints, then confirm.
"""
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

from qwen_decision.v3 import architecture_search as search, delta_memory as dm, modeling, train
from qwen_decision.v3 import pointer_experiment as pilot, readout_experiment as previous
from qwen_decision.v3 import recurrent_experiment as old, robust_adaptation as views, robust_experiment as evaluation
from qwen_decision.v3 import fresh_eval as fresh
from qwen_decision.v3.fresh_holdback import state_key
from qwen_decision.v3.reference_comparison import as_example

OUT=Path('runs/delta-memory/block4-v1')
OLD=old.OUT
TAIL=Path('runs/decision-tail/trained-v1')
BATCH=old.BATCH


def read(path):return json.loads(Path(path).read_text())


def implementation():
    from qwen_decision.v3 import engine
    modules=[dm,modeling,train,views,old,evaluation,fresh,engine]
    return {str(p):train.sha256_file(p) for p in [Path(m.__file__) for m in modules]+[Path(__file__)]}


def prepare(out):
    if out.exists():raise ValueError('preserve trials; choose a new output directory')
    prior=old.verify(OLD)
    p={**prior['plan'],'arms':['vanilla','delta'],'architecture':{
        'kind':'gated_block_delta_memory','block_size':4,'groups':6,'parameters':10240,
        'sources':'per-token block output minus block input, plus fixed zero source',
        'routing':'RMS-normalized delta keys, learned query, depth softmax, zero-initialized tanh channel gate',
        'backbone_calls':1,'native_layers_executed':24},
       'router_lr':.003,'gate_lr':None,
       'selection':'independent best trained snapshot 256/512 on old selection; no step-0 fallback',
       'confirmation':'one preregistered delta-vs-native comparison, source macro, 5000 paired document bootstrap draws, 95% CI',
       'fresh_sampling':'remaining gold6-dev plus local pinned official fresh-v1 inventory excluding GoEmotions; >=20 unobserved groups/source; <=256 groups/source, one fixed-hash representative/group; max4096 tokens, no truncation',
       'control':'reuse complete verified native fit; no new native training claimed',
       'routing_optimizer':'same AdamW master, no decay for new query/gate parameters',
       'checkpointing':'six explicit tensor-carrying four-native-layer groups; native control used per-layer checkpointing; no dropout or extra training examples',
       'no_model_or_seed_averaging':True,'fixed_data_update_budget':True,'matched_compute_budget':False}
    out.mkdir(parents=True);train.atomic_json(out/'plan.json',p)
    for name in ['train.jsonl','selection.jsonl']:shutil.copyfile(OLD/name,out/name)
    dev=pilot.read_rows(Path('data/mix/v3-gold6.dev.jsonl'))
    candidates=[{**r,'suite':'remaining_gold6'} for r in dev]
    raw=list(fresh.rows())
    candidates += [{**as_example(r),'label_origin':'human','suite':'official_fresh'} for r in raw if r['source']!='goemotions']
    # Official exports may repeat upstream IDs (e.g. question variants). Keep
    # that upstream identity for grouping, but use distinct evaluation IDs.
    unique={};duplicates=0
    for r in candidates:
        rid=r['id'];signature=search.digest([r['state'],r['question']])[:20]
        uid=r['suite']+':'+rid+'#'+signature
        item={**r,'upstream_id':r.get('upstream_id',rid),'id':uid}
        if uid in unique:
            if unique[uid]['target']!=item['target']:raise ValueError('conflicting labels for identical input')
            duplicates+=1
        else:unique[uid]=item
    candidates=list(unique.values())
    groups=old.groups_for(candidates);key_ids=defaultdict(set)
    for r in candidates:
        for key in old.keys(r):key_ids[key].add(r['id'])
    blocked=set();audit={}
    def mark(r):
        for key in old.keys(r):blocked.update(key_ids.get(key,()))
    exposed=[Path('data/mix/v3-gold3.dev.jsonl'),Path('data/fresh/fresh-v1.jsonl'),
             Path('runs/robust/fresh-v2-holdback/examples.jsonl'),OLD/'selection.jsonl',OLD/'confirmation.jsonl',
             TAIL/'confirmation.jsonl']
    for path in exposed:
        before=len(blocked)
        for r in previous.rows(path):mark(r)
        audit[str(path)]={'new_blocked':len(blocked)-before,'sha256':train.sha256_file(path)}
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile('data/jeval/data/test.parquet').iter_batches(columns=['state']):
        for r in batch.to_pylist():blocked.update(key_ids.get(('canonical_state',state_key(r['state'])),()))
    audit['after_entire_jeval']=len(blocked)
    # Official holdback cases also exclude every model-development dev state.
    # Unobserved gold6-dev cases remain eligible only in their own suite.
    all_dev_keys=set()
    for name in ['v3-gold2','v3-gold3','v3-gold6']:
        path=Path('data/mix')/(name+'.dev.jsonl')
        for r in previous.rows(path):all_dev_keys.update(old.keys(r))
    for r in candidates:
        if r['suite']=='official_fresh' and any(k in all_dev_keys for k in old.keys(r)):blocked.add(r['id'])
    for path_s,sha in prior['input_sha256'].items():
        path=Path(path_s)
        if train.sha256_file(path)!=sha:raise ValueError('upstream training mix changed')
        before=len(blocked);n=0
        for line in path.open():
            r=json.loads(line);n+=1;mark(r)
            if n%200000==0:print('AUDIT',path_s,n,flush=True)
        audit[path_s]={'rows':n,'sha256':sha,'new_blocked':len(blocked)-before}
    blocked_groups={groups[i] for i in blocked}
    renderer,_=search.renderer_for(search.BASE)
    by=defaultdict(lambda:defaultdict(list))
    for r in candidates:
        if groups[r['id']] in blocked_groups:continue
        if len(views.view(renderer,r)['input_ids'])>p['max_eval_tokens']:continue
        by[r['source']][groups[r['id']]].append({**r,'group':groups[r['id']]})
    chosen=[];available={}
    for source,gg in sorted(by.items()):
        available[source]=len(gg)
        if len(gg)<20:continue
        for g in sorted(gg,key=lambda g:search.digest(['delta-memory-confirm-v1',g]))[:256]:
            chosen.append(min(gg[g],key=lambda r:search.digest(['delta-memory-representative-v1',r['id']])))
    if len(chosen)<500:raise ValueError('insufficient fresh cases')
    splits={'train':previous.rows(out/'train.jsonl'),'selection':previous.rows(out/'selection.jsonl'),'confirmation':chosen}
    pilot.check_disjoint(splits);search.save_records(out/'confirmation.jsonl',chosen)
    control=OLD/'vanilla';control_files={}
    for s in p['snapshots']:
        for path in list((control/f'step-{s}').iterdir())+[control/f'selection-{s}.json',control/f'selection-{s}.jsonl']:
            if path.name in ['model.safetensors','config.json','v3_config.json'] or path.parent==control:
                control_files[str(path)]=train.sha256_file(path)
    control_files[str(control/'train.jsonl')]=train.sha256_file(control/'train.jsonl')
    if len(previous.rows(control/'train.jsonl'))!=512:raise ValueError('incomplete cached control')
    source=implementation();(out/'source').mkdir()
    for path in source:shutil.copyfile(path,out/'source'/Path(path).name)
    manifest={'plan':p,'counts':{k:len(v) for k,v in splits.items()},
              'sources':{k:dict(Counter(r['source'] for r in v)) for k,v in splits.items()},
              'confirmation_suites':dict(Counter(r['suite'] for r in chosen)),'exact_candidate_duplicates_removed':duplicates,
              'confirmation_groups':len({r['group'] for r in chosen}),'available_unobserved_groups':available,
              'files':{k+'.jsonl':train.sha256_file(out/(k+'.jsonl')) for k in splits},
              'implementation':source,'base_sha256':prior['base_sha256'],'cached_control_files':control_files,
              'audit':audit,'raw_snapshots':read('runs/robust/fresh-raw-snapshots.json'),
              'scope':'exclude whole linked upstream/document/state/normalized-state groups matching full gold2/3/6 training, old dev/fresh/recurrent/tail confirmation and all JEVal; official rows additionally exclude all gold2/3/6 dev; not semantic paraphrase audit'}
    train.atomic_json(out/'manifest.json',manifest)
    print('PREPARED',manifest['counts'],len(manifest['sources']['confirmation']),manifest['confirmation_suites'],flush=True)


def verify(out):
    m=read(out/'manifest.json')
    if implementation()!=m['implementation']:raise ValueError('source changed after preparation')
    for name,sha in m['files'].items():
        if train.sha256_file(out/name)!=sha:raise ValueError('experimental input changed')
    for path,sha in m['cached_control_files'].items():
        if train.sha256_file(Path(path))!=sha:raise ValueError('cached control changed')
    if train.sha256_file(search.BASE/'model.safetensors')!=m['base_sha256']:raise ValueError('base changed')
    return m


def load_arm(path=search.BASE,install=True):
    model=dm.load(path);text=model.model.language_model
    if install and not hasattr(text,'_decision_delta_memory'):dm.install(text,4)
    model.requires_grad_(False);text.requires_grad_(True);model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(text,'_decision_delta_memory'):dm.disable_native_checkpointing(text)
    return model


def optimizer_for(model,p):
    opt=train.MasterAdamW(model,p['body_lr']);text=model.model.language_model
    ids={id(q) for q in dm.routing_parameters(text).values()}
    masters=[m for q,m in opt.pairs if id(q) in ids];mid={id(m) for m in masters}
    for group in opt.param_groups:group['params']=[m for m in group['params'] if id(m) not in mid]
    opt.optimizer.add_param_group({'params':masters,'weight_decay':0.,'lr_scale':p['router_lr']/p['body_lr']})
    return opt


def fit(out,smoke=False):
    m=verify(out);p=m['plan'];dest=out/('smoke' if smoke else 'delta')
    if dest.exists():raise ValueError('preserve existing fit')
    dest.mkdir();renderer,cfg=search.renderer_for(search.BASE)
    rows=previous.rows(out/'train.jsonl');selection=previous.rows(out/'selection.jsonl')
    control=previous.rows(OLD/'vanilla/train.jsonl')
    train.seed_all(p['seed']);model=load_arm();opt=optimizer_for(model,p)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    if not smoke:
        score,records=old.evaluate(model,renderer,selection)
        train.atomic_json(dest/'selection-0.json',score);search.save_records(dest/'selection-0.jsonl',records)
        print('SELECTION',0,score['macro'],flush=True)
    rng=random.Random(p['seed']+1000);steps=3 if smoke else p['steps']
    with (dest/'train.jsonl').open('w') as log:
        for step,(indices,_) in enumerate(train.batches(len(rows),p['batch_size'],steps,random.Random(p['seed'])),1):
            model.train();train.set_checkpointing(model,True);opt.zero_grad()
            encoded=[views.random_view(renderer,rows[i],rng) for i in indices]
            h=search.digest([r['input_ids'] for r in encoded])
            if h!=control[step-1]['prompt_sha256']:raise ValueError('native control prompt differs')
            loss_sum=0.
            for group in train.microbatches(encoded,BATCH):
                z,target=train.logits_for(model,group,renderer.tokenizer.pad_token_id)
                loss=train.soft_ce(z,target).sum()/len(indices)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();loss_sum+=loss.item()
            norm=opt.clip_grad_norm(1.)
            if not math.isfinite(norm):raise ValueError('nonfinite gradients')
            scale=min(step/p['warmup'],1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-p['warmup'])/(p['steps']-p['warmup']))))
            for group in opt.param_groups:group['lr']=p['body_lr']*scale*group['lr_scale']
            opt.step();params=dm.routing_parameters(model.model.language_model)
            gates=torch.stack([v.detach() for k,v in params.items() if k.endswith('gate')])
            entry={'step':step,'loss':loss_sum,'grad_norm':norm,'seconds':time.monotonic()-started,
                   'prompt_sha256':h,'rows_seen':step*len(indices),'gate_rms':gates.square().mean().sqrt().item()}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%16==0 or smoke:print('FIT',entry,flush=True)
            if not smoke and step in p['snapshots']:
                dm.save(model,renderer.tokenizer,dest/f'step-{step}',cfg|{'steps':step,'trained_from':str(search.BASE),'experiment':'delta-memory-block4-v1'})
                score,records=old.evaluate(model,renderer,selection)
                train.atomic_json(dest/f'selection-{step}.json',score);search.save_records(dest/f'selection-{step}.jsonl',records)
                print('SELECTION',step,score['macro'],flush=True)
    summary={'steps':steps,'seconds':time.monotonic()-started,
             'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad),
             'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'all_prompt_hashes_match_native':True}
    if smoke:
        opt.zero_grad();_,expected=old.evaluate(model,renderer,rows[:8])
        dm.save(model,renderer.tokenizer,dest/'checkpoint',cfg)
        del opt,model;gc.collect();torch.cuda.empty_cache()
        model=dm.load(dest/'checkpoint').requires_grad_(False)
        _,actual=old.evaluate(model,renderer,rows[:8])
        mismatch=sum(a['prediction']!=b['prediction'] for a,b in zip(expected,actual))
        delta=max(abs(a-b) for aa,bb in zip(expected,actual) for a,b in zip(aa['probabilities'],bb['probabilities']))
        if mismatch or delta>1e-6:raise ValueError('smoke checkpoint changed computation')
        summary.update(reload_prediction_mismatches=mismatch,reload_max_probability_delta=delta)
    train.atomic_json(dest/'result.json',summary)


def freeze(out):
    m=verify(out)
    if (out/'frozen.json').exists():raise ValueError('already frozen')
    choices={}
    for arm,dest in [('vanilla',OLD/'vanilla'),('delta',out/'delta')]:
        choices_at=[(read(dest/f'selection-{s}.json')['macro'],s) for s in m['plan']['snapshots']]
        score,step=sorted(choices_at,key=lambda x:(-x[0],x[1]))[0];path=dest/f'step-{step}'
        choices[arm]={'path':str(path),'step':step,'selection_macro':score,
                      'files':{p.name:train.sha256_file(p) for p in path.iterdir() if p.name in
                               {'model.safetensors','config.json','v3_config.json','decision_architecture.json','delta_memory.safetensors'}}}
    train.atomic_json(out/'frozen.json',{'choices':choices,'confirmation_sha256':m['files']['confirmation.jsonl']})
    print('FROZEN',choices,flush=True)


def confirm(out):
    m=verify(out);frozen=read(out/'frozen.json')
    if (out/'confirmation-started.json').exists():raise ValueError('confirmation already opened')
    train.atomic_json(out/'confirmation-started.json',{'frozen_sha256':train.sha256_file(out/'frozen.json')})
    renderer,_=search.renderer_for(search.BASE);rows=previous.rows(out/'confirmation.jsonl')
    result={'manifest':m,'frozen':frozen,'arms':{}};records_by={}
    for arm,choice in frozen['choices'].items():
        path=Path(choice['path'])
        for name,sha in choice['files'].items():
            if train.sha256_file(path/name)!=sha:raise ValueError('frozen checkpoint changed')
        model=load_arm(path,arm=='delta').requires_grad_(False)
        metrics,records=old.evaluate(model,renderer,rows)
        result['arms'][arm]=metrics;records_by[arm]=records
        search.save_records(out/f'confirmation-{arm}.jsonl',records)
        train.atomic_json(out/f'confirmation-{arm}.json',metrics)
        print('CONFIRM',arm,metrics['macro'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    delta=evaluation.paired_interval(records_by['vanilla'],records_by['delta'],confidence=.95)
    delta.pop('paired_document_bootstrap_97_5ci');result['difference']=delta
    train.atomic_json(out/'results.json',result);print('DIFFERENCE',delta,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','fit','freeze','confirm'])
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage in ['smoke','fit']:fit(a.output,a.stage=='smoke')
    elif a.stage=='freeze':freeze(a.output)
    else:confirm(a.output)


if __name__=='__main__':main()
