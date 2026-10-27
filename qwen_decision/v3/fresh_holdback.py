"""One new gold holdback for the already fixed g6/g3 soup; no recipe search.

Uses unused official validation cases from the same source inventory as fresh-v1.
Exclude whole normalized-state groups in fresh-v1, JEVal, training and gold dev.
"""
import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from qwen_decision.v3 import architecture_search as search,fresh_eval as fresh,modeling,train
from qwen_decision.v3 import readout_experiment as previous,robust_adaptation as robust,robust_experiment as study
from qwen_decision.v3.reference_comparison import REBUILT,REVISION,as_example,skill,FRESH

OUT=Path('runs/robust/fresh-v2-holdback')
EXPECTED_SOUP='c90f56aea77f0dfa5fc4c50b7fe08cbe1ca5a168eefa8648ee97b5d15b18ac2c'


def state_key(value):
    if isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:value=json.loads(value)
        except (ValueError,TypeError):pass
    text=value if isinstance(value,str) else json.dumps(value,sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(fresh.norm(text).encode()).hexdigest()


def keys(row):
    return [('id',str(row.get('upstream_id',row['id']))),('state',state_key(row['state']))]


def prepare(out):
    if out.exists():raise ValueError('preserve holdback: use a new output directory')
    if train.sha256_file(REBUILT/'model.safetensors')!=EXPECTED_SOUP:raise ValueError('fixed soup changed')
    # Plan is fixed before any model execution on these new cases.
    plan={'candidate':str(REBUILT),'baseline':str(search.BASE),'recipe':'fixed g6 .67 + g3 epoch2 .33; never reselect',
          'sampling':'all non-GoEmotions fresh-v1 source families with >=20 unused normalized-state groups; <=200 groups/source; lowest fixed hash, one representative/group',
          'selection':'none; this file is evaluation only','metric':'source macro accuracy','bootstrap':5000,'confidence':.95,
          'max_tokens':4096,'no_truncation':True,'teacher_labels_generated':False,
          'source_inventory':'exact fresh-v1 builder, official validation/test rows; exclude all old fresh-v1 and JEVal states',
          'implementation_sha256':train.sha256_file(Path(__file__))}
    out.mkdir(parents=True);train.atomic_json(out/'plan.json',plan)
    original=list(fresh.rows())
    candidates=[as_example(r) for r in original if r['source']!='goemotions']
    key_ids=defaultdict(set)
    for row in candidates:
        for key in keys(row):key_ids[key].add(row['id'])
    blocked=set();audit={}
    def mark(row):
        found=False
        for key in keys(row):
            matches=key_ids.get(key,())
            if matches:blocked.update(matches);found=True
        return found
    for row in previous.rows(FRESH):mark(row)
    audit['after_fresh_v1_blocked_ids']=len(blocked)
    # Exclude the entire local JEVal file, not just the earlier scored subset.
    pf=pq.ParquetFile('data/jeval/data/test.parquet')
    for batch in pf.iter_batches(columns=['state']):
        for row in batch.to_pylist():
            blocked.update(key_ids.get(('state',state_key(row['state'])),()))
    audit['after_entire_jeval_blocked_ids']=len(blocked)
    # Verify that the restored files are the actual training corpora of all
    # three model ingredients/baseline, not new samples of those datasets.
    expected={}
    for name,run in [('v3-gold2','g3-gold2-ep3-lr3e-6-a0'),('v3-gold6','g6-gold6-pad0')]:
        path=hf_hub_download('Kyunghwan-kim/kkh-decision-model',f'runs/v3/{run}/manifest.json',revision=REVISION,
                             local_dir='runs/robust/hf-reference-a0b44b4')
        expected[name]=study.read(path)['data']['mix_sha256']
    expected['v3-gold3']=study.read(Path('runs/robust/search-v1/manifest.json'))['mix_sha256']['data/mix/v3-gold3.jsonl']
    for name in ['v3-gold2','v3-gold3','v3-gold6']:
        path=Path('data/mix')/(name+'.jsonl')
        sha=train.sha256_file(path)
        if sha!=expected[name]:raise ValueError('actual training mix hash differs: '+name)
        before=len(blocked);n=matches=0
        for line in path.open():
            row=json.loads(line);n+=1;matches+=mark(row)
            if n%200000==0:print('TRAIN AUDIT',name,n,flush=True)
        audit[name]={'rows':n,'sha256':sha,'training_rows_sharing_candidate_keys':matches,'new_blocked_candidate_ids':len(blocked)-before}
        dev=path.with_suffix('.dev.jsonl')
        if dev.exists():
            for row in previous.rows(dev):mark(row)
    # Every sibling sharing a state with a blocked row must leave the holdback.
    blocked_states={state_key(row['state']) for row in candidates if row['id'] in blocked}
    by=defaultdict(lambda:defaultdict(list))
    for row in candidates:
        group=state_key(row['state'])
        if row['id'] not in blocked and group not in blocked_states:by[row['source']][group].append(row)
    renderer,_=search.renderer_for(search.BASE)
    chosen=[];available={};dropped_long=0
    for source,groups in sorted(by.items()):
        usable=[]
        for group,rows in groups.items():
            row=min(rows,key=lambda r:search.digest(['fresh-v2-row',r['id']]))
            row={**row,'group':group}
            if len(robust.view(renderer,row)['input_ids'])<=4096:usable.append(row)
            else:dropped_long+=1
        available[source]=len(usable)
        if len(usable)>=20:
            chosen += sorted(usable,key=lambda r:search.digest(['fresh-v2-20261019',r['group']]))[:200]
    if not chosen:raise ValueError('no unused holdback remains')
    search.save_records(out/'examples.jsonl',chosen)
    manifest={'plan':plan,'rows':len(chosen),'sources':dict(Counter(r['source'] for r in chosen)),
              'available_unseen_groups':available,'original_raw_rows':len(original),'audit':audit,
              'excluded_groups':len(blocked_states),'dropped_over_4096':dropped_long,
              'audit_scope':'upstream id and lowercase punctuation/whitespace-normalized complete state, including parsed/canonical JSON; not semantic near-duplicate detection',
              'no_training_or_previous_eval_shared_keys_in_retained_cases':True,
              'examples_sha256':train.sha256_file(out/'examples.jsonl'),
              'model_sha256':{str(p):train.sha256_file(p/'model.safetensors') for p in [search.BASE,REBUILT]},
              'raw_dataset_snapshots':study.read('runs/robust/fresh-raw-snapshots.json'),
              'prior_fresh_sha256':train.sha256_file(FRESH)}
    train.atomic_json(out/'manifest.json',manifest)
    print('PREPARED NEW HOLDOUT',manifest['rows'],manifest['sources'],flush=True)


def confirm(out):
    if (out/'results.json').exists():raise ValueError('holdback already observed')
    m=study.read(out/'manifest.json')
    if train.sha256_file(Path(__file__))!=m['plan']['implementation_sha256']:raise ValueError('implementation changed')
    if train.sha256_file(out/'examples.jsonl')!=m['examples_sha256']:raise ValueError('holdback changed')
    for p,sha in m['model_sha256'].items():
        if train.sha256_file(Path(p)/'model.safetensors')!=sha:raise ValueError('fixed checkpoint changed')
    examples=previous.rows(out/'examples.jsonl');renderer,_=search.renderer_for(search.BASE)
    result={'manifest':m,'models':{},'new_cases_not_used_to_select_recipe':True};record_map={}
    for name,path in [('g4b',search.BASE),('soup',REBUILT)]:
        cfg=modeling.read_config(path)
        model=modeling.load(path,attention=cfg['attention'],deltanet=cfg['deltanet'],kernel='sdpa').requires_grad_(False)
        metrics,records=study.evaluate(model,renderer,examples)
        record_map[name]=records
        search.save_records(out/(name+'.jsonl'),records)
        result['models'][name]=skill(examples,records)|{'runtime':metrics}
        train.atomic_json(out/'progress.json',result)
        print('NEW HOLDOUT',name,result['models'][name]['macro_accuracy'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    delta=study.paired_interval(record_map['g4b'],record_map['soup'],confidence=.95)
    delta['paired_document_bootstrap_95ci']=delta.pop('paired_document_bootstrap_97_5ci')
    result['difference']=delta
    train.atomic_json(out/'results.json',result)
    train.atomic_json(Path('wiki/project/g6-soup-holdback-results.json'),result)
    print('NEW HOLDOUT DIFFERENCE',delta,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','confirm']);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    (prepare if args.stage=='prepare' else confirm)(args.output)


if __name__=='__main__':main()
