"""Re-evaluate a published, fixed g6/g3 soup against g4b on fresh-v1.

This is a replication on an already observed evaluation suite, not a new
held-out success for the independently trained robust adaptation experiment.
"""
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import statistics

import torch

from qwen_decision.v3 import architecture_search as search, format as fmt, modeling, train
from qwen_decision.v3 import readout_experiment as previous, robust_adaptation as robust, robust_experiment as study

ROOT = Path('runs/robust/reference-comparison')
REBUILT = Path('runs/robust/rebuilt-g6w67-g3e2')
FRESH = Path('data/fresh/fresh-v1.jsonl')
FRESH_SHA = '685535c3a81e0f27d56e74cbb21649b1aa715dae05ed1f7118542fd6eb42c89f'
REVISION = 'a0b44b4d531145ecb30136cea886381205132d93'


def as_example(row):
    keys,_ = fmt.options(row['question'])
    label = (1 if row['label'] else 0) if row['question']['type']=='noul' else keys.index(row['label'])
    return {**row,'target':[float(i==label) for i in range(len(keys))], 'ordered':False}


def skill(examples,records,exclude_emotions=False):
    by=defaultdict(list);chance=defaultdict(list)
    for row,record in zip(examples,records):
        if exclude_emotions and row['source']=='goemotions':continue
        by[row['source']].append(record['correct'])
        chance[row['source']].append(1/len(row['target']))
    sources={name:{'n':len(v),'accuracy':sum(v)/len(v),
                   'skill':max(0.,(sum(v)/len(v)-statistics.mean(chance[name]))/(1-statistics.mean(chance[name])))}
             for name,v in sorted(by.items())}
    return {'sources':sources,'macro_accuracy':statistics.mean(v['accuracy'] for v in sources.values()),
            'macro_skill':statistics.mean(v['skill'] for v in sources.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();out=args.output
    if out.exists():raise ValueError('preserve reference comparison: use a new directory')
    if train.sha256_file(FRESH)!=FRESH_SHA:raise ValueError('published fresh suite differs')
    raw=[as_example(r) for r in previous.rows(FRESH)]
    groups=previous.document_groups(raw)
    examples=[{**r,'group':groups[r['id']]} for r in raw]
    renderer,_=search.renderer_for(search.BASE)
    other,_=search.renderer_for(REBUILT)
    max_length=0
    for row in examples:
        a,b=robust.view(renderer,row),robust.view(other,row)
        if a['input_ids']!=b['input_ids']:raise ValueError('model prompts or tokenizer differ')
        max_length=max(max_length,len(a['input_ids']))
    out.mkdir(parents=True)
    checkpoint_paths=[search.BASE/'model.safetensors',REBUILT/'model.safetensors']
    plan={'models':{'g4b':str(search.BASE),'g6w67_g3e2':str(REBUILT)},
          'recipe':'0.67*g6-last + 0.33*g3-epoch-2, FP32 merge stored BF16; upstream recipe fixed before this replication',
          'hub_model_revision':REVISION,'hub_data_revision':'c15b054aafa3013cb421597031aee87ee150ad4b',
          'model_sha256':{str(p):train.sha256_file(p) for p in checkpoint_paths},
          'data_sha256':FRESH_SHA,'rows':len(examples),'max_tokens':max_length,'no_truncation':True,
          'primary':'15 sources excluding GoEmotions, a training source; also report complete 16-source suite for replication',
          'holdback':'published recipe was already evaluated on fresh-v1; this is repeat measurement, not a new untouched test',
          'upstream_aliases':{'g6-best':'last, both step 21874','g3-best':'epoch-2, selected step12500 of 6250 updates/epoch'},
          'precision':'same BF16 padded SDPA and same input token IDs for both models',
          'timing':'GPU may be shared with the separate adaptation training; do not use these times as latency comparisons'}
    train.atomic_json(out/'plan.json',plan)
    result={'plan':plan,'models':{}}
    record_map={}
    for name,path in plan['models'].items():
        config=modeling.read_config(Path(path))
        model=modeling.load(Path(path),attention=config['attention'],deltanet=config['deltanet'],kernel='sdpa')
        model.requires_grad_(False)
        metrics,records=study.evaluate(model,renderer,examples)
        search.save_records(out/(name+'.jsonl'),records)
        record_map[name]=records
        result['models'][name]={'all_16':skill(examples,records),'excluding_goemotions_15':skill(examples,records,True),
                                'runtime':metrics}
        train.atomic_json(out/'progress.json',result)
        print('FRESH REPLICATION',name,{k:v['macro_accuracy'] for k,v in result['models'][name].items() if k!='runtime'},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    base=[r for r in record_map['g4b'] if r['source']!='goemotions']
    new=[r for r in record_map['g6w67_g3e2'] if r['source']!='goemotions']
    result['primary_difference']=study.paired_interval(base,new,confidence=.95)
    result['primary_difference']['paired_document_bootstrap_95ci']=result['primary_difference'].pop('paired_document_bootstrap_97_5ci')
    result['source']=str(out)
    train.atomic_json(out/'results.json',result)
    train.atomic_json(Path('wiki/project/g6-soup-replication-results.json'),result)
    print('FRESH DIFFERENCE',result['primary_difference'],flush=True)


if __name__=='__main__':main()
