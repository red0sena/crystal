"""Candidate shortlisting plus probability-preserving reranking, no new labels.

Uses robust_experiment's prepared splits. Freeze this independent family before
any shared confirmation results. Joint reports adjust for all three frozen arms.
This is a shortlist reranker inspired by Runoff, not its full yes/no algorithm.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from qwen_decision.v3 import architecture_search as search, format as fmt, train
from qwen_decision.v3 import readout_experiment as previous
from qwen_decision.v3 import robust_adaptation as robust, robust_experiment as study


def shortlist_row(example, selected):
    keys = list(example['question']['criteria'])
    if len(set(selected))!=len(selected) or not selected:
        raise ValueError('distinct nonempty candidates required')
    q = {**example['question'], 'criteria':{keys[i]:example['question']['criteria'][keys[i]] for i in selected}}
    # These targets are never used to select candidates. Keep their original
    # entries even when the shortlist excludes every gold option.
    return {**example,'question':q,'target':[example['target'][i] for i in selected]}


def distribute(original, selected, conditional, alpha):
    """Redistribute original shortlist mass; retain all outside probabilities."""
    p = list(original)
    mass = sum(p[i] for i in selected)
    for i,q in zip(selected,conditional):
        p[i] = (1-alpha)*p[i]+alpha*mass*q
    return p


@torch.no_grad()
def conditional_predictions(model, renderer, examples, base, topk, cyclic):
    encoded = []
    selected = {}
    for row in examples:
        k = len(row['target'])
        protected = row['ordered'] or renderer.positional(row) or row['question']['type']!='choice'
        if protected or k<=2:
            continue
        # Select from model probabilities only; never inspect target values.
        keep = sorted(sorted(range(k),key=lambda i:(-base[row['id']]['probabilities'][i],i))[:topk])
        selected[row['id']] = keep
        smaller = shortlist_row(row,keep)
        encoded += robust.cyclic_views(renderer,smaller) if cyclic else [robust.view(renderer,smaller)]
    predictions = defaultdict(list)
    for group in train.microbatches(encoded,study.BATCH):
        z,_ = train.logits_for(model,group,renderer.tokenizer.pad_token_id)
        p = robust.align(z,group).softmax(-1).cpu()
        for row,probs in zip(group,p):
            predictions[row['id']].append(probs[:len(row['target'])].tolist())
    return {'selected':selected,'conditional':{key:torch.tensor(p).mean(0).tolist() for key,p in predictions.items()},
            'prompt_evaluations':len(encoded),'tokens':sum(len(r['input_ids']) for r in encoded)}


def records_for(examples,base,conditionals,alpha):
    records=[]
    for row in examples:
        key=row['id']
        p=base[key]['probabilities']
        if key in conditionals['selected']:
            p=distribute(p,conditionals['selected'][key],conditionals['conditional'][key],alpha)
        pred=max(range(len(p)),key=p.__getitem__)
        records.append({k:row[k] for k in ('id','source','group')} |
                       {'probabilities':p,'prediction':pred,'correct':row['target'][pred]>=max(row['target'])-1e-6})
    return records


def plan(out):
    destination=out/'shortlist-plan.json'
    if destination.exists():
        return
    if (out/'shortlist-frozen.json').exists() or (out/'confirmation-started.json').exists():
        raise ValueError('cannot add a method after shared confirmation')
    train.atomic_json(destination,{'topk':[2,4],'cyclic':[False,True],'alpha':[.5,1.],
        'candidate_count':8,'selection':'original fixed 1474 rows only; one family winner, deterministic tie by smaller K then no cyclic then smaller alpha',
        'preserve':'ordered, positional, noul, and already binary questions retain their base predictions',
        'probabilities':'redistribute existing shortlist mass; all outside probabilities unchanged',
        'gold':'gold used only for scoring, never for choosing the shortlist; no new teacher labels',
        'confirmation':'same unobserved documents as robust study; freeze before ANY of the three confirm arms',
        'joint_multiple_comparisons':'three frozen arms: 98.333333% two-sided Bonferroni intervals',
        'implementation':{Path(__file__).name:train.sha256_file(Path(__file__)),Path(robust.__file__).name:train.sha256_file(Path(robust.__file__))}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','screen','confirm'])
    p.add_argument('--output',type=Path,default=study.OUT)
    args=p.parse_args();out=args.output
    if args.stage=='plan':
        plan(out);return
    protocol=study.read(out/'shortlist-plan.json')
    for name,sha in protocol['implementation'].items():
        if train.sha256_file(Path(__file__).with_name(name))!=sha:
            raise ValueError('shortlist source changed')
    manifest=study.read(out/'manifest.json');study.verify(out,manifest)
    if args.stage=='screen':
        if (out/'shortlist-frozen.json').exists() or (out/'confirmation-started.json').exists():
            raise ValueError('preserve frozen screen')
        examples=previous.rows(out/'selection.jsonl')
        base={r['id']:r for r in previous.rows(out/'selection-base-single.jsonl')}
    else:
        if (out/'shortlist-confirmation.json').exists():raise ValueError('already confirmed')
        frozen=study.read(out/'shortlist-frozen.json')
        if not (out/'frozen.json').exists():raise ValueError('robust candidates not frozen')
        # Robust confirmation supplies identical baseline probabilities. It is
        # only read after this family's model-independent recipe is frozen.
        examples=previous.rows(out/'confirmation.jsonl')
        base={r['id']:r for r in previous.rows(out/'confirmation-base.jsonl')}
    model=study.load_model();renderer,_=search.renderer_for(search.BASE)
    if args.stage=='screen':
        candidates=[]
        for topk in protocol['topk']:
            for cyclic in protocol['cyclic']:
                tick=__import__('time').monotonic()
                cond=conditional_predictions(model,renderer,examples,base,topk,cyclic)
                for alpha in protocol['alpha']:
                    records=records_for(examples,base,cond,alpha)
                    name=f'shortlist-k{topk}-cyclic{int(cyclic)}-a{alpha:g}'
                    score=search.metrics(records)
                    item={'name':name,'topk':topk,'cyclic':cyclic,'alpha':alpha,'selection':score,
                          'rerank_prompt_evaluations':cond['prompt_evaluations'],'rerank_tokens':cond['tokens'],
                          'shared_rerank_seconds':__import__('time').monotonic()-tick}
                    candidates.append(item)
                    search.save_records(out/f'selection-{name}.jsonl',records)
                    print('SHORTLIST',name,score['macro'],flush=True)
        winner=sorted(candidates,key=lambda c:(-c['selection']['macro'],c['topk'],c['cyclic'],c['alpha']))[0]
        train.atomic_json(out/'shortlist-frozen.json',{'winner':winner,'candidates':candidates,'before_shared_confirmation':True,
            'plan_sha256':train.sha256_file(out/'shortlist-plan.json'),'confirmation_sha256':manifest['files']['confirmation.jsonl']})
    else:
        spec=frozen['winner']
        # Re-run the selected selection recipe from its frozen specification.
        selected=previous.rows(out/'selection.jsonl')
        selected_base={r['id']:r for r in previous.rows(out/'selection-base-single.jsonl')}
        cond=conditional_predictions(model,renderer,selected,selected_base,spec['topk'],spec['cyclic'])
        reloaded=records_for(selected,selected_base,cond,spec['alpha'])
        old={r['id']:r['prediction'] for r in previous.rows(out/f"selection-{spec['name']}.jsonl")}
        mismatch=sum(old[r['id']]!=r['prediction'] for r in reloaded)
        if mismatch:raise ValueError('shortlist selection reload differs')
        tick=__import__('time').monotonic()
        cond=conditional_predictions(model,renderer,examples,base,spec['topk'],spec['cyclic'])
        records=records_for(examples,base,cond,spec['alpha'])
        search.save_records(out/'confirmation-shortlist.jsonl',records)
        result={'candidate':spec,'metrics':search.metrics(records),'reload_mismatches':mismatch,
                'vs_base':study.paired_interval(list(base.values()),records,confidence=1-.05/3),
                'additional_prompt_evaluations':cond['prompt_evaluations'],'additional_tokens':cond['tokens'],
                'additional_seconds':__import__('time').monotonic()-tick}
        # The legacy key names the 97.5 interval in the robust-only two-arm
        # helper. Label the actual, wider joint interval explicitly here.
        result['vs_base']['paired_document_bootstrap_joint_98_333ci']=result['vs_base'].pop('paired_document_bootstrap_97_5ci')
        train.atomic_json(out/'shortlist-confirmation.json',result)
        print('CONFIRM SHORTLIST',result['metrics']['macro'],result['vs_base'],flush=True)


if __name__=='__main__':main()
