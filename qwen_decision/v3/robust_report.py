"""Freeze all combination families before shared confirmation; export joint CIs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from qwen_decision.v3 import train
from qwen_decision.v3 import readout_experiment as previous, robust_experiment as study


def confirmation(out):
    required=['frozen.json','shortlist-frozen.json','memory-frozen.json']
    for name in required:
        if not (out/name).exists():raise ValueError('freeze every family first: '+name)
    protocol=out/'joint-confirmation-plan.json'
    memory=study.read(out/'memory-frozen.json')
    count=4 if memory['confirmation_eligible'] else 3
    if not protocol.exists():
        if (out/'confirmation-started.json').exists():
            raise ValueError('joint freeze must precede all confirmation scores')
        train.atomic_json(protocol,{'frozen_sha256':{name:train.sha256_file(out/name) for name in required},
            'comparisons':count,'two_sided_confidence':1-.05/count,'bootstrap_draws':5000,
            'criterion':'lower bound of joint Bonferroni paired source-stratified document bootstrap interval > 0',
            'all_candidates_frozen_before_shared_confirmation':True,
            'memory_included':memory['confirmation_eligible']})
    plan=study.read(protocol)
    for name,sha in plan['frozen_sha256'].items():
        if train.sha256_file(out/name)!=sha:raise ValueError('frozen family changed: '+name)
    stages=[('robust_experiment','confirmation-results.json'),('shortlist_experiment','shortlist-confirmation.json')]
    if plan['memory_included']:stages.append(('memory_experiment','memory-confirmation.json'))
    for module,result in stages:
        if (out/result).exists():continue
        subprocess.run([sys.executable,'-m','qwen_decision.v3.'+module,'confirm','--output',str(out)],check=True)


def export(out,destination):
    plan=study.read(out/'joint-confirmation-plan.json')
    manifest=study.read(out/'manifest.json');study.verify(out,manifest)
    for name,sha in plan['frozen_sha256'].items():
        if train.sha256_file(out/name)!=sha:raise ValueError('candidate changed since joint freeze')
    base=previous.rows(out/'confirmation-base.jsonl')
    expected={r['id'] for r in previous.rows(out/'confirmation.jsonl')}
    if {r['id'] for r in base}!=expected:raise ValueError('baseline confirmation ids differ')
    robust_result=study.read(out/'confirmation-results.json')
    shortlist=study.read(out/'shortlist-confirmation.json')
    arms={}
    for family in ['single','multi','shortlist']+(['memory'] if plan['memory_included'] else []):
        rows=previous.rows(out/f'confirmation-{family}.jsonl')
        if {r['id'] for r in rows}!=expected:raise ValueError('candidate ids differ')
        delta=study.paired_interval(base,rows,draws=plan['bootstrap_draws'],confidence=plan['two_sided_confidence'])
        delta['paired_document_bootstrap_joint_ci']=delta.pop('paired_document_bootstrap_97_5ci')
        detail=robust_result['arms'][family] if family in ('single','multi') else shortlist if family=='shortlist' else study.read(out/'memory-confirmation.json')
        arms[family]={'candidate':detail['candidate'],'metrics':study.search.metrics(rows),'vs_base':delta}
    fits=[]
    for p in sorted((out/'fits').glob('*/result.json')):
        result=study.read(p)
        steps=previous.rows(p.parent/'train.jsonl')
        result['last_64_mean_ce']=sum(r['ce'] for r in steps[-64:])/len(steps[-64:])
        result['last_64_mean_js']=sum(r['js'] for r in steps[-64:])/len(steps[-64:])
        fits.append(result)
    report={'baseline':study.search.metrics(base),'arms':arms,'joint_plan':plan,
        'confirmed_improvements':[name for name,arm in arms.items() if arm['vs_base']['confirmed_improvement']],
        'manifest':manifest,'fits':fits,'baseline_selection':study.read(out/'baseline.json'),
        'environment':study.read(out/'environment.json') if (out/'environment.json').exists() else None,
        'memory_screen':study.read(out/'memory-frozen.json'),
        'shortlist_screen':study.read(out/'shortlist-frozen.json'),
        'robust_screen':study.read(out/'frozen.json'),
        'reload_mismatches':robust_result['reload_mismatches']|{'shortlist':shortlist['reload_mismatches']},
        'artifact_sha256':{str(p.relative_to(out)):train.sha256_file(p) for p in sorted(out.glob('confirmation-*.jsonl'))},
        'limitations':['g4b historically used full gold3 dev for early stopping',
            'selection documents have been reused in multiple searches',
            'training is a small 4096-example two-view adaptation per seed, not full-corpus retraining',
            'source-stratified bootstrap is conditional on these 39 sources; no population-wide architecture claim',
            'confirmation documents are now observed and must not be used to select another recipe']}
    train.atomic_json(destination,report)
    print(json.dumps({'baseline':report['baseline']['macro'],'methods':{name:{'macro':arm['metrics']['macro'],**arm['vs_base']} for name,arm in arms.items()}},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['confirm','export'])
    p.add_argument('--output',type=Path,default=study.OUT)
    p.add_argument('--destination',type=Path,default=Path('wiki/project/robust-combinations-results.json'))
    args=p.parse_args()
    if args.stage=='confirm':confirmation(args.output)
    else:export(args.output,args.destination)


if __name__=='__main__':main()
