"""Verify rebuilt soup via the existing Decision Index Scorer API.

PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.validate_reference_engine
"""
from pathlib import Path

from qwen_decision.v3 import train
from qwen_decision.v3.engine import Scorer
from qwen_decision.v3.reference_comparison import FRESH, ROOT, REBUILT
from qwen_decision.v3.readout_experiment import rows, read_json


def main():
    source=rows(FRESH)
    saved={r['id']:r for r in rows(ROOT/'g6w67_g3e2.jsonl')}
    model=Scorer(REBUILT,batch_tokens=4096,max_batch=4,orders=1)
    calls=[]
    hook=model.model.model.language_model.register_forward_hook(lambda *unused:calls.append(1))
    requests=[(r['state'],{'q':r['question']}) for r in source]
    items=[((i,key),ids,codes,order) for i,(state,questions) in enumerate(requests)
           for key,ids,codes,order in model.prompts(state,questions)]
    expected_batches=len(list(model.batches(items)))
    output=model.score_many(requests)
    hook.remove()
    mismatch=0;max_difference=0.
    for row,result in zip(source,output):
        p=result['q'];previous=saved[row['id']]
        mismatch+=max(range(len(p)),key=p.__getitem__)!=previous['prediction']
        max_difference=max(max_difference,max(abs(a-b) for a,b in zip(p,previous['probabilities'])))
    evidence={'rows':len(source),'prediction_mismatches':mismatch,'max_probability_difference':max_difference,
              'actual_text_forward_calls':len(calls),'expected_batches':expected_batches,
              'one_forward_per_batch':len(calls)==expected_batches,'orders':1,
              'checkpoint_sha256':train.sha256_file(REBUILT/'model.safetensors'),
              'decision_index_commit':'f5a6e4fd85fed05e77878dfc4966d0568bfb45ec',
              'dependency_access':'pinned repository submodule on PYTHONPATH; original huggingface-hub 2.2.0 retained',
              'implementation_sha256':train.sha256_file(Path(__file__))}
    if mismatch or len(calls)!=expected_batches:raise ValueError(evidence)
    train.atomic_json(ROOT/'engine-validation.json',evidence)
    train.atomic_json(REBUILT/'engine-validation.json',evidence)
    result=read_json(ROOT/'results.json');result['engine_validation']=evidence
    train.atomic_json(ROOT/'results.json',result)
    train.atomic_json(Path('wiki/project/g6-soup-replication-results.json'),result)
    print(evidence,flush=True)


if __name__=='__main__':main()
