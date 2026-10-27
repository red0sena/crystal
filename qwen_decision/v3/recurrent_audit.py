"""Reference metrics and real Decision Index engine validation after recipe freeze.

PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.recurrent_audit
"""
from collections import Counter
import argparse
import gc
from pathlib import Path

import torch

from qwen_decision.v3 import architecture_search as search, recurrent, train
from qwen_decision.v3 import readout_experiment as previous
from qwen_decision.v3.recurrent_experiment import OUT, read, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    out = parser.parse_args().output
    result = read(out/"results.json")
    rows = previous.rows(out/"confirmation.jsonl")
    renderer,_ = search.renderer_for(search.BASE)
    model = recurrent.load(search.BASE).requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    metrics, records = evaluate(model,renderer,rows)
    metrics["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated()/1e9
    search.save_records(out/"confirmation-original.jsonl",records)
    train.atomic_json(out/"original-reference.json",metrics)
    del model; gc.collect(); torch.cuda.empty_cache()

    choice = result["frozen"]["choices"]["recurrent"]
    path = Path(choice["path"])
    for name,sha in choice["checkpoint_files"].items():
        if train.sha256_file(path/name)!=sha: raise ValueError("frozen model changed")
    sample = previous.rows(out/"selection.jsonl")[:128]
    model = recurrent.load(path).requires_grad_(False)
    _, expected = evaluate(model,renderer,sample)
    text = model.model.language_model
    loop = recurrent.loop_module(text) if hasattr(text,"_decision_recurrence") else None
    gate = loop.recurrence_gate.detach().cpu() if loop is not None else None
    gate_stats = ({"rms":gate.square().mean().sqrt().item(), "min":gate.min().item(),
                   "max":gate.max().item(), "nonzero_channels":int(gate.ne(0).sum())} if gate is not None else None)
    del text,loop,model; gc.collect(); torch.cuda.empty_cache()

    from qwen_decision.v3.engine import Scorer
    scorer = Scorer(path,batch_tokens=4096,max_batch=4,orders=1)
    calls,layer_calls = [], Counter()
    branch_energy = Counter()
    text = scorer.model.model.language_model
    handle = text.register_forward_hook(lambda *unused:calls.append(1))
    handles = []
    for i,layer in enumerate(text.layers):
        if not isinstance(layer,recurrent.RecurrentPass):
            handles.append(layer.register_forward_hook(
                lambda m,a,o,i=i:layer_calls.update([i])))
        else:
            def measure_branch(module,args,kwargs,output):
                valid=kwargs["attention_mask"][:,0,0].bool()
                original=args[0][valid].float()
                delta=output[valid].float()-original
                branch_energy.update(delta_squared=delta.square().sum().item(),
                                     input_squared=original.square().sum().item(),
                                     changed=int(delta.ne(0).sum()),elements=delta.numel())
            handles.append(layer.register_forward_hook(measure_branch,with_kwargs=True))
    requests=[(r["state"],{"q":r["question"]}) for r in sample]
    items=[((i,key),ids,codes,order) for i,(state,questions) in enumerate(requests)
           for key,ids,codes,order in scorer.prompts(state,questions)]
    expected_batches=len(list(scorer.batches(items)))
    output=scorer.score_many(requests)
    handle.remove()
    for h in handles:h.remove()
    mismatch=0;max_delta=0.
    for ref,actual in zip(expected,output):
        p=actual["q"]
        mismatch+=max(range(len(p)),key=p.__getitem__)!=ref["prediction"]
        max_delta=max(max_delta,max(abs(a-b) for a,b in zip(p,ref["probabilities"])))
    executed=sum(layer_calls.values())/len(calls)
    expected_depth=28 if gate is not None else 24
    if mismatch or max_delta>1e-5 or executed!=expected_depth or len(calls)!=expected_batches:
        raise ValueError({"mismatches":mismatch,"max_delta":max_delta,"depth":executed})
    audit={"rows":len(sample),"prediction_mismatches":mismatch,"max_probability_delta":max_delta,
           "text_forward_calls":len(calls),"executed_transformer_layers_per_forward":executed,
           "expected_batches":expected_batches,"one_text_forward_per_batch":len(calls)==expected_batches,
           "orders":1,"gate":gate_stats,"original_reference":metrics,
           "implementation_sha256":train.sha256_file(Path(__file__))}
    if branch_energy:
        audit["extra_path_relative_activation_norm"]=(branch_energy["delta_squared"]/branch_energy["input_squared"])**.5
        audit["extra_path_changed_feature_fraction"]=branch_energy["changed"]/branch_energy["elements"]
    train.atomic_json(out/"audit.json",audit)
    print(audit,flush=True)


if __name__=="__main__":main()
