"""Verify exported engine behavior and matched warmed inference cost, after freeze."""
from collections import Counter
from contextlib import nullcontext
import gc
from pathlib import Path
import statistics
import time

import torch

from qwen_decision.v3 import architecture_search as search, decision_tail as dt, modeling, train
from qwen_decision.v3 import readout_experiment as previous
from qwen_decision.v3 import decision_tail_experiment as experiment


def main():
    out = experiment.OUT
    m = experiment.verify(out); frozen = experiment.read(out/"frozen.json")
    if not (out/"results.json").exists(): raise ValueError("complete primary confirmation first")
    renderer,_ = search.renderer_for(search.BASE)
    rows = sorted(previous.rows(out/"selection.jsonl"),
                  key=lambda r:search.digest(["decision-tail-feasibility-v1",r["id"]]))[:512]
    encoded = [experiment.encode(renderer,r) for r in rows]
    inputs = []
    for group in train.microbatches(encoded,experiment.BATCH):
        ids,mask,codes,valid,_ = train.collate(group,renderer.tokenizer.pad_token_id)
        positions = torch.tensor([r["option_positions"]+[0]*(codes.shape[1]-len(r["target"]))
                                  for r in group],device=ids.device)
        inputs.append((ids,mask,codes,valid,positions))
    report = {"rows":len(rows),"sources":len({r["source"] for r in rows}),"batches":len(inputs),
              "tokens":sum(len(r["input_ids"]) for r in encoded),"arms":{},
              "timing_protocol":"BF16 SDPA, microbatch4/4096 tokens; fixed prepared GPU inputs; 8 warm batches per arm, 3 timed passes, synchronize; excludes loading/tokenization/collation; arms loaded separately, not concurrent or round-robin",
              "implementation_sha256":train.sha256_file(Path(__file__))}
    from qwen_decision.v3.engine import Scorer
    requests_all=[(r["state"],{"q":r["question"]}) for r in rows]
    expected = None
    for arm,choice in frozen["choices"].items():
        path = Path(choice["path"])
        for name,sha in choice["files"].items():
            if train.sha256_file(path/name)!=sha: raise ValueError("frozen model changed")
        scorer=Scorer(path,max_batch=4,batch_tokens=4096,orders=1)
        model=scorer.model.requires_grad_(False).eval()
        train.set_checkpointing(model,False)
        @torch.no_grad()
        def run(subset=None):
            for ids,mask,codes,valid,positions in (inputs[:subset] if subset else inputs):
                if arm=="vanilla": modeling.code_logits(model,ids,mask,codes,valid)
                else: dt.logits(model,ids,mask,codes,valid,positions,arm)
            torch.cuda.synchronize()
        run(8); timings=[]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize(); started=time.perf_counter(); run()
            timings.append(time.perf_counter()-started)
        report["arms"][arm]={"seconds":timings,"median_seconds":statistics.median(timings),
                              "peak_cuda_allocated_gb":torch.cuda.max_memory_allocated()/1e9}
        scorer.score_many(requests_all[:16]); torch.cuda.synchronize()
        end_to_end=[]
        for _ in range(3):
            started=time.perf_counter(); scorer.score_many(requests_all); torch.cuda.synchronize()
            end_to_end.append(time.perf_counter()-started)
        report["arms"][arm]["engine_end_to_end_seconds"]=end_to_end
        report["arms"][arm]["engine_end_to_end_median_seconds"]=statistics.median(end_to_end)
        if arm=="candidates":
            _,replayed=experiment.evaluate(model,renderer,previous.rows(out/'selection.jsonl'),arm)
            recorded=previous.rows(path.parent/f"selection-{choice['step']}.jsonl")
            reload_mismatch=sum(a['prediction']!=b['prediction'] for a,b in zip(replayed,recorded))
            reload_delta=max(abs(a-b) for aa,bb in zip(replayed,recorded) for a,b in zip(aa['probabilities'],bb['probabilities']))
            if len(replayed)!=len(recorded) or reload_mismatch or reload_delta>1e-6:
                raise ValueError('trained candidate checkpoint reload changed selection predictions')
            report['training_checkpoint_reload']={'rows':len(recorded),'prediction_mismatches':reload_mismatch,'max_probability_delta':reload_delta}
            _,expected=experiment.evaluate(model,renderer,rows[:128],arm)
            tail=model.model.language_model.layers[-1]
            original=tail.down.weight.detach().clone()
            report["adapter_down_rms"]=original.square().mean().sqrt().item()
            with torch.no_grad(): tail.down.weight.zero_()
            _,without=experiment.evaluate(model,renderer,rows[:128],arm)
            report["conditioner_zero_ablation_selection"]={
                "rows":128,"changed_predictions":sum(a["prediction"]!=b["prediction"] for a,b in zip(expected,without)),
                "max_probability_delta":max(abs(a-b) for aa,bb in zip(expected,without) for a,b in zip(aa["probabilities"],bb["probabilities"]))}
            with torch.no_grad(): tail.down.weight.copy_(original)
        del scorer,model; gc.collect(); torch.cuda.empty_cache()
    # An independent exported-model load through the actual Decision Index path.
    scorer=Scorer(frozen["choices"]["candidates"]["path"],max_batch=4,batch_tokens=4096,orders=1)
    requests=[(r["state"],{"q":r["question"]}) for r in rows[:128]]
    calls=[]; layer_calls=Counter(); query_rows=[];memory_rows=[]
    text=scorer.model.model.language_model
    handles=[text.register_forward_hook(lambda *a:calls.append(1))]
    for i,layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(lambda m,a,o,i=i:layer_calls.update([i])))
    tail=text.layers[-1]
    handles += [tail.native.self_attn.q_proj.register_forward_pre_hook(lambda m,a:query_rows.append(a[0].shape[1])),
                tail.native.self_attn.k_proj.register_forward_pre_hook(lambda m,a:memory_rows.append(a[0].shape[1]))]
    items=[((r,item[0]),*item[1:]) for r,(state,questions) in enumerate(requests)
           for item in scorer.prompts(state,questions)]
    groups=list(scorer.batches(items))
    actual=scorer.score_many(requests)
    for h in handles:h.remove()
    mismatch=sum(max(range(len(a['q'])),key=a['q'].__getitem__)!=r['prediction'] for a,r in zip(actual,expected))
    max_delta=max(abs(a-b) for aa,bb in zip(actual,expected) for a,b in zip(aa['q'],bb['probabilities']))
    expected_q=[max(len(items[i][2]) for i in g) for g in groups]
    expected_t=[max(len(items[i][1]) for i in g) for g in groups]
    if (mismatch or max_delta>1e-5 or len(calls)!=len(groups) or
        set(layer_calls.values())!={len(groups)} or query_rows!=expected_q or memory_rows!=expected_t):
        raise ValueError({"mismatch":mismatch,"probability_delta":max_delta,"calls":len(calls),"batches":len(groups)})
    report["engine"]={"rows":128,"prediction_mismatches":mismatch,"max_probability_delta":max_delta,
                       "one_backbone_call_per_batch":True,"executed_blocks_per_forward":24,
                       "query_rows_equal_padded_candidate_count":True,"key_rows_equal_prompt_length":True,
                       "calls":len(calls),"orders":1}
    native=report['arms']['vanilla']['median_seconds']
    report['relative_median_time']={k:v['median_seconds']/native for k,v in report['arms'].items()}
    engine_native=report['arms']['vanilla']['engine_end_to_end_median_seconds']
    report['relative_engine_end_to_end_median_time']={k:v['engine_end_to_end_median_seconds']/engine_native for k,v in report['arms'].items()}
    report['engine_timing_protocol']='score_many on same 512 raw requests, includes tokenization/option offsets/padding/host-to-device transfer/probabilities; 16-request warmup, 3 passes; no model loading, orders=1'
    # Timing only: hold the final candidate weights fixed and alternate the
    # original computation and trained candidate computation on the same GPU.
    # This is not another accuracy candidate or a model combination.
    paired={'vanilla':[],'candidates':[]}
    try:
        for mode in ['vanilla','candidates']:
            scorer.decision_tail_mode=None if mode=='vanilla' else 'candidates'
            with dt.native(text) if mode=='vanilla' else nullcontext():
                scorer.score_many(requests_all[:16])
            torch.cuda.synchronize()
        for order in [('vanilla','candidates'),('candidates','vanilla'),('vanilla','candidates')]:
            for mode in order:
                scorer.decision_tail_mode=None if mode=='vanilla' else 'candidates'
                with dt.native(text) if mode=='vanilla' else nullcontext():
                    torch.cuda.synchronize(); started=time.perf_counter()
                    scorer.score_many(requests_all); torch.cuda.synchronize()
                    paired[mode].append(time.perf_counter()-started)
    finally:
        scorer.decision_tail_mode='candidates'
    report['paired_engine_timing']={'seconds':paired,
        'median_seconds':{k:statistics.median(v) for k,v in paired.items()},
        'candidate_to_native_ratio':statistics.median(paired['candidates'])/statistics.median(paired['vanilla']),
        'protocol':'same trained candidate checkpoint; native/full final block versus candidate final block; 512 identical raw requests, 16-request warmup, 3 alternating-order passes; end-to-end score_many including option offsets; timing only, native path is not used as an accuracy arm'}
    train.atomic_json(out/'audit.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
