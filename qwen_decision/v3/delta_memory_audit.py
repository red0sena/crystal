"""Post-freeze checkpoint, engine, active depth-routing and latency verification."""
from collections import Counter
from contextlib import nullcontext
import gc
from pathlib import Path
import statistics
import time

import torch

from qwen_decision.v3 import architecture_search as search, delta_memory as dm, modeling, train
from qwen_decision.v3 import readout_experiment as previous
from qwen_decision.v3 import delta_memory_experiment as e


def main():
    out=e.OUT;m=e.verify(out);f=e.read(out/'frozen.json')
    if not (out/'results.json').exists():raise ValueError('complete primary confirmation first')
    renderer,_=search.renderer_for(search.BASE)
    selection=previous.rows(out/'selection.jsonl')
    rows=sorted(selection,key=lambda r:search.digest(['delta-memory-timing-v1',r['id']]))[:512]
    encoded=[e.views.view(renderer,r) for r in rows]
    inputs=[train.collate(g,renderer.tokenizer.pad_token_id)[:4] for g in train.microbatches(encoded,e.BATCH)]
    requests=[(r['state'],{'q':r['question']}) for r in rows]
    report={'rows':len(rows),'sources':len({r['source'] for r in rows}),
            'tokens':sum(len(r['input_ids']) for r in encoded),'batches':len(inputs),'arms':{},
            'timing_protocol':'RTX3090 BF16 padded SDPA microbatch4/4096; 8 warm GPU batches; 3 timed passes; separate checkpoint loads; no concurrent training',
            'implementation_sha256':train.sha256_file(Path(__file__))}
    from qwen_decision.v3.engine import Scorer
    expected=None
    for arm,choice in f['choices'].items():
        path=Path(choice['path'])
        for name,sha in choice['files'].items():
            if train.sha256_file(path/name)!=sha:raise ValueError('frozen checkpoint changed')
        scorer=Scorer(path,max_batch=4,batch_tokens=4096,orders=1)
        model=scorer.model.requires_grad_(False).eval()
        @torch.no_grad()
        def run(subset=None):
            for ids,mask,codes,valid in inputs[:subset] if subset else inputs:
                modeling.code_logits(model,ids,mask,codes,valid)
            torch.cuda.synchronize()
        run(8);timings=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter();run();timings.append(time.perf_counter()-start)
        scorer.score_many(requests[:16]);torch.cuda.synchronize();end_to_end=[]
        for _ in range(3):
            start=time.perf_counter();scorer.score_many(requests);torch.cuda.synchronize()
            end_to_end.append(time.perf_counter()-start)
        report['arms'][arm]={'gpu_seconds':timings,'gpu_median_seconds':statistics.median(timings),
                              'engine_seconds':end_to_end,'engine_median_seconds':statistics.median(end_to_end),
                              'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9}
        print('TIMING',arm,report['arms'][arm],flush=True)
        if arm=='delta':
            _,replay=e.old.evaluate(model,renderer,selection)
            original=previous.rows(path.parent/f"selection-{choice['step']}.jsonl")
            mismatch=sum(a['prediction']!=b['prediction'] for a,b in zip(replay,original))
            diff=max(abs(a-b) for aa,bb in zip(replay,original) for a,b in zip(aa['probabilities'],bb['probabilities']))
            if len(replay)!=len(original) or mismatch or diff>1e-6:raise ValueError('checkpoint reload changed selection')
            report['training_checkpoint_reload']={'rows':len(replay),'prediction_mismatches':mismatch,'max_probability_delta':diff}
            _,expected=e.old.evaluate(model,renderer,rows[:128])
            params=dm.routing_parameters(model.model.language_model)
            gates={k:v.detach().clone() for k,v in params.items() if k.endswith('gate')}
            with torch.no_grad():
                for k in gates:params[k].zero_()
            _,without=e.old.evaluate(model,renderer,rows[:128])
            with torch.no_grad():
                for k,v in gates.items():params[k].copy_(v)
            report['zero_gate_ablation_selection']={'rows':128,
                'changed_predictions':sum(a['prediction']!=b['prediction'] for a,b in zip(expected,without)),
                'max_probability_delta':max(abs(a-b) for aa,bb in zip(expected,without) for a,b in zip(aa['probabilities'],bb['probabilities']))}
            report['gate_rms']={k:v.square().mean().sqrt().item() for k,v in gates.items()}
            report['query_rms']={k:v.detach().square().mean().sqrt().item() for k,v in params.items() if k.endswith('query')}
            # Selection-only diagnostic: keep native weights/gates and local
            # block updates, but erase access to earlier block-update sources.
            def erase_history(module,args):
                packed=args[0].reshape(*args[0].shape[:-1],module.groups+1,module.hidden).clone()
                packed[...,1:,:]=0
                return (packed.flatten(-2),)+args[1:]
            hooks=[b.register_forward_pre_hook(erase_history) for b in model.model.language_model.layers if b.index>0]
            try:_,local_only=e.old.evaluate(model,renderer,rows[:128])
            finally:
                for h in hooks:h.remove()
            report['erase_earlier_delta_ablation_selection']={'rows':128,
                'changed_predictions':sum(a['prediction']!=b['prediction'] for a,b in zip(expected,local_only)),
                'max_probability_delta':max(abs(a-b) for aa,bb in zip(expected,local_only) for a,b in zip(aa['probabilities'],bb['probabilities']))}
        del scorer,model;gc.collect();torch.cuda.empty_cache()
    # Independent native engine load, not a copy of the evaluator object.
    scorer=Scorer(f['choices']['delta']['path'],max_batch=4,batch_tokens=4096,orders=1)
    text=scorer.model.model.language_model
    calls=[];layer_calls=Counter();energy=Counter();raw_end={};routing={}
    handles=[text.register_forward_hook(lambda *a:calls.append(1))]
    for i,block in enumerate(text.layers):
        for j,layer in enumerate(block.native):
            handles.append(layer.register_forward_hook(lambda m,a,o,k=(i,j):layer_calls.update([k])))
        if hasattr(block,'gate'):
            handles.append(block.native[-1].register_forward_hook(lambda m,a,o,i=i:raw_end.update({i:o})))
            def measure(module,args,kwargs,output,i=i):
                baseline=raw_end.pop(i)
                main=output[...,:module.hidden]
                valid=kwargs['attention_mask'][0][:,0,0]
                base=baseline[valid].float();delta=main[valid].float()-base
                energy.update(delta_squared=delta.square().sum().item(),input_squared=base.square().sum().item(),
                              changed=delta.ne(0).sum().item(),elements=delta.numel())
                states=output.reshape(*output.shape[:-1],module.groups+1,module.hidden)
                values=torch.stack([torch.zeros_like(baseline)]+[states[...,j+1,:] for j in range(module.index+1)]).float()
                keys=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+1e-6)
                weights=(keys*module.query).sum(-1).softmax(0)
                stat=routing.setdefault(i,Counter())
                stat.update(max_weight_sum=weights.max(0).values[valid].sum().item(),
                            null_weight_sum=weights[0][valid].sum().item(),
                            earlier_weight_sum=weights[1:-1].sum(0)[valid].sum().item(),tokens=valid.sum().item())
            handles.append(block.register_forward_hook(measure,with_kwargs=True))
    sample=requests[:128]
    items=[((i,item[0]),*item[1:]) for i,(state,questions) in enumerate(sample) for item in scorer.prompts(state,questions)]
    batches=len(list(scorer.batches(items)));actual=scorer.score_many(sample)
    for h in handles:h.remove()
    mismatch=sum(max(range(len(a['q'])),key=a['q'].__getitem__)!=r['prediction'] for a,r in zip(actual,expected))
    diff=max(abs(a-b) for aa,bb in zip(actual,expected) for a,b in zip(aa['q'],bb['probabilities']))
    if mismatch or diff>1e-5 or len(calls)!=batches or len(layer_calls)!=24 or set(layer_calls.values())!={batches}:
        raise ValueError({'mismatch':mismatch,'max_delta':diff,'text_calls':len(calls),'batches':batches,'layers':dict(layer_calls)})
    report['engine']={'rows':128,'prediction_mismatches':mismatch,'max_probability_delta':diff,
                      'one_backbone_call_per_batch':True,'native_blocks_executed':24,'calls':len(calls),'orders':1,
                      'router_relative_activation_norm':(energy['delta_squared']/energy['input_squared'])**.5,
                      'router_changed_feature_fraction':energy['changed']/energy['elements']}
    report['routing_selection']={str(i):{'mean_max_weight':v['max_weight_sum']/v['tokens'],
        'uniform_max_weight':1/(i+2),'mean_null_weight':v['null_weight_sum']/v['tokens'],
        'mean_earlier_block_weight':v['earlier_weight_sum']/v['tokens'],'tokens':v['tokens']} for i,v in routing.items()}
    # Hold weights fixed, alternate only the computation graph for timing.
    paired={'native':[],'delta':[]}
    for mode in ['native','delta']:
        with dm.native(text) if mode=='native' else nullcontext():scorer.score_many(requests[:16])
        torch.cuda.synchronize()
    for order in [('native','delta'),('delta','native'),('native','delta')]:
        for mode in order:
            with dm.native(text) if mode=='native' else nullcontext():
                torch.cuda.synchronize();start=time.perf_counter();scorer.score_many(requests);torch.cuda.synchronize()
                paired[mode].append(time.perf_counter()-start)
    report['paired_engine_timing']={'seconds':paired,'median_seconds':{k:statistics.median(v) for k,v in paired.items()},
        'delta_to_native_ratio':statistics.median(paired['delta'])/statistics.median(paired['native']),
        'protocol':'same trained delta checkpoint, native versus delta graph; same512 raw requests, 16-request warmup and 3 alternating-order score_many passes; includes tokenization/padding/transfer/probability conversion, excludes model loading; timing only'}
    train.atomic_json(out/'audit.json',report);print('AUDIT',report,flush=True)


if __name__=='__main__':main()
