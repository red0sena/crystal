"""Numerical/latency feasibility only: no accuracy training or holdback search."""
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from qwen_decision.v3 import architecture_search as search, decision_tail as dt, format as fmt
from qwen_decision.v3 import modeling, readout_experiment as previous, train

OUT=Path('runs/decision-tail/feasibility-v1')


@torch.no_grad()
def main():
    if OUT.exists():raise ValueError('preserve probes: choose a new output directory')
    OUT.mkdir(parents=True)
    renderer,cfg=search.renderer_for(search.BASE)
    rows=sorted(previous.rows('runs/recurrent/internal4-v1/selection.jsonl'),
                key=lambda r:search.digest(['decision-tail-feasibility-v1',r['id']]))[:512]
    encoded=[]
    for r in rows:
        codes=renderer.codes[:len(r['target'])]
        e=fmt.encode_options(renderer.tokenizer,r['state'],r['question'],codes)
        encoded.append({**r,**e,'code_ids':renderer.code_ids[:len(codes)]})
    groups=list(train.microbatches(encoded,SimpleNamespace(pack=False,micro_batch=4,token_budget=4096)))
    inputs=[]
    for group in groups:
        ids,mask,codes,valid,target=train.collate(group,renderer.tokenizer.pad_token_id)
        positions=torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in group],device='cuda')
        inputs.append((ids,mask,codes,valid,positions))
    model=dt_model=modeling.load(search.BASE,attention=cfg['attention'],deltanet=cfg['deltanet']).requires_grad_(False)
    text=model.model.language_model
    tail=dt.install(text)
    model.eval()
    def run(mode,save=False,subset=None):
        outputs=[]
        ctx=dt.native(text) if mode=='native' else nullcontext()
        with ctx:
            for ids,mask,codes,valid,positions in inputs[:subset] if subset is not None else inputs:
                if mode=='native':z=modeling.code_logits(model,ids,mask,codes,valid)
                else:z=dt.logits(model,ids,mask,codes,valid,positions,mode)
                if save:outputs.append(z.float().cpu())
        torch.cuda.synchronize()
        return outputs
    for mode in ['native','single','candidates']:run(mode,subset=8)
    reference=run('native',save=True)
    check={}
    for mode in ['single','candidates']:
        actual=run(mode,save=True)
        check[mode]={'prediction_mismatches':sum((a.argmax(-1)!=b.argmax(-1)).sum().item() for a,b in zip(reference,actual)),
                     'max_abs_logit_delta':max((a-b).abs().max().item() for a,b in zip(reference,actual)),
                     'max_abs_probability_delta':max((a.softmax(-1)-b.softmax(-1)).abs().max().item() for a,b in zip(reference,actual))}
    timings={k:[] for k in ['native','single','candidates']}
    orders=[['native','single','candidates'],['single','candidates','native'],['candidates','native','single']]
    for order in orders:
        for mode in order:
            torch.cuda.synchronize();started=time.perf_counter();run(mode)
            timings[mode].append(time.perf_counter()-started)
            print('TIMING',mode,round(timings[mode][-1],4),flush=True)
    med={k:statistics.median(v) for k,v in timings.items()}
    report={'base':str(search.BASE),'base_sha256':train.sha256_file(search.BASE/'model.safetensors'),
            'scope':'untrained zero-adapter numerical equivalence and prepared-GPU-input forward timing; not an accuracy improvement experiment',
            'rows':len(rows),'sources':len({r['source'] for r in rows}),'batches':len(inputs),
            'tokens':sum(len(r['input_ids']) for r in encoded),
            'median_candidates':statistics.median(len(r['target']) for r in rows),
            'candidate_count_histogram':dict(Counter(len(r['target']) for r in rows)),
            'adapter_parameters':sum(p.numel() for m in [tail.up,tail.down] for p in m.parameters()),
            'original_weights_unchanged':True,'check':check,'timings_seconds':timings,'median_seconds':med,
            'relative_median_time':{k:v/med['native'] for k,v in med.items()},
            'timing_protocol':'all native weights fixed, BF16, padded SDPA, batch4/4096 tokens; warm 8 batches per mode; three rotated-order passes, synchronized; pre-tokenized/collated GPU inputs',
            'implementation':{p.name:train.sha256_file(p) for p in [Path(__file__),Path(dt.__file__)]}}
    train.atomic_json(OUT/'results.json',report)
    print('PROBE',json.dumps(report),flush=True)


if __name__=='__main__':main()
