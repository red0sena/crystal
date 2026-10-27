"""Predeclared screen requiring all answers to leave through candidate workspace states.

No new confirmation is opened by this script. Reuse the verified native control,
including its per-update prompt hashes. An architecture must beat its best
selection score by at least 0.5 percentage points to merit a fresh confirmation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from qwen_decision.v3 import architecture_search as search, workspace_bottleneck as ws, modeling, train
from qwen_decision.v3 import hypothesis_workspace as core
from qwen_decision.v3 import decision_tail_experiment as encoding
from qwen_decision.v3 import readout_experiment as records, recurrent_experiment as control
from qwen_decision.v3 import robust_adaptation as views

OUT = Path('runs/hypothesis-workspace/bottleneck16-v1')
OLD = control.OUT


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    modules = [ws, core, modeling, train, search, encoding, encoding.fmt, records, control, views]
    return {str(p):train.sha256_file(p) for p in [Path(m.__file__) for m in modules] + [Path(__file__)]}


def prepare(out):
    if out.exists():
        raise ValueError('preserve existing trials; select a new output directory')
    prior = control.verify(OLD)
    plan = {**prior['plan'], 'arms':['cached_native', 'candidate_workspace_bottleneck'],
            'architecture':{'kind':'candidate_workspace_bottleneck','start':16,'scale':.5,
                'readout':'candidate workspace states only, diagonal native code rows; no direct context decision-token exit',
                'slots':'one per option, last + .5*rms(last)*identity_projection(rms(option-mean(options)))',
                'updates':'upper full-attention blocks update context and slots jointly; upper DeltaNet updates context, shared MLP updates slots',
                'position':'all slots use original decision position', 'new_parameters':1048576,
                'active_at_initialization':True, 'new_zero_gates':False},
            'workspace_lr':1e-4, 'gate_lr':None,
            'selection':'old exposed selection only; independent best of trained snapshots; no step-0 fallback',
            'early_stop':'stop at step256 if source macro is >0.5pp below cached step256 native',
            'advance_threshold_pp':.5,
            'confirmation':'none in this screening run; a new audited holdback is required if advanced',
            'no_new_teacher_labels_or_rationales':True,
            'comparison_scope':'same base, data, draw order, gold loss, body optimizer and updates; not matched FLOPs',
            'validation_limit':'selection is already exposed and repeatedly reused; no confirmatory accuracy claim'}
    out.mkdir(parents=True)
    train.atomic_json(out/'plan.json', plan)
    for name in ['train.jsonl','selection.jsonl']:
        shutil.copyfile(OLD/name, out/name)
    files = {name:train.sha256_file(out/name) for name in ['train.jsonl','selection.jsonl']}
    cached = {str(OLD/'vanilla/train.jsonl'):train.sha256_file(OLD/'vanilla/train.jsonl')}
    for step in plan['snapshots']:
        for ext in ['json','jsonl']:
            path = OLD/f'vanilla/selection-{step}.{ext}'
            cached[str(path)] = train.sha256_file(path)
        for name in ['model.safetensors','config.json','v3_config.json']:
            path = OLD/f'vanilla/step-{step}'/name
            cached[str(path)] = train.sha256_file(path)
    source = implementation()
    (out/'source').mkdir()
    for path in source:
        shutil.copyfile(path, out/'source'/Path(path).name)
    train.atomic_json(out/'manifest.json', {'plan':plan,'files':files,'implementation':source,
        'cached_files':cached,'base_sha256':prior['base_sha256'],
        'upstream_manifest_sha256':train.sha256_file(OLD/'manifest.json'),
        'upstream_split_audit':prior.get('audit',{}),
        'new_confirmation_cases_scored':0,
        'motivation':'previous workspace contents changed no predictions in the fixed 512-case path audit; remove the context exit rather than assuming added workspace is used'})
    print('PREPARED',plan['architecture'],flush=True)


def verify(out):
    m = read(out/'manifest.json')
    if implementation() != m['implementation']:
        raise ValueError('source changed after preparation')
    for name, sha in m['files'].items():
        if train.sha256_file(out/name) != sha:
            raise ValueError('input changed')
    for path, sha in m['cached_files'].items():
        if train.sha256_file(Path(path)) != sha:
            raise ValueError('cached control changed')
    if train.sha256_file(search.BASE/'model.safetensors') != m['base_sha256']:
        raise ValueError('base checkpoint changed')
    return m


def logits_for(model, rows, pad, **kwargs):
    ids, mask, codes, valid, targets = train.collate(rows,pad)
    positions = torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in rows],
                             device=ids.device)
    return ws.logits(model,ids,mask,codes,valid,positions,**kwargs), targets


@torch.no_grad()
def evaluate(model,renderer,rows,**kwargs):
    model.eval();train.set_checkpointing(model,False)
    encoded = [encoding.encode(renderer,r) for r in rows]
    probabilities = {}
    torch.cuda.synchronize();started = time.monotonic()
    for group in train.microbatches(encoded,control.BATCH):
        z,_ = logits_for(model,group,renderer.tokenizer.pad_token_id,**kwargs)
        for row,p in zip(group,z.softmax(-1).cpu()):
            probabilities[row['id']] = p[:len(row['target'])].tolist()
    torch.cuda.synchronize()
    result = []
    for row in rows:
        p = probabilities[row['id']];pred = max(range(len(p)),key=p.__getitem__)
        result.append({'id':row['id'],'source':row['source'],'group':row.get('group',row['id']),
                       'prediction':pred,'correct':row['target'][pred]>=max(row['target'])-1e-6,
                       'probabilities':p,'views':1})
    return search.metrics(result)|{'seconds':time.monotonic()-started},result


def load_arm(path=search.BASE):
    model = ws.load(path)
    text = model.model.language_model
    if not hasattr(text,'hypothesis_workspace'):
        ws.install(text,start=16,scale=.5)
    model.requires_grad_(False);text.requires_grad_(True);model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model


def optimizer(model,p):
    opt = train.MasterAdamW(model,p['body_lr'])
    ids = {id(q) for q in model.model.language_model.hypothesis_workspace.parameters()}
    masters = [m for q,m in opt.pairs if id(q) in ids];mids = {id(m) for m in masters}
    for group in opt.param_groups:
        group['params'] = [m for m in group['params'] if id(m) not in mids]
    opt.optimizer.add_param_group({'params':masters,'weight_decay':.01,
                                  'lr_scale':p['workspace_lr']/p['body_lr']})
    return opt


def fit(out,smoke=False):
    m = verify(out);p = m['plan'];dest = out/('smoke' if smoke else 'workspace')
    if dest.exists():
        raise ValueError('preserve existing fit')
    dest.mkdir()
    renderer,cfg = search.renderer_for(search.BASE)
    rows,selection = records.rows(out/'train.jsonl'),records.rows(out/'selection.jsonl')
    native_log = records.rows(OLD/'vanilla/train.jsonl')
    train.seed_all(p['seed']);model = load_arm();opt = optimizer(model,p)
    torch.cuda.reset_peak_memory_stats();started = time.monotonic()
    selected = []
    if not smoke:
        score,rr = evaluate(model,renderer,selection)
        train.atomic_json(dest/'selection-0.json',score);search.save_records(dest/'selection-0.jsonl',rr)
        print('SELECTION',0,score['macro'],flush=True)
    rng = random.Random(p['seed']+1000)
    steps = 3 if smoke else p['steps'];gate_gradient = None
    with (dest/'train.jsonl').open('w') as log:
        for step,(indices,_) in enumerate(train.batches(len(rows),p['batch_size'],steps,random.Random(p['seed'])),1):
            model.train();train.set_checkpointing(model,True);opt.zero_grad()
            encoded = [encoding.encode(renderer,rows[i],rng) for i in indices]
            prompt_hash = search.digest([r['input_ids'] for r in encoded])
            if prompt_hash != native_log[step-1]['prompt_sha256']:
                raise ValueError('native control prompt differs')
            loss_sum = 0.
            for group in train.microbatches(encoded,control.BATCH):
                z,target = logits_for(model,group,renderer.tokenizer.pad_token_id)
                loss = train.soft_ce(z,target).sum()/len(indices)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                loss.backward();loss_sum += loss.item()
            projection = model.model.language_model.hypothesis_workspace.projection.weight
            if step == 1:
                if any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters() if q.requires_grad):
                    raise ValueError('missing/nonfinite trainable gradient')
                gate_gradient = projection.grad.norm().item()
                if gate_gradient == 0:
                    raise ValueError('new route has no gradient')
            norm = opt.clip_grad_norm(1.)
            if not math.isfinite(norm):
                raise ValueError('nonfinite gradients')
            scale = min(step/p['warmup'],1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-p['warmup'])/(p['steps']-p['warmup']))))
            for group in opt.param_groups:
                group['lr'] = p['body_lr']*scale*group['lr_scale']
            opt.step()
            entry = {'step':step,'loss':loss_sum,'grad_norm':norm,'seconds':time.monotonic()-started,
                     'prompt_sha256':prompt_hash,'rows_seen':step*len(indices)}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%16 == 0 or smoke:
                print('FIT',entry,flush=True)
            if not smoke and step in p['snapshots']:
                ws.save(model,renderer.tokenizer,dest/f'step-{step}',cfg|{'steps':step,'trained_from':str(search.BASE),'experiment':'hypothesis-workspace-bottleneck16-v1'})
                score,rr = evaluate(model,renderer,selection)
                train.atomic_json(dest/f'selection-{step}.json',score);search.save_records(dest/f'selection-{step}.jsonl',rr)
                native = read(OLD/f'vanilla/selection-{step}.json')['macro']
                selected.append({'step':step,'macro':score['macro'],'native_macro':native})
                print('SELECTION',step,score['macro'],'native',native,flush=True)
                if step == 256 and score['macro'] < native-.005:
                    print('EARLY_STOP: predeclared selection deficit >0.5pp',flush=True)
                    break
    summary = {'steps':step,'seconds':time.monotonic()-started,'initial_workspace_gradient_norm':gate_gradient,
               'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad),
               'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
               'all_prompt_hashes_match_native':True,'new_confirmation_cases_scored':0}
    if smoke:
        opt.zero_grad();_,expected = evaluate(model,renderer,rows[:12])
        ws.save(model,renderer.tokenizer,dest/'checkpoint',cfg)
        del opt,model;gc.collect();torch.cuda.empty_cache()
        model = ws.load(dest/'checkpoint').requires_grad_(False)
        _,actual = evaluate(model,renderer,rows[:12])
        mismatch = sum(a['prediction']!=b['prediction'] for a,b in zip(expected,actual))
        delta = max(abs(a-b) for aa,bb in zip(expected,actual) for a,b in zip(aa['probabilities'],bb['probabilities']))
        if mismatch or delta>1e-6:
            raise ValueError('saved workspace changed computation')
        summary.update(reload_prediction_mismatches=mismatch,reload_max_probability_delta=delta)
    else:
        best = sorted(selected,key=lambda x:(-x['macro'],x['step']))[0]
        native_best = max(read(OLD/f'vanilla/selection-{s}.json')['macro'] for s in p['snapshots'])
        delta = best['macro']-native_best
        summary.update(snapshots=selected,selected=best,native_selected_macro=native_best,
                       selection_delta_pp=100*delta,advance_to_fresh_confirmation=delta>=p['advance_threshold_pp']/100,
                       conclusion='selection screen only; independent accuracy improvement is untested')
    train.atomic_json(dest/'result.json',summary)
    print('RESULT',summary,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','fit'])
    p.add_argument('--output',type=Path,default=OUT)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.output)
    else:
        fit(a.output,smoke=a.stage=='smoke')


if __name__ == '__main__':
    main()
