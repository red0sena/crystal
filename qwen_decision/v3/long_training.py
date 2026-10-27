"""Long, matched adaptation of native and previously screened architectures.

Data/plan preparation -> cache validated prompts -> real-model resume smoke
-> freeze -> run -> audit.
Original short trials remain immutable. No accuracy-based early stopping.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch

from qwen_decision.v3 import bdh_experiment as prior,bdh_memory as bdh
from qwen_decision.v3 import hypothesis_workspace,workspace_bottleneck as workspace,recurrent
from qwen_decision.v3 import decision_tail_experiment as encoding
from qwen_decision.v3 import modeling,train,architecture_search as search
from qwen_decision.v3 import readout_experiment as records,robust_adaptation as views
from qwen_decision.v3 import robust_experiment as statistics_helper

OUT = Path('runs/long-training/64k-4096-v1')
BATCH = prior.BATCH


def read(path):return json.loads(Path(path).read_text())


def implementation():
    modules = [prior,bdh,hypothesis_workspace,workspace,recurrent,encoding,modeling,
               train,search,records,views,statistics_helper,encoding.fmt,prior.old,prior.pilot]
    paths = [Path(m.__file__) for m in modules]+[Path(__file__),Path('qwen_decision/v3/fresh_holdback.py'),
             Path('qwen_decision/v3/mix.py')]
    return {str(p):train.sha256_file(p) for p in paths}


def prepare(out,scope):
    if out.exists():raise ValueError('preserve previous output')
    previous = prior.verify(prior.OUT)
    path = Path('data/mix/v3-gold6.jsonl')
    if train.sha256_file(path)!=previous['full_mix_sha256']:raise ValueError('training corpus changed')
    renderer,_ = search.renderer_for(search.BASE)
    old = records.rows(prior.OUT/'large.jsonl');sources = sorted({r['source'] for r in old})
    by = defaultdict(list)
    for r in prior.pilot.read_rows(path):
        if r['source'] in sources and prior.old.allowed_origin(r,{'human','programmatic'}):
            by[r['source']].append(r)
    base_by = defaultdict(list)
    for r in old:base_by[r['source']].append(r)
    pools = {}
    for s in sources:
        pool = list(base_by[s]);ids = {r['id'] for r in pool}
        candidates = sorted((r for r in by[s] if r['id'] not in ids),
                            key=lambda r:search.digest(['long-64k-v1',r['id']]))
        for r in candidates:
            if len(pool)>=1024:break
            if len(views.view(renderer,r)['input_ids'])<=1536:pool.append(r)
        pools[s] = pool
        print('POOL',s,len(pool),flush=True)
    del by;gc.collect()
    quota = {s:len(base_by[s]) for s in sources}
    while sum(quota.values())<65536:
        changed = False
        for s in sources:
            if quota[s]<len(pools[s]) and sum(quota.values())<65536:
                quota[s]+=1;changed=True
        if not changed:raise ValueError('insufficient eligible unique examples')
    rows = sorted([r for s in sources for r in pools[s][:quota[s]]],key=lambda r:r['id'])
    if len(rows)!=65536 or len({r['id'] for r in rows})!=65536:raise ValueError('wrong data size')
    if not {r['id'] for r in old}<={r['id'] for r in rows}:raise ValueError('prior data missing')
    selection = records.rows(prior.OUT/'selection.jsonl')
    prior.pilot.check_disjoint({'train':rows,'selection':selection})
    arms = ['native','bdh','recurrent','workspace'] if scope=='structures' else ['native','bdh']
    seeds = [41] if scope=='structures' else [41,73]
    p = {'scope':scope,'base':str(search.BASE),'arms':arms,'seeds':seeds,'steps':4096,'batch_size':16,
         'training_rows':65536,'exposures_per_fit':65536,'body_lr':3e-6,'bdh_lr':1e-4,
         'workspace_lr':1e-4,'recurrence_gate_lr':.001,'warmup':64,
         'schedule':'linear warmup then cosine to 10% over4096 updates',
         'snapshots':[1024,2048,4096],'max_training_tokens':1536,'max_eval_tokens':4096,
         'micro_batch':BATCH.micro_batch,'padded_token_budget':BATCH.token_budget,
         'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
         'optimizer':'BF16 forward/backward; FP32-master AdamW betas(.9,.95), WD.01, global clip1',
         'native':'original24 hybrid blocks and original tied code head',
         'bdh':'native16 plus shared BDH core3 cycles, omitted upper8 frozen/CPU',
         'recurrent':'native24 plus gated shared extra pass through layers16:20',
         'workspace':'candidate_workspace_bottleneck: native24 with candidate states after16; candidate-only code readout',
         'training':'all used language parameters and tied embedding/head; vision frozen',
         'initialization':'same original g4b for all arms; fresh fit, not optimizer-reset continuation',
         'primary':'fixed4096 update score; same prompt/target/code hashes across architectures per seed',
         'secondary':'1024/2048 learning trajectory; no best-snapshot reselection for primary comparison',
         'no_accuracy_early_stop':True,'no_model_or_prediction_averaging':True,'gold_ce_only':True,
         'resume':'full FP32 masters, Adam moments, parameter names and RNG saved at every snapshot',
         'confirmation':'no fresh inference in this runner; requires separate frozen candidate and audited holdback',
         'limits':['exposed selection, not independent confirmation','larger dataset and longer LR schedule than old pilots; not a step-only intervention',
                   'unchanged1536-token training filter, so longer-context training remains untested',
                   'same updates/data, not matched parameters/FLOPs','selected previous architectures, not all historical heads/settings',
                   'one seed in structures scope; two seeds in seeds scope do not fully estimate training variance']}
    out.mkdir(parents=True)
    train.atomic_json(out/'plan.json',p)
    for name,rr in [('train',rows),('selection',selection),('train-probe',records.rows(prior.OUT/'train-probe.jsonl'))]:
        search.save_records(out/(name+'.jsonl'),rr)
    train.atomic_json(out/'manifest.json',{'plan':p,'frozen':False,'base_sha256':previous['base_sha256'],
        'full_mix_sha256':previous['full_mix_sha256'],'upstream_manifest_sha256':train.sha256_file(prior.OUT/'manifest.json'),
        'files':{n:train.sha256_file(out/n) for n in ['train.jsonl','selection.jsonl','train-probe.jsonl','plan.json']},
        'sources':dict(Counter(r['source'] for r in rows)),
        'distinct_states':len({prior.state_key(r['state']) for r in rows}),
        'inherited_train_selection_audit':previous['scope'],'direct_disjoint_check':True,
        'new_confirmation_cases_scored':0})
    print('PREPARED',len(rows),len(sources),p['arms'],p['seeds'],flush=True)


def verify(out,*,frozen=True):
    m = read(out/'manifest.json')
    if frozen and not m['frozen']:raise ValueError('run smoke then freeze first')
    if m['frozen'] and implementation()!=m['implementation']:raise ValueError('frozen implementation changed')
    for name,sha in m['files'].items():
        if train.sha256_file(out/name)!=sha:raise ValueError('input/plan changed')
    if train.sha256_file(search.BASE/'model.safetensors')!=m['base_sha256']:raise ValueError('base changed')
    return m


def freeze(out):
    m = verify(out,frozen=False)
    if m['frozen']:raise ValueError('already frozen')
    if any(f'encoded-seed{s}.jsonl' not in m['files'] for s in m['plan']['seeds']):
        raise ValueError('cache all training inputs first')
    for arm in m['plan']['arms']:
        s = read(out/f'smoke-{arm}/result.json')
        if s['implementation']!=implementation() or not s['exact_optimizer_resume']:raise ValueError('smoke missing/stale')
    m.update(frozen=True,implementation=implementation())
    (out/'source').mkdir()
    for path in m['implementation']:shutil.copyfile(path,out/'source'/Path(path).name)
    train.atomic_json(out/'manifest.json',m)
    print('FROZEN',flush=True)


def load_arm(arm,path=search.BASE):
    torch.use_deterministic_algorithms(True)
    if arm=='recurrent':
        model = recurrent.load(path);text = model.model.language_model
        if not hasattr(text,'_decision_recurrence'):recurrent.install(text,16,20)
    elif arm=='workspace':
        model = workspace.load(path);text = model.model.language_model
        if not hasattr(text,'hypothesis_workspace'):workspace.install(text,16,.5)
    else:
        return prior.load_arm(arm,path)
    model.requires_grad_(False);text.requires_grad_(True);model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model


def optimizer(model,arm,p):
    opt = train.MasterAdamW(model,p['body_lr'])
    text = model.model.language_model
    if arm=='native':return opt
    if arm=='bdh':extra = list(text.bdh_memory.parameters());lr = p['bdh_lr'];wd = .01
    elif arm=='workspace':extra = list(text.hypothesis_workspace.parameters());lr = p['workspace_lr'];wd = .01
    else:extra = [recurrent.loop_module(text).recurrence_gate];lr = p['recurrence_gate_lr'];wd = 0.
    ids = {id(q) for q in extra};masters = [m for q,m in opt.pairs if id(q) in ids]
    mids = {id(q) for q in masters}
    for g in opt.param_groups:g['params'] = [q for q in g['params'] if id(q) not in mids]
    opt.optimizer.add_param_group({'params':masters,'weight_decay':wd,'lr_scale':lr/p['body_lr']})
    return opt


def encoded_batch(renderer,rows,indices,seed,step):
    return [encoding.encode(renderer,rows[i],random.Random(int(search.digest(['long-view-v1',seed,step,j])[:16],16)))
            for j,i in enumerate(indices)]


def cache(out):
    m = verify(out,frozen=False);p = m['plan']
    if m['frozen']:raise ValueError('do not mutate frozen inputs')
    renderer,_ = search.renderer_for(search.BASE);rows = records.rows(out/'train.jsonl')
    for seed in p['seeds']:
        path = out/f'encoded-seed{seed}.jsonl'
        if path.exists():
            if path.name in m['files'] and train.sha256_file(path)==m['files'][path.name]:continue
            raise ValueError('preserve existing unverified cache')
        tmp = path.with_suffix('.jsonl.tmp')
        with tmp.open('w') as f:
            for step,(indices,_) in enumerate(train.batches(len(rows),p['batch_size'],p['steps'],random.Random(seed)),1):
                rr = encoded_batch(renderer,rows,indices,seed,step)
                f.write(json.dumps({'step':step,'indices':indices,'rows':rr},ensure_ascii=False)+'\n')
                if step%256==0:print('CACHE',seed,step,flush=True)
        tmp.rename(path);m['files'][path.name] = train.sha256_file(path)
        train.atomic_json(out/'manifest.json',m)
    print('ALL_PROMPTS_VALIDATED',flush=True)


def logits_for(model,rr,pad,arm):
    ids,mask,codes,valid,targets = train.collate(rr,pad)
    if arm=='bdh':z = bdh.logits(model,ids,mask,codes,valid)
    elif arm=='workspace':
        pos = torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in rr],device=ids.device)
        z = workspace.logits(model,ids,mask,codes,valid,pos)
    else:z = modeling.code_logits(model,ids,mask,codes,valid)
    return z,targets


@torch.no_grad()
def evaluate(model,renderer,rows,arm):
    model.eval();train.set_checkpointing(model,False);probabilities = {}
    encoded = [encoding.encode(renderer,r) for r in rows]
    for group in train.microbatches(encoded,BATCH):
        z,_ = logits_for(model,group,renderer.tokenizer.pad_token_id,arm)
        for r,p in zip(group,z.softmax(-1).cpu()):probabilities[r['id']] = p[:len(r['target'])].tolist()
    result = []
    for r in rows:
        p = probabilities[r['id']];pred = max(range(len(p)),key=p.__getitem__)
        result.append({'id':r['id'],'source':r['source'],'group':r.get('group',r['id']),
                       'prediction':pred,'correct':r['target'][pred]>=max(r['target'])-1e-6,'probabilities':p})
    return search.metrics(result),result


def update(model,opt,renderer,rr,arm,p,step):
    model.train();train.set_checkpointing(model,True);opt.zero_grad();loss_sum = 0.
    for group in train.microbatches(rr,BATCH):
        z,target = logits_for(model,group,renderer.tokenizer.pad_token_id,arm)
        loss = train.soft_ce(z,target).sum()/len(rr)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();loss_sum+=loss.item()
    if step==1 and any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters() if q.requires_grad):
        raise ValueError('missing/nonfinite trainable gradient')
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):raise ValueError('nonfinite gradients')
    scale = min(step/p['warmup'],1.)*(.1+.45*(1+math.cos(math.pi*max(0,step-p['warmup'])/(p['steps']-p['warmup']))))
    for g in opt.param_groups:g['lr'] = p['body_lr']*g['lr_scale']*scale
    opt.step();return loss_sum,norm


def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x


def training_state(model,opt,step):
    return {'step':step,'names':[n for n,q in model.named_parameters() if q.requires_grad],
            'masters':[m.detach().cpu() for _,m in opt.pairs],
            'optimizer':cpu_tree(opt.optimizer.state_dict()),'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all()}


def save_checkpoint(model,opt,renderer,path,arm,cfg,step):
    if path.exists():raise ValueError('preserve existing checkpoint')
    opt.zero_grad();config = cfg|{'steps':step,'trained_from':str(search.BASE),'experiment':'long-training-64k-v1','arm':arm}
    save = {'native':modeling.save,'bdh':bdh.save,'recurrent':recurrent.save,'workspace':workspace.save}[arm]
    save(model,renderer.tokenizer,path,config)
    payload = training_state(model,opt,step)
    torch.save(payload,path/'optimizer.pt.tmp');(path/'optimizer.pt.tmp').rename(path/'optimizer.pt')
    del payload;gc.collect()
    files = {f.name:train.sha256_file(f) for f in path.iterdir() if f.name in
        ['model.safetensors','optimizer.pt','config.json','v3_config.json','decision_architecture.json',
         'bdh_memory.safetensors','hypothesis_workspace.safetensors','recurrence.safetensors']}
    train.atomic_json(path/'complete.json',{'step':step,'arm':arm,'files':files})


def verify_checkpoint(path):
    marker = read(path/'complete.json')
    for name,sha in marker['files'].items():
        if train.sha256_file(path/name)!=sha:raise ValueError('checkpoint changed')
    return marker


def restore_optimizer(model,opt,path):
    verify_checkpoint(path)
    # This pickle was created locally by this runner and is hash-checked above.
    state = torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    names = [n for n,q in model.named_parameters() if q.requires_grad]
    if names!=state['names'] or len(opt.pairs)!=len(state['masters']):raise ValueError('optimizer parameter order differs')
    with torch.no_grad():
        for (q,m),saved in zip(opt.pairs,state['masters']):m.copy_(saved);q.copy_(m)
    opt.optimizer.load_state_dict(state['optimizer'])
    # load_state_dict replaces the optimizer's group list. The wrapper's alias
    # must follow it, or future scheduler writes silently update stale groups.
    opt.param_groups = opt.optimizer.param_groups
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def master_digest(opt):
    h = hashlib.sha256()
    for _,m in opt.pairs:h.update(m.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for n,q in model.named_parameters():
        if q.requires_grad:
            h.update(str((n,q.dtype,tuple(q.stride()))).encode())
            h.update(q.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizer_digest(opt):
    h = hashlib.sha256()
    for group in opt.optimizer.param_groups:
        h.update(json.dumps({k:v for k,v in group.items() if k!='params'},sort_keys=True).encode())
    for _,m in opt.pairs:
        for k,v in sorted(opt.optimizer.state[m].items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def smoke(out,arm):
    m = verify(out,frozen=False);p = m['plan'];seed = p['seeds'][0];dest = out/f'smoke-{arm}'
    if dest.exists():raise ValueError('preserve smoke evidence')
    dest.mkdir();renderer,cfg = search.renderer_for(search.BASE);rows = records.rows(out/'train.jsonl')
    schedule = list(train.batches(len(rows),16,4,random.Random(seed)))
    train.seed_all(seed);model = load_arm(arm);opt = optimizer(model,arm,p);losses = []
    for step,(ii,_) in enumerate(schedule[:3],1):
        loss,_ = update(model,opt,renderer,encoded_batch(renderer,rows,ii,seed,step),arm,p,step);losses.append(loss)
    save_checkpoint(model,opt,renderer,dest/'step-3',arm,cfg,3)
    before = {'masters':master_digest(opt),'model':model_digest(model),'optimizer':optimizer_digest(opt)}
    expected_types = {n:str(q.dtype) for n,q in model.named_parameters() if q.requires_grad}
    if {id(q) for q,_ in opt.pairs}!={id(q) for q in model.parameters() if q.requires_grad}:
        raise ValueError('saving changed trainable parameter objects')
    rr = encoded_batch(renderer,rows,schedule[3][0],seed,4)
    expected_loss,expected_norm = update(model,opt,renderer,rr,arm,p,4);expected = master_digest(opt)
    expected_masters = [q.detach().cpu() for _,q in opt.pairs]
    _,expected_pred = evaluate(model,renderer,rows[:12],arm)
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model = load_arm(arm,dest/'step-3');opt = optimizer(model,arm,p)
    assert restore_optimizer(model,opt,dest/'step-3')==3
    restored = {'masters':master_digest(opt),'model':model_digest(model),'optimizer':optimizer_digest(opt)}
    actual_types = {n:str(q.dtype) for n,q in model.named_parameters() if q.requires_grad}
    actual_loss,actual_norm = update(model,opt,renderer,rr,arm,p,4);actual = master_digest(opt)
    _,actual_pred = evaluate(model,renderer,rows[:12],arm)
    delta = max(abs(a-b) for x,y in zip(expected_pred,actual_pred) for a,b in zip(x['probabilities'],y['probabilities']))
    differences = []
    names = [n for n,q in model.named_parameters() if q.requires_grad]
    for name,(_,q),a in zip(names,opt.pairs,expected_masters):
        d = (q.detach().cpu()-a).abs()
        if d.max().item():differences.append({'name':name,'max_delta':d.max().item(),'changed':int(d.count_nonzero())})
    train.atomic_json(dest/'resume-comparison.json',{'expected_loss':expected_loss,'actual_loss':actual_loss,
        'expected_norm':expected_norm,'actual_norm':actual_norm,'before':before,'restored':restored,
        'expected_master_sha256':expected,'actual_master_sha256':actual,'max_probability_delta':delta,
        'type_mismatches':[n for n,v in expected_types.items() if v!=actual_types[n]],
        'master_differences':sorted(differences,key=lambda r:-r['max_delta'])})
    if expected!=actual or expected_loss!=actual_loss or delta:raise ValueError('optimizer resume changed next update')
    r = {'arm':arm,'implementation':implementation(),'three_initial_losses':losses,
         'exact_optimizer_resume':True,'next_update_loss':actual_loss,'master_sha256_after_next_update':actual,
         'reload_max_probability_delta':delta,'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad)}
    train.atomic_json(dest/'result.json',r);print('SMOKE',arm,r['exact_optimizer_resume'],flush=True)
    del model,opt;gc.collect();torch.cuda.empty_cache()


def fit(out,arm,seed):
    m = verify(out);p = m['plan'];dest = out/f'{arm}-seed{seed}';dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():print('SKIP_COMPLETE',arm,seed,flush=True);return
    renderer,cfg = search.renderer_for(search.BASE);rows = records.rows(out/'train.jsonl')
    selection = records.rows(out/'selection.jsonl');probe = records.rows(out/'train-probe.jsonl')
    schedule = list(train.batches(len(rows),p['batch_size'],p['steps'],random.Random(seed)))
    prepared = records.rows(out/f'encoded-seed{seed}.jsonl')
    if len(prepared)!=p['steps']:raise ValueError('incomplete prepared inputs')
    checkpoints = sorted((q for q in dest.glob('step-*') if q.name[5:].isdigit() and (q/'complete.json').exists()),
                         key=lambda q:int(q.name[5:]))
    path = checkpoints[-1] if checkpoints else search.BASE
    train.seed_all(seed);model = load_arm(arm,path);opt = optimizer(model,arm,p)
    start = restore_optimizer(model,opt,path) if checkpoints else 0
    log_path = dest/'train.jsonl';logs = records.rows(log_path) if log_path.exists() else []
    if len(logs)<start:raise ValueError('checkpoint/log discrepancy')
    if len(logs)>start:
        shutil.copyfile(log_path,dest/f'train-uncommitted-{time.time_ns()}.jsonl')
        search.save_records(log_path,logs[:start])
    reference_path = out/f'native-seed{seed}/train.jsonl'
    reference = records.rows(reference_path) if arm!='native' else None
    torch.cuda.reset_peak_memory_stats();started = time.monotonic()
    if start==0 and not (dest/'selection-0.json').exists():
        metric,rr = evaluate(model,renderer,selection,arm)
        train.atomic_json(dest/'selection-0.json',metric);search.save_records(dest/'selection-0.jsonl',rr)
        print('SELECTION',arm,seed,0,metric['macro'],flush=True)
    with log_path.open('a') as log:
        for step,(indices,_) in enumerate(schedule,1):
            if step<=start:continue
            item = prepared[step-1]
            if item['step']!=step or item['indices']!=indices:raise ValueError('cached draw schedule differs')
            rr = item['rows']
            digest = search.digest([[r['input_ids'],r['code_ids'],r['target']] for r in rr])
            if reference is not None and digest!=reference[step-1]['input_target_sha256']:raise ValueError('matched inputs differ')
            loss,norm = update(model,opt,renderer,rr,arm,p,step)
            entry = {'step':step,'loss':loss,'grad_norm':norm,'input_target_sha256':digest,
                     'seconds_this_session':time.monotonic()-started,'examples_seen':step*len(indices)}
            log.write(json.dumps(entry)+'\n');log.flush()
            if step%64==0:print('FIT',arm,seed,entry,flush=True)
            if step in p['snapshots']:
                metric,rr = evaluate(model,renderer,selection,arm)
                train.atomic_json(dest/f'selection-{step}.json',metric);search.save_records(dest/f'selection-{step}.jsonl',rr)
                metric_probe,pr = evaluate(model,renderer,probe,arm)
                train.atomic_json(dest/f'train-probe-{step}.json',metric_probe);search.save_records(dest/f'train-probe-{step}.jsonl',pr)
                target = dest/f'step-{step}'
                if target.exists():target.rename(dest/f'step-{step}.incomplete-{time.time_ns()}')
                save_checkpoint(model,opt,renderer,target,arm,cfg,step)
                print('SELECTION',arm,seed,step,metric['macro'],'PROBE',metric_probe['macro'],flush=True)
    result = {'arm':arm,'seed':seed,'steps':p['steps'],'examples_seen':p['steps']*p['batch_size'],
              'snapshots':{str(s):read(dest/f'selection-{s}.json')['macro'] for s in p['snapshots']},
              'peak_cuda_allocated_gb_this_session':torch.cuda.max_memory_allocated()/1e9,
              'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad),
              'seconds_this_session':time.monotonic()-started,'resume_start_step':start,
              'all_inputs_targets_match_native':True,'new_confirmation_cases_scored':0}
    train.atomic_json(dest/'result.json',result);print('RESULT',result,flush=True)
    del model,opt;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def audit(out):
    m = verify(out);p = m['plan'];renderer,_ = search.renderer_for(search.BASE)
    rows = records.rows(out/'selection.jsonl');arms = {};comparisons = {}
    for seed in p['seeds']:
        predictions = {};native_trace = records.rows(out/f'native-seed{seed}/train.jsonl')
        for arm in p['arms']:
            dest = out/f'{arm}-seed{seed}';result = read(dest/'result.json')
            trace = records.rows(dest/'train.jsonl')
            if len(trace)!=p['steps'] or [r['input_target_sha256'] for r in trace]!=[r['input_target_sha256'] for r in native_trace]:
                raise ValueError('incomplete or mismatched training trace')
            marker = verify_checkpoint(dest/f"step-{p['steps']}")
            model = load_arm(arm,dest/f"step-{p['steps']}").requires_grad_(False).eval()
            metric,rr = evaluate(model,renderer,rows,arm);expected = records.rows(dest/f"selection-{p['steps']}.jsonl")
            mismatch = sum(a['prediction']!=b['prediction'] for a,b in zip(rr,expected))
            delta = max(abs(a-b) for x,y in zip(rr,expected) for a,b in zip(x['probabilities'],y['probabilities']))
            if mismatch or delta:raise ValueError('reload changed predictions')
            predictions[arm] = rr;arms[f'{arm}-seed{seed}'] = {'fit':result,'selection':metric,
                'exact_selection_reload':True,'checkpoint':marker}
            del model;gc.collect();torch.cuda.empty_cache();print('AUDIT_RELOAD',arm,seed,flush=True)
        for arm in p['arms'][1:]:
            x = statistics_helper.paired_interval(predictions['native'],predictions[arm],
                                                  confidence=1-.05/(len(p['arms'])-1))
            x['familywise_95ci_within_seed'] = x.pop('paired_document_bootstrap_97_5ci')
            x['exploratory_adjusted_interval_above_zero'] = x.pop('confirmed_improvement')
            x['scope'] = 'repeated selection, conditional on trained models; not independent confirmation or a training-seed interval'
            comparisons[f'{arm}-seed{seed}'] = x
    r = {'manifest':m,'arms':arms,'comparisons':comparisons,'environment':train.versions(),
         'accuracy_early_stopping':False,'new_confirmation_cases_scored':0,
         'candidate_for_separate_fresh_confirmation':[k for k,v in comparisons.items() if v['macro_delta']>=.005]}
    train.atomic_json(out/'report.json',r);train.atomic_json(Path('wiki/project/long-training-results.json'),r)
    print('FINAL_COMPARISONS',comparisons,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','cache','smoke','freeze','run','audit'])
    p.add_argument('--scope',choices=['structures','seeds'],default='structures')
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--arm',choices=['native','bdh','recurrent','workspace'])
    a = p.parse_args()
    if a.stage=='prepare':prepare(a.output,a.scope)
    elif a.stage=='cache':cache(a.output)
    elif a.stage=='smoke':
        for arm in ([a.arm] if a.arm else read(a.output/'plan.json')['arms']):smoke(a.output,arm)
    elif a.stage=='freeze':freeze(a.output)
    elif a.stage=='audit':audit(a.output)
    else:
        plan = verify(a.output)['plan']
        for seed in plan['seeds']:
            for arm in plan['arms']:fit(a.output,arm,seed)
        audit(a.output)


if __name__=='__main__':main()
