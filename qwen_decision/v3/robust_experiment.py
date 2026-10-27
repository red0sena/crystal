"""Reproducible LoRA + permutation consistency + update soup experiment.

Stages: prepare, baseline, fit, screen, confirm. No confirmation scoring before
the candidate recipes and model hashes have been frozen by screen.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from qwen_decision.v3 import architecture_search as search
from qwen_decision.v3 import pointer_experiment as pilot
from qwen_decision.v3 import readout_experiment as previous
from qwen_decision.v3 import robust_adaptation as robust
from qwen_decision.v3 import train

OUT = Path('runs/robust/search-v1')
SEEDS = (41, 73)
BATCH = SimpleNamespace(pack=False, micro_batch=4, token_budget=4096)
PRIOR = ['runs/architecture/search-v1/selection.jsonl',
         'runs/architecture/search-v1/confirm.jsonl',
         'runs/inputs/search-v2/confirmation-ids.jsonl',
         'runs/readout/search-v1/confirmation.jsonl',
         'runs/pointer/g4b-gold3-head100-tune20-s41/code/frozen/dev-step-0.jsonl']


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {p.name: train.sha256_file(p) for p in
            (Path(__file__), Path(robust.__file__), Path(train.__file__), Path(search.modeling.__file__))}


def verify(out, manifest):
    if manifest['implementation'] != source_hashes():
        raise ValueError('implementation changed; preserve this run and use a new directory')
    for name, digest in manifest['files'].items():
        if train.sha256_file(out / name) != digest:
            raise ValueError('prepared data or manifest input changed: ' + name)
    if train.sha256_file(search.BASE / 'model.safetensors') != manifest['base_sha256']:
        raise ValueError('baseline weights changed')


def prepare(args):
    out = args.output
    if out.exists():
        raise SystemExit('preserve runs: prepare requires a new directory')
    raw = pilot.read_rows(Path('data/mix/v3-gold3.dev.jsonl'))
    groups = previous.document_groups(raw)
    lookup = {r['id']: r for r in raw}
    exposed = {r['id'] for path in PRIOR for r in previous.rows(path)}
    exposed_groups = {groups[i] for i in exposed if i in groups}
    # Audit every original training row before scoring any candidate. This also
    # catches overlaps beyond the smaller subset used in this experiment.
    key_ids = defaultdict(set)
    for row in raw:
        for key in previous.group_keys(row):
            key_ids[key].add(row['id'])
    train_overlaps = set()
    count = 0
    for line in Path('data/mix/v3-gold3.jsonl').open():
        row = json.loads(line)
        count += 1
        for key in previous.group_keys(row):
            train_overlaps.update(key_ids.get(key, ()))
        if count % 100000 == 0:
            print('full training audit', count, flush=True)
    overlap_groups = {groups[i] for i in train_overlaps}
    renderer, config = search.renderer_for(search.BASE)
    raw_confirmation = []
    skipped = []
    for row in raw:
        if groups[row['id']] in exposed_groups | overlap_groups:
            continue
        variants = robust.cyclic_views(renderer, row)
        if max(len(v['input_ids']) for v in variants) <= 4096:
            raw_confirmation.append({**row, 'group': groups[row['id']]})
        else:
            skipped.append(row['id'])
    # Retain the old selection set; all newly unobserved documents remain a
    # single confirmation set, never a succession of small test-driven retries.
    selection_ids = [r['id'] for r in previous.rows(PRIOR[0])]
    selection = [{**lookup[i], 'group': groups[i]} for i in selection_ids]
    old_encoded = previous.rows(PRIOR[0])
    for row, old in zip(selection, old_encoded):
        if robust.view(renderer, row)['input_ids'] != old['input_ids']:
            raise ValueError('canonical baseline prompt differs')
    reserved = set(key_ids)
    eligible = []
    for row in pilot.read_rows(Path('data/mix/v3-gold3.jsonl'), 32768):
        if any(key in reserved for key in previous.group_keys(row)):
            continue
        if len(robust.view(renderer, row)['input_ids']) <= 1500:
            eligible.append(row)
    training = sorted(eligible, key=lambda r: search.digest(['robust-training-v1', r['id']]))[:16384]
    if not training or not raw_confirmation:
        raise ValueError('empty training or confirmation')
    out.mkdir(parents=True)
    for name, rows in [('train', training), ('selection', selection), ('confirmation', raw_confirmation)]:
        search.save_records(out / (name + '.jsonl'), rows)
    plan = {'seeds': SEEDS, 'objectives': {'ce': 0., 'consistent': .5},
            'rank': 8, 'lora_alpha': 16, 'learning_rate': .0001,
            'steps': 256, 'effective_semantic_batch': 16, 'views_per_example': 2,
            'warmup_steps': 16, 'optimizer': 'AdamW FP32 adapters, betas(.9,.95), weight_decay .01, clip 1',
            'snapshots': [128, 256], 'mixing_coefficients': [.5, 1.],
            'seed_combination': 'equal mean of the two BA updates, same step and coefficient; never average factors',
            'inference': 'actual merged BF16 projections, frozen original embeddings and code output rows',
            'screen': 'four step/coefficient combinations per objective on the old selection only; tie favors smaller coefficient then fewer steps',
            'multiview': 'equal probability mean over distinct cyclic shifts 0,k//4,k//2,3*k//4; ordered/positional/noul stay in original order',
            'candidate_freeze': 'single best nonbaseline soup; separately best cyclic method among base and two selected soups',
            'confirmation_decision': 'paired source-stratified document bootstrap, 5000 draws; two frozen comparisons use Bonferroni 97.5% intervals; positive lower bound required for confirmed improvement',
            'no_new_teacher_labels_or_rationales': True,
            'no_confirmation_feedback_for_retraining_or_candidate_selection': True}
    manifest = {'base': str(search.BASE), 'config': config, 'counts': {'train': len(training), 'selection': len(selection), 'confirmation': len(raw_confirmation)},
                'sources': {name: len({r['source'] for r in rr}) for name, rr in [('train',training),('selection',selection),('confirmation',raw_confirmation)]},
                'confirmation_source_counts': dict(Counter(r['source'] for r in raw_confirmation)),
                'confirmation_groups': len({r['group'] for r in raw_confirmation}),
                'full_train_audit_rows': count, 'full_train_overlapping_dev_ids': len(train_overlaps),
                'confirmation_original_train_shared_keys': 0, 'confirmation_prior_exposure_shared_groups': 0,
                'audit_scope': 'upstream id, full state, whitespace-normalized state, explicit item/document; not semantic near-duplicate audit',
                'skipped_confirmation_over_4096': skipped, 'prior_exposed_ids': len(exposed),
                'selection_max_tokens': 1536, 'confirmation_max_tokens': 4096, 'train_original_max_tokens': 1500,
                'holdback': 'unobserved by earlier searches in this chat; g4b historically selected with the full gold3 dev',
                'base_sha256': train.sha256_file(search.BASE/'model.safetensors'),
                'mix_sha256': {str(p): train.sha256_file(p) for p in [Path('data/mix/v3-gold3.jsonl'),Path('data/mix/v3-gold3.dev.jsonl')]},
                'implementation': source_hashes(), 'plan': plan,
                'files': {name+'.jsonl': train.sha256_file(out/(name+'.jsonl')) for name in ('train','selection','confirmation')}}
    train.atomic_json(out/'plan.json', plan)
    train.atomic_json(out/'manifest.json', manifest)
    print('PREPARED', json.dumps(manifest['counts']), manifest['sources'], flush=True)


def load_model():
    config = search.modeling.read_config(search.BASE)
    model = search.modeling.load(search.BASE, attention=config['attention'], deltanet=config['deltanet'], kernel='sdpa')
    model.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate(model, renderer, rows, cyclic=False):
    model.eval()
    encoded = []
    for row in rows:
        encoded += robust.cyclic_views(renderer, row) if cyclic else [robust.view(renderer, row)]
    probabilities = defaultdict(list)
    started = time.monotonic()
    torch.cuda.synchronize()
    calls = 0
    for group in train.microbatches(encoded, BATCH):
        z, _ = train.logits_for(model, group, renderer.tokenizer.pad_token_id)
        probs = robust.align(z, group).softmax(-1).cpu()
        for row, p in zip(group, probs):
            probabilities[row['id']].append(p[:len(row['target'])].tolist())
        calls += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    records = []
    for row in rows:
        p = torch.tensor(probabilities[row['id']]).mean(0).tolist()
        pred = max(range(len(p)), key=p.__getitem__)
        records.append({k: row[k] for k in ('id','group','source')} |
                       {'prediction': pred, 'correct': row['target'][pred] >= max(row['target'])-1e-6,
                        'probabilities': p, 'views': len(probabilities[row['id']])})
    return search.metrics(records) | {'seconds': elapsed, 'prompt_evaluations': len(encoded),
                                     'batches': calls, 'tokens': sum(len(r['input_ids']) for r in encoded)}, records


def baseline(args, manifest):
    out = args.output
    if (out/'baseline.json').exists():
        raise SystemExit('baseline already exists')
    model = load_model()
    renderer, _ = search.renderer_for(search.BASE)
    rows = previous.rows(out/'selection.jsonl')
    result = {}
    for name, cyclic in [('single',False),('cyclic',True)]:
        metrics, records = evaluate(model, renderer, rows, cyclic)
        result[name] = metrics
        search.save_records(out/f'selection-base-{name}.jsonl', records)
        print('BASE', name, metrics['macro'], metrics['seconds'], flush=True)
    train.atomic_json(out/'baseline.json', result)


def fit(args, manifest):
    out = args.output
    if (out/'frozen.json').exists():
        raise SystemExit('candidates already frozen')
    plan = manifest['plan']
    rows = previous.rows(out/'train.jsonl')
    renderer, _ = search.renderer_for(search.BASE)
    for objective, js_weight in plan['objectives'].items():
        for seed in SEEDS:
            dest = out/'fits'/f'{objective}-s{seed}'
            if (dest/'result.json').exists():
                continue
            if dest.exists():
                raise SystemExit('partial run exists: preserve it and use a new output directory')
            dest.mkdir(parents=True)
            train.seed_all(seed)
            model = load_model()
            text = model.model.language_model
            projections = robust.install(text, plan['rank'])
            params = [p for p in text.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=plan['learning_rate'], betas=(.9,.95), weight_decay=.01)
            rng = random.Random(seed+1000)
            elapsed_start = time.monotonic()
            n_seen = 0
            first_loss = None
            with (dest/'train.jsonl').open('w') as log:
                for step, (indices, _) in enumerate(train.batches(len(rows), 16, plan['steps'], random.Random(seed)), 1):
                    model.train()
                    optimizer.zero_grad()
                    pairs = [(robust.random_view(renderer, rows[i], rng), robust.random_view(renderer, rows[i], rng)) for i in indices]
                    pairs.sort(key=lambda pair: max(len(v['input_ids']) for v in pair))
                    loss_sum = ce_sum = js_sum = 0.
                    at = 0
                    while at < len(pairs):
                        group = [pairs[at]]
                        at += 1
                        while at < len(pairs) and len(group) < 2 and 2*(len(group)+1)*max(len(v['input_ids']) for pair in group+[pairs[at]] for v in pair) <= 4096:
                            group.append(pairs[at]); at += 1
                        flat = [pair[0] for pair in group] + [pair[1] for pair in group]
                        if max(len(v['input_ids']) for v in flat) > 1536:
                            raise ValueError('augmented training prompt exceeds budget; never truncate')
                        z, _ = train.logits_for(model, flat, renderer.tokenizer.pad_token_id)
                        z = robust.align(z, flat)
                        a, b = z.chunk(2)
                        kmax = z.shape[1]
                        target = torch.tensor(
                            [[pair[0]['target'][pair[0]['order'].index(i)] for i in range(len(pair[0]['order']))]
                             + [0.]*(kmax-len(pair[0]['order'])) for pair in group], device=z.device)
                        ce = .5*(train.soft_ce(a, target)+train.soft_ce(b, target))
                        js = robust.jensen_shannon(a,b)
                        loss = (ce+js_weight*js).sum()/len(indices)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite loss')
                        loss.backward()
                        loss_sum += loss.item()
                        ce_sum += ce.sum().item()/len(indices)
                        js_sum += js.sum().item()/len(indices)
                    norm = torch.nn.utils.clip_grad_norm_(params,1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('nonfinite gradient')
                    scale = min(step/16, 1.) * (.1+.9*.5*(1+math.cos(math.pi*max(step-16,0)/(plan['steps']-16))))
                    for pg in optimizer.param_groups:
                        pg['lr'] = plan['learning_rate']*scale
                    optimizer.step()
                    n_seen += len(indices)
                    first_loss = loss_sum if first_loss is None else first_loss
                    entry = {'step':step, 'loss':loss_sum, 'ce':ce_sum, 'js':js_sum,
                             'grad_norm':float(norm), 'rows_seen':n_seen, 'lr':optimizer.param_groups[0]['lr']}
                    log.write(json.dumps(entry)+'\n'); log.flush()
                    if step%16==0:
                        print(dest.name, step, 'loss',round(loss_sum,4),'JS',round(js_sum,5), 'seconds',round(time.monotonic()-elapsed_start), flush=True)
                    if step in plan['snapshots']:
                        save_file(robust.adapter_state(text), str(dest/f'adapter-step-{step}.safetensors'))
            train.atomic_json(dest/'result.json', {'objective':objective, 'js_weight':js_weight, 'seed':seed,
                'steps':plan['steps'], 'rows_seen':n_seen, 'prompt_views_seen':2*n_seen,
                'trainable_parameters':sum(p.numel() for p in params), 'projection_count':len(projections),
                'first_loss':first_loss, 'last_loss':loss_sum, 'seconds':time.monotonic()-elapsed_start,
                'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
                'files':{f.name:train.sha256_file(f) for f in dest.glob('*.safetensors')}})
            del params, optimizer, text, model
            gc.collect(); torch.cuda.empty_cache()


def states_for(out, spec):
    return [load_file(str(out/'fits'/f"{spec['objective']}-s{seed}"/f"adapter-step-{spec['step']}.safetensors")) for seed in SEEDS]


def screen(args, manifest):
    out = args.output
    if (out/'frozen.json').exists():
        raise SystemExit('screen already frozen')
    base = read(out/'baseline.json')
    rows = previous.rows(out/'selection.jsonl')
    renderer, _ = search.renderer_for(search.BASE)
    model = load_model()
    text = model.model.language_model
    robust.install(text, manifest['plan']['rank'])
    candidates = []
    for objective in manifest['plan']['objectives']:
        for step in manifest['plan']['snapshots']:
            for coefficient in manifest['plan']['mixing_coefficients']:
                spec = {'objective':objective,'step':step,'coefficient':coefficient}
                name = f'{objective}-{step}-a{coefficient:g}'
                path = out/f'selection-{name}.json'
                if path.exists():
                    item = read(path)
                else:
                    states = states_for(out,spec)
                    with robust.merged(text, coefficient, states):
                        score, records = evaluate(model,renderer,rows)
                    item = {'spec':spec,'selection':score,'name':name}
                    train.atomic_json(path,item)
                    search.save_records(out/f'selection-{name}.jsonl',records)
                candidates.append(item)
                print('SCREEN',name,item['selection']['macro'],flush=True)
    ranked = sorted(candidates, key=lambda c:(-c['selection']['macro'],c['spec']['coefficient'],c['spec']['step'],c['name']))
    single = ranked[0]
    # Fix two family winners before evaluating any cyclic variants. No per-source
    # gating, seed picking, or test-set search is permitted.
    best_by_family = [next(c for c in ranked if c['spec']['objective']==objective) for objective in manifest['plan']['objectives']]
    cyclic = [{'name':'base-cyclic','spec':None,'selection':base['cyclic']}]
    for candidate in best_by_family:
        states = states_for(out,candidate['spec'])
        with robust.merged(text,candidate['spec']['coefficient'],states):
            score, records = evaluate(model,renderer,rows,True)
        item = {**candidate,'name':candidate['name']+'-cyclic','selection':score}
        cyclic.append(item)
        search.save_records(out/f"selection-{item['name']}.jsonl",records)
        print('CYCLIC',item['name'],score['macro'],flush=True)
    multi = sorted(cyclic,key=lambda c:(-c['selection']['macro'],c['name']))[0]
    files = {str(p.relative_to(out)):train.sha256_file(p) for p in (out/'fits').glob('*/*.safetensors')}
    frozen = {'single':single,'multi':multi,'ranked_single':ranked,'cyclic_comparisons':cyclic,
              'adapter_sha256':files,'implementation':source_hashes(),
              'selection_sha256':manifest['files']['selection.jsonl'],
              'confirmation_sha256':manifest['files']['confirmation.jsonl'],
              'before_confirmation':True}
    train.atomic_json(out/'frozen.json',frozen)
    print('FROZEN',json.dumps({'single':single['name'],'multi':multi['name']}),flush=True)


def paired_interval(base, candidate, draws=5000, confidence=.975):
    # Resample documents within each source. Return both ordinary 95% and
    # Bonferroni-adjusted 97.5% intervals for the two pre-frozen comparisons.
    import numpy as np
    a, b = {r['id']:r for r in base}, {r['id']:r for r in candidate}
    if a.keys()!=b.keys():
        raise ValueError('paired ids differ')
    groups = defaultdict(lambda:defaultdict(list))
    for key,row in a.items():
        groups[row['source']][row['group']].append(int(b[key]['correct'])-int(row['correct']))
    rng = np.random.default_rng(20261019)
    by_source = [[np.asarray(v) for v in g.values()] for g in groups.values()]
    samples = []
    for _ in range(draws):
        samples.append(np.mean([np.concatenate([g[i] for i in rng.integers(len(g),size=len(g))]).mean() for g in by_source]))
    q = (1-confidence)/2
    return {'macro_delta':search.metrics(candidate)['macro']-search.metrics(base)['macro'],
            'paired_document_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist(),
            'paired_document_bootstrap_97_5ci':np.quantile(samples,[q,1-q]).tolist(),
            'wins':sum(b[k]['correct'] and not a[k]['correct'] for k in a),
            'losses':sum(a[k]['correct'] and not b[k]['correct'] for k in a),
            'confirmed_improvement':bool(np.quantile(samples,q)>0)}


def confirm(args, manifest):
    out = args.output
    if (out/'confirmation-results.json').exists() or (out/'confirmation-started.json').exists():
        raise SystemExit('confirmation already accessed; preserve it, no reselection')
    frozen = read(out/'frozen.json')
    for name,digest in frozen['adapter_sha256'].items():
        if train.sha256_file(out/name)!=digest:
            raise ValueError('frozen adapter changed')
    train.atomic_json(out/'confirmation-started.json',{'frozen_sha256':train.sha256_file(out/'frozen.json')})
    rows = previous.rows(out/'confirmation.jsonl')
    selection = previous.rows(out/'selection.jsonl')
    renderer, _ = search.renderer_for(search.BASE)
    model = load_model()
    base_metrics, base_records = evaluate(model,renderer,rows)
    search.save_records(out/'confirmation-base.jsonl',base_records)
    print('CONFIRM baseline',base_metrics['macro'],flush=True)
    robust.install(model.model.language_model,manifest['plan']['rank'])
    result = {'base':base_metrics,'arms':{},'reload_mismatches':{},'scope':manifest['holdback']}
    for kind in ('single','multi'):
        candidate = frozen[kind]
        spec = candidate['spec']
        states = states_for(out,spec) if spec else None
        strength = spec['coefficient'] if spec else 0.
        with robust.merged(model.model.language_model,strength,states):
            _, reloaded = evaluate(model,renderer,selection,kind=='multi')
            expected_file = out/f"selection-{candidate['name']}.jsonl"
            old = {r['id']:r['prediction'] for r in previous.rows(expected_file)}
            mismatch = sum(old[r['id']]!=r['prediction'] for r in reloaded)
            if mismatch:
                raise ValueError('reloaded merged selection predictions differ')
            score, records = evaluate(model,renderer,rows,kind=='multi')
        search.save_records(out/f'confirmation-{kind}.jsonl',records)
        result['reload_mismatches'][kind] = mismatch
        result['arms'][kind] = {'candidate':candidate,'metrics':score,'vs_base':paired_interval(base_records,records)}
        train.atomic_json(out/'confirmation-progress.json',result)
        print('CONFIRM',kind,score['macro'],result['arms'][kind]['vs_base'],flush=True)
    train.atomic_json(out/'confirmation-results.json',result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','baseline','fit','screen','confirm'])
    p.add_argument('--output',type=Path,default=OUT)
    args = p.parse_args()
    if args.stage=='prepare':
        prepare(args)
    else:
        manifest = read(args.output/'manifest.json')
        verify(args.output,manifest)
        globals()[args.stage](args,manifest)


if __name__=='__main__':
    main()
