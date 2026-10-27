"""Gold-only, training-free input experiments; keep the serving format unchanged.

Screen on the existing architecture selection set. Freeze one candidate before
checking different gold3-dev documents; never reuse the exposed confirmation or
JEVal restricted examples for model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from qwen_decision.v3 import architecture_search as search
from qwen_decision.v3 import format as fmt, modeling, pointer_experiment as pilot, train

MODES = ('base', 'definitions', 'echo', 'repeat', 'prefill')
ALPHAS = (0.25, 0.5, 1.0)
CONTENT_FREE = ('N/A', '[MASK]', '')


def prompt(tokenizer, example, codes, mode, state_override=None):
    if mode not in MODES:
        raise ValueError(mode)
    state = example['state'] if state_override is None else state_override
    question = example['question']
    if mode == 'definitions' and question['type'] == 'choice':
        # Work with the rendered texts, not a new dict keyed by descriptions:
        # distinct answer keys can legitimately share the same description.
        texts = [key if value is None or value == '' else fmt.describe(value)
                 for key, value in question['criteria'].items()]
        lines = ['State:', fmt.describe(state) if state not in (None, '') else '(empty)', '',
                 'Question:', fmt.describe(question.get('instructions') or fmt.DEFAULT_INSTRUCTIONS),
                 '', 'Options:']
        lines += [f'{code}: {text}' for code, text in zip(codes, texts)]
        user = '\n'.join(lines + ['', fmt.CLOSING_LINE])
    else:
        user = fmt.user_prompt(state, question, codes, 'echo' if mode == 'echo' else 'd3')
    if mode == 'repeat':
        user = user + '\n\n' + user
    text = tokenizer.apply_chat_template([
        {'role': 'system', 'content': fmt.SYSTEM_PROMPT}, {'role': 'user', 'content': user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # A newline avoids BPE merging ':' with the following answer letter.
    return text + ('The correct option is:\n' if mode == 'prefill' else '')


def encode(tokenizer, examples, groups, mode, state_override=None):
    codes, code_ids = fmt.answer_codes(tokenizer)
    result = []
    for e in examples:
        k = len(e['target'])
        text = prompt(tokenizer, e, codes[:k], mode, state_override)
        # Prefilling must preserve the contract that each answer code is one token.
        ids = tokenizer.encode(text, add_special_tokens=False)
        if mode == 'prefill':
            for code, tid in zip(codes[:k], code_ids[:k]):
                if tokenizer.encode(text + code, add_special_tokens=False) != ids + [tid]:
                    raise ValueError('prefill changes answer-code token boundary')
        result.append({'id': e['id'], 'source': e['source'], 'group': groups[e['id']],
                       'input_ids': ids, 'code_ids': code_ids[:k], 'target': e['target']})
    return result


@torch.no_grad()
def infer(model, rows, pad_id, max_tokens=4096):
    if any(len(r['input_ids']) > max_tokens for r in rows):
        raise ValueError('input exceeds fixed window; no silent truncation or row dropping')
    # Identical question/options with content-free state share a computation.
    unique, slots = {}, defaultdict(list)
    for i, row in enumerate(rows):
        key = tuple(row['input_ids'])
        unique.setdefault(key, row)
        slots[key].append(i)
    outputs = [None] * len(rows)
    args = SimpleNamespace(pack=False, micro_batch=4, token_budget=4096)
    started = time.monotonic()
    batches = 0
    torch.cuda.synchronize()
    for batch in train.microbatches(list(unique.values()), args):
        logits, _ = train.logits_for(model, batch, pad_id)
        values = logits.cpu().tolist()
        for row, value in zip(batch, values):
            for i in slots[tuple(row['input_ids'])]:
                outputs[i] = value[:len(row['target'])]
        batches += 1
        if batches % 100 == 0:
            print(f'inference {batches} batches, {time.monotonic()-started:.0f}s', flush=True)
    torch.cuda.synchronize()
    return outputs, {'seconds': time.monotonic()-started, 'unique_prompts': len(unique), 'batches': batches,
                     'tokens': sum(len(r['input_ids']) for r in unique.values()),
                     'max_tokens': max(len(r['input_ids']) for r in rows)}


def records(rows, logits):
    result = []
    for row, z in zip(rows, logits):
        p = int(np.argmax(z))
        correct = row['target'][p] >= max(row['target']) - 1e-6
        result.append({k: row[k] for k in ('id', 'source', 'group')} |
                      {'prediction': p, 'correct': correct, 'logits': z})
    return result


def calibrated(logits, priors, alpha):
    # p(y|x) / mean_c p(y|content_free_c), or tempered prior for alpha<1.
    result = []
    for i, z in enumerate(logits):
        probabilities = [torch.tensor(p[i], dtype=torch.float64).softmax(0) for p in priors]
        log_prior = torch.stack(probabilities).mean(0).clamp_min(1e-100).log()
        result.append((torch.tensor(z) - alpha * log_prior).tolist())
    return result


def screen(args):
    if args.output.exists():
        raise SystemExit('Preserve completed experiments: use a new output directory')
    old = args.previous
    old_manifest = json.loads((old / 'manifest.json').read_text())
    base = Path(old_manifest['args']['base'])
    tokenizer = AutoTokenizer.from_pretrained(base)
    raw = pilot.read_rows(Path('data/mix/v3-gold3.dev.jsonl'))
    groups = {r['id']: g for r, (_, g) in zip(raw, search.partition_dev(raw))}
    lookup = {r['id']: r for r in raw}
    previous = search.read_rows(old, 'selection')
    examples = [lookup[r['id']] for r in previous]
    baseline_rows = encode(tokenizer, examples, groups, 'base')
    if any(a['input_ids'] != b['input_ids'] or a['target'] != b['target']
           for a, b in zip(baseline_rows, previous)):
        raise ValueError('selection re-render differs from prior fixed inputs')
    rendered = {mode: (baseline_rows if mode == 'base' else encode(tokenizer, examples, groups, mode))
                for mode in MODES}
    if any(len(row['input_ids']) > 4096 for rows in rendered.values() for row in rows):
        raise ValueError('a candidate exceeds the fixed input window')
    args.output.mkdir(parents=True)
    plan = {'base': str(base), 'previous': str(old), 'modes': MODES, 'calibration_alphas': ALPHAS,
            'content_free': CONTENT_FREE, 'max_tokens': 4096, 'selection_n': len(examples),
            'confirmation_gate_macro_delta': .005, 'selection_sha256': search.digest(previous),
            'implementation_sha256': train.sha256_file(Path(__file__)),
            'dev_sha256': train.sha256_file(Path('data/mix/v3-gold3.dev.jsonl')),
            'holdback': 'exclude all prior architecture selection/confirmation and Pointer dev document groups',
            'note': 'base historically used full gold3 dev for early stopping; no pristine benchmark claim'}
    train.atomic_json(args.output / 'plan.json', plan)
    model = search.load_model(old_manifest)
    # Warm up outside recorded inference durations.
    infer(model, baseline_rows[:4], tokenizer.pad_token_id)
    summaries, all_logits = {}, {}
    for mode in MODES:
        rows = rendered[mode]
        z, runtime = infer(model, rows, tokenizer.pad_token_id)
        all_logits[mode] = z
        rec = records(rows, z)
        search.save_records(args.output / f'selection-{mode}.jsonl', rec)
        summaries[mode] = {'metrics': search.metrics(rec), 'runtime': runtime}
        print(mode, json.dumps(summaries[mode]), flush=True)
    priors = []
    for j, state in enumerate(CONTENT_FREE):
        rows = encode(tokenizer, examples, groups, 'base', state_override=state)
        z, runtime = infer(model, rows, tokenizer.pad_token_id)
        priors.append(z)
        search.save_records(args.output / f'prior-{j}.jsonl', records(rows, z))
        summaries[f'prior-{j}'] = {'runtime': runtime}
    for alpha in ALPHAS:
        name = f'calibration-{alpha}'
        rec = records(baseline_rows, calibrated(all_logits['base'], priors, alpha))
        search.save_records(args.output / f'selection-{name}.jsonl', rec)
        summaries[name] = {'metrics': search.metrics(rec)}
    candidates = [k for k, v in summaries.items() if 'metrics' in v and k != 'base']
    winner = max(candidates, key=lambda k: summaries[k]['metrics']['macro'])
    delta = summaries[winner]['metrics']['macro'] - summaries['base']['metrics']['macro']
    frozen = {'candidate': winner, 'selection_macro_delta': delta, 'run_confirmation': delta >= .005,
              'criterion': 'at least +0.5 percentage points macro on selection, fixed before screening'}
    train.atomic_json(args.output / 'selection-summary.json', summaries)
    train.atomic_json(args.output / 'frozen-candidate.json', frozen)
    print('FROZEN', json.dumps(frozen), flush=True)


def confirm(args):
    out = args.output
    if (out / 'confirmation.json').exists():
        raise SystemExit('confirmation already evaluated')
    frozen = json.loads((out / 'frozen-candidate.json').read_text())
    if not frozen['run_confirmation']:
        raise SystemExit('No candidate passed the predeclared screening gate')
    plan = json.loads((out / 'plan.json').read_text())
    if plan['implementation_sha256'] != train.sha256_file(Path(__file__)):
        raise ValueError('implementation changed after candidate freeze')
    old = Path(plan['previous'])
    manifest = json.loads((old / 'manifest.json').read_text())
    raw = pilot.read_rows(Path('data/mix/v3-gold3.dev.jsonl'))
    groups = {r['id']: g for r, (_, g) in zip(raw, search.partition_dev(raw))}
    exposed = {r['group'] for split in ('selection', 'confirm') for r in search.read_rows(old, split)}
    pointer_file = Path('runs/pointer/g4b-gold3-head100-tune20-s41/code/frozen/dev-step-0.jsonl')
    for r in (json.loads(line) for line in pointer_file.open()):
        if r['id'] in groups:
            exposed.add(groups[r['id']])
    tokenizer = AutoTokenizer.from_pretrained(plan['base'])
    by_source = defaultdict(list)
    for r in raw:
        if groups[r['id']] not in exposed:
            by_source[r['source']].append(r)
    examples = []
    for source, rows in sorted(by_source.items()):
        chosen = 0
        for row in sorted(rows, key=lambda r: search.digest(['input-confirm-v1', r['id']])):
            if len(encode(tokenizer, [row], groups, 'base')[0]['input_ids']) <= 1536:
                examples.append(row)
                chosen += 1
            if chosen == 64:
                break
    baseline_rows = encode(tokenizer, examples, groups, 'base')
    search.save_records(out / 'confirmation-ids.jsonl',
                        [{k: r[k] for k in ('id', 'group', 'source')} for r in baseline_rows])
    model = search.load_model(manifest)
    a, _ = infer(model, baseline_rows, tokenizer.pad_token_id)
    name = frozen['candidate']
    if name.startswith('calibration-'):
        priors = [infer(model, encode(tokenizer, examples, groups, 'base', s), tokenizer.pad_token_id)[0]
                  for s in CONTENT_FREE]
        b = calibrated(a, priors, float(name.split('-')[1]))
    else:
        b, _ = infer(model, encode(tokenizer, examples, groups, name), tokenizer.pad_token_id)
    ra, rb = records(baseline_rows, a), records(baseline_rows, b)
    search.save_records(out / 'confirmation-base.jsonl', ra)
    search.save_records(out / 'confirmation-candidate.jsonl', rb)
    delta = search.paired_delta(ra, rb)
    result = {'frozen': frozen, 'n': len(examples), 'base': search.metrics(ra),
              'candidate': search.metrics(rb), 'delta': delta,
              'evidence_of_improvement': delta['paired_document_bootstrap_95ci'][0] > 0}
    train.atomic_json(out / 'confirmation.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('screen', 'confirm'))
    p.add_argument('--previous', type=Path, default=Path('runs/architecture/search-v1'))
    p.add_argument('--output', type=Path, default=Path('runs/inputs/search-v1'))
    args = p.parse_args()
    (screen if args.stage == 'screen' else confirm)(args)


if __name__ == '__main__':
    main()
