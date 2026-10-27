"""Audit the selected workspace checkpoint on exposed selection cases only."""
from collections import defaultdict
import gc
from pathlib import Path
import statistics
import time

import torch

from qwen_decision.v3 import architecture_search as search, hypothesis_workspace as ws, modeling, train
from qwen_decision.v3 import hypothesis_experiment as experiment
from qwen_decision.v3 import readout_experiment as records


def compare(a,b):
    if [r['id'] for r in a] != [r['id'] for r in b]:
        raise ValueError('unaligned predictions')
    return {'prediction_mismatches':sum(x['prediction']!=y['prediction'] for x,y in zip(a,b)),
            'max_probability_delta':max(abs(p-q) for x,y in zip(a,b)
                                        for p,q in zip(x['probabilities'],y['probabilities']))}


@torch.no_grad()
def main():
    out = experiment.OUT
    manifest = experiment.verify(out)
    summary = experiment.read(out/'workspace/result.json')
    step = summary['selected']['step']
    actual_log = records.rows(out/'workspace/train.jsonl')
    cached_log = records.rows(experiment.OLD/'vanilla/train.jsonl')[:len(actual_log)]
    if [r['prompt_sha256'] for r in actual_log] != [r['prompt_sha256'] for r in cached_log]:
        raise ValueError('independent audit found mismatched training draws')
    path = out/f'workspace/step-{step}'
    renderer,_ = search.renderer_for(search.BASE)
    rows = records.rows(out/'selection.jsonl')
    model = ws.load(path).requires_grad_(False).eval()
    metrics,reloaded = experiment.evaluate(model,renderer,rows)
    expected = records.rows(out/f'workspace/selection-{step}.jsonl')
    reload_check = compare(expected,reloaded)
    if reload_check['prediction_mismatches'] or reload_check['max_probability_delta']:
        raise ValueError('checkpoint did not reproduce full selection')
    by = defaultdict(list)
    for r in rows:
        by[r['source']].append(r)
    sample = []
    for source in sorted(by):
        sample += sorted(by[source],key=lambda r:search.digest(['workspace-audit',r['id']]))[:8]
    sample = sorted(sample,key=lambda r:search.digest(['workspace-audit-sample',r['id']]))[:512]
    active_metrics,active = experiment.evaluate(model,renderer,sample)
    blocked_metrics,blocked = experiment.evaluate(model,renderer,sample,write_back=False)
    disabled_metrics,disabled = experiment.evaluate(model,renderer,sample,enabled=False)
    projection = model.model.language_model.hypothesis_workspace.projection.weight
    trained_projection = projection.clone()
    projection.zero_()
    neutral_metrics,neutral = experiment.evaluate(model,renderer,sample)
    projection.copy_(trained_projection)
    perturbation = (projection-torch.eye(len(projection),device=projection.device)).norm().item()/len(projection)**.5
    encoded = [experiment.encoding.encode(renderer,r) for r in sample]
    groups = list(train.microbatches(encoded,experiment.control.BATCH))
    native_max = 0.;permutation_max = 0.;permutation_mismatches = 0
    for group in groups:
        ids,mask,codes,valid,_ = train.collate(group,renderer.tokenizer.pad_token_id)
        positions = torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in group],device=ids.device)
        native = modeling.code_logits(model,ids,mask,codes,valid)
        removed = ws.logits(model,ids,mask,codes,valid,positions,enabled=False)
        native_max = max(native_max,(native-removed).abs().max().item())
        current = ws.logits(model,ids,mask,codes,valid,positions)
        perm = torch.arange(codes.shape[1]-1,-1,-1,device=ids.device)
        shuffled = ws.logits(model,ids,mask,codes[:,perm],valid[:,perm],positions[:,perm])[:,perm]
        permutation_max = max(permutation_max,(current.softmax(-1)-shuffled.softmax(-1)).abs().max().item())
        permutation_mismatches += (current.argmax(-1)!=shuffled.argmax(-1)).sum().item()
    shapes = {};handles = [];counts = [0]*len(model.model.language_model.layers)
    for i,layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_hook(lambda m,a,o,i=i:counts.__setitem__(i,counts[i]+1)))
        mixer = layer.self_attn.q_proj if layer.block_type=='full_attention' else layer.linear_attn
        if layer.block_type=='full_attention':
            handles.append(mixer.register_forward_pre_hook(lambda m,a,i=i:shapes.__setitem__(str(i),int(a[0].shape[1]))))
        else:
            handles.append(mixer.register_forward_pre_hook(lambda m,a,kw,i=i:shapes.__setitem__(str(i),int(kw['hidden_states'].shape[1])),with_kwargs=True))
    group = groups[0]
    experiment.logits_for(model,group,renderer.tokenizer.pad_token_id)
    for handle in handles:
        handle.remove()
    width = max(len(r['input_ids']) for r in group)
    k = max(len(r['target']) for r in group)
    if counts != [1]*24 or any(shapes[str(i)] != width+(k if i in [19,23] else 0) for i in range(24)):
        raise ValueError('unexpected native mixer execution')
    # Padded GPU batches are held on device. Tokenization/model loading excluded.
    cached = []
    for group in groups:
        ids,mask,codes,valid,_ = train.collate(group,renderer.tokenizer.pad_token_id)
        positions = torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in group],device=ids.device)
        cached.append((ids,mask,codes,valid,positions))
    timing = defaultdict(list)
    def run(mode,batches):
        torch.cuda.synchronize();started = time.monotonic()
        for args in batches:
            if mode=='native_same_weights':
                modeling.code_logits(model,*args[:4])
            else:
                ws.logits(model,*args)
        torch.cuda.synchronize()
        return time.monotonic()-started
    for mode in ['native_same_weights','workspace']:
        run(mode,cached[:16])
    for i in range(3):
        for mode in (['native_same_weights','workspace'] if i%2==0 else ['workspace','native_same_weights']):
            timing[mode].append(run(mode,cached))
    med = {k:statistics.median(v) for k,v in timing.items()}
    speed = {'samples':len(sample),'batches':len(groups),'passes':dict(timing),'median_seconds':med,
             'overhead_percent':100*(med['workspace']/med['native_same_weights']-1),
             'scope':'same trained weights, GPU prepared padded inputs including Python/mask/slot operations; excludes tokenization/loading; no service latency claim'}
    baseline = records.rows(experiment.OLD/f'vanilla/selection-{step}.jsonl')
    if [r['id'] for r in baseline] != [r['id'] for r in reloaded]:
        raise ValueError('cached same-step control is unaligned')
    wins = sum(not a['correct'] and b['correct'] for a,b in zip(baseline,reloaded))
    losses = sum(a['correct'] and not b['correct'] for a,b in zip(baseline,reloaded))
    result = {'screen':summary,'manifest':manifest,'checkpoint':str(path),
              'audit_implementation_sha256':train.sha256_file(Path(__file__)),
              'environment':train.versions(),
              'encoding_dependency_sha256':train.sha256_file(Path(experiment.encoding.fmt.__file__)),
              'independently_checked_training_prompt_hashes':len(actual_log),
              'checkpoint_files':{p.name:train.sha256_file(p) for p in path.iterdir() if p.name in
                  ['model.safetensors','v3_config.json','config.json','decision_architecture.json','hypothesis_workspace.safetensors']},
              'reload_all_selection':reload_check,'selection_metrics':metrics,'same_step_wins':wins,'same_step_losses':losses,
              'ablation_samples':len(sample),'ablation_scope':'exposed selection, same trained weights, descriptive causal path checks; not an independent comparison',
              'active':active_metrics,'write_back_blocked':blocked_metrics,'workspace_removed':disabled_metrics,
              'contentless_seeds':neutral_metrics,
              'block_write_effect':compare(active,blocked),'remove_workspace_effect':compare(active,disabled),
              'remove_content_effect':compare(active,neutral),'blocked_vs_native':compare(blocked,disabled),
              'projection_relative_change_from_identity':perturbation,
              'native_manual_forward_max_logit_delta':native_max,
              'slot_order_only_max_probability_delta':permutation_max,'slot_order_only_prediction_mismatches':permutation_mismatches,
              'mixer_execution_probe':{'native_block_calls':counts,'mixer_rows':shapes,'original_width':width,'candidate_width':k},
              'timing':speed,'new_confirmation_cases_scored':0,
              'limits':['one seed, 512 or predeclared early stop; exposed selection is not confirmation',
                        'not a reproduction of the cited latent reasoning methods; no reasoning traces',
                        'additional tokens vs candidate content not separated by a separately trained control',
                        'prototype requires its custom load/logits; not integrated into standard engine']}
    train.atomic_json(out/'audit.json',result)
    train.atomic_json(Path('wiki/project/hypothesis-workspace-results.json'),result)
    print('AUDIT',reload_check,speed,'wins/losses',wins,losses,flush=True)


if __name__=='__main__':
    main()
