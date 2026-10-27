"""Reproduce and intervene on the mandatory candidate exit; old selection only."""
from collections import defaultdict
from pathlib import Path
import statistics
import time

import torch

from qwen_decision.v3 import architecture_search as search, bottleneck_experiment as ex
from qwen_decision.v3 import hypothesis_workspace as core, modeling, train, workspace_bottleneck as bt
from qwen_decision.v3 import readout_experiment as records
from qwen_decision.v3.hypothesis_audit import compare


def score(model,args,blocked=False):
    ids,mask,codes,valid,positions = args
    if not blocked:
        return bt.logits(model,*args)
    _,slots = core.hidden(model,ids,mask,positions,valid,write_back=False,return_workspace=True)
    h = model.model.language_model.norm(slots).float()
    return (h*model.get_output_embeddings().weight[codes].float()).sum(-1).masked_fill(~valid,modeling.MASK)


def tensors(rows,pad):
    ids,mask,codes,valid,_ = train.collate(rows,pad)
    positions = torch.tensor([r['option_positions']+[0]*(codes.shape[1]-len(r['target'])) for r in rows],device=ids.device)
    return ids,mask,codes,valid,positions


@torch.no_grad()
def probe(model,renderer,rows,blocked=False):
    probabilities = {}
    encoded = [ex.encoding.encode(renderer,r) for r in rows]
    for group in train.microbatches(encoded,ex.control.BATCH):
        z = score(model,tensors(group,renderer.tokenizer.pad_token_id),blocked)
        for r,p in zip(group,z.softmax(-1).cpu()):
            probabilities[r['id']] = p[:len(r['target'])].tolist()
    result = []
    for r in rows:
        p = probabilities[r['id']];pred = max(range(len(p)),key=p.__getitem__)
        result.append({'id':r['id'],'source':r['source'],'group':r.get('group',r['id']),
                       'prediction':pred,'correct':r['target'][pred]>=max(r['target'])-1e-6,
                       'probabilities':p,'views':1})
    return search.metrics(result),result


@torch.no_grad()
def main():
    out = ex.OUT
    manifest = ex.verify(out)
    screen = ex.read(out/'workspace/result.json')
    step = screen['selected']['step'];path = out/f'workspace/step-{step}'
    actual_log = records.rows(out/'workspace/train.jsonl')
    cached_log = records.rows(ex.OLD/'vanilla/train.jsonl')[:len(actual_log)]
    if [r['prompt_sha256'] for r in actual_log] != [r['prompt_sha256'] for r in cached_log]:
        raise ValueError('training prompts differ')
    rows = records.rows(out/'selection.jsonl');renderer,_ = search.renderer_for(search.BASE)
    model = bt.load(path).requires_grad_(False).eval()
    metrics,actual = ex.evaluate(model,renderer,rows)
    expected = records.rows(out/f'workspace/selection-{step}.jsonl')
    reload_check = compare(expected,actual)
    if reload_check['prediction_mismatches'] or reload_check['max_probability_delta']:
        raise ValueError('selected model did not reproduce')
    baseline = records.rows(ex.OLD/f'vanilla/selection-{step}.jsonl')
    if [r['id'] for r in baseline] != [r['id'] for r in actual]:
        raise ValueError('unaligned matched-step native')
    wins = sum(not a['correct'] and b['correct'] for a,b in zip(baseline,actual))
    losses = sum(a['correct'] and not b['correct'] for a,b in zip(baseline,actual))
    by = defaultdict(list)
    for r in rows:
        by[r['source']].append(r)
    sample = []
    for source in sorted(by):
        sample += sorted(by[source],key=lambda r:search.digest(['workspace-audit',r['id']]))[:8]
    sample = sorted(sample,key=lambda r:search.digest(['workspace-audit-sample',r['id']]))[:512]
    active_metrics,active = probe(model,renderer,sample)
    blocked_metrics,blocked = probe(model,renderer,sample,blocked=True)
    projection = model.model.language_model.hypothesis_workspace.projection.weight
    saved = projection.clone();projection.zero_()
    neutral_metrics,neutral = probe(model,renderer,sample)
    projection.copy_(saved)
    norm_change = (projection-torch.eye(len(projection),device=projection.device)).norm().item()/len(projection)**.5
    encoded = [ex.encoding.encode(renderer,r) for r in sample]
    cached = [tensors(group,renderer.tokenizer.pad_token_id) for group in train.microbatches(encoded,ex.control.BATCH)]
    order_mismatch = 0;order_delta = 0.
    for args in cached:
        ids,mask,codes,valid,positions = args
        perm = torch.arange(codes.shape[1]-1,-1,-1,device=codes.device)
        a = bt.logits(model,*args).softmax(-1)
        b = bt.logits(model,ids,mask,codes[:,perm],valid[:,perm],positions[:,perm])[:,perm].softmax(-1)
        order_mismatch += (a.argmax(-1)!=b.argmax(-1)).sum().item()
        order_delta = max(order_delta,(a-b).abs().max().item())
    timing = defaultdict(list)
    def run(mode,batches):
        torch.cuda.synchronize();started = time.monotonic()
        for args in batches:
            if mode=='native_same_weights':
                modeling.code_logits(model,*args[:4])
            else:
                bt.logits(model,*args)
        torch.cuda.synchronize()
        return time.monotonic()-started
    for mode in ['native_same_weights','candidate_bottleneck']:
        run(mode,cached[:16])
    for i in range(3):
        for mode in (['native_same_weights','candidate_bottleneck'] if i%2==0 else ['candidate_bottleneck','native_same_weights']):
            timing[mode].append(run(mode,cached))
    med = {k:statistics.median(v) for k,v in timing.items()}
    result = {'screen':screen,'manifest':manifest,'checkpoint':str(path),
              'audit_implementation_sha256':train.sha256_file(Path(__file__)),
              'environment':train.versions(),'independently_checked_training_prompt_hashes':len(actual_log),
              'checkpoint_files':{p.name:train.sha256_file(p) for p in path.iterdir() if p.name in
                  ['model.safetensors','config.json','v3_config.json','decision_architecture.json','hypothesis_workspace.safetensors']},
              'reload_all_selection':reload_check,'selection_metrics':metrics,
              'matched_step_native_macro':ex.read(ex.OLD/f'vanilla/selection-{step}.json')['macro'],
              'same_step_wins':wins,'same_step_losses':losses,
              'ablation_samples':len(sample),'active':active_metrics,'write_back_blocked':blocked_metrics,
              'contentless_seeds':neutral_metrics,'block_write_effect':compare(active,blocked),
              'remove_content_effect':compare(active,neutral),
              'projection_relative_change_from_identity':norm_change,
              'slot_order_only_prediction_mismatches':order_mismatch,'slot_order_only_max_probability_delta':order_delta,
              'timing':{'samples':len(sample),'passes':dict(timing),'median_seconds':med,
                  'overhead_percent':100*(med['candidate_bottleneck']/med['native_same_weights']-1),
                  'scope':'same trained weights; prepared GPU inputs; tokenization/loading excluded; not a service benchmark'},
              'new_confirmation_cases_scored':0,
              'limits':['single seed, fixed small retrofit budget; selection is exposed, not confirmation',
                        'early stop compares the same update; final best-native comparison may use more updates',
                        'path interventions use the same trained weights, not separate trained controls',
                        'no claim of superiority from gradient size, path activity, or source-specific point estimates']}
    train.atomic_json(out/'audit.json',result)
    train.atomic_json(Path('wiki/project/workspace-bottleneck-results.json'),result)
    print('AUDIT',reload_check,'wins/losses',wins,losses,'timing',result['timing'],flush=True)


if __name__=='__main__':
    main()
