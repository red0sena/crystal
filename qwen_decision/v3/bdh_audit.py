"""Compare fixed-budget BDH/data arms and verify the actual saved computation."""
from collections import Counter,defaultdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from qwen_decision.v3 import bdh_experiment as ex,bdh_memory as bdh,modeling,train
from qwen_decision.v3 import architecture_search as search,readout_experiment as records
from qwen_decision.v3 import robust_experiment as statistics_helper
from qwen_decision.v3.hypothesis_audit import compare


def interval(a,b):
    result = statistics_helper.paired_interval(a,b,confidence=.95)
    result.pop('paired_document_bootstrap_97_5ci',None)
    result['exploratory_interval_above_zero'] = result.pop('confirmed_improvement')
    result['scope'] = 'exploratory paired interval on repeatedly exposed selection; not confirmation'
    return result


def exposure_counts(out,schedule):
    probe = {r['id'] for r in records.rows(out/'train-probe.jsonl')}
    result = {}
    for condition in ['small','large']:
        rows = records.rows(out/(condition+'.jsonl'));seen = Counter();result[condition] = {}
        for step in schedule:
            seen.update(rows[i]['id'] for i in step[condition])
            if step['step'] in [256,512,1024]:
                result[condition][str(step['step'])] = {
                    'probe_seen':sum(seen[i]>0 for i in probe),
                    'probe_mean_visits':sum(seen[i] for i in probe)/len(probe),
                    'unique_training_rows':len(seen)}
    return result


def initial_training_pool_overlap(out):
    """Additional adaptation rows need not be new to the original decision pool."""
    path = Path('data/mix/v3-gold3.jsonl')
    expected = ex.read(ex.old.OUT/'manifest.json')['input_sha256'][str(path)]
    if train.sha256_file(path)!=expected:raise ValueError('initial training corpus changed')
    datasets = {k:records.rows(out/(k+'.jsonl')) for k in ['small','large']}
    wanted_ids = {r['id'] for rr in datasets.values() for r in rr}
    wanted_states = {ex.state_key(r['state']) for rr in datasets.values() for r in rr}
    found_ids = set();found_states = set();count = 0
    with path.open() as f:
        for line in f:
            r = json.loads(line);count+=1
            if r['id'] in wanted_ids:found_ids.add(r['id'])
            key = ex.state_key(r['state'])
            if key in wanted_states:found_states.add(key)
    return {'file':str(path),'sha256':expected,'rows':count,
            'conditions':{k:{'rows':len(rr),
                'ids_in_initial_training_file':sum(r['id'] in found_ids for r in rr),
                'states_in_initial_training_file':sum(ex.state_key(r['state']) in found_states for r in rr)}
                for k,rr in datasets.items()},
            'scope':'membership in original gold3 decision-training file; not proof an exact row was drawn during base fitting, nor an audit of foundation pretraining'}


@torch.no_grad()
def main():
    out = ex.OUT;m = ex.verify(out);p = m['plan']
    if ex.read(out/'plan.json')!=p:raise ValueError('published plan differs from frozen manifest')
    rows = records.rows(out/'selection.jsonl');renderer,_ = search.renderer_for(search.BASE)
    traces = {arm:records.rows(out/arm/'train.jsonl') for arm in p['arms']}
    if any(len(v)!=1024 for v in traces.values()):raise ValueError('incomplete arm')
    reference = traces['bdh_large']
    for arm,logs in traces.items():
        if [r['source_trace_sha256'] for r in logs]!=[r['source_trace_sha256'] for r in reference]:
            raise ValueError('source trace differs')
    if ([r['prompt_sha256'] for r in traces['native_large']]!=
            [r['prompt_sha256'] for r in reference]):raise ValueError('architecture prompts differ')
    predictions = {};arms = {};checks = {}
    for arm in p['arms']:
        kind = arm.split('_')[0];path = out/arm/'step-1024'
        model = ex.load_arm(kind,path).requires_grad_(False).eval()
        metrics,rr = ex.evaluate(model,renderer,rows,kind)
        expected = records.rows(out/arm/'selection-1024.jsonl')
        check = compare(expected,rr)
        if check['prediction_mismatches'] or check['max_probability_delta']:
            raise ValueError('saved model did not reproduce selection')
        predictions[arm] = rr;checks[arm] = check
        arms[arm] = {'fit':ex.read(out/arm/'result.json'),'selection':metrics,
                     'train_probe':ex.read(out/arm/'train-probe-1024.json'),
                     'checkpoint_files':{f.name:train.sha256_file(f) for f in path.iterdir() if f.name in
                        ['model.safetensors','config.json','v3_config.json','decision_architecture.json','bdh_memory.safetensors']}}
        print('RELOAD',arm,check,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    architecture = interval(predictions['native_large'],predictions['bdh_large'])
    data = interval(predictions['bdh_small'],predictions['bdh_large'])
    probe_records = {arm:records.rows(out/arm/'train-probe-1024.jsonl') for arm in p['arms']}
    common_sources = set(r['source'] for r in rows)&set(r['source'] for r in probe_records['bdh_large'])
    common_gap = {}
    for arm in p['arms']:
        training = search.metrics([r for r in probe_records[arm] if r['source'] in common_sources])
        heldout = search.metrics([r for r in predictions[arm] if r['source'] in common_sources])
        common_gap[arm] = {'sources':len(common_sources),'train_probe_macro':training['macro'],
                           'selection_macro':heldout['macro'],'gap_pp':100*(training['macro']-heldout['macro'])}
    lengths = {r['id']:len(ex.views.view(renderer,r)['input_ids']) for r in rows}
    options = {r['id']:len(r['target']) for r in rows}
    diagnostic_groups = {
        'length_up_to_512':{i for i,n in lengths.items() if n<=512},
        'length_513_to_1536':{i for i,n in lengths.items() if 512<n<=1536},
        'length_above_train_limit_1536':{i for i,n in lengths.items() if n>1536},
        'options_up_to_4':{i for i,n in options.items() if n<=4},
        'options_5_to_16':{i for i,n in options.items() if 4<n<=16},
        'options_above_16':{i for i,n in options.items() if n>16}}
    diagnostics = {name:{arm:search.metrics([r for r in rr if r['id'] in ids])
                         for arm,rr in predictions.items()} for name,ids in diagnostic_groups.items() if ids}
    by = defaultdict(list)
    for r in rows:by[r['source']].append(r)
    sample = []
    for source in sorted(by):
        sample += sorted(by[source],key=lambda r:search.digest(['bdh-audit',r['id']]))[:8]
    sample = sorted(sample,key=lambda r:search.digest(['bdh-audit-sample',r['id']]))[:512]
    model = ex.load_arm('bdh',out/'bdh_large/step-1024').requires_grad_(False).eval()
    active_metric,active = ex.evaluate(model,renderer,sample,'bdh')
    removed_metric,removed = ex.evaluate(model,renderer,sample,'bdh',memory_enabled=False)
    cycle_sensitivity = {'3':active_metric}
    trained_cycles = model.model.language_model.bdh_memory.cycles
    try:
        for cycles in [1,2,4,6]:
            model.model.language_model.bdh_memory.cycles = cycles
            metric,_ = ex.evaluate(model,renderer,sample,'bdh')
            cycle_sensitivity[str(cycles)] = metric
    finally:
        model.model.language_model.bdh_memory.cycles = trained_cycles
    encoded = [ex.views.view(renderer,r) for r in sample]
    groups = list(train.microbatches(encoded,ex.BATCH));group = groups[0]
    ids,mask,codes,valid,_ = train.collate(group,renderer.tokenizer.pad_token_id)
    text = model.model.language_model;core = text.bdh_memory
    counts = [0]*24;captured = {};handles = []
    for i,layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(lambda m,a,o,i=i:counts.__setitem__(i,counts[i]+1)))
    handles.append(text.layers[15].register_forward_hook(lambda m,a,o:captured.__setitem__('prefix',o)))
    handles.append(core.register_forward_hook(lambda m,a,o:captured.__setitem__('core',o)))
    bdh.logits(model,ids,mask,codes,valid)
    for handle in handles:handle.remove()
    if counts!=[1]*16+[0]*8:raise ValueError('unexpected native blocks executed')
    x = bdh.norm(captured['prefix'])*mask[:,:,None];changes = []
    for _ in range(core.cycles):
        before = x;x = core.step(x,mask.bool())
        keep = mask[:,:,None]
        changes.append({'relative_state_change_norm':((x-before).float()*keep).norm().item()/
                        (before.float()*keep).norm().item()})
    manual_delta = (x-captured['core']).abs().max().item()
    drift = {}
    for name,source in [('encoder',text.layers[16].mlp.up_proj.weight),
                        ('encoder_v',text.layers[16].mlp.gate_proj.weight),
                        ('decoder',text.layers[16].mlp.down_proj.weight)]:
        initial = source.detach().cpu().float();actual = getattr(core,name).detach().cpu().float()
        drift[name] = (actual-initial).norm().item()/initial.norm().item()
    # Real CUDA padding comparison; report numeric error rather than claiming bitwise invariance.
    longest = max(len(r['input_ids']) for r in group)
    batched,_ = ex.logits_for(model,group,renderer.tokenizer.pad_token_id,'bdh')
    solo_max = 0.;solo_mismatches = 0
    for i,r in enumerate(group):
        z,_ = ex.logits_for(model,[r],renderer.tokenizer.pad_token_id,'bdh')
        n = len(r['target']);a = batched[i,:n].softmax(-1);b = z[0,:n].softmax(-1)
        solo_max = max(solo_max,(a-b).abs().max().item())
        solo_mismatches += int(a.argmax()!=b.argmax())
    native = ex.load_arm('native',out/'native_large/step-1024').requires_grad_(False).eval()
    cached = [train.collate(g,renderer.tokenizer.pad_token_id)[:4] for g in groups]
    timing = defaultdict(list)
    def run(kind,batches):
        torch.cuda.synchronize();started = time.monotonic()
        for args in batches:
            if kind=='native_large':modeling.code_logits(native,*args)
            else:bdh.logits(model,*args)
        torch.cuda.synchronize()
        return time.monotonic()-started
    for kind in ['native_large','bdh_large']:run(kind,cached[:16])
    for i in range(3):
        for kind in (['native_large','bdh_large'] if i%2==0 else ['bdh_large','native_large']):
            timing[kind].append(run(kind,cached))
    med = {k:statistics.median(v) for k,v in timing.items()}
    result = {'manifest':m,'arms':arms,'reload_all_selection':checks,
              'fixed_1024_update_architecture_difference':architecture,
              'fixed_1024_update_data_difference':data,
              'common_source_train_selection_gap':common_gap,
              'posthoc_length_and_candidate_diagnostics':diagnostics,
              'diagnostic_scope':'descriptive exposed-selection bins; not extra confirmation tests or selection criteria',
              'all_1024_architecture_prompt_hashes_match':True,'all_three_1024_source_traces_match':True,
              'train_probe_exposures':exposure_counts(out,records.rows(out/'schedule.jsonl')),
              'initial_training_pool_overlap':initial_training_pool_overlap(out),
              'ablation_samples':len(sample),'ablation_active':active_metric,'ablation_no_memory':removed_metric,
              'posthoc_refinement_cycle_sensitivity':cycle_sensitivity,
              'cycle_probe_scope':'fixed512 exposed selection; model trained at3; diagnostic only, no reselection from this probe',
              'memory_ablation_effect':compare(active,removed),'native_block_calls':counts,
              'core_replay_max_hidden_delta':manual_delta,'core_cycle_changes':changes,'core_relative_weight_drift':drift,
              'cuda_padding_probe':{'rows':len(group),'padded_width':longest,'prediction_mismatches':solo_mismatches,
                                    'max_probability_delta':solo_max},
              'timing':{'samples':len(sample),'passes':dict(timing),'median_seconds':med,
                  'overhead_percent':100*(med['bdh_large']/med['native_large']-1),
                  'scope':'trained models, same prepared GPU batches, alternated warm passes; excludes tokenization/loading; not a service benchmark'},
              'advance_to_fresh_confirmation':architecture['macro_delta']>=.005,
              'environment':train.versions(),'audit_implementation_sha256':train.sha256_file(Path(__file__)),
              'audit_dependency_sha256':{str(f):train.sha256_file(f) for f in
                  [Path(statistics_helper.__file__),Path('qwen_decision/v3/hypothesis_audit.py'),
                   Path('qwen_decision/v3/fresh_holdback.py')]},
              'new_confirmation_cases_scored':0,
              'limits':p['limits']+['native-small not trained; no architecture-data interaction estimate',
                'different logical parameter/FLOP counts; BDH replaced eight pretrained blocks with a shared small core',
                'same initial decision checkpoint, not equal pretraining from scratch',
                'data intervention is source-matched nested sampling, not a guarantee that only semantic information quantity differs']}
    train.atomic_json(out/'report.json',result)
    train.atomic_json(Path('wiki/project/bdh-memory-results.json'),result)
    print('FINAL_ARCHITECTURE',architecture,flush=True)
    print('FINAL_DATA',data,flush=True)
    print('MEMORY_ABLATION',result['memory_ablation_effect'],'TIMING',result['timing'],flush=True)


if __name__=='__main__':main()
