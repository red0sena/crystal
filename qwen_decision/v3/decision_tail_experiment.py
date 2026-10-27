"""Matched candidate-query tail experiment with a fresh, sealed confirmation set.

Reuse the fully audited native control only if its data, recipe, implementation,
weights and all 512 per-update prompt hashes match. Fit single/candidate queries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from qwen_decision.v3 import architecture_search as search, decision_tail as dt
from qwen_decision.v3 import format as fmt, modeling, train
from qwen_decision.v3 import pointer_experiment as pilot, readout_experiment as previous
from qwen_decision.v3 import recurrent_experiment as old, robust_adaptation as views
from qwen_decision.v3 import robust_experiment as evaluation

OUT = Path("runs/decision-tail/trained-v1")
OLD = Path("runs/recurrent/internal4-v1")
BATCH = old.BATCH


def read(path): return json.loads(Path(path).read_text())


def source_files():
    from qwen_decision.v3 import engine
    return [Path(m.__file__) for m in [dt, modeling, train, fmt, views, old, evaluation, engine]]+[Path(__file__)]


def implementation(): return {str(p): train.sha256_file(p) for p in source_files()}


def encode(renderer, row, rng=None):
    e = views.view(renderer, row) if rng is None else views.random_view(renderer, row, rng)
    q = row["question"]
    if q["type"] == "choice":
        keys = list(q["criteria"])
        q = {**q, "criteria": {keys[i]:q["criteria"][keys[i]] for i in e["order"]}}
    by_id = dict(zip(renderer.code_ids, renderer.codes))
    result = fmt.encode_options(renderer.tokenizer, row["state"], q,
                                [by_id[i] for i in e["code_ids"]], renderer.args.layout)
    if result["input_ids"] != e["input_ids"]:
        raise ValueError("offset rendering changed the matched prompt")
    return {**e, "option_positions":result["option_positions"]}


def logits_for(model, rows, pad_id, mode):
    ids, mask, codes, valid, targets = train.collate(rows, pad_id)
    if mode == "vanilla":
        z = modeling.code_logits(model, ids, mask, codes, valid)
    else:
        positions = torch.tensor([r["option_positions"]+[0]*(codes.shape[1]-len(r["target"]))
                                  for r in rows], device=ids.device)
        z = dt.logits(model, ids, mask, codes, valid, positions, mode)
    return z, targets


@torch.no_grad()
def evaluate(model, renderer, rows, mode):
    model.eval(); train.set_checkpointing(model, False)
    encoded = [encode(renderer, r) for r in rows]
    probabilities = {}
    torch.cuda.synchronize(); started = time.monotonic(); calls = 0
    for group in train.microbatches(encoded, BATCH):
        z, _ = logits_for(model, group, renderer.tokenizer.pad_token_id, mode)
        for row, p in zip(group, z.softmax(-1).cpu()):
            probabilities[row["id"]] = p[:len(row["target"])].tolist()
        calls += 1
    torch.cuda.synchronize(); elapsed = time.monotonic()-started
    records = []
    for row in rows:
        p = probabilities[row["id"]]; pred = max(range(len(p)), key=p.__getitem__)
        records.append({k:row[k] for k in ["id","source"]} |
                       {"group":row.get("group",row["id"]), "prediction":pred,
                        "correct":row["target"][pred] >= max(row["target"])-1e-6,
                        "probabilities":p, "views":1})
    return search.metrics(records) | {"seconds":elapsed,"batches":calls,
        "tokens":sum(len(r["input_ids"]) for r in encoded),"prompt_evaluations":len(encoded)}, records


def prepare(out):
    if out.exists(): raise ValueError("preserve experiments: choose a new output directory")
    old.verify(OLD)
    old_m = read(OLD/"manifest.json"); old_p = old_m["plan"]
    plan = {**old_p, "arms":["vanilla","single","candidates"],
            "architecture":{"rank":64,"adapter_parameters":262144,
                            "query":"decision + down(silu(up([norm(decision),norm(option),product])))",
                            "queries":"K; all retain original decision-token RoPE position",
                            "shared_blocks":23,"final_block":"native weights, Q/FFN only on decision queries"},
            "adapter_lr":3e-4,"gate_lr":None,
            "selection":"best of trained steps 256/512 per arm; old selection set reused, no step-0 substitution",
            "confirmation":"one primary candidates-vs-vanilla comparison; 5000 source-stratified document bootstrap draws, 95% CI; single-query ablation secondary",
            "native_control":"reuse prior native fit after complete recipe/data/source/prompt/checkpoint verification; no retraining claimed",
            "fixed_data_update_budget":True,"matched_compute_budget":False}
    out.mkdir(parents=True)
    train.atomic_json(out/"plan.json", plan)
    for name in ["train.jsonl","selection.jsonl"]: shutil.copyfile(OLD/name,out/name)
    # Build fresh groups without consulting any candidate predictions/accuracy.
    dev = pilot.read_rows(Path("data/mix/v3-gold6.dev.jsonl"))
    groups = old.groups_for(dev); key_ids = defaultdict(set)
    for row in dev:
        for key in old.keys(row): key_ids[key].add(row["id"])
    blocked = set(); audit = {}
    paths = [Path("data/mix/v3-gold3.dev.jsonl"),Path("data/fresh/fresh-v1.jsonl"),
             Path("runs/robust/fresh-v2-holdback/examples.jsonl"),
             OLD/"selection.jsonl",OLD/"confirmation.jsonl"]
    for path in paths:
        before = len(blocked)
        for row in previous.rows(path):
            for key in old.keys(row): blocked.update(key_ids.get(key,()))
        audit[str(path)] = {"new_blocked":len(blocked)-before,"sha256":train.sha256_file(path)}
    import pyarrow.parquet as pq
    from qwen_decision.v3.fresh_holdback import state_key
    for batch in pq.ParquetFile("data/jeval/data/test.parquet").iter_batches(columns=["state"]):
        for row in batch.to_pylist(): blocked.update(key_ids.get(("canonical_state",state_key(row["state"])),()))
    audit["after_jeval"] = len(blocked)
    for path_s, sha in old_m["input_sha256"].items():
        path = Path(path_s)
        if train.sha256_file(path) != sha: raise ValueError("upstream training data changed")
        count = 0; before = len(blocked)
        for line in path.open():
            row = json.loads(line); count += 1
            for key in old.keys(row): blocked.update(key_ids.get(key,()))
            if count%200000 == 0: print("AUDIT",path_s,count,flush=True)
        audit[path_s] = {"rows":count,"new_blocked":len(blocked)-before,"sha256":sha}
    blocked_groups = {groups[i] for i in blocked}
    renderer,_ = search.renderer_for(search.BASE)
    pools = defaultdict(lambda:defaultdict(list))
    for row in dev:
        if groups[row["id"]] in blocked_groups: continue
        if len(views.view(renderer,row)["input_ids"]) > plan["max_eval_tokens"]: continue
        pools[row["source"]][groups[row["id"]]].append({**row,"group":groups[row["id"]]})
    confirmation = []
    for source, gg in sorted(pools.items()):
        if len(gg) < 2: continue
        selected = sorted(gg,key=lambda g:search.digest(["decision-tail-confirm-v1",g]))[:128]
        for g in selected: confirmation += gg[g]
    if len(confirmation)<200: raise ValueError("insufficient fresh confirmation; preserve audit")
    splits = {"train":previous.rows(out/"train.jsonl"),"selection":previous.rows(out/"selection.jsonl"),
              "confirmation":confirmation}
    pilot.check_disjoint(splits)
    search.save_records(out/"confirmation.jsonl",confirmation)
    # Verify cached control really followed the declared, common training recipe.
    for key in ["seed","steps","batch_size","body_lr","warmup","snapshots","optimizer","augmentation"]:
        if plan[key] != old_p[key]: raise ValueError("control recipe mismatch: "+key)
    control = OLD/"vanilla"
    logs = previous.rows(control/"train.jsonl")
    if len(logs)!=plan["steps"] or logs[-1]["step"]!=plan["steps"]: raise ValueError("incomplete cached control")
    native_options = [(read(control/f"selection-{s}.json")["macro"],s) for s in plan["snapshots"]]
    _, native_step = sorted(native_options,key=lambda x:(-x[0],x[1]))[0]
    cached = {"path":str(control),"selected_step":native_step,
              "train_log_sha256":train.sha256_file(control/"train.jsonl"),
              "selection_files":{str(control/f"selection-{s}.{ext}"):train.sha256_file(control/f"selection-{s}.{ext}")
                                   for s in plan["snapshots"] for ext in ["json","jsonl"]},
              "implementation":old_m["implementation"],
              "checkpoint_files":{str(p):train.sha256_file(p) for s in plan["snapshots"]
                for p in (control/f"step-{s}").iterdir() if p.name in
                {"model.safetensors","config.json","v3_config.json"}}}
    (out/"source").mkdir()
    for p in source_files(): shutil.copyfile(p,out/"source"/p.name)
    manifest = {"plan":plan,"counts":{k:len(v) for k,v in splits.items()},
                "sources":{k:dict(Counter(r["source"] for r in v)) for k,v in splits.items()},
                "groups":{k:len({r.get('group',r['id']) for r in v}) for k,v in splits.items()},
                "files":{k+".jsonl":train.sha256_file(out/(k+".jsonl")) for k in splits},
                "implementation":implementation(),"base_sha256":old_m["base_sha256"],
                "audit":audit,"cached_control":cached,
                "scope":"fresh remaining gold6-dev groups; exclude all previously exposed old dev/fresh/JEVal/recurrent selection+confirmation and full gold2/3/6 training; shared source domains remain; no semantic-paraphrase audit"}
    train.atomic_json(out/"manifest.json",manifest)
    print("PREPARED",manifest["counts"],{k:len(v) for k,v in manifest["sources"].items()},flush=True)


def verify(out):
    m = read(out/"manifest.json")
    if m["implementation"] != implementation(): raise ValueError("implementation changed after freeze")
    for name, sha in m["files"].items():
        if train.sha256_file(out/name)!=sha: raise ValueError("experimental data changed")
    if train.sha256_file(search.BASE/"model.safetensors") != m["base_sha256"]: raise ValueError("base changed")
    c = m["cached_control"]
    if train.sha256_file(Path(c["path"])/"train.jsonl") != c["train_log_sha256"]: raise ValueError("control log changed")
    for path,sha in (c["selection_files"] | c["checkpoint_files"]).items():
        if train.sha256_file(Path(path)) != sha: raise ValueError("cached control changed")
    return m


def load_arm(arm, path=search.BASE):
    model = dt.load(path)
    text = model.model.language_model
    if arm != "vanilla" and not isinstance(text.layers[-1],dt.DecisionTail): dt.install(text)
    model.requires_grad_(False); text.requires_grad_(True); model.lm_head.requires_grad_(True)
    if arm == "single":
        text.layers[-1].up.requires_grad_(False); text.layers[-1].down.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    return model


def optimizer_for(model,plan,arm):
    optimizer = train.MasterAdamW(model,plan["body_lr"])
    if arm == "candidates":
        tail = model.model.language_model.layers[-1]
        params = {id(tail.up.weight),id(tail.down.weight)}
        masters = [m for p,m in optimizer.pairs if id(p) in params]
        master_ids = {id(m) for m in masters}
        for group in optimizer.param_groups: group["params"] = [p for p in group["params"] if id(p) not in master_ids]
        optimizer.optimizer.add_param_group({"params":masters,"weight_decay":.01,
                                             "lr_scale":plan["adapter_lr"]/plan["body_lr"]})
    return optimizer


def fit(out,arm,smoke=False):
    m = verify(out); plan = m["plan"]
    dest = out/("smoke-"+arm if smoke else arm)
    if dest.exists(): raise ValueError("preserve existing fit")
    dest.mkdir()
    renderer,cfg = search.renderer_for(search.BASE)
    rows = previous.rows(out/"train.jsonl")
    selection = previous.rows(out/"selection.jsonl")
    control = previous.rows(Path(m["cached_control"]["path"])/"train.jsonl")
    train.seed_all(plan["seed"])
    model = load_arm(arm); optimizer = optimizer_for(model,plan,arm)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    if not smoke:
        score,records = evaluate(model,renderer,selection,arm)
        train.atomic_json(dest/"selection-0.json",score); search.save_records(dest/"selection-0.jsonl",records)
        print("SELECTION",arm,0,score["macro"],flush=True)
    rng = random.Random(plan["seed"]+1000)
    total_steps = 3 if smoke else plan["steps"]
    with (dest/"train.jsonl").open("w") as log:
        for step,(indices,_) in enumerate(train.batches(len(rows),plan["batch_size"],total_steps,random.Random(plan["seed"])),1):
            model.train(); train.set_checkpointing(model,True); optimizer.zero_grad()
            encoded = [encode(renderer,rows[i],rng) for i in indices]
            prompt_hash = search.digest([r["input_ids"] for r in encoded])
            if prompt_hash != control[step-1]["prompt_sha256"]: raise ValueError("cached native control prompt mismatch")
            loss_sum = 0.
            for group in train.microbatches(encoded,BATCH):
                z,target = logits_for(model,group,renderer.tokenizer.pad_token_id,arm)
                loss = train.soft_ce(z,target).sum()/len(indices)
                if not torch.isfinite(loss): raise ValueError("nonfinite loss")
                loss.backward(); loss_sum += loss.item()
            adapter_grad = None
            if arm == "candidates":
                grad = model.model.language_model.layers[-1].down.weight.grad
                if grad is None or not torch.isfinite(grad).all(): raise ValueError("missing/nonfinite adapter gradient")
                adapter_grad = grad.float().norm().item()
            norm = optimizer.clip_grad_norm(1.)
            if not math.isfinite(norm): raise ValueError("nonfinite gradient")
            scale = min(step/plan["warmup"],1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-plan["warmup"])/(plan["steps"]-plan["warmup"]))))
            for group in optimizer.param_groups: group["lr"] = plan["body_lr"]*scale*group["lr_scale"]
            optimizer.step()
            entry = {"step":step,"loss":loss_sum,"grad_norm":norm,"seconds":time.monotonic()-started,
                     "rows_seen":step*len(indices),"prompt_sha256":prompt_hash}
            if arm == "candidates":
                tail = model.model.language_model.layers[-1]
                entry["adapter_down_rms"] = tail.down.weight.detach().square().mean().sqrt().item()
                entry["adapter_down_grad_norm"] = adapter_grad
            log.write(json.dumps(entry)+"\n"); log.flush()
            if step%16==0 or smoke: print("FIT",arm,entry,flush=True)
            if not smoke and step in plan["snapshots"]:
                dt.save(model,renderer.tokenizer,dest/f"step-{step}",cfg|{"steps":step,
                        "trained_from":str(search.BASE),"experiment":"decision-tail-trained-v1","arm":arm},mode=arm)
                score,records = evaluate(model,renderer,selection,arm)
                train.atomic_json(dest/f"selection-{step}.json",score)
                search.save_records(dest/f"selection-{step}.jsonl",records)
                print("SELECTION",arm,step,score["macro"],flush=True)
    summary = {"arm":arm,"steps":total_steps,"seconds":time.monotonic()-started,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_allocated_gb":torch.cuda.max_memory_allocated()/1e9,"all_prompt_hashes_match_cached_native":True}
    if smoke:
        optimizer.zero_grad(); _,expected = evaluate(model,renderer,rows[:8],arm)
        dt.save(model,renderer.tokenizer,dest/"checkpoint",cfg,mode=arm)
        del optimizer,model; gc.collect(); torch.cuda.empty_cache()
        model = dt.load(dest/"checkpoint").requires_grad_(False)
        _,actual = evaluate(model,renderer,rows[:8],arm)
        summary["reload_prediction_mismatches"] = sum(a["prediction"]!=b["prediction"] for a,b in zip(expected,actual))
        summary["reload_max_probability_delta"] = max(abs(a-b) for aa,bb in zip(expected,actual) for a,b in zip(aa["probabilities"],bb["probabilities"]))
        if summary["reload_prediction_mismatches"] or summary["reload_max_probability_delta"]>1e-6: raise ValueError("reload differs")
    train.atomic_json(dest/"result.json",summary)


def freeze(out):
    m = verify(out)
    if (out/"frozen.json").exists(): raise ValueError("already frozen")
    choices = {}
    for arm in m["plan"]["arms"]:
        dest = Path(m["cached_control"]["path"]) if arm=="vanilla" else out/arm
        options = [(read(dest/f"selection-{s}.json")["macro"],s) for s in m["plan"]["snapshots"]]
        score,step = sorted(options,key=lambda x:(-x[0],x[1]))[0]
        path = dest/f"step-{step}"
        choices[arm] = {"path":str(path),"step":step,"selection_macro":score,
                        "files":{p.name:train.sha256_file(p) for p in path.iterdir() if p.name in
                        {"model.safetensors","decision_tail.safetensors","decision_architecture.json","config.json","v3_config.json"}}}
    train.atomic_json(out/"frozen.json",{"choices":choices,"confirmation_sha256":m["files"]["confirmation.jsonl"]})
    print("FROZEN",choices,flush=True)


def confirm(out):
    m = verify(out); frozen = read(out/"frozen.json")
    if (out/"results.json").exists(): raise ValueError("confirmation already exposed")
    rows = previous.rows(out/"confirmation.jsonl"); renderer,_ = search.renderer_for(search.BASE)
    result = {"manifest":m,"frozen":frozen,"arms":{}}; records_by = {}
    for arm,choice in frozen["choices"].items():
        path = Path(choice["path"])
        for name,sha in choice["files"].items():
            if train.sha256_file(path/name)!=sha: raise ValueError("frozen checkpoint changed")
        model = load_arm(arm,path).requires_grad_(False)
        metric,records = evaluate(model,renderer,rows,arm)
        result["arms"][arm] = metric; records_by[arm] = records
        search.save_records(out/f"confirmation-{arm}.jsonl",records)
        train.atomic_json(out/f"confirmation-{arm}.json",metric)
        print("CONFIRM",arm,metric["macro"],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    for base,candidate,key in [("vanilla","candidates","primary"),("vanilla","single","single_ablation"),("single","candidates","conditioning_ablation")]:
        delta = evaluation.paired_interval(records_by[base],records_by[candidate],confidence=.95)
        delta.pop("paired_document_bootstrap_97_5ci")
        result[key] = delta
    train.atomic_json(out/"results.json",result)
    print("PRIMARY",result["primary"],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","smoke","fit","freeze","confirm"])
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--arm",choices=["single","candidates"])
    a = p.parse_args()
    if a.stage=="prepare": prepare(a.output)
    elif a.stage in {"smoke","fit"}:
        if a.arm is None: p.error("--arm required")
        fit(a.output,a.arm,a.stage=="smoke")
    elif a.stage=="freeze": freeze(a.output)
    else: confirm(a.output)


if __name__=="__main__": main()
