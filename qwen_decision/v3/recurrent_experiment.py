"""Matched full-backbone adaptation: vanilla versus an internal recurrent block.

prepare -> fit (vanilla/recurrent) -> freeze -> confirm. Confirmation examples
are never read by training/selection. No model/seed weight averaging is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import heapq
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from qwen_decision.v3 import architecture_search as search, modeling, recurrent, train
from qwen_decision.v3 import pointer_experiment as pilot, readout_experiment as previous
from qwen_decision.v3 import robust_adaptation as views, robust_experiment as evaluation
from qwen_decision.v3.fresh_holdback import state_key
from qwen_decision.v3.mix import allowed_origin

OUT = Path("runs/recurrent/internal4-v1")
BATCH = SimpleNamespace(pack=False, micro_batch=4, token_budget=4096)


def read(path):
    return json.loads(Path(path).read_text())


def keys(row):
    return previous.group_keys(row) + [("canonical_state", state_key(
        row["state"] if row["state"] not in (None, "", {}) else row["question"]))]


def groups_for(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in keys(row):
            if key in seen: parent[find(i)] = find(seen[key])
            seen[key] = i
    names = {}
    for i, row in enumerate(rows):
        root = find(i)
        names[root] = min(names.get(root, row["id"]), row["id"])
    return {row["id"]: names[find(i)] for i, row in enumerate(rows)}


def implementation():
    return {p.name: train.sha256_file(p) for p in
            [Path(__file__), Path(recurrent.__file__), Path(modeling.__file__), Path(train.__file__)]}


def prepare(out):
    if out.exists(): raise ValueError("preserve experiments: use a new directory")
    plan = {"base": str(search.BASE), "arms": ["vanilla", "recurrent"], "seed": 41,
            "architecture": {"shared_block": [16, 20], "extra_passes": 1,
                             "gate": "1024 FP32 channels, tanh, zero initialized"},
            "training": "full language backbone and tied embedding/LM head; vision frozen",
            "train_sampling": "lowest stable hash <=128 gold6 examples/source before length filtering",
            "steps": 512, "batch_size": 16, "body_lr": 3e-6, "gate_lr": .001,
            "micro_batch": 4, "padded_token_budget": 4096,
            "warmup": 32, "schedule": "linear warmup then cosine to 10%",
            "snapshots": [256, 512], "selection": "independent best step per arm, including step 0; source macro",
            "optimizer": "AdamW FP32 masters, betas(.9,.95), weight decay .01, clip global norm 1",
            "augmentation": "same per-step choice/code permutation draws in both arms; one view, CE gold only",
            "max_training_tokens": 1536, "max_eval_tokens": 4096,
            "confirmation": "one frozen recurrent-vs-matched-vanilla primary comparison, 5000 source-stratified document bootstrap draws, 95% CI",
            "fixed_data_update_budget": True, "matched_compute_budget": False,
            "no_model_or_seed_averaging": True, "no_new_teacher_labels_or_rationales": True}
    out.mkdir(parents=True)
    train.atomic_json(out / "plan.json", plan)
    dev = pilot.read_rows(Path("data/mix/v3-gold6.dev.jsonl"))
    groups = groups_for(dev)
    key_ids = defaultdict(set)
    for row in dev:
        for key in keys(row): key_ids[key].add(row["id"])
    blocked, audit = set(), {}
    def mark(row):
        for key in keys(row): blocked.update(key_ids.get(key, ()))
    # Entire old dev, not just selected/scored rows: all earlier search families
    # in this chat used gold3 dev. Also exclude both previously observed fresh sets.
    for path in ["data/mix/v3-gold3.dev.jsonl", "data/fresh/fresh-v1.jsonl",
                 "runs/robust/fresh-v2-holdback/examples.jsonl"]:
        for row in previous.rows(path): mark(row)
    audit["blocked_after_previous_dev_and_fresh"] = len(blocked)
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile("data/jeval/data/test.parquet").iter_batches(columns=["state"]):
        for row in batch.to_pylist():
            blocked.update(key_ids.get(("canonical_state", state_key(row["state"])), ()))
    audit["blocked_after_entire_jeval"] = len(blocked)
    pools = defaultdict(list)
    full_hashes = {}
    serial = 0
    for name in ["v3-gold2", "v3-gold3", "v3-gold6"]:
        path = Path("data/mix") / (name + ".jsonl")
        full_hashes[str(path)] = train.sha256_file(path)
        before, count = len(blocked), 0
        for line in path.open():
            row = json.loads(line); count += 1
            kk = keys(row)
            overlap = False
            for key in kk:
                found = key_ids.get(key)
                if found: blocked.update(found); overlap = True
            if name == "v3-gold6" and not overlap and allowed_origin(row, {"human", "programmatic"}):
                h = int(search.digest(["recurrent-train-v1", row["id"], row["question"]]), 16)
                pool = pools[row["source"]]
                item = (-h, serial, row); serial += 1
                if len(pool) < 128: heapq.heappush(pool, item)
                elif h < -pool[0][0]: heapq.heapreplace(pool, item)
            if count % 200000 == 0: print("AUDIT", name, count, flush=True)
        audit[name] = {"rows": count, "new_blocked_dev_ids": len(blocked)-before}
    blocked_groups = {groups[i] for i in blocked}
    renderer, _ = search.renderer_for(search.BASE)
    by = defaultdict(list)
    for row in dev:
        if groups[row["id"]] in blocked_groups: continue
        if len(views.view(renderer, row)["input_ids"]) <= plan["max_eval_tokens"]:
            by[row["source"]].append({**row, "group": groups[row["id"]]})
    splits = {"selection": [], "confirmation": []}
    for source, rows in sorted(by.items()):
        by_group = defaultdict(list)
        for row in rows: by_group[row["group"]].append(row)
        ordered = sorted(by_group, key=lambda g: search.digest(["recurrent-dev-v1", g]))
        if len(ordered) < 8: continue
        n_select = min(32, max(4, len(ordered)//3))
        for name, gg in [("selection", ordered[:n_select]),
                         ("confirmation", ordered[n_select:n_select+96])]:
            for g in gg: splits[name] += by_group[g]
    training = []
    for source, pool in sorted(pools.items()):
        for _, _, row in sorted(pool, reverse=True):
            row = train.gold_targets([row])[0]
            row = {**row, "upstream_id": row["id"],
                   "id": row["id"]+"#"+search.digest([row["state"], row["question"]])[:16]}
            if len(views.view(renderer, row)["input_ids"]) <= plan["max_training_tokens"]:
                training.append(row)
    splits["train"] = training
    pilot.check_disjoint(splits)
    if min(len(v) for v in splits.values()) == 0: raise ValueError("empty experimental split")
    for name, rows in splits.items(): search.save_records(out / (name+".jsonl"), rows)
    manifest = {"plan": plan, "counts": {k: len(v) for k,v in splits.items()},
                "sources": {k: dict(Counter(r["source"] for r in v)) for k,v in splits.items()},
                "audit": audit, "base_sha256": train.sha256_file(search.BASE/"model.safetensors"),
                "input_sha256": full_hashes, "implementation": implementation(),
                "files": {k+".jsonl": train.sha256_file(out/(k+".jsonl")) for k in splits},
                "scope": "new gold6 dev document groups excluding gold2/3/6 training and entire old gold3 dev/fresh-v1/v2/JEVal states; upstream/source overlap may remain; not semantic paraphrase audit"}
    train.atomic_json(out / "manifest.json", manifest)
    print("PREPARED", manifest["counts"], {k:len(v) for k,v in manifest["sources"].items()}, flush=True)


def verify(out):
    m = read(out/"manifest.json")
    if m["implementation"] != implementation(): raise ValueError("implementation changed; preserve this run")
    for name, sha in m["files"].items():
        if train.sha256_file(out/name) != sha: raise ValueError("prepared data changed")
    if train.sha256_file(search.BASE/"model.safetensors") != m["base_sha256"]: raise ValueError("base changed")
    return m


def load_arm(arm):
    model = recurrent.load(search.BASE)
    model.requires_grad_(False)
    model.model.language_model.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    if arm == "recurrent": recurrent.install(model.model.language_model)
    elif arm != "vanilla": raise ValueError(arm)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def optimizer_for(model, plan):
    optimizer = train.MasterAdamW(model, plan["body_lr"])
    text = model.model.language_model
    if hasattr(text, "_decision_recurrence"):
        gate = recurrent.loop_module(text).recurrence_gate
        master = next(m for p,m in optimizer.pairs if p is gate)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p is not master]
        optimizer.optimizer.add_param_group({"params": [master], "weight_decay": 0.,
                                             "lr_scale": plan["gate_lr"]/plan["body_lr"]})
    return optimizer


@torch.no_grad()
def evaluate(model, renderer, rows):
    train.set_checkpointing(model, False)
    grouped = [{**r, "group":r.get("group", r.get("upstream_id",r["id"]))} for r in rows]
    return evaluation.evaluate(model, renderer, grouped)


def fit(out, arm, smoke=False):
    manifest = verify(out); plan = manifest["plan"]
    dest = out / ("smoke-"+arm if smoke else arm)
    if dest.exists(): raise ValueError("preserve fits: destination already exists")
    dest.mkdir()
    renderer, cfg = search.renderer_for(search.BASE)
    rows = previous.rows(out/"train.jsonl")
    selection = previous.rows(out/"selection.jsonl")
    train.seed_all(plan["seed"])
    model = load_arm(arm)
    optimizer = optimizer_for(model, plan)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if not smoke:
        score, records = evaluate(model, renderer, selection)
        train.atomic_json(dest/"selection-0.json", score)
        search.save_records(dest/"selection-0.jsonl", records)
        print("SELECTION", arm, 0, score["macro"], flush=True)
    rng = random.Random(plan["seed"]+1000)
    total_steps = 3 if smoke else plan["steps"]
    optimizer.zero_grad()
    with (dest/"train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(train.batches(len(rows), plan["batch_size"], total_steps,
                                                          random.Random(plan["seed"])), 1):
            model.train(); train.set_checkpointing(model, True)
            optimizer.zero_grad()
            encoded = [views.random_view(renderer, rows[i], rng) for i in indices]
            loss_sum = 0.
            for group in train.microbatches(encoded, BATCH):
                z, targets = train.logits_for(model, group, renderer.tokenizer.pad_token_id)
                loss = train.soft_ce(z, targets).sum()/len(indices)
                if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
                loss.backward(); loss_sum += loss.item()
            norm = optimizer.clip_grad_norm(1.)
            if not math.isfinite(norm): raise RuntimeError("nonfinite gradient")
            scale = min(step/plan["warmup"], 1.) * (.1+.9*.5*(1+math.cos(
                math.pi*max(0,step-plan["warmup"])/(plan["steps"]-plan["warmup"]))))
            for group in optimizer.param_groups:
                group["lr"] = plan["body_lr"]*scale*group["lr_scale"]
            optimizer.step()
            entry = {"step":step, "loss":loss_sum, "grad_norm":norm,
                     "seconds":time.monotonic()-started, "rows_seen":step*len(indices),
                     "prompt_sha256":search.digest([r["input_ids"] for r in encoded])}
            if arm == "recurrent":
                gate = recurrent.loop_module(model.model.language_model).recurrence_gate.detach()
                entry["gate_rms"] = gate.square().mean().sqrt().item()
                entry["gate_max_abs"] = gate.abs().max().item()
            log.write(json.dumps(entry)+"\n"); log.flush()
            if step%16 == 0 or smoke: print("FIT",arm,entry,flush=True)
            if not smoke and step in plan["snapshots"]:
                recurrent.save(model, renderer.tokenizer, dest/f"step-{step}",
                               cfg | {"steps":step,"trained_from":str(search.BASE),
                                      "experiment":"internal4-v1", "arm":arm})
                score, records = evaluate(model, renderer, selection)
                train.atomic_json(dest/f"selection-{step}.json", score)
                search.save_records(dest/f"selection-{step}.jsonl", records)
                print("SELECTION",arm,step,score["macro"],flush=True)
    summary = {"arm":arm, "steps":total_steps,
        "seconds":time.monotonic()-started, "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_allocated_gb":torch.cuda.max_memory_allocated()/1e9}
    if smoke:
        optimizer.zero_grad()
        before, expected = evaluate(model, renderer, rows[:8])
        recurrent.save(model, renderer.tokenizer, dest/"checkpoint", cfg)
        del optimizer, model
        gc.collect(); torch.cuda.empty_cache()
        loaded = recurrent.load(dest/"checkpoint").requires_grad_(False)
        after, actual = evaluate(loaded, renderer, rows[:8])
        summary["reload_prediction_mismatches"] = sum(a["prediction"]!=b["prediction"] for a,b in zip(expected,actual))
        summary["reload_max_probability_delta"] = max(abs(a-b) for aa,bb in zip(expected,actual)
            for a,b in zip(aa["probabilities"],bb["probabilities"]))
        if summary["reload_prediction_mismatches"] or summary["reload_max_probability_delta"] > 1e-6:
            raise ValueError("serialized architecture does not reproduce the trained network")
    train.atomic_json(dest/"result.json", summary)


def freeze(out):
    m = verify(out)
    if (out/"frozen.json").exists(): raise ValueError("already frozen")
    choices = {}
    for arm in m["plan"]["arms"]:
        options = [(read(out/arm/f"selection-{step}.json")["macro"], step)
                   for step in [0]+m["plan"]["snapshots"]]
        score, step = sorted(options, key=lambda item:(-item[0],item[1]))[0]
        # Step 0 is a legitimate no-change control. A recurrent step-0 winner
        # means this architecture failed to improve selection, not a new model.
        path = out/arm/f"step-{step}" if step else search.BASE
        choices[arm] = {"step":step, "selection_macro":score, "path":str(path),
                        "weights_sha256":train.sha256_file(path/"model.safetensors"),
                        "checkpoint_files": {p.name:train.sha256_file(p) for p in path.iterdir()
                            if p.name in {"model.safetensors", "recurrence.safetensors",
                                          "decision_architecture.json", "v3_config.json", "config.json"}}}
    train.atomic_json(out/"frozen.json", {"choices":choices,"confirmation_sha256":m["files"]["confirmation.jsonl"]})
    print("FROZEN",choices,flush=True)


def confirm(out):
    m = verify(out); frozen = read(out/"frozen.json")
    if (out/"results.json").exists(): raise ValueError("confirmation already observed")
    rows = previous.rows(out/"confirmation.jsonl")
    renderer,_ = search.renderer_for(search.BASE)
    result = {"manifest":m,"frozen":frozen,"arms":{}}
    records_by = {}
    for arm, choice in frozen["choices"].items():
        path = Path(choice["path"])
        if train.sha256_file(path/"model.safetensors") != choice["weights_sha256"]: raise ValueError("frozen weights changed")
        for name, sha in choice["checkpoint_files"].items():
            if train.sha256_file(path/name) != sha: raise ValueError("frozen architecture changed: "+name)
        model = recurrent.load(path).requires_grad_(False)
        metrics, records = evaluate(model,renderer,rows)
        result["arms"][arm] = metrics
        records_by[arm] = records
        search.save_records(out/f"confirmation-{arm}.jsonl",records)
        # Verify serialized architecture by replaying independently loaded weights
        # on selection, without tuning on confirmation results.
        sel_metrics, sel_records = evaluate(model,renderer,previous.rows(out/"selection.jsonl"))
        expected = previous.rows(out/arm/f"selection-{choice['step']}.jsonl")
        mismatches = sum(a["prediction"]!=b["prediction"] for a,b in zip(expected,sel_records))
        if len(expected)!=len(sel_records) or mismatches: raise ValueError("checkpoint reload predictions differ")
        result["arms"][arm]["reload_selection_prediction_mismatches"] = mismatches
        print("CONFIRM",arm,metrics["macro"],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    delta = evaluation.paired_interval(records_by["vanilla"],records_by["recurrent"],confidence=.95)
    delta.pop("paired_document_bootstrap_97_5ci")
    result["difference"] = delta
    train.atomic_json(out/"results.json",result)
    print("DIFFERENCE",delta,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","smoke","fit","freeze","confirm"])
    p.add_argument("--arm",choices=["vanilla","recurrent"])
    p.add_argument("--output",type=Path,default=OUT)
    a=p.parse_args()
    if a.stage=="prepare": prepare(a.output)
    elif a.stage in ("smoke","fit"):
        if not a.arm: p.error("fit/smoke requires --arm")
        fit(a.output,a.arm,a.stage=="smoke")
    elif a.stage=="freeze": freeze(a.output)
    else: confirm(a.output)


if __name__=="__main__": main()
