"""Frozen native/reverse x CE/MNTP conversion factorial, two paired seeds.

Stage one: 1,024 matched extra updates. Stage two: 4,096 gold decision updates.
Every arm starts at the original base, with fresh Adam at each phase boundary.
Only the existing audited training pool is used for masked token targets.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from qwen_decision.v3 import long_training as common, masked_adaptation as masked
from qwen_decision.v3 import reverse_scan as reverse, reverse_scan_experiment as previous
from qwen_decision.v3 import train, modeling, architecture_search as search
from qwen_decision.v3 import readout_experiment as records

OUT = Path("runs/mntp-conversion/64k-factorial-v1")
REFERENCE = common.OUT
ARMS = ["native_ce", "native_mntp", "reverse_ce", "reverse_mntp"]


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    return previous.implementation() | {str(Path(m.__file__)): train.sha256_file(Path(m.__file__))
                                       for m in [masked, __import__(__name__, fromlist=["_"])]}


def prepare(out):
    if out.exists():
        raise ValueError("preserve existing experiments")
    original = common.verify(REFERENCE)
    renderer, _ = search.renderer_for(search.BASE)
    mask_ids = renderer.tokenizer.encode("_", add_special_tokens=False)
    if len(mask_ids) != 1 or mask_ids[0] in renderer.tokenizer.all_special_ids:
        raise ValueError("underscore must be one existing ordinary token")
    plan = original["plan"] | {
        "experiment": "mntp-conversion-factorial-v1", "arms": ARMS, "seeds": [41, 73],
        "pre_steps": 1024, "steps": 4096, "total_steps": 5120, "gate_lr": .001,
        "pre_lr": 3e-6, "body_lr": 3e-6, "warmup": 64,
        "phase_boundary": "preserve weights; reset FP32-master AdamW in ALL four arms; reseed downstream RNG",
        "pre_inputs": "every fourth cached CE batch (1,5,...4093); 16,384 examples across the 64k pool",
        "ce_inputs": "the exact original 4096-update augmented stream for each seed",
        "mask_id": mask_ids[0], "mask_token": "_", "mask_rate": .2, "head_chunk": 64,
        "masking": "mask 20% of natural-language state/question/option tokens; every selected token replaced; no unchanged/random target leakage",
        "mask_protection": "chat scaffold, section headings, option labels and special tokens excluded by exact offset spans",
        "mntp_readout": "full tied vocabulary at position i-1 predicts the original hidden token i; BF16 projection, FP32 CE",
        "mntp_weighting": "mean masked-token CE within each example, then mean across 16 examples",
        "mask_determinism": "SHA-derived seed for phase, seed, original cache step and example id; shared across architectures",
        "checkpoints": [1024, 3072, 5120], "selection_steps": [1024, 3072, 5120],
        "probe_steps": [256, 1024, 2048, 3072, 4096, 5120],
        "primary": "fixed final source-macro interaction (reverse_mntp-reverse_ce)-(native_mntp-native_ce), average of two paired seeds",
        "secondary": "MNTP effects within each structure, final reverse_mntp versus native_ce, learning curves, train-only masked probe",
        "primary_interval": "source-stratified document bootstrap paired across all 4 arms and both seeds; conditional on the two trained seeds",
        "no_best_snapshot_reselection": True, "no_accuracy_early_stop": True,
        "no_model_or_prediction_averaging": True, "teacher_targets": False,
        "gold_ce_only": False, "exposures_per_fit": 81920,
        "snapshots": [1024, 3072, 5120],
        "training": "full used language backbone and tied head; vision frozen; original 1536 token limit",
        "confirmation": "repeatedly exposed 4147-case selection only; no new holdback evaluated automatically",
        "limits": ["2 seeds do not establish population training variance", "no hyperparameter sweep; one frozen LR and mask recipe",
                   "matched examples and updates, not FLOPs or wall time: MNTP full-vocabulary CE and reverse scans cost more",
                   "20% fully replaced masking differs from BERT's unchanged/random replacement strategy",
                   "LLM2Vec evidence is from embedding tasks, not this hybrid 0.8B decision model",
                   "fixed train probes measure fitting, not unseen-language generalization",
                   "selection is repeatedly exposed and not independent confirmation"],
    }
    out.mkdir(parents=True)
    train.atomic_json(out / "plan.json", plan)
    files = [REFERENCE / name for name in ["train.jsonl", "selection.jsonl", "train-probe.jsonl",
                                          "manifest.json", "encoded-seed41.jsonl"]]
    train.atomic_json(out / "manifest.json", {
        "frozen": False, "plan": plan, "plan_sha256": train.sha256_file(out / "plan.json"),
        "base_sha256": original["base_sha256"], "files": {str(f): train.sha256_file(f) for f in files},
        "inherited_train_selection_audit": original["inherited_train_selection_audit"],
        "new_confirmation_cases_scored": 0})
    print("PREPARED", out, plan["arms"], plan["seeds"], flush=True)


def verify(out, *, frozen=True):
    manifest = read(out / "manifest.json")
    if frozen and not manifest["frozen"]:
        raise ValueError("smoke and freeze before fitting")
    if manifest["frozen"] and implementation() != manifest["implementation"]:
        raise ValueError("frozen implementation changed")
    if train.sha256_file(out / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("plan changed")
    if train.sha256_file(search.BASE / "model.safetensors") != manifest["base_sha256"]:
        raise ValueError("base weights changed")
    for filename, digest in manifest["files"].items():
        if train.sha256_file(Path(filename)) != digest:
            raise ValueError(f"frozen input changed: {filename}")
    return manifest


def cache(out):
    manifest = verify(out, frozen=False)
    if manifest["frozen"]:
        raise ValueError("do not change frozen inputs")
    plan = manifest["plan"]
    renderer, _ = search.renderer_for(search.BASE)
    rows = records.rows(REFERENCE / "train.jsonl")
    by_id = {r["id"]: r for r in rows}
    for seed in plan["seeds"]:
        dest = out / f"encoded-seed{seed}.jsonl"
        if dest.exists():
            if str(dest) in manifest["files"] and train.sha256_file(dest) == manifest["files"][str(dest)]:
                continue
            raise ValueError("preserve unverified cache")
        upstream = (REFERENCE / "encoded-seed41.jsonl").open() if seed == 41 else None
        try:
            with dest.with_suffix(".jsonl.tmp").open("w") as f:
                schedule = train.batches(len(rows), 16, 4096, random.Random(seed))
                for step, (indices, _) in enumerate(schedule, 1):
                    item = json.loads(next(upstream)) if upstream else {
                        "step": step, "indices": indices,
                        "rows": common.encoded_batch(renderer, rows, indices, seed, step)}
                    if item["step"] != step or item["indices"] != indices:
                        raise ValueError("sample stream differs from declared schedule")
                    for row in item["rows"]:
                        row["eligible_positions"] = masked.content_positions(renderer, by_id[row["id"]], row)
                        if len(row["input_ids"]) > 1536:
                            raise ValueError("training input exceeds original limit")
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if step % 512 == 0:
                        print("CACHE", seed, step, flush=True)
        finally:
            if upstream:
                upstream.close()
        dest.with_suffix(".jsonl.tmp").rename(dest)
        manifest["files"][str(dest)] = train.sha256_file(dest)
        train.atomic_json(out / "manifest.json", manifest)
    probe = []
    for raw in records.rows(REFERENCE / "train-probe.jsonl")[:128]:
        encoded = common.encoding.encode(renderer, raw)
        encoded["eligible_positions"] = masked.content_positions(renderer, raw, encoded)
        probe.append(encoded)
    probe_path = out / "masked-train-probe.jsonl"
    search.save_records(probe_path, probe)
    manifest["files"][str(probe_path)] = train.sha256_file(probe_path)
    train.atomic_json(out / "manifest.json", manifest)
    print("CACHE_READY", flush=True)


def load(arm, path=search.BASE):
    return previous.load(path) if arm.startswith("reverse_") else common.load_arm("native", path)


def optimizer(model, plan):
    return train.MasterAdamW(model, plan["body_lr"], gate_lr=plan["gate_lr"])


def view_for(row, plan, seed, cache_step, *, probe=False):
    key = ["mntp-train-probe-v1" if probe else "mntp-conversion-v1", seed, cache_step, row["id"]]
    return masked.masked_view(row, mask_id=plan["mask_id"], rate=plan["mask_rate"],
                              seed=int(search.digest(key)[:16], 16))


def update(model, opt, renderer, rows, arm, plan, step, seed, cache_step):
    pre = step <= plan["pre_steps"]
    phase_step = step if pre else step - plan["pre_steps"]
    local_plan = plan | {"steps": plan["pre_steps"] if pre else plan["steps"],
                         "body_lr": plan["pre_lr"] if pre else plan["body_lr"]}
    if not pre or not arm.endswith("_mntp"):
        loss, norm = common.update(model, opt, renderer, rows, "native", local_plan, phase_step)
        return loss, norm, {"objective": "decision_ce", "masked_tokens": 0}
    model.train()
    train.set_checkpointing(model, True)
    opt.zero_grad()
    total, count = 0., 0
    for group in train.microbatches(rows, common.BATCH):
        views = [view_for(r, plan, seed, cache_step) for r in group]
        loss = masked.loss(model, views, renderer.tokenizer.pad_token_id,
                           chunk_size=plan["head_chunk"]) / len(rows)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite masked loss")
        loss.backward()
        total += loss.item()
        count += sum(len(v["labels"]) for v in views)
    if phase_step == 1 and any(p.grad is None or not torch.isfinite(p.grad).all()
                               for p in model.parameters() if p.requires_grad):
        raise ValueError("missing or nonfinite MNTP gradients")
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError("nonfinite MNTP gradient norm")
    scale = min(phase_step / plan["warmup"], 1.) * (.1 + .45 * (1 + math.cos(
        math.pi * max(0, phase_step - plan["warmup"]) / (plan["pre_steps"] - plan["warmup"]))))
    for group in opt.param_groups:
        group["lr"] = plan["pre_lr"] * group["lr_scale"] * scale
    opt.step()
    return total, norm, {"objective": "masked_next_token", "masked_tokens": count}


@torch.no_grad()
def masked_probe(model, renderer, rows, plan):
    model.eval()
    train.set_checkpointing(model, False)
    total = 0.
    for group in train.microbatches(rows, common.BATCH):
        views = [view_for(r, plan, 0, 0, probe=True) for r in group]
        total += masked.loss(model, views, renderer.tokenizer.pad_token_id,
                             chunk_size=plan["head_chunk"], recompute=False).item()
    return {"n": len(rows), "mean_per_example_mntp_ce": total / len(rows),
            "scope": "fixed masks on training examples; fitting diagnostic, not unseen-language validation"}


def save_checkpoint(model, opt, renderer, dest, cfg, step, arm, seed):
    if dest.exists():
        raise ValueError("preserve existing checkpoint")
    opt.zero_grad()
    save = reverse.save if arm.startswith("reverse_") else modeling.save
    save(model, renderer.tokenizer, dest, cfg | {"experiment": "mntp-conversion-factorial-v1",
         "steps": step, "arm": arm, "seed": seed, "trained_from": str(search.BASE)})
    state = common.training_state(model, opt, step)
    torch.save(state, dest / "optimizer.pt.tmp")
    (dest / "optimizer.pt.tmp").rename(dest / "optimizer.pt")
    del state
    gc.collect()
    train.atomic_json(dest / "complete.json", {"step": step, "arm": arm, "seed": seed,
        "files": {f.name: train.sha256_file(f) for f in dest.iterdir() if f.is_file()}})


def smoke(out, arm):
    manifest = verify(out, frozen=False)
    plan = manifest["plan"]
    dest = out / f"smoke-{arm}"
    if dest.exists():
        raise ValueError("preserve smoke evidence")
    dest.mkdir()
    renderer, cfg = search.renderer_for(search.BASE)
    with (out / "encoded-seed41.jsonl").open() as f:
        prepared = [json.loads(next(f)) for _ in range(5)]
    probe = records.rows(REFERENCE / "selection.jsonl")[:12]
    train.seed_all(41)
    model = load(arm)
    opt = optimizer(model, plan)
    torch.cuda.reset_peak_memory_stats()
    before_masked = masked_probe(model, renderer, prepared[0]["rows"][:4], plan)
    first_loss, _, _ = update(model, opt, renderer, prepared[0]["rows"], arm, plan, 1, 41, 1)
    save_checkpoint(model, opt, renderer, dest / "step-1", cfg, 1, arm, 41)
    expected_loss, _, _ = update(model, opt, renderer, prepared[4]["rows"], arm, plan, 2, 41, 5)
    expected = common.master_digest(opt)
    _, expected_predictions = previous.evaluate(model, renderer, probe)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    model = load(arm, dest / "step-1")
    opt = optimizer(model, plan)
    if common.restore_optimizer(model, opt, dest / "step-1") != 1:
        raise ValueError("wrong restored step")
    actual_loss, _, _ = update(model, opt, renderer, prepared[4]["rows"], arm, plan, 2, 41, 5)
    _, actual_predictions = previous.evaluate(model, renderer, probe)
    if actual_loss != expected_loss or common.master_digest(opt) != expected or actual_predictions != expected_predictions:
        raise ValueError("next loss, FP32 master weights, or predictions did not resume exactly")
    # Exercise the true phase reset and a decision update after masked adaptation.
    del opt
    gc.collect()
    opt = optimizer(model, plan)
    train.seed_all(41001)
    transition_loss, _, _ = update(model, opt, renderer, prepared[0]["rows"], arm, plan,
                                  plan["pre_steps"] + 1, 41, 1)
    worst = None
    if arm.endswith("_mntp"):
        longest = []
        with (out / "encoded-seed41.jsonl").open() as f:
            for line in f:
                longest = sorted(longest + json.loads(line)["rows"],
                                 key=lambda r: len(r["input_ids"]), reverse=True)[:2]
        worst_loss, _, _ = update(model, opt, renderer, longest, arm, plan, 3, 41, 99999)
        worst = {"lengths": [len(r["input_ids"]) for r in longest], "loss": worst_loss}
    result = {"implementation": implementation(), "arm": arm, "exact_optimizer_resume": True,
              "first_loss": first_loss, "second_loss": actual_loss, "transition_ce": transition_loss,
              "initial_masked_train_probe": before_masked, "worst_length_training": worst,
              "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "gate_summary": reverse.gate_summary(model.model.language_model) if arm.startswith("reverse_") else None}
    train.atomic_json(dest / "result.json", result)
    print("SMOKE_PASS", arm, {k: result[k] for k in ["first_loss", "second_loss", "transition_ce",
          "exact_optimizer_resume", "peak_cuda_allocated_gb", "worst_length_training"]}, flush=True)


def freeze(out):
    manifest = verify(out, frozen=False)
    if manifest["frozen"]:
        raise ValueError("already frozen")
    for arm in ARMS:
        result = read(out / f"smoke-{arm}/result.json")
        if result["implementation"] != implementation() or not result["exact_optimizer_resume"]:
            raise ValueError("missing or stale real-model smoke")
    for objective in ["ce", "mntp"]:
        native = read(out / f"smoke-native_{objective}/result.json")
        altered = read(out / f"smoke-reverse_{objective}/result.json")
        if native["first_loss"] != altered["first_loss"]:
            raise ValueError("zero-gate reverse does not preserve the initial objective exactly")
    for seed in manifest["plan"]["seeds"]:
        if str(out / f"encoded-seed{seed}.jsonl") not in manifest["files"]:
            raise ValueError("missing validated input cache")
    manifest.update(frozen=True, implementation=implementation())
    (out / "source").mkdir()
    for filename in manifest["implementation"]:
        shutil.copyfile(filename, out / "source" / Path(filename).name)
    train.atomic_json(out / "manifest.json", manifest)
    print("FROZEN", flush=True)


def evaluation(model, renderer, out, dest, step, plan, *, selection):
    probe = records.rows(REFERENCE / "train-probe.jsonl")
    metrics, predictions = previous.evaluate(model, renderer, probe)
    train.atomic_json(dest / f"train-probe-{step}.json", metrics)
    search.save_records(dest / f"train-probe-{step}.jsonl", predictions)
    train.atomic_json(dest / f"masked-probe-{step}.json", masked_probe(
        model, renderer, records.rows(out / "masked-train-probe.jsonl"), plan))
    if selection:
        metrics, predictions = previous.evaluate(model, renderer, records.rows(REFERENCE / "selection.jsonl"))
        train.atomic_json(dest / f"selection-{step}.json", metrics)
        search.save_records(dest / f"selection-{step}.jsonl", predictions)
        print("SELECTION", dest.name, step, metrics["macro"], flush=True)


def fit(out, arm, seed):
    manifest = verify(out)
    plan = manifest["plan"]
    dest = out / f"{arm}-seed{seed}"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return
    renderer, cfg = search.renderer_for(search.BASE)
    checkpoints = sorted((p for p in dest.glob("step-*") if p.name[5:].isdigit()
                          and (p / "complete.json").exists()), key=lambda p: int(p.name[5:]))
    train.seed_all(seed)
    model = load(arm, checkpoints[-1] if checkpoints else search.BASE)
    opt = optimizer(model, plan)
    start = common.restore_optimizer(model, opt, checkpoints[-1]) if checkpoints else 0
    log_path = dest / "train.jsonl"
    trace = records.rows(log_path) if log_path.exists() else []
    if len(trace) < start:
        raise ValueError("checkpoint ahead of log")
    if len(trace) > start:
        shutil.copyfile(log_path, dest / f"uncommitted-{time.time_ns()}.jsonl")
        trace = trace[:start]
        search.save_records(log_path, trace)
    if not start:
        evaluation(model, renderer, out, dest, 0, plan, selection=False)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    schedule = [(s, s * 4 - 3) for s in range(1, plan["pre_steps"] + 1)]
    schedule += [(plan["pre_steps"] + s, s) for s in range(1, plan["steps"] + 1)]
    # Offsets allow both phases to read exactly the same frozen cache without retaining it all in RAM.
    offsets = []
    with (out / f"encoded-seed{seed}.jsonl").open("rb") as cache:
        while True:
            pos = cache.tell()
            if not cache.readline():
                break
            offsets.append(pos)
        if len(offsets) != 4096:
            raise ValueError("incomplete cache")
        with log_path.open("a") as log:
            for step, cache_step in schedule:
                if step <= start:
                    continue
                if step == plan["pre_steps"] + 1:
                    # A reload of the pre-phase checkpoint must take this reset too.
                    del opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    opt = optimizer(model, plan)
                    train.seed_all(seed * 1000 + 1)
                cache.seek(offsets[cache_step - 1])
                item = json.loads(cache.readline())
                if item["step"] != cache_step:
                    raise ValueError("cache offset mismatch")
                rows = item["rows"]
                t0 = time.monotonic()
                loss, norm, extra = update(model, opt, renderer, rows, arm, plan, step, seed, cache_step)
                entry = {"step": step, "phase": "pre" if step <= plan["pre_steps"] else "ce",
                         "cache_step": cache_step, "loss": loss, "grad_norm": norm,
                         "input_target_sha256": search.digest([[r["input_ids"], r["code_ids"], r["target"]] for r in rows]),
                         "mask_sha256": search.digest([view_for(r, plan, seed, cache_step) for r in rows])
                           if extra["masked_tokens"] else None,
                         "update_seconds": time.monotonic() - t0,
                         "seconds_this_session": time.monotonic() - started} | extra
                trace.append(entry)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                if step % 32 == 0 or step == 1:
                    window = [r for r in trace[-64:] if r["phase"] == entry["phase"]]
                    status = entry | {"arm": arm, "seed": seed,
                        "mean_loss_last64": sum(r["loss"] for r in window) / len(window),
                        "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
                    train.atomic_json(out / "status.json", status)
                    train.atomic_json(dest / "latest.json", status)
                    print("FIT", json.dumps(status), flush=True)
                if step in plan["probe_steps"]:
                    evaluation(model, renderer, out, dest, step, plan, selection=step in plan["selection_steps"])
                if step in plan["checkpoints"]:
                    checkpoint = dest / f"step-{step}"
                    if checkpoint.exists():
                        checkpoint.rename(dest / f"step-{step}.incomplete-{time.time_ns()}")
                    save_checkpoint(model, opt, renderer, checkpoint, cfg, step, arm, seed)
    if len(trace) != plan["total_steps"]:
        raise ValueError("incomplete fit")
    # Each final artifact is reloaded independently before its results enter the report.
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    common.verify_checkpoint(dest / "step-5120")
    model = load(arm, dest / "step-5120").requires_grad_(False)
    metrics, predictions = previous.evaluate(model, renderer, records.rows(REFERENCE / "selection.jsonl"))
    if predictions != records.rows(dest / "selection-5120.jsonl"):
        raise ValueError("reloaded full-selection predictions changed")
    train.atomic_json(dest / "result.json", {"steps": plan["total_steps"], "seed": seed, "arm": arm,
        "resume_start": start, "exact_full_selection_reload": True,
        "update_seconds": sum(r["update_seconds"] for r in trace), "final_selection": metrics,
        "gate_summary": reverse.gate_summary(model.model.language_model) if arm.startswith("reverse_") else None})
    del model
    gc.collect()
    torch.cuda.empty_cache()


def factorial_bootstrap(predictions, *, draws=5000):
    """Paired source-stratified document resampling, averaging scores across seeds.

    This never averages model predictions and does not estimate seed-population uncertainty.
    """
    seeds = sorted(predictions)
    first = predictions[seeds[0]][ARMS[0]]
    ids = [r["id"] for r in first]
    sources = sorted({r["source"] for r in first})
    by_source = {s: {} for s in sources}
    for i, row in enumerate(first):
        by_source[row["source"]].setdefault(row.get("group", row["id"]), []).append(i)
    correctness = []
    for seed in seeds:
        per_arm = []
        for arm in ARMS:
            rows = predictions[seed][arm]
            if [(r["id"], r["source"], r.get("group", r["id"])) for r in rows] != [
                    (r["id"], r["source"], r.get("group", r["id"])) for r in first]:
                raise ValueError("factorial predictions are not paired")
            per_arm.append([float(r["correct"]) for r in rows])
        correctness.append(per_arm)
    values = np.asarray(correctness).mean(axis=0)
    # Contrasts: MNTP effect native, MNTP effect reverse, interaction, final candidate vs native CE.
    contrast = np.stack([values[1] - values[0], values[3] - values[2],
                         values[3] - values[2] - values[1] + values[0], values[3] - values[0]], axis=1)
    point = np.mean([contrast[[i for group in by_source[s].values() for i in group]].mean(axis=0)
                     for s in sources], axis=0)
    rng = np.random.default_rng(20261021)
    boot = np.zeros((draws, 4))
    for source in sources:
        groups = list(by_source[source].values())
        sums = np.asarray([contrast[group].sum(axis=0) for group in groups])
        sizes = np.asarray([len(group) for group in groups])
        sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
        boot += sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)[:, None] / len(sources)
    return {name: {"macro_delta": float(point[i]), "paired_document_bootstrap_95ci":
                  np.quantile(boot[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(["native_mntp_effect", "reverse_mntp_effect", "interaction",
                                       "reverse_mntp_vs_native_ce"])}


def report(out):
    manifest = verify(out)
    plan = manifest["plan"]
    scores, predictions, checks = {}, {}, {}
    for seed in plan["seeds"]:
        paths = {arm: out / f"{arm}-seed{seed}" for arm in ARMS}
        if not all((p / "result.json").exists() for p in paths.values()):
            continue
        scores[seed] = {a: read(p / "result.json")["final_selection"] for a, p in paths.items()}
        predictions[seed] = {a: records.rows(p / "selection-5120.jsonl") for a, p in paths.items()}
        logs = {a: records.rows(p / "train.jsonl") for a, p in paths.items()}
        reference = logs[ARMS[0]]
        for arm, rows in logs.items():
            if len(rows) != 5120 or [(r["step"], r["cache_step"], r["input_target_sha256"]) for r in rows] != [
                    (r["step"], r["cache_step"], r["input_target_sha256"]) for r in reference]:
                raise ValueError("factorial inputs or budgets differ")
        if [r["mask_sha256"] for r in logs["native_mntp"]] != [r["mask_sha256"] for r in logs["reverse_mntp"]]:
            raise ValueError("architectures received different masks")
        checks[seed] = {"all5120_inputs_match": True, "identical_mntp_masks": True,
                        "all_final_checkpoints_reloaded_exactly": True}
    result = {"complete": len(scores) == len(plan["seeds"]), "completed_seeds": list(scores),
              "scores": scores, "comparisons": factorial_bootstrap(predictions) if predictions else None,
              "audit": checks, "primary": plan["primary"], "new_confirmation_cases_scored": 0,
              "limits": plan["limits"], "plan_sha256": manifest["plan_sha256"]}
    train.atomic_json(out / "report.json", result)
    train.atomic_json(Path("wiki/project/mntp-conversion-results.json"), result)
    lines = ["# MNTP 전환 학습 실험", "", f"상태: {'완료' if result['complete'] else '진행 중'}", "",
             "원본·역방향 구조 × 추가 CE·MNTP, seed 41/73. 각 조건은 같은 g4b에서 시작해 전환 학습 1,024회, 판단 CE 4,096회를 수행한다.",
             "두 단계 사이 Adam은 모든 조건에서 초기화한다. 같은 입력과 업데이트 수를 맞추며, 계산량은 다르다.", "",
             "| seed | 원본+CE | 원본+MNTP | 역방향+CE | 역방향+MNTP |", "|---|---:|---:|---:|---:|"]
    for seed, arms in scores.items():
        lines.append(f"| {seed} | " + " | ".join(f"{arms[a]['macro'] * 100:.3f}%" for a in ARMS) + " |")
    lines += ["", "주 비교는 (역방향 MNTP−역방향 CE)−(원본 MNTP−원본 CE)다. 문서 부트스트랩 구간은 훈련된 두 seed에 조건부이며 seed 모집단의 변동을 나타내지 않는다.",
              "기존에 반복 노출된 4,147개 선택 집합을 사용한다. 새로운 독립 홀드백의 개선을 주장하지 않는다.",
              "마스킹 대상은 학습 입력의 자연어 토큰이며 매번 실제로 숨긴다. 정답 위치 i의 원래 토큰은 i−1의 벡터로 복원한다. 교사 모델은 사용하지 않는다."]
    if result["comparisons"]:
        for name, comparison in result["comparisons"].items():
            lo, hi = comparison["paired_document_bootstrap_95ci"]
            lines.append(f"- {name}: {comparison['macro_delta'] * 100:+.3f}%p, 95% CI [{lo * 100:+.3f}, {hi * 100:+.3f}]%p")
    Path("wiki/project/mntp-conversion.md").write_text("\n".join(lines) + "\n")
    print("REPORT", result["complete"], list(scores), result["comparisons"], flush=True)


def run(out):
    manifest = verify(out)
    for seed in manifest["plan"]["seeds"]:
        for arm in ["reverse_mntp", "reverse_ce", "native_mntp", "native_ce"]:
            fit(out, arm, seed)
            report(out)
    train.atomic_json(out / "status.json", {"state": "complete", "completed_fits": 8})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "cache", "smoke", "freeze", "run", "report"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "smoke":
        if args.arm is None:
            parser.error("smoke requires --arm")
        smoke(args.output, args.arm)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
