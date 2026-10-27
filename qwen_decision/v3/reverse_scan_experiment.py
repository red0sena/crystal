"""Matched 4,096-update reverse-DeltaNet trial with an immutable native control.

No new labels, model ensembles, accuracy early stopping or fresh holdback use.
The previous native fit supplies the exact input schedule and comparison.
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

import torch

from qwen_decision.v3 import long_training as common, reverse_scan as architecture
from qwen_decision.v3 import modeling, train, architecture_search as search
from qwen_decision.v3 import readout_experiment as records, robust_experiment as statistics

OUT = Path("runs/reverse-scan/64k-4096-v1")
REFERENCE = common.OUT


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    return common.implementation() | {str(Path(m.__file__)): train.sha256_file(Path(m.__file__))
                                      for m in [architecture, __import__(__name__, fromlist=["_"])]}


def prepare(out):
    if out.exists():
        raise ValueError("preserve previous experiment")
    original = common.verify(REFERENCE)
    if not read(REFERENCE / "worker-result.json")["returncode"] == 0:
        raise ValueError("native reference is not complete")
    native = REFERENCE / "native-seed41"
    files = [REFERENCE / f for f in ["manifest.json", "plan.json", "train.jsonl", "selection.jsonl",
                                    "train-probe.jsonl", "encoded-seed41.jsonl"]]
    files += [native / f for f in ["train.jsonl", "result.json", "selection-0.jsonl"]]
    files += [native / f"{kind}-{s}.{extension}" for s in [1024, 2048, 4096]
              for kind in ["selection", "train-probe"] for extension in ["json", "jsonl"]]
    plan = original["plan"] | {
        "experiment": "reverse-deltanet-64k-v1", "arms": ["native_reused", "reverse"],
        "seeds": [41], "gate_lr": .001, "new_gate_parameters": 18432,
        "snapshots": [1024, 2048, 4096], "probe_steps": [0, 64, 256, 512, 1024, 2048, 4096],
        "architecture": "all18 DeltaNets share forward/reverse weights; valid-row reversal; F+tanh(g)*(R-F)",
        "initialization": "zero FP32 channel gates; initial code probabilities must equal native exactly",
        "native_control": str(native), "native_reuse": "same frozen source, base, input/target/code hashes, seed and full4096 LR curve",
        "primary": "fixed4096 source macro accuracy versus native4096; one predeclared contrast",
        "secondary": "matched1024/2048 accuracy; paired online CE windows; fixed512 seen-training probe CE; reverse-off intervention",
        "confirmation": "no new holdback scored automatically; selection is repeatedly exposed",
        "limits": ["one seed, exploratory exposed selection", "shared pretrained weights on reversed text may need adaptation",
                   "same data/updates, extra DeltaNet compute; no FLOP match", "unchanged1536 training token limit",
                   "zero gates require learning the new route; no guarantee that all18 reverse scans help",
                   "this tests gold CE adaptation only, not a masked-prediction objective"],
    }
    out.mkdir(parents=True)
    train.atomic_json(out / "plan.json", plan)
    train.atomic_json(out / "manifest.json", {
        "frozen": False, "plan": plan, "base_sha256": original["base_sha256"],
        "files": {str(f): train.sha256_file(f) for f in files},
        "plan_sha256": train.sha256_file(out / "plan.json"),
        "inherited_audit": original["inherited_train_selection_audit"], "new_confirmation_cases_scored": 0})
    print("PREPARED", out, flush=True)


def verify(out, frozen=True):
    m = read(out / "manifest.json")
    if frozen and not m["frozen"]:
        raise ValueError("smoke then freeze before fitting")
    if m["frozen"] and m["implementation"] != implementation():
        raise ValueError("frozen source changed")
    if train.sha256_file(out / "plan.json") != m["plan_sha256"]:
        raise ValueError("plan changed")
    if train.sha256_file(search.BASE / "model.safetensors") != m["base_sha256"]:
        raise ValueError("base checkpoint changed")
    for filename, digest in m["files"].items():
        if train.sha256_file(Path(filename)) != digest:
            raise ValueError(f"reference input or control changed: {filename}")
    return m


def load(path=search.BASE):
    torch.use_deterministic_algorithms(True)
    model = architecture.load(path)
    model.requires_grad_(False)
    model.model.language_model.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def optimizer(model, plan):
    return train.MasterAdamW(model, plan["body_lr"], gate_lr=plan["gate_lr"])


def save_checkpoint(model, opt, renderer, dest, cfg, step):
    if dest.exists():
        raise ValueError("preserve existing checkpoint")
    opt.zero_grad()
    architecture.save(model, renderer.tokenizer, dest,
                      cfg | {"steps": step, "experiment": "reverse-deltanet-64k-v1", "trained_from": str(search.BASE)})
    state = common.training_state(model, opt, step)
    torch.save(state, dest / "optimizer.pt.tmp")
    (dest / "optimizer.pt.tmp").rename(dest / "optimizer.pt")
    del state
    gc.collect()
    train.atomic_json(dest / "complete.json", {
        "step": step, "arm": "reverse", "files": {f.name: train.sha256_file(f)
        for f in dest.iterdir() if f.name != "complete.json"}})


def add_ce(metrics, predictions, rows):
    labels = {r["id"]: r["target"] for r in rows}
    losses = [-sum(t * math.log(max(p, 1e-30)) for t, p in zip(labels[r["id"]], r["probabilities"]))
              for r in predictions]
    return metrics | {"mean_gold_ce_from_probabilities": sum(losses) / len(losses),
                      "ce_probability_floor": 1e-30}


def evaluate(model, renderer, rows):
    metrics, predictions = common.evaluate(model, renderer, rows, "native")
    return add_ce(metrics, predictions, rows), predictions


def intervention(model, renderer, rows):
    active, a = evaluate(model, renderer, rows)
    with architecture.disabled(model.model.language_model):
        off, b = evaluate(model, renderer, rows)
    return {"n": len(rows), "active": active, "off": off,
            "prediction_changes": sum(x["prediction"] != y["prediction"] for x, y in zip(a, b)),
            "max_probability_delta": max(abs(x - y) for arow, brow in zip(a, b)
                                          for x, y in zip(arow["probabilities"], brow["probabilities"]))}


def smoke(out):
    manifest = verify(out, frozen=False)
    plan = manifest["plan"]
    dest = out / "smoke"
    if dest.exists():
        raise ValueError("preserve smoke evidence")
    dest.mkdir()
    renderer, cfg = search.renderer_for(search.BASE)
    with (REFERENCE / "encoded-seed41.jsonl").open() as f:
        prepared = [json.loads(next(f)) for _ in range(4)]
    rows = records.rows(REFERENCE / "selection.jsonl")[:32]
    train.seed_all(41)
    model = load()
    opt = optimizer(model, plan)
    _, initial = evaluate(model, renderer, rows)
    # A subset sorts/pads differently from the original full selection. Compare
    # the stock route with precisely the same rows/batches, not subset-vs-full.
    with architecture.disabled(model.model.language_model):
        _, native_zero = evaluate(model, renderer, rows)
    initial_delta = max(abs(a - b) for r, expected_row in zip(initial, native_zero)
                        for a, b in zip(r["probabilities"], expected_row["probabilities"]))
    if initial_delta:
        raise ValueError("zero gate did not preserve native initialization")
    losses = []
    gate_gradient = None
    for step, item in enumerate(prepared[:3], 1):
        losses.append(common.update(model, opt, renderer, item["rows"], "native", plan, step)[0])
        if step == 1:
            gate_gradient = sum(float(m.grad.abs().sum()) for p, m in opt.pairs
                                if p is model.model.language_model.layers[0].linear_attn.backward_gate)
    expected_first = records.rows(REFERENCE / "native-seed41/train.jsonl")[0]["loss"]
    if losses[0] != expected_first or not gate_gradient:
        raise ValueError("first CE or reverse gate gradient differs from the declared contract")
    save_checkpoint(model, opt, renderer, dest / "step-3", cfg, 3)
    expected_loss, _ = common.update(model, opt, renderer, prepared[3]["rows"], "native", plan, 4)
    expected = common.master_digest(opt)
    _, expected_predictions = evaluate(model, renderer, rows[:12])
    effect = intervention(model, renderer, rows[:16])
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    model = load(dest / "step-3")
    opt = optimizer(model, plan)
    assert common.restore_optimizer(model, opt, dest / "step-3") == 3
    actual_loss, _ = common.update(model, opt, renderer, prepared[3]["rows"], "native", plan, 4)
    _, predictions = evaluate(model, renderer, rows[:12])
    delta = max(abs(x - y) for a, b in zip(expected_predictions, predictions)
                for x, y in zip(a["probabilities"], b["probabilities"]))
    if actual_loss != expected_loss or common.master_digest(opt) != expected or delta:
        raise ValueError("optimizer resume did not reproduce the next exact update")
    result = {"implementation": implementation(), "initial_native_max_probability_delta": initial_delta,
              "first_native_loss_exact": True, "first_reverse_gate_master_gradient_l1": gate_gradient,
              "initial_losses": losses, "exact_optimizer_resume": True,
              "reload_max_probability_delta": delta, "reverse_off_after4": effect,
              "gate_summary": architecture.gate_summary(model.model.language_model),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    train.atomic_json(dest / "result.json", result)
    print("SMOKE_PASS", {k: v for k, v in result.items() if k not in ["implementation", "gate_summary", "reverse_off_after4"]}, flush=True)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def freeze(out):
    m = verify(out, frozen=False)
    result = read(out / "smoke/result.json")
    if result["implementation"] != implementation() or not result["exact_optimizer_resume"]:
        raise ValueError("smoke is stale or missing")
    m.update(frozen=True, implementation=implementation())
    (out / "source").mkdir()
    for filename in m["implementation"]:
        shutil.copyfile(filename, out / "source" / Path(filename).name)
    train.atomic_json(out / "manifest.json", m)
    print("FROZEN", flush=True)


def fit(out):
    manifest = verify(out)
    plan = manifest["plan"]
    dest = out / "reverse-seed41"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return
    renderer, cfg = search.renderer_for(search.BASE)
    selection = records.rows(REFERENCE / "selection.jsonl")
    probe = records.rows(REFERENCE / "train-probe.jsonl")
    reference = records.rows(REFERENCE / "native-seed41/train.jsonl")
    checkpoints = sorted((p for p in dest.glob("step-*") if p.name[5:].isdigit()
                          and (p / "complete.json").exists()), key=lambda p: int(p.name[5:]))
    train.seed_all(41)
    model = load(checkpoints[-1] if checkpoints else search.BASE)
    opt = optimizer(model, plan)
    start = common.restore_optimizer(model, opt, checkpoints[-1]) if checkpoints else 0
    log_path = dest / "train.jsonl"
    trace = records.rows(log_path) if log_path.exists() else []
    if len(trace) < start:
        raise ValueError("checkpoint is ahead of training trace")
    if len(trace) > start:
        shutil.copyfile(log_path, dest / f"uncommitted-{time.time_ns()}.jsonl")
        search.save_records(log_path, trace[:start])
        trace = trace[:start]
    schedule = list(train.batches(65536, 16, 4096, random.Random(41)))
    if not (dest / "selection-0.json").exists():
        metrics, predictions = evaluate(model, renderer, selection)
        expected = records.rows(REFERENCE / "native-seed41/selection-0.jsonl")
        if predictions != expected:
            raise ValueError("initial full selection differs from native")
        train.atomic_json(dest / "selection-0.json", metrics)
        search.save_records(dest / "selection-0.jsonl", predictions)
    if not (dest / "train-probe-0.json").exists():
        metrics, predictions = evaluate(model, renderer, probe)
        train.atomic_json(dest / "train-probe-0.json", metrics)
        search.save_records(dest / "train-probe-0.jsonl", predictions)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with log_path.open("a") as log, (REFERENCE / "encoded-seed41.jsonl").open() as cache:
        for step, line in enumerate(cache, 1):
            if step <= start:
                continue
            item = json.loads(line)
            if item["step"] != step or item["indices"] != schedule[step - 1][0]:
                raise ValueError("cached sample draw differs")
            digest = search.digest([[r["input_ids"], r["code_ids"], r["target"]] for r in item["rows"]])
            if digest != reference[step - 1]["input_target_sha256"]:
                raise ValueError("native control received different input/target/code")
            loss, norm = common.update(model, opt, renderer, item["rows"], "native", plan, step)
            entry = {"step": step, "loss": loss, "grad_norm": norm, "input_target_sha256": digest,
                     "seconds_this_session": time.monotonic() - started, "examples_seen": step * 16}
            trace.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 64 == 0:
                window = trace[-64:]
                entry = entry | {"reverse_mean_ce_last64": sum(r["loss"] for r in window) / len(window),
                                 "native_mean_ce_same64": sum(r["loss"] for r in reference[step - 64:step]) / 64}
                print("FIT", entry, flush=True)
                train.atomic_json(dest / "latest.json", entry | {
                    "gate_summary": architecture.gate_summary(model.model.language_model)})
            if step in plan["probe_steps"]:
                metrics, predictions = evaluate(model, renderer, probe)
                train.atomic_json(dest / f"train-probe-{step}.json", metrics)
                search.save_records(dest / f"train-probe-{step}.jsonl", predictions)
                train.atomic_json(dest / f"reverse-off-{step}.json", intervention(model, renderer, probe[:128]))
                print("TRAIN_PROBE", step, metrics["macro"], metrics["mean_gold_ce_from_probabilities"], flush=True)
            if step in plan["snapshots"]:
                metrics, predictions = evaluate(model, renderer, selection)
                train.atomic_json(dest / f"selection-{step}.json", metrics)
                search.save_records(dest / f"selection-{step}.jsonl", predictions)
                checkpoint = dest / f"step-{step}"
                if checkpoint.exists():
                    checkpoint.rename(dest / f"step-{step}.incomplete-{time.time_ns()}")
                save_checkpoint(model, opt, renderer, checkpoint, cfg, step)
                print("SELECTION", step, metrics["macro"], flush=True)
    if len(trace) != plan["steps"]:
        raise ValueError("incomplete cached schedule")
    train.atomic_json(dest / "result.json", {"steps": 4096, "resume_start_step": start,
        "all4096_inputs_match_native": True, "seconds_this_session": time.monotonic() - started,
        "peak_cuda_allocated_gb_this_session": torch.cuda.max_memory_allocated() / 1e9,
        "gate_summary": architecture.gate_summary(model.model.language_model)})
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def audit(out):
    manifest = verify(out)
    dest = out / "reverse-seed41"
    reference = REFERENCE / "native-seed41"
    selection = records.rows(REFERENCE / "selection.jsonl")
    probe = records.rows(REFERENCE / "train-probe.jsonl")
    renderer, _ = search.renderer_for(search.BASE)
    common.verify_checkpoint(dest / "step-4096")
    model = load(dest / "step-4096").requires_grad_(False).eval()
    metrics, candidate = evaluate(model, renderer, selection)
    expected = records.rows(dest / "selection-4096.jsonl")
    if candidate != expected:
        raise ValueError("full checkpoint reload changed predictions or probabilities")
    native = records.rows(reference / "selection-4096.jsonl")
    contrast = statistics.paired_interval(native, candidate, confidence=.95)
    contrast["paired_document_bootstrap_95ci_primary"] = contrast.pop("paired_document_bootstrap_97_5ci")
    contrast["scope"] = "one predeclared contrast, exposed selection; conditional on trained models, not seed variation or independent confirmation"
    contrast["exploratory_interval_above_zero"] = contrast.pop("confirmed_improvement")
    trajectory = {}
    for step in [1024, 2048, 4096]:
        n = read(reference / f"selection-{step}.json")
        c = read(dest / f"selection-{step}.json")
        np = records.rows(reference / f"train-probe-{step}.jsonl")
        cp = records.rows(dest / f"train-probe-{step}.jsonl")
        trajectory[str(step)] = {"native_macro": n["macro"], "reverse_macro": c["macro"],
            "macro_delta": c["macro"] - n["macro"],
            "native_train_probe": add_ce(read(reference / f"train-probe-{step}.json"), np, probe),
            "reverse_train_probe": add_ce(read(dest / f"train-probe-{step}.json"), cp, probe)}
    effect = intervention(model, renderer, selection[:512])
    result = {"manifest": manifest, "fit": read(dest / "result.json"),
              "native": read(reference / "selection-4096.json"), "reverse": metrics,
              "trajectory": trajectory, "comparison": contrast, "exact_full_selection_reload": True,
              "reverse_off_selection512": effect, "new_confirmation_cases_scored": 0,
              "candidate_for_separate_fresh_confirmation": contrast["macro_delta"] >= .005}
    train.atomic_json(out / "report.json", result)
    train.atomic_json(Path("wiki/project/reverse-scan-results.json"), result)
    print("FINAL_COMPARISON", contrast, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "smoke", "freeze", "run", "audit"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "run":
        fit(args.output)
        audit(args.output)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
