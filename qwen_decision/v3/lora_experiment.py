"""Attention-only LoRA from the untrained base model, on the training-method screen's data, seeds and views.

LoRA placement in hybrid models (arXiv 2604.22127) fine-tuned Qwen3.5-0.8B with LoRA on the softmax attention
pathway only (softmax_only: 24 modules, 1.08M parameters at rank 16) and found it beat all-layer LoRA, while
LoRA on the Gated DeltaNet path was destructive (GSM8K .297 -> .398 attention only, .375 all layers, .148 DeltaNet
only; one seed). Their recipe: rank 16, alpha 32, dropout .05, lr 2e-4 cosine, effective batch 16. Here the same
adapters go on q/k/v/o of the 6 full-attention layers (16 x (in + out) per module gives exactly 1,081,344) and
everything else, the code readout's tied embedding included, stays at the base weights. Two arms:
  lora-causal     stock attention, as in the paper.
  lora-noncausal  d3 attention (the 6 full-attention layers bidirectional), as the screen's noncausal and frozen.
The paper's 1,024-token truncation, 3 epochs and 8-bit Adam are not copied: every fit reads the screen's 16,384
untruncated rows once in 1,024 updates of 16 with FP32-master AdamW, as training_method_experiment does, and is
checked step by step against the views of its causal fit. Its causal, noncausal and frozen evaluations are the
baselines; their evaluation code must hash as it did then.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment check
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment freeze
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment run
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_experiment report
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import save_file
import torch

from qwen_decision.v3 import long_context_experiment as lce, modeling, readout_experiment as records
from qwen_decision.v3 import robust_adaptation as views, train, training_method_experiment as tme

OUT = Path("runs/from-base/lora-v1")
SCREEN = tme.OUT
ARMS = ("lora-causal", "lora-noncausal")
SEEDS = tme.SEEDS
RANK, ALPHA, DROPOUT, LORA_LR = 16, 32, .05, 2e-4
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
COMPARISONS = {"lora_causal-causal": ("causal", "lora-causal"), "lora_noncausal-noncausal": ("noncausal", "lora-noncausal"),
               "lora_noncausal-frozen": ("frozen", "lora-noncausal")}
REPORTED = {"lora_noncausal-lora_causal": ("lora-causal", "lora-noncausal")}
read = tme.read


class LoRALinear(torch.nn.Module):
    """base(x) + alpha / rank * B(A(dropout(x))); B starts at zero, so the wrapped layer starts equal to the base."""

    def __init__(self, base: torch.nn.Linear):
        super().__init__()
        self.base, self.scale, self.dropout = base, ALPHA / RANK, torch.nn.Dropout(DROPOUT)
        w = base.weight
        a = torch.empty(RANK, base.in_features, device=w.device)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # peft's default for A
        self.lora_a = torch.nn.Parameter(a.to(w.dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, RANK, device=w.device, dtype=w.dtype))

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_a.T @ self.lora_b.T * self.scale


def load_model(arm: str):
    model = tme.load_model(arm.removeprefix("lora-"))
    model.requires_grad_(False)
    for layer in model.model.language_model.layers:
        if hasattr(layer, "self_attn"):
            for name in TARGETS:
                setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))
    return model


def adapters(model) -> dict:
    return {n: p for n, p in model.named_parameters() if ".lora_" in n}


def make_optimizer(model):
    opt = train.MasterAdamW(model, tme.BODY_LR)
    for g in opt.param_groups:  # tme.update sets lr = BODY_LR * lr_scale * schedule
        g["lr_scale"] = LORA_LR / tme.BODY_LR
    return opt


def baseline_evals(names, seeds) -> dict:
    return {arm: {s: {name: records.rows(SCREEN / f"{arm}-seed{s}/eval-{name}.jsonl") for name in names} for s in seeds}
            for arm in {a for a, _ in (COMPARISONS | REPORTED).values()} - set(ARMS)}


def check(out: Path) -> None:
    """Before any fit: the screen's evaluation code is unchanged, only the 24 adapters train, the adapted model
    starts equal to the base on both attentions, and one update moves B but no base weight."""
    out.mkdir(parents=True, exist_ok=True)
    screen = read(SCREEN / "plan.json")
    report = {"screen_implementation_unchanged": tme.implementation() == screen["implementation"],
              "screen_complete": read(SCREEN / "report.json")["complete"],
              "eval_sets": {name: len(v) for name, v in tme.eval_sets().items()}}
    if not report["screen_implementation_unchanged"] or not report["screen_complete"]:
        raise ValueError(f"screen baselines unusable: {report}")
    if report["eval_sets"] != screen["eval_sets"]:
        raise ValueError("evaluation sets differ from the screen")
    renderer = lce.renderers()[0]["native"]
    rows = [views.view(renderer, r) for r in records.rows(tme.POOL)[:8]]
    for arm in ARMS:
        with torch.no_grad():
            logits = {}
            for name, model in (("base", lambda: tme.load_model(arm.removeprefix("lora-"))), ("lora", lambda: load_model(arm))):
                model = model().eval()
                train.set_checkpointing(model, False)
                batch, _ = tme.fb.packed(rows)
                logits[name] = modeling.packed_code_logits(model, batch)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        model = load_model(arm)
        params = dict(model.named_parameters())
        lora = adapters(model)
        trainable = {n for n, p in params.items() if p.requires_grad}
        frozen = {n: p.detach().clone() for n, p in params.items() if n not in lora}
        opt = make_optimizer(model)
        masters = {id(p): m for p, m in opt.pairs}
        b_start = {n: masters[id(p)].clone() for n, p in lora.items() if n.endswith("lora_b")}
        tme.update(model, opt, arm, rows, tme.WARMUP, 0.)
        report[arm] = {"init_max_logit_diff": float((logits["lora"] - logits["base"]).abs().max()),
                       "modules": len(lora) // 2, "lora_parameters": sum(p.numel() for p in lora.values()),
                       "trainable_equals_lora": trainable == set(lora),
                       "lr": opt.param_groups[0]["lr"],  # the schedule is 1 at step WARMUP
                       "base_changed": any(bool((params[n].detach() != w).any()) for n, w in frozen.items()),
                       "b_master_changed": all(bool((masters[id(lora[n])] != v).any()) for n, v in b_start.items())}
        print("CHECK", arm, json.dumps(report[arm]), flush=True)
        if (report[arm]["init_max_logit_diff"] != 0 or report[arm]["modules"] != 24
                or report[arm]["lora_parameters"] != 1081344 or not report[arm]["trainable_equals_lora"]
                or report[arm]["base_changed"] or not report[arm]["b_master_changed"]
                or not math.isclose(report[arm]["lr"], LORA_LR)):
            raise ValueError(f"{arm} wrong: {report[arm]}")
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    train.atomic_json(out / "check.json", report)


def smoke(out: Path) -> None:
    """The screen's views for the first 12 steps of seed 131, update timing, and the pool's longest rows."""
    rows = records.rows(tme.POOL)
    renderer = lce.renderers()[0]["native"]
    reference = read(SCREEN / "smoke.json")
    schedule = list(train.batches(len(rows), tme.BATCH_SIZE, 12, random.Random(SEEDS[0])))
    lengths = [len(views.view(renderer, r)["input_ids"]) for r in rows]
    longest = sorted(range(len(rows)), key=lengths.__getitem__)[-3:]
    report = {}
    for arm in ARMS:
        train.seed_all(SEEDS[0])
        model = load_model(arm)
        opt = make_optimizer(model)
        digests, times = [], []
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = lce.step_rows(renderer, rows, indices, SEEDS[0], step)
            digests.append(lce.view_digest(encoded))
            tick = time.monotonic()
            loss, _, _ = tme.update(model, opt, arm, encoded, step, 0.)
            torch.cuda.synchronize()
            times.append({"seconds": time.monotonic() - tick, "tokens": sum(len(r["input_ids"]) for r in encoded), "loss": loss})
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        tme.update(model, opt, arm, [views.view(renderer, rows[i]) for i in longest], 13, 0.)
        torch.cuda.synchronize()
        median = float(np.median([t["seconds"] for t in times[3:]]))
        report[arm] = {"view_digests": digests, "updates": times, "median_update_s": median,
                       "estimated_fit_minutes": median * tme.STEPS / 60,
                       "long_update": {"tokens": [lengths[i] for i in longest], "seconds": time.monotonic() - tick,
                                       "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}}
        print("SMOKE", arm, json.dumps({k: v for k, v in report[arm].items() if k not in ("view_digests", "updates")}), flush=True)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    # the screen measured its causal fit at 1.6x the smoke estimate; that ratio is kept for planning
    report["screen_causal_estimate_vs_actual"] = [reference["causal"]["estimated_fit_minutes"],
                                                  read(SCREEN / f"causal-seed{SEEDS[0]}/result.json")["train_seconds"] / 60]
    report["matched_views"] = all(report[arm]["view_digests"] == reference["causal"]["view_digests"] for arm in ARMS)
    train.atomic_json(out / "smoke.json", report)
    if not report["matched_views"]:
        raise ValueError("views differ from the screen")


def implementation() -> dict:
    return tme.implementation() | {str(Path(__file__)): train.sha256_file(Path(__file__))}


def freeze(out: Path) -> dict:
    path = out / "plan.json"
    if path.exists():
        raise ValueError("plan already frozen")
    checked, smoked = read(out / "check.json"), read(out / "smoke.json")
    if not smoked["matched_views"]:
        raise ValueError("smoke did not match views")
    screen = read(SCREEN / "plan.json")
    plan = {"experiment": "from-base-lora-v1", "base": screen["base"], "base_sha256": screen["base_sha256"],
            "pool": screen["pool"], "pool_sha256": train.sha256_file(tme.POOL),
            "screen": str(SCREEN), "screen_plan_sha256": train.sha256_file(SCREEN / "plan.json"),
            "screen_report_sha256": train.sha256_file(SCREEN / "report.json"),
            "arms": {"lora-causal": "stock attention, LoRA on q/k/v/o of the 6 full-attention layers, all else frozen",
                     "lora-noncausal": "d3 attention, LoRA on q/k/v/o of the 6 full-attention layers, all else frozen"},
            "baselines": "the screen's causal, noncausal and frozen fits of the same seeds (full fine-tuning, lr 1e-5)",
            "lora": {"rank": RANK, "alpha": ALPHA, "dropout": DROPOUT, "targets": list(TARGETS), "modules": 24,
                     "parameters": checked[ARMS[0]]["lora_parameters"], "init": "A kaiming uniform (a=sqrt 5), B zeros",
                     "lr": LORA_LR, "source": "arXiv 2604.22127 softmax_only"},
            "seeds": list(SEEDS), "order": "seed-major", "steps": tme.STEPS, "batch_size": tme.BATCH_SIZE,
            "warmup": tme.WARMUP, "schedule": "linear warmup then cosine to 10%, as the screen",
            "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1, as the screen",
            "views": "lce.step_rows, checked per step against the screen's causal fit of the seed",
            "eval_sets": checked["eval_sets"], "primary_metric": screen["primary_metric"],
            "secondary_metrics": screen["secondary_metrics"],
            "tests": {k: {"baseline": a, "candidate": b, "metrics": ["selection NLL", "long NLL"]} for k, (a, b) in COMPARISONS.items()},
            "decision": "candidate better on a metric if at least 4 of the 5 seed deltas favour it and the hierarchical "
                        "seed x item bootstrap 95% interval excludes 0 on its side; worse by the mirror rule. 6 tests. "
                        "A LoRA arm better than its full fine-tuning baseline advances to a 65,536-row confirmation.",
            "reported_only": "secondary metrics, per set, " + ", ".join(REPORTED),
            "implementation": implementation(), "check": checked,
            "smoke": {arm: {k: smoked[arm][k] for k in ("median_update_s", "estimated_fit_minutes", "long_update")} for arm in ARMS}}
    train.atomic_json(path, plan)
    return plan


def fit(out: Path, arm: str, seed: int) -> None:
    plan = read(out / "plan.json")
    if implementation() != plan["implementation"]:
        raise ValueError("frozen implementation changed")
    dest = out / f"{arm}-seed{seed}"
    if (dest / "result.json").exists():
        print("SKIP_COMPLETE", arm, seed, flush=True)
        return
    if dest.exists():
        dest.rename(out / f"{dest.name}.incomplete-{time.time_ns()}")
    dest.mkdir()
    rows = records.rows(tme.POOL)
    if train.sha256_file(tme.POOL) != plan["pool_sha256"]:
        raise ValueError("training pool changed")
    renderer = lce.renderers()[0]["native"]
    reference = records.rows(SCREEN / f"causal-seed{seed}/train.jsonl")
    train.seed_all(seed)
    model = load_model(arm)
    opt = make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (dest / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(train.batches(len(rows), tme.BATCH_SIZE, tme.STEPS, random.Random(seed)), 1):
            encoded = lce.step_rows(renderer, rows, indices, seed, step)
            digest = lce.view_digest(encoded)
            if digest != reference[step - 1]["view_sha256"]:
                raise ValueError(f"step {step}: rows or views differ from the screen's causal fit")
            loss, _, norm = tme.update(model, opt, arm, encoded, step, 0.)
            entry = {"step": step, "loss": loss, "grad_norm": norm, "view_sha256": digest,
                     "tokens": sum(len(r["input_ids"]) for r in encoded), "seconds": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0:
                print("FIT", arm, seed, step, round(loss, 4), round(entry["seconds"]), flush=True)
                train.atomic_json(out / "status.json", {"state": "training", "arm": arm, "seed": seed, "step": step,
                                                        "steps": tme.STEPS, "seconds": entry["seconds"]})
    trained_seconds = time.monotonic() - started
    # the adapters only: the base weights are models/qwen3.5-0.8b unchanged
    lora = adapters(model)
    masters = {id(p): m for p, m in opt.pairs}
    save_file({n: p.detach().contiguous() for n, p in lora.items()}
              | {f"master.{n}": masters[id(p)].contiguous() for n, p in lora.items()}, str(dest / "adapter.safetensors"))
    (dest / "adapter.json").write_text(json.dumps({"attention": arm.removeprefix("lora-"), "base": plan["base"],
                                                   **plan["lora"], "experiment": plan["experiment"], "seed": seed}, indent=1) + "\n")
    peak = torch.cuda.max_memory_allocated() / 1e9
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    train.atomic_json(out / "status.json", {"state": "evaluating", "arm": arm, "seed": seed})
    summary = tme.evaluate_all(model, arm, renderer, dest, 0.)
    train.atomic_json(dest / "result.json", {"arm": arm, "seed": seed, "train_seconds": trained_seconds,
                                             "peak_cuda_gb": peak, "eval": summary})
    print("RESULT", arm, seed, json.dumps({k: round(v, 5) for k, v in summary["long"].items()}),
          json.dumps({m: round(summary["selection"][m], 5) for m in tme.METRICS}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report(out: Path) -> None:
    plan = read(out / "plan.json")
    names = list(tme.eval_sets())
    done = [s for s in plan["seeds"] if all((out / f"{arm}-seed{s}/result.json").exists() for arm in ARMS)]
    evals = {arm: {s: {name: records.rows(out / f"{arm}-seed{s}/eval-{name}.jsonl") for name in names} for s in done}
             for arm in ARMS} | baseline_evals(names, done)
    result = {"plan_sha256": train.sha256_file(out / "plan.json"), "completed_seeds": done,
              "scores": {arm: {s: read(out / f"{arm}-seed{s}/result.json")["eval"] for s in done} for arm in ARMS}}
    if done:
        groups = {"selection": ["selection"], "long": list(lce.LONG_SETS)}
        result["comparisons"] = {label: {f"{g} {m}": tme.hierarchical(evals[a], evals[b], sets, m, draws=10000 if m == "nll" else 2000)
                                         for g, sets in groups.items() for m in tme.METRICS}
                                 for label, (a, b) in (COMPARISONS | REPORTED).items()}
        result["tests"] = {k: {g: result["comparisons"][k][f"{g} nll"] for g in groups} for k in COMPARISONS}
        result["complete"] = len(done) == len(plan["seeds"])
        for label, v in result["tests"].items():
            print("TEST", label, json.dumps({g: {k: v[g][k] for k in ("mean", "ci95", "better", "worse")} for g in v}), flush=True)
    train.atomic_json(out / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "smoke", "freeze", "run", "report"])
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        check(args.output)
    elif args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "report":
        report(args.output)
    else:
        for seed in map(int, args.seeds.split(",")):
            for arm in ARMS:
                fit(args.output, arm, seed)
        report(args.output)


if __name__ == "__main__":
    main()
