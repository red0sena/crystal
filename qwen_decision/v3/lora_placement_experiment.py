"""LoRA placement from the untrained base model: where the adapters go, and how many parameters they get.

lora_experiment found attention-only LoRA (q/k/v/o of the 6 full-attention layers, rank 16, 1.08M parameters)
equal to full fine-tuning on short items and far better on long ones (+4.8 points accuracy, 5/5 seeds, d3
attention). That leaves two readings: the attention pathway is the right place (LoRA placement, arXiv 2604.22127),
or changing little is what keeps the base model's long-context ability. Five arms, all d3 attention, on the
training-method screen's data, seeds and views; each module count matches the paper's table at rank 16:
  gdn        the 5 Gated DeltaNet projections of the 18 linear-attention layers (90 modules, 4.43M; gdn_only)
  mlp        gate/up/down of all 24 MLPs (72 modules, 5.31M; mlp_only)
  attn-mlp   attention q/k/v/o and the MLPs (96 modules, 6.39M; softmax_plus_mlp)
  all        every projection above (186 modules, 10.82M; all_layers)
  attn-r160  attention q/k/v/o at rank 160 (10.81M): as many parameters as all, in the attention pathway only
alpha is twice the rank everywhere (the paper's 32 at 16); dropout .05, lr 2e-4, as lora_experiment. The base
model's own scores (causal and d3 attention, no training) are evaluated first, once, as a reference.
Baselines: lora_experiment's lora-noncausal (attn) and the screen's noncausal and frozen fits.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment check
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment freeze
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment run
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.lora_placement_experiment report
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

from qwen_decision.v3 import long_context_experiment as lce, lora_experiment as lora, modeling, readout_experiment as records
from qwen_decision.v3 import robust_adaptation as views, train, training_method_experiment as tme

OUT = Path("runs/from-base/lora-placement-v1")
SCREEN, LORA = tme.OUT, lora.OUT
ATTN = ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))
GDN = ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"))
MLP = ("mlp", ("gate_proj", "up_proj", "down_proj"))
ARMS = {"gdn": ((GDN,), 16), "mlp": ((MLP,), 16), "attn-mlp": ((ATTN, MLP), 16), "all": ((ATTN, GDN, MLP), 16),
        "attn-r160": ((ATTN,), 160)}
EXPECTED = {"gdn": (90, 4432896), "mlp": (72, 5308416), "attn-mlp": (96, 6389760), "all": (186, 10822656),
            "attn-r160": (24, 10813440)}
SEEDS = tme.SEEDS
TESTS = {"gdn-attn": ("attn", "gdn"), "mlp-attn": ("attn", "mlp"), "attn_mlp-attn": ("attn", "attn-mlp"),
         "all-attn": ("attn", "all"), "all-attn_r160": ("attn-r160", "all")}
REPORTED = {f"{a}-{b}": (b, a) for a in ARMS for b in ("noncausal", "frozen")} | {"attn_r160-attn": ("attn", "attn-r160")}
read = tme.read


class LoRALinear(torch.nn.Module):
    """lora_experiment.LoRALinear with the rank as an argument; alpha = 2 x rank, so the scale is 2 at every rank."""

    def __init__(self, base: torch.nn.Linear, rank: int):
        super().__init__()
        self.base, self.scale, self.dropout = base, 2., torch.nn.Dropout(lora.DROPOUT)
        w = base.weight
        a = torch.empty(rank, base.in_features, device=w.device)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a = torch.nn.Parameter(a.to(w.dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_a.T @ self.lora_b.T * self.scale


def load_model(arm: str):
    blocks, rank = ARMS[arm]
    model = tme.load_model("noncausal")
    model.requires_grad_(False)
    for layer in model.model.language_model.layers:
        for block, names in blocks:
            if hasattr(layer, block):
                module = getattr(layer, block)
                for name in names:
                    setattr(module, name, LoRALinear(getattr(module, name), rank))
    return model


def baseline_path(arm: str, seed: int) -> Path:
    return LORA / f"lora-noncausal-seed{seed}" if arm == "attn" else SCREEN / f"{arm}-seed{seed}"


def evaluate_base(out: Path) -> None:
    """The untrained model on every set, stock and d3 attention: no seed, no training."""
    renderer = lce.renderers()[0]["native"]
    for attention in ("causal", "noncausal"):
        dest = out / f"base-{attention}"
        if (dest / "result.json").exists():
            print("SKIP_COMPLETE base", attention, flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        model = tme.load_model(attention)
        summary = tme.evaluate_all(model, attention, renderer, dest, 0.)
        train.atomic_json(dest / "result.json", {"arm": f"base-{attention}", "eval": summary})
        print("RESULT", f"base-{attention}", json.dumps({k: round(v, 5) for k, v in summary["long"].items()}),
              json.dumps({m: round(summary["selection"][m], 5) for m in tme.METRICS}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def check(out: Path) -> None:
    """Before any fit: the baselines' code is unchanged, each arm trains exactly the paper's module and parameter
    count and nothing else, starts equal to the base model, and one update moves B but no base weight."""
    out.mkdir(parents=True, exist_ok=True)
    report = {"screen_implementation_unchanged": tme.implementation() == read(SCREEN / "plan.json")["implementation"],
              "lora_implementation_unchanged": lora.implementation() == read(LORA / "plan.json")["implementation"],
              "baselines_complete": read(SCREEN / "report.json")["complete"] and read(LORA / "report.json")["complete"],
              "eval_sets": {name: len(v) for name, v in tme.eval_sets().items()}}
    if not (report["screen_implementation_unchanged"] and report["lora_implementation_unchanged"] and report["baselines_complete"]):
        raise ValueError(f"baselines unusable: {report}")
    if report["eval_sets"] != read(SCREEN / "plan.json")["eval_sets"]:
        raise ValueError("evaluation sets differ from the screen")
    renderer = lce.renderers()[0]["native"]
    rows = [views.view(renderer, r) for r in records.rows(tme.POOL)[:8]]
    with torch.no_grad():
        model = tme.load_model("noncausal").eval()
        train.set_checkpointing(model, False)
        batch, _ = tme.fb.packed(rows)
        base_logits = modeling.packed_code_logits(model, batch)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for arm in ARMS:
        with torch.no_grad():
            model = load_model(arm).eval()
            train.set_checkpointing(model, False)
            init_diff = float((modeling.packed_code_logits(model, batch) - base_logits).abs().max())
            del model
            gc.collect()
            torch.cuda.empty_cache()
        model = load_model(arm)
        params = dict(model.named_parameters())
        adapters = lora.adapters(model)
        trainable = {n for n, p in params.items() if p.requires_grad}
        frozen = {n: p.detach().clone() for n, p in params.items() if n not in adapters}
        opt = lora.make_optimizer(model)
        masters = {id(p): m for p, m in opt.pairs}
        b_start = {n: masters[id(p)].clone() for n, p in adapters.items() if n.endswith("lora_b")}
        tme.update(model, opt, arm, rows, tme.WARMUP, 0.)
        report[arm] = {"init_max_logit_diff": init_diff, "modules": len(adapters) // 2,
                       "lora_parameters": sum(p.numel() for p in adapters.values()),
                       "trainable_equals_lora": trainable == set(adapters),
                       "lr": opt.param_groups[0]["lr"],  # the schedule is 1 at step WARMUP
                       "base_changed": any(bool((params[n].detach() != w).any()) for n, w in frozen.items()),
                       "b_master_changed": all(bool((masters[id(adapters[n])] != v).any()) for n, v in b_start.items())}
        print("CHECK", arm, json.dumps(report[arm]), flush=True)
        if (report[arm]["init_max_logit_diff"] != 0 or (report[arm]["modules"], report[arm]["lora_parameters"]) != EXPECTED[arm]
                or not report[arm]["trainable_equals_lora"] or report[arm]["base_changed"]
                or not report[arm]["b_master_changed"] or not math.isclose(report[arm]["lr"], lora.LORA_LR)):
            raise ValueError(f"{arm} wrong: {report[arm]}")
        del model, opt, frozen
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
        opt = lora.make_optimizer(model)
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
    report["matched_views"] = all(report[arm]["view_digests"] == reference["causal"]["view_digests"] for arm in ARMS)
    train.atomic_json(out / "smoke.json", report)
    if not report["matched_views"]:
        raise ValueError("views differ from the screen")


def implementation() -> dict:
    return lora.implementation() | {str(Path(__file__)): train.sha256_file(Path(__file__))}


def freeze(out: Path) -> dict:
    path = out / "plan.json"
    if path.exists():
        raise ValueError("plan already frozen")
    checked, smoked = read(out / "check.json"), read(out / "smoke.json")
    if not smoked["matched_views"]:
        raise ValueError("smoke did not match views")
    screen = read(SCREEN / "plan.json")
    plan = {"experiment": "from-base-lora-placement-v1", "base": screen["base"], "base_sha256": screen["base_sha256"],
            "pool": screen["pool"], "pool_sha256": train.sha256_file(tme.POOL),
            "screen_plan_sha256": train.sha256_file(SCREEN / "plan.json"), "lora_plan_sha256": train.sha256_file(LORA / "plan.json"),
            "arms": {arm: {"blocks": {b: list(n) for b, n in blocks}, "rank": rank, "alpha": 2 * rank,
                           "modules": EXPECTED[arm][0], "parameters": EXPECTED[arm][1]} for arm, (blocks, rank) in ARMS.items()},
            "attention": "d3 (noncausal) in every arm", "dropout": lora.DROPOUT, "lr": lora.LORA_LR,
            "baselines": {"attn": "lora_experiment lora-noncausal", "noncausal": "screen noncausal (full FT)",
                          "frozen": "screen frozen (full FT, DeltaNet mixers frozen)"},
            "base_reference": "untrained model, causal and d3 attention, evaluated once before the fits; reported only",
            "seeds": list(SEEDS), "order": "base reference, then seed-major", "steps": tme.STEPS,
            "batch_size": tme.BATCH_SIZE, "warmup": tme.WARMUP, "schedule": "linear warmup then cosine to 10%, as the screen",
            "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1, as the screen",
            "views": "lce.step_rows, checked per step against the screen's causal fit of the seed",
            "eval_sets": checked["eval_sets"], "primary_metric": screen["primary_metric"],
            "secondary_metrics": screen["secondary_metrics"],
            "tests": {k: {"baseline": a, "candidate": b, "metrics": ["selection NLL", "long NLL"]} for k, (a, b) in TESTS.items()},
            "decision": "candidate better on a metric if at least 4 of the 5 seed deltas favour it and the hierarchical "
                        "seed x item bootstrap 95% interval excludes 0 on its side; worse by the mirror rule. 10 tests: "
                        "each placement against attention-only LoRA, and all layers against attention-only LoRA of the "
                        "same parameter count (placement versus amount).",
            "reported_only": "secondary metrics, per set, base reference, " + ", ".join(REPORTED),
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
    opt = lora.make_optimizer(model)
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
    adapters = lora.adapters(model)
    masters = {id(p): m for p, m in opt.pairs}
    save_file({n: p.detach().contiguous() for n, p in adapters.items()}
              | {f"master.{n}": masters[id(p)].contiguous() for n, p in adapters.items()}, str(dest / "adapter.safetensors"))
    (dest / "adapter.json").write_text(json.dumps({"attention": "noncausal", "base": plan["base"], "arm": arm, **plan["arms"][arm],
                                                   "dropout": lora.DROPOUT, "lr": lora.LORA_LR,
                                                   "experiment": plan["experiment"], "seed": seed}, indent=1) + "\n")
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
             for arm in ARMS}
    evals |= {arm: {s: {name: records.rows(baseline_path(arm, s) / f"eval-{name}.jsonl") for name in names} for s in done}
              for arm in ("attn", "noncausal", "frozen")}
    result = {"plan_sha256": train.sha256_file(out / "plan.json"), "completed_seeds": done,
              "base_reference": {a: read(out / f"base-{a}/result.json")["eval"] for a in ("causal", "noncausal")},
              "scores": {arm: {s: read((out / f"{arm}-seed{s}" if arm in ARMS else baseline_path(arm, s)) / "result.json")["eval"]
                               for s in done} for arm in evals}}
    if done:
        groups = {"selection": ["selection"], "long": list(lce.LONG_SETS)}
        result["comparisons"] = {label: {f"{g} {m}": tme.hierarchical(evals[a], evals[b], sets, m, draws=10000 if m == "nll" else 2000)
                                         for g, sets in groups.items() for m in tme.METRICS}
                                 for label, (a, b) in (TESTS | REPORTED).items()}
        result["tests"] = {k: {g: result["comparisons"][k][f"{g} nll"] for g in groups} for k in TESTS}
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
        if implementation() != read(args.output / "plan.json")["implementation"]:
            raise ValueError("frozen implementation changed")
        evaluate_base(args.output)
        for seed in map(int, args.seeds.split(",")):
            for arm in ARMS:
                fit(args.output, arm, seed)
        report(args.output)


if __name__ == "__main__":
    main()
