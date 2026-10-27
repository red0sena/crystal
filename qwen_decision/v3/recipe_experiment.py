"""Training-recipe changes against the matched native 4,096-update fit.

Every recipe reads the long-training run's cached seed-41 prompts, targets, codes and batch
order, so only the optimisation differs from runs/long-training/64k-4096-v1/native-seed41
(selection macro 79.14%). Weight averages are kept in FP32 next to the AdamW masters and
exported as ordinary single checkpoints; nothing is ensembled at inference.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.recipe_experiment smoke --recipe ema
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.recipe_experiment run --recipe ema
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.recipe_experiment report
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.recipe_experiment confirm

`confirm` follows <output>/preregistration.json, written before any full result: it scores the one
variant that gained most on selection (at least +0.5%p) and native once on an untouched set.

Recipes:
  ema    native's own run, resumed from its exact step-2048 optimizer state, with an EMA
         (decay .999, started at 2048) and a uniform average of steps 3073-4096. Its raw
         weights must reproduce native step 4096, which checks that nothing else changed.
  lr1e5  body LR 1e-5 instead of 3e-6 from step 0, same warmup/cosine shape, same averages.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from qwen_decision.v3 import architecture_search as search, long_training as lt
from qwen_decision.v3 import modeling, readout_experiment as records, robust_experiment as statistics, train

SOURCE = lt.OUT
NATIVE = SOURCE / "native-seed41"
OUT = Path("runs/recipe/64k-4096-v1")
SEED = 41
RECIPES = {"ema": {"body_lr": 3e-6, "resume": "step-2048"},
           "lr1e5": {"body_lr": 1e-5, "resume": None}}
EMA_DECAY = .999
TAIL_START = 3073  # uniform average over the last quarter of the schedule
MIDPOINT = 2048


def read(path):
    return json.loads(Path(path).read_text())


def inputs() -> dict:
    """The long-training plan, after checking its cached inputs against its frozen manifest."""
    manifest = read(SOURCE / "manifest.json")
    for name in ("train.jsonl", "selection.jsonl", "train-probe.jsonl", "plan.json", f"encoded-seed{SEED}.jsonl"):
        if train.sha256_file(SOURCE / name) != manifest["files"][name]:
            raise ValueError(f"long-training input changed: {name}")
    return manifest["plan"]


class Averages:
    """FP32 EMA and uniform tail average of the optimizer's master weights."""

    def __init__(self, masters, decay=EMA_DECAY, tail_start=TAIL_START):
        self.masters, self.decay, self.tail_start = masters, decay, tail_start
        self.ema = [m.detach().clone() for m in masters]
        self.tail, self.count = None, 0

    @torch.no_grad()
    def update(self, step: int) -> None:
        torch._foreach_lerp_(self.ema, self.masters, 1 - self.decay)
        if step < self.tail_start:
            return
        self.count += 1
        if self.tail is None:
            self.tail = [m.detach().clone() for m in self.masters]
        else:  # running mean: avg_n = avg_{n-1} + (x_n - avg_{n-1}) / n
            torch._foreach_lerp_(self.tail, self.masters, 1 / self.count)


@torch.no_grad()
def use_weights(opt, weights) -> None:
    """Write FP32 weights (masters, EMA or tail) into the BF16 model."""
    for (parameter, _), weight in zip(opt.pairs, weights):
        parameter.copy_(weight)


def score(model, renderer, rows, dest: Path, name: str) -> float:
    metric, result = lt.evaluate(model, renderer, rows, "native")
    train.atomic_json(dest / f"{name}.json", metric)
    search.save_records(dest / f"{name}.jsonl", result)
    print("SCORE", name, round(metric["macro"], 6), flush=True)
    return metric["macro"]


def fit(recipe: str, out: Path, smoke_steps: int = 0) -> None:
    plan, spec = inputs(), RECIPES[recipe]
    p = {**plan, "body_lr": spec["body_lr"]}
    dest = out / (f"smoke-{recipe}" if smoke_steps else recipe)
    if (dest / "result.json").exists():
        print("SKIP_COMPLETE", recipe, flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    renderer, config = search.renderer_for(search.BASE)
    prepared = records.rows(SOURCE / f"encoded-seed{SEED}.jsonl")
    reference = records.rows(NATIVE / "train.jsonl")
    train.seed_all(SEED)
    if spec["resume"]:
        model = lt.load_arm("native", NATIVE / spec["resume"])
        opt = lt.optimizer(model, "native", p)
        start = lt.restore_optimizer(model, opt, NATIVE / spec["resume"])
    else:
        model, start = lt.load_arm("native"), 0
        opt = lt.optimizer(model, "native", p)
    averages = Averages([m for _, m in opt.pairs])
    stop = start + smoke_steps if smoke_steps else plan["steps"]
    same_recipe = spec["body_lr"] == plan["body_lr"]
    torch.cuda.reset_peak_memory_stats()
    started, mismatched_losses = time.monotonic(), 0
    with (dest / "train.jsonl").open("w") as log:
        for step in range(start + 1, stop + 1):
            rows = prepared[step - 1]["rows"]
            digest = search.digest([[r["input_ids"], r["code_ids"], r["target"]] for r in rows])
            if prepared[step - 1]["step"] != step or digest != reference[step - 1]["input_target_sha256"]:
                raise ValueError(f"step {step}: inputs differ from the native fit")
            loss, norm = lt.update(model, opt, renderer, rows, "native", p, step)
            averages.update(step)
            # With native's recipe every loss must equal native's; with another LR only step 1 can.
            if (same_recipe or step == 1) and loss != reference[step - 1]["loss"]:
                mismatched_losses += 1
            log.write(json.dumps({"step": step, "loss": loss, "grad_norm": norm, "native_loss": reference[step - 1]["loss"],
                                  "seconds": time.monotonic() - started}) + "\n")
            log.flush()
            if step % 64 == 0 or smoke_steps:
                print("FIT", recipe, step, round(loss, 5), round(norm, 3), flush=True)
                train.atomic_json(out / "status.json", {"recipe": recipe, "step": step, "steps": stop,
                                                        "mismatched_losses": mismatched_losses})
            if step == MIDPOINT and not smoke_steps:
                selection = records.rows(SOURCE / "selection.jsonl")
                score(model, renderer, selection, dest, f"selection-raw-{step}")
                use_weights(opt, averages.ema)
                score(model, renderer, selection, dest, f"selection-ema-{step}")
                use_weights(opt, [m for _, m in opt.pairs])
    peak = torch.cuda.max_memory_allocated() / 1e9
    if smoke_steps:
        selection = records.rows(SOURCE / "selection.jsonl")[:32]
        raw = score(model, renderer, selection, dest, "selection32-raw")
        use_weights(opt, averages.ema)
        ema = score(model, renderer, selection, dest, "selection32-ema")
        train.atomic_json(dest / "result.json", {"recipe": recipe, "steps": [start + 1, stop],
            "mismatched_losses": mismatched_losses, "peak_cuda_allocated_gb": peak,
            "selection32_raw": raw, "selection32_ema": ema})
        print("SMOKE", recipe, "mismatched_losses", mismatched_losses, "peak_gb", round(peak, 2), flush=True)
        return
    selection, probe = records.rows(SOURCE / "selection.jsonl"), records.rows(SOURCE / "train-probe.jsonl")
    summary = {}
    for name, weights in (("raw", [m for _, m in opt.pairs]), ("ema", averages.ema), ("tail", averages.tail)):
        use_weights(opt, weights)
        summary[name] = {"selection": score(model, renderer, selection, dest, f"selection-{name}"),
                         "probe": score(model, renderer, probe, dest, f"probe-{name}")}
        modeling.save(model, renderer.tokenizer, dest / f"model-{name}",
                      config | {"steps": stop, "trained_from": str(search.BASE), "experiment": "recipe-64k-4096-v1",
                                "recipe": recipe, "weights": name, "body_lr": p["body_lr"]})
    train.atomic_json(dest / "result.json", {"recipe": recipe, "body_lr": p["body_lr"], "resumed_from": start,
        "ema_decay": EMA_DECAY, "tail_steps": [TAIL_START, stop], "tail_count": averages.count,
        "mismatched_losses": mismatched_losses, "peak_cuda_allocated_gb": peak,
        "seconds": time.monotonic() - started, "scores": summary})
    print("RESULT", recipe, json.dumps(summary), flush=True)
    del model, opt, averages
    gc.collect()
    torch.cuda.empty_cache()


def report(out: Path) -> None:
    native = records.rows(NATIVE / "selection-4096.jsonl")
    comparisons = {}
    for recipe in RECIPES:
        for name in ("raw", "ema", "tail"):
            path = out / recipe / f"selection-{name}.jsonl"
            if path.exists():
                comparisons[f"{recipe}-{name}"] = statistics.paired_interval(native, records.rows(path), confidence=.95)
    result = {"baseline": str(NATIVE / "selection-4096.jsonl"), "baseline_macro": search.metrics(native)["macro"],
              "comparisons": comparisons,
              "scope": "repeatedly used selection set, one training seed: screening, not confirmation"}
    train.atomic_json(out / "report.json", result)
    for key, value in comparisons.items():
        print(f"{key:12s} {value['macro_delta'] * 100:+.3f}%p  95% {[round(x * 100, 3) for x in value['paired_document_bootstrap_95ci']]}"
              f"  wins {value['wins']} losses {value['losses']}", flush=True)


def confirm(out: Path) -> None:
    """The preregistered pick against native on the untouched confirmation set, once."""
    from types import SimpleNamespace

    from qwen_decision.v3 import fresh_eval

    rule = read(out / "preregistration.json")
    if (out / "confirm.json").exists() or (out / "confirm-started.json").exists():
        raise SystemExit("confirmation already accessed; no reselection")
    if train.sha256_file(Path(rule["confirmation_set"])) != rule["confirmation_sha256"]:
        raise ValueError("confirmation set changed")
    eligible = {k: v for k, v in read(out / "report.json")["comparisons"].items() if v["macro_delta"] >= .005}
    if not eligible:
        train.atomic_json(out / "confirm.json", {"picked": None, "reason": "no variant reached +0.5%p on selection"})
        print("NO_CANDIDATE", flush=True)
        return
    picked = max(eligible, key=lambda k: eligible[k]["macro_delta"])
    recipe, weights = picked.rsplit("-", 1)
    models = {"native": NATIVE / "step-4096", picked: out / recipe / f"model-{weights}"}
    train.atomic_json(out / "confirm-started.json", {"picked": picked, "models": {k: str(v) for k, v in models.items()}})
    renderer, _ = search.renderer_for(search.BASE)
    rows = records.rows(Path(rule["confirmation_set"]))
    predictions, fresh = {}, {}
    for name, path in models.items():
        model = lt.load_arm("native", path).requires_grad_(False).eval()
        metric, predictions[name] = lt.evaluate(model, renderer, rows, "native")
        search.save_records(out / f"confirm-{name}.jsonl", predictions[name])
        print("CONFIRM", name, round(metric["macro"], 6), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        fresh_eval.score(SimpleNamespace(model=str(path), attention=None, out=out / f"fresh-{name}.json"))
        fresh[name] = read(out / f"fresh-{name}.json")["macro_accuracy"]
    comparison = statistics.paired_interval(predictions["native"], predictions[picked], confidence=.95)
    result = {"picked": picked, "native_macro": search.metrics(predictions["native"])["macro"],
              "picked_macro": search.metrics(predictions[picked])["macro"], "comparison": comparison,
              "success": comparison["paired_document_bootstrap_95ci"][0] > 0, "fresh_v1_macro_accuracy": fresh}
    train.atomic_json(out / "confirm.json", result)
    print("CONFIRM_RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "run", "report", "confirm"])
    parser.add_argument("--recipe", choices=sorted(RECIPES))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage in ("report", "confirm"):
        (report if args.stage == "report" else confirm)(args.output)
        return
    for recipe in [args.recipe] if args.recipe else list(RECIPES):
        fit(recipe, args.output, smoke_steps=3 if args.stage == "smoke" else 0)


if __name__ == "__main__":
    main()
