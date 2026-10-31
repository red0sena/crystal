"""Loss screen on the best LoRA recipe: all-layer LoRA r16 with d3 attention, six training losses, five seeds.

lora_placement_experiment's `all` arm (LoRA rank 16 on every attention, Gated DeltaNet and MLP projection, 10.82M
parameters) beat the screen's frozen recipe on short and long items. Every arm here is that model on the same
16,384 rows, seeds and matched views; only the loss changes. The `all` fits are the baseline. The arms:
  ls        CE against the gold smoothed over the shown codes, eps .1 (Szegedy et al. 2016)
  focal     focal loss, gamma 2 (Lin et al. 2017; calibration: Mukhoti et al. 2020)
  brier     CE + 1 x Brier over the shown codes, the screen's brier recipe on LoRA
  twoview   the matched view and a second random view of each row, the mean CE of the two (the control of consist)
  consist   twoview + 1 x Jensen-Shannon between the two views' option distributions in semantic order. Rows whose
            options keep their order (ordered, positional, noul) get the same view twice, which differ by dropout.
            robust_experiment tried lambda .5 on g4b (256 updates, two seeds) with no confirmed change.
  mntp      CE + .5 x masked next-token prediction on the matched view: 20% of the natural-language tokens replaced
            by "_", h[i-1] predicts x[i] through the tied LM head (LLM2Vec's adaptation of bidirectional attention;
            masked_adaptation). The weight is one untuned value.
ls and focal change how confident the loss asks the model to be, so their NLL mixes ranking and calibration; they
are tested on accuracy as well. The one-forward arms run first for every seed, then the two-forward arms.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment check
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment freeze
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment run
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.loss_experiment report
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import save_file
import torch

from qwen_decision.v3 import architecture_search as search, long_context_experiment as lce, lora_experiment as lora
from qwen_decision.v3 import lora_placement_experiment as placement, masked_adaptation as masked, modeling
from qwen_decision.v3 import readout_experiment as records, robust_adaptation as views, train
from qwen_decision.v3 import training_method_experiment as tme

OUT = Path("runs/from-base/loss-v1")
SCREEN, PLACEMENT = tme.OUT, placement.OUT
BASELINE = "all"
PHASES = (("ls", "focal", "brier"), ("twoview", "consist", "mntp"))
ARMS = tuple(a for phase in PHASES for a in phase)
SEEDS = tme.SEEDS
SMOOTHING, GAMMA, JSD_WEIGHT, MNTP_WEIGHT, MASK_RATE, HEAD_CHUNK = .1, 2., 1., .5, .2, 64
TESTS = {"ls-all": ("all", "ls", ("nll", "correct")), "focal-all": ("all", "focal", ("nll", "correct")),
         "brier-all": ("all", "brier", ("nll",)), "twoview-all": ("all", "twoview", ("nll",)),
         "consist-twoview": ("twoview", "consist", ("nll",)), "mntp-all": ("all", "mntp", ("nll",))}
REPORTED = {"consist-all": ("all", "consist")} | {f"{a}-frozen": ("frozen", a) for a in ARMS}
read = tme.read


def smooth(targets, code_mask):
    q = targets / targets.sum(-1, keepdim=True)
    shown = code_mask.float()
    return (1 - SMOOTHING) * q + SMOOTHING * shown / shown.sum(-1, keepdim=True)


def focal(z, targets, gamma=GAMMA):
    log_p = z.float().log_softmax(-1)
    return -(targets * (1 - log_p.exp()).pow(gamma) * log_p).sum(-1)


def second_views(renderer, rows, indices, seed, step):
    """A second random view of each row of the step, drawn independently of the matched one."""
    return [views.random_view(renderer, rows[i], random.Random(int(search.digest(["loss-v1-second-view", seed, step, j])[:16], 16)))
            for j, i in enumerate(indices)]


def masked_views(renderer, rows, indices, encoded, seed, step, mask_id):
    out = []
    for j, (i, e) in enumerate(zip(indices, encoded)):
        row = e | {"eligible_positions": masked.content_positions(renderer, rows[i], e)}
        out.append(masked.masked_view(row, mask_id=mask_id, rate=MASK_RATE,
                                      seed=int(search.digest(["loss-v1-mntp", seed, step, j])[:16], 16))
                   | {"code_ids": e["code_ids"]})
    return out


def extras(arm, renderer, rows, indices, encoded, seed, step, mask_id):
    if arm in ("twoview", "consist"):
        return second_views(renderer, rows, indices, seed, step)
    if arm == "mntp":
        return masked_views(renderer, rows, indices, encoded, seed, step, mask_id)
    return None


def mntp_loss(model, group):
    """masked_adaptation.loss on a packed batch: sum over rows of the mean masked-token CE."""
    batch = modeling.pack_rows([v["input_ids"] for v in group], [v["code_ids"] for v in group], "cuda")
    hidden = model.model.language_model(
        input_ids=batch["ids"][None], position_ids=batch["positions"][None], cu_seq_lens_q=batch["cu"],
        cu_seq_lens_k=batch["cu"], max_length_q=batch["longest"], max_length_k=batch["longest"],
        cu_seqlens_cpu=batch["cu_cpu"], use_cache=False).last_hidden_state[0]
    starts = batch["cu_cpu"].tolist()
    reads = [starts[j] + p for j, v in enumerate(group) for p in v["read_positions"]]
    labels = [t for v in group for t in v["labels"]]
    weights = [1 / len(v["labels"]) for v in group for _ in v["labels"]]
    device = hidden.device
    return masked.chunked_loss(hidden[torch.tensor(reads, device=device)], model.get_output_embeddings().weight,
                               torch.tensor(labels, device=device), torch.tensor(weights, device=device),
                               chunk_size=HEAD_CHUNK)


def update(model, opt, arm, encoded, step, extra=None):
    """tme.update with the arm's loss. `extra` holds each row's second view (twoview, consist) or masked view (mntp).
    Returns the objective, its parts (ce is the matched view's CE in every arm) and the gradient norm."""
    model.train()
    opt.zero_grad()
    paired = {id(r): x for r, x in zip(encoded, extra)} if extra is not None else {}
    parts = defaultdict(float)
    for group in train.microbatches(encoded, lce.TRAIN_ARGS):
        other = [paired[id(r)] for r in group] if paired else []
        tokens = sum(len(r["input_ids"]) for r in group)
        train.set_checkpointing(model, tokens + (arm != "mntp") * sum(len(o["input_ids"]) for o in other) > lce.CHECKPOINT_ABOVE)
        batch, targets = tme.fb.packed(group)
        z = modeling.packed_code_logits(model, batch)
        ce = train.soft_ce(z, targets)
        if arm == "ls":
            rows = train.soft_ce(z, smooth(targets, batch["code_mask"]))
        elif arm == "focal":
            rows = focal(z, targets)
        elif arm == "brier":
            rows = ce + tme.BRIER_WEIGHT * tme.brier(z, targets)
        elif arm in ("twoview", "consist"):
            batch2, targets2 = tme.fb.packed(other)
            z2 = modeling.packed_code_logits(model, batch2)
            ce2 = train.soft_ce(z2, targets2)
            rows = (ce + ce2) / 2
            parts["ce2"] += ce2.sum().item() / len(encoded)
            js = views.jensen_shannon(views.align(z, group), views.align(z2, other))
            parts["jsd"] += js.sum().item() / len(encoded)
            if arm == "consist":
                rows = rows + JSD_WEIGHT * js
        else:
            rows = ce
        loss = rows.sum() / len(encoded)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        loss.backward()
        parts["loss"] += loss.item()
        parts["ce"] += ce.sum().item() / len(encoded)
        if arm == "mntp":
            train.set_checkpointing(model, tokens > lce.CHECKPOINT_ABOVE)
            m = mntp_loss(model, other) / len(encoded)
            if not torch.isfinite(m):
                raise ValueError(f"nonfinite masked loss at step {step}")
            (MNTP_WEIGHT * m).backward()
            parts["mntp"] += m.item()
            parts["loss"] += MNTP_WEIGHT * m.item()
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError(f"nonfinite gradients at step {step}")
    lr = min(step / tme.WARMUP, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - tme.WARMUP) / (tme.STEPS - tme.WARMUP))))
    for g in opt.param_groups:
        g["lr"] = tme.BODY_LR * g["lr_scale"] * lr
    opt.step()
    return dict(parts), norm


def mask_id(renderer) -> int:
    ids = renderer.tokenizer.encode("_", add_special_tokens=False)
    if len(ids) != 1 or ids[0] in renderer.tokenizer.all_special_ids:
        raise ValueError("underscore must be one existing ordinary token")
    return ids[0]


def baseline_path(arm: str, seed: int) -> Path:
    return PLACEMENT / f"all-seed{seed}" if arm == BASELINE else SCREEN / f"{arm}-seed{seed}"


def check(out: Path) -> None:
    """Before any fit: the baseline is unchanged and complete, every loss gives known values on known inputs, the
    second view aligns to the same semantic targets, every pool row has maskable text, and one update of each arm
    moves the adapters only."""
    out.mkdir(parents=True, exist_ok=True)
    report = {"placement_implementation_unchanged": placement.implementation() == read(PLACEMENT / "plan.json")["implementation"],
              "baseline_complete": read(PLACEMENT / "report.json")["complete"] and read(SCREEN / "report.json")["complete"],
              "eval_sets": {name: len(v) for name, v in tme.eval_sets().items()}}
    if not (report["placement_implementation_unchanged"] and report["baseline_complete"]):
        raise ValueError(f"baseline unusable: {report}")
    if report["eval_sets"] != read(SCREEN / "plan.json")["eval_sets"]:
        raise ValueError("evaluation sets differ from the screen")
    z = torch.tensor([[2., 0., modeling.MASK], [0., 1., 0.]])
    t = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    m = torch.tensor([[True, True, False], [True, True, True]])
    p = z.softmax(-1)
    report["smooth_known"] = {"value": smooth(t, m).tolist(), "expected": [[.95, .05, 0.], [.1 / 3, .9 + .1 / 3, .1 / 3]]}
    report["focal_known"] = {"value": focal(z, t).tolist(),
                             "expected": [float(-(1 - p[0, 0]) ** 2 * p[0, 0].log()), float(-(1 - p[1, 1]) ** 2 * p[1, 1].log())],
                             "gamma0_minus_ce": float((focal(z, t, 0.) - train.soft_ce(z, t)).abs().max())}
    if (not torch.allclose(smooth(t, m), torch.tensor(report["smooth_known"]["expected"]))
            or not np.allclose(report["focal_known"]["value"], report["focal_known"]["expected"])
            or report["focal_known"]["gamma0_minus_ce"] > 1e-6):
        raise ValueError(f"loss values wrong: {report}")
    renderer = lce.renderers()[0]["native"]
    pool = records.rows(tme.POOL)
    token = mask_id(renderer)
    eligible = []
    for r in pool:
        eligible.append(len(masked.content_positions(renderer, r, views.view(renderer, r))))
    report["mntp"] = {"mask_id": token, "rows": len(eligible), "eligible_min": min(eligible),
                      "eligible_mean": float(np.mean(eligible))}
    indices = list(range(8))
    first = lce.step_rows(renderer, pool, indices, SEEDS[0], 1)
    second = second_views(renderer, pool, indices, SEEDS[0], 1)
    semantic = [torch.tensor(r["target"]) for r in pool[:8]]
    report["second_view"] = {"orders_differ": sum(a["order"] != b["order"] for a, b in zip(first, second)),
                             "aligned_targets_equal": all(
                                 torch.equal(views.align(torch.tensor([a["target"]]), [a])[0], s)
                                 and torch.equal(views.align(torch.tensor([b["target"]]), [b])[0], s)
                                 for a, b, s in zip(first, second, semantic))}
    if not report["second_view"]["aligned_targets_equal"]:
        raise ValueError("second view does not align to the semantic targets")
    with torch.no_grad():
        model = placement.load_model(BASELINE).eval()
        train.set_checkpointing(model, False)
        batch, _ = tme.fb.packed(first)
        batch2, _ = tme.fb.packed(second)
        za, zb = modeling.packed_code_logits(model, batch), modeling.packed_code_logits(model, batch2)
        report["base_jsd_between_views"] = views.jensen_shannon(views.align(za, first), views.align(zb, second)).tolist()
        report["base_jsd_same_view"] = float(views.jensen_shannon(za, za).abs().max())
        report["base_mntp_loss_per_row"] = float(mntp_loss(model, masked_views(renderer, pool, indices, first, SEEDS[0], 1, token)) / 8)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for arm in ARMS:
        model = placement.load_model(BASELINE)
        params = dict(model.named_parameters())
        adapters = lora.adapters(model)
        frozen = {n: p.detach().clone() for n, p in params.items() if n not in adapters}
        opt = lora.make_optimizer(model)
        masters = {id(p): v for p, v in opt.pairs}
        b_start = {n: masters[id(p)].clone() for n, p in adapters.items() if n.endswith("lora_b")}
        parts, norm = update(model, opt, arm, first, tme.WARMUP, extras(arm, renderer, pool, indices, first, SEEDS[0], 1, token))
        report[arm] = {"parts": parts, "grad_norm": norm, "lr": opt.param_groups[0]["lr"],
                       "base_changed": any(bool((params[n].detach() != w).any()) for n, w in frozen.items()),
                       "b_master_changed": all(bool((masters[id(adapters[n])] != v).any()) for n, v in b_start.items())}
        print("CHECK", arm, json.dumps(report[arm]), flush=True)
        if (report[arm]["base_changed"] or not report[arm]["b_master_changed"] or not math.isfinite(norm)
                or not math.isclose(report[arm]["lr"], lora.LORA_LR)):
            raise ValueError(f"{arm} wrong: {report[arm]}")
        del model, opt, frozen
        gc.collect()
        torch.cuda.empty_cache()
    train.atomic_json(out / "check.json", report)
    print("CHECK", json.dumps({k: v for k, v in report.items() if k not in ARMS}), flush=True)


def smoke(out: Path) -> None:
    """The screen's views for the first 12 steps of seed 131, update timing, and the pool's longest rows."""
    rows = records.rows(tme.POOL)
    renderer = lce.renderers()[0]["native"]
    token = mask_id(renderer)
    reference = read(SCREEN / "smoke.json")
    schedule = list(train.batches(len(rows), tme.BATCH_SIZE, 12, random.Random(SEEDS[0])))
    lengths = [len(views.view(renderer, r)["input_ids"]) for r in rows]
    longest = sorted(range(len(rows)), key=lengths.__getitem__)[-3:]
    report = {}
    for arm in ARMS:
        train.seed_all(SEEDS[0])
        model = placement.load_model(BASELINE)
        opt = lora.make_optimizer(model)
        digests, times = [], []
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = lce.step_rows(renderer, rows, indices, SEEDS[0], step)
            digests.append(lce.view_digest(encoded))
            extra = extras(arm, renderer, rows, indices, encoded, SEEDS[0], step, token)
            tick = time.monotonic()
            parts, _ = update(model, opt, arm, encoded, step, extra)
            torch.cuda.synchronize()
            times.append({"seconds": time.monotonic() - tick, "tokens": sum(len(r["input_ids"]) for r in encoded)} | parts)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        encoded = [views.view(renderer, rows[i]) for i in longest]
        update(model, opt, arm, encoded, 13, extras(arm, renderer, rows, longest, encoded, SEEDS[0], 13, token))
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
    return placement.implementation() | {str(Path(f)): train.sha256_file(Path(f)) for f in (masked.__file__, __file__)}


def freeze(out: Path) -> dict:
    path = out / "plan.json"
    if path.exists():
        raise ValueError("plan already frozen")
    checked, smoked = read(out / "check.json"), read(out / "smoke.json")
    if not smoked["matched_views"]:
        raise ValueError("smoke did not match views")
    screen, where = read(SCREEN / "plan.json"), read(PLACEMENT / "plan.json")
    plan = {"experiment": "from-base-loss-v1", "base": screen["base"], "base_sha256": screen["base_sha256"],
            "pool": screen["pool"], "pool_sha256": train.sha256_file(tme.POOL),
            "screen_plan_sha256": train.sha256_file(SCREEN / "plan.json"),
            "placement_plan_sha256": train.sha256_file(PLACEMENT / "plan.json"),
            "model": {"attention": "d3 (noncausal)", "lora": where["arms"][BASELINE], "dropout": lora.DROPOUT, "lr": lora.LORA_LR},
            "arms": {"ls": f"CE against the gold smoothed over the shown codes, eps {SMOOTHING}",
                     "focal": f"focal loss, gamma {GAMMA}",
                     "brier": f"CE + {tme.BRIER_WEIGHT} x Brier over the shown codes",
                     "twoview": "mean CE of the matched view and a second random view of the same rows",
                     "consist": f"twoview + {JSD_WEIGHT} x Jensen-Shannon between the two views in semantic order",
                     "mntp": f"CE + {MNTP_WEIGHT} x masked next-token CE (mean per row) on the matched view, "
                             f"{MASK_RATE:.0%} of natural-language tokens replaced by '_', h[i-1] predicts x[i]"},
            "baselines": {BASELINE: "lora_placement_experiment all (CE)", "frozen": "screen frozen (full FT, DeltaNet mixers frozen)"},
            "seeds": list(SEEDS), "order": "phase by phase (" + "; ".join(", ".join(p) for p in PHASES) + "), seed-major within a phase",
            "steps": tme.STEPS, "batch_size": tme.BATCH_SIZE, "warmup": tme.WARMUP,
            "schedule": "linear warmup then cosine to 10%, as the screen",
            "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1, as the screen",
            "views": "lce.step_rows, checked per step against the screen's causal fit of the seed; second views and masks "
                     "drawn from sha256 of (seed, step, row slot)",
            "eval_sets": checked["eval_sets"], "primary_metric": screen["primary_metric"],
            "secondary_metrics": screen["secondary_metrics"],
            "tests": {k: {"baseline": a, "candidate": b, "metrics": [f"{g} {m}" for m in ms for g in ("selection", "long")]}
                      for k, (a, b, ms) in TESTS.items()},
            "decision": "candidate better on a metric if at least 4 of the 5 seed deltas favour it and the hierarchical "
                        "seed x item bootstrap 95% interval excludes 0 on its side; worse by the mirror rule. 16 tests: "
                        "each loss against CE on selection and long NLL, ls and focal also on accuracy (their NLL mixes "
                        "calibration and ranking), and consist against twoview (the consistency term alone).",
            "reported_only": "secondary metrics, per set, training loss parts, " + ", ".join(REPORTED),
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
    token = mask_id(renderer)
    reference = records.rows(SCREEN / f"causal-seed{seed}/train.jsonl")
    train.seed_all(seed)
    model = placement.load_model(BASELINE)
    opt = lora.make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (dest / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(train.batches(len(rows), tme.BATCH_SIZE, tme.STEPS, random.Random(seed)), 1):
            encoded = lce.step_rows(renderer, rows, indices, seed, step)
            digest = lce.view_digest(encoded)
            if digest != reference[step - 1]["view_sha256"]:
                raise ValueError(f"step {step}: rows or views differ from the screen's causal fit")
            extra = extras(arm, renderer, rows, indices, encoded, seed, step, token)
            parts, norm = update(model, opt, arm, encoded, step, extra)
            entry = {"step": step, **parts, "grad_norm": norm, "view_sha256": digest,
                     "second_view_sha256": lce.view_digest(extra) if arm in ("twoview", "consist") else None,
                     "tokens": sum(len(r["input_ids"]) for r in encoded), "seconds": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0:
                print("FIT", arm, seed, step, round(parts["loss"], 4), round(entry["seconds"]), flush=True)
                train.atomic_json(out / "status.json", {"state": "training", "arm": arm, "seed": seed, "step": step,
                                                        "steps": tme.STEPS, "seconds": entry["seconds"]})
    trained_seconds = time.monotonic() - started
    adapters = lora.adapters(model)
    masters = {id(p): m for p, m in opt.pairs}
    save_file({n: p.detach().contiguous() for n, p in adapters.items()}
              | {f"master.{n}": masters[id(p)].contiguous() for n, p in adapters.items()}, str(dest / "adapter.safetensors"))
    (dest / "adapter.json").write_text(json.dumps({"attention": "noncausal", "base": plan["base"], "arm": arm,
                                                   "loss": plan["arms"][arm], **plan["model"]["lora"],
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


def last_updates(path: Path, n=64) -> dict:
    """Mean loss parts over the last n updates of a fit."""
    entries = records.rows(path)[-n:]
    return {k: float(np.mean([e[k] for e in entries])) for k in ("loss", "ce", "ce2", "jsd", "mntp") if k in entries[-1]}


def report(out: Path) -> None:
    """Every comparison whose two arms are complete for all seeds, so it can run after the first phase."""
    plan = read(out / "plan.json")
    names = list(tme.eval_sets())
    seeds = plan["seeds"]
    ready = [a for a in ARMS if all((out / f"{a}-seed{s}/result.json").exists() for s in seeds)]
    evals = {arm: {s: {name: records.rows(out / f"{arm}-seed{s}/eval-{name}.jsonl") for name in names} for s in seeds}
             for arm in ready}
    evals |= {arm: {s: {name: records.rows(baseline_path(arm, s) / f"eval-{name}.jsonl") for name in names} for s in seeds}
              for arm in (BASELINE, "frozen")}
    result = {"plan_sha256": train.sha256_file(out / "plan.json"), "complete_arms": ready,
              "scores": {arm: {s: read((out / f"{arm}-seed{s}" if arm in ARMS else baseline_path(arm, s)) / "result.json")["eval"]
                               for s in seeds} for arm in evals},
              "training": {arm: {s: last_updates(out / f"{arm}-seed{s}/train.jsonl") for s in seeds} for arm in ready}}
    groups = {"selection": ["selection"], "long": list(lce.LONG_SETS)}
    pairs = {k: (a, b) for k, (a, b, _) in TESTS.items()} | REPORTED
    result["comparisons"] = {label: {f"{g} {m}": tme.hierarchical(evals[a], evals[b], sets, m, draws=10000 if m == "nll" else 2000)
                                     for g, sets in groups.items() for m in tme.METRICS}
                             for label, (a, b) in pairs.items() if a in evals and b in evals}
    result["tests"] = {k: {f"{g} {m}": result["comparisons"][k][f"{g} {m}"] for m in ms for g in groups}
                       for k, (_, _, ms) in TESTS.items() if k in result["comparisons"]}
    result["complete"] = len(ready) == len(ARMS)
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
        for phase in PHASES:
            for seed in map(int, args.seeds.split(",")):
                for arm in phase:
                    fit(args.output, arm, seed)
            report(args.output)


if __name__ == "__main__":
    main()
