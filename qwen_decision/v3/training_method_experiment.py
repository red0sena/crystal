"""Training-method screen from the untrained base model: five recipes, 16,384 rows per fit, five seeds, gold NLL first.

Every arm starts from base Qwen3.5-0.8B (models/qwen3.5-0.8b) and reads the same 16,384 v3-gold6 rows
(data/longctx/train-nocap104-16k.jsonl: about 158 per source of all 104, none truncated) once, in the same
order, with the same option permutations and code draws, for 1,024 updates of 16 rows. The recipes:
  causal     the stock model, gold CE.
  noncausal  d3 attention (the 6 full-attention layers bidirectional), gold CE.
  frozen     noncausal with every Gated DeltaNet token mixer (linear_attn) frozen. LoRA placement in hybrid
             models (arXiv 2604.22127) found adapting Qwen3.5-0.8B's DeltaNet path destructive (GSM8K -14.8pp).
  brier      noncausal, gold CE + Brier score over the shown codes (NDM, arXiv 2610.25553; OneJev; DKnown-Jev).
  hoptrain   noncausal trained like from_base_experiment's hop arm (two reads, half the CE each, the second
             read takes the first read's state detached) but answering from the first read: one forward at
             inference. In that experiment the hop model's first and second reads had the same NLL and it beat
             causal on selection NLL in all three seeds (post hoc).

The primary metric is the gold NLL, -log of the probability on the gold option(s), averaged per source
(Signal and Noise, arXiv 2508.13144: continuous metrics separate small-scale decisions better than accuracy).
Evaluation: selection (4,147 short items) and the long sets of long_context_experiment with BABILong cut
at 16k tokens; LongBench v2 is left out. The decision rule is fixed in plan.json by `freeze`, before any fit.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment pool
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment check
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment freeze
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment run
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.training_method_experiment report
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from qwen_decision.v3 import architecture_search as search, from_base_experiment as fb, long_context_experiment as lce
from qwen_decision.v3 import long_eval, long_training as lt, modeling, readout_experiment as records
from qwen_decision.v3 import robust_adaptation as views, train

OUT = Path("runs/from-base/methods-v1")
POOL = long_eval.OUT / "train-nocap104-16k.jsonl"
ARMS = ("causal", "noncausal", "frozen", "brier", "hoptrain")
SEEDS = (131, 151, 173, 191, 211)  # none used by an earlier from-base fit
STEPS, BATCH_SIZE, BODY_LR, WARMUP = 1024, 16, 1e-5, 64
BRIER_WEIGHT = 1.  # NDM adds the Brier score to the log-loss unweighted
SETS = ("selection", *lce.LONG_SETS)
BABILONG = {f"babilong-{n}" for n in ("0k", "1k", "2k", "4k", "8k", "16k")}
CONFIRMATORY = {"noncausal-causal": ("causal", "noncausal", "long"), "hoptrain-causal": ("causal", "hoptrain", "selection")}
SCREENS = {"frozen-noncausal": ("noncausal", "frozen"), "brier-noncausal": ("noncausal", "brier"),
           "hoptrain-noncausal": ("noncausal", "hoptrain")}
METRICS = ("nll", "correct", "brier")
LOWER_IS_BETTER = {"nll": True, "brier": True, "correct": False}


def read(path):
    return json.loads(Path(path).read_text())


def pool() -> None:
    """16,384 rows of the 104-source pool: per source the first rows by sha256, round robin."""
    by = defaultdict(list)
    for r in records.rows(fb.POOL):
        by[r["source"]].append(r)
    for s in by:
        by[s].sort(key=lambda r: search.digest(["methods16k-v1", r["id"]]))
    taken, rows = dict.fromkeys(sorted(by), 0), []
    while len(rows) < 16384:
        for s in taken:
            if taken[s] < len(by[s]) and len(rows) < 16384:
                rows.append(by[s][taken[s]])
                taken[s] += 1
    rows.sort(key=lambda r: r["id"])
    sha = long_eval.write(POOL, rows)
    POOL.with_suffix(".manifest.json").write_text(json.dumps({
        "rows": len(rows), "sha256": sha, "parent": str(fb.POOL), "parent_sha256": train.sha256_file(fb.POOL),
        "sampler": "per source the first rows by sha256(['methods16k-v1', id]), round robin to 16,384",
        "sources": dict(Counter(r["source"] for r in rows))}, indent=1) + "\n")
    print("POOL", len(rows), len(taken), flush=True)


def eval_sets() -> dict[str, list[dict]]:
    sets = lce.eval_sets()
    return {"selection": sets["selection"], "babilong": [r for r in sets["babilong"] if r["source"] in BABILONG],
            **{name: sets[name] for name in lce.LONG_SETS if name != "babilong"}}


def load_model(arm: str):
    model = fb.load_model("causal" if arm == "causal" else "noncausal")
    if arm == "frozen":
        for name, p in model.model.language_model.named_parameters():
            if ".linear_attn." in name:
                p.requires_grad_(False)
    return model


def brier(z, targets):
    q = targets / targets.sum(-1, keepdim=True)
    return (z.float().softmax(-1) - q).pow(2).sum(-1)


def update(model, opt, arm, encoded, step, scale):
    model.train()
    opt.zero_grad()
    loss_sum, first_sum = 0., 0.
    for group in train.microbatches(encoded, lce.TRAIN_ARGS):
        train.set_checkpointing(model, sum(len(r["input_ids"]) for r in group) > lce.CHECKPOINT_ABOVE)
        batch, targets = fb.packed(group)
        if arm == "hoptrain":
            h = fb.first_read(model, batch)
            first = train.soft_ce(fb.readout(model, batch, h), targets).sum() / len(encoded)
            (first / 2).backward()
            first_sum += first.item()
            z = fb.readout(model, batch, fb.second_read(model, batch, fb.hop_layout(group), h.detach(), scale))
        else:
            z = modeling.packed_code_logits(model, batch)
        ce = train.soft_ce(z, targets)
        loss = (ce + BRIER_WEIGHT * brier(z, targets) if arm == "brier" else ce).sum() / len(encoded)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        (loss / 2 if arm == "hoptrain" else loss).backward()
        loss_sum += loss.item()
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError(f"nonfinite gradients at step {step}")
    lr = min(step / WARMUP, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - WARMUP) / (STEPS - WARMUP))))
    for g in opt.param_groups:
        g["lr"] = BODY_LR * g["lr_scale"] * lr
    opt.step()
    # first_sum is the first read's CE for hoptrain (whose answer comes from it); loss_sum is the second read's
    return loss_sum, (first_sum if arm == "hoptrain" else None), norm


def scores(p, target) -> dict:
    best = max(target)
    gold = [i for i, v in enumerate(target) if v >= best - 1e-6]
    q = np.asarray(target) / sum(target)
    pred = max(range(len(p)), key=p.__getitem__)
    return {"prediction": pred, "correct": pred in gold, "nll": -math.log(max(sum(p[i] for i in gold), 1e-12)),
            "brier": float(((np.asarray(p) - q) ** 2).sum())}


@torch.no_grad()
def evaluate(model, arm, renderer, rows, scale):
    """fb.evaluate with per-item NLL and Brier; hoptrain answers from the first read and keeps the second."""
    model.eval()
    train.set_checkpointing(model, False)
    encoded = [views.view(renderer, r) for r in rows]
    answer, second, unsupported = {}, {}, {}
    for r in encoded:
        if len(r["input_ids"]) + (arm == "hoptrain") > long_eval.MAX_CONTEXT:
            unsupported[r["id"]] = "longer than the model context"
    fits = [r for r in encoded if r["id"] not in unsupported]

    def run(group):
        batch, _ = fb.packed(group)
        if arm == "hoptrain":
            h = fb.first_read(model, batch)
            z, z2 = fb.readout(model, batch, h), fb.readout(model, batch, fb.second_read(model, batch, fb.hop_layout(group), h, scale))
        else:
            z, z2 = modeling.packed_code_logits(model, batch), None
        for i, r in enumerate(group):
            answer[r["id"]] = z[i].float().softmax(-1)[:len(r["target"])].tolist()
            if z2 is not None:
                second[r["id"]] = z2[i].float().softmax(-1)[:len(r["target"])].tolist()

    for group in train.microbatches(fits, lce.EVAL_ARGS):
        try:
            run(group)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for r in group:
                try:
                    run([r])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    unsupported[r["id"]] = "GPU memory"
    result = []
    for r, e in zip(rows, encoded):
        k = len(r["target"])
        p = answer.get(r["id"])
        record = {"id": r["id"], "source": r["source"], "group": r.get("group", r["id"]), "tokens": len(e["input_ids"]),
                  "unsupported": unsupported.get(r["id"]), "probabilities": p,
                  **(scores(p, r["target"]) if p is not None else scores([1 / k] * k, r["target"]) | {"correct": False})}
        if arm == "hoptrain" and r["id"] in second:
            record["second"] = {"probabilities": second[r["id"]], **scores(second[r["id"]], r["target"])}
        result.append(record)
    return result


def macro(result, metric) -> float:
    by = defaultdict(list)
    for r in result:
        by[r["source"]].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in by.values()]))


def evaluate_all(model, arm, renderer, dest: Path, scale) -> dict:
    summary = {}
    for name, rows in eval_sets().items():
        started = time.monotonic()
        result = evaluate(model, arm, renderer, rows, scale)
        search.save_records(dest / f"eval-{name}.jsonl", result)
        summary[name] = {m: macro(result, m) for m in METRICS} | {
            "n": len(result), "unsupported": sum(r["unsupported"] is not None for r in result),
            "seconds": time.monotonic() - started}
        if arm == "hoptrain":
            summary[name]["second_read"] = {m: macro([r["second"] | {"source": r["source"]} for r in result if "second" in r], m)
                                            for m in METRICS}
        print("EVAL", dest.name, name, {m: round(summary[name][m], 5) for m in METRICS}, round(summary[name]["seconds"]), flush=True)
    summary["long"] = {m: float(np.mean([summary[name][m] for name in lce.LONG_SETS])) for m in METRICS}
    return summary


def check(out: Path) -> None:
    """Before any fit: the frozen arm trains nothing in the DeltaNet mixers and everything else, the Brier term and
    the item scores behave on known inputs, and the hoptrain answer equals the noncausal arm on base weights."""
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    z = torch.tensor([[2., 0., modeling.MASK], [0., 0., modeling.MASK]])
    t = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    p = z.softmax(-1)
    report["brier_known"] = {"value": brier(z, t).tolist(), "expected": [float((p[0, 0] - 1) ** 2 + p[0, 1] ** 2),
                                                                      float(p[1, 0] ** 2 + (p[1, 1] - 1) ** 2)]}
    report["scores_known"] = scores([.7, .2, .1], [0., 1., 0.])
    if abs(report["scores_known"]["nll"] + math.log(.2)) > 1e-9 or report["scores_known"]["correct"]:
        raise ValueError("item scores wrong")
    renderer = lce.renderers()[0]["native"]
    rows = [views.view(renderer, r) for r in records.rows(POOL)[:8]]
    model = load_model("frozen")
    trainable = {n for n, p in model.model.language_model.named_parameters() if p.requires_grad}
    mixers = {n for n, _ in model.model.language_model.named_parameters() if ".linear_attn." in n}
    params = dict(model.model.language_model.named_parameters())
    others = [n for n in params if n.endswith("layers.0.mlp.down_proj.weight") or n.endswith("layers.3.self_attn.q_proj.weight")]
    weights = {n: params[n].detach().clone() for n in mixers}
    opt = train.MasterAdamW(model, BODY_LR)
    masters = {id(p): m for p, m in opt.pairs}
    # one step at lr 1e-5 is below BF16 resolution, so the trained side is checked on the FP32 masters
    started = {n: masters[id(params[n])].clone() for n in others}
    update(model, opt, "frozen", rows, WARMUP, 0.)
    report["frozen"] = {"mixer_tensors": len(mixers), "mixer_parameters": sum(params[n].numel() for n in mixers),
                        "trainable_parameters": sum(params[n].numel() for n in trainable),
                        "mixers_trainable": len(mixers & trainable),
                        "mixers_in_optimizer": sum(id(params[n]) in masters for n in mixers),
                        "mixer_changed": any(bool((params[n].detach() != w).any()) for n, w in weights.items()),
                        "others_master_changed": [bool((masters[id(params[n])] != v).any()) for n, v in started.items()]}
    if (report["frozen"]["mixers_trainable"] or report["frozen"]["mixers_in_optimizer"] or report["frozen"]["mixer_changed"]
            or not all(report["frozen"]["others_master_changed"])):
        raise ValueError(f"frozen arm wrong: {report['frozen']}")
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits = {}
        for arm in ("noncausal", "hoptrain"):
            model = load_model(arm).eval()
            train.set_checkpointing(model, False)
            batch, _ = fb.packed(rows)
            logits[arm] = (modeling.packed_code_logits(model, batch) if arm == "noncausal"
                           else fb.readout(model, batch, fb.first_read(model, batch)))
            if arm == "hoptrain":
                report["embedding_rms"] = fb.embedding_rms(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    report["hoptrain_answer_vs_noncausal_max_diff"] = float((logits["hoptrain"] - logits["noncausal"]).abs().max())
    report["eval_sets"] = {name: len(v) for name, v in eval_sets().items()}
    train.atomic_json(out / "check.json", report)
    print("CHECK", json.dumps(report), flush=True)


def smoke(out: Path) -> None:
    """Matched views across arms, update timing, and the pool's longest rows through one update."""
    rows = records.rows(POOL)
    renderer = lce.renderers()[0]["native"]
    scale = read(out / "check.json")["embedding_rms"]
    schedule = list(train.batches(len(rows), BATCH_SIZE, 12, random.Random(SEEDS[0])))
    lengths = [len(views.view(renderer, r)["input_ids"]) for r in rows]
    longest = sorted(range(len(rows)), key=lengths.__getitem__)[-3:]
    report = {"pool_tokens": {"max": max(lengths), "mean": float(np.mean(lengths)), "over_1536": sum(n > 1536 for n in lengths)}}
    for arm in ARMS:
        train.seed_all(SEEDS[0])
        model = load_model(arm)
        opt = train.MasterAdamW(model, BODY_LR)
        digests, times = [], []
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = lce.step_rows(renderer, rows, indices, SEEDS[0], step)
            digests.append(lce.view_digest(encoded))
            tick = time.monotonic()
            loss, first, _ = update(model, opt, arm, encoded, step, scale)
            torch.cuda.synchronize()
            times.append({"seconds": time.monotonic() - tick, "tokens": sum(len(r["input_ids"]) for r in encoded),
                          "loss": loss, "first_read_loss": first})
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        update(model, opt, arm, [views.view(renderer, rows[i]) for i in longest], 13, scale)
        torch.cuda.synchronize()
        median = float(np.median([t["seconds"] for t in times[3:]]))
        report[arm] = {"view_digests": digests, "updates": times, "median_update_s": median,
                       "estimated_fit_minutes": median * STEPS / 60,
                       "long_update": {"tokens": [lengths[i] for i in longest], "seconds": time.monotonic() - tick,
                                       "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}}
        print("SMOKE", arm, json.dumps({k: v for k, v in report[arm].items() if k not in ("view_digests", "updates")}), flush=True)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    report["matched_views"] = all(report[arm]["view_digests"] == report[ARMS[0]]["view_digests"] for arm in ARMS)
    train.atomic_json(out / "smoke.json", report)
    if not report["matched_views"]:
        raise ValueError("arms see different rows or views")


def implementation() -> dict:
    return fb.implementation() | {str(Path(__file__)): train.sha256_file(Path(__file__))}


def freeze(out: Path) -> dict:
    path = out / "plan.json"
    if path.exists():
        raise ValueError("plan already frozen")
    checked, smoked = read(out / "check.json"), read(out / "smoke.json")
    if not smoked["matched_views"]:
        raise ValueError("smoke did not match views")
    plan = {"experiment": "from-base-methods-v1", "base": str(fb.BASE),
            "base_sha256": train.sha256_file(fb.BASE / "model.safetensors-00001-of-00001.safetensors"),
            "pool": str(POOL), "pool_sha256": train.sha256_file(POOL),
            "test_manifest_sha256": train.sha256_file(long_eval.TEST / "manifest.json"),
            "audit_sha256": train.sha256_file(long_eval.OUT / "audit.json"),
            "arms": {"causal": "stock attention, gold CE", "noncausal": "d3 attention, gold CE",
                     "frozen": "noncausal, Gated DeltaNet token mixers (linear_attn) frozen",
                     "brier": f"noncausal, gold CE + {BRIER_WEIGHT} x Brier over shown codes",
                     "hoptrain": "noncausal, two reads with half the CE each, answer from the first read"},
            "hop_embedding_rms": checked["embedding_rms"], "seeds": list(SEEDS),
            "order": "seed-major: every arm of a seed before the next seed",
            "steps": STEPS, "batch_size": BATCH_SIZE, "rows_per_fit": STEPS * BATCH_SIZE, "body_lr": BODY_LR,
            "warmup": WARMUP, "schedule": "linear warmup then cosine to 10%",
            "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1",
            "views": "lce.step_rows, identical across arms, checked per step",
            "eval_sets": checked["eval_sets"],
            "primary_metric": "gold NLL: -log of the probability on the gold option(s), mean per source; "
                              "selection = its source macro, long = mean of the source macros of " + ", ".join(lce.LONG_SETS)
                              + " (BABILong 0k-16k)",
            "secondary_metrics": "accuracy (correct) and Brier score, same aggregation",
            "confirmatory": {k: {"baseline": a, "candidate": b, "metric": f"{m} NLL"} for k, (a, b, m) in CONFIRMATORY.items()},
            "confirmatory_reason": "post hoc NLL findings of runs/from-base/structures-v1 (continuous-reanalysis.json), new seeds",
            "screens": {k: {"baseline": a, "candidate": b, "metrics": ["selection NLL", "long NLL"]} for k, (a, b) in SCREENS.items()},
            "decision": "candidate better on a metric if at least 4 of the 5 seed deltas favour it and the hierarchical "
                        "seed x item bootstrap 95% interval excludes 0 on its side; worse by the mirror rule. "
                        "Confirmatory: 2 tests. Screens: 6 tests, a pass advances the recipe to a 65,536-row confirmation, "
                        "it is not a claim.",
            "reported_only": "secondary metrics, per set, hoptrain second read, every other arm pair",
            "implementation": implementation(), "check": checked,
            "smoke": {arm: {k: smoked[arm][k] for k in ("median_update_s", "estimated_fit_minutes", "long_update")} for arm in ARMS}
                     | {"pool_tokens": smoked["pool_tokens"]}}
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
    rows = records.rows(POOL)
    if train.sha256_file(POOL) != plan["pool_sha256"]:
        raise ValueError("training pool changed")
    renderer = lce.renderers()[0]["native"]
    reference = None if arm == ARMS[0] else records.rows(out / f"{ARMS[0]}-seed{seed}/train.jsonl")
    scale = plan["hop_embedding_rms"]
    train.seed_all(seed)
    model = load_model(arm)
    opt = train.MasterAdamW(model, BODY_LR)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (dest / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(train.batches(len(rows), BATCH_SIZE, STEPS, random.Random(seed)), 1):
            encoded = lce.step_rows(renderer, rows, indices, seed, step)
            digest = lce.view_digest(encoded)
            if reference is not None and digest != reference[step - 1]["view_sha256"]:
                raise ValueError(f"step {step}: rows or views differ from {ARMS[0]}")
            loss, first, norm = update(model, opt, arm, encoded, step, scale)
            entry = {"step": step, "loss": loss, "first_read_loss": first, "grad_norm": norm, "view_sha256": digest,
                     "tokens": sum(len(r["input_ids"]) for r in encoded), "seconds": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0:
                print("FIT", arm, seed, step, round(loss, 4), round(entry["seconds"]), flush=True)
                train.atomic_json(out / "status.json", {"state": "training", "arm": arm, "seed": seed, "step": step,
                                                        "steps": STEPS, "seconds": entry["seconds"]})
    trained_seconds = time.monotonic() - started
    _, config = lce.renderers()
    modeling.save(model, renderer.tokenizer, dest / "model",
                  config | {"attention": "causal" if arm == "causal" else "noncausal", "layout": "d3", "steps": STEPS,
                            "trained_from": str(fb.BASE), "experiment": plan["experiment"], "arm": arm, "seed": seed})
    peak = torch.cuda.max_memory_allocated() / 1e9
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    train.atomic_json(out / "status.json", {"state": "evaluating", "arm": arm, "seed": seed})
    summary = evaluate_all(model, arm, renderer, dest, scale)
    train.atomic_json(dest / "result.json", {"arm": arm, "seed": seed, "train_seconds": trained_seconds,
                                             "peak_cuda_gb": peak, "eval": summary})
    print("RESULT", arm, seed, json.dumps({k: round(v, 5) for k, v in summary["long"].items()}),
          json.dumps({m: round(summary["selection"][m], 5) for m in METRICS}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def hierarchical(base: dict, cand: dict, names, metric: str, draws=10000) -> dict:
    """lce.hierarchical for any per-item metric: seeds and, once per draw for all seeds, groups within each source."""
    rng = np.random.default_rng(20261027)
    seeds = sorted(base)
    blocks = []
    for name in names:
        sources = defaultdict(lambda: defaultdict(lambda: [np.zeros(len(seeds)), 0]))
        for j, seed in enumerate(seeds):
            c = {r["id"]: r for r in cand[seed][name]}
            for r in base[seed][name]:
                cell = sources[r["source"]][r["group"]]
                cell[0][j] += float(c[r["id"]][metric]) - float(r[metric])
                if j == 0:
                    cell[1] += 1
        blocks.append([(np.stack([v[0] for v in g.values()], 1), np.array([v[1] for v in g.values()]))
                       for g in sources.values()])

    def deltas(picks=None):
        values = []
        for b, block in enumerate(blocks):
            per_source = []
            for k, (d, n) in enumerate(block):
                idx = slice(None) if picks is None else picks[b][k]
                per_source.append(d[:, idx].sum(1) / n[idx].sum())
            values.append(np.mean(per_source, 0))
        return np.mean(values, 0)

    observed = deltas()
    samples = [deltas([[rng.integers(len(n), size=len(n)) for _, n in block] for block in blocks])
               [rng.integers(len(seeds), size=len(seeds))].mean() for _ in range(draws)]
    sign = -1 if LOWER_IS_BETTER[metric] else 1
    ci = np.quantile(samples, [.025, .975]).tolist()
    favour = int((sign * observed > 0).sum())
    return {"seeds": seeds, "per_seed": dict(zip(map(str, seeds), map(float, observed))), "mean": float(observed.mean()),
            "ci95": ci, "seeds_favouring_candidate": favour,
            "better": favour >= len(seeds) - 1 and min(sign * x for x in ci) > 0,
            "worse": len(seeds) - favour >= len(seeds) - 1 and max(sign * x for x in ci) < 0}


def report(out: Path) -> None:
    plan = read(out / "plan.json")
    names = list(eval_sets())
    done = [s for s in plan["seeds"] if all((out / f"{arm}-seed{s}/result.json").exists() for arm in ARMS)]
    evals = {arm: {s: {name: records.rows(out / f"{arm}-seed{s}/eval-{name}.jsonl") for name in names} for s in done}
             for arm in ARMS}
    result = {"plan_sha256": train.sha256_file(out / "plan.json"), "completed_seeds": done,
              "scores": {arm: {s: read(out / f"{arm}-seed{s}/result.json")["eval"] for s in done} for arm in ARMS}}
    if done:
        groups = {"selection": ["selection"], "long": list(lce.LONG_SETS)}
        pairs = {k: (a, b) for k, (a, b, _) in CONFIRMATORY.items()} | SCREENS
        result["comparisons"] = {label: {f"{g} {m}": hierarchical(evals[a], evals[b], sets, m, draws=10000 if m == "nll" else 2000)
                                         for g, sets in groups.items() for m in METRICS}
                                 for label, (a, b) in pairs.items()}
        result["confirmatory"] = {k: result["comparisons"][k][f"{g} nll"] | {"metric": f"{g} nll"}
                                  for k, (_, _, g) in CONFIRMATORY.items()}
        result["screens"] = {k: {g: result["comparisons"][k][f"{g} nll"] for g in groups} for k in SCREENS}
        second = {s: {name: [r | r.get("second", {}) for r in v] for name, v in evals["hoptrain"][s].items()} for s in done}
        result["hoptrain_second_minus_answer"] = {f"{g} {m}": hierarchical(evals["hoptrain"], second, sets, m, draws=2000)
                                                 for g, sets in groups.items() for m in ("nll", "correct")}
        result["complete"] = len(done) == len(plan["seeds"])
        for label, v in result["confirmatory"].items():
            print("CONFIRM", label, json.dumps(v), flush=True)
        for label, v in result["screens"].items():
            print("SCREEN", label, json.dumps({g: {k: v[g][k] for k in ("mean", "ci95", "better", "worse")} for g in v}), flush=True)
    train.atomic_json(out / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pool", "check", "smoke", "freeze", "run", "report"])
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "pool":
        pool()
    elif args.stage == "check":
        check(args.output)
    elif args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "report":
        report(args.output)
    else:
        # the bootstrap report runs once at the end so the GPU does not wait on it between seeds
        for seed in map(int, args.seeds.split(",")):
            for arm in ARMS:
                fit(args.output, arm, seed)
        report(args.output)


if __name__ == "__main__":
    main()
