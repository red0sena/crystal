"""Native vs question-first (echo) prompt layout, trained on long-weighted gold data, tested on long inputs.

Both arms start from g4b and see the same 32,768 v3-gold6 rows (a quarter over 1,536 tokens, none
truncated: data/longctx/train-long25-32k.jsonl) in the same order, with the same option permutations
and code draws. Only the layout of the prompt differs:
  native  d3: State, Question, Options. The 18 causal Gated DeltaNet layers read the state before the question.
  echo    Question and Options, then State, Question, Options. They read the state knowing the question.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment run --seeds 41
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.long_context_experiment report

Primary metric (plan.json, fixed before any fit): the long score, the mean of the macro accuracies of
BABILong (900 items: 20 per task, the same items at every length 0k-128k), JEVal long (684), QASPER (284)
and DocNLI long (1,188). Selection (4,147 short items) must not drop more than 0.5%p. LongBench v2 short is
reported, not scored. A prompt longer than the model's context or the GPU memory is unsupported and wrong.
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
from types import SimpleNamespace

import numpy as np
import torch

from qwen_decision.v3 import architecture_search as search, long_eval, long_training as lt, modeling
from qwen_decision.v3 import readout_experiment as records, robust_adaptation as views, train

OUT = Path("runs/long-context/echo-v1")
ARMS = {"native": "d3", "echo": "echo"}
STEPS, BATCH_SIZE, BODY_LR, WARMUP = 2048, 16, 3e-6, 64
TRAIN_ARGS = SimpleNamespace(pack=True, micro_batch=16, token_budget=16384)
EVAL_ARGS = SimpleNamespace(pack=True, micro_batch=16, token_budget=32768)
CHECKPOINT_ABOVE = 2048  # on the 24GB RTX 3090 an uncheckpointed 8k-token microbatch ran out of memory
LONG_SETS = ("babilong", "jeval-long", "qasper", "docnli-long")
BABILONG_PER_TASK = 20


def read(path):
    return json.loads(Path(path).read_text())


def load_model(path=search.BASE):
    modeling.compile_elementwise()
    model = modeling.load(path, attention="noncausal", kernel="varlen", dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.model.language_model.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def renderers():
    base, config = search.renderer_for(search.BASE)
    return {arm: type(base)(base.tokenizer, SimpleNamespace(**{**vars(base.args), "layout": layout}), [])
            for arm, layout in ARMS.items()}, config


def eval_sets() -> dict[str, list[dict]]:
    excluded = {name: set(v["excluded"]) for name, v in read(long_eval.OUT / "audit.json")["overlap"].items()}
    sets = {"selection": records.rows(lt.OUT / "selection.jsonl")}
    for name in (*LONG_SETS, "longbench-v2"):
        rows = [r for r in records.rows(long_eval.TEST / f"{name}.jsonl") if r["id"] not in excluded[name]]
        if name == "babilong":
            groups = defaultdict(list)
            for g in sorted({r["group"] for r in rows}, key=lambda g: search.digest(["babilong-eval-v1", g])):
                groups[g.split(":")[1]].append(g)
            keep = {g for gs in groups.values() for g in gs[:BABILONG_PER_TASK]}
            rows = [r for r in rows if r["group"] in keep]
        if name == "longbench-v2":
            rows = [r for r in rows if r["source"] == "longbench-v2-short"]
        sets[name] = rows
    return sets


def step_rows(renderer, rows, indices, seed, step):
    return [views.random_view(renderer, rows[i], random.Random(int(search.digest(["long-ctx-view-v1", seed, step, j])[:16], 16)))
            for j, i in enumerate(indices)]


def view_digest(encoded):
    return search.digest([[r["id"], r["order"], r["code_ids"], r["target"]] for r in encoded])


def update(model, opt, pad, encoded, step):
    model.train()
    opt.zero_grad()
    loss_sum = 0.
    for group in train.microbatches(encoded, TRAIN_ARGS):
        train.set_checkpointing(model, sum(len(r["input_ids"]) for r in group) > CHECKPOINT_ABOVE)
        logits, targets = train.logits_for(model, group, pad, packed=True)
        loss = train.soft_ce(logits, targets).sum() / len(encoded)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        loss.backward()
        loss_sum += loss.item()
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError(f"nonfinite gradients at step {step}")
    scale = min(step / WARMUP, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - WARMUP) / (STEPS - WARMUP))))
    for g in opt.param_groups:
        g["lr"] = BODY_LR * g["lr_scale"] * scale
    opt.step()
    return loss_sum, norm


@torch.no_grad()
def evaluate(model, renderer, rows):
    """Records like long_training.evaluate; unsupported (too long for the context or GPU) counts as wrong."""
    model.eval()
    train.set_checkpointing(model, False)
    pad = renderer.tokenizer.pad_token_id
    encoded = [views.view(renderer, r) for r in rows]
    probabilities, unsupported = {}, {}
    for r in encoded:
        if len(r["input_ids"]) > long_eval.MAX_CONTEXT:
            unsupported[r["id"]] = "longer than the model context"
    fits = [r for r in encoded if r["id"] not in unsupported]

    def run(group):
        z, _ = train.logits_for(model, group, pad, packed=True)
        for r, p in zip(group, z.float().softmax(-1).cpu()):
            probabilities[r["id"]] = p[:len(r["target"])].tolist()

    for group in train.microbatches(fits, EVAL_ARGS):
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
        p = probabilities.get(r["id"])
        pred = None if p is None else max(range(len(p)), key=p.__getitem__)
        result.append({"id": r["id"], "source": r["source"], "group": r.get("group", r["id"]), "tokens": len(e["input_ids"]),
                       "prediction": pred, "correct": pred is not None and r["target"][pred] >= max(r["target"]) - 1e-6,
                       "probabilities": p, "unsupported": unsupported.get(r["id"])})
    return search.metrics(result), result


def evaluate_all(model, renderer, dest: Path) -> dict:
    summary = {}
    for name, rows in eval_sets().items():
        started = time.monotonic()
        metric, result = evaluate(model, renderer, rows)
        search.save_records(dest / f"eval-{name}.jsonl", result)
        summary[name] = {"macro": metric["macro"], "accuracy": metric["accuracy"], "n": metric["n"],
                         "unsupported": sum(r["unsupported"] is not None for r in result),
                         "seconds": time.monotonic() - started, "sources": metric["sources"]}
        print("EVAL", dest.name, name, round(metric["macro"], 5), "unsupported", summary[name]["unsupported"],
              round(summary[name]["seconds"]), flush=True)
    summary["long_score"] = float(np.mean([summary[name]["macro"] for name in LONG_SETS]))
    return summary


def prepare(out: Path) -> dict:
    plan_path = out / "plan.json"
    if plan_path.exists():
        return read(plan_path)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"experiment": "long-context-echo-v1", "base": str(search.BASE),
            "base_sha256": train.sha256_file(search.BASE / "model.safetensors"),
            "pool": str(long_eval.LONG25), "pool_sha256": train.sha256_file(long_eval.LONG25),
            "test_manifest_sha256": train.sha256_file(long_eval.TEST / "manifest.json"),
            "audit_sha256": train.sha256_file(long_eval.OUT / "audit.json"),
            "arms": ARMS, "steps": STEPS, "batch_size": BATCH_SIZE, "body_lr": BODY_LR, "warmup": WARMUP,
            "schedule": "linear warmup then cosine to 10%", "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1",
            "loss": "gold soft CE over shown codes", "packing": "varlen packed rows, no truncation",
            "checkpoint_above_tokens": CHECKPOINT_ABOVE, "views": "robust_adaptation.random_view, identical across arms",
            "primary": "long score: mean of macro accuracy over " + ", ".join(LONG_SETS),
            "non_inferiority": "selection macro drop no more than 0.005",
            "decision": "echo is better if, for every completed seed, the paired long-score delta is positive and its "
                        "95% source-stratified group bootstrap interval excludes 0, and selection passes non-inferiority",
            "reported_only": "longbench-v2 short; BABILong per length"}
    train.atomic_json(plan_path, plan)
    return plan


def fit(out: Path, arm: str, seed: int) -> None:
    plan = prepare(out)
    dest = out / f"{arm}-seed{seed}"
    if (dest / "result.json").exists():
        print("SKIP_COMPLETE", arm, seed, flush=True)
        return
    if dest.exists():
        dest.rename(out / f"{dest.name}.incomplete-{time.time_ns()}")
    dest.mkdir()
    rows = records.rows(long_eval.LONG25)
    if train.sha256_file(long_eval.LONG25) != plan["pool_sha256"]:
        raise ValueError("training pool changed")
    renderer = renderers()[0][arm]
    reference = records.rows(out / f"native-seed{seed}/train.jsonl") if arm != "native" else None
    train.seed_all(seed)
    model = load_model()
    opt = train.MasterAdamW(model, BODY_LR)
    pad = renderer.tokenizer.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    schedule = train.batches(len(rows), BATCH_SIZE, STEPS, random.Random(seed))
    with (dest / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = step_rows(renderer, rows, indices, seed, step)
            digest = view_digest(encoded)
            if reference is not None and digest != reference[step - 1]["view_sha256"]:
                raise ValueError(f"step {step}: rows or views differ from native")
            loss, norm = update(model, opt, pad, encoded, step)
            entry = {"step": step, "loss": loss, "grad_norm": norm, "view_sha256": digest,
                     "tokens": sum(len(r["input_ids"]) for r in encoded), "seconds": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0:
                print("FIT", arm, seed, step, round(loss, 4), round(entry["seconds"]), flush=True)
                train.atomic_json(out / "status.json", {"state": "training", "arm": arm, "seed": seed, "step": step,
                                                        "steps": STEPS, "seconds": entry["seconds"]})
    trained_seconds = time.monotonic() - started
    _, config = renderers()
    modeling.save(model, renderer.tokenizer, dest / "model",
                  config | {"layout": ARMS[arm], "steps": STEPS, "trained_from": str(search.BASE),
                            "experiment": plan["experiment"], "arm": arm, "seed": seed})
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    train.atomic_json(out / "status.json", {"state": "evaluating", "arm": arm, "seed": seed})
    summary = evaluate_all(model, renderer, dest)
    train.atomic_json(dest / "result.json", {"arm": arm, "seed": seed, "layout": ARMS[arm], "train_seconds": trained_seconds,
                                             "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9, "eval": summary})
    print("RESULT", arm, seed, round(summary["long_score"], 5), round(summary["selection"]["macro"], 5), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def smoke(out: Path) -> None:
    """Matched views across arms, timing of the first real updates, and the longest row through one update."""
    prepare(out)
    rows = records.rows(long_eval.LONG25)
    rends, _ = renderers()
    schedule = list(train.batches(len(rows), BATCH_SIZE, 24, random.Random(41)))
    report = {}
    for arm, renderer in rends.items():
        train.seed_all(41)
        model = load_model()
        opt = train.MasterAdamW(model, BODY_LR)
        pad = renderer.tokenizer.pad_token_id
        digests, times = [], []
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = step_rows(renderer, rows, indices, 41, step)
            digests.append(view_digest(encoded))
            tick = time.monotonic()
            update(model, opt, pad, encoded, step)
            torch.cuda.synchronize()
            times.append({"seconds": time.monotonic() - tick, "tokens": sum(len(r["input_ids"]) for r in encoded)})
        longest = sorted(rows, key=lambda r: len(json.dumps(r["state"], ensure_ascii=False)))[-3:]
        encoded = [views.view(renderer, r) for r in longest]
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        update(model, opt, pad, encoded, 25)
        torch.cuda.synchronize()
        long_update = {"tokens": [len(r["input_ids"]) for r in encoded], "seconds": time.monotonic() - tick,
                       "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        probe = [r for r in eval_sets()["babilong"] if r["source"] == "babilong-128k"][:2]
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        metric, _ = evaluate(model, renderer, probe)
        eval_probe = {"rows": len(probe), "seconds": time.monotonic() - tick,
                      "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
        report[arm] = {"view_digests": digests, "update_times": times[4:], "long_update": long_update,
                       "eval_128k": eval_probe}
        print("SMOKE", arm, json.dumps({"median_update_s": float(np.median([t["seconds"] for t in times[4:]])),
                                        "long_update": long_update, "eval_128k": eval_probe}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    matched = report["native"]["view_digests"] == report["echo"]["view_digests"]
    train.atomic_json(out / "smoke.json", {"matched_views": matched, **report})
    if not matched:
        raise ValueError("arms see different rows or views")
    print("SMOKE_MATCHED", matched, flush=True)


def interval(base: dict[str, list[dict]], cand: dict[str, list[dict]], names, draws=5000):
    """Paired delta of the mean of per-set source macros, groups resampled within source within set."""
    rng = np.random.default_rng(20261024)
    structure, observed = [], []
    for name in names:
        b, c = {r["id"]: r for r in base[name]}, {r["id"]: r for r in cand[name]}
        if b.keys() != c.keys():
            raise ValueError(f"{name}: paired ids differ")
        groups = defaultdict(lambda: defaultdict(list))
        for key, r in b.items():
            groups[r["source"]][r["group"]].append(int(c[key]["correct"]) - int(r["correct"]))
        sources = [[np.asarray(v) for v in g.values()] for g in groups.values()]
        structure.append(sources)
        observed.append(np.mean([np.concatenate(g).mean() for g in sources]))
    samples = []
    for _ in range(draws):
        samples.append(np.mean([np.mean([np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]).mean()
                                         for g in sources]) for sources in structure]))
    return {"delta": float(np.mean(observed)), "per_set": dict(zip(names, map(float, observed))),
            "ci95": np.quantile(samples, [.025, .975]).tolist()}


def report(out: Path) -> None:
    plan = prepare(out)
    seeds = sorted({int(p.name.split("seed")[1]) for p in out.glob("echo-seed*") if (p / "result.json").exists()
                    and (out / p.name.replace("echo", "native") / "result.json").exists()})
    result = {"plan": plan, "seeds": {}}
    for seed in seeds:
        load = lambda arm: {name: records.rows(out / f"{arm}-seed{seed}/eval-{name}.jsonl") for name in eval_sets()}
        native, echo = load("native"), load("echo")
        long = interval(native, echo, LONG_SETS)
        selection = interval(native, echo, ["selection"])
        per_length = {s: search.metrics([r for r in echo["babilong"] if r["source"] == s])["accuracy"]
                      - search.metrics([r for r in native["babilong"] if r["source"] == s])["accuracy"]
                      for s in sorted({r["source"] for r in native["babilong"]})}
        scores = {arm: read(out / f"{arm}-seed{seed}/result.json")["eval"] for arm in ARMS}
        result["seeds"][seed] = {"long": long, "selection": selection, "babilong_by_length_delta": per_length,
                                 "longbench_v2_short": {arm: scores[arm]["longbench-v2"] for arm in ARMS},
                                 "scores": {arm: {k: v["macro"] if isinstance(v, dict) else v for k, v in s.items()}
                                            for arm, s in scores.items()},
                                 "passes": long["ci95"][0] > 0 and selection["delta"] >= -.005}
        print("SEED", seed, json.dumps({"long": long, "selection": selection["delta"]}), flush=True)
    result["echo_better"] = bool(seeds) and all(v["passes"] for v in result["seeds"].values())
    train.atomic_json(out / "report.json", result)


def hierarchical(base: dict, cand: dict, names, draws=10000) -> dict:
    """Seed x item bootstrap of the mean paired delta (significance-plan.json).

    base/cand: {seed: {set name: records}}. Every draw resamples the seeds with replacement and, once
    for all drawn seeds, the document groups within each source of each set (the test items are the
    same for every seed).
    """
    rng = np.random.default_rng(20261024)
    seeds = sorted(base)
    blocks = []  # per set: per source (diff sums [seed, group], item counts [group])
    for name in names:
        sources = defaultdict(lambda: defaultdict(lambda: [np.zeros(len(seeds)), 0]))
        for j, seed in enumerate(seeds):
            c = {r["id"]: r for r in cand[seed][name]}
            for r in base[seed][name]:
                cell = sources[r["source"]][r["group"]]
                cell[0][j] += int(c[r["id"]]["correct"]) - int(r["correct"])
                if j == 0:
                    cell[1] += 1
        blocks.append([(np.stack([v[0] for v in g.values()], 1), np.array([v[1] for v in g.values()]))
                       for g in sources.values()])

    def deltas(picks=None):  # [seed]: mean over sets of the source macro of the paired delta
        values = []
        for b, block in enumerate(blocks):
            per_source = []
            for k, (d, n) in enumerate(block):
                idx = slice(None) if picks is None else picks[b][k]
                per_source.append(d[:, idx].sum(1) / n[idx].sum())
            values.append(np.mean(per_source, 0))
        return np.mean(values, 0)

    observed = deltas()
    samples = []
    for _ in range(draws):
        delta = deltas([[rng.integers(len(n), size=len(n)) for _, n in block] for block in blocks])
        samples.append(delta[rng.integers(len(seeds), size=len(seeds))].mean())
    return {"seeds": seeds, "per_seed": dict(zip(map(str, seeds), map(float, observed))),
            "mean": float(observed.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "all_seeds_positive": bool((observed > 0).all())}


def significance(out: Path) -> None:
    plan = read(out / "significance-plan.json")
    names = list(eval_sets())
    load = lambda arm, seed: {name: records.rows(out / f"{arm}-seed{seed}/eval-{name}.jsonl") for name in names}
    done = [s for s in plan["seeds"] if all((out / f"{arm}-seed{s}/result.json").exists() for arm in ARMS)]
    native, echo = {s: load("native", s) for s in done}, {s: load("echo", s) for s in done}
    result = {"plan": plan, "completed_seeds": done}
    for label, seeds in (("all", done), ("confirmatory", [s for s in done if s != 41])):
        if not seeds:
            continue
        b, c = {s: native[s] for s in seeds}, {s: echo[s] for s in seeds}
        long = hierarchical(b, c, LONG_SETS)
        result[label] = {"long": long, "selection": hierarchical(b, c, ["selection"]),
                         "per_set": {name: hierarchical(b, c, [name], draws=2000) for name in (*LONG_SETS, "longbench-v2")},
                         "success": long["all_seeds_positive"] and long["ci95"][0] > 0}
        print("SIGNIFICANCE", label, seeds, json.dumps({"long": long, "selection": result[label]["selection"]}), flush=True)
    result["claim_supported"] = len(done) == len(plan["seeds"]) and result["all"]["success"]
    train.atomic_json(out / "significance.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "run", "report", "significance"])
    parser.add_argument("--seeds", default="41")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "report":
        report(args.output)
    elif args.stage == "significance":
        significance(args.output)
    else:
        for seed in map(int, args.seeds.split(",")):
            for arm in ARMS:
                fit(args.output, arm, seed)
        report(args.output)


if __name__ == "__main__":
    main()
