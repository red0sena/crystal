"""Native vs matched residual readers: architecture and path-intervention trial.

Ordinary independent-example gold CE only. All arms start from the EXACT same
native checkpoint; new branches initially output zero. Final document tests
open only after all twelve fixed-step fits finish. No model ensembling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from qwen_decision.v3 import single_forward_data as data, semantic_matching as semantic
from qwen_decision.v3 import evidence_pair_experiment as previous, long_training as common
from qwen_decision.v3 import modeling, train, retrofit_recovery as context

OUT = Path("runs/single-forward/semantic-v1")
BASE = common.OUT / "native-seed41/step-4096"
ARMS = ("native", "generic", "semantic", "self")
SEEDS = (41, 73, 101)


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    return previous.implementation() | {str(Path(m.__file__)): train.sha256_file(Path(m.__file__))
                                       for m in (data, semantic)} | {str(Path(__file__)): train.sha256_file(Path(__file__))}


def prepare(out):
    if (out / "manifest.json").exists():
        raise ValueError("preserve existing plan")
    inputs = read(out / "inputs.json")
    for path, sha in inputs["files"].items():
        if train.sha256_file(Path(path)) != sha:
            raise ValueError(f"input changed: {path}")
    plan = {"experiment": "single-forward-semantic-v1", "arms": list(ARMS), "seeds": list(SEEDS),
        "steps": 512, "batch_size": 16, "body_lr": 3e-6, "matching_lr": 1e-4, "warmup": 32,
        "initialization": str(BASE), "new_branch": "two shared matching layers, width1024, heads8; scorer exactly zero at initialization",
        "same_initial_function": "native backbone and tied code head start with the same weights in all arms; new-branch seed initialization matched across three controls",
        "native": "original 24 hybrid layers and code readout",
        "generic": "native logits + matching correction; entire state vs generic task instruction+option",
        "semantic": "native logits + matching correction; actual evidence vs actual question/claim/hypothesis+option; fallback state/instruction for unsupported schemas",
        "self": "same semantic spans and exact same new parameters/initial weights; self attention on their concatenation instead of cross attention",
        "training": "all used language and tied embedding/head parameters; vision frozen; ordinary gold CE only; identical independent cases, target, code and option views per seed",
        "data": inputs["summary"], "training_mix": "8 target-family cases +8 broad anchors per update; 8192 unique training IDs; no paired-example objective",
        "objective": "mean independent-example gold CE across16 cases; no pair loss, distillation, ranking loss, ensemble, or generated labels",
        "optimizer": "BF16 model, FP32-master AdamW(.9,.95), WD.01, clip1; warmup32 then cosine to10%",
        "max_train_tokens": 1536, "max_eval_tokens": 4096, "dev_steps": [0, 128, 256, 512],
        "selection": "fixed final512; no accuracy early stop or best-checkpoint reselection",
        "primary": "ordinary per-example source-macro accuracy on five new document reading/verification/NLI families",
        "primary_contrasts": ["semantic-native", "semantic-generic", "semantic-self"],
        "mechanism_contrasts": ["(semantic-generic) active gap minus branch-off gap", "semantic question-zero harm minus same-span self-control question-zero harm"],
        "mechanism_scope": "partial interventions AFTER the unchanged joint encoder; encoder states retain evidence and query information; not total information removal or universal proof of a single-forward limit",
        "test_protocol": "new primary and retention documents scored only after all12 final models and exact input traces frozen; all head interventions use the same encoder states",
        "statistics": "5000 paired document draws within source; all arm/mode comparisons share draws; also resample the three paired training seeds; familywise95 bootstrap intervals for five preregistered primary/mechanism contrasts",
        "advance": "all five hierarchical familywise interval lower bounds >0; three architecture contrasts positive for every seed; retention one-sided95 lower bound >-1pp; exploratory gate, no automatic expansion",
        "retention": "six separate new general families; source-macro semantic-native and one-sided95 bootstrap interval; prespecified noninferiority margin1pp",
        "runtime": "audit exactly one language-model call and one execution of each24 backbone layers per inference batch; benchmark active scoring only with identical prepared batches; report latency/memory, no equal-FLOPs claim",
        "resume": "final weights-only checkpoints; incomplete current fit is archived and restarted deterministically from the common base; full optimizer resume verified in separate smoke",
        "limits": ["targets a semantic aggregation hypothesis, not all possible single-forward bottlenecks", "three additional-training seeds share one old seed41 initial checkpoint",
                   "bootstrap with only three training seeds has limited precision and coverage", "same data/updates and matched added parameters; attention FLOPs and wall time differ",
                   "identity initialization and residual readouts were tried previously; the new hypothesis is semantic routing, not residual novelty",
                   "case accuracy is primary; no official full CONDAQA/VitaminC leaderboard reproduction", *inputs["summary"]["limits"]]}
    files = dict(inputs["files"])
    files[str(out / "inputs.json")] = train.sha256_file(out / "inputs.json")
    common.verify_checkpoint(BASE)
    for name in ("model.safetensors", "config.json", "v3_config.json", "complete.json"):
        path = BASE / name
        files[str(path)] = train.sha256_file(path)
    train.atomic_json(out / "plan.json", plan)
    files[str(out / "plan.json")] = train.sha256_file(out / "plan.json")
    train.atomic_json(out / "manifest.json", {"frozen": False, "plan": plan, "files": files})
    print("PREPARED", ARMS, SEEDS, flush=True)


def verify(out, *, frozen=True):
    m = read(out / "manifest.json")
    if frozen and not m["frozen"]:
        raise ValueError("validate and freeze first")
    if m["frozen"] and m["implementation"] != implementation():
        raise ValueError("frozen implementation changed")
    for path, sha in m["files"].items():
        if train.sha256_file(Path(path)) != sha:
            raise ValueError(f"input changed: {path}")
    return m


def cache(out):
    m = verify(out, frozen=False)
    if m["frozen"]:
        raise ValueError("preserve frozen inputs")
    renderer, _ = common.search.renderer_for(common.search.BASE)
    targets = list(data.lines(out / "target-train.jsonl"))
    anchors = list(data.lines(out / "anchor-train.jsonl"))
    roles = Counter()
    for seed in SEEDS:
        path = out / f"encoded-seed{seed}.jsonl"
        if path.exists():
            if str(path) in m["files"] and train.sha256_file(path) == m["files"][str(path)]:
                continue
            raise ValueError("unverified cache exists")
        ti, ai = list(range(4096)), list(range(4096))
        random.Random(seed).shuffle(ti); random.Random(seed + 1).shuffle(ai)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for step in range(1, 513):
                raw = [targets[i] for i in ti[(step - 1) * 8:step * 8]] + [anchors[i] for i in ai[(step - 1) * 8:step * 8]]
                rows = [data.encode(renderer, r, random.Random(int(data.digest(["single-forward-view", seed, step, j])[:16], 16)))
                        for j, r in enumerate(raw)]
                if len(rows) != 16 or max(len(r["input_ids"]) for r in rows) > 1536:
                    raise ValueError("wrong matched batch")
                roles.update(r["semantic_role"] for r in rows)
                sha = data.digest([[r["id"], r["input_ids"], r["code_ids"], r["target"]] for r in rows])
                f.write(json.dumps({"step": step, "rows": rows, "input_target_sha256": sha}, ensure_ascii=False) + "\n")
                if step % 128 == 0:
                    print("CACHE", seed, step, flush=True)
        tmp.rename(path)
        m["files"][str(path)] = train.sha256_file(path)
        train.atomic_json(out / "manifest.json", m)
    for name in ("dev", "primary-test", "retention-test"):
        path = out / f"{name}-encoded.jsonl"
        rows = [data.encode(renderer, r) for r in data.lines(out / f"{name}.jsonl")]
        common.search.save_records(path, rows)
        m["files"][str(path)] = train.sha256_file(path)
    train.atomic_json(out / "cache-audit.json", {"training_roles_across_encoded_seeds": dict(roles),
        "same_inputs_targets_for_all_arms": True, "role_selection_uses_labels": False,
        "interventions": list(semantic.INTERVENTIONS)})
    m["files"][str(out / "cache-audit.json")] = train.sha256_file(out / "cache-audit.json")
    train.atomic_json(out / "manifest.json", m)
    print("CACHE_READY", flush=True)


def load(arm, seed, path=None):
    torch.use_deterministic_algorithms(True)
    model = semantic.load(path or BASE)
    text = model.model.language_model
    if arm != "native" and not hasattr(text, "semantic_matching"):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            semantic.install(text, arm)
    if arm == "native" and hasattr(text, "semantic_matching"):
        raise ValueError("native has a residual module")
    if arm != "native" and text.semantic_matching.arm != arm:
        raise ValueError("arm and saved architecture differ")
    model.requires_grad_(False); text.requires_grad_(True); model.lm_head.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model._retrofit_evaluation_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    return model


def optimizer(model, plan):
    opt = train.MasterAdamW(model, plan["body_lr"])
    module = getattr(model.model.language_model, "semantic_matching", None)
    if module is not None:
        ids = {id(p) for p in module.parameters()}
        masters = [m for p, m in opt.pairs if id(p) in ids]
        mids = {id(p) for p in masters}
        for group in opt.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in mids]
        opt.optimizer.add_param_group({"params": masters, "weight_decay": .01,
            "lr_scale": plan["matching_lr"] / plan["body_lr"]})
    return opt


def update(model, opt, renderer, item, plan, step):
    model.train(); train.set_checkpointing(model, True); opt.zero_grad()
    loss_sum = 0.
    for rows in train.microbatches(item["rows"], common.BATCH):
        ids, mask, codes, valid, targets = train.collate(rows, renderer.tokenizer.pad_token_id)
        logits = semantic.logits(model, ids, mask, codes, valid, rows)
        loss = train.soft_ce(logits, targets).sum() / len(item["rows"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite ordinary CE")
        loss.backward(); loss_sum += loss.item()
    if step <= 3 and any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise ValueError("missing or nonfinite gradient")
    module = getattr(model.model.language_model, "semantic_matching", None)
    branch = float(sum(p.grad.float().square().sum() for p in module.parameters()).sqrt()) if module else 0.
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradients")
    scale = min(step / plan["warmup"], 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - plan["warmup"]) / (plan["steps"] - plan["warmup"]))))
    for group in opt.param_groups:
        group["lr"] = plan["body_lr"] * group["lr_scale"] * scale
    opt.step()
    return {"ce": loss_sum, "grad_norm": norm, "branch_grad_norm": branch}


@torch.no_grad()
def _evaluate(model, renderer, rows, modes):
    model.eval(); train.set_checkpointing(model, False)
    result = {mode: {} for mode in modes}
    for batch in train.microbatches(rows, common.BATCH):
        ids, mask, codes, valid, _ = train.collate(batch, renderer.tokenizer.pad_token_id)
        scores = semantic.multi_logits(model, ids, mask, codes, valid, batch, modes)
        for mode, z in scores.items():
            for row, value in zip(batch, z.softmax(-1).cpu()):
                p = value[:len(row["target"])].tolist(); pred = max(range(len(p)), key=p.__getitem__)
                result[mode][row["id"]] = {"id": row["id"], "source": row["source"], "group": row["group"],
                    "prediction": pred, "correct": row["target"][pred] >= max(row["target"]) - 1e-6, "probabilities": p}
    return {mode: [values[r["id"]] for r in rows] for mode, values in result.items()}


def evaluate(model, renderer, rows, modes=("active",)):
    return context.canonical_evaluation(model, _evaluate, renderer, rows, modes)


def dev(model, renderer, out, dest, step):
    rr = evaluate(model, renderer, list(data.lines(out / "dev-encoded.jsonl")))["active"]
    common.search.save_records(dest / f"dev-{step}.jsonl", rr)
    metrics = common.search.metrics(rr)
    train.atomic_json(dest / f"dev-{step}.json", metrics)
    print("DEV", dest.name, step, metrics["macro"], flush=True)


def save_checkpoint(model, opt, renderer, path, cfg, arm, seed, step, *, full=False):
    if path.exists():
        raise ValueError("preserve checkpoint")
    if opt is not None:
        opt.zero_grad()
    semantic.save(model, renderer.tokenizer, path, cfg | {"experiment": "single-forward-semantic-v1",
        "steps": step, "arm": arm, "seed": seed, "trained_from": str(BASE)})
    if full:
        payload = common.training_state(model, opt, step)
        torch.save(payload, path / "optimizer.pt.tmp"); (path / "optimizer.pt.tmp").rename(path / "optimizer.pt")
        del payload; gc.collect()
    files = {p.name: train.sha256_file(p) for p in path.iterdir() if p.suffix == ".safetensors" or
             p.name in ("optimizer.pt", "config.json", "v3_config.json", "decision_architecture.json")}
    train.atomic_json(path / "complete.json", {"step": step, "full_optimizer": full, "files": files})


@torch.no_grad()
def runtime_audit(model, renderer, rows):
    model.eval(); train.set_checkpointing(model, False)
    groups = list(train.microbatches(rows, common.BATCH))
    prepared = [(g, train.collate(g, renderer.tokenizer.pad_token_id)[:4]) for g in groups]
    def score():
        for g, (ids, mask, codes, valid) in prepared:
            semantic.logits(model, ids, mask, codes, valid, g)
    counts = Counter()
    hooks = [model.model.language_model.register_forward_hook(lambda *args: counts.update(["backbone"]))]
    for i, layer in enumerate(model.model.language_model.layers):
        hooks.append(layer.register_forward_hook(lambda *args, i=i: counts.update([f"layer{i}"])))
    g, args = prepared[0]
    semantic.logits(model, *args, g)
    for hook in hooks:
        hook.remove()
    if counts != Counter({"backbone": 1, **{f"layer{i}": 1 for i in range(24)}}):
        raise ValueError("inference did not execute exactly one native backbone pass")
    for g, args in prepared[:4]:
        semantic.logits(model, *args, g)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(3):
        t0 = time.perf_counter(); score(); torch.cuda.synchronize(); timings.append(time.perf_counter() - t0)
    return {"cases": len(rows), "batches": len(groups), "seconds_three_runs": timings,
        "median_seconds": float(np.median(timings)), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "forward_counts_one_batch": dict(counts), "parameters": sum(p.numel() for p in model.parameters()),
        "scope": "active single-pass scoring, identical prepared device batches; excludes tokenization/loading and head interventions; no service latency claim"}


def smoke(out, arm):
    m = verify(out, frozen=False); plan = m["plan"]; dest = out / f"smoke-{arm}"
    if dest.exists():
        raise ValueError("preserve smoke evidence")
    dest.mkdir(); renderer, cfg = common.search.renderer_for(common.search.BASE)
    with (out / "encoded-seed41.jsonl").open() as f:
        items = [json.loads(next(f)) for _ in range(3)]
    encoded = list(data.lines(out / "dev-encoded.jsonl"))[:12]
    train.seed_all(41); model = load(arm, 41)
    with torch.no_grad():
        model.eval(); train.set_checkpointing(model, False)
        rows = encoded[:2]; ids, mask, codes, valid, _ = train.collate(rows, renderer.tokenizer.pad_token_id)
        actual = semantic.logits(model, ids, mask, codes, valid, rows)
        expected = modeling.code_logits(model, ids, mask, codes, valid)
        if not torch.equal(actual, expected):
            raise ValueError("initial native function was not preserved exactly")
    opt = optimizer(model, plan)
    first = update(model, opt, renderer, items[0], plan, 1)
    second = update(model, opt, renderer, items[1], plan, 2)
    full = arm in ("native", "semantic")
    if full:
        save_checkpoint(model, opt, renderer, dest / "step-2", cfg, arm, 41, 2, full=True)
    third = update(model, opt, renderer, items[2], plan, 3)
    expected = evaluate(model, renderer, encoded, semantic.INTERVENTIONS)
    master, moments = (common.master_digest(opt), common.optimizer_digest(opt)) if full else (None, None)
    if not full:
        save_checkpoint(model, opt, renderer, dest / "step-3", cfg, arm, 41, 3)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    model = load(arm, 41, dest / ("step-2" if full else "step-3"))
    if full:
        opt = optimizer(model, plan)
        if common.restore_optimizer(model, opt, dest / "step-2") != 2:
            raise ValueError("wrong resume update")
        other = update(model, opt, renderer, items[2], plan, 3)
        if third != other or master != common.master_digest(opt) or moments != common.optimizer_digest(opt):
            raise ValueError("optimizer did not resume exactly")
        del opt; gc.collect(); torch.cuda.empty_cache()
    model.requires_grad_(False)
    actual = evaluate(model, renderer, encoded, semantic.INTERVENTIONS)
    if actual != expected or any(p.requires_grad for p in model.parameters()):
        raise ValueError("canonical frozen reload differs")
    # runtime audit uses canonical flags as well, matching ordinary evaluation.
    runtime = context.canonical_evaluation(model, runtime_audit, renderer, encoded)
    result = {"implementation": implementation(), "arm": arm, "initial_function_exact": True,
        "exact_reload_all_interventions": True, "exact_optimizer_resume": full, "first_update": first,
        "second_update": second, "third_update": third, "runtime": runtime}
    train.atomic_json(dest / "result.json", result)
    print("SMOKE_PASS", arm, flush=True)


def freeze(out):
    m = verify(out, frozen=False)
    if m["frozen"]:
        raise ValueError("already frozen")
    for arm in ARMS:
        r = read(out / f"smoke-{arm}/result.json")
        if r["implementation"] != implementation() or not r["initial_function_exact"] or not r["exact_reload_all_interventions"]:
            raise ValueError("stale real-model smoke")
        if arm in ("native", "semantic") and not r["exact_optimizer_resume"]:
            raise ValueError("missing optimizer resume validation")
    validation = read(out / "validation.json")
    if not validation["passed"] or validation["implementation"] != implementation():
        raise ValueError("unit validation missing or stale")
    for path in (out / "validation.json", out / "worker.py", Path("tests/test_v3_semantic_matching.py")):
        m["files"][str(path)] = train.sha256_file(path)
    (out / "source").mkdir()
    for path in implementation():
        shutil.copyfile(path, out / "source" / Path(path).name)
    m.update(frozen=True, implementation=implementation())
    train.atomic_json(out / "manifest.json", m)
    print("FROZEN", flush=True)


def fit(out, arm, seed):
    m = verify(out); plan = m["plan"]; dest = out / f"{arm}-seed{seed}"
    if (dest / "result.json").exists():
        return
    final = dest / "step-512"
    completed = (final / "complete.json").exists()
    if dest.exists() and not completed:
        archive = out / "attempts" / f"{dest.name}-{time.time_ns()}"
        archive.parent.mkdir(exist_ok=True); dest.rename(archive)
    dest.mkdir(exist_ok=True)
    train.atomic_json(out / "status.json", {"state": "starting", "fit": dest.name,
        "completed_fits": sum((out / f"{a}-seed{s}/result.json").exists() for a in ARMS for s in SEEDS), "total_fits": 12})
    renderer, cfg = common.search.renderer_for(common.search.BASE)
    train.seed_all(seed); model = load(arm, seed, final if completed else None)
    started = time.monotonic()
    if not completed:
        opt = optimizer(model, plan)
        dev(model, renderer, out, dest, 0)
        with (out / f"encoded-seed{seed}.jsonl").open() as cache, (dest / "train.jsonl").open("w") as log:
            for step, line in enumerate(cache, 1):
                item = json.loads(line)
                if item["step"] != step:
                    raise ValueError("wrong cached schedule")
                t0 = time.monotonic(); entry = update(model, opt, renderer, item, plan, step)
                entry.update(step=step, seconds_update=time.monotonic() - t0, input_target_sha256=item["input_target_sha256"])
                log.write(json.dumps(entry) + "\n"); log.flush()
                if step == 1 or step % 16 == 0:
                    train.atomic_json(out / "status.json", {"state": "running", "fit": dest.name,
                        "step": step, "steps": 512, "completed_fits": sum((out / f"{a}-seed{s}/result.json").exists() for a in ARMS for s in SEEDS), "total_fits": 12})
                    print("FIT", dest.name, step, entry["ce"], entry["branch_grad_norm"], flush=True)
                if step in (128, 256, 512):
                    dev(model, renderer, out, dest, step)
        if final.exists():
            final.rename(dest / f"step-512-incomplete-{time.time_ns()}")
        save_checkpoint(model, opt, renderer, final, cfg, arm, seed, 512)
        del opt
    common.verify_checkpoint(final)
    expected = list(data.lines(dest / "dev-512.jsonl"))
    del model; gc.collect(); torch.cuda.empty_cache()
    model = load(arm, seed, final)
    actual = evaluate(model, renderer, list(data.lines(out / "dev-encoded.jsonl")))["active"]
    if actual != expected:
        raise ValueError("final reload does not reproduce exact dev probabilities")
    trace = list(data.lines(dest / "train.jsonl"))
    hashes = [r["input_target_sha256"] for r in data.lines(out / f"encoded-seed{seed}.jsonl")]
    if len(trace) != 512 or [r["input_target_sha256"] for r in trace] != hashes:
        raise ValueError("incomplete or unmatched input/target stream")
    train.atomic_json(dest / "result.json", {"arm": arm, "seed": seed, "steps": 512, "examples_seen": 8192,
        "same_initial_base": str(BASE), "all_input_target_hashes_match": True, "exact_final_reload": True,
        "seconds_this_session": time.monotonic() - started, "dev": read(dest / "dev-512.json")})
    print("FIT_COMPLETE", dest.name, flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def contrast_vectors(predictions, primary=True):
    """Per-source seed x case x contrast arrays, with exact ID alignment."""
    first = predictions[SEEDS[0]]["native"]["active"]
    sources = defaultdict(list)
    for r in first:
        sources[r["source"]].append(r["id"])
    maps = {s: {a: {mode: {r["id"]: r for r in rr} for mode, rr in modes.items()}
                for a, modes in aa.items()} for s, aa in predictions.items()}
    names = ["semantic_minus_native", "semantic_minus_generic", "semantic_minus_self",
             "semantic_generic_gap_lost_with_branch_off", "semantic_minus_self_question_path_dependence"] if primary else ["semantic_minus_native"]
    result = []
    for source, ids in sorted(sources.items()):
        vals = []
        for seed in SEEDS:
            def v(arm, mode="active"):
                rr = maps[seed][arm][mode]
                if set(rr) != {r["id"] for r in first}:
                    raise ValueError("prediction IDs differ")
                return np.asarray([rr[k]["correct"] for k in ids], dtype=float)
            n, g, s, c = (v(a) for a in ARMS)
            vv = [s - n, s - g, s - c, s - g - v("semantic", "branch_off") + v("generic", "branch_off"),
                  s - v("semantic", "question_zero") - c + v("self", "question_zero")] if primary else [s - n]
            vals.append(np.stack(vv, axis=-1))
        result.append(np.stack(vals))
    return names, result


def summarize_contrasts(predictions, *, primary=True, draws=5000):
    names, sources = contrast_vectors(predictions, primary)
    point = np.mean([v.mean((0, 1)) for v in sources], axis=0)
    per_seed = np.mean([v.mean(1) for v in sources], axis=0)
    rng = np.random.default_rng(20261023)
    conditional, hierarchical = [], []
    for _ in range(draws):
        seeds = rng.integers(len(SEEDS), size=len(SEEDS))
        resampled = [v[:, rng.integers(v.shape[1], size=v.shape[1]), :] for v in sources]
        conditional.append(np.mean([v.mean((0, 1)) for v in resampled], axis=0))
        hierarchical.append(np.mean([v[seeds].mean((0, 1)) for v in resampled], axis=0))
    conditional, hierarchical = np.asarray(conditional), np.asarray(hierarchical)
    q = .025 / len(names)
    return {name: {"delta": float(point[i]), "per_seed": {str(s): float(per_seed[j, i]) for j, s in enumerate(SEEDS)},
        "conditional_document_95ci": np.quantile(conditional[:, i], [.025, .975]).tolist(),
        "hierarchical_seed_document_95ci": np.quantile(hierarchical[:, i], [.025, .975]).tolist(),
        "familywise_hierarchical_95ci": np.quantile(hierarchical[:, i], [q, 1 - q]).tolist(),
        "hierarchical_one_sided95_lower": float(np.quantile(hierarchical[:, i], .05))}
        for i, name in enumerate(names)}


def confirm(out):
    m = verify(out)
    fits = {f"{a}-seed{s}": (a, s, out / f"{a}-seed{s}/step-512") for s in SEEDS for a in ARMS}
    sealed = {}
    for name, (_, _, path) in fits.items():
        if not (out / name / "result.json").exists():
            raise ValueError("finish all twelve fixed-step fits before final testing")
        sealed[name] = common.verify_checkpoint(path)
    marker = out / "confirmation-started.json"
    if marker.exists() and read(marker) != sealed:
        raise ValueError("fixed final models changed")
    train.atomic_json(marker, sealed)
    renderer, _ = common.search.renderer_for(common.search.BASE)
    encoded = {k: list(data.lines(out / f"{k}-test-encoded.jsonl")) for k in ("primary", "retention")}
    dev_rows = list(data.lines(out / "dev-encoded.jsonl"))
    bench = sorted(dev_rows, key=lambda r: data.digest(["single-forward-cost-v1", r["id"]]))[:64]
    results, predictions = {}, {k: defaultdict(dict) for k in encoded}
    for name, (arm, seed, path) in fits.items():
        dest = out / "final-test" / name; dest.mkdir(parents=True, exist_ok=True)
        train.atomic_json(out / "status.json", {"state": "final_evaluation", "model": name, "completed_fits": 12, "total_fits": 12})
        modes = ("active",) if arm == "native" else semantic.INTERVENTIONS
        if not (dest / "result.json").exists():
            model = load(arm, seed, path)
            result = {}
            for split, rows in encoded.items():
                rr = evaluate(model, renderer, rows, modes)
                result[split] = {mode: common.search.metrics(v) for mode, v in rr.items()}
                for mode, values in rr.items():
                    common.search.save_records(dest / f"{split}-{mode}.jsonl", values)
            result["runtime"] = context.canonical_evaluation(model, runtime_audit, renderer, bench)
            train.atomic_json(dest / "result.json", result)
            del model; gc.collect(); torch.cuda.empty_cache()
        results[name] = read(dest / "result.json")
        for split in encoded:
            predictions[split][seed][arm] = {mode: list(data.lines(dest / f"{split}-{mode}.jsonl")) for mode in modes}
    effects = summarize_contrasts(predictions["primary"])
    retention = summarize_contrasts(predictions["retention"], primary=False)
    advance = all(r["familywise_hierarchical_95ci"][0] > 0 for r in effects.values())
    advance &= all(min(effects[n]["per_seed"].values()) > 0 for n in list(effects)[:3])
    advance &= retention["semantic_minus_native"]["hierarchical_one_sided95_lower"] > -.01
    means = {a: {k: float(np.mean([results[f"{a}-seed{s}"][k]["active"]["macro"] for s in SEEDS]))
                  for k in ("primary", "retention")} for a in ARMS}
    report = {"complete": True, "plan": m["plan"], "arm_mean_accuracy": means, "models": results,
        "primary_and_mechanism_contrasts": effects, "retention_contrast": retention,
        "exploratory_advance_gate": bool(advance), "interpretation_scope": m["plan"]["mechanism_scope"],
        "limits": m["plan"]["limits"]}
    train.atomic_json(out / "report.json", report)
    train.atomic_json(Path("wiki/project/single-forward-semantic-results.json"), report)
    train.atomic_json(out / "status.json", {"state": "complete", "completed_fits": 12, "total_fits": 12, "report": str(out / "report.json")})
    print("FINAL_STRUCTURE", means, effects, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("data", "prepare", "cache", "smoke", "freeze", "run", "confirm"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(); out = args.output
    if args.stage == "data":
        out.mkdir(parents=True, exist_ok=True)
        if (out / "inputs.json").exists():
            raise ValueError("preserve prepared data")
        data.build(out)
    elif args.stage == "prepare": prepare(out)
    elif args.stage == "cache": cache(out)
    elif args.stage == "smoke":
        for arm in (args.arm,) if args.arm else ARMS: smoke(out, arm)
    elif args.stage == "freeze": freeze(out)
    elif args.stage == "confirm": confirm(out)
    else:
        for seed in SEEDS:
            for arm in ARMS: fit(out, arm, seed)
        confirm(out)


if __name__ == "__main__":
    main()
