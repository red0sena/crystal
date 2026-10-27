"""Native/DUMA x gold CE/paired evidence-change loss, two paired seeds.

Warm-start the completed equal-history 64k CE checkpoints. Four aligned pairs
plus eight ordinary anchor cases per update in EVERY condition. A fixed small
mechanism trial; final unused-document tests open only after all fits finish.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from qwen_decision.v3 import evidence_pairs as ep, evidence_pair_data as data
from qwen_decision.v3 import long_training as common, retrofit_experiment as prior
from qwen_decision.v3 import retrofit_recovery as context, dual_matching as dual, modeling, train

OUT = Path("runs/evidence-pairs/factorial-v1")
ARMS = ["native_ce", "native_pair", "dual_ce", "dual_pair"]
BASES = {"native": common.OUT / "native-seed41/step-4096",
         "dual": prior.OUT / "dual-seed41/step-4096"}


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    paths = [Path(m.__file__) for m in (ep, data, context)] + [Path(__file__)]
    return prior.implementation() | {str(p): train.sha256_file(p) for p in paths}


def prepare(out):
    if (out / "manifest.json").exists():
        raise ValueError("preserve existing plan")
    inputs = read(out / "inputs.json")
    for name, sha in inputs["files"].items():
        if train.sha256_file(Path(name)) != sha:
            raise ValueError("prepared input changed")
    for split, minimum in (("pair-dev", 8), ("pair-test", 40)):
        rr = common.records.rows(out / f"{split}.jsonl")
        for source in ep.SOURCES:
            if sum(p["source"] == source and p["kind"] == "flip" for p in rr) < minimum:
                raise ValueError("insufficient source-specific evaluation pairs")
        if sum(p["kind"] == "invariant" for p in rr) < minimum:
            raise ValueError("insufficient invariant controls")
    p = {"experiment": "evidence-pair-factorial-v1", "arms": ARMS, "seeds": [41, 73],
         "steps": 512, "batch_size": 16, "pairs_per_update": 4, "anchors_per_update": 8,
         "pair_source_quota": {s: 2 for s in ep.SOURCES}, "body_lr": 3e-6, "matching_lr": 1e-5,
         "warmup": 32, "margin": 1., "pair_weight": .25, "snapshots": [256, 512],
         "dev_steps": [0, 128, 256, 512], "micro_batch": 4, "padded_token_budget": 4096,
         "architecture_initialization": {k: str(v) for k, v in BASES.items()},
         "initialization_scope": "each architecture starts from its completed seed41 64k/4096 CE checkpoint; both objectives and both new seeds share that architecture's exact initial weights",
         "training": "full used language backbone and tied input embeddings; DUMA matching readout; vision frozen; fresh FP32-master AdamW",
         "objective_ce": "mean gold CE across the exact same 16 cases, including the same four pairs",
         "objective_pair": "same CE + .25 * mean softplus(1 - [(za[ya]-za[yb])-(zb[ya]-zb[yb])]) across four flip pairs",
         "augmentation": "same option permutation and code assignment within each pair; exactly matched cached examples across all four arms per seed",
         "data": inputs["summary"], "max_training_tokens": 1536, "max_eval_tokens": 4096,
         "primary": "final flip-pair both-correct interaction (dual_pair-dual_ce)-(native_pair-native_ce), mean over the two paired seeds",
         "secondary": "within-architecture paired-loss effects; dual_pair-native_pair; invariant both-correct; fresh general source-macro accuracy and original-correct preservation",
         "final_testing": "unused document pairs and general retention evaluated only after all eight fixed-step models are frozen; no checkpoint selection or accuracy early stop",
         "mechanism_ablation": "DUMA passage inputs to matching zeroed after unchanged encoder; active/off use the exact same entire test batches; diagnostic does not remove evidence already in encoder states",
         "advance": "only exploratory candidate: adjusted interaction interval above0, interaction positive in both seeds, and observed dual_pair-native_pair general macro and invariant both-correct >=-.005; point guards are not statistical noninferiority; no automatic long training",
         "bootstrap": "2000 document-cluster draws within each source, paired across all four conditions and both seeds; Bonferroni four comparisons; conditional on these two seeds",
         "model_combination": False, "new_teacher_labels": False,
         "optimizer": "BF16 forward/backward, FP32-master AdamW betas(.9,.95), WD.01, global clip1; linear warmup then cosine to10%",
         "limits": ["specialized mechanism test on two evidence-change tasks, not general architectural superiority",
                    "two seeds do not characterize population training variance", "updates/examples matched, not FLOPs",
                    "seeds41/73 vary new training only; architecture initial checkpoints both use old seed41",
                    "CONDAQA documents are original passages, not entire Wikipedia articles",
                    "unknown original pretraining and semantic paraphrases cannot be exhaustively excluded"]}
    files = dict(inputs["files"])
    for name in ("inputs.json", "split-audit.json", "data-summary.json"):
        files[str(out / name)] = train.sha256_file(out / name)
    for path in BASES.values():
        for name in ("model.safetensors", "dual_matching.safetensors", "decision_architecture.json", "v3_config.json", "config.json"):
            f = path / name
            if f.exists():
                files[str(f)] = train.sha256_file(f)
    train.atomic_json(out / "plan.json", p)
    files[str(out / "plan.json")] = train.sha256_file(out / "plan.json")
    train.atomic_json(out / "manifest.json", {"frozen": False, "plan": p, "files": files})
    print("PREPARED", ARMS, p["seeds"], flush=True)


def verify(out, *, frozen=True):
    m = read(out / "manifest.json")
    if frozen and not m["frozen"]:
        raise ValueError("validate and freeze first")
    if m["frozen"] and m["implementation"] != implementation():
        raise ValueError("frozen implementation changed")
    for name, sha in m["files"].items():
        if train.sha256_file(Path(name)) != sha:
            raise ValueError(f"frozen input changed: {name}")
    return m


def cache(out):
    m = verify(out, frozen=False)
    if m["frozen"]:
        raise ValueError("preserve frozen cache")
    renderer, _ = common.search.renderer_for(common.search.BASE)
    training = common.records.rows(out / "train-pairs.jsonl")
    anchors = common.records.rows(out / "anchor-train.jsonl")
    by = {s: [p for p in training if p["source"] == s] for s in ep.SOURCES}
    for seed in m["plan"]["seeds"]:
        path = out / f"encoded-seed{seed}.jsonl"
        if path.exists():
            raise ValueError("preserve existing cache")
        queues, counters = {}, defaultdict(int)
        for s, pp in by.items():
            queues[s] = list(range(len(pp)))
            random.Random(int(common.search.digest([seed, s])[:16], 16)).shuffle(queues[s])
        ai = list(range(len(anchors)))
        random.Random(seed).shuffle(ai)
        with path.open("w") as f:
            for step in range(1, 513):
                selected = []
                for s in ep.SOURCES:
                    for j in range(2):
                        idx = queues[s][counters[s] % len(queues[s])]
                        counters[s] += 1
                        rng = random.Random(int(common.search.digest(["pair-view", seed, step, s, j])[:16], 16))
                        selected.append(ep.encode_pair(renderer, by[s][idx], rng))
                aa = [ep.encode(renderer, anchors[i], random.Random(int(common.search.digest(["anchor-view", seed, step, j])[:16], 16)))
                      for j, i in enumerate(ai[(step - 1) * 8:step * 8])]
                flat = [r for p in selected for r in p["rows"]] + aa
                if len(flat) != 16 or max(len(r["input_ids"]) for r in flat) > 1536:
                    raise ValueError("invalid matched training batch")
                digest = common.search.digest([[r["id"], r["input_ids"], r["code_ids"], r["target"]] for r in flat])
                f.write(json.dumps({"step": step, "pairs": selected, "anchors": aa, "input_target_sha256": digest}, ensure_ascii=False) + "\n")
        m["files"][str(path)] = train.sha256_file(path)
    for split in ("pair-dev", "pair-test", "retention-test"):
        raw = common.records.rows(out / f"{split}.jsonl")
        if split.startswith("pair-"):
            unique = {p[k]["id"]: p[k] for p in raw for k in ("a", "b")}
            raw = [unique[k] for k in sorted(unique)]
        encoded = [ep.encode(renderer, r) for r in raw]
        path = out / f"{split}-encoded.jsonl"
        common.search.save_records(path, encoded)
        m["files"][str(path)] = train.sha256_file(path)
    train.atomic_json(out / "manifest.json", m)
    print("CACHE_READY", flush=True)


def load(arch, path=None):
    torch.use_deterministic_algorithms(True)
    path = path or BASES[arch]
    model = common.load_arm("native", path) if arch == "native" else context.load("dual", path)
    model._retrofit_evaluation_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    return model


def optimizer(model, arch, plan):
    return train.MasterAdamW(model, plan["body_lr"]) if arch == "native" else prior.optimizer(model, "dual", plan)


def logits(model, rows, renderer, arch, *, disable_evidence=False):
    ids, mask, codes, valid, target = train.collate(rows, renderer.tokenizer.pad_token_id)
    value = (modeling.code_logits(model, ids, mask, codes, valid) if arch == "native" else
             dual.logits(model, ids, mask, rows, disable_evidence=disable_evidence))
    return value, target


def update(model, opt, renderer, item, arch, objective, plan, step):
    model.train()
    train.set_checkpointing(model, True)
    opt.zero_grad()
    count = sum(len(p["rows"]) for p in item["pairs"]) + len(item["anchors"])
    ce_sum, pair_sum, delta_sum = 0., 0., 0.
    for pair in item["pairs"]:
        z, target = logits(model, pair["rows"], renderer, arch)
        ce = train.soft_ce(z, target).sum() / count
        cf, delta = ep.margin_loss(z, target, plan["margin"])
        loss = ce + (plan["pair_weight"] * cf / len(item["pairs"]) if objective == "pair" else 0.)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite paired objective")
        loss.backward()
        ce_sum += ce.item()
        pair_sum += cf.item() / len(item["pairs"])
        delta_sum += delta.item() / len(item["pairs"])
    for group in train.microbatches(item["anchors"], common.BATCH):
        z, target = logits(model, group, renderer, arch)
        ce = train.soft_ce(z, target).sum() / count
        if not torch.isfinite(ce):
            raise ValueError("nonfinite anchor loss")
        ce.backward()
        ce_sum += ce.item()
    if step in (1, 2, 3):
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise ValueError("missing or nonfinite trainable gradient")
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradients")
    scale = min(step / plan["warmup"], 1.) * (.1 + .45 * (1 + math.cos(
        math.pi * max(0, step - plan["warmup"]) / (plan["steps"] - plan["warmup"]))))
    for g in opt.param_groups:
        g["lr"] = plan["body_lr"] * g["lr_scale"] * scale
    opt.step()
    return {"ce": ce_sum, "pair_loss": pair_sum, "pair_delta": delta_sum, "grad_norm": norm,
            "loss": ce_sum + (plan["pair_weight"] * pair_sum if objective == "pair" else 0.)}


@torch.no_grad()
def _evaluate(model, renderer, encoded, arch, **kwargs):
    model.eval()
    train.set_checkpointing(model, False)
    predictions = {}
    for group in train.microbatches(encoded, common.BATCH):
        z, _ = logits(model, group, renderer, arch, **kwargs)
        for r, value in zip(group, z.softmax(-1).cpu()):
            p = value[:len(r["target"])].tolist()
            choice = max(range(len(p)), key=p.__getitem__)
            predictions[r["id"]] = {"id": r["id"], "source": r["source"], "group": r["group"],
                "prediction": choice, "correct": r["target"][choice] >= max(r["target"]) - 1e-6, "probabilities": p}
    rr = [predictions[r["id"]] for r in encoded]
    return common.search.metrics(rr), rr


def evaluate(model, renderer, encoded, arch, **kwargs):
    return context.canonical_evaluation(model, _evaluate, renderer, encoded, arch, **kwargs)


def dev(model, renderer, out, dest, arch, step):
    encoded = common.records.rows(out / "pair-dev-encoded.jsonl")
    _, rr = evaluate(model, renderer, encoded, arch)
    outcomes = ep.pair_outcomes(common.records.rows(out / "pair-dev.jsonl"), {r["id"]: r for r in rr})
    metrics = ep.response_metrics(outcomes)
    train.atomic_json(dest / f"pair-dev-{step}.json", metrics)
    common.search.save_records(dest / f"pair-dev-{step}.jsonl", rr)
    print("PAIR_DEV", dest.name, step, metrics["flip"]["macro_both_correct"], flush=True)


def save_checkpoint(model, opt, renderer, path, arch, cfg, step, *, full):
    if path.exists():
        raise ValueError("preserve existing checkpoint")
    opt.zero_grad()
    (modeling.save if arch == "native" else dual.save)(model, renderer.tokenizer, path,
        cfg | {"experiment": "evidence-pair-factorial-v1", "steps": step, "trained_from": str(BASES[arch])})
    if full:
        payload = common.training_state(model, opt, step)
        torch.save(payload, path / "optimizer.pt.tmp")
        (path / "optimizer.pt.tmp").rename(path / "optimizer.pt")
        del payload
        gc.collect()
    files = {p.name: train.sha256_file(p) for p in path.iterdir() if p.suffix == ".safetensors" or
             p.name in ("optimizer.pt", "config.json", "v3_config.json", "decision_architecture.json")}
    train.atomic_json(path / "complete.json", {"step": step, "full_optimizer": full, "files": files})


def smoke(out, arch):
    m = verify(out, frozen=False)
    plan = m["plan"]
    dest = out / f"smoke-{arch}"
    if dest.exists():
        raise ValueError("preserve smoke evidence")
    dest.mkdir()
    items = []
    with (out / "encoded-seed41.jsonl").open() as f:
        for _ in range(4):
            items.append(json.loads(next(f)))
    renderer, cfg = common.search.renderer_for(common.search.BASE)
    train.seed_all(41)
    model = load(arch)
    opt = optimizer(model, arch, plan)
    first = update(model, opt, renderer, items[0], arch, "pair", plan, 1)
    update(model, opt, renderer, items[1], arch, "ce", plan, 2)
    save_checkpoint(model, opt, renderer, dest / "step-2", arch, cfg, 2, full=True)
    expected_losses = [update(model, opt, renderer, items[s - 1], arch, "pair" if s == 3 else "ce", plan, s) for s in (3, 4)]
    expected_master = common.master_digest(opt)
    expected_optimizer = common.optimizer_digest(opt)
    encoded = common.records.rows(out / "pair-dev-encoded.jsonl")[:12]
    _, expected = evaluate(model, renderer, encoded, arch)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    model = load(arch, dest / "step-2")
    opt = optimizer(model, arch, plan)
    if common.restore_optimizer(model, opt, dest / "step-2") != 2:
        raise ValueError("wrong resume step")
    actual_losses = [update(model, opt, renderer, items[s - 1], arch, "pair" if s == 3 else "ce", plan, s) for s in (3, 4)]
    if expected_losses != actual_losses or expected_master != common.master_digest(opt) or expected_optimizer != common.optimizer_digest(opt):
        raise ValueError("CE/pair optimizer updates did not resume exactly")
    model.requires_grad_(False)
    _, actual = evaluate(model, renderer, encoded, arch)
    if actual != expected or any(p.requires_grad for p in model.parameters()):
        raise ValueError("canonical frozen-caller evaluation differs")
    r = {"implementation": implementation(), "architecture": arch, "first_update": first,
         "exact_optimizer_resume": True, "both_objectives_exercised": True, "frozen_caller_exact": True}
    train.atomic_json(dest / "result.json", r)
    print("SMOKE_PASS", arch, flush=True)


def freeze(out):
    m = verify(out, frozen=False)
    if m["frozen"]:
        raise ValueError("already frozen")
    for arch in BASES:
        r = read(out / f"smoke-{arch}/result.json")
        if r["implementation"] != implementation() or not r["exact_optimizer_resume"] or not r["frozen_caller_exact"]:
            raise ValueError("missing or stale real-model validation")
    r = read(out / "validation.json")
    if not r["passed"] or r["implementation"] != implementation():
        raise ValueError("missing unit validation")
    for path in (out / "worker.py", out / "validation.json", Path("tests/test_v3_evidence_pairs.py")):
        m["files"][str(path)] = train.sha256_file(path)
    m.update(frozen=True, implementation=implementation())
    (out / "source").mkdir()
    for path in m["implementation"]:
        shutil.copyfile(path, out / "source" / Path(path).name)
    train.atomic_json(out / "manifest.json", m)
    print("FROZEN", flush=True)


def fit(out, arm, seed):
    m = verify(out)
    plan = m["plan"]
    arch, objective = arm.split("_")
    dest = out / f"{arm}-seed{seed}"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return
    train.atomic_json(out / "status.json", {"state": "starting", "fit": dest.name})
    renderer, cfg = common.search.renderer_for(common.search.BASE)
    final = dest / "step-512"
    finals = (final / "complete.json").exists()
    resume = dest / "step-256"
    resumes = (resume / "complete.json").exists()
    train.seed_all(seed)
    model = load(arch, final if finals else resume if resumes else None)
    opt = None if finals else optimizer(model, arch, plan)
    start = 512 if finals else common.restore_optimizer(model, opt, resume) if resumes else 0
    logpath = dest / "train.jsonl"
    trace = common.records.rows(logpath) if logpath.exists() else []
    if len(trace) < start:
        raise ValueError("checkpoint ahead of trace")
    if len(trace) > start:
        shutil.copyfile(logpath, dest / f"uncommitted-{time.time_ns()}.jsonl")
        common.search.save_records(logpath, trace[:start])
    if not start:
        dev(model, renderer, out, dest, arch, 0)
    started = time.monotonic()
    with (out / f"encoded-seed{seed}.jsonl").open() as cache, logpath.open("a") as log:
        for step, line in enumerate(cache, 1):
            if step <= start:
                continue
            item = json.loads(line)
            if item["step"] != step:
                raise ValueError("cache schedule differs")
            t0 = time.monotonic()
            entry = update(model, opt, renderer, item, arch, objective, plan, step)
            entry |= {"step": step, "input_target_sha256": item["input_target_sha256"],
                      "seconds_update": time.monotonic() - t0, "seconds_session": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step == 1 or step % 16 == 0:
                train.atomic_json(out / "status.json", {"state": "running", "fit": dest.name, "step": step, "steps": 512,
                    "completed_fits": sum((out / f"{a}-seed{s}/result.json").exists() for a in ARMS for s in (41, 73))})
                print("FIT", dest.name, step, entry["ce"], entry["pair_loss"], flush=True)
            if step in (128, 256, 512):
                dev(model, renderer, out, dest, arch, step)
            if step in (256, 512):
                path = dest / f"step-{step}"
                if path.exists():
                    path.rename(dest / f"step-{step}.incomplete-{time.time_ns()}")
                save_checkpoint(model, opt, renderer, path, arch, cfg, step, full=step == 256)
    common.verify_checkpoint(final)
    expected = common.records.rows(dest / "pair-dev-512.jsonl")
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    model = load(arch, final).requires_grad_(False)
    _, actual = evaluate(model, renderer, common.records.rows(out / "pair-dev-encoded.jsonl"), arch)
    if expected != actual:
        raise ValueError("final canonical reload differs")
    trace = common.records.rows(logpath)
    cache_hashes = [json.loads(s)["input_target_sha256"] for s in (out / f"encoded-seed{seed}.jsonl").read_text().splitlines()]
    if len(trace) != 512 or [r["input_target_sha256"] for r in trace] != cache_hashes:
        raise ValueError("incomplete or unmatched fit")
    train.atomic_json(dest / "result.json", {"architecture": arch, "objective": objective, "seed": seed,
        "steps": 512, "examples_seen": 8192, "all_input_target_hashes_match": True,
        "exact_final_reload": True, "seconds_session": time.monotonic() - started,
        "dev": read(dest / "pair-dev-512.json")})
    print("FIT_COMPLETE", dest.name, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def factorial_bootstrap(by_seed, draws=2000):
    groups = defaultdict(lambda: defaultdict(list))
    first = by_seed[41][ARMS[0]]
    maps = {s: {a: {r["id"]: r for r in rr} for a, rr in arms.items()} for s, arms in by_seed.items()}
    for r in first:
        if r["kind"] == "flip":
            groups[r["source"]][r["group"]].append(r["id"])
    sources = []
    for source, docs in groups.items():
        ids = [k for rr in docs.values() for k in rr]
        positions = {k: i for i, k in enumerate(ids)}
        clusters = [np.asarray([positions[k] for k in rr]) for rr in docs.values()]
        values = np.asarray([[[maps[s][a][k]["correct"] for k in ids] for a in ARMS] for s in (41, 73)], dtype=float)
        sources.append((values, clusters))
    rng = np.random.default_rng(20261022)
    samples = []
    for _ in range(draws):
        samples.append(np.mean([values[:, :, np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])].mean((0, 2))
                                for values, clusters in sources], axis=0))
    samples = np.asarray(samples)
    point = np.mean([v.mean((0, 2)) for v, _ in sources], axis=0)
    contrasts = {"native_pair_effect": np.array([-1, 1, 0, 0]), "dual_pair_effect": np.array([0, 0, -1, 1]),
                 "interaction": np.array([1, -1, -1, 1]), "dual_pair_minus_native_pair": np.array([0, -1, 0, 1])}
    result = {}
    for name, vector in contrasts.items():
        x = samples @ vector
        result[name] = {"delta": float(point @ vector), "paired_document_95ci": np.quantile(x, [.025, .975]).tolist(),
                        "familywise_95ci_four_comparisons": np.quantile(x, [.00625, .99375]).tolist()}
    return result


def confirm(out):
    m = verify(out)
    fits = {f"{a}-seed{s}": out / f"{a}-seed{s}/step-512" for s in (41, 73) for a in ARMS}
    for name in fits:
        if not (out / name / "result.json").exists():
            raise ValueError("finish every fixed-step fit before final testing")
        common.verify_checkpoint(fits[name])
    identity = {name: read(path / "complete.json") for name, path in fits.items()}
    marker = out / "confirmation-started.json"
    sealed = {"final_checkpoints": identity, "pair_test_sha256": train.sha256_file(out / "pair-test.jsonl"),
              "retention_test_sha256": train.sha256_file(out / "retention-test.jsonl")}
    if marker.exists():
        if read(marker) != sealed:
            raise ValueError("fixed final models or tests changed")
    else:
        train.atomic_json(marker, sealed)
    renderer, _ = common.search.renderer_for(common.search.BASE)
    test = common.records.rows(out / "pair-test.jsonl")
    encoded = common.records.rows(out / "pair-test-encoded.jsonl")
    general = common.records.rows(out / "retention-test-encoded.jsonl")
    results, outcomes, retention = {}, defaultdict(dict), {}
    for name, path in ({"baseline_native": BASES["native"], "baseline_dual": BASES["dual"]} | fits).items():
        arch = "dual" if "dual" in name else "native"
        dest = out / "final-test" / name
        dest.mkdir(parents=True, exist_ok=True)
        train.atomic_json(out / "status.json", {"state": "final_evaluation", "model": name, "completed_fits": 8})
        if not (dest / "result.json").exists():
            model = load(arch, path).requires_grad_(False)
            metric, rr = evaluate(model, renderer, encoded, arch)
            general_metric, gg = evaluate(model, renderer, general, arch)
            common.search.save_records(dest / "pairs.jsonl", rr)
            common.search.save_records(dest / "retention.jsonl", gg)
            response = ep.response_metrics(ep.pair_outcomes(test, {r["id"]: r for r in rr}))
            ablated = None
            if arch == "dual" and not name.startswith("baseline"):
                _, off = evaluate(model, renderer, encoded, arch, disable_evidence=True)
                common.search.save_records(dest / "matching-evidence-zero.jsonl", off)
                ablated = ep.response_metrics(ep.pair_outcomes(test, {r["id"]: r for r in off}))
            train.atomic_json(dest / "result.json", {"response": response, "case_metric": metric,
                "retention": general_metric, "matching_evidence_zero": ablated})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        results[name] = read(dest / "result.json")
        rr = common.records.rows(dest / "pairs.jsonl")
        retention[name] = {r["id"]: r for r in common.records.rows(dest / "retention.jsonl")}
        if not name.startswith("baseline"):
            arm, seed = name.split("-seed")
            outcomes[int(seed)][arm] = ep.pair_outcomes(test, {r["id"]: r for r in rr})
    base = retention["baseline_native"]
    correct = {k for k, v in base.items() if v["correct"]}
    for name, rr in retention.items():
        results[name]["preserve_original_correct"] = sum(rr[k]["correct"] for k in correct) / len(correct)
        results[name]["original_correct_regressions"] = sum(not rr[k]["correct"] for k in correct)
    factorial = factorial_bootstrap(outcomes)
    interactions = {}
    for seed in (41, 73):
        values = {a: results[f"{a}-seed{seed}"]["response"]["flip"]["macro_both_correct"] for a in ARMS}
        interactions[seed] = values["dual_pair"] - values["dual_ce"] - values["native_pair"] + values["native_ce"]
    retention_delta = np.mean([results[f"dual_pair-seed{s}"]["retention"]["macro"] -
                               results[f"native_pair-seed{s}"]["retention"]["macro"] for s in (41, 73)])
    invariant_delta = np.mean([results[f"dual_pair-seed{s}"]["response"]["invariant"]["macro_both_correct"] -
                               results[f"native_pair-seed{s}"]["response"]["invariant"]["macro_both_correct"] for s in (41, 73)])
    ablated_gain = np.mean([results[f"dual_pair-seed{s}"]["matching_evidence_zero"]["flip"]["macro_both_correct"] -
                            results[f"dual_ce-seed{s}"]["matching_evidence_zero"]["flip"]["macro_both_correct"] for s in (41, 73)])
    advance = (factorial["interaction"]["familywise_95ci_four_comparisons"][0] > 0 and
               min(interactions.values()) > 0 and retention_delta >= -.005 and invariant_delta >= -.005)
    report = {"complete": True, "plan": m["plan"], "models": results, "factorial": factorial,
              "per_seed_interaction": interactions, "observed_retention_delta": float(retention_delta),
              "observed_invariant_delta": float(invariant_delta), "exploratory_advance_gate": bool(advance),
              "matching_path_diagnostic": {"active_paired_loss_effect": factorial["dual_pair_effect"]["delta"],
                  "passage_zero_paired_loss_effect": float(ablated_gain),
                  "difference": float(factorial["dual_pair_effect"]["delta"] - ablated_gain),
                  "scope": m["plan"]["mechanism_ablation"]},
              "new_test_scope": read(out / "split-audit.json"),
              "limits": m["plan"]["limits"]}
    train.atomic_json(out / "report.json", report)
    train.atomic_json(Path("wiki/project/evidence-pairs-results.json"), report)
    train.atomic_json(out / "status.json", {"state": "complete", "completed_fits": 8, "report": str(out / "report.json")})
    print("FINAL_FACTORIAL", report["factorial"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cache", "smoke", "freeze", "run", "confirm"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--arch", choices=list(BASES))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "cache":
        cache(args.output)
    elif args.stage == "smoke":
        for arch in [args.arch] if args.arch else BASES:
            smoke(args.output, arch)
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "confirm":
        confirm(args.output)
    else:
        for seed in (41, 73):
            for arm in ARMS:
                fit(args.output, arm, seed)
        confirm(args.output)


if __name__ == "__main__":
    main()
