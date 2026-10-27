"""Matched DUMA-style readout and Theseus-style gradual replacement trials.

Existing immutable native and abrupt-BDH controls are reused after verifying
their data, recipe, per-update hashes and final checkpoints. Three new fits:
dual matching, parameter-matched self matching, and gradual BDH replacement.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from qwen_decision.v3 import long_training as common, dual_matching as dual
from qwen_decision.v3 import progressive_replacement as progressive, bdh_memory as bdh
from qwen_decision.v3 import modeling, train, architecture_search as search
from qwen_decision.v3 import readout_experiment as records

OUT = Path("runs/retrofit/dual-theseus-64k-v1")
REFERENCE = common.OUT
ARMS = ["dual", "gradual", "self"]


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def implementation():
    paths = [Path(dual.__file__), Path(progressive.__file__), Path(__file__)]
    return common.implementation() | {str(p): train.sha256_file(p) for p in paths}


def prepare(out):
    if out.exists():
        raise ValueError("preserve previous experiments")
    previous = common.verify(REFERENCE)
    for arm in ("native", "bdh"):
        result = read(REFERENCE / f"{arm}-seed41/result.json")
        if result["steps"] != 4096 or not result["all_inputs_targets_match_native"]:
            raise ValueError("incomplete reference fit")
        common.verify_checkpoint(REFERENCE / f"{arm}-seed41/step-4096")
    plan = previous["plan"] | {
        "experiment": "dual-theseus-64k-v1", "arms": ARMS, "seeds": [41],
        "matching_lr": 1e-4, "matching_heads": 8, "matching_layers": 2,
        "matching_parameters": 4196352, "transition_steps": 1024, "initial_successor_probability": .1,
        "dual": "one unchanged joint Qwen encoding; two layers of shared bidirectional passage <-> question+candidate token attention; concatenate masked means; shared scalar score",
        "self": "exactly the same new weights, parameter count, spans and scalar readout; self attention on concatenated passage/question/candidate instead of cross attention",
        "gradual": "upper8 native predecessor frozen; route to existing 3-cycle BDH successor with linearly increasing probability .1->1 over updates1:1024; successor only thereafter",
        "training": "all used language parameters and tied input embedding train; vision frozen; gradual predecessor upper8 frozen; absent successor gradient skips its Adam and decay on predecessor updates",
        "theseus_adaptation": "coarse single-module replacement; common prefix and head train throughout, unlike original Theseus frozen embedding/head phase; no distillation or extra data",
        "duma_adaptation": "one joint encoding of all choices, unlike original DUMA separate candidate encodings; full width1024 MHA, no FFN/residual/gate; code LM readout replaced",
        "reference_native": str(REFERENCE / "native-seed41"),
        "reference_abrupt_bdh": str(REFERENCE / "bdh-seed41"),
        "primary": "fixed4096 updates: dual-native; dual-self; gradual-abrupt_bdh; gradual-native",
        "reference_reuse": "previous exact seed41 cached input/target stream, optimizer and cosine schedule; no fresh control training claimed",
        "snapshots": [1024, 2048, 4096], "probe_steps": [0, 256, 1024, 2048, 4096],
        "successor_route": "SHA256(seed, update); one path for entire update; deterministic across checkpoint replay/resume; evaluation always successor",
        "new_confirmation_cases_scored": 0, "no_best_snapshot_reselection": True,
        "limits": ["one training seed; exploratory exposed selection, not independent confirmation",
                   "same examples and optimizer updates, not equal FLOPs or parameter counts",
                   "DUMA and Theseus inspired adaptations, not literal reproductions",
                   "Theseus addresses conversion damage, not an expectation of exceeding the original larger backbone"],
    }
    out.mkdir(parents=True)
    train.atomic_json(out / "plan.json", plan)
    files = [REFERENCE / n for n in ["manifest.json", "plan.json", "train.jsonl", "selection.jsonl",
                                     "train-probe.jsonl", "encoded-seed41.jsonl"]]
    for arm in ("native", "bdh"):
        dest = REFERENCE / f"{arm}-seed41"
        files += [dest / n for n in ["result.json", "train.jsonl"]]
        files += [dest / f"selection-{step}.{ext}" for step in [1024, 2048, 4096] for ext in ("json", "jsonl")]
        files += [dest / "step-4096/complete.json"]
    train.atomic_json(out / "manifest.json", {
        "plan": plan, "frozen": False, "base_sha256": previous["base_sha256"],
        "plan_sha256": train.sha256_file(out / "plan.json"),
        "files": {str(p): train.sha256_file(p) for p in files},
        "inherited_train_selection_audit": previous["inherited_train_selection_audit"],
        "new_confirmation_cases_scored": 0,
    })
    print("PREPARED", out, ARMS, flush=True)


def verify(out, *, frozen=True):
    manifest = read(out / "manifest.json")
    if frozen and not manifest["frozen"]:
        raise ValueError("validate and freeze before training")
    if manifest["frozen"] and manifest["implementation"] != implementation():
        raise ValueError("frozen implementation changed")
    if train.sha256_file(out / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("plan changed")
    if train.sha256_file(search.BASE / "model.safetensors") != manifest["base_sha256"]:
        raise ValueError("base changed")
    for filename, sha in manifest["files"].items():
        if train.sha256_file(Path(filename)) != sha:
            raise ValueError(f"input/reference changed: {filename}")
    return manifest


def cache(out):
    manifest = verify(out, frozen=False)
    if manifest["frozen"]:
        raise ValueError("preserve frozen cache")
    renderer, _ = search.renderer_for(search.BASE)
    raw = {r["id"]: r for r in records.rows(REFERENCE / "train.jsonl")}
    path = out / "encoded-seed41.jsonl"
    if path.exists():
        raise ValueError("preserve existing cache")
    reference = records.rows(REFERENCE / "native-seed41/train.jsonl")
    counts = {"rows": 0, "max_passage_tokens": 0, "max_qa_tokens": 0, "max_choices": 0}
    with (REFERENCE / "encoded-seed41.jsonl").open() as source, path.with_suffix(".tmp").open("w") as dest:
        for step, line in enumerate(source, 1):
            item = json.loads(line)
            if item["step"] != step or len(item["rows"]) != 16:
                raise ValueError("invalid original cache")
            rr = item["rows"]
            digest = search.digest([[r["input_ids"], r["code_ids"], r["target"]] for r in rr])
            if digest != reference[step - 1]["input_target_sha256"]:
                raise ValueError("cache differs from native training")
            for r in rr:
                r.update(dual.content_spans(renderer, raw[r["id"]], r))
                counts["rows"] += 1
                counts["max_passage_tokens"] = max(counts["max_passage_tokens"], len(r["passage_positions"]))
                counts["max_qa_tokens"] = max(counts["max_qa_tokens"], len(r["question_positions"]) + max(map(len, r["candidate_positions"])))
                counts["max_choices"] = max(counts["max_choices"], len(r["target"]))
            dest.write(json.dumps(item, ensure_ascii=False) + "\n")
            if step % 512 == 0:
                print("CACHE", step, counts, flush=True)
    if step != 4096 or counts["rows"] != 65536:
        raise ValueError("incomplete cache")
    path.with_suffix(".tmp").rename(path)
    manifest["files"][str(path)] = train.sha256_file(path)
    for split in ("selection", "train-probe"):
        rr = []
        for r in records.rows(REFERENCE / f"{split}.jsonl"):
            e = common.encoding.encode(renderer, r)
            rr.append(e | dual.content_spans(renderer, r, e))
        target = out / f"{split}-encoded.jsonl"
        search.save_records(target, rr)
        manifest["files"][str(target)] = train.sha256_file(target)
    train.atomic_json(out / "manifest.json", manifest)
    train.atomic_json(out / "input-audit.json", counts | {"all_65536_inputs_targets_match_native": True,
                                                       "all_content_spans_nonempty": True})
    print("CACHE_READY", counts, flush=True)


def load(arm, path=search.BASE):
    torch.use_deterministic_algorithms(True)
    if arm in ("dual", "self"):
        model = dual.load(path)
        text = model.model.language_model
        if not hasattr(text, "dual_matching"):
            dual.install(text, arm)
        if text.dual_matching.mode != arm:
            raise ValueError("matching mode differs")
        model.requires_grad_(False)
        text.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model = bdh.load(path)
        text = model.model.language_model
        if not hasattr(text, "bdh_memory"):
            bdh.install(text, 16, 4, 3)
        model.requires_grad_(False)
        text.embed_tokens.requires_grad_(True)
        text.norm.requires_grad_(True)
        text.bdh_memory.requires_grad_(True)
        model.lm_head.requires_grad_(True)
        for layer in text.layers[:16]:
            layer.requires_grad_(True)
        # Retain the frozen predecessor ON GPU while conversion is possible.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def optimizer(model, arm, plan):
    if arm == "gradual":
        return common.optimizer(model, "bdh", plan)
    opt = train.MasterAdamW(model, plan["body_lr"])
    ids = {id(p) for p in model.model.language_model.dual_matching.parameters()}
    masters = [m for q, m in opt.pairs if id(q) in ids]
    mids = {id(m) for m in masters}
    for group in opt.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in mids]
    opt.optimizer.add_param_group({"params": masters, "weight_decay": .01,
                                  "lr_scale": plan["matching_lr"] / plan["body_lr"]})
    return opt


def logits_for(model, rows, pad, arm, *, successor=True, disable_evidence=False):
    ids, mask, codes, valid, targets = train.collate(rows, pad)
    if arm == "gradual":
        value = progressive.logits(model, ids, mask, codes, valid, successor=successor)
    else:
        value = dual.logits(model, ids, mask, rows, disable_evidence=disable_evidence)
    return value, targets


def update(model, opt, renderer, rows, arm, plan, step, *, force_successor=None):
    successor = True if arm != "gradual" else progressive.use_successor(
        41, step, plan["transition_steps"], plan["initial_successor_probability"])
    if force_successor is not None:
        successor = force_successor
    model.train()
    train.set_checkpointing(model, True)
    opt.zero_grad()
    total = 0.
    for group in train.microbatches(rows, common.BATCH):
        values, targets = logits_for(model, group, renderer.tokenizer.pad_token_id, arm, successor=successor)
        loss = train.soft_ce(values, targets).sum() / len(rows)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite CE")
        loss.backward()
        total += loss.item()
    if step in (1, 2, 3, 1024, 1025):
        missing, nonfinite = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            dormant = arm == "gradual" and not successor and ".bdh_memory." in name
            if p.grad is None and not dormant:
                missing.append(name)
            elif p.grad is not None and not torch.isfinite(p.grad).all():
                nonfinite.append(name)
        if missing or nonfinite:
            raise ValueError(f"invalid gradients: missing={missing}, nonfinite={nonfinite}")
    # Record actual new-module activity; the old route intentionally leaves it dormant.
    module = (model.model.language_model.bdh_memory if arm == "gradual" else
              model.model.language_model.dual_matching)
    activity = {k: None if p.grad is None else p.grad.float().norm().item()
                for k, p in module.named_parameters()}
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    scale = min(step / plan["warmup"], 1.) * (.1 + .45 * (1 + math.cos(
        math.pi * max(0, step - plan["warmup"]) / (plan["steps"] - plan["warmup"]))))
    for group in opt.param_groups:
        group["lr"] = plan["body_lr"] * group["lr_scale"] * scale
    opt.step()
    return total, norm, {"successor": successor, "new_module_gradient_norms": activity}


@torch.no_grad()
def evaluate(model, renderer, encoded, arm, **kwargs):
    model.eval()
    train.set_checkpointing(model, False)
    predictions = {}
    started = time.monotonic()
    for group in train.microbatches(encoded, common.BATCH):
        value, _ = logits_for(model, group, renderer.tokenizer.pad_token_id, arm, **kwargs)
        for row, prob in zip(group, value.softmax(-1).cpu()):
            p = prob[:len(row["target"])].tolist()
            choice = max(range(len(p)), key=p.__getitem__)
            predictions[row["id"]] = {"id": row["id"], "source": row["source"], "group": row["group"],
                "prediction": choice, "correct": row["target"][choice] >= max(row["target"]) - 1e-6,
                "probabilities": p}
    result = [predictions[r["id"]] for r in encoded]
    return search.metrics(result) | {"seconds": time.monotonic() - started}, result


def save_checkpoint(model, opt, renderer, path, arm, cfg, step):
    if path.exists():
        raise ValueError("preserve checkpoint")
    opt.zero_grad()
    config = cfg | {"steps": step, "trained_from": str(search.BASE),
                    "experiment": "dual-theseus-64k-v1", "arm": arm}
    (bdh.save if arm == "gradual" else dual.save)(model, renderer.tokenizer, path, config)
    state = common.training_state(model, opt, step)
    torch.save(state, path / "optimizer.pt.tmp")
    (path / "optimizer.pt.tmp").rename(path / "optimizer.pt")
    del state
    gc.collect()
    files = {p.name: train.sha256_file(p) for p in path.iterdir()
             if p.suffix == ".safetensors" or p.name in ("optimizer.pt", "config.json", "v3_config.json", "decision_architecture.json")}
    train.atomic_json(path / "complete.json", {"step": step, "arm": arm, "files": files})


def smoke(out, arm):
    manifest = verify(out, frozen=False)
    plan = manifest["plan"]
    dest = out / f"smoke-{arm}"
    if dest.exists():
        raise ValueError("preserve smoke evidence")
    dest.mkdir()
    renderer, cfg = search.renderer_for(search.BASE)
    batches, worst = [], []
    with (out / "encoded-seed41.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            if len(batches) < 3:
                batches.append(item["rows"])
            candidates = worst + item["rows"]
            longest = max(candidates, key=lambda r: len(r["input_ids"]))
            most_choices = max(candidates, key=lambda r: len(r["target"]))
            worst = [longest, most_choices]
    probe = records.rows(out / "train-probe-encoded.jsonl")[:12]
    train.seed_all(41)
    model = load(arm)
    opt = optimizer(model, arm, plan)
    torch.cuda.reset_peak_memory_stats()
    equivalence = None
    if arm == "gradual":
        group = batches[0][:2]
        ids, mask, codes, valid, _ = train.collate(group, renderer.tokenizer.pad_token_id)
        model.eval()
        with torch.no_grad():
            a = modeling.code_logits(model, ids, mask, codes, valid)
            b, _ = logits_for(model, group, renderer.tokenizer.pad_token_id, arm, successor=False)
            if not torch.equal(a, b):
                raise ValueError("predecessor route differs from native")
            a = bdh.logits(model, ids, mask, codes, valid)
            b, _ = logits_for(model, group, renderer.tokenizer.pad_token_id, arm)
            if not torch.equal(a, b):
                raise ValueError("final successor route differs from abrupt BDH")
        equivalence = {"predecessor_equals_native": True, "successor_equals_abrupt_bdh": True}
    first, _, activity = update(model, opt, renderer, batches[0], arm, plan, 1, force_successor=True)
    if any(v is None or v <= 0 for v in activity["new_module_gradient_norms"].values()):
        raise ValueError("new module is inactive")
    save_checkpoint(model, opt, renderer, dest / "step-1", arm, cfg, 1)
    expected_losses = []
    # Gradual deliberately tests a dormant-core update followed by an active update.
    for step in (2, 3):
        loss, _, _ = update(model, opt, renderer, batches[step - 1], arm, plan, step,
                            force_successor=(step == 3) if arm == "gradual" else True)
        expected_losses.append(loss)
    expected = common.master_digest(opt)
    expected_opt = common.optimizer_digest(opt)
    _, expected_pred = evaluate(model, renderer, probe, arm)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    model = load(arm, dest / "step-1")
    opt = optimizer(model, arm, plan)
    if common.restore_optimizer(model, opt, dest / "step-1") != 1:
        raise ValueError("wrong resume step")
    actual_losses = []
    for step in (2, 3):
        loss, _, _ = update(model, opt, renderer, batches[step - 1], arm, plan, step,
                            force_successor=(step == 3) if arm == "gradual" else True)
        actual_losses.append(loss)
    _, actual_pred = evaluate(model, renderer, probe, arm)
    if (actual_losses != expected_losses or common.master_digest(opt) != expected
            or common.optimizer_digest(opt) != expected_opt or expected_pred != actual_pred):
        raise ValueError("optimizer restart changes losses, masters, moments or predictions")
    worst_loss, _, _ = update(model, opt, renderer, worst, arm, plan, 4, force_successor=True)
    result = {"arm": arm, "implementation": implementation(), "exact_optimizer_resume": True,
              "first_loss": first, "next_two_losses": actual_losses, "initial_gradients": activity,
              "route_equivalence": equivalence, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "worst_cases": {"lengths": [len(r["input_ids"]) for r in worst],
                              "choices": [len(r["target"]) for r in worst], "loss": worst_loss},
              "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    train.atomic_json(dest / "result.json", result)
    print("SMOKE_PASS", arm, result["first_loss"], result["peak_cuda_allocated_gb"], flush=True)


def freeze(out):
    manifest = verify(out, frozen=False)
    if manifest["frozen"]:
        raise ValueError("already frozen")
    for arm in ARMS:
        result = read(out / f"smoke-{arm}/result.json")
        if result["implementation"] != implementation() or not result["exact_optimizer_resume"]:
            raise ValueError("stale or missing real-model validation")
    validation = read(out / "validation.json")
    if not validation["passed"] or validation["implementation"] != implementation():
        raise ValueError("missing or stale unit validation")
    for path in (out / "worker.py", out / "README.md", out / "validation.json", Path("tests/test_v3_dual_matching.py")):
        manifest["files"][str(path)] = train.sha256_file(path)
    manifest.update(frozen=True, implementation=implementation(), frozen_utc=now())
    (out / "source").mkdir()
    for path in manifest["implementation"]:
        shutil.copyfile(path, out / "source" / Path(path).name)
    train.atomic_json(out / "manifest.json", manifest)
    print("FROZEN", flush=True)


def write_evaluation(model, renderer, encoded, arm, dest, prefix, step, **kwargs):
    metric, predictions = evaluate(model, renderer, encoded, arm, **kwargs)
    train.atomic_json(dest / f"{prefix}-{step}.json", metric)
    search.save_records(dest / f"{prefix}-{step}.jsonl", predictions)
    print(prefix.upper(), arm, step, metric["macro"], flush=True)
    return metric


def fit(out, arm):
    manifest = verify(out)
    plan = manifest["plan"]
    dest = out / f"{arm}-seed41"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return
    renderer, cfg = search.renderer_for(search.BASE)
    checkpoints = sorted((p for p in dest.glob("step-*") if p.name[5:].isdigit() and (p / "complete.json").exists()),
                         key=lambda p: int(p.name[5:]))
    train.seed_all(41)
    model = load(arm, checkpoints[-1] if checkpoints else search.BASE)
    opt = optimizer(model, arm, plan)
    start = common.restore_optimizer(model, opt, checkpoints[-1]) if checkpoints else 0
    path = dest / "train.jsonl"
    trace = records.rows(path) if path.exists() else []
    if len(trace) < start:
        raise ValueError("checkpoint ahead of training log")
    if len(trace) > start:
        shutil.copyfile(path, dest / f"uncommitted-{time.time_ns()}.jsonl")
        search.save_records(path, trace[:start])
    reference = records.rows(REFERENCE / "native-seed41/train.jsonl")
    selection = records.rows(out / "selection-encoded.jsonl")
    probe = records.rows(out / "train-probe-encoded.jsonl")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if not start:
        write_evaluation(model, renderer, probe, arm, dest, "train-probe", 0)
        write_evaluation(model, renderer, selection, arm, dest, "selection", 0)
    with (out / "encoded-seed41.jsonl").open() as cache, path.open("a") as log:
        for step, line in enumerate(cache, 1):
            if step <= start:
                continue
            item = json.loads(line)
            rr = item["rows"]
            digest = search.digest([[r["input_ids"], r["code_ids"], r["target"]] for r in rr])
            if item["step"] != step or digest != reference[step - 1]["input_target_sha256"]:
                raise ValueError("inputs/targets differ from native control")
            t0 = time.monotonic()
            loss, norm, extra = update(model, opt, renderer, rr, arm, plan, step)
            entry = {"step": step, "loss": loss, "grad_norm": norm, "input_target_sha256": digest,
                     "seconds_update": time.monotonic() - t0, "seconds_this_session": time.monotonic() - started,
                     "examples_seen": step * 16} | extra
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0 or step == 1:
                train.atomic_json(out / "status.json", {"state": "running", "arm": arm, "step": step,
                    "steps_per_fit": plan["steps"], "updated_utc": now(), "loss": loss,
                    "completed_arms": [a for a in ARMS if (out / f"{a}-seed41/result.json").exists()]})
                print("FIT", arm, step, loss, extra["successor"], flush=True)
            if step in plan["probe_steps"]:
                write_evaluation(model, renderer, probe, arm, dest, "train-probe", step)
            if step in plan["snapshots"]:
                write_evaluation(model, renderer, selection, arm, dest, "selection", step)
                target = dest / f"step-{step}"
                if target.exists():
                    target.rename(dest / f"step-{step}.incomplete-{time.time_ns()}")
                save_checkpoint(model, opt, renderer, target, arm, cfg, step)
    if step != plan["steps"]:
        raise ValueError("incomplete stream")
    metric, prediction = evaluate(model, renderer, selection, arm)
    expected = records.rows(dest / "selection-4096.jsonl")
    if prediction != expected:
        raise ValueError("saving changed live model predictions")
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    common.verify_checkpoint(dest / "step-4096")
    model = load(arm, dest / "step-4096").requires_grad_(False)
    _, reloaded = evaluate(model, renderer, selection, arm)
    if reloaded != expected:
        raise ValueError("final checkpoint reload changed predictions")
    ablation = None
    if arm == "dual":
        off, off_predictions = evaluate(model, renderer, selection[:512], arm, disable_evidence=True)
        active = expected[:512]
        ablation = {"cases": 512, "changed_predictions": sum(a["prediction"] != b["prediction"] for a, b in zip(active, off_predictions)),
                    "evidence_zero_macro": off["macro"], "active_macro": search.metrics(active)["macro"],
                    "scope": "zero passage input to matching only; encoder still saw full original input; diagnostic, not a clean causal evidence-removal test"}
    trace = records.rows(path)
    result = {"arm": arm, "seed": 41, "steps": 4096, "examples_seen": 65536,
              "snapshots": {str(s): read(dest / f"selection-{s}.json")["macro"] for s in plan["snapshots"]},
              "new_module_updates": sum(r["successor"] for r in trace),
              "exact_final_reload": True, "all_inputs_targets_match_native": True,
              "evidence_ablation": ablation, "seconds_this_session": time.monotonic() - started,
              "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "new_confirmation_cases_scored": 0}
    train.atomic_json(dest / "result.json", result)
    print("RESULT", result, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report(out):
    manifest = verify(out)
    arms = {}
    for arm in ("native", "bdh", *ARMS):
        dest = (REFERENCE if arm in ("native", "bdh") else out) / f"{arm}-seed41"
        result_path = dest / "result.json"
        arms[arm] = {"complete": result_path.exists(), "path": str(dest),
                     "snapshots": {str(s): read(dest / f"selection-{s}.json")["macro"]
                                   for s in manifest["plan"]["snapshots"] if (dest / f"selection-{s}.json").exists()}}
        if result_path.exists():
            arms[arm]["fit"] = read(result_path)
    comparisons = {}
    for baseline, candidate in (("native", "dual"), ("self", "dual"), ("bdh", "gradual"), ("native", "gradual")):
        if not arms[candidate]["complete"] or not arms[baseline]["complete"]:
            continue
        a = records.rows(Path(arms[baseline]["path"]) / "selection-4096.jsonl")
        b = records.rows(Path(arms[candidate]["path"]) / "selection-4096.jsonl")
        interval = common.statistics_helper.paired_interval(a, b, confidence=1 - .05 / 4)
        interval["familywise_95ci_four_comparisons"] = interval.pop("paired_document_bootstrap_97_5ci")
        interval["exploratory_interval_above_zero"] = interval.pop("confirmed_improvement")
        interval["scope"] = "exposed selection, conditional on one trained seed; not independent confirmation"
        comparisons[f"{candidate}_minus_{baseline}"] = interval
    result = {"updated_utc": now(), "arms": arms, "comparisons": comparisons,
              "complete": all(arms[a]["complete"] for a in ARMS), "new_confirmation_cases_scored": 0}
    train.atomic_json(out / "report.json", result)
    lines = ["# DUMA / Theseus 구조 변경 실험", "", "64K개 입력, 4,096 업데이트, seed41. 기존 native/즉시 BDH 대조군은 해시 검증 후 재사용.", "",
             "DUMA는 모든 선택지를 한 번에 인코딩하는 변형이다. Theseus는 공통 prefix/head도 함께 학습하는 변형이다.", "",
             "| 구조 | 1024 | 2048 | 4096 | 상태 |", "|---|---:|---:|---:|---|"]
    for arm, r in arms.items():
        values = [f"{100 * r['snapshots'][str(s)]:.3f}%" if str(s) in r["snapshots"] else "—" for s in (1024, 2048, 4096)]
        lines.append("| " + " | ".join([arm, *values, "완료" if r["complete"] else "대기/진행 중"]) + " |")
    lines += ["", "반복 노출된 selection 4,147건의 탐색 결과이며, 독립 검증이나 학습 seed 간 오차 추정이 아니다.",
              "동일 데이터·업데이트 기준이며 계산량과 파라미터 수를 맞춘 비교는 아니다.", "",
              "근거: [DUMA](https://arxiv.org/abs/2001.09415), [BERT-of-Theseus](https://aclanthology.org/2020.emnlp-main.633/).", ""]
    Path("wiki/project/dual-theseus.md").write_text("\n".join(lines))
    if result["complete"]:
        train.atomic_json(out / "status.json", {"state": "complete", "updated_utc": now(), "report": str(out / "report.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cache", "smoke", "freeze", "run", "report"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "cache":
        cache(args.output)
    elif args.stage == "smoke":
        for arm in [args.arm] if args.arm else ARMS:
            smoke(args.output, arm)
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "report":
        report(args.output)
    else:
        verify(args.output)
        for arm in [args.arm] if args.arm else ARMS:
            fit(args.output, arm)
            report(args.output)


if __name__ == "__main__":
    main()
