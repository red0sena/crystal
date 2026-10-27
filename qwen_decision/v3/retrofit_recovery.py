"""Resume the frozen retrofit trial with a canonical no-grad evaluation context.

The original runner freezes every parameter only for final reload evaluation.
That changes BF16 numerical outputs in this stack. Preserve each arm's
original parameter trainability during no-grad evaluation, then restore flags.
No weights, training updates, data, learning rates or architecture are changed.
The original frozen source files and failure evidence remain immutable.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from qwen_decision.v3 import retrofit_experiment as original
import torch

_load = original.load
_evaluate = original.evaluate


def load(arm, path=original.search.BASE):
    model = _load(arm, path)
    model._retrofit_evaluation_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    return model


def canonical_evaluation(model, evaluate_fn, *args, **kwargs):
    desired = model._retrofit_evaluation_flags
    parameters = dict(model.named_parameters())
    if parameters.keys() != desired.keys():
        raise ValueError("evaluation parameter names changed")
    previous = {n: p.requires_grad for n, p in parameters.items()}
    try:
        for name, parameter in parameters.items():
            parameter.requires_grad_(desired[name])
        with torch.no_grad():
            return evaluate_fn(model, *args, **kwargs)
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(previous[name])


def evaluate(model, *args, **kwargs):
    return canonical_evaluation(model, _evaluate, *args, **kwargs)


def install():
    original.load = load
    original.evaluate = evaluate


def files():
    return [Path(__file__), Path("tests/test_v3_retrofit_recovery.py")]


def verify(out):
    manifest = original.read(out / "recovery-manifest.json")
    if original.train.sha256_file(out / "manifest.json") != manifest["original_manifest_sha256"]:
        raise ValueError("original manifest changed")
    original.verify(out)
    for filename, sha in manifest["files"].items():
        if original.train.sha256_file(Path(filename)) != sha:
            raise ValueError(f"recovery evidence/source changed: {filename}")
    return manifest


def prepare(out):
    original.verify(out)
    path = out / "recovery-manifest.json"
    if path.exists():
        raise ValueError("preserve previous recovery")
    diagnostic = original.read(out / "reload-diagnostic.json")
    if (diagnostic["trainable"]["prediction_mismatches"] or
            diagnostic["trainable"]["probability_mismatches"]):
        raise ValueError("canonical reload did not exactly reproduce the saved predictions")
    validation = original.read(out / "recovery-validation.json")
    source = {str(p): original.train.sha256_file(p) for p in files()}
    if not validation["passed"] or validation["source"] != source:
        raise ValueError("unit tests are missing or stale")
    extra = [out / "reload-diagnostic.json", out / "recovery-validation.json", out / "recovery-worker.py"]
    manifest = {"prepared_utc": original.now(), "original_manifest_sha256": original.train.sha256_file(out / "manifest.json"),
        "files": source | {str(p): original.train.sha256_file(p) for p in extra},
        "change": "preserve arm-specific training requires_grad flags during torch.no_grad evaluation; restore caller flags afterwards",
        "unchanged": ["training forwards/backwards", "weights", "data and target stream", "update budget", "optimizer and LR schedule", "architecture"],
        "original_failure": "final checkpoint reload changed predictions after requires_grad_(False)",
        "diagnostic": diagnostic}
    original.train.atomic_json(path, manifest)
    dest = out / "recovery-source"
    dest.mkdir()
    for p in files():
        shutil.copyfile(p, dest / p.name)
    print("RECOVERY_PREPARED", flush=True)


def finalize_dual(out):
    verify(out)
    dest = out / "dual-seed41"
    if (dest / "result.json").exists():
        return
    trace = original.records.rows(dest / "train.jsonl")
    reference = original.records.rows(original.REFERENCE / "native-seed41/train.jsonl")
    if len(trace) != 4096 or [r["input_target_sha256"] for r in trace] != [r["input_target_sha256"] for r in reference]:
        raise ValueError("incomplete or unmatched existing training")
    checkpoint = dest / "step-4096"
    original.common.verify_checkpoint(checkpoint)
    renderer, _ = original.search.renderer_for(original.search.BASE)
    encoded = original.records.rows(out / "selection-encoded.jsonl")
    expected = original.records.rows(dest / "selection-4096.jsonl")
    model = load("dual", checkpoint).requires_grad_(False)
    started = time.monotonic()
    metrics, reloaded = evaluate(model, renderer, encoded, "dual")
    if reloaded != expected:
        raise ValueError("canonical frozen-model reload still differs")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("evaluation did not restore caller trainability")
    original.search.save_records(dest / "canonical-reload-4096.jsonl", reloaded)
    off, off_predictions = evaluate(model, renderer, encoded[:512], "dual", disable_evidence=True)
    active = expected[:512]
    ablation = {"cases": 512, "changed_predictions": sum(a["prediction"] != b["prediction"] for a, b in zip(active, off_predictions)),
                "evidence_zero_macro": off["macro"], "active_macro": original.search.metrics(active)["macro"],
                "scope": "zero passage input to matching only; encoder still saw full original input; diagnostic only"}
    result = {"arm": "dual", "seed": 41, "steps": 4096, "examples_seen": 65536,
              "snapshots": {str(s): original.read(dest / f"selection-{s}.json")["macro"] for s in (1024, 2048, 4096)},
              "new_module_updates": 4096, "exact_final_reload": True,
              "evaluation_context": "canonical training requires_grad flags under torch.no_grad; caller flags restored",
              "recovery_manifest_sha256": original.train.sha256_file(out / "recovery-manifest.json"),
              "all_inputs_targets_match_native": True, "evidence_ablation": ablation,
              "training_seconds_logged": trace[-1]["seconds_this_session"],
              "recovery_evaluation_seconds": time.monotonic() - started,
              "new_confirmation_cases_scored": 0}
    original.train.atomic_json(dest / "result.json", result)
    original.train.atomic_json(dest / "recovery-audit.json", {
        "original_frozen_parameter_evaluation": original.read(out / "reload-diagnostic.json"),
        "canonical_reload": {"cases": len(expected), "prediction_mismatches": 0,
                             "probability_mismatches": 0, "macro": metrics["macro"]},
        "training_repeated": False, "caller_flags_restored": True})
    print("FINALIZED_DUAL", metrics["macro"], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run(out):
    verify(out)
    install()
    original.train.atomic_json(out / "status.json", {"state": "recovering", "arm": "dual", "updated_utc": original.now()})
    finalize_dual(out)
    original.report(out)
    for arm in ("gradual", "self"):
        verify(out)
        original.train.atomic_json(out / "status.json", {"state": "starting", "arm": arm, "updated_utc": original.now()})
        original.fit(out, arm)
        original.train.atomic_json(out / f"{arm}-seed41/recovery-context.json", {
            "recovery_manifest_sha256": original.train.sha256_file(out / "recovery-manifest.json"),
            "canonical_evaluation_only": True, "training_recipe_changed": False})
        original.report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, default=original.OUT)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)


if __name__ == "__main__":
    main()
