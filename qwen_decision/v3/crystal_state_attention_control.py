"""Pre-evaluation attention controls for the matched Crystal experiment.

The primary runner holds causal state extraction fixed. This supplement
also evaluates Crystal in its native noncausal mode, with an untrained
noncausal encoder as the corresponding attention-matched baseline.
No weights or hyperparameters are selected using these test scores.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from qwen_decision.v3 import crystal_state_compare as exp, modeling, state_rotation as sr, train

OUT = exp.OUT / "noncausal-control"


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any(OUT.glob("*/features.pt"))
    assert not (exp.OUT / "report.json").exists(), "Supplement must be fixed before test analysis"
    plan = {"reason": "Guard inference-mode mismatch: evaluate native noncausal Crystal states AND noncausal untrained states.",
            "primary_plan_sha256": train.sha256_file(exp.OUT / "plan.json"),
            "source_sha256": train.sha256_file(Path(__file__)),
            "attention": "noncausal 6 softmax / causal 18 DeltaNet, same in both supplementary arms",
            "supervision": "no new training beyond the original 8922 text/seen rows",
            "data": "exact primary scenes, question holdouts, candidates, train-only PCA and rotation",
            "arms": ["untrained-causal", "untrained-noncausal", "crystal-271", "crystal-283", "crystal-307"],
            "downstream": "exact original state_rotation.analyze, 3 identical head seeds, 160epochs, 20 shuffle maps",
            "uncertainty": "5000 paired crossed LM-seed/head-seed/whole-scene bootstrap draws",
            "selection": "none; report both causal primary and noncausal controls regardless of direction",
            "created_before_any_evaluation_complete": not any(exp.OUT.glob("*/evaluation-complete.json")),
            "baseline_amendment": "Re-extract untrained causal features with identical FLA convolution, compiled elementwise operations and SDPA inference as fitted encoders. Parent v2 cached features used stock HF kernels. Replace baseline in final primary aggregation; preserve original aggregation as report-cached-baseline.json and quantify kernel drift. Primary fitted-arm contrast is unchanged.",
            "prior_plan_sha256": train.sha256_file(OUT / "plan-initial.json") if (OUT / "plan-initial.json").exists() else None}
    assert plan["created_before_any_evaluation_complete"]
    train.atomic_json(OUT / "plan.json", plan)


def run():
    torch.set_num_threads(8)
    exp.check_plan(exp.OUT)
    # Main run enabled these fused functions while training. Match that numerical path.
    modeling.compile_elementwise()
    plan = exp.read(OUT / "plan.json")
    assert plan["source_sha256"] == train.sha256_file(Path(__file__))
    assert plan["primary_plan_sha256"] == train.sha256_file(exp.OUT / "plan.json")
    rows, specs, splits = exp.dataset()
    directories = []
    for name in plan["arms"]:
        directory = OUT / name
        directories.append(directory)
        if (directory / "complete.json").exists():
            continue
        directory.mkdir(exist_ok=True)
        checkpoint = None if name.startswith("untrained-") else exp.OUT / name / "best.safetensors"
        mode = "causal" if name == "untrained-causal" else "crystal"
        model = exp.inference_model(mode, checkpoint)
        processor = AutoProcessor.from_pretrained(exp.BASE, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        features = exp.extract_in_memory(model, processor, rows, specs, directory)
        # The untrained noncausal native readout is an attention-matched zero-shot control.
        if name == "untrained-noncausal":
            native = {}
            for modality in ("text", "image"):
                array, records = exp.native_records(model, processor, rows, specs, splits["test"], modality)
                np.save(directory / f"native-{modality}.npy", array)
                sr.dump(directory / f"native-{modality}-items.json", records)
                native[modality] = sr.summarize(array[None])
            sr.dump(directory / "native.json", native)
        del model
        gc.collect(); torch.cuda.empty_cache()
        report = sr.analyze(rows, specs, features, directory)
        report.update({"experiment": "crystal-state-noncausal-control", "encoder": name,
                       "state_inference_attention": mode, "primary_report_metadata_inherited": True})
        sr.dump(directory / "report.json", report)
        sr.dump(directory / "complete.json", {"complete": True})
    arrays, summaries, alignments = {}, {}, {}
    for arm in ("untrained", "crystal"):
        dirs = [directories[1]] * 3 if arm == "untrained" else directories[2:]
        arrays[arm] = {name: np.stack([np.load(d / "test-scene-records.npz")[name] for d in dirs])
                       for name in ("image_mean_only", "image_rotation", "text_reference")}
        summaries[arm] = {name: sr.summarize(a.reshape(-1, *a.shape[2:])) for name, a in arrays[arm].items()}
        alignments[arm] = {name: {metric: float(np.mean([exp.read(d / "report.json")["alignment_test"][name][metric]
                                                       for d in dirs])) for metric in ("paired_cosine", "retrieval_R1", "retrieval_R5")}
                           for name in ("image_mean_only", "image_rotation")}
    intervals = {}
    for name in arrays["crystal"]:
        rng = np.random.default_rng(sr.SEED + 500)
        samples = {g: {m: [] for m in ("accuracy", "nll")} for g in ("seen", "unseen")}
        for _ in range(5000):
            lm, hs, scenes = rng.integers(3, size=3), rng.integers(3, size=3), rng.integers(256, size=256)
            scores = {}
            for arm in arrays:
                a = arrays[arm][name][lm[:, None], hs[None, :]][:, :, scenes]
                scores[arm] = sr.summarize(a.reshape(-1, *a.shape[2:]))
            for g in samples:
                for m in samples[g]:
                    samples[g][m].append(scores["crystal"][g]["macro_" + m] - scores["untrained"][g]["macro_" + m])
        intervals[name] = {g: {m: {"delta": summaries["crystal"][name][g]["macro_" + m] - summaries["untrained"][name][g]["macro_" + m],
                                   "ci95": np.quantile(v, [.025, .975]).tolist()} for m, v in metrics.items()}
                           for g, metrics in samples.items()}
    sr.dump(OUT / "report.json", {"complete": True, "state_transfer_noncausal": summaries,
                                   "alignment_noncausal": alignments, "crystal_minus_untrained_ci95": intervals})
    # Reuse unchanged original aggregation on the numerical-path-matched baseline.
    original = exp.read(exp.OUT / "report.json")
    sr.dump(exp.OUT / "report-cached-baseline.json", original)
    old_parent = exp.PARENT
    exp.PARENT = directories[0]
    exp.all_results(exp.OUT)
    exp.PARENT = old_parent
    corrected = exp.read(exp.OUT / "report.json")
    corrected["baseline_numerical_path_correction"] = {
        "reason": plan["baseline_amendment"], "supplement_plan": "noncausal-control/plan.json",
        "original_cached_baseline": original["state_transfer"]["plain"],
        "matched_baseline": "noncausal-control/untrained-causal",
        "seen_rotation_accuracy_drift": corrected["state_transfer"]["plain"]["image_rotation"]["seen"]["macro_accuracy"] - original["state_transfer"]["plain"]["image_rotation"]["seen"]["macro_accuracy"]}
    sr.dump(exp.OUT / "report.json", corrected)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    else:
        run()
