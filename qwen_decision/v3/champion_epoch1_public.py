"""Canonical full public-suite comparison of epoch1 against the current <=1B Full leaders.

Keep checkpoint BF16 adapters unmerged and use the existing native padded SDPA engines.
Reuse provenance-verified complete leader predictions; never turn public results into training labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from safetensors.torch import load_file

from qwen_decision.v3 import champion_audit as audit, champion_full_train as fit
from qwen_decision.v3 import vision_train as vt, lora_placement_experiment as place, train
from qwen_decision.v3.engine import V3Engine
from qwen_decision.v3.vision_engine import VisionEngine

ROOT = fit.ROOT / "epoch1-leader-eval"
PUBLIC = Path("runs/champion-v1/public-eval-v1")
INTERNAL = fit.ROOT / "epoch1-eval"
PLAN = ROOT / "plan.json"


def state(stage, **extra):
    train.atomic_json(ROOT / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})


def overlay(model):
    cp = audit.read(INTERNAL / "request.json")["epoch1_checkpoint"]
    assert cp["step"] == 70659
    path = Path(cp["directory"]) / "adapter.safetensors"
    assert audit.sha(path) == cp["files"][path.name]
    vt.LM_TARGETS = place.ARMS["all"][0]
    vt.add_lora(model, False)
    saved = {n: t for n, t in load_file(str(path), device="cpu").items() if not n.startswith("master.")}
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert set(params) == set(saved) and len(params) == 372
    with torch.no_grad():
        for name, param in params.items():
            assert param.dtype == saved[name].dtype and torch.isfinite(saved[name]).all()
            param.copy_(saved[name])
    assert all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params.items())
    model.requires_grad_(False)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == 863808576 < 10**9
    return cp


class EpochText(V3Engine):
    name = "full-g7-epoch1-native-text"

    def __init__(self, **options):
        if options:
            raise TypeError(options)
        super().__init__(model=str(fit.data.BASE), attention="noncausal", deltanet="causal", layout="d3",
                         batch_tokens=8192, max_batch=8, orders=1)
        cp = overlay(self.scorer.model)
        self.provenance.update(checkpoint_step=cp["step"], adapter_sha256=cp["files"]["adapter.safetensors"],
                               kernel="native padded sdpa", adapter="stored BF16 weights, unmerged",
                               served_parameters=863808576)


class EpochVision(VisionEngine):
    name = "full-g7-epoch1-native-vision"

    def __init__(self, image_root=None, **options):
        if options:
            raise TypeError(options)
        super().__init__(model=str(fit.data.BASE), attention="noncausal", prompt="decide",
                         processor=str(fit.data.BASE), image_root=image_root, max_pixels=1638400)
        cp = overlay(self.model)
        self.provenance.update(checkpoint_step=cp["step"], adapter_sha256=cp["files"]["adapter.safetensors"],
                               kernel="native padded sdpa", adapter="stored BF16 weights, unmerged",
                               served_parameters=863808576)


def reference_files():
    paths = [PUBLIC / "text-counted-rows.jsonl", PUBLIC / "text-probes.jsonl",
             PUBLIC / "text-counted-verification.json", PUBLIC / "vision-verification.json",
             PUBLIC / "gliclass-published-score/scores.json", PUBLIC / "gliclass-published-score/parity.json",
             PUBLIC / "qev-vision/completion.json", PUBLIC / "qev-vision/vision-scores.json",
             PUBLIC / "qev-vision/results.jsonl", PUBLIC / "qev-vision/plan.json",
             PUBLIC / "vision-suite/vision-rows.jsonl", PUBLIC / "gliclass-reference-audit.json"]
    paths += [Path(x["local"]) for x in audit.read(PUBLIC / "references.json")
              if x["repo"] == "knowledgator/decision-index-results"]
    paths += list((ROOT / "board/data").glob("*.json"))
    paths += [ROOT / "current-leaders.json"]
    return paths


def prepare():
    """CPU checks and immutable queue plan; GPU preflight runs after internal evaluation releases the GPU."""
    ROOT.mkdir(parents=True, exist_ok=True)
    leaders = audit.read(ROOT / "current-leaders.json")
    assert leaders["text_top"][0]["engine"] == "gliclass-v4-edge"
    assert leaders["vision_top"][0]["engine"] == "qev-0.8b"
    parity = audit.read(PUBLIC / "gliclass-published-score/parity.json")
    assert parity["match"] and parity["recomputed"] == 47.49
    qev = audit.read(PUBLIC / "qev-vision/completion.json")
    assert qev["finished"] and qev["counts"] == {"ok": 12210}
    assert audit.read(PUBLIC / "vision-verification.json")["ok"]
    assert audit.read(PUBLIC / "text-counted-verification.json")["actual_inputs_match_published_reference"]
    for ref in audit.read(PUBLIC / "references.json"):
        if ref["repo"] == "knowledgator/decision-index-results":
            assert audit.sha(ref["local"]) == ref["sha256"]
    cp = audit.read(INTERNAL / "request.json")["epoch1_checkpoint"]
    for name, sha in cp["files"].items():
        assert audit.sha(Path(cp["directory"]) / name) == sha
    # Prove every cached vision result refers to the exact same pinned public request.
    expected = {r["id"]: r for r in audit.rows(PUBLIC / "vision-suite/vision-rows.jsonl")}
    seen = set()
    for row in audit.rows(PUBLIC / "qev-vision/results.jsonl"):
        rid = row["run_id"]
        assert rid in expected and rid not in seen and row["status"] == "ok"
        assert row["payload_sha256"] == expected[rid]["_evaluation"]["payload_sha256"]
        assert row["image_sha256"] == expected[rid]["_evaluation"]["image_sha256"]
        seen.add(rid)
    assert len(seen) == len(expected) == 12210
    plan = {"checkpoint": cp, "training_plan_sha256": audit.sha(fit.PLAN),
        "implementation": {str(Path(__file__)): audit.sha(__file__), **fit.implementation()},
        "reference_files": {str(p): audit.sha(p) for p in reference_files()},
        "current_leaders": leaders, "public_counts": {"text": 110087, "vision": 12210},
        "engines": {"text": "qwen_decision.v3.champion_epoch1_public:EpochText",
                    "vision": "qwen_decision.v3.champion_epoch1_public:EpochVision"},
        "text_reference": "GLiClass author-published full predictions, pinned payloads verified and rescored; not a fresh full local GLiClass run",
        "vision_reference": "QEV accurate-mode complete local run, pinned payload and image hashes matched",
        "policy": "Unmerged checkpoint BF16 LoRA; native padded SDPA, orders1, fixed d3, no tuning or truncation",
        "official_full_rank": None, "submission": False,
        "caveat": "Public benchmarks observed during development; Full includes private tests; public score cannot establish Full rank."}
    if PLAN.exists():
        assert audit.read(PLAN) == plan
    else:
        train.atomic_json(PLAN, plan)
    state("queued", checkpoint_step=cp["step"], after="epoch1 internal Crystal/original comparison")
    return plan


def verify():
    p = audit.read(PLAN)
    assert audit.sha(fit.PLAN) == p["training_plan_sha256"]
    for category in ("implementation", "reference_files"):
        for path, sha in p[category].items():
            assert audit.sha(path) == sha, path
    for name, sha in p["checkpoint"]["files"].items():
        assert audit.sha(Path(p["checkpoint"]["directory"]) / name) == sha
    return p


def check(modality):
    verify()
    state("preflight", modality=modality)
    from decision_index.engines import validate
    engine = EpochText() if modality == "text" else EpochVision(image_root=str(PUBLIC / "vision-suite"))
    cases = []
    if modality == "text":
        rows = list(audit.rows(PUBLIC / "text-probes.jsonl"))
    else:
        rows = []
        seen = set()
        for row in audit.rows(PUBLIC / "vision-suite/vision-rows.jsonl"):
            if row["benchmark"] not in seen:
                rows.append(row)
                seen.add(row["benchmark"])
    for row in rows:
        if modality == "text":
            response, _ = engine(row["state"], row["questions"])
            rid = row["_evaluation"]["run_id"]
        else:
            response, _ = engine({}, row["questions"], images=row["images"], prompt=row.get("prompt"))
            rid = row["id"]
        validate(row["questions"], response)
        cases.append({"run_id": rid, "status": "ok"})
        print("PREFLIGHT", modality, rid, flush=True)
    train.atomic_json(ROOT / ("check-" + modality + ".json"),
                      {"passed": True, "cases": cases, "provenance": engine.provenance})


def run(modality):
    p = verify()
    assert audit.read(ROOT / ("check-" + modality + ".json"))["passed"]
    state("public_evaluation", modality=modality)
    dest = ROOT / ("epoch1-" + modality)
    if modality == "text":
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        from decision_index.suite.io import Suite
        runner(p["engines"][modality], {}, PUBLIC / "text-counted-rows.jsonl", dest, compact=True,
               corpus_sha256=p["reference_files"][str(PUBLIC / "text-counted-rows.jsonl")])
        scores = score_run(Suite(PUBLIC / "text-suite", "0.3"), dest / "results.jsonl", "full-g7-epoch1", dest)
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner(p["engines"][modality], {}, PUBLIC / "vision-suite", dest, workers=1, chunk=16, compact=True)
        scores = score_run(PUBLIC / "vision-suite/vision-rows.jsonl", dest / "results.jsonl", dest)
    rows = list(audit.rows(dest / "results.jsonl"))
    counts = Counter(r["status"] for r in rows)
    expected = p["public_counts"][modality]
    assert len(rows) == len({r["run_id"] for r in rows}) == expected
    # Preserve unsupported/errors in the official metric denominator and expose them explicitly.
    train.atomic_json(dest / "completion.json", {"complete": True, "counts": dict(counts),
        "visited_requests": expected, "public_index": scores.get("decision_index", scores.get("public_vision_index")),
        "errors_require_review": bool(counts["error"] or counts["unsupported"]), "official_full_rank": None})


def report():
    p = verify()
    ours_text = audit.read(ROOT / "epoch1-text/scores.json")
    ours_vision = audit.read(ROOT / "epoch1-vision/vision-scores.json")
    leader_text = audit.read(PUBLIC / "gliclass-published-score/scores.json")
    leader_vision = audit.read(PUBLIC / "qev-vision/vision-scores.json")
    for modality in ("text", "vision"):
        assert audit.read(ROOT / ("epoch1-" + modality) / "completion.json")["complete"]
    text = {key: {"dataset": b["dataset"], "metric": b["metric"], "epoch1_raw": b["score"],
                 "leader_raw": leader_text["benchmarks"][key]["score"],
                 "epoch1_skill": ours_text["index_benchmarks"][key]["skill"],
                 "leader_skill": leader_text["index_benchmarks"][key]["skill"]}
            for key, b in ours_text["benchmarks"].items() if b["in_index"]}
    vision = {key: {"epoch1_accuracy": b["accuracy"], "leader_accuracy": leader_vision["benchmarks"][key]["accuracy"],
                   "epoch1_skill": b["skill"], "leader_skill": leader_vision["benchmarks"][key]["skill"]}
              for key, b in ours_vision["benchmarks"].items()}
    result = {"complete": True, "checkpoint_step": 70659, "leaders": p["current_leaders"],
        "text": {"epoch1_public": ours_text["decision_index"], "leader_public": leader_text["decision_index"],
                 "delta": ours_text["decision_index"]-leader_text["decision_index"], "benchmarks": text,
                 "reference_kind": p["text_reference"]},
        "vision": {"epoch1_public": ours_vision["public_vision_index"], "leader_local_public": leader_vision["public_vision_index"],
                   "leader_board_public": p["current_leaders"]["vision_top"][0]["pub"],
                   "delta": ours_vision["public_vision_index"]-leader_vision["public_vision_index"], "benchmarks": vision,
                   "reference_kind": p["vision_reference"]},
        "official_full_rank": None, "caveat": p["caveat"]}
    train.atomic_json(ROOT / "comparison.json", result)
    train.atomic_json(ROOT / "completion.json", {"complete": True, "comparison": str(ROOT / "comparison.json")})
    state("complete", checkpoint_step=70659)


def pipeline():
    for command in ("check-text", "check-vision", "run-text", "run-vision", "report"):
        state("pipeline", command=command)
        with (ROOT / (command + ".log")).open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "qwen_decision.v3.champion_epoch1_public", command],
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "pipeline", "check-text", "check-vision", "run-text", "run-vision", "report"))
    command = parser.parse_args().command
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    try:
        if command == "prepare":
            prepare()
        elif command == "pipeline":
            pipeline()
        elif command == "report":
            report()
        else:
            kind, modality = command.split("-")
            (check if kind == "check" else run)(modality)
    except BaseException:
        state("failed", command=command, traceback=traceback.format_exc())
        raise
