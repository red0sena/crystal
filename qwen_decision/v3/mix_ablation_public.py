"""Full public text (37 counted panels, 110,087 requests) and vision (11 benchmarks, 12,210) for a mix-ablation arm.

Same engines/policy as the epoch1 public comparison: unmerged BF16 LoRA, native padded SDPA, d3, orders1,
no tuning or truncation. Public results are a development observation, never training labels or checkpoint choice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from safetensors.torch import load_file

from qwen_decision.v3 import champion_audit as audit, champion_full_train as fit, mix_ablation as m
from qwen_decision.v3 import vision_train as vt, lora_placement_experiment as place, train
from qwen_decision.v3.engine import V3Engine
from qwen_decision.v3.vision_engine import VisionEngine

PUBLIC = Path("runs/champion-v1/public-eval-v1")
SERVED = 863808576


def root(arm):
    return m.arm_root(arm) / "public"


def state(arm, stage, **extra):
    train.atomic_json(root(arm) / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})


def overlay(model, arm):
    path, sha = m.adapter_for(arm)
    vt.LM_TARGETS = place.ARMS["all"][0]
    vt.add_lora(model, False)
    saved = {n: t for n, t in load_file(str(path), device="cpu").items() if not n.startswith("master.")}
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert set(params) == set(saved) and len(params) == 372
    with torch.no_grad():
        for name, param in params.items():
            assert param.dtype == saved[name].dtype and torch.isfinite(saved[name]).all()
            param.copy_(saved[name])
    model.requires_grad_(False)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == SERVED < 10**9
    return sha


class ArmText(V3Engine):
    name = "mix-ablation-native-text"

    def __init__(self, arm, **options):
        if options:
            raise TypeError(options)
        super().__init__(model=str(fit.data.BASE), attention="noncausal", deltanet="causal", layout="d3",
                         batch_tokens=8192, max_batch=8, orders=1)
        sha = overlay(self.scorer.model, arm)
        self.provenance.update(arm=arm, adapter_sha256=sha, kernel="native padded sdpa",
                               adapter="stored BF16 weights, unmerged", served_parameters=SERVED)


class ArmVision(VisionEngine):
    name = "mix-ablation-native-vision"

    def __init__(self, arm, image_root=None, **options):
        if options:
            raise TypeError(options)
        super().__init__(model=str(fit.data.BASE), attention="noncausal", prompt="decide",
                         processor=str(fit.data.BASE), image_root=image_root, max_pixels=1638400)
        sha = overlay(self.model, arm)
        self.provenance.update(arm=arm, adapter_sha256=sha, kernel="native padded sdpa",
                               adapter="stored BF16 weights, unmerged", served_parameters=SERVED)


def check(arm, modality):
    from decision_index.engines import validate
    root(arm).mkdir(parents=True, exist_ok=True)
    state(arm, "preflight", modality=modality)
    engine = ArmText(arm) if modality == "text" else ArmVision(arm, image_root=str(PUBLIC / "vision-suite"))
    if modality == "text":
        rows = list(audit.rows(PUBLIC / "text-probes.jsonl"))
    else:
        rows, seen = [], set()
        for row in audit.rows(PUBLIC / "vision-suite/vision-rows.jsonl"):
            if row["benchmark"] not in seen:
                rows.append(row)
                seen.add(row["benchmark"])
    for row in rows:
        if modality == "text":
            response, _ = engine(row["state"], row["questions"])
        else:
            response, _ = engine({}, row["questions"], images=row["images"], prompt=row.get("prompt"))
        validate(row["questions"], response)
    train.atomic_json(root(arm) / f"check-{modality}.json", {"passed": True, "cases": len(rows),
                      "provenance": engine.provenance})


def run(arm, modality):
    assert audit.read(root(arm) / f"check-{modality}.json")["passed"]
    state(arm, "public_evaluation", modality=modality)
    dest = root(arm) / modality
    if (dest / "completion.json").exists():
        return
    if modality == "text":
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        from decision_index.suite.io import Suite
        runner("qwen_decision.v3.mix_ablation_public:ArmText", {"arm": arm}, PUBLIC / "text-counted-rows.jsonl", dest,
               compact=True, corpus_sha256=audit.sha(PUBLIC / "text-counted-rows.jsonl"))
        scores = score_run(Suite(PUBLIC / "text-suite", "0.3"), dest / "results.jsonl", f"mix-ablation-{arm}", dest)
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner("qwen_decision.v3.mix_ablation_public:ArmVision", {"arm": arm}, PUBLIC / "vision-suite", dest,
               workers=1, chunk=16, compact=True)
        scores = score_run(PUBLIC / "vision-suite/vision-rows.jsonl", dest / "results.jsonl", dest)
    rows = list(audit.rows(dest / "results.jsonl"))
    counts = Counter(r["status"] for r in rows)
    expected = 110087 if modality == "text" else 12210
    assert len(rows) == len({r["run_id"] for r in rows}) == expected
    train.atomic_json(dest / "completion.json", {"complete": True, "counts": dict(counts), "visited_requests": expected,
        "public_index": scores.get("decision_index", scores.get("public_vision_index")),
        "errors_require_review": bool(counts["error"] or counts["unsupported"]), "official_full_rank": None})


def pipeline(arm):
    root(arm).mkdir(parents=True, exist_ok=True)
    for command, modality in (("check", "text"), ("check", "vision"), ("run", "text"), ("run", "vision")):
        if m.paused():
            state(arm, "paused_by_request")
            return
        name = f"{command}-{modality}"
        state(arm, "pipeline", command=name)
        with (root(arm) / (name + ".log")).open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "qwen_decision.v3.mix_ablation_public", command, modality,
                            "--arm", arm], stdout=log, stderr=subprocess.STDOUT, check=True)
    state(arm, "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pipeline", "check", "run"))
    parser.add_argument("modality", nargs="?", choices=("text", "vision"))
    parser.add_argument("--arm", choices=m.ARMS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    try:
        if args.command == "pipeline":
            pipeline(args.arm)
        else:
            (check if args.command == "check" else run)(args.arm, args.modality)
    except BaseException:
        state(args.arm, "failed", command=args.command, modality=args.modality, traceback=traceback.format_exc())
        raise
