"""Full public text/vision evaluation of the R3 adapter with the exact R1 public engines and policy."""
from __future__ import annotations

import argparse
import traceback

import torch

from qwen_decision.v3 import mix_ablation_public as P, skill_mix as sm, champion_full_train as fit

_adapter_for, _root = P.m.adapter_for, P.root


def adapter_for(arm):
    return sm.adapter() if arm == "r3" else _adapter_for(arm)


def root(arm):
    return sm.ROOT / "public" if arm == "r3" else _root(arm)


P.m.adapter_for, P.root = adapter_for, root


class R3Text(P.ArmText):
    name = "skill-mix-r3-native-text"

    def __init__(self, **options):
        super().__init__("r3", **options)


class R3Vision(P.ArmVision):
    name = "skill-mix-r3-native-vision"

    def __init__(self, image_root=None, **options):
        super().__init__("r3", image_root=image_root, **options)


def run(modality):
    """P.run with the R3 engine paths (P.run hard-codes the ArmText/ArmVision class paths)."""
    from collections import Counter
    from qwen_decision.v3 import champion_audit as audit, train
    dest = root("r3") / modality
    if (dest / "completion.json").exists():
        return
    P.state("r3", "public_evaluation", modality=modality)
    if modality == "text":
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        from decision_index.suite.io import Suite
        runner("qwen_decision.v3.skill_mix_public:R3Text", {}, P.PUBLIC / "text-counted-rows.jsonl", dest, compact=True,
               corpus_sha256=audit.sha(P.PUBLIC / "text-counted-rows.jsonl"))
        scores = score_run(Suite(P.PUBLIC / "text-suite", "0.3"), dest / "results.jsonl", "skill-mix-r3", dest)
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner("qwen_decision.v3.skill_mix_public:R3Vision", {}, P.PUBLIC / "vision-suite", dest, workers=1, chunk=16, compact=True)
        scores = score_run(P.PUBLIC / "vision-suite/vision-rows.jsonl", dest / "results.jsonl", dest)
    rows = list(audit.rows(dest / "results.jsonl"))
    counts = Counter(r["status"] for r in rows)
    expected = 110087 if modality == "text" else 12210
    assert len(rows) == len({r["run_id"] for r in rows}) == expected
    train.atomic_json(dest / "completion.json", {"complete": True, "counts": dict(counts), "visited_requests": expected,
        "public_index": scores.get("decision_index", scores.get("public_vision_index")),
        "errors_require_review": bool(counts["error"] or counts["unsupported"]), "official_full_rank": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("modality", choices=("text", "vision"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    try:
        if args.command == "check":
            P.check("r3", args.modality)
        else:
            run(args.modality)
    except BaseException:
        P.state("r3", "failed", command=args.command, modality=args.modality, traceback=traceback.format_exc())
        raise
