"""Per-question forgetting between two proxy_eval runs on the same rows: right->wrong and wrong->right.

  python -m qwen_decision.v3.forget_report --before runs/proxy/base-noncausal-dev --after runs/proxy/<run> \
      [--out runs/proxy/<run>/forgetting.json]

A higher accuracy can hide forgetting: a run can gain on some questions while losing others the starting
model answered. Counting the two flips apart (as in per-item forgetting studies, e.g. arXiv 2510.17776)
keeps "learned less and forgot less" from looking like "learned more". Compare runs against the base model
under the same attention mode and prompt layout (the `before` run), on the same split and sampling, or the
counts mix two changes. Rows must match by id; a mismatch is an error.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from qwen_decision.v3.proxy_eval import group_of


def read(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (run / "predictions.jsonl").open())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    before = read(args.before)
    report = {"before": str(args.before), "runs": {}}
    for run in args.after:
        after = read(run)
        if set(after) != set(before):
            raise SystemExit(f"{run}: {len(set(after) ^ set(before))} rows differ from {args.before}; use the same split and sampling")
        by = defaultdict(lambda: {"n": 0, "kept": 0, "forgot": 0, "learned": 0})
        for i, b in before.items():
            for key in (b["source"], "group:" + group_of(b["source"]), "all"):
                s = by[key]
                s["n"] += 1
                s["kept"] += b["correct"] and after[i]["correct"]
                s["forgot"] += b["correct"] and not after[i]["correct"]
                s["learned"] += not b["correct"] and after[i]["correct"]
        for s in by.values():
            s["net"] = s["learned"] - s["forgot"]
            right = s["kept"] + s["forgot"]
            s["forgot_rate"] = s["forgot"] / right if right else 0.
        report["runs"][str(run)] = dict(sorted(by.items()))
        overall = by["all"]
        print(f"{run}: forgot {overall['forgot']} of {overall['kept'] + overall['forgot']} the base answered "
              f"({overall['forgot_rate']:.1%}), learned {overall['learned']}, net {overall['net']:+d}")
        worst = sorted((k for k in by if ":" not in k and k != "all"), key=lambda k: -by[k]["forgot"])[:8]
        for k in worst:
            s = by[k]
            print(f"  {k:28s} forgot {s['forgot']:3d}  learned {s['learned']:3d}  net {s['net']:+4d}  (n {s['n']})")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
