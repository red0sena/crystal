"""Checks for scorer_v2: shared-state equivalence and speed, limits, no generation.

Usage: test_scorer_v2.py --model <dir> --output <json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

import decision_format as fmt
from scorer_v2 import DecisionScorer

ROOT = Path(__file__).resolve().parent


def question_of(r):
    q = {"type": r["qtype"], "question": r["request"]["question"]}
    if r["qtype"] != "noul":
        q["options"] = r["request"]["options"]
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--articles", type=int, default=12)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    scorer = DecisionScorer(args.model, calibration=None, dtype=torch.float32 if args.fp32 else None)
    calls = []
    original_forward = scorer.lm.forward

    def counted(*a, **k):
        calls.append(1)
        return original_forward(*a, **k)
    scorer.lm.forward = counted
    with (ROOT / "data/text-decisions-v2/test.jsonl").open() as f:
        test = [json.loads(x) for x in f]
    by_article = defaultdict(list)
    for r in test:
        if r["task"] == "long_reading":
            by_article[r["request"]["state"]].append(r)
    articles = sorted((rr for rr in by_article.values() if len(rr) >= 8), key=lambda rr: rr[0]["id"])[:args.articles]
    rows, max_dp, agree, total = [], 0., 0, 0
    for rr in articles:
        request = {"state": rr[0]["request"]["state"], "questions": {r["id"]: question_of(r) for r in rr[:8]}}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        calls.clear()
        shared = scorer.score(request, share_state=True, calibrated=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        shared_calls = len(calls)
        calls.clear()
        alone = scorer.score(request, share_state=False, calibrated=False)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        for name in request["questions"]:
            a, b = shared["answers"][name], alone["answers"][name]
            max_dp = max(max_dp, max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
            agree += a["choice"] == b["choice"]
            total += 1
        rows.append({"state_tokens_plus_suffixes": shared["usage"]["input_tokens"], "independent_tokens": alone["usage"]["input_tokens"],
                     "mode": shared["usage"]["state_reading"], "shared_seconds": t1 - t0, "independent_seconds": t2 - t1,
                     "shared_forward_calls": shared_calls, "independent_forward_calls": len(calls)})
    # Mixed question types over one state.
    mixed = {"state": "I was charged twice for one order and the app keeps crashing when I open my receipts.",
             "questions": {
                 "route": {"type": "choice", "question": "Which team should handle this first?",
                           "options": {"billing": "Charges and refunds", "technical": "App errors", "sales": "New purchases"}},
                 "refund": {"type": "noul", "question": "Is the customer asking for money back?"},
                 "urgency": {"type": "score", "question": "How urgent is this request?",
                             "options": {"1": "Not urgent", "2": "Low", "3": "Medium", "4": "High", "5": "Critical"}}}}
    mixed_shared = scorer.score(mixed, share_state=True, calibrated=False)
    mixed_alone = scorer.score(mixed, share_state=False, calibrated=False)
    refused = False
    try:
        scorer.score({"state": "x", "question": "q", "options": {f"o{i}": f"option {i}" for i in range(65)}})
    except ValueError:
        refused = True
    result = {
        "model": str(args.model), "dtype": "float32" if args.fp32 else "bfloat16", "articles": len(articles), "questions": total,
        "shared_vs_independent_max_probability_difference": max_dp,
        "shared_vs_independent_choice_agreement": agree / total,
        "mean_shared_seconds": sum(r["shared_seconds"] for r in rows) / len(rows),
        "mean_independent_seconds": sum(r["independent_seconds"] for r in rows) / len(rows),
        "mean_input_tokens_shared": sum(r["state_tokens_plus_suffixes"] for r in rows) / len(rows),
        "mean_input_tokens_independent": sum(r["independent_tokens"] for r in rows) / len(rows),
        "all_shared": all(r["mode"] == "shared_state" for r in rows), "per_article": rows,
        "mixed_types_shared": mixed_shared["answers"], "mixed_types_max_probability_difference": max(
            abs(mixed_shared["answers"][q]["probabilities"][k] - mixed_alone["answers"][q]["probabilities"][k])
            for q in mixed["questions"] for k in mixed_shared["answers"][q]["probabilities"]),
        "over_64_options_refused": refused, "output_tokens": mixed_shared["usage"]["output_tokens"],
        "criterion": "pass if max probability difference <= 0.02 and choice agreement >= 0.99",
    }
    result["passed"] = (result["shared_vs_independent_max_probability_difference"] <= .02
                        and result["shared_vs_independent_choice_agreement"] >= .99 and refused and result["all_shared"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_article", "mixed_types_shared")}, indent=1))


if __name__ == "__main__":
    main()
