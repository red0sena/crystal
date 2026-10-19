"""Fit per-bucket temperatures on held-out development logits (evaluate_v2.py
calibration pass) by minimizing cross-entropy against the true labels.

Buckets: choice by option count (2 / 3-5 / 6-10 / 11-26 / 27-64), noul, score.
Soft-target rows (STS-B) use their target distribution; all others one-hot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import qwen_decision.core.decision_format as fmt


def log_softmax(z, t):
    m = max(v / t for v in z)
    s = math.log(sum(math.exp(v / t - m) for v in z)) + m
    return [v / t - s for v in z]


def nll(rows, t):
    total = 0.
    for r in rows:
        lp = log_softmax(r["logits"], t)
        if r.get("target"):
            total -= sum(p * lp[r["names"].index(k)] for k, p in r["target"].items())
        else:
            total -= lp[r["gold"]]
    return total / len(rows)


def golden(f, lo=math.log(.05), hi=math.log(20.), iterations=60):
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="calibration.jsonl from evaluate_v2.py")
    parser.add_argument("--output", type=Path, required=True, nargs="+")
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()
    with args.predictions.open() as f:
        rows = [json.loads(x) for x in f]
    buckets = defaultdict(list)
    for r in rows:
        buckets[fmt.calibration_key(r["qtype"], r["n_options"])].append(r)
    temperatures, fit = {}, {}
    # log(T) 공간에서 개발 cross-entropy를 최소화한다. 표본 200개 미만인 bucket은 T=1.
    # 낮은 calibration loss가 더 높은 argmax 정확도를 뜻하지는 않는다.
    for key, rr in sorted(buckets.items()):
        before = nll(rr, 1.)
        if len(rr) < args.min_rows:
            fit[key] = {"n": len(rr), "temperature": 1.0, "nll_before": before, "note": "too few rows; T=1"}
            temperatures[key] = 1.0
            continue
        t = math.exp(golden(lambda u: nll(rr, math.exp(u))))
        temperatures[key] = t
        fit[key] = {"n": len(rr), "temperature": t, "nll_before": before, "nll_after": nll(rr, t),
                    "tasks": sorted({r["task"] for r in rr})}
    result = {"temperatures": temperatures, "fit": fit, "objective": "cross-entropy against true labels",
              "source": str(args.predictions), "source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              "partition": "development rows outside the model-selection subset; test never used"}
    for path in args.output:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (round(v["temperature"], 3), v["n"]) for k, v in fit.items()}))


if __name__ == "__main__":
    main()
