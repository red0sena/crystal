"""Distribution of a training mix: area, source, question type, option count, length, label kind, difficulty.

  python -m qwen_decision.v3.mix_report --mix data/mix/v3-main.jsonl [--probe models/d3-lite --per-source 100]

Lengths are rendered prompt tokens (format.py, as the engine renders them; training augmentation can add
padded options on top). Difficulty comes from a reference model run on a per-source sample (--probe): the
probability it gives the target's top option, binned easy (>= .9), medium (.5-.9) and hard (< .5); for
soft-target rows the top option is the target's argmax. The probe is a reference point, not ground truth:
d3-lite is itself a 0.8B decision model, so "hard" means hard for a model of our size.
Writes <mix>.report.json next to the mix.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
from transformers import AutoTokenizer

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.mix import area_of

ROOT = Path(__file__).resolve().parents[2]
LENGTH_BINS = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def bin_of(n: int, edges: list[int]) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if n < hi:
            return f"{lo}-{hi}"
    return f">={edges[-1]}"


def lengths(tokenizer, rows) -> list[int]:
    codes, _ = fmt.answer_codes(tokenizer)
    texts = [fmt.render(tokenizer, r["state"], r["question"], codes[:len(r["target"])]) for r in rows]
    out = []
    for i in range(0, len(texts), 512):
        out += [len(x) for x in tokenizer(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
    return out


def probe(model: Path, rows: list[dict], per_source: int, seed: int) -> dict[str, float]:
    from qwen_decision.v3.engine import Scorer

    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    rng = random.Random(seed)
    sample = [r for s in sorted(by) for r in rng.sample(by[s], min(per_source, len(by[s])))]
    scorer = Scorer(model, attention="noncausal")  # d3 checkpoints carry no v3_config; d3 runs noncausal
    out = {}
    for i in range(0, len(sample), 256):
        chunk = sample[i:i + 256]
        for r, s in zip(chunk, scorer.score_many([(r["state"], {"q": r["question"]}) for r in chunk])):
            out[r["id"]] = s["q"][max(range(len(r["target"])), key=r["target"].__getitem__)]
        print(f"probe {i + len(chunk)}/{len(sample)}", flush=True)
    return out


def share(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {k: round(v / total, 4) for k, v in counter.most_common()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", type=Path, required=True)
    p.add_argument("--probe", type=Path, help="reference model for difficulty (e.g. models/d3-lite)")
    p.add_argument("--per-source", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.mix.open()]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen3.5-0.8b")
    L = lengths(tokenizer, rows)
    report = {"rows": len(rows), "tokens_total": int(sum(L))}
    area, source, kind, k, label, lang, length = (Counter() for _ in range(7))
    area_tokens, area_len = Counter(), defaultdict(list)
    for r, n in zip(rows, L):
        a = area_of(r["source"])
        area[a] += 1
        area_tokens[a] += n
        area_len[a].append(n)
        source[r["source"]] += 1
        kind["noul" if r["question"]["type"] == "noul" else "ordered" if r.get("ordered") else "choice"] += 1
        k[bin_of(len(r["target"]), [2, 3, 5, 10, 17, 65, 161, 256])] += 1
        label["soft" if max(r["target"]) < .99 else "hard"] += 1
        text = json.dumps(r["state"], ensure_ascii=False) + (r["question"].get("instructions") or "")
        lang["non-ascii" if sum(ord(c) > 127 for c in text) > .1 * len(text) else "mostly ascii"] += 1
        length[bin_of(n, LENGTH_BINS)] += 1
    report |= {"area_rows": share(area), "area_tokens": share(area_tokens),
               "area_length_p50_p90_max": {a: [int(np.percentile(v, 50)), int(np.percentile(v, 90)), int(max(v))]
                                          for a, v in area_len.items()},
               "source_rows": dict(source.most_common()), "type": share(kind), "options": share(k), "label": share(label),
               "language": share(lang), "length_rows": {b: round(length[b] / len(rows), 4) for b in
                                                        [bin_of(e, LENGTH_BINS) for e in LENGTH_BINS[:-1]] if length[b]}}
    if args.probe:
        p_gold = probe(args.probe, rows, args.per_source, args.seed)
        by_area, by_source = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["id"] in p_gold:
                by_area[area_of(r["source"])].append(p_gold[r["id"]])
                by_source[r["source"]].append(p_gold[r["id"]])

        def bins(v):
            v = np.array(v)
            return {"easy": round(float((v >= .9).mean()), 3), "medium": round(float(((v >= .5) & (v < .9)).mean()), 3),
                    "hard": round(float((v < .5).mean()), 3), "mean_p": round(float(v.mean()), 3), "n": len(v)}
        report["difficulty_probe"] = str(args.probe)
        report["difficulty_by_area"] = {a: bins(v) for a, v in sorted(by_area.items())}
        report["difficulty_by_source"] = {s: bins(v) for s, v in sorted(by_source.items())}
    out = args.mix.with_suffix(".report.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "tokens_total", "area_rows", "type", "label", "length_rows")}, indent=1))


if __name__ == "__main__":
    main()
