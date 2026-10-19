"""Per-source composition check of the training data: what each source teaches and how varied it is.

  python -m qwen_decision.v3.data_composition --out runs/data-audit

For every corpus source and phase-2 source: size, question types, option counts, how many distinct
instructions (digits masked) and states there are, duplicate states, the gold option's position in the
stored order, the soft target's shape (top probability, entropy), noul yes-rate, non-ASCII share, and
for the corpus its domain spread. Writes composition.json and samples.json (a few raw examples each).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from qwen_decision.v3 import jev_data
from qwen_decision.v3.sources import common

ROOT = Path(__file__).resolve().parents[2]
DIGITS = re.compile(r"\d+")


def text_of(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)


def describe(rows: list[dict], domains: Counter | None = None) -> dict:
    n = len(rows)
    instructions = Counter(DIGITS.sub("#", r["question"].get("instructions") or "") for r in rows)
    states = Counter(text_of(r["state"]) for r in rows)
    types = Counter("noul" if r["question"]["type"] == "noul" else "ordered" if r.get("ordered") else "choice" for r in rows)
    k = [len(r["target"]) for r in rows]
    top, ent, gold_pos, yes = [], [], Counter(), []
    for r in rows:
        t = r["target"]
        g = max(range(len(t)), key=t.__getitem__)
        top.append(t[g])
        ent.append(-sum(p * math.log(p) for p in t if p > 0) / math.log(len(t)))
        if r["question"]["type"] == "noul":
            yes.append(t[1])
        else:
            gold_pos["first" if g == 0 else "last" if g == len(t) - 1 else "middle"] += 1
    choice_n = sum(gold_pos.values())
    texts = [text_of(r["state"]) + (r["question"].get("instructions") or "") for r in rows]
    non_ascii = sum(any(ord(c) > 127 for c in s) for s in texts) / n
    out = {"rows": n, "types": {t: c / n for t, c in types.items()},
           "options_mean": sum(k) / n, "options_max": max(k),
           "distinct_instructions": len(instructions), "top_instruction_share": instructions.most_common(1)[0][1] / n,
           "distinct_states_share": len(states) / n, "rows_with_repeated_state": sum(c for c in states.values() if c > 1) / n,
           "target_top_mean": sum(top) / n, "target_hard_share": sum(p >= .99 for p in top) / n,
           "target_normalized_entropy_mean": sum(ent) / n,
           "gold_position": {p: c / choice_n for p, c in gold_pos.items()} if choice_n else {},
           "noul_yes_mean": sum(yes) / len(yes) if yes else None,
           "noul_yes_over_half": sum(p >= .5 for p in yes) / len(yes) if yes else None,
           "non_ascii_share": non_ascii}
    if domains:
        out["domains"] = len(domains)
        out["top_domains"] = domains.most_common(8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "runs/data-audit")
    p.add_argument("--samples", type=int, default=4)
    args = p.parse_args()
    raw_domains = defaultdict(Counter)
    with (jev_data.CORPUS / "train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            raw_domains[r["source"]][f"{r.get('family')}/{r.get('domain')}"] += 1
    by = defaultdict(list)
    for r in jev_data.load("train"):
        by["corpus:" + r["source"]].append(r)
    names = sorted(f.name[:-6] for f in (ROOT / "data/phase2").glob("*.jsonl"))
    for r in common.load(names):
        by["phase2:" + r["source"]].append(r)
    report, samples = {}, {}
    rng = random.Random(0)
    for name, rows in sorted(by.items()):
        report[name] = describe(rows, raw_domains.get(name.split(":", 1)[1]) if name.startswith("corpus:") else None)
        samples[name] = [{"state": text_of(r["state"])[:600], "question": r["question"].get("instructions"),
                          "options": list(r["question"].get("criteria") or {})[:8], "target": [round(t, 3) for t in r["target"][:8]]}
                         for r in rng.sample(rows, min(args.samples, len(rows)))]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "composition.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (args.out / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n")
    print(f"{'source':34}{'rows':>8}{'instr':>7}{'top_i':>7}{'uniqS':>7}{'hard':>6}{'top_p':>7}{'ent':>6}{'gold1st':>8}{'yes':>6}{'k':>6}")
    for name, s in report.items():
        print(f"{name[:34]:34}{s['rows']:8}{s['distinct_instructions']:7}{s['top_instruction_share']:7.2f}"
              f"{s['distinct_states_share']:7.2f}{s['target_hard_share']:6.2f}{s['target_top_mean']:7.2f}"
              f"{s['target_normalized_entropy_mean']:6.2f}{s['gold_position'].get('first', 0):8.2f}"
              f"{(s['noul_yes_over_half'] if s['noul_yes_over_half'] is not None else float('nan')):6.2f}{s['options_mean']:6.1f}")


if __name__ == "__main__":
    main()
