"""Public vision suite baseline (2026-10-31): the base model and wise80 g6+g7e2, before any image training.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.vision_baseline rows
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.vision_baseline run base-chat [--limit N]
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.vision_baseline score

The suite is built by `decision-index suite rebuild --vision --work data/vision-work`. Winoground is gated and this
account has no access yet, so `rows` writes the kit's rows (rows.select / rows.to_row) for the sets that were built
and skips the rest; the kit's scorer then renormalizes the index over the benchmarks present (complete=false).
The Mind2Web box labels use Liberation Sans Bold instead of Arial Bold, so its image bytes differ from the board's.
Four runs: each model in its own prompt (base: chat reference path; wise80: v3 decide format) and in the other's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_index.vision import rows as vrows
from decision_index.vision import spec
from decision_index.vision.run import run as kit_run
from decision_index.vision.score import score_run

ROOT = Path("data/vision-work/vision")
ROWS = ROOT / "rows-built.jsonl"
OUT = Path("runs/vision")
BASE, WISE80 = Path("models/qwen3.5-0.8b"), Path("runs/v3/wise80-g6-g7e2")
RUNS = {"base-chat": (BASE, "chat"), "wise80-decide": (WISE80, "decide"),
        "base-decide": (BASE, "decide"), "wise80-chat": (WISE80, "chat")}
ENGINE = "qwen_decision.v3.vision_engine:VisionEngine"


def write_rows() -> None:
    built = [s for s in spec.SETS if (ROOT / "full" / s / "items.jsonl").exists()]
    rows = [vrows.to_row(ROOT, it) for s in built for it in vrows.select(s, vrows.load_items(ROOT, s))]
    with open(ROWS, "w", encoding="utf-8") as f:
        f.writelines(vrows.dumps(r) + "\n" for r in rows)
    counts = {}
    for r in rows:
        counts[r["benchmark"]] = counts.get(r["benchmark"], 0) + 1
    print(json.dumps({"rows": len(rows), "sets_built": built, "sets_missing": [s for s in spec.SETS if s not in built],
                      "per_benchmark": counts}, indent=1))


def run(name: str, limit: int | None, out: Path | None = None) -> None:
    model, prompt = RUNS[name]
    kit_run(ENGINE, {"model": str(model), "prompt": prompt}, ROOT, out or OUT / name, rows_path=ROWS, workers=1,
            limit=limit)


def score() -> None:
    total = sum(b["weight"] for b in spec.BENCHMARKS)
    table = {}
    for name in RUNS:
        results = OUT / name / "results.jsonl"
        if not results.exists():
            continue
        s = score_run(ROWS, results, OUT / name)
        present = sum(b["weight"] for b in s["benchmarks"].values())
        table[name] = {"index_over_present": round(s["public_vision_index"], 2),
                       "index_missing_as_zero": round(s["public_vision_index"] * present / total, 2),
                       "missing_rows": s["missing_rows"],
                       **{b: round(v["skill"], 3) for b, v in s["benchmarks"].items()}}
    (OUT / "summary.json").write_text(json.dumps(table, indent=1) + "\n")
    print(json.dumps(table, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["rows", "run", "score"])
    p.add_argument("name", nargs="?", choices=list(RUNS))
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.command == "rows":
        write_rows()
    elif a.command == "run":
        run(a.name, a.limit, a.out)
    else:
        score()


if __name__ == "__main__":
    main()
