"""SargeDev/jev-distill-corpus-v3 rows as v3 training examples.

A row ``{kind, options, target, state, question, ...}`` becomes
``{"state", "question": System One question, "target": probabilities in
options() order, "ordered": keep option order}``.

- noul: targets are [P(false), P(true)], the order fmt.options() uses.
- choice: criteria {option: None}, so each line renders as the option text.
- score: a 0..5 choice whose level order is kept (never shuffled).
- yuri_v1 rows are dropped: their [0.5, 0.5] targets are placeholders, not teacher labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/jev-distill-v3"
DROPPED_SOURCES = {"yuri_v1"}


def convert(row: dict) -> dict | None:
    if row["source"] in DROPPED_SOURCES:
        return None
    options, target = row["options"], [float(p) for p in row["target"]]
    if len(options) != len(target) or len(set(options)) != len(options) or not 2 <= len(options) <= 255:
        return None
    if row["kind"] == "noul":
        if options != ["false", "true"]:
            return None
        question = {"type": "noul", "instructions": row["question"]}
    elif row["kind"] in ("choice", "score"):
        question = {"type": "choice", "instructions": row["question"], "criteria": {o: None for o in options}}
    else:
        return None
    total = sum(target)
    if not total > 0:
        return None
    return {"id": row["id"], "state": row["state"], "question": question, "target": [p / total for p in target],
            "ordered": row["kind"] == "score", "kind": row["kind"], "family": row.get("family"),
            "domain": row.get("domain"), "source": row["source"]}


def load(split: str, limit: int | None = None, seed: int = 0) -> list[dict]:
    """Converted rows of one corpus split; `limit` draws a seeded random subset."""
    with (CORPUS / f"{split}.jsonl").open() as f:
        rows = [r for r in (convert(json.loads(line)) for line in f) if r is not None]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows
