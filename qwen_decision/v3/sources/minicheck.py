"""MiniCheck synthetic grounding data (2026-10-19): is a claim fully supported by a document?

  python -m qwen_decision.v3.sources.minicheck --fetch   # download the pinned upstream files
  python -m qwen_decision.v3.sources.minicheck           # build data/phase2/minicheck-synth.jsonl

Upstream: lytang/C2D-and-D2C-MiniCheck (Tang et al. 2024, MiniCheck), MIT. 14,395 (claim, document, label)
rows, label 1 = supported.
- C2D (7,076): ~400 Wikipedia claims with cited web articles, decomposed into atomic facts by GPT-3.5; GPT-4
  wrote documents that state every fact (supported by construction), and documents with a sentence removed
  that GPT-4's entailment check found no longer support the claim (unsupported).
- D2C (7,319): ~300 Google News articles from November 2023 on; GPT-4 summarised chunks into claims, and
  chunks with sentences removed were relabelled by an entailment check.
The labels were decided by a generating-model pipeline, so label_origin is model. That is allowed for this
source only: the user's decision of 2026-10-19 adds it to mix.MODEL_ORIGIN_ALLOWED (otherwise tools only).
The other teacher sources (typed-decisions, SII-Decision teacher distributions, yuri_v3) stay out.

Questions: 80% yes/no (a quarter of them asking whether the claim has unsupported content, so the truth
flips), 20% a two-way choice between "supported" and "not supported" texts. Our own phrasings.
Item key (first id segment): a connected group of rows that share a claim, a document, or a document
sentence (one source article's chunks, one Wikipedia claim's documents), so no claim or document text
straddles the dev split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase5 import hub_dir

NAME = "minicheck-synth"
REPO, REVISION = "lytang/C2D-and-D2C-MiniCheck", "1f698000d2f0e39c6830c74d66f793c02a1d9259"
SEED = 2026101921
CHOICE, FLIPPED = .2, .25
MIN_SENTENCE = 60  # normalized characters for a shared sentence to link two documents
SUPPORTED = [
    "Is every part of the claim supported by the document?",
    "Does the document fully support the claim?",
    "Can the claim be verified from the document alone, with nothing in it left unsupported?",
    "Based only on this text, is the statement fully backed up?",
    "Is the claim consistent with and entirely grounded in the document?",
]
UNSUPPORTED = [
    "Does the claim contain anything the document does not support?",
    "Is any part of the statement missing from, or contradicted by, the document?",
    "Would checking this claim against the document find at least one unsupported fact?",
]
CHOICE_ASK = ["How does the document relate to the claim?", "Check the claim against the document.",
              "Is the claim grounded in the document?"]
CHOICE_OPTIONS = [("supported", "Every fact in the claim is supported by the document"),
                  ("not supported", "At least one fact in the claim is not supported by the document")]
STATE_KEYS = [("document", "claim"), ("source", "statement"), ("context", "claim"), ("evidence", "claim to check")]


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def groups(rows: list[dict]) -> list[str]:
    """Connected components over shared claims, documents and long document sentences; one key per row."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for n, r in enumerate(rows):
        union(f"row:{n}", f"claim:{norm(r['claim'])}")
        union(f"row:{n}", f"doc:{norm(r['doc'])}")
        for s in re.split(r"(?<=[.!?])\s+", r["doc"]):
            if len(norm(s)) >= MIN_SENTENCE:
                union(f"row:{n}", f"sent:{norm(s)}")
    roots = [find(f"row:{n}") for n in range(len(rows))]
    return [hashlib.sha1(root.encode()).hexdigest()[:12] for root in roots]


def build(stats: dict):
    rng = random.Random(SEED)
    local = hub_dir(REPO, REVISION)
    for split in ("c2d", "d2c"):
        rows = pq.read_table(next((local / "data").glob(f"{split}-*.parquet"))).to_pylist()
        keys = groups(rows)
        sizes = collections.Counter(keys)
        stats["groups"][split] = {"groups": len(sizes), "largest_group_rows": max(sizes.values()),
                                  "rows_in_groups_over_100": sum(v for v in sizes.values() if v > 100)}
        for n, (r, key) in enumerate(zip(rows, keys)):
            supported = r["label"] == 1
            doc_key, claim_key = rng.choice(STATE_KEYS)
            state = {doc_key: r["doc"].strip(), claim_key: r["claim"].strip()}
            if rng.random() < CHOICE:
                options = list(CHOICE_OPTIONS)
                rng.shuffle(options)
                gold = [k for k, _ in options].index("supported" if supported else "not supported")
                question, target = common.choice(state, rng.choice(CHOICE_ASK), dict(options), gold)
                stats["kinds"]["choice"] += 1
            elif rng.random() < FLIPPED:
                question, target = common.noul(rng.choice(UNSUPPORTED), not supported)
                stats["kinds"]["noul_flipped"] += 1
            else:
                question, target = common.noul(rng.choice(SUPPORTED), supported)
                stats["kinds"]["noul"] += 1
            stats["labels"][f"{split}:{'supported' if supported else 'unsupported'}"] += 1
            yield common.example(NAME, f"{split}-{key}:{n}", "grounded-fact-check", state, question, target, split)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download the pinned upstream files")
    args = p.parse_args()
    if args.fetch:
        local, revision = common.fetch(REPO, ["README.md", "data/*"], revision=REVISION)
        print(json.dumps({"name": NAME, "repo": REPO, "revision": revision, "path": str(local)}))
    else:
        stats = {"groups": {}, "kinds": collections.Counter(), "labels": collections.Counter()}
        meta = {"upstream": f"{REPO}@{REVISION[:12]} (c2d + d2c parquet)", "revision": REVISION, "split": "c2d+d2c (all)",
                "licence": "MIT", "label_origin": "model: GPT-4 generation with GPT-4/entailment-check labels (Tang et al. 2024); "
                "allowed for this source by the user's decision of 2026-10-19 (mix.MODEL_ORIGIN_ALLOWED)",
                "provenance": "C2D: ~400 Wikipedia claims with cited web articles; D2C: ~300 Google News articles (Nov 2023+)",
                "stats": stats}
        common.write(NAME, build(stats), meta)
