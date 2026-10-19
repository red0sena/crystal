"""BANKING77 train (10,003 utterances) as a 77-option banking-intent choice.

Upstream: legacy-datasets/banking77 (the PolyAI-LDN task-specific-datasets release, CC-BY-4.0) at a
pinned Hub revision; train split only. The index scores all 3,080 test utterances (its raw copy of
test.csv is used here only to drop train utterances identical to test; the plan measured 9).

Options are the 77 category names made readable (underscores removed, lower case); half the
questions list them alphabetically, half shuffled.
"""
from __future__ import annotations

import csv
import json
import random

import pyarrow.parquet as pq

from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

NAME = "banking77-train"
REPO = "legacy-datasets/banking77"
REVISION = "f54121560de48f2852f90be299010d1d6dc612ec"
INDEX_TEST = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/banking77/test.csv"
PHRASINGS = [
    "Which banking support topic is this customer message about?",
    "A customer wrote to their bank's help desk. Pick the category that fits what they are asking for.",
    "Route this customer question to the right banking intent.",
    "Which of these support intents best describes what the customer needs?",
    "Classify the customer's request into one of the bank's support categories.",
    "What is this bank customer trying to get help with?",
]
STATE_KEYS = ["customer_message", "message", "query", "text"]


def readable(label: str) -> str:
    return label.replace("_", " ").lower()


def build(stats: dict):
    local, revision = common.fetch(REPO, ["data/train-*", "README.md"], revision=REVISION)
    table = pq.read_table(next(local.glob("data/train-*.parquet")))
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    assert len(names) == 77
    test = set()
    if INDEX_TEST.exists():
        with INDEX_TEST.open(newline="", encoding="utf-8") as f:
            test = {" ".join(words(r["text"])) for r in csv.DictReader(f)}
    labels = [readable(x) for x in names]
    rng = random.Random(77)
    for i, r in enumerate(table.to_pylist()):
        if " ".join(words(r["text"])) in test:
            stats["dropped_test_duplicates"] += 1
            continue
        options = sorted(labels) if rng.random() < .5 else rng.sample(labels, len(labels))
        state = r["text"] if rng.random() < .3 else {rng.choice(STATE_KEYS): r["text"]}
        question, target = common.choice(state, common.instruction(rng, PHRASINGS), options, options.index(labels[r["label"]]))
        yield common.example(NAME, f"train:{i}", "intent", state, question, target, "train")


if __name__ == "__main__":
    stats = {"dropped_test_duplicates": 0}
    rows = list(build(stats))
    common.write(NAME, rows, {"upstream": REPO, "revision": REVISION, "split": "train", "licence": "CC-BY-4.0", **stats})
