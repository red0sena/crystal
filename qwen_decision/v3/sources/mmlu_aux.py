"""MMLU auxiliary_train as native four-way multiple choice; a seeded 20,000-row subset.

Upstream: cais/mmlu, config auxiliary_train (99,842 rows), fetched with the pattern
auxiliary_train/train-* only. Revision pinned in the manifest. Licence: MIT.

What it is: Hendrycks et al.'s auxiliary training pool, drawn from the train splits of ARC, MC_TEST,
OBQA and RACE (most rows are RACE passages with their question appended), not from the MMLU subject
tests. Overlap: MMLU-Pro (index catalog 57 and JEVal) is MMLU test (filtered) plus TheoremQA,
SciBench and STEM-site items; none of that is read here, but the filter drops any row sharing text
with MMLU-Pro or any other index/JEVal item, and the manifest records the drop rate. ARC and OBQA
train rows also appear in science_mcq; that is duplication across our sources, not contamination.
"""
from __future__ import annotations

import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common, mcq_util

NAME = "mmlu-aux-train"
REPO = "cais/mmlu"
CAP = 20_000
PHRASINGS = [
    "Select the correct answer to the question in the state.",
    "Which of the listed options is right?",
    "Pick the choice that answers the question correctly.",
    "Read the state, then choose the option that is true or best supported.",
    "Which answer would be marked correct on this exam question?",
]
KEYS = ("question", "exam_question", "item")


def build(local):
    rows = [r["train"] for r in pq.read_table(next(local.glob("auxiliary_train/train-*.parquet"))).to_pylist()]
    rng = random.Random(99842)
    order = list(range(len(rows)))
    rng.shuffle(order)
    kept = 0
    for i in order:
        r = rows[i]
        e = mcq_util.cast(NAME, str(i), "knowledge-mcq", "auxiliary_train", rng, PHRASINGS, r["question"],
                          list(r["choices"]), int(r["answer"]), keys=KEYS)
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


if __name__ == "__main__":
    local, revision = common.fetch(REPO, ["auxiliary_train/train-*"])
    common.write(NAME, build(local), {"upstream": f"{REPO} (auxiliary_train)", "revision": revision,
                                      "split": "auxiliary_train", "licence": "MIT", "cap": CAP,
                                      "cast": "native 4-way choice"})
