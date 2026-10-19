"""MedMCQA train as native four-way multiple choice (explanations stripped); a seeded 10,000-row subset.

Upstream: openlifescienceai/medmcqa, split train (182,822 questions from Indian AIIMS/NEET-PG
entrance exams), fetched with the pattern data/train-* only. Revision pinned in the manifest.
Licence: Apache-2.0.

Overlap: MedMCQA is in neither the index nor JEVal (JEVal's medical items are MedQA test, a USMLE
set). The filter still checks every row; it also drops rows whose options repeat text found in the
index or JEVal. Rows with a missing or repeated option are skipped.
"""
from __future__ import annotations

import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common, mcq_util

NAME = "medmcqa-train"
REPO = "openlifescienceai/medmcqa"
CAP = 10_000
PHRASINGS = [
    "Which option is correct for this medical exam question?",
    "Using medical knowledge, select the right answer.",
    "Pick the choice an examiner would accept as correct.",
    "Which of the options answers the clinical question correctly?",
    "Select the correct option for the medicine question in the state.",
]
KEYS = ("question", "exam_question", "clinical_question")


def build(local):
    rows = pq.read_table(next(local.glob("data/train-*.parquet")), columns=["id", "question", "opa", "opb", "opc", "opd", "cop"]).to_pylist()
    rng = random.Random(182822)
    order = list(range(len(rows)))
    rng.shuffle(order)
    kept = 0
    for i in order:
        r = rows[i]
        if r["cop"] not in (0, 1, 2, 3):
            continue
        options = [r[k] or "" for k in ("opa", "opb", "opc", "opd")]
        e = mcq_util.cast(NAME, r["id"], "knowledge-mcq", "train", rng, PHRASINGS, r["question"] or "", options,
                          int(r["cop"]), keys=KEYS)
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


if __name__ == "__main__":
    local, revision = common.fetch(REPO, ["data/train-*"])
    common.write(NAME, build(local), {"upstream": REPO, "revision": revision, "split": "train",
                                      "licence": "Apache-2.0", "cap": CAP, "cast": "native 4-way choice"})
