"""Science and commonsense multiple choice from train splits only, in each dataset's native options.

Upstream (revisions pinned in the manifests; cap 10,000 rows per source):
  - arc-train: allenai/ai2_arc, ARC-Easy and ARC-Challenge train (2,251 + 1,119). CC-BY-SA-4.0.
  - openbookqa-train: allenai/openbookqa, config main, train (4,957). Licence: Hub says "unknown";
    the AllenAI release is Apache-2.0 (plan lists it as unverified).
  - sciq-train: allenai/sciq, train (11,679). CC-BY-NC-3.0 (non-commercial; the plan flags it gray).
    SciQ has no option order, so the gold and its three distractors are shuffled with a seed; a third
    of the rows carry the support paragraph as context.
  - commonsenseqa-train: tau/commonsense_qa, train (9,741). MIT.

Overlap: JEVal samples ARC from test and CommonsenseQA from dev, and MMLU-Pro (index and JEVal) is
built from MMLU test plus STEM sites; none of those splits is read here (fetch patterns take train
files only). The filter checks every row against the index and JEVal, which also catches the same
question reappearing inside another benchmark.
"""
from __future__ import annotations

import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common, mcq_util

CAP = 10_000
SCIENCE = [
    "Pick the option that correctly answers the science question.",
    "Which answer choice is right?",
    "Using general science knowledge, select the correct option.",
    "Which of the choices best answers or completes the question?",
    "Select the option a science teacher would mark as correct.",
]
COMMONSENSE = [
    "Which option is the most sensible answer to the question?",
    "Using everyday common sense, pick the best answer.",
    "Which choice fits the question best?",
    "Select the answer most people would give to this question.",
    "Which of these options answers the question most reasonably?",
]
KEYS = ("question", "query", "item")


def labelled(rows, name, family, phrasings, rng, keys=KEYS, prefix=""):
    """Rows with {'choices': {'text', 'label'}, 'answerKey'} in a seeded order, capped."""
    kept = 0
    for r in mcq_util.capped(rng, rows, len(rows)):
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] not in labels:
            continue
        question = r.get("question") or r.get("question_stem")
        e = mcq_util.cast(name, prefix + r["id"], family, "train", rng, phrasings, question, texts,
                          labels.index(r["answerKey"]), keys=keys)
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


def arc(local):
    rows = []
    for config in ("ARC-Easy", "ARC-Challenge"):
        for r in pq.read_table(next(local.glob(f"{config}/train-*.parquet"))).to_pylist():
            r["id"] = f"{config}/{r['id']}"
            rows.append(r)
    yield from labelled(rows, "arc-train", "science-mcq", SCIENCE, random.Random(3370))


def openbookqa(local):
    rows = pq.read_table(next(local.glob("main/train-*.parquet"))).to_pylist()
    yield from labelled(rows, "openbookqa-train", "science-mcq", SCIENCE, random.Random(4957),
                        keys=("question", "statement", "prompt"))


def commonsenseqa(local):
    rows = pq.read_table(next(local.glob("data/train-*.parquet"))).to_pylist()
    yield from labelled(rows, "commonsenseqa-train", "commonsense-mcq", COMMONSENSE, random.Random(9741))


def sciq(local):
    rows = pq.read_table(next(local.glob("data/train-*.parquet"))).to_pylist()
    rng = random.Random(11679)
    order = list(range(len(rows)))
    rng.shuffle(order)
    kept = 0
    for i in order:
        r = rows[i]
        options = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        rng.shuffle(options)
        support = (r.get("support") or "").strip()
        context = support if support and rng.random() < 1 / 3 else None
        e = mcq_util.cast("sciq-train", str(i), "science-mcq", "train", rng, SCIENCE, r["question"], options,
                          options.index(r["correct_answer"]), context=context, keys=KEYS,
                          context_keys=("support", "passage", "background"))
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


SOURCES = [
    ("arc-train", "allenai/ai2_arc", ["ARC-Easy/train-*", "ARC-Challenge/train-*"], arc, "CC-BY-SA-4.0",
     "ARC-Easy + ARC-Challenge"),
    ("openbookqa-train", "allenai/openbookqa", ["main/train-*"], openbookqa,
     "unknown on the Hub (AllenAI release Apache-2.0; unverified)", "main"),
    ("sciq-train", "allenai/sciq", ["data/train-*"], sciq, "CC-BY-NC-3.0 (non-commercial)", "default"),
    ("commonsenseqa-train", "tau/commonsense_qa", ["data/train-*"], commonsenseqa, "MIT", "default"),
]

if __name__ == "__main__":
    for name, repo, patterns, builder, licence, config in SOURCES:
        local, revision = common.fetch(repo, patterns)
        common.write(name, builder(local), {"upstream": f"{repo} ({config})", "revision": revision, "split": "train",
                                            "licence": licence, "cap": CAP, "cast": "native multiple choice"})
