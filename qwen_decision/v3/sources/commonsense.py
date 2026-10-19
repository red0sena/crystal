"""HellaSwag train (4-way continuation) and WinoGrande xl train (2-way blank filling): every train row since
2026-10-18 (15,000 sampled each before; the phase-2 files of that build are what v3-gold/v3-gold2 used).

HellaSwag: Rowan/hellaswag on the Hub, `data/train-*.parquet` at the commit the index pins.
Licence: MIT. The index (catalog 29) and JEVal both use the validation split, so it is never a
training source; it is read here only to drop the train contexts that repeat a validation context
(25, measured in the plan).

WinoGrande: allenai/winogrande on the Hub, `winogrande_xl/train-*.parquet` at the commit the index
pins. Licence: Apache-2.0 (GitHub release; the Hub card is blank). The index (catalog 28) and JEVal
both use validation, which is not read; train has no validation duplicate (measured in the plan).
70% show the sentence with its blank and the two fillers; 30% show the two filled-in sentences.
"""
from __future__ import annotations

import random

import pyarrow.parquet as pq

from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

CAP = None  # 2026-10-18: every train row (was 15,000 sampled each); g4 uses the larger pools

HELLASWAG = "hellaswag-train"
HELLASWAG_REPO = "Rowan/hellaswag"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
CONTINUE = [
    "Which ending continues the passage most sensibly?",
    "Pick the sentence that most plausibly comes next.",
    "What happens next? Choose the most likely continuation.",
    "Select the option that best completes the description.",
    "Only one of these endings is what actually follows; the others were written to mislead. Which one follows?",
]
CONTINUE_STATE_KEYS = [("activity", "context"), ("topic", "story so far"), ("activity", "beginning")]

WINOGRANDE = "winogrande-train"
WINOGRANDE_REPO = "allenai/winogrande"
WINOGRANDE_REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
BLANK = [
    "Who or what does the blank (_) stand for?",
    "Fill in the blank with the option that makes the sentence make sense.",
    "Which choice belongs in place of the underscore?",
    "Pick the option that correctly completes the sentence.",
    "Read the sentence and decide which of the two candidates the _ refers to.",
]
FILLED = [
    "Which version of the sentence makes sense?",
    "Only one of these sentences is logically coherent. Which one?",
    "Choose the sentence in which the right person or thing has been filled in.",
]


def clean(text: str) -> str:
    return " ".join(str(text).split())


def hellaswag(stats: dict):
    path, _ = common.fetch(HELLASWAG_REPO, ["data/train-*", "data/validation-*"], revision=HELLASWAG_REVISION)
    seen = {" ".join(words(clean(r["ctx"]))) for r in pq.read_table(next(path.glob("data/validation-*.parquet")), columns=["ctx"]).to_pylist()}
    rows = pq.read_table(next(path.glob("data/train-*.parquet"))).to_pylist()
    rng = random.Random(2026101629)
    repeats = {i for i, r in enumerate(rows) if " ".join(words(clean(r["ctx"]))) in seen}
    candidates = [i for i in range(len(rows)) if i not in repeats]
    ties = 0
    for i in (candidates if CAP is None else sorted(rng.sample(candidates, CAP))):
        r = rows[i]
        endings = [clean(e) for e in r["endings"]]
        if len(set(endings)) != len(endings):
            ties += 1
            continue
        context = clean(r["ctx"])
        if rng.random() < .4:
            a, b = rng.choice(CONTINUE_STATE_KEYS)
            state = {a: r["activity_label"], b: context}
        else:
            state = context
        question, target = common.choice(state, rng.choice(CONTINUE), endings, int(r["label"]))
        yield common.example(HELLASWAG, f"{i}:{r['ind']}", "commonsense", state, question, target, "train")
    stats.update(train_rows=len(rows), validation_context_repeats_dropped=len(repeats), duplicate_endings_skipped=ties,
                 sampled=f"{CAP} (seed 2026101629)")


def winogrande(stats: dict):
    path, _ = common.fetch(WINOGRANDE_REPO, ["winogrande_xl/train-*"], revision=WINOGRANDE_REVISION)
    rows = pq.read_table(next(path.glob("winogrande_xl/train-*.parquet"))).to_pylist()
    rng = random.Random(2026101628)
    skipped = 0
    for i in (range(len(rows)) if CAP is None else sorted(rng.sample(range(len(rows)), CAP))):
        r = rows[i]
        options = [r["option1"].strip(), r["option2"].strip()]
        if options[0] == options[1] or r["sentence"].count("_") != 1 or r["answer"] not in ("1", "2"):
            skipped += 1
            continue
        gold = int(r["answer"]) - 1
        if rng.random() < .3:
            filled = [r["sentence"].replace("_", o) for o in options]
            question, target = common.choice("", rng.choice(FILLED), filled, gold)
            state = ""
        else:
            state = r["sentence"] if rng.random() < .7 else {"sentence": r["sentence"]}
            question, target = common.choice(state, rng.choice(BLANK), options, gold)
        yield common.example(WINOGRANDE, str(i), "commonsense", state, question, target, "train")
    stats.update(train_rows=len(rows), malformed_skipped=skipped, sampled=f"{CAP} (seed 2026101628)")


if __name__ == "__main__":
    meta = {"upstream": HELLASWAG_REPO, "revision": HELLASWAG_REVISION, "split": "train", "licence": "MIT"}
    common.write(HELLASWAG, hellaswag(meta), meta)
    meta = {"upstream": f"{WINOGRANDE_REPO} (winogrande_xl)", "revision": WINOGRANDE_REVISION, "split": "train",
            "licence": "Apache-2.0 (GitHub release; Hub card blank)"}
    common.write(WINOGRANDE, winogrande(meta), meta)
