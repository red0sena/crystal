"""VAST train + dev as 3-way stance choices, and iSarcasmEval train as sarcasm decisions.

VAST (Allaway & McKeown 2020): emilyallaway/zero-shot-stance `data/VAST/vast_train.csv` and
`vast_dev.csv` at the commit the index pins (the suite rebuild already cloned it; read-only).
Licence: no licence file in the repo (research release; the posts are NYT comments). The index
scores `vast_test.csv` (catalog 41); `vast_test-sentswap.csv` is built from the test posts and is
never read. Train/dev posts share none of the test posts (measured). Rows are grouped by (post,
topic); the topic is the annotated `new_topic`, except for the synthetic-neutral rows (type 4),
whose topic is `topic_str`. When duplicates of a pair disagree, the target is their label mix.

iSarcasmEval (SemEval-2022 Task 6): iabufarha/iSarcasmEval `train/train.En.csv` and
`train/train.Ar.csv` at the commit the index pins (read-only). Licence: MIT. The index scores the
`test/` files (catalog 40); `third-party annotations/` holds test items and is never read.
Three casts, all from the train columns: A, is the text sarcastic (yes/no, both polarities, En
and Ar); C, which of a sarcastic text and its author's plain rephrase is the sarcastic one (both
polarities); B (English), per ironic-speech category yes/no, for every sarcastic text and a
quarter of the others.
"""
from __future__ import annotations

import collections
import csv
import random

from qwen_decision.v3.sources import common

RAW = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw"

# --- VAST ---------------------------------------------------------------------------------------------
VAST = "vast-train"
VAST_COMMIT = "e7c4775182b184730f350995f8260579c9e066fe"
# Option sets in (con, pro, neutral) order, matching the upstream labels 0, 1, 2.
STANCES = [
    [("against", "the author opposes or criticises it"), ("in favor", "the author supports or defends it"),
     ("neutral", "the author takes no clear side on it")],
    [("con", None), ("pro", None), ("neutral", None)],
    [("opposes", None), ("supports", None), ("neither", None)],
    [("negative stance", "argues against the topic"), ("positive stance", "argues for the topic"),
     ("no stance", "mentions or ignores the topic without taking a side")],
]
STANCE_WITH_TOPIC = [
    "What position does the comment take on \"{t}\"?",
    "Is the author of this post for or against {t}, or neutral?",
    "How does the writer feel about the topic \"{t}\"?",
    "Classify the post's stance toward: {t}",
    "Judging from the comment, does its author back {t}, oppose it, or neither?",
]
STANCE_GENERIC = [
    "Decide the stance the post takes toward the given topic.",
    "Read the comment and say whether it is for, against, or neutral on the topic.",
    "Which attitude toward the topic does the text express?",
]
STANCE_STATE_KEYS = [("comment", "topic"), ("post", "target"), ("text", "subject")]


def vast(stats: dict):
    rows = []
    for split in ("train", "dev"):
        with (RAW / f"vast/data/VAST/vast_{split}.csv").open(newline="", encoding="utf-8") as f:
            rows += [dict(r, _split=split) for r in csv.DictReader(f)]
    groups = collections.defaultdict(list)
    for r in rows:
        topic = r["topic_str"] if r["type_idx"] == "4" else r["new_topic"]
        groups[(r["post"], " ".join(topic.split()))].append(r)
    rng = random.Random(2026101641)
    mixed = 0
    for (post, topic), group in groups.items():
        counts = collections.Counter(int(r["label"]) for r in group)
        mixed += len(counts) > 1
        labels = rng.choice(STANCES)
        order = list(range(3))
        rng.shuffle(order)
        criteria = {labels[i][0]: labels[i][1] for i in order}
        target = [counts[i] / len(group) for i in order]
        if rng.random() < .6:
            state = post if rng.random() < .6 else {rng.choice(STANCE_STATE_KEYS)[0]: post}
            instructions = rng.choice(STANCE_WITH_TOPIC).format(t=topic)
        else:
            a, b = rng.choice(STANCE_STATE_KEYS)
            state = {a: post, b: topic}
            instructions = rng.choice(STANCE_GENERIC)
        question, target = common.choice(state, instructions, criteria, target=target)
        split = "+".join(sorted({r["_split"] for r in group}))
        yield common.example(VAST, "|".join(sorted(r["new_id"] for r in group)), "stance", state, question, target, split)
    stats.update(source_rows=len(rows), post_topic_pairs=len(groups), pairs_with_mixed_labels=mixed)


# --- iSarcasmEval -------------------------------------------------------------------------------------
ISARCASM = "isarcasm-train"
ISARCASM_COMMIT = "dfc708b53bde1bb571abfb5692f63231c2232195"
SARCASTIC = [
    "Did the author mean this text sarcastically?",
    "Is this post sarcastic?",
    "The writer is being sarcastic here.",
    "Does the text say something other than what its author really means, in a mocking way?",
]
SINCERE = [
    "Is this text meant literally, without sarcasm?",
    "The author means exactly what this post says.",
    "Is the writer being sincere rather than sarcastic?",
    "This text contains no sarcasm.",
]
INTENT = ["Is the following text sarcastic or sincere?", "How did the author mean this post?",
          "Classify what the writer intended."]
PAIR_SARCASTIC = [
    "Both texts carry the same message, but only one of them is sarcastic. Which one?",
    "One of these was written sarcastically and the other states the same point plainly. Pick the sarcastic one.",
    "Which version uses sarcasm?",
]
PAIR_PLAIN = [
    "Both texts carry the same message. Which one says it plainly, without sarcasm?",
    "One of these is a sarcastic post and the other is its literal rewording. Pick the literal one.",
]
CATEGORIES = {
    "sarcasm": "sarcasm (mocking irony aimed at a person or group)",
    "irony": "irony (saying the opposite of what is meant, without a clear victim)",
    "satire": "satire (ridiculing something, often public life, by humorous imitation)",
    "understatement": "understatement (making something sound smaller or less serious than it is)",
    "overstatement": "overstatement (exaggerating for effect)",
    "rhetorical_question": "a rhetorical question (a question asked to make a point, not to get an answer)",
}
CATEGORY = [
    "Does this text use {c}?",
    "Is {c} present in the post?",
    "The text is an example of {c}.",
    "Judge only this one category of ironic speech: does the post contain {c}?",
]


def isarcasm(stats: dict):
    rng = random.Random(2026101640)
    counts = collections.Counter()
    for lang, text_col in (("En", "tweet"), ("Ar", "text")):
        with (RAW / f"repos/isarcasm/train/train.{lang}.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            text = (r[text_col] or "").strip()
            if not text or r["sarcastic"] not in ("0", "1"):
                counts["skipped_empty"] += 1
                continue
            sarcastic = r["sarcastic"] == "1"
            state = text if rng.random() < .7 else {"text": text}
            if rng.random() < .25:
                keys = ["sarcastic", "not sarcastic"]
                rng.shuffle(keys)
                question, target = common.choice(state, rng.choice(INTENT), keys,
                                                 keys.index("sarcastic" if sarcastic else "not sarcastic"))
            elif rng.random() < .5:
                question, target = common.noul(rng.choice(SARCASTIC), sarcastic)
            else:
                question, target = common.noul(rng.choice(SINCERE), not sarcastic)
            counts[f"A_{lang}"] += 1
            yield common.example(ISARCASM, f"A:{lang}:{i}", "sarcasm", state, question, target, "train")
            rephrase = (r.get("rephrase") or "").strip()
            if sarcastic and rephrase and rephrase != text:
                pair = [text, rephrase]
                rng.shuffle(pair)
                plain = rng.random() < .3
                gold = pair.index(rephrase if plain else text)
                instructions = rng.choice(PAIR_PLAIN if plain else PAIR_SARCASTIC)
                question, target = common.choice("", instructions, pair, gold)
                counts[f"C_{lang}"] += 1
                yield common.example(ISARCASM, f"C:{lang}:{i}", "sarcasm", "", question, target, "train")
            if lang == "En" and (sarcastic or rng.random() < .25):
                names = list(CATEGORIES) if sarcastic else [rng.choice(list(CATEGORIES))]
                for name in names:
                    if r[name] not in ("0", "1"):
                        continue
                    question, target = common.noul(rng.choice(CATEGORY).format(c=CATEGORIES[name]), r[name] == "1")
                    counts["B_En"] += 1
                    yield common.example(ISARCASM, f"B:{lang}:{i}:{name}", "sarcasm", state, question, target, "train")
    stats.update(questions=dict(counts))


if __name__ == "__main__":
    meta = {"upstream": "emilyallaway/zero-shot-stance (data/VAST)", "revision": VAST_COMMIT, "split": "train+dev",
            "licence": "no licence file (research release; NYT comments)"}
    common.write(VAST, vast(meta), meta)
    meta = {"upstream": "iabufarha/iSarcasmEval (train/train.En.csv, train/train.Ar.csv)", "revision": ISARCASM_COMMIT,
            "split": "train", "licence": "MIT"}
    common.write(ISARCASM, isarcasm(meta), meta)
