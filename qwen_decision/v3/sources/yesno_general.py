"""New-domain yes/no judgement: BoolQ, StrategyQA, PubMedQA (labeled) and SQuAD 2.0 answerability.

All four are Hub datasets fetched at a pinned commit; at most 10,000 questions each. Every yes/no
source mixes three casts so the model has to read what is asked: a direct yes/no (true = "yes"),
an inverted yes/no (true = "no", e.g. "is the answer no?"), and a yes/no choice.

BoolQ: google/boolq train (9,427). Licence: CC-BY-SA-3.0. Not in the index; JEVal's boolq task
samples 200 items from train + validation, and the filter drops those (and any train question
that shares their passage). Validation is not used.

StrategyQA: ChilleD/StrategyQA train (1,603; a labelled re-split of the original StrategyQA
train). Licence: MIT. Not in the index or JEVal. No passage: it tests broad knowledge. The
re-split's test half is not used.

PubMedQA: qiaojin/PubMedQA pqa_labeled (1,000 expert-labelled). Licence: MIT. Not in the index or
JEVal. 3-way yes / maybe / no choice (ordered) over the abstract without its conclusion
(`long_answer` would give the answer away).

SQuAD 2.0: rajpurkar/squad_v2 train (130,319). Licence: CC-BY-SA-4.0. Not in the index or JEVal.
Answerability: can the question be answered from the paragraph? 5,000 answerable and 5,000
unanswerable questions, at most two per paragraph.
"""
from __future__ import annotations

import collections
import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

CAP = 10_000
YES_NO = [{"yes": None, "no": None}, {"yes": "the answer is yes", "no": "the answer is no"}]


def question_text(q: str) -> str:
    q = " ".join(q.split()).rstrip("?").strip()
    return q[:1].upper() + q[1:] + "?"


def yes_no(rng: random.Random, state, answer: bool, direct: list[str], inverted: list[str], choice: list[str], **fields):
    """A direct, inverted or choice-form yes/no question whose gold is `answer` (True = yes)."""
    roll = rng.random()
    if roll < .4:
        return common.noul(rng.choice(direct).format(**fields), answer)
    if roll < .7:
        return common.noul(rng.choice(inverted).format(**fields), not answer)
    criteria = dict(rng.choice(YES_NO))
    keys = list(criteria)
    rng.shuffle(keys)
    return common.choice(state, rng.choice(choice).format(**fields), {k: criteria[k] for k in keys},
                         keys.index("yes" if answer else "no"))


def table(repo: str, revision: str, pattern: str) -> list[dict]:
    path, _ = common.fetch(repo, [pattern], revision=revision)
    return pq.read_table(sorted(path.glob(pattern))[0]).to_pylist()


# --- BoolQ ----------------------------------------------------------------------------------------------------
BOOLQ = "boolq-train"
BOOLQ_REPO, BOOLQ_REVISION = "google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5"
BOOLQ_DIRECT = [
    "{q} Answer from the passage.",
    "Based on the passage, is the answer to this question yes?\n{q}",
    "According to the text: {q}",
    "Read the passage. {q}",
]
BOOLQ_INVERTED = [
    "Based on the passage, is the answer to this question no?\n{q}",
    "The passage indicates that the answer to the following question is no: {q}",
    "Using the text, decide whether this should be answered with \"no\": {q}",
]
BOOLQ_CHOICE = ["{q}", "Answer the question using the passage: {q}", "What does the text say? {q}"]
PASSAGE_KEYS = [("passage", "question"), ("text", "question"), ("article", "query")]


def boolq(stats: dict):
    rows = table(BOOLQ_REPO, BOOLQ_REVISION, "data/train-*.parquet")
    rng = random.Random(2026101661)
    picked = sorted(rng.sample(range(len(rows)), min(CAP, len(rows))))
    for i in picked:
        r = rows[i]
        q = question_text(r["question"])
        if rng.random() < .7:
            state = r["passage"] if rng.random() < .5 else {"passage": r["passage"]}
            question, target = yes_no(rng, state, r["answer"], BOOLQ_DIRECT, BOOLQ_INVERTED, BOOLQ_CHOICE, q=q)
        else:
            a, b = rng.choice(PASSAGE_KEYS)
            state = {a: r["passage"], b: q}
            question, target = yes_no(rng, state, r["answer"], ["Does the passage support answering yes to the question?"],
                                      ["Does the passage support answering no to the question?"],
                                      ["Answer the question from the passage."])
        yield common.example(BOOLQ, str(i), "yesno-qa", state, question, target, "train")
    stats.update(train_rows=len(rows), used=len(picked))


# --- StrategyQA ---------------------------------------------------------------------------------------------
STRATEGYQA = "strategyqa-train"
STRATEGYQA_REPO, STRATEGYQA_REVISION = "ChilleD/StrategyQA", "705562638fe1d8ca6bb98c66fc8f94d45fda8c83"
STRATEGY_DIRECT = [
    "{q}",
    "Using general knowledge and a bit of reasoning, answer yes or no: {q}",
    "Think it through: {q}",
]
STRATEGY_INVERTED = [
    "Is the correct answer to this question \"no\"? {q}",
    "Would a well-informed person answer the following with no? {q}",
]
STRATEGY_CHOICE = ["{q}", "Answer from what you know: {q}"]


def strategyqa(stats: dict):
    rows = table(STRATEGYQA_REPO, STRATEGYQA_REVISION, "data/train-*.parquet")
    rng = random.Random(2026101662)
    for r in rows:
        q = question_text(r["question"])
        if rng.random() < .7:
            state = ""
            question, target = yes_no(rng, state, r["answer"], STRATEGY_DIRECT, STRATEGY_INVERTED, STRATEGY_CHOICE, q=q)
        else:
            state = {"question": q}
            question, target = yes_no(rng, state, r["answer"], ["Is the answer to the question yes?"],
                                      ["Is the answer to the question no?"], ["Answer the question."])
        yield common.example(STRATEGYQA, r["qid"], "yesno-qa", state, question, target, "train")
    stats["train_rows"] = len(rows)


# --- PubMedQA -----------------------------------------------------------------------------------------------
PUBMEDQA = "pubmedqa-labeled"
PUBMEDQA_REPO, PUBMEDQA_REVISION = "qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318"
PUBMED_WITH_QUESTION = [
    "Based on this abstract, what is the answer to the research question? {q}",
    "Do the study's findings answer yes, no, or maybe to: {q}",
    "Read the abstract (its conclusion is omitted). {q}",
]
PUBMED_GENERIC = [
    "Does the study described answer its research question with yes, no, or maybe?",
    "Judging from the abstract, how should the question be answered?",
]
LEVELS = [
    [("yes", None), ("maybe", None), ("no", None)],
    [("yes", "the results support it"), ("maybe", "the evidence is mixed or inconclusive"), ("no", "the results do not support it")],
]


def pubmedqa(stats: dict):
    rows = table(PUBMEDQA_REPO, PUBMEDQA_REVISION, "pqa_labeled/train-*.parquet")
    rng = random.Random(2026101663)
    gold_counts = collections.Counter()
    for r in rows:
        ctx = r["context"]
        abstract = "\n".join(f"{label.title()}: {text}" if label else text
                             for label, text in zip(ctx["labels"] or [None] * len(ctx["contexts"]), ctx["contexts"]))
        q = question_text(r["question"])
        levels = rng.choice(LEVELS)
        if rng.random() < .5:
            levels = levels[::-1]
        criteria = dict(levels)
        gold = list(criteria).index(r["final_decision"])
        gold_counts[r["final_decision"]] += 1
        if rng.random() < .6:
            state = abstract if rng.random() < .5 else {"abstract": abstract}
            instructions = rng.choice(PUBMED_WITH_QUESTION).format(q=q)
        else:
            state = {"question": q, "abstract": abstract}
            instructions = rng.choice(PUBMED_GENERIC)
        question, target = common.choice(state, instructions, criteria, gold)
        yield common.example(PUBMEDQA, str(r["pubid"]), "yesno-qa", state, question, target, "train", ordered=True)
    stats.update(rows=len(rows), labels=dict(gold_counts))


# --- SQuAD 2.0 answerability ------------------------------------------------------------------------------
SQUAD = "squad2-answerability"
SQUAD_REPO, SQUAD_REVISION = "rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412"
ANSWERABLE = [
    "Can this question be answered from the paragraph? {q}",
    "Does the passage contain the information needed to answer: {q}",
    "The text gives an answer to the following question: {q}",
    "Is the answer to \"{q}\" stated in the passage?",
]
UNANSWERABLE = [
    "Is the paragraph missing the answer to this question? {q}",
    "This question cannot be answered using only the passage: {q}",
    "Would you have to say \"the text doesn't say\" to the question \"{q}\"?",
]
ANSWERABILITY = [  # (answerable option, unanswerable option)
    (("answerable", "the passage contains the answer"), ("unanswerable", "the passage does not contain the answer")),
    (("answered in the text", None), ("not answered in the text", None)),
]
ANSWERABILITY_CHOICE = ["Can the question \"{q}\" be answered from the text?", "Check the passage: does it answer {q}"]


def squad(stats: dict):
    rows = table(SQUAD_REPO, SQUAD_REVISION, "squad_v2/train-*.parquet")
    rng = random.Random(2026101660)
    order = list(range(len(rows)))
    rng.shuffle(order)
    per_context, picked, quota = collections.Counter(), [], {True: CAP // 2, False: CAP // 2}
    for i in order:
        r = rows[i]
        answerable = len(r["answers"]["text"]) > 0
        if quota[answerable] == 0 or per_context[r["context"]] >= 2:
            continue
        quota[answerable] -= 1
        per_context[r["context"]] += 1
        picked.append(i)
    for i in sorted(picked):
        r = rows[i]
        answerable = len(r["answers"]["text"]) > 0
        q = " ".join(r["question"].split())
        state = r["context"] if rng.random() < .5 else {"title": r["title"].replace("_", " "), "paragraph": r["context"]}
        roll = rng.random()
        if roll < .4:
            question, target = common.noul(rng.choice(ANSWERABLE).format(q=q), answerable)
        elif roll < .7:
            question, target = common.noul(rng.choice(UNANSWERABLE).format(q=q), not answerable)
        else:
            yes, no = rng.choice(ANSWERABILITY)
            labels = [yes, no]
            rng.shuffle(labels)
            gold = labels.index(yes if answerable else no)
            criteria = dict(labels)
            question, target = common.choice(state, rng.choice(ANSWERABILITY_CHOICE).format(q=q), criteria, gold)
        yield common.example(SQUAD, r["id"], "yesno-qa", state, question, target, "train")
    stats.update(train_rows=len(rows), answerable=CAP // 2 - quota[True], unanswerable=CAP // 2 - quota[False],
                 paragraphs=len(per_context))


if __name__ == "__main__":
    for name, fn, repo, revision, split, licence in [
            (BOOLQ, boolq, BOOLQ_REPO, BOOLQ_REVISION, "train", "CC-BY-SA-3.0"),
            (STRATEGYQA, strategyqa, STRATEGYQA_REPO, STRATEGYQA_REVISION, "train", "MIT"),
            (PUBMEDQA, pubmedqa, f"{PUBMEDQA_REPO} (pqa_labeled)", PUBMEDQA_REVISION, "train (pqa_labeled)", "MIT"),
            (SQUAD, squad, SQUAD_REPO, SQUAD_REVISION, "train", "CC-BY-SA-4.0")]:
        meta = {"upstream": repo, "revision": revision, "split": split, "licence": licence}
        common.write(name, fn(meta), meta)
