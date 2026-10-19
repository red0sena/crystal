"""MuSiQue train (multi-hop QA over ~20 Wikipedia paragraphs) as answerability, answer and evidence questions.

Upstream: bdsaglam/musique on the Hub (pinned below), which holds the official MuSiQue v1.0 files
(Trivedi et al. 2022, github.com/StonyBrookNLP/musique) byte for byte (same sizes as the release
and as voidful/MuSiQue): musique_ans_v1.0_train.jsonl (19,938 answerable questions) and
musique_full_v1.0_train.jsonl (each of those questions twice: once answerable, once as an
unanswerable contrast version whose paragraph set lacks a needed supporting paragraph). Licence:
CC-BY-4.0 (official release). Only the two train files are downloaded; dev and test are not used.
The index and JEVal contain no MuSiQue rows.

State: the question's paragraphs in upstream order, numbered, each with its Wikipedia title on the
same line as its text ("Paragraph 3 (Title): text", "[3] Title: text", a {"Paragraph 3": "Title:
text"} object, ...). The title is kept inline on purpose: on a line of its own a title such as
"Statue of Liberty" or "In the End" is an exact short-text match of some index row, which flagged a
fifth of all questions for nothing but an entity name.

Questions (family "multi-doc"), each question id used by only one of the three tasks:
  (a) answerability (noul, from MuSiQue-Full): "Do the passages contain enough information to
      answer: <q>?" in several phrasings; both versions of a question are used (one true, one false),
      so (a) is balanced and contrastive.
  (b) answer choice (from MuSiQue-Ans): the gold answer plus 3-15 distractors drawn from other
      questions' answers of the same coarse type (year / date / number / proper name / other),
      sometimes including this question's own intermediate-hop answers of that type; no distractor
      equals or contains the gold answer or an alias (normalised). Shuffled; gold index recorded.
  (c) evidence choice (from MuSiQue-Ans): which paragraph is needed, over the paragraph labels; gold
      = the last hop's supporting paragraph, which contains the final answer. Either all paragraphs
      are options and the question asks for the one that states the final answer, or the other
      supporting paragraphs are left out of the options and the question asks which listed paragraph
      is needed (so exactly one option is correct either way).
PER_TASK examples per task (12,000 in all).

Overlap: every candidate example is checked with the strict contamination filter (the same
decontam.Filter that common.write runs) before it counts towards its task's quota, so flagged
questions are replaced by clean ones and the tasks stay balanced; the manifest records the
pre-check counts and reasons. Measured: 13-18% of candidates are flagged (any of ~20 paragraphs
sharing >= 2 kept 13-gram shingles); tracing a 1,500-question sample, the flagged paragraphs are
Wikipedia text that also appears in index HoVer rows (most), JEVal BoolQ, LongBench retrieval,
FEVER and HaluEval-QA states; those drops are legitimate.
common.write then runs the strict filter again (documents=False).

Length: prompts are rendered with format.render and the Qwen3.5 tokenizer; items over MAX_TOKENS
(64k) would be dropped, never truncated (20 paragraphs are ~2-4k tokens).
"""
from __future__ import annotations

import json
import random
import re

import numpy as np

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.decontam import Filter, strings
from qwen_decision.v3.sources import common

NAME = "musique-train"
REPO = "bdsaglam/musique"
REVISION = "22873a405dd809893b22ada0b499299fb612d2df"
ANS, FULL = "musique_ans_v1.0_train.jsonl", "musique_full_v1.0_train.jsonl"
PER_TASK = 4000
MAX_TOKENS = 64_000
FAMILY = "multi-doc"

ANSWERABLE = [
    "Do the passages contain enough information to answer: {q}",
    "Can the following question be answered from the paragraphs above alone?\n{q}",
    "Is everything needed to answer this question stated in the provided paragraphs?\n\n{q}",
    "Using only these paragraphs, is it possible to work out the answer to: {q}",
    "Question: {q}\nDo the documents give you what you need to answer it?",
    "Would a reader who knows nothing but these passages be able to answer \"{q}\"?",
]
ANSWERABLE_CRITERIA = [None, {"true": "every fact needed is in the passages", "false": "at least one needed fact is missing"},
                       {"true": "answerable from the paragraphs", "false": "not answerable from the paragraphs"}]
ANSWER = [
    "Answer the question using the paragraphs.\n\nQuestion: {q}",
    "{q}",
    "Based on the passages, {lq}",
    "Combine facts from the paragraphs to answer: {q}",
    "Several of the passages are needed for this question. {q}",
    "Using the documents above, choose the answer.\n{q}",
]
FINAL_EVIDENCE = [
    "Which paragraph states the final answer to this question?\n{q}",
    "To answer \"{q}\", several paragraphs are combined. Which one gives the final answer?",
    "Which passage do you need for the last step of answering: {q}",
    "Find the paragraph that contains the answer to: {q}",
]
NEEDED_EVIDENCE = [
    "Which paragraph is needed to answer the question below?\n{q}",
    "Which of the listed paragraphs do you have to read to answer: {q}",
    "Only one of the paragraphs offered is needed for this question. Which one?\n{q}",
    "Which passage supports the answer to: {q}",
]
LOWER_OK = {"what", "who", "whom", "whose", "which", "where", "when", "how", "why", "in", "the", "a", "an", "is", "was",
            "are", "were", "did", "does", "do", "at", "on", "for", "from", "of", "by", "after", "during", "since",
            "until", "to", "as", "with"}
MONTHS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)


def layouts():
    """(make_state(paragraphs), label(n)) pairs; titles stay on the paragraph's line."""
    def text(fmt_line):
        return lambda ps: "\n\n".join(fmt_line(i + 1, p["title"], p["paragraph_text"]) for i, p in enumerate(ps))
    return [
        (text(lambda n, t, x: f"Paragraph {n} ({t}): {x}"), lambda n: f"Paragraph {n}"),
        (text(lambda n, t, x: f"[{n}] {t}: {x}"), lambda n: f"[{n}]"),
        (text(lambda n, t, x: f"Passage {n}. {t} - {x}"), lambda n: f"Passage {n}"),
        (text(lambda n, t, x: f"#{n} | {t} | {x}"), lambda n: f"#{n}"),
        (lambda ps: {f"Paragraph {i + 1}": f"{p['title']}: {p['paragraph_text']}" for i, p in enumerate(ps)},
         lambda n: f"Paragraph {n}"),
        (lambda ps: {"documents": "\n".join(f"Document {i + 1} - {p['title']}: {p['paragraph_text']}"
                                            for i, p in enumerate(ps))}, lambda n: f"Document {n}"),
    ]


LAYOUTS = layouts()


def lower(q: str) -> str:
    return q[0].lower() + q[1:] if q.split(" ", 1)[0].lower() in LOWER_OK else q


def norm(a: str) -> str:
    a = re.sub(r"[^a-z0-9 ]", " ", a.lower())
    return " ".join(w for w in a.split() if w not in {"a", "an", "the"})


def kind(a: str) -> str:
    a = a.strip()
    if re.fullmatch(r"(c\. ?)?(1\d{3}|20\d{2})s?", a) or re.fullmatch(r"\d{1,4} (BC|BCE|AD|CE)", a):
        return "year"
    if MONTHS.search(a) and re.search(r"\d", a):
        return "date"
    if re.search(r"\d", a):
        return "number"
    if a[:1].isupper():
        return "name"
    return "other"


def build(stats: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(common.ROOT / "models/qwen3.5-0.8b")
    codes, _ = fmt.answer_codes(tokenizer)
    flt = Filter(common.DECONTAM)
    rng = random.Random(2026101631)
    path, _ = common.fetch(REPO, [ANS, FULL, "README.md"], revision=REVISION)
    ans = {}
    with (path / ANS).open() as f:
        for line in f:
            r = json.loads(line)
            ans[r["id"]] = r
    full: dict[str, dict[bool, dict]] = {}
    with (path / FULL).open() as f:
        for line in f:
            r = json.loads(line)
            full.setdefault(r["id"], {})[r["answerable"]] = r
    pools: dict[str, list[str]] = {}
    for a in sorted({r["answer"].strip() for r in ans.values()}):
        pools.setdefault(kind(a), []).append(a)

    ids = sorted(ans)
    rng.shuffle(ids)
    third = len(ids) // 3
    slices = {"answerable": ids[:third], "answer": ids[third:2 * third], "evidence": ids[2 * third:]}
    lengths, too_long = [], 0
    pre = {task: {"candidates": 0, "flagged": 0, "skipped_short": 0, "reasons": []} for task in slices}

    def clean(task, e):
        q = e["question"]
        why = flt.flagged(strings(e["state"]) + strings(q.get("instructions")), options=strings(q.get("criteria")))
        if why:
            pre[task]["flagged"] += 1
            if len(pre[task]["reasons"]) < 15:
                pre[task]["reasons"].append({"id": e["id"], "why": why})
        return not why

    def fits(e):
        nonlocal too_long
        n = len(tokenizer.encode(fmt.render(tokenizer, e["state"], e["question"], codes), add_special_tokens=False))
        if n > MAX_TOKENS:
            too_long += 1
            return False
        lengths.append(n)
        return True

    # (a) answerability: both versions of a question, true and false.
    made = 0
    for qid in slices["answerable"]:
        if made >= PER_TASK:
            break
        versions = full.get(qid, {})
        if set(versions) != {True, False} or min(len(v["paragraphs"]) for v in versions.values()) < 10:
            pre["answerable"]["skipped_short"] += 1
            continue
        pair = []
        for truth in (True, False):
            r = versions[truth]
            make, _ = rng.choice(LAYOUTS)
            q = r["question"].strip()
            criteria = rng.choice(ANSWERABLE_CRITERIA)
            question, target = common.noul(rng.choice(ANSWERABLE).format(q=q), truth, criteria)
            pair.append(common.example(NAME, f"answerable:{qid}:{int(truth)}", FAMILY, make(r["paragraphs"]),
                                       question, target, "train"))
        pre["answerable"]["candidates"] += 1
        if all(clean("answerable", e) for e in pair) and all(fits(e) for e in pair):
            made += 2
            yield from pair

    # (b) answer choice among same-type answers of other questions.
    made, n_options = 0, []
    for qid in slices["answer"]:
        if made >= PER_TASK:
            break
        r = ans[qid]
        if len(r["paragraphs"]) < 10:
            pre["answer"]["skipped_short"] += 1
            continue
        gold = r["answer"].strip()
        banned = {norm(a) for a in [gold, *r["answer_aliases"]]} - {""}

        def ok(a):
            n = norm(a)
            return n and all(n != b and n not in b and b not in n for b in banned)

        k = rng.randint(3, 15)
        hops = [h["answer"].strip() for h in r["question_decomposition"][:-1]]
        picked = [a for a in dict.fromkeys(hops) if kind(a) == kind(gold) and ok(a) and rng.random() < .5]
        pool = pools[kind(gold)]
        tries = 0
        while len(picked) < k and tries < 200:
            tries += 1
            a = rng.choice(pool)
            if ok(a) and a not in picked and all(norm(a) != norm(p) for p in picked):
                picked.append(a)
        if len(picked) < 3:
            pre["answer"]["skipped_short"] += 1
            continue
        options = picked[:k] + [gold]
        rng.shuffle(options)
        make, _ = rng.choice(LAYOUTS)
        q = r["question"].strip()
        question, target = common.choice(None, rng.choice(ANSWER).format(q=q, lq=lower(q)), options,
                                         gold=options.index(gold))
        e = common.example(NAME, f"answer:{qid}", FAMILY, make(r["paragraphs"]), question, target, "train")
        pre["answer"]["candidates"] += 1
        if clean("answer", e) and fits(e):
            made += 1
            n_options.append(len(options))
            yield e

    # (c) evidence: the paragraph with the final answer.
    made, final_mode = 0, 0
    for qid in slices["evidence"]:
        if made >= PER_TASK:
            break
        r = ans[qid]
        ps = r["paragraphs"]
        if len(ps) < 10:
            pre["evidence"]["skipped_short"] += 1
            continue
        supporting = [i for i, p in enumerate(ps) if p["is_supporting"]]
        last = r["question_decomposition"][-1]["paragraph_support_idx"]
        if last is None or last not in supporting:
            last = supporting[-1]
        make, label = rng.choice(LAYOUTS)
        q = r["question"].strip()
        if rng.random() < .5:
            keep = list(range(len(ps)))
            instructions = rng.choice(FINAL_EVIDENCE).format(q=q)
        else:
            keep = [i for i in range(len(ps)) if i == last or i not in supporting]
            instructions = rng.choice(NEEDED_EVIDENCE).format(q=q)
        options = [label(i + 1) for i in keep]
        question, target = common.choice(None, instructions, options, gold=keep.index(last))
        e = common.example(NAME, f"evidence:{qid}", FAMILY, make(ps), question, target, "train")
        pre["evidence"]["candidates"] += 1
        if clean("evidence", e) and fits(e):
            made += 1
            final_mode += len(keep) == len(ps)
            yield e

    q = np.percentile(lengths, [50, 90]).round().tolist()
    stats.update(questions_ans=len(ans), questions_full=sum(len(v) for v in full.values()),
                 per_task=PER_TASK, strict_precheck=pre, over_token_cap=too_long,
                 answer_options={"min": min(n_options), "max": max(n_options), "mean": round(float(np.mean(n_options)), 2)},
                 evidence_final_answer_mode=final_mode,
                 answer_type_pool={k: len(v) for k, v in pools.items()},
                 prompt_tokens={"p50": q[0], "p90": q[1], "max": int(max(lengths)), "n": len(lengths)})


if __name__ == "__main__":
    # build() adds its statistics to meta; write() reads meta only after consuming the rows.
    meta = {"upstream": f"{REPO} (official MuSiQue v1.0 files: {ANS}, {FULL})", "revision": REVISION,
            "split": "train", "licence": "CC-BY-4.0", "max_prompt_tokens": MAX_TOKENS,
            "filter_note": ("strict mode; every candidate was pre-checked with the same strict filter and flagged "
                            "questions were replaced by clean ones (counts in strict_precheck); flagged paragraphs traced to "
                            "index HoVer, JEVal BoolQ/LongBench-retrieval/FEVER/HaluEval-QA Wikipedia text")}
    common.write(NAME, build(meta), meta, documents=False)
