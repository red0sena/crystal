"""QuALITY train (150 long articles, 2,523 questions) as 4-option reading-comprehension choices.

Upstream: tasksource/QuALITY on the Hub (parquet, pinned below), a mirror of the official QuALITY
v1.0.1 release (Pang et al. 2022, github.com/nyu-mll/quality): it keeps the official fields
(article_id, set_unique_id, writer_label, gold_label, validation, difficult) and the official train
size (2,523 questions over 150 articles; dev 2,086). Licence: the dataset is CC-BY-4.0; the
articles are Project Gutenberg texts (public domain in the US, Gutenberg licence), Slate articles
from the Open American National Corpus, and CC-BY-4.0 pieces (The Long+Short, Freesouls, MIT Press
open access); each row's `license` field says which. Only the train split is used; the index and
JEVal contain no QuALITY rows, and dev/test are left alone.

Framing: one choice per (article, question) over the four options in their original order (training
shuffles); gold = `gold_label` (the validators' majority answer, 1-based upstream), not the writer's
label. Instructions come from our own phrasings; the state is the article as plain text or under a
varying key ("article", "story", "document", "text", sometimes with the title). Family "long-reading".

Overlap: before the filter, every train article is compared with every long evaluation state: the
JEVal test states (data/jeval/data/test.parquet) and the suite-0.3 rows whose state text is longer
than 2,000 characters. An article is dropped when more than 20% of its 13-grams (decontam.words)
occur in one evaluation state, or when it shares at least MAX_SHARED_GRAMS 13-grams with one (a
copied passage). Because that document-level check is done here, the filter then runs in documents
mode: the strict filter flags 19 of the 150 articles, all for one short line of ordinary dialogue or
prose ("that's right", "I don't know", "all rights reserved") that also occurs as a line in some
index row. The manifest records both checks.

Length: prompts are rendered with format.render and the Qwen3.5 tokenizer; items over MAX_TOKENS
(64k) are dropped, never truncated. QuALITY articles are 2k-8k words, so none comes close.
"""
from __future__ import annotations

import glob
import gzip
import json
import random

import numpy as np
import pyarrow.parquet as pq

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

NAME = "quality-train"
REPO = "tasksource/QuALITY"
REVISION = "4892b065e6a40f44fa61bc406df4c35fe91e9290"
MAX_TOKENS = 64_000
MAX_EVAL_CONTAINMENT = .2  # share of an article's 13-grams found in one evaluation state
MAX_SHARED_GRAMS = 100  # or this many 13-grams shared with one evaluation state
SUITE_MIN_CHARS = 2000
WITH_QUESTION = [
    "Answer the question about the text above.\n\n{q}",
    "Based on the story, {lq}",
    "Read the passage carefully, then answer: {q}",
    "{q}",
    "Using only what the article says or implies, choose the best answer.\n\nQuestion: {q}",
    "According to the text, {lq}",
    "Question about the reading: {q}",
    "Pick the option that best answers this question about the document: {q}",
]
GENERIC = [
    "Answer the question about the article.",
    "Read the text and choose the best answer to the question.",
    "Which option answers the question correctly, given the passage?",
]
LOWER_OK = {"what", "why", "how", "who", "whom", "whose", "which", "where", "when", "is", "are", "does", "do", "did",
            "was", "were", "in", "after", "before", "during", "at", "the", "a", "an", "of", "if", "can", "could",
            "would", "will", "should", "according", "based", "throughout", "by", "from", "for", "to"}
TEXT_KEYS = ["article", "story", "document", "text", "passage"]


def grams(text: str) -> set[int]:
    w = words(text)
    return {hash(tuple(w[i:i + 13])) for i in range(len(w) - 12)}


def eval_states():
    """(name, text) of every JEVal test state and of every long suite-0.3 state."""
    for r in pq.read_table(common.ROOT / "data/jeval/data/test.parquet", columns=["id", "state"]).to_pylist():
        yield f"jeval:{r['id']}", r["state"]
    for path in sorted(glob.glob(str(common.ROOT / "external/decision-index/suite-0.3/*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                state = json.loads(line).get("state")
                text = state if isinstance(state, str) else "\n".join(common.strings(state)) if state else ""
                if len(text) > SUITE_MIN_CHARS:
                    yield f"suite:{path.rsplit('/', 1)[-1]}:{i}", text


def eval_overlap(articles: dict[int, str]) -> dict:
    """Per article: the largest share of its 13-grams, and count, found in a single evaluation state."""
    owner: dict[int, list[int]] = {}
    sizes = {}
    for key, text in articles.items():
        g = grams(text)
        sizes[key] = len(g)
        for h in g:
            owner.setdefault(h, []).append(key)
    best = {key: {"containment": 0., "shared": 0, "state": None} for key in articles}
    scanned = 0
    for name, text in eval_states():
        scanned += 1
        counts: dict[int, int] = {}
        for h in grams(text):
            for key in owner.get(h, ()):
                counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            if n > best[key]["shared"]:
                best[key] = {"containment": n / max(1, sizes[key]), "shared": n, "state": name}
    return {"scanned": scanned, "best": best}


def build(stats: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(common.ROOT / "models/qwen3.5-0.8b")
    codes, _ = fmt.answer_codes(tokenizer)
    rng = random.Random(2026101621)
    path, _ = common.fetch(REPO, ["data/train-*.parquet", "README.md"], revision=REVISION)
    rows = pq.read_table(glob.glob(str(path / "data/train-*.parquet"))[0]).to_pylist()
    articles = {r["article_id"]: r["article"] for r in rows}
    overlap = eval_overlap(articles)
    near = sorted(k for k, b in overlap["best"].items()
                  if b["containment"] > MAX_EVAL_CONTAINMENT or b["shared"] >= MAX_SHARED_GRAMS)
    top = sorted(overlap["best"].items(), key=lambda kv: -kv[1]["shared"])[:5]
    lengths, too_long, bad_options = [], 0, 0
    for r in rows:
        if r["article_id"] in near:
            continue
        options = [o.strip() for o in r["options"]]
        if len(set(options)) != 4 or not all(options):
            bad_options += 1
            continue
        q = r["question"].strip()
        article = r["article"]
        if rng.random() < .65:
            first = q.split(" ", 1)[0]
            lq = q[0].lower() + q[1:] if first.lower() in LOWER_OK else q
            instructions = rng.choice(WITH_QUESTION).format(q=q, lq=lq)
            style = rng.random()
            if style < .35:
                state = article
            elif style < .85:
                state = {rng.choice(TEXT_KEYS): article}
            else:
                state = {"title": r["title"], rng.choice(TEXT_KEYS): article}
        else:
            state = {rng.choice(TEXT_KEYS): article, "question": q}
            instructions = rng.choice(GENERIC)
        question, target = common.choice(state, instructions, options, gold=r["gold_label"] - 1)
        n = len(tokenizer.encode(fmt.render(tokenizer, state, question, codes), add_special_tokens=False))
        if n > MAX_TOKENS:
            too_long += 1
            continue
        lengths.append(n)
        yield common.example(NAME, r["question_unique_id"], "long-reading", state, question, target, "train")
    q = np.percentile(lengths, [50, 90]).round().tolist()
    stats.update(questions=len(rows), articles=len(articles),
                 eval_overlap_check={"states_scanned": overlap["scanned"], "articles_dropped": near,
                                     "questions_dropped": sum(r["article_id"] in near for r in rows),
                                     "top_overlaps": [{"article_id": k, **b, "containment": round(b["containment"], 4)}
                                                      for k, b in top]},
                 skipped_bad_options=bad_options, over_token_cap=too_long,
                 prompt_tokens={"p50": q[0], "p90": q[1], "max": int(max(lengths)), "n": len(lengths)})


if __name__ == "__main__":
    # build() adds its statistics to meta; write() reads meta only after consuming the rows.
    meta = {"upstream": f"{REPO} (mirror of QuALITY v1.0.1, data/train-*.parquet)", "revision": REVISION,
            "split": "train", "licence": "CC-BY-4.0 (dataset); articles: Project Gutenberg / OANC (Slate) / CC-BY-4.0",
            "gold": "gold_label (validator majority), 1-based upstream", "max_prompt_tokens": MAX_TOKENS,
            "max_eval_containment": MAX_EVAL_CONTAINMENT, "max_shared_grams": MAX_SHARED_GRAMS,
            "filter_note": ("documents mode: every train article was checked against all JEVal test states and all "
                            f"suite-0.3 states over {SUITE_MIN_CHARS} chars by 13-gram containment in build(); the "
                            "strict filter flags 19/150 articles, each for one common short line of dialogue/prose")}
    common.write(NAME, build(meta), meta, documents=True)
