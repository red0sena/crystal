"""CUAD train contracts x 41 clause categories as "does the contract contain this clause?" nouls.

Upstream: theatticusproject/cuad-qa on the Hub, the Hub's parquet conversion
(refs/convert/parquet at REVISION; the repo itself only has a loading script, which is not run).
Only `default/train/*.parquet` (408 contracts, SQuAD-style rows) is used for questions; the test
parquet (102 contracts) is read only for the overlap check. Licence: CC-BY-4.0 (card YAML and the
card's Licensing Information: "CUAD is licensed under the Creative Commons Attribution 4.0 (CC BY
4.0) license"; the underlying contracts are public SEC EDGAR filings).

Framing: one noul per (contract, category): true when the category has at least one answer span in
that contract. The instruction names the category and its CUAD description (the "Details:" text)
in one of several own phrasings (the index's "should be reviewed by a lawyer" line is not used), or
the category and description go into the state with a generic instruction. Two train titles share
one identical text; it is kept once. Balance: every positive is kept and negatives are sampled so
positives are POSITIVE_SHARE of the questions (upstream ~33% positive).

Overlap: a training contract is dropped when its words equal, or its 13-gram containment exceeds
MAX_TEST_CONTAINMENT in either direction with, any JEVal `cuad` state or any CUAD test contract.
Because this document-level check is done here, the filter runs in documents mode (as for
ContractNLI: standard clauses and signature blocks are shared by many contracts). The documents-mode
filter still flags ~45% of the contracts (>= 10 shared 13-grams, mostly confidentiality boilerplate
shared with ContractNLI test NDAs in the index); build() applies it to every question before sampling
negatives, so the kept questions keep the target balance, and write() re-checks every question.

Lengths: a contract is kept only if its rendered prompt (echo layout, the longer one, with the
longest instruction and each state form used) is <= MAX_PROMPT_TOKENS Qwen3.5 tokens; nothing is
truncated. The manifest records drops, the share above 32k and p50/p90/max.
"""
from __future__ import annotations

from collections import Counter
import random

import numpy as np
import pandas as pd

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.decontam import Filter, strings
from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.lexglue_train import JEVAL, Overlap, REPORT_TOKENS

NAME = "cuad-train"
REPO = "theatticusproject/cuad-qa"
REVISION = "d9c4ee0250ae2eb97bdb5b50773ab14ea62d0631"  # refs/convert/parquet; main is 6d9a118fb629814c4384a1e63cfc84f5c01894b3
MAX_PROMPT_TOKENS = 64_000
MAX_TEST_CONTAINMENT = .2
POSITIVE_SHARE = .45
WITH_CATEGORY = [
    'Does this contract contain a clause of the type "{cat}"? ({desc})',
    "Clause type: {cat}. What it covers: {desc}\nIs there any provision of this kind in the agreement?",
    'A reviewer is checking the agreement for "{cat}" language ({desc}). Is such language present?',
    "Is any part of this contract about {cat}? Category description: {desc}",
    'Check the agreement for the "{cat}" category, described as: {desc}. Does the agreement include it?',
    "Contract review question - {cat} ({desc}): does the document have a clause on this?",
]
GENERIC = [
    "Does the contract contain at least one clause of the given category?",
    "Decide whether the agreement includes language that falls under the named clause type.",
    "Is the clause category described alongside the contract present anywhere in the document?",
]
CRITERIA = [None, {"true": "the contract has at least one clause of this kind", "false": "no clause of this kind appears"},
            {"true": "present", "false": "absent"}]
CONTRACT_KEYS = ["contract", "agreement", "document"]
GENERIC_KEYS = [("contract", "clause type", "description"), ("agreement", "category", "what it covers")]


def read(split: str) -> pd.DataFrame:
    local, _ = common.fetch(REPO, [f"default/{split}/*.parquet"], revision=REVISION)
    df = pd.concat([pd.read_parquet(p) for p in sorted((local / "default" / split).glob("*.parquet"))], ignore_index=True)
    df["category"] = df.question.str.extract(r'related to "(.+?)" that')[0]
    df["description"] = df.question.str.extract(r"Details: (.*)$", expand=False).str.strip().str.rstrip(".")
    assert df.category.notna().all() and df.description.notna().all()
    return df


def build(stats: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(common.ROOT / "models/qwen3.5-0.8b")
    codes, _ = fmt.answer_codes(tokenizer)
    rng = random.Random(2026101631)
    train, test = read("train"), read("test")
    jeval = pd.read_parquet(JEVAL, columns=["source", "state"])
    overlap = Overlap([str(s) for s in jeval[jeval.source == "cuad"].state] + list(test.context.unique()))
    categories = train.drop_duplicates("category").set_index("category").description.to_dict()
    assert len(categories) == 41, len(categories)
    longest = max(([t.format(cat=c, desc=d) for t in WITH_CATEGORY for c, d in categories.items()]
                   + [f"{g}\n{c} {d}" for g in GENERIC for c, d in categories.items()]), key=len)
    longest_q = {"type": "noul", "instructions": longest, "criteria": CRITERIA[1]}

    def tokens(state) -> int:
        return len(tokenizer.encode(fmt.render(tokenizer, state, longest_q, codes[:2], "echo")))

    flt = Filter(common.DECONTAM)
    counts, lengths, contracts, seen = Counter(), {}, [], set()
    for title, g in train.groupby("title", sort=True):
        text = g.context.iloc[0]
        counts["contracts"] += 1
        if text in seen:
            counts["duplicate_text_contracts"] += 1
            continue
        seen.add(text)
        if overlap.near(text, MAX_TEST_CONTAINMENT):
            counts["contracts_near_test"] += 1
            stats.setdefault("near_test_titles", []).append(title)
            continue
        forms = [text] + [{k: text} for k in CONTRACT_KEYS] + [{a: text, b: "x", c: "x"} for a, b, c in GENERIC_KEYS]
        n = max(tokens(f) for f in forms)
        counts["contracts_length_checked"] += 1
        if n > REPORT_TOKENS:
            counts["contracts_over_32k_tokens"] += 1
        if n > MAX_PROMPT_TOKENS:
            counts["contracts_over_token_cap"] += 1
            continue
        lengths[title] = n
        present = g.groupby("category").answers.apply(lambda s: any(len(a["text"]) > 0 for a in s)).to_dict()
        assert set(present) == set(categories), title
        contracts.append((title, text, present))

    def question(title, text, cat, truth):
        desc = categories[cat]
        if rng.random() < .7:
            state = text if rng.random() < .5 else {rng.choice(CONTRACT_KEYS): text}
            instructions = rng.choice(WITH_CATEGORY).format(cat=cat, desc=desc)
        else:
            a, b, c = rng.choice(GENERIC_KEYS)
            state = {a: text, b: cat, c: desc}
            instructions = rng.choice(GENERIC)
        q, target = common.noul(instructions, truth, rng.choice(CRITERIA))
        return common.example(NAME, f"train:{title}:{cat}", "legal-clause-detection", state, q, target, "train")

    # The index filter (documents mode) is applied here exactly as write() applies it, before negatives are
    # sampled, so the class balance holds after filtering; write() then re-checks every row (a no-op).
    positives, negatives = [], []
    for title, text, present in contracts:
        for cat, truth in sorted(present.items()):
            e = question(title, text, cat, truth)
            q = e["question"]
            if flt.flagged(strings(e["state"]) + strings(q["instructions"]), documents=True, options=strings(q.get("criteria"))):
                counts["questions_flagged_by_index_filter"] += 1
                continue
            (positives if truth else negatives).append(e)
    n_neg = min(len(negatives), round(len(positives) * (1 - POSITIVE_SHARE) / POSITIVE_SHARE))
    chosen = positives + rng.sample(negatives, n_neg)
    chosen.sort(key=lambda e: e["id"])
    yield from chosen
    kept = {e["id"].split(":")[2] for e in chosen}
    counts["contracts_fully_flagged_by_index_filter"] = len({t for t, _, _ in contracts} - {e["id"].split(":")[2] for e in positives + negatives})

    def pct(values):
        q = np.percentile(values, [50, 90]).round().tolist()
        return {"p50": q[0], "p90": q[1], "max": int(max(values))}

    stats.update(counts, contracts_kept=len(kept), questions=len(chosen), positives=len(positives),
                 negatives_available=len(negatives), negatives_sampled=n_neg,
                 positive_share=round(len(positives) / max(1, len(chosen)), 4),
                 share_over_32k_tokens=round(counts["contracts_over_32k_tokens"] / counts["contracts_length_checked"], 4),
                 prompt_tokens_length_checked=pct(list(lengths.values())),
                 prompt_tokens=pct([lengths[t] for t in kept]))
    print({k: v for k, v in stats.items() if k != "near_test_titles"}, flush=True)


if __name__ == "__main__":
    # build() adds its statistics to meta; write() reads meta only after consuming the rows.
    meta = {"upstream": f"{REPO} (Hub parquet conversion, default/train)", "revision": REVISION, "split": "train",
            "licence": "CC-BY-4.0 (card YAML and Licensing Information; contracts from SEC EDGAR)",
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "prompt_tokens_measured": "per contract: fmt.render echo layout with the longest instruction, max over state forms",
            "max_test_containment": MAX_TEST_CONTAINMENT, "positive_share_target": POSITIVE_SHARE,
            "overlap_reference": "JEVal source == cuad states + the 102 CUAD test contracts (exact words or >20% 13-gram containment either way)"}
    meta["filter_note"] = "documents mode: per-contract overlap with JEVal cuad and the CUAD test split is checked in build()"
    common.write(NAME, build(meta), meta, documents=True)
