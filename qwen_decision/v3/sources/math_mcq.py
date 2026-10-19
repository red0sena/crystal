"""MathQA and AQuA-RAT train splits as their native five-way answer choices (rationales stripped).

Upstream:
  - allenai/math_qa: the Hub repo holds only a loading script, so we read the Hub's own parquet
    conversion (refs/convert/parquet, default/train), pinned to that commit. 29,837 train problems.
    Licence: Apache-2.0.
  - deepmind/aqua_rat, config raw, split train (97,467 problems, 78.5k distinct questions).
    Licence: Apache-2.0.
Cap: 15,000 rows each, a seeded random subset of the valid train problems.

Overlap: neither dataset is in the index or JEVal (plan K2/K3). MathQA re-annotates AQuA problems,
so the two sources share some problems with each other; that is duplication, not contamination.
GSM8K test and its rewrites are not involved; the filter still checks every row against the index
(GSM8K test, MMLU-Pro and the routing rows) and JEVal.

Cleaning: MathQA text is tokenised lower case ("rs . 36 . what is ?"), so spaces before punctuation
are removed; option strings are parsed from "a ) x , b ) y" (or the list form some rows use). AQuA
has noisy items, so a problem is dropped when its rationale names a different letter than the gold
(the plan's rationale-answer agreement check), and repeated question+options are kept once.
"""
from __future__ import annotations

import ast
import random
import re

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common, mcq_util

CAP = 15_000
MATHQA = ("allenai/math_qa", "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6")  # refs/convert/parquet
AQUA = "deepmind/aqua_rat"
PHRASINGS = [
    "Solve the problem and pick the matching answer choice.",
    "Which choice gives the right result for this problem?",
    "Work out the answer, then select the option equal to it.",
    "Select the answer choice that solves the quantitative problem in the state.",
    "Which of the listed answers is correct for this math question?",
]
KEYS = ("problem", "question", "math_problem")
_PUNCT = re.compile(r" ([.,?!;%])(?=\s|$)")
_LETTERS = "abcde"
_CLAIM = re.compile(r"(?i)\b(?:answer|ans|option|choice|correct)\b(?:\s+(?:is|will be|should be|would be|choice|option))*"
                    r"\s*[:=\-]*\s*(?:option\s*)?\(?([A-E])\b(?![a-z])")


def detok(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r" (['’]) s\b", r"\1s", text)
    text = _PUNCT.sub(r"\1", text)
    text = text.replace("( ", "(").replace(" )", ")")
    return re.sub(r"\s+", " ", text).strip()


def mathqa_options(raw: str) -> list[str] | None:
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parts = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    else:
        parts = re.split(r"\s*,\s*(?=[a-e] \) )", raw)
    if len(parts) != 5:
        return None
    out = []
    for letter, part in zip(_LETTERS, parts):
        m = re.match(rf"{letter} \) (.*)$", str(part).strip(), re.S)
        if not m:
            return None
        out.append(detok(m.group(1)))
    return out


def build_mathqa(local):
    rows = pq.read_table(next(local.glob("default/train/*.parquet"))).to_pylist()
    rng = random.Random(29837)
    order = list(range(len(rows)))
    rng.shuffle(order)
    kept = 0
    for i in order:
        r = rows[i]
        options = mathqa_options(r["options"])
        if options is None or r["correct"] not in _LETTERS:
            continue
        e = mcq_util.cast("mathqa-train", str(i), "math", "train", rng, PHRASINGS, detok(r["Problem"]), options,
                          _LETTERS.index(r["correct"]), keys=KEYS)
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


def claimed_letter(rationale: str) -> str | None:
    tail = rationale.strip()[-160:]
    last = tail.splitlines()[-1].strip() if tail else ""
    if re.fullmatch(r"\(?([A-E])\)?\.?", last):
        return last.strip("().")
    found = _CLAIM.findall(tail)
    return found[-1].upper() if found else None


def build_aqua(local):
    rows = pq.read_table(next(local.glob("raw/train-*.parquet"))).to_pylist()
    rng = random.Random(97467)
    order = list(range(len(rows)))
    rng.shuffle(order)
    kept, seen = 0, set()
    for i in order:
        r = rows[i]
        letters = [o[:2] for o in r["options"]]
        if letters != [f"{c})" for c in "ABCDE"] or r["correct"] not in "ABCDE":
            continue
        claim = claimed_letter(r["rationale"])
        if claim is not None and claim != r["correct"]:
            continue
        options = [o[2:] for o in r["options"]]
        key = " ".join(r["question"].lower().split()) + "||" + "|".join(options)
        if key in seen:
            continue
        seen.add(key)
        e = mcq_util.cast("aqua-rat-train", str(i), "math", "train", rng, PHRASINGS, r["question"], options,
                          "ABCDE".index(r["correct"]), keys=KEYS)
        if e:
            yield e
            kept += 1
            if kept == CAP:
                return


if __name__ == "__main__":
    local, revision = common.fetch(MATHQA[0], ["default/train/*"], revision=MATHQA[1])
    common.write("mathqa-train", build_mathqa(local), {
        "upstream": f"{MATHQA[0]} (refs/convert/parquet, default)", "revision": revision, "split": "train",
        "licence": "Apache-2.0", "cap": CAP, "cast": "native 5-way choice, rationale stripped, light detokenisation"})
    local, revision = common.fetch(AQUA, ["raw/train-*"])
    common.write("aqua-rat-train", build_aqua(local), {
        "upstream": f"{AQUA} (raw)", "revision": revision, "split": "train", "licence": "Apache-2.0", "cap": CAP,
        "cast": "native 5-way choice, rationale stripped; rationale-answer disagreement and repeats dropped"})
