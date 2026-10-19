"""Corr2Cause train as yes/no: does the causal hypothesis follow from the stated correlations?

Upstream: causalnlp/corr2cause (same repo as causal-nlp/corr2cause), file train.csv only (205,734
premise/hypothesis pairs over closed systems of 4-6 variables, 18.5% valid). Revision pinned in the
manifest. Licence: MIT (GitHub causalNLP/corr2cause; the Hub card carries no licence tag).

Overlap: the index's causal benchmark is CLadder (5,000 rows of cladder-v1-q-balanced), a different
construction (story-based causal-ladder questions); no CLadder file is read here, and Corr2Cause is
not in JEVal. The filter checks every row anyway. The test/dev and perturbation files are not read.

Sampling (10,000 rows): the train split is 81.5% "no", and the "child" template has no positive in
train at all. We draw up to 800 valid rows from each of the five templates that have them and 1,000
invalid rows from each of the six templates, so labels are 40/60 and every hypothesis type appears.
Half the phrasings ask whether the hypothesis follows and half whether it does not, so the model has
to read the polarity of the question rather than lean on the base rate.
"""
from __future__ import annotations

import random

import pandas as pd

from qwen_decision.v3.sources import common

NAME = "corr2cause-train"
REPO = ("causalnlp/corr2cause", "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53")
POSITIVES, NEGATIVES = 800, 1000
FOLLOWS = [
    "Given only these statistical facts, must the hypothesis be true?",
    "Does the causal claim necessarily follow from the stated correlations and independences?",
    "Is the hypothesis logically implied by what the premise says about this closed system?",
    "The correlational information is enough to conclude the causal hypothesis.",
]
FAILS = [
    "Is the hypothesis left undetermined, or ruled out, by the statistical relations described?",
    "The stated correlations and independences are not enough to conclude the hypothesis.",
    "Would it be a mistake to infer the causal claim from the premise alone?",
    "Does the premise fail to establish the causal hypothesis?",
]
STATE_KEYS = [("premise", "hypothesis"), ("observations", "claim"), ("statistical_relations", "causal_hypothesis")]


def build(local):
    df = pd.read_csv(local / "train.csv")
    rng = random.Random(2023)
    picked = []
    for (template, label), group in df.groupby(["template", "label"], sort=True):
        n = min(len(group), POSITIVES if label == 1 else NEGATIVES)
        picked += [(i, template) for i in group.sample(n=n, random_state=rng.randrange(2**31)).index]
    rng.shuffle(picked)
    for i, template in picked:
        text, label = df.at[i, "input"], bool(df.at[i, "label"])
        premise, hypothesis = text.split("\nHypothesis:", 1)
        premise = premise.removeprefix("Premise:").strip()
        hypothesis = hypothesis.strip()
        a, b = rng.choice(STATE_KEYS)
        if rng.random() < .7:
            state = {a: premise, b: hypothesis}
        else:
            state = f"{a.replace('_', ' ').title()}:\n{premise}\n\n{b.replace('_', ' ').title()}:\n{hypothesis}"
        if rng.random() < .5:
            question, target = common.noul(rng.choice(FOLLOWS), label)
        else:
            question, target = common.noul(rng.choice(FAILS), not label)
        yield common.example(NAME, f"{i}:{template}", "causal", state, question, target, "train")


if __name__ == "__main__":
    local, revision = common.fetch(REPO[0], ["train.csv"], revision=REPO[1])
    common.write(NAME, build(local), {
        "upstream": f"{REPO[0]} (train.csv)", "revision": revision, "split": "train",
        "licence": "MIT (GitHub causalNLP/corr2cause; no licence tag on the Hub card)",
        "sampling": f"up to {POSITIVES} valid per template (5 templates have any) + {NEGATIVES} invalid per template "
                    "(6 templates); upstream train is 18.5% valid", "cast": "noul, half the phrasings inverted"})
