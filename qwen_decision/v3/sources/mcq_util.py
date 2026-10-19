"""Shared cast for the native multiple-choice builders (math_mcq, science_mcq, mmlu_aux, medmcqa).

The option texts become the criteria keys in their native order (so "all of the above" or
"both A and B" still point at the right codes), the gold stays in `target`, and the question goes
in the state, either as plain text or under a randomly chosen key, so the model does not learn one
layout. A row whose options are empty or repeat is skipped.
"""
from __future__ import annotations

import random
import re

from qwen_decision.v3.sources import common

_SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    return _SPACE.sub(" ", str(text)).strip()


def state_of(rng: random.Random, question: str, context: str | None = None,
             keys: tuple[str, ...] = ("question",), context_keys: tuple[str, ...] = ("passage",)):
    if context:
        ckey, qkey = rng.choice(context_keys), rng.choice(keys)
        if rng.random() < .6:
            return {ckey: context, qkey: question}
        return f"{ckey.replace('_', ' ').title()}:\n{context}\n\n{qkey.replace('_', ' ').title()}:\n{question}"
    if rng.random() < .5:
        return question
    return {rng.choice(keys): question}


def cast(name: str, item_id: str, family: str, split: str, rng: random.Random, phrasings: list[str],
         question: str, options: list[str], gold: int, context: str | None = None,
         keys: tuple[str, ...] = ("question",), context_keys: tuple[str, ...] = ("passage",)) -> dict | None:
    options = [clean(o) for o in options]
    question = question.strip()
    if not question or any(not o for o in options) or len(set(options)) != len(options):
        return None
    if not 0 <= gold < len(options):
        return None
    state = state_of(rng, question, context, keys, context_keys)
    q, target = common.choice(state, common.instruction(rng, phrasings), options, gold)
    return common.example(name, item_id, family, state, q, target, split)


def capped(rng: random.Random, items: list, cap: int) -> list:
    """A seeded random subset of at most `cap` items, in a shuffled order."""
    items = list(items)
    rng.shuffle(items)
    return items[:cap]
