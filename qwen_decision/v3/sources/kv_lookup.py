"""Synthetic JSON key-value lookup up to ~32k tokens (the only synthetic long-context family kept).

The long-context families built from real text (paragraph finding, buried QA, in-context labels in
long_context.py) are replaced by real long-document train splits (lexglue-train, cuad-train,
quality-train, musique-train). Key-value lookup stays synthetic: the test task it prepares for (HELMET's
JSON lookup in the proxy set) is itself synthetic and has no real counterpart.

Each example is a JSON object of 50-550 random identifier pairs (up to ~32k Qwen3.5 tokens) and a
choice of the value stored under one key among 10-255 values that all occur in the object.
Nothing here comes from any dataset.
"""
from __future__ import annotations

import json
import random

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.long_context import LOOKUP, ident

NAME = "kv-lookup"
COUNT = 4000
SIZES = [50, 100, 200, 350, 550]  # 550 pairs ~ 30k tokens (450 measured at 24.8k max)


def build():
    rng = random.Random(20261017)
    for i in range(COUNT):
        size = rng.choice(SIZES)
        data, values = {}, set()
        while len(data) < size:
            k_, v_ = ident(rng), ident(rng)
            if k_ not in data and v_ not in values:  # values are distinct so every option is unambiguous
                data[k_] = v_
                values.add(v_)
        key = rng.choice(list(data))
        k = min(len(data), rng.choice([10, 30, 50, 100, 255]))
        options = rng.sample([v for kk, v in data.items() if kk != key], k - 1) + [data[key]]
        rng.shuffle(options)
        state = json.dumps(data, indent=rng.choice([None, 2]))
        question, target = common.choice(state, common.instruction(rng, LOOKUP, key=key), options,
                                         gold=options.index(data[key]))
        yield common.example(NAME, f"kv:{i}", "long-lookup", state, question, target, "generated")


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "generated", "licence": "ours", "sizes": SIZES})
