"""GSM8K main train split (7,473 problems) as number-choice questions, no working shown.

Upstream: openai/gsm8k, config main, split train only (fetched with the pattern main/train-*, so the
test parquet is never downloaded). Revision: the head at build time, pinned in the manifest.
Licence: MIT.

Overlap: the index scores the 1,319 test problems (catalog 30, GSM8K-4 and GSM8K-10); the plan
measured 0 train/test duplicates, and the filter still runs on every row (it masks digits, so a
numeric rewrite of a test problem is caught too). GSM-Hard, GSM-Plus, GSM-Symbolic and
GSM8K-Platinum are rewrites of the test set and are never used.

Distractors follow the idea of the index's 0.3 answer groups, rebuilt inside train only. Problems
are sorted by gold and cut into bands; a band of 10*m problems, in which no value repeats more than
m times, is dealt round-robin into m groups of ten, so each group holds ten distinct golds of
similar size. A problem's ten options are its group's ten golds: every value shown is the right
answer for exactly one problem that sees that same set, so the values alone carry no signal. Its
four options are a window of four neighbours on the group's sorted ring of values; where the gold
sits in the window is the same within a group and balanced across groups, so each value appears in
exactly four windows and is the gold of one. The few problems left over (n mod 10) are padded with
the nearest other train golds. Both a 4-option and a 10-option row are written per problem.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

NAME = "gsm8k-train"
REPO = "openai/gsm8k"
SEED = 7473
K, WINDOW = 10, 4
PHRASINGS = [
    "Which number is the answer to this word problem?",
    "Solve the problem in the state and select its final numeric answer.",
    "Work the problem out and pick the value it asks for.",
    "Which of these values correctly answers the question?",
    "Select the number that the word problem resolves to.",
    "Pick the candidate equal to the problem's final answer.",
]
STATE_KEYS = ["problem", "question", "word_problem"]


def gold(answer: str) -> Decimal:
    return Decimal(answer.split("####")[-1].strip().replace(",", ""))


def text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def longest_run(values: list[Decimal]) -> int:
    best = run = 1
    for a, b in zip(values, values[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def cut_bands(values: list[Decimal]) -> list[tuple[int, int]]:
    """(start, m) bands over sorted values: 10*m items each, no value more than m times in a band."""
    bands, start, n = [], 0, len(values)
    while start < n:
        m = 1
        while start + K * m <= n and longest_run(values[start:start + K * m]) > m:
            m += 1
        if start + K * m <= n:
            bands.append((start, m))
            start += K * m
            continue
        # The tail cannot form a band of its own: merge it backwards until the merged band is valid.
        while True:
            start = bands.pop()[0] if bands else 0
            m = (n - start) // K
            if longest_run(values[start:n]) <= m:
                bands.append((start, m))
                return bands
            if start == 0:
                raise ValueError("one value fills more than a tenth of the problems")
    return bands


def answer_groups(golds: dict[int, Decimal], rng: random.Random) -> list[dict]:
    salt = {i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest() for i in golds}
    order = sorted(golds, key=lambda i: (golds[i], salt[i]))
    # Set aside n mod 10 problems with distinct golds; they get padded groups at the end.
    spare = len(order) % K
    left, seen = [], set()
    for i in rng.sample(order, len(order)):
        if len(left) == spare:
            break
        if golds[i] not in seen:
            left.append(i)
            seen.add(golds[i])
    taken = set(left)
    rest = [i for i in order if i not in taken]
    values = [golds[i] for i in rest]
    groups = []
    for start, m in cut_bands(values):
        for j in range(m):
            members = [rest[start + j + t * m] for t in range(K)]
            groups.append({"members": members, "values": [golds[i] for i in members]})
    if left:
        own = {golds[i] for i in left}
        distinct = sorted(set(golds.values()) - own)
        lo, hi = min(own), max(own)
        near = sorted(distinct, key=lambda v: (min(abs(v - lo), abs(v - hi)), v))[:K - len(left)]
        groups.append({"members": left, "values": [golds[i] for i in left] + near})
    for g in groups:
        assert len(set(g["values"])) == K, g
    return groups


def option_sets(golds: dict[int, Decimal]) -> dict[int, dict[int, list[Decimal]]]:
    rng = random.Random(SEED)
    groups = answer_groups(golds, rng)
    offsets = []
    while len(offsets) < len(groups):
        block = list(range(WINDOW))
        rng.shuffle(block)
        offsets += block
    sets = {}
    for g, offset in zip(groups, offsets):
        ring = sorted(g["values"])
        for i in g["members"]:
            at = ring.index(golds[i])
            four = [ring[(at - offset + t) % K] for t in range(WINDOW)]
            sets[i] = {K: list(g["values"]), WINDOW: four}
    return sets


def build(local):
    rows = pq.read_table(next(local.glob("main/train-*.parquet"))).to_pylist()
    golds = {i: gold(r["answer"]) for i, r in enumerate(rows)}
    sets = option_sets(golds)
    rng = random.Random(SEED + 1)
    for i, r in enumerate(rows):
        for count in (WINDOW, K):
            options = [text(v) for v in sets[i][count]]
            rng.shuffle(options)
            problem = r["question"].strip()
            state = problem if rng.random() < .5 else {rng.choice(STATE_KEYS): problem}
            question, target = common.choice(state, common.instruction(rng, PHRASINGS), options,
                                             options.index(text(golds[i])))
            yield common.example(NAME, f"{i}:{count}", "math", state, question, target, "train")


if __name__ == "__main__":
    local, revision = common.fetch(REPO, ["main/train-*"])
    common.write(NAME, build(local), {
        "upstream": f"{REPO} (main)", "revision": revision, "split": "train", "licence": "MIT",
        "cast": "choice over 4 and over 10 numbers; distractors are other train problems' golds from the same "
                "answer group (10 distinct golds of similar size), so option values alone score at chance"})
