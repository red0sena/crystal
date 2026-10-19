"""ACOS train + dev (Laptop and Restaurant) as per-(aspect category, sentiment) yes/no questions about a review.

Upstream: NUSTM/ACOS, `data/*/*_quad_train.tsv` and `*_quad_dev.tsv` at the commit the index pins
(the suite rebuild already cloned it; read-only). Licence: no licence file in the repo (research
release; Restaurant-ACOS extends SemEval-2016 restaurant reviews, Laptop-ACOS is Amazon reviews).
The index scores only the `*_quad_test.tsv` sentences (catalog 38); train/dev feed only its
category inventory. 30 train/dev sentences repeat a test sentence (measured in the plan); they are
dropped here by normalised text before the filter runs, and the manifest records how many and
whether the filter alone flags them too.

Per sentence: every gold (category, sentiment) pair is a "yes" question; "no" questions are the
same category with another sentiment, a category of the same entity, and a random category of the
domain. Each question is a yes/no (both polarities: is the pair present / is it absent) or a
yes/no choice; categories are shown as words ("laptop / operation performance") or in the raw
ENTITY#ATTRIBUTE form.
"""
from __future__ import annotations

import random

from qwen_decision.v3.decontam import Filter, row_texts, words
from qwen_decision.v3.sources import common

NAME = "acos-train"
REPO = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/repos/acos"
COMMIT = "45d179a3dcc6a3dedd848d81b16f2552454805fe"
SENTIMENTS = {"0": "negative", "1": "neutral", "2": "positive"}
DOMAINS = {"Laptop-ACOS": "laptop", "Restaurant-ACOS": "restaurant"}
PRESENT = [
    "Does the review express a {s} opinion about {c}?",
    "Is there {a} {s} judgement of {c} anywhere in this review?",
    "The reviewer says something {s} about {c}.",
    "Consider only the aspect category {c}: does the text voice a {s} sentiment toward it?",
]
ABSENT = [
    "Is the review free of any {s} opinion about {c}?",
    "The text expresses no {s} sentiment toward {c}.",
    "Is it true that the reviewer never comments on {c} in a {s} way?",
]
CHOICE = [
    "Does this review contain the aspect/sentiment pair ({c}, {s})?",
    "Is {c} mentioned with {s} sentiment in the review?",
    "Answer yes only if the review talks about {c} and the opinion on it is {s}.",
]
YES_NO = [
    {"yes": "the pair occurs in the review", "no": "the pair does not occur"},
    {"yes": None, "no": None},
]


def human(category: str, rng: random.Random) -> str:
    if rng.random() < .3:
        return category
    entity, attribute = category.lower().split("#")
    return f"{entity.replace('_', ' ')} / {attribute.replace('_', ' ')}"


def parse(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        review, *quads = line.rstrip("\n").split("\t")
        yield review, {(q.split()[1], q.split()[2]) for q in quads}


def build(stats: dict):
    rng = random.Random(2026101638)
    test = {" ".join(words(review)) for path in REPO.glob("data/*/*_quad_test.tsv") for review, _ in parse(path)}
    flt, duplicates, flagged = Filter(common.DECONTAM), [], 0
    for domain, short in DOMAINS.items():
        files = sorted((REPO / "data" / domain).glob("*_quad_train.tsv")) + sorted((REPO / "data" / domain).glob("*_quad_dev.tsv"))
        parsed = {path: list(parse(path)) for path in files}
        categories = sorted({c for rows in parsed.values() for _, pairs in rows for c, _ in pairs})
        for path in files:
            split = "train" if path.stem.endswith("train") else "dev"
            for line_no, (review, gold) in enumerate(parsed[path], 1):
                if " ".join(words(review)) in test:
                    duplicates.append(f"{path.name}:{line_no}")
                    flagged += bool(flt.flagged([review]))
                    continue
                negatives = set()
                for c, s in gold:
                    others = [x for x in SENTIMENTS if (c, x) not in gold]
                    if others:
                        negatives.add((c, rng.choice(others)))
                    entity = c.split("#")[0]
                    same = [x for x in categories if x.startswith(entity + "#") and x != c]
                    if same:
                        negatives.add((rng.choice(same), rng.choice(list(SENTIMENTS))))
                negatives.add((rng.choice(categories), rng.choice(list(SENTIMENTS))))
                negatives -= gold
                state = review if rng.random() < .5 else rng.choice([{"review": review}, {"text": review, "domain": f"{short} review"}])
                for c, s in sorted(gold) + sorted(negatives):
                    present = (c, s) in gold
                    fields = dict(c=human(c, rng), s=SENTIMENTS[s])
                    fields["a"] = "an" if fields["s"][0] in "aeiou" else "a"
                    roll = rng.random()
                    if roll < .4:
                        question, target = common.noul(rng.choice(PRESENT).format(**fields), present)
                    elif roll < .6:
                        question, target = common.noul(rng.choice(ABSENT).format(**fields), not present)
                    else:
                        criteria = dict(rng.choice(YES_NO))
                        keys = list(criteria)
                        rng.shuffle(keys)
                        criteria = {k: criteria[k] for k in keys}
                        question, target = common.choice(state, rng.choice(CHOICE).format(**fields), criteria,
                                                         keys.index("yes" if present else "no"))
                    item = f"{domain}:{path.stem}:{line_no}:{c}:{s}"
                    yield common.example(NAME, item, "aspect-sentiment", state, question, target, split)
    stats.update(test_duplicate_sentences=len(duplicates), test_duplicates_flagged_by_filter_alone=flagged,
                 test_duplicate_ids=duplicates)


if __name__ == "__main__":
    meta = {"upstream": "NUSTM/ACOS (data/*/*_quad_train.tsv, *_quad_dev.tsv)", "revision": COMMIT, "split": "train+dev",
            "licence": "no licence file (research release; SemEval-2016 restaurant reviews, Amazon laptop reviews)"}
    common.write(NAME, build(meta), meta)
