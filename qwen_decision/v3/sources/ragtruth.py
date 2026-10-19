"""RAGTruth train split (15,090 responses) as hallucination yes/no questions.

The index scores the 2,700 test responses (catalog 59); their 450 sources share none of the train
responses' 2,515 sources. Source: ParticleMedia/RAGTruth at the commit the index pins (already
cloned by the suite rebuild). Licence: MIT annotations; contexts from CNN/DM, MS MARCO and Yelp
(training use only, never redistributed).

Half of the phrasings ask whether something is unsupported and half whether everything is
supported, so the model has to read the polarity of the question, not just spot errors.
"""
from __future__ import annotations

import json
import random

from qwen_decision.v3.sources import common

NAME = "ragtruth-train"
REPO = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/repos/cand-RAGTruth"
COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
HALLUCINATED = [
    "Does the response include any claim that the provided context does not support?",
    "The answer adds information that cannot be verified from the given source material.",
    "Is any part of this response unsupported by, or in conflict with, the reference text?",
    "The generated text contains at least one statement that goes beyond or contradicts its source.",
]
FAITHFUL = [
    "Is every statement in the response backed by the context it was given?",
    "The response is fully faithful to the source material: nothing in it is invented or contradicted.",
    "Can each claim in the answer be traced back to the provided context?",
    "All content of the generated text is supported by the reference passage.",
]
STATE_KEYS = [("prompt", "response"), ("context", "answer"), ("source", "generated_text")]


def build():
    sources = {s["source_id"]: s for s in map(json.loads, (REPO / "dataset/source_info.jsonl").open())}
    rng = random.Random(59)
    for r in map(json.loads, (REPO / "dataset/response.jsonl").open()):
        if r["split"] != "train":
            continue
        s = sources[r["source_id"]]
        context = s.get("prompt") or json.dumps(s["source_info"], ensure_ascii=False)
        a, b = rng.choice(STATE_KEYS)
        state = {a: context, b: r["response"]} if rng.random() < .7 else f"{a.title()}:\n{context}\n\n{b.title()}:\n{r['response']}"
        hallucinated = len(r["labels"]) > 0
        if rng.random() < .5:
            question, target = common.noul(rng.choice(HALLUCINATED), hallucinated)
        else:
            question, target = common.noul(rng.choice(FAITHFUL), not hallucinated)
        yield common.example(NAME, r["id"], "hallucination", state, question, target, "train")


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "ParticleMedia/RAGTruth", "revision": COMMIT, "split": "train",
                                 "licence": "MIT annotations; source texts CNN/DM, MS MARCO, Yelp (training only)"})
