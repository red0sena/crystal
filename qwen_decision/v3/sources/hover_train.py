"""HoVer train (18,171 multi-hop claims) with their oracle Wikipedia intro paragraphs, 2-way supported / not supported.

Upstream: hover-nlp/hover `data/hover/hover_train_release_v1.1.json` at the commit the index pins,
plus the HoVer wiki dump `wiki_wo_links.db` (both CC-BY-SA-4.0), reused read-only from the suite's
raw copy. The index scores all 4,000 dev claims (oracle documents from the same db); train claims
are distinct (0 exact overlaps), but about 1,900 train claims cite a Wikipedia article that a dev
claim also cites, so the contamination filter drops some rows for sharing a document, not a claim.

The evidence is every distinct supporting-fact article (title + full intro paragraph), shuffled so
the order carries no signal. 70% are 2-option choices, 30% yes/no questions whose polarity varies.
"""
from __future__ import annotations

import json
import random
import sqlite3
import unicodedata

from qwen_decision.v3.sources import common

NAME = "hover-train"
RAW = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw"
CLAIMS = RAW / "repos/cand-hover/data/hover/hover_train_release_v1.1.json"
DB = RAW / "hover/wiki_wo_links.db"
COMMIT = "39b84697f196308f398a251a7aea9b82ae0f0562"
CHOICE = [
    "Do these Wikipedia passages, taken together, back up the claim?",
    "Check the claim against the evidence articles. Does the evidence confirm every part of it?",
    "Read the claim and the supporting articles. Is the claim verified by them?",
    "Combining the facts in these passages, is the claim supported or not?",
    "Decide whether the evidence establishes the claim. Several passages may need to be connected.",
]
OPTIONS = [
    ({"supported": "Every part of the claim follows from the passages.",
      "not supported": "Some part of the claim is contradicted by the passages or does not follow from them."}, ("supported", "not supported")),
    ({"verified": None, "not verified": None}, ("verified", "not verified")),
    ({"true according to the evidence": None, "false or unverifiable from the evidence": None},
     ("true according to the evidence", "false or unverifiable from the evidence")),
]
SUPPORTED = [
    "Taken together, the evidence passages fully support the claim.",
    "Is every part of the claim confirmed by the given Wikipedia articles?",
    "The claim can be verified from these passages.",
]
REFUTED = [
    "At least one detail of the claim is contradicted by, or missing from, the evidence.",
    "Does the claim contain something the given articles do not back up?",
]


def evidence(db, facts, rng):
    docs = []
    for title in dict.fromkeys(t for t, _ in facts):
        found = db.execute("SELECT text FROM documents WHERE id=?", (unicodedata.normalize("NFD", title),)).fetchall()
        assert len(found) == 1, title
        docs.append({"title": title, "text": found[0][0]})
    rng.shuffle(docs)
    return docs


def build():
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rng = random.Random(61)
    for r in json.loads(CLAIMS.read_text(encoding="utf-8")):
        docs = evidence(db, r["supporting_facts"], rng)
        if rng.random() < .6:
            state = {"claim": r["claim"], "evidence": docs}
        else:
            text = "\n\n".join(f"{d['title']}: {d['text']}" for d in docs)
            state = {"claim": r["claim"], "passages": text} if rng.random() < .5 else f"Claim: {r['claim']}\n\nEvidence:\n{text}"
        supported = r["label"] == "SUPPORTED"
        if rng.random() < .7:
            options, (yes, no) = rng.choice(OPTIONS)
            keys = list(options) if rng.random() < .5 else list(options)[::-1]
            options = {k: options[k] for k in keys}
            question, target = common.choice(state, common.instruction(rng, CHOICE), options, keys.index(yes if supported else no))
        elif rng.random() < .6:
            question, target = common.noul(rng.choice(SUPPORTED), supported)
        else:
            question, target = common.noul(rng.choice(REFUTED), not supported)
        yield common.example(NAME, r["uid"], "claim-verification", state, question, target, "train")
    db.close()


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "hover-nlp/hover (train release v1.1) + wiki_wo_links.db", "revision": COMMIT,
                                 "split": "train", "licence": "CC-BY-SA-4.0"})
