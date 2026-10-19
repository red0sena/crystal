"""CLINC150 train + val (+ oos_train / oos_val) as a 151-option intent choice (150 intents + out of scope).

Upstream: clinc/oos-eval `data/data_full.json` (CC-BY-3.0), the same file the index pinned by sha256
(reused from the suite's raw copy when present, else fetched from GitHub at commit 828f809).
Splits: train (15,000), val (3,000), oos_train (100), oos_val (100). Never test / oos_test: the
index scores all 5,500 test rows and JEVal clinc150 samples test. The plan measured 27 train and
4 val utterances identical to test utterances; they are dropped here by normalized text (and the
contamination filter would also catch them).

Options are readable intent names (underscores removed, a few abbreviations spelled out) plus one
out-of-scope option worded several ways; half the questions list them alphabetically, half shuffled.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request

from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

NAME = "clinc-train"
LOCAL = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/clinc150/data_full.json"
COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{COMMIT}/data/data_full.json"
SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
SPLITS = ("train", "val", "oos_train", "oos_val")
RENAME = {
    "apr": "APR", "pto_balance": "PTO balance", "pto_request": "PTO request", "pto_request_status": "PTO request status",
    "pto_used": "PTO used", "w2": "W-2 tax form", "rollover_401k": "401k rollover", "mpg": "MPG (fuel economy)",
    "oil_change_how": "how to change oil", "oil_change_when": "when to change oil", "uber": "Uber ride",
    "text": "send a text message", "yes": "yes (affirm)", "no": "no (deny)", "maybe": "maybe (unsure)",
    "pin_change": "PIN change", "change_ai_name": "change AI name", "are_you_a_bot": "are you a bot", "what_can_i_ask_you": "what can I ask you",
    "min_payment": "minimum payment", "smart_home": "smart home control", "gas": "gas level", "food_last": "how long food lasts",
}
OOS = ["out of scope: no listed intent applies", "none of these (out-of-scope request)",
       "other / unsupported request", "out of domain: the assistant does not handle this"]
PHRASINGS = [
    "What does the user want? Pick the intent this request expresses; if it falls outside every listed intent, pick the out-of-scope option.",
    "Map the user's message to one of the assistant's supported intents, or mark it as out of scope.",
    "Which of these intents is the user expressing? Requests the assistant does not support belong to the out-of-scope option.",
    "Route this request to the right intent handler. A request that no handler covers goes to the out-of-scope option.",
    "Identify the single intent behind the user's request, or choose out of scope when none of the intents fits.",
]
STATE_KEYS = ["utterance", "user_message", "request", "query"]


def source() -> tuple[dict, str]:
    if LOCAL.exists():
        path, origin = LOCAL, "index raw copy"
    else:
        path = common.RAW / f"clinc__oos-eval@{COMMIT[:12]}/data_full.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, path)
        origin = URL
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, path
    return json.loads(path.read_text()), origin


def readable(label: str) -> str:
    return RENAME.get(label, label.replace("_", " "))


def build(data: dict, stats: dict):
    labels = sorted({label for _, label in data["train"]})
    assert len(labels) == 150
    test = {" ".join(words(t)) for split in ("test", "oos_test") for t, _ in data[split]}
    rng = random.Random(150)
    for split in SPLITS:
        for i, (text, label) in enumerate(data[split]):
            if " ".join(words(text)) in test:
                stats["dropped_test_duplicates"][split] += 1
                continue
            oos = rng.choice(OOS)
            names = [readable(x) for x in labels]
            if rng.random() < .5:
                names = sorted(names, key=str.lower) + [oos]
            else:
                names = names + [oos]
                rng.shuffle(names)
            gold = oos if label == "oos" else readable(label)
            r = rng.random()
            state = text if r < .3 else {rng.choice(STATE_KEYS): text}
            question, target = common.choice(state, common.instruction(rng, PHRASINGS), names, names.index(gold))
            yield common.example(NAME, f"{split}:{i}", "intent", state, question, target, split)


if __name__ == "__main__":
    data, origin = source()
    stats = {"dropped_test_duplicates": {s: 0 for s in SPLITS}}
    rows = list(build(data, stats))
    common.write(NAME, rows, {"upstream": "clinc/oos-eval data/data_full.json", "revision": COMMIT, "sha256": SHA256,
                              "origin": origin, "split": "+".join(SPLITS), "licence": "CC-BY-3.0", **stats})
