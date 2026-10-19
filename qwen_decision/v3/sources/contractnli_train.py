"""ContractNLI train + dev (423 + 61 NDAs x 17 hypotheses) as supported / contradicted / not-mentioned choices.

Upstream: stanfordnlp/contract-nli, `resources/contract-nli.zip` at the commit the index pins (the
suite rebuild already cloned it; read-only). Licence: CC-BY-4.0 (Hitachi America terms of use).
The index scores only `test.json` (123 NDAs, catalog 11); train and dev are separate contracts.
The 17 hypotheses are the same in every split, so they are not item text: the index repeats each
one in all 123 test rows and the filter treats it as a template.

Overlap: some train/dev NDAs are the same template as a test NDA (measured: 6 contain ~all of a
test contract's 13-grams, ~30 more contain over 20%). Those are dropped here, before the filter,
by 13-gram containment in any single test contract (test.json is read only for this). Because that
document-level check is done here, the filter runs in documents mode: a strict filter dropped 86% of
the contracts for boilerplate lines ("NOW, THEREFORE, the parties agree as follows", "Governing Law",
signature blocks) and standard clauses that every NDA shares. The manifest records the counts.

One question per (contract, hypothesis). Documents are long (median ~2k tokens); every train/dev
contract is under 12k Qwen3.5 tokens, so none is cut, and the manifest records the token lengths.
"""
from __future__ import annotations

import json
import random
import zipfile

import numpy as np

from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

NAME = "contractnli-train"
REPO = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/repos/contractnli"
COMMIT = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
MAX_DOC_TOKENS = 12_000
MAX_TEST_CONTAINMENT = .2  # share of a contract's 13-grams found in one test contract
GOLD = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}
# Option sets in (entailment, contradiction, not mentioned) order.
LABELS = [
    [("supported", "the contract clearly provides for this"), ("contradicted", "the contract says the opposite"),
     ("not mentioned", "the contract does not address this")],
    [("entailed", None), ("contradicted", None), ("not mentioned", None)],
    [("yes", "the agreement establishes this"), ("no", "the agreement rules this out"),
     ("not stated", "the agreement is silent on this point")],
    [("confirmed", "a clause of the document backs it"), ("refuted", "a clause of the document conflicts with it"),
     ("absent", "no clause speaks to it")],
]
WITH_HYPOTHESIS = [
    "Read the agreement. Is the following statement supported by it, contradicted by it, or not covered at all?\n{h}",
    "Does this contract back up, contradict, or stay silent on the claim below?\n{h}",
    "Check the statement against the non-disclosure agreement: {h}",
    "According to the contract text, what is the status of this provision?\n{h}",
    "Judging only from the document, how does it treat this point: {h}",
]
GENERIC = [
    "Compare the hypothesis with the contract and choose its status.",
    "Decide whether the agreement supports, contradicts, or does not mention the statement.",
    "Using only the contract, classify the statement that accompanies it.",
]
STATE_KEYS = [("contract", "hypothesis"), ("agreement", "statement"), ("document", "claim")]


def grams(text: str) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + 13]) for i in range(len(w) - 12)}


def build(stats: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(common.ROOT / "models/qwen3.5-0.8b")
    rng = random.Random(2026101611)
    lengths, too_long, near_test = [], 0, []
    with zipfile.ZipFile(REPO / "resources/contract-nli.zip") as z:
        test = [grams(d["text"]) for d in json.loads(z.read("contract-nli/test.json"))["documents"]]
        for split in ("train", "dev"):
            data = json.loads(z.read(f"contract-nli/{split}.json"))
            hypotheses = {key: label["hypothesis"] for key, label in data["labels"].items()}
            for doc in data["documents"]:
                n = len(tokenizer.encode(doc["text"]))
                lengths.append(n)
                if n > MAX_DOC_TOKENS:
                    too_long += 1
                    continue
                g = grams(doc["text"])
                containment = max(len(g & t) for t in test) / max(1, len(g))
                if containment >= MAX_TEST_CONTAINMENT:
                    near_test.append(f"{split}:{doc['id']}")
                    continue
                (annotations,) = [s["annotations"] for s in doc["annotation_sets"]]
                for key in sorted(annotations):
                    labels = rng.choice(LABELS)
                    order = list(range(3))
                    rng.shuffle(order)
                    criteria = {labels[i][0]: labels[i][1] for i in order}
                    gold = order.index(GOLD[annotations[key]["choice"]])
                    h = hypotheses[key]
                    if rng.random() < .6:
                        state = doc["text"] if rng.random() < .6 else {rng.choice(STATE_KEYS)[0]: doc["text"]}
                        instructions = rng.choice(WITH_HYPOTHESIS).format(h=h)
                    else:
                        a, b = rng.choice(STATE_KEYS)
                        state = {a: doc["text"], b: h}
                        instructions = rng.choice(GENERIC)
                    question, target = common.choice(state, instructions, criteria, gold)
                    yield common.example(NAME, f"{split}:{doc['id']}:{key}", "nli", state, question, target, split)
    q = np.percentile(lengths, [50, 99]).round().tolist()
    stats.update(documents=len(lengths), document_tokens={"p50": q[0], "p99": q[1], "max": int(max(lengths))},
                 documents_over_token_cap=too_long, documents_near_test_contract=len(near_test),
                 near_test_contract_ids=near_test)


if __name__ == "__main__":
    # build() adds its statistics to meta; write() reads meta only after consuming the rows.
    meta = {"upstream": "stanfordnlp/contract-nli (resources/contract-nli.zip)", "revision": COMMIT, "split": "train+dev",
            "licence": "CC-BY-4.0", "max_document_tokens": MAX_DOC_TOKENS,
            "max_test_containment": MAX_TEST_CONTAINMENT}
    meta["filter_note"] = "documents mode: per-contract 13-gram containment against every test contract is checked in build()"
    common.write(NAME, build(meta), meta, documents=True)
