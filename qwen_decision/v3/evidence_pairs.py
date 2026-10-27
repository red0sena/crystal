"""Existing-gold evidence-change pairs, aligned augmentation, and paired loss.

No evidence or labels are generated. The only canonicalization is the generic
instruction and equivalent three-choice rendering. Pair partners keep the same
question/claim, option permutation and answer-code assignment.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from difflib import SequenceMatcher
import itertools
import json
import math
import random
import re

import torch
from torch.nn import functional as F

from qwen_decision.v3 import architecture_search as search, dual_matching as dual
from qwen_decision.v3 import robust_adaptation as views

SOURCES = ("condaqa-train", "vitaminc-train")
LABELS = {"condaqa-train": ["yes", "no", "the passage does not say"],
          "vitaminc-train": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]}
INSTRUCTIONS = {"condaqa-train": "Read the passage and answer the question.",
                "vitaminc-train": "How does the evidence bear on the claim?"}


def evidence(row):
    return row["state"]["passage" if row["source"] == "condaqa-train" else "evidence"]


def prompt_key(row):
    key = "passage" if row["source"] == "condaqa-train" else "evidence"
    return search.digest({k: v for k, v in row["state"].items() if k != key})


def canonical(row):
    source = row["source"]
    if (source not in SOURCES or row.get("label_origin") != "human"
            or row["question"]["type"] != "choice" or len(row["target"]) != 3):
        return None
    keys = list(row["question"]["criteria"])
    if set(keys) != set(LABELS[source]) or sum(t > .999 for t in row["target"]) != 1:
        return None
    gold = keys[max(range(3), key=row["target"].__getitem__)]
    target = [float(k == gold) for k in LABELS[source]]
    return row | {"question": {"type": "choice", "instructions": INSTRUCTIONS[source],
                               "criteria": dict.fromkeys(LABELS[source])},
                  "target": target, "ordered": False, "kind": "choice"}


def edits(left, right):
    a, b = re.findall(r"\w+|[^\w\s]", left.lower()), re.findall(r"\w+|[^\w\s]", right.lower())
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    regions = [(i0, i1, j0, j1) for tag, i0, i1, j0, j1 in matcher.get_opcodes() if tag != "equal"]
    changed = sum(max(i1 - i0, j1 - j0) for i0, i1, j0, j1 in regions)
    lexical = any(re.search(r"\w", token) for i0, i1, j0, j1 in regions for token in a[i0:i1] + b[j0:j1])
    return {"similarity": matcher.ratio(), "changed_tokens": changed, "max_tokens": max(len(a), len(b)),
            "edit_regions": len(regions), "lexical_change": lexical}


def make_pairs(rows):
    """One close flip and one close invariant pair per item and literal question."""
    groups = defaultdict(list)
    rejected = Counter()
    for row in rows:
        r = canonical(row)
        if r is None:
            rejected["not_hard_three_way_gold"] += 1
            continue
        groups[(r["source"], r["item"], prompt_key(r))].append(r)
    pairs = []
    for key, rr in sorted(groups.items()):
        seen, unique, conflict = {}, [], False
        for r in sorted(rr, key=lambda x: x["id"]):
            text = " ".join(evidence(r).split())
            y = r["target"].index(1.)
            if text in seen:
                conflict |= seen[text] != y
            else:
                seen[text] = y
                unique.append(r)
        if conflict:
            rejected["conflicting_identical_evidence_group"] += 1
            continue
        candidates = defaultdict(list)
        for a, b in itertools.combinations(unique, 2):
            edit = edits(evidence(a), evidence(b))
            if (edit["similarity"] < .8 or edit["edit_regions"] != 1 or not edit["lexical_change"]
                    or not 0 < edit["changed_tokens"] <= min(16, max(8, .15 * edit["max_tokens"]))):
                rejected["not_close_edit_pair"] += 1
                continue
            kind = "flip" if a["target"] != b["target"] else "invariant"
            pair_id = search.digest(["evidence-pair-v1", a["id"], b["id"]])
            candidates[kind].append({"id": pair_id, "source": a["source"], "group": a["group"],
                "kind": kind, "a": a, "b": b, "edit": edit})
        for kind, cc in candidates.items():
            pairs.append(min(cc, key=lambda p: (p["edit"]["changed_tokens"], p["id"])))
    return pairs, dict(rejected)


def encode(renderer, row, rng=None, *, order=None, codes=None):
    r = views.random_view(renderer, row, rng) if rng is not None else views.view(renderer, row, order, codes)
    return r | dual.content_spans(renderer, row, r)


def encode_pair(renderer, pair, rng=None):
    order = list(range(3))
    codes = list(range(3))
    if rng is not None:
        rng.shuffle(order)
        if rng.random() < .2:
            rng.shuffle(codes)
    a, b = [encode(renderer, pair[k], order=order, codes=codes) for k in ("a", "b")]
    if a["order"] != b["order"] or a["code_ids"] != b["code_ids"]:
        raise ValueError("pair labels/codes are not aligned")
    return {k: pair[k] for k in ("id", "source", "group", "kind")} | {"rows": [a, b]}


def margin_loss(logits, targets, margin=1.):
    """Demand the correct *change* in relative preference for two gold labels.

    delta = (z_a[y_a]-z_a[y_b]) - (z_b[y_a]-z_b[y_b]).
    Gold CE still trains the correctness of each member separately. The paired
    term emphasizes evidence-sensitive changes and is invariant to logit shifts.
    """
    if logits.shape != (2, 3) or targets.shape != (2, 3):
        raise ValueError("one aligned, three-choice pair required")
    y = targets.argmax(-1)
    if y[0] == y[1]:
        raise ValueError("the contrast margin is only defined for flip pairs")
    delta = logits[0, y[0]] - logits[0, y[1]] - logits[1, y[0]] + logits[1, y[1]]
    return F.softplus(margin - delta), delta


def pair_outcomes(pairs, predictions):
    result = []
    for pair in pairs:
        a, b = [predictions[pair[k]["id"]] for k in ("a", "b")]
        result.append({"id": pair["id"], "source": pair["source"], "group": pair["group"], "kind": pair["kind"],
                       "correct": bool(a["correct"] and b["correct"]),
                       "prediction_changed": a["prediction"] != b["prediction"]})
    return result


def response_metrics(outcomes):
    result = {}
    for kind in ("flip", "invariant"):
        rr = [r for r in outcomes if r["kind"] == kind]
        if not rr:
            continue
        by = defaultdict(list)
        for r in rr:
            by[r["source"]].append(r)
        result[kind] = {"pairs": len(rr), "documents": len({r["group"] for r in rr}),
            "both_correct": sum(r["correct"] for r in rr) / len(rr),
            "macro_both_correct": sum(sum(r["correct"] for r in v) / len(v) for v in by.values()) / len(by),
            "prediction_change_rate": sum(r["prediction_changed"] for r in rr) / len(rr),
            "sources": {s: {"pairs": len(v), "both_correct": sum(r["correct"] for r in v) / len(v)} for s, v in by.items()}}
    return result
