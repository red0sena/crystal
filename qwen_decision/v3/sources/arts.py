"""Arts and human taste: Habermas Machine training rounds, New Yorker caption matching, Humicroedit pairs.

Habermas Machine (Tessler et al. 2024): google-deepmind/habermas_machine
`hm_all_candidate_comparisons.parquet` as the suite rebuild cloned it (read-only; the file's sha256
is recorded). Licence: CC-BY-4.0 data, Apache-2.0 code. The index (catalog 50) scores only the
EVAL_* cohorts with question.split IID_TEST/OOD_TEST; here only metadata.version TRAINING_DATA_V1..V5
with question.split TRAIN are read (no question overlaps the eval questions, measured). As in the
index, a question is one round's candidate set with the complete human panel; the target is spread
over the statements with the lowest summed rank (ties accepted). Both iterations are used; for the
second one the participants' critiques of the first draft are part of the state.

New Yorker caption contest (Hessel et al. 2023): jmhessel/newyorker_caption_contest config
`matching` (cross-validation split 0) train + validation at the revision the index pins. Licence:
CC-BY-4.0. The index (catalog 64) scores split-0 `matching` test (141 contests); the other folds put
those contests in train, so they are never read, and any row whose contest_number is a test contest
is dropped (0 found; the test file is read only for its contest numbers). Text only: scene,
description, the unusual detail and the linked entities.

Humicroedit (SemEval-2020 Task 7): subtask-2 `train.csv` + `dev.csv` from the task zip the suite
rebuild downloaded (read-only; sha256 pinned by the index). Licence: none stated (SemEval-2020 task
release; the Hub mirror says unknown). The index (catalog 21) scores subtask-2 test; no train/dev
pair uses a test headline (measured: 0 shared originals). Ties (label 0) are dropped, as the
official scorer does. 20% ask for the less funny headline.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import random
import re
import zipfile

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

RAW = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw"

# --- Habermas Machine ---------------------------------------------------------------------------------
HABERMAS = "habermas-train"
HABERMAS_REPO = RAW / "repos/habermas_machine"
HABERMAS_COMMIT = "7923b71966c14136077c78d7841bc9e1a182dfe0"
CONSENSUS = [
    "Several people gave their views on the question below. Which candidate statement would this group, taken together, rank highest?",
    "Predict which statement the participants as a whole would endorse most, judging from their opinions. Each person counts equally.",
    "Which of these group statements would the participants themselves rank best on average?",
    "Given everyone's opinion, choose the statement the group would prefer overall, not the one you personally agree with.",
    "These participants were asked to rank draft consensus statements. Which draft comes out on top when their rankings are combined?",
]
COLUMNS = ["metadata.version", "metadata.status", "round_id", "iteration_index", "question.split", "question.text",
           "rankings.metadata.status", "rankings.candidate_ids", "rankings.numerical_ranks", "candidates.metadata.id",
           "candidates.text", "own_opinion.metadata.id", "own_opinion.text", "other_opinions.metadata.id", "critique.text"]


def habermas(stats: dict):
    source = HABERMAS_REPO / "hm_all_candidate_comparisons.parquet"
    with source.open("rb") as f:
        stats["source_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    groups, filters = collections.defaultdict(list), collections.Counter()
    for r in pq.read_table(source, columns=COLUMNS).to_pylist():
        if not r["metadata.version"].startswith("TRAINING_DATA") or r["question.split"] != "TRAIN":
            filters["not_training_cohort"] += 1
            continue
        if r["metadata.status"] != "COMPLETED" or r["rankings.metadata.status"] != "COMPLETED":
            filters["incomplete_response"] += 1
            continue
        groups[(r["round_id"], r["iteration_index"], tuple(sorted(r["rankings.candidate_ids"])))].append(r)
    rng = random.Random(2026101650)
    for (round_id, iteration, candidates), panel in sorted(groups.items()):
        ref = panel[0]
        expected = set(ref["other_opinions.metadata.id"]) | {ref["own_opinion.metadata.id"]}
        members = [r["own_opinion.metadata.id"] for r in panel]
        if set(members) != expected or len(members) != len(set(members)):
            filters["incomplete_panel"] += 1
            continue
        texts = dict(zip(ref["candidates.metadata.id"], ref["candidates.text"]))
        if any(not (texts.get(c) or "").strip() for c in candidates) or \
                len({" ".join(texts[c].split()) for c in candidates}) != len(candidates):
            filters["blank_or_identical_candidates"] += 1
            continue
        sums = dict.fromkeys(candidates, 0)
        for r in panel:
            for c, rank in zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"]):
                sums[c] += rank
        best = min(sums.values())
        order = list(candidates)
        rng.shuffle(order)
        winners = [c for c in order if sums[c] == best]
        target = [1 / len(winners) if c in winners else 0. for c in order]
        panel = sorted(panel, key=lambda r: r["own_opinion.metadata.id"])
        opinions = [r["own_opinion.text"].strip() for r in panel]
        critiques = [(r["critique.text"] or "").strip() for r in panel] if iteration else []
        if rng.random() < .5:
            state = {"question": ref["question.text"], "opinions": opinions}
            if any(critiques):
                state["critiques of the previous draft"] = [c for c in critiques if c]
        else:
            lines = [f"Question: {ref['question.text']}", ""]
            for k, (o, c) in enumerate(zip(opinions, critiques or [""] * len(opinions)), 1):
                lines.append(f"Participant {k}: {o}")
                if c:
                    lines.append(f"Participant {k}, on the previous draft: {c}")
            state = "\n".join(lines)
        question, target = common.choice(state, rng.choice(CONSENSUS), [" ".join(texts[c].split()) for c in order],
                                         target=target)
        filters[f"kept_iteration_{iteration}"] += 1
        yield common.example(HABERMAS, f"{round_id}:{iteration}", "consensus", state, question, target, "train")
    stats["groups"] = dict(filters)


# --- New Yorker caption matching ------------------------------------------------------------------------
NEWYORKER = "newyorker-matching-train"
NEWYORKER_REPO = "jmhessel/newyorker_caption_contest"
NEWYORKER_REVISION = "d81cbab7d0392708d5371d3a4960e69261824db4"
MATCH = [
    "Which caption goes with the cartoon described here?",
    "Only one of these captions was written for this cartoon; the rest belong to other contests. Pick it.",
    "Match the cartoon description to its caption.",
    "Which caption fits the scene in this cartoon best?",
    "A cartoon is described below. Which of the captions was actually submitted for it?",
]
SCENE_KEYS = [("scene", "what is shown", "what is unusual", "related concepts"),
              ("location", "picture", "odd detail", "entities"),
              ("setting", "description", "the funny part", "topics")]


def entity(url: str) -> str:
    return url.rsplit("/", 1)[-1].replace("_", " ")


def newyorker(stats: dict):
    path, _ = common.fetch(NEWYORKER_REPO, ["matching/train-*", "matching/validation-*"], revision=NEWYORKER_REVISION)
    test = RAW / "newyorker/matching/test-00000-of-00001.parquet"
    test_contests = {r["contest_number"] for r in pq.read_table(test, columns=["contest_number"]).to_pylist()}
    rng = random.Random(2026101664)
    skipped = collections.Counter()
    for split in ("train", "validation"):
        for f in sorted(path.glob(f"matching/{split}-*.parquet")):
            names = [c for c in pq.ParquetFile(f).schema_arrow.names if c != "image"]
            for r in pq.read_table(f, columns=names).to_pylist():
                if r["contest_number"] in test_contests:
                    skipped["test_contest"] += 1
                    continue
                captions = [" ".join(c.split()) for c in r["caption_choices"]]
                if len(set(captions)) != len(captions):
                    skipped["duplicate_captions"] += 1
                    continue
                gold = "ABCDE".index(r["label"])
                values = [r["image_location"], r["image_description"], r["image_uncanny_description"],
                          [entity(u) for u in r["entities"]]]
                if rng.random() < .6:
                    state = dict(zip(rng.choice(SCENE_KEYS), values))
                else:
                    keys = rng.choice(SCENE_KEYS)
                    state = "\n".join(f"{k.capitalize()}: {v if isinstance(v, str) else ', '.join(v)}" for k, v in zip(keys, values))
                question, target = common.choice(state, rng.choice(MATCH), captions, gold)
                yield common.example(NEWYORKER, r["instance_id"], "caption", state, question, target, split)
    stats.update(skipped=dict(skipped), test_contests_excluded=len(test_contests))


# --- Humicroedit --------------------------------------------------------------------------------------------
HUMICROEDIT = "humicroedit-train"
HUMICROEDIT_ZIP = RAW / "downloads/humicroedit-full.zip"
HUMICROEDIT_SHA256 = "12a6cbf28c8b698ad80be42a65ac867b57e4c71662eedab607805e167ba791ab"
FUNNIER = [
    "Each headline had one word swapped to make it funny. Which version is funnier?",
    "Which edited headline would readers find more amusing?",
    "A single word was replaced in each of these headlines. Pick the one that gets the bigger laugh.",
    "Which of the two rewritten headlines works better as a joke?",
]
LESS_FUNNY = [
    "Each headline had one word swapped to make it funny. Which attempt falls flatter?",
    "Which of these edited headlines is less amusing?",
]


def humicroedit(stats: dict):
    with HUMICROEDIT_ZIP.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == HUMICROEDIT_SHA256
    rng = random.Random(2026101621)
    counts = collections.Counter()
    with zipfile.ZipFile(HUMICROEDIT_ZIP) as z:
        for split in ("train", "dev"):
            member = f"semeval-2020-task-7-dataset/subtask-2/{split}.csv"
            for r in csv.DictReader(io.StringIO(z.read(member).decode("utf-8-sig"))):
                if r["label"] not in ("1", "2"):
                    counts["ties"] += 1
                    continue
                headlines, marks = [], []
                for n in (1, 2):
                    text, k = re.subn(r"<[^<>]+/>", lambda _: r[f"edit{n}"], r[f"original{n}"])
                    headlines.append(" ".join(text.strip(' "').split()))
                    marks.append(k)
                if marks != [1, 1] or headlines[0] == headlines[1]:
                    counts["malformed_or_identical"] += 1
                    continue
                funnier = int(r["label"]) - 1
                original = " ".join(re.sub(r"<([^<>]+)/>", r"\1", r["original1"]).strip(' "').split())
                order = [0, 1] if rng.random() < .5 else [1, 0]
                options = [headlines[i] for i in order]
                less = rng.random() < .2
                gold = order.index(1 - funnier if less else funnier)
                state = "" if rng.random() < .5 else {"original headline": original}
                question, target = common.choice(state, rng.choice(LESS_FUNNY if less else FUNNIER), options, gold)
                counts[split] += 1
                yield common.example(HUMICROEDIT, f"{split}:{r['id']}", "humor", state, question, target, split)
    stats["counts"] = dict(counts)


if __name__ == "__main__":
    meta = {"upstream": "google-deepmind/habermas_machine (hm_all_candidate_comparisons.parquet)",
            "revision": HABERMAS_COMMIT, "split": "TRAINING_DATA_V1..V5 / question.split TRAIN",
            "licence": "CC-BY-4.0 (data), Apache-2.0 (code)"}
    common.write(HABERMAS, habermas(meta), meta)
    meta = {"upstream": f"{NEWYORKER_REPO} (matching, cross-validation split 0)", "revision": NEWYORKER_REVISION,
            "split": "train+validation", "licence": "CC-BY-4.0"}
    common.write(NEWYORKER, newyorker(meta), meta)
    meta = {"upstream": "SemEval-2020 Task 7 (semeval-2020-task-7-dataset.zip, subtask-2)", "revision": f"sha256:{HUMICROEDIT_SHA256}",
            "split": "train+dev", "licence": "none stated (SemEval-2020 task release; Hub mirror: unknown)"}
    common.write(HUMICROEDIT, humicroedit(meta), meta)
