"""Phase-7 sources (2026-10-19): stance toward a target (VAST's skill) and CommonsenseQA 2.0 yes/no.

  python -m qwen_decision.v3.sources.phase7 --fetch [key ...]   # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase7 [name ...]          # build data/phase2/<name>.jsonl

Chosen for the largest public-sample gaps to d3-lite after BPoMP (VAST -0.28) and for JEVal's commonsense
yes/no (BoolQ -5, CommonsenseQA -6 of 60 questions; the BoolQ/CSQA/WinoGrande train splits are already used).
Vetted on 2026-10-19 (label origin, licence, files):
- EZ-STANCE (Zhao & Caragea, ACL 2024): 47k tweet-target pairs, May 2021-Jan 2023 tweets. Targets and stances
  by human annotators (Cogitotech): noun-phrase targets by majority vote of 3 annotators, claim targets
  written by an annotator for each stance. Released as one zip on Google Drive (the repo README links it;
  sha256 pinned below). No licence file: flagged, like VAST, for review before a public release.
  Subtask A "mixed" (noun-phrase and claim targets) train + val; test never read.
- P-Stance (Li et al., ACL Findings 2021): 21.6k tweets on Trump, Biden and Sanders, favor/against by crowd
  annotators. Google Drive folder linked from the MIT-licensed repo; train + val CSVs (sha256 pinned).
- args.me corpus (Wachsmuth et al. 2017; webis/args_me, CC-BY-4.0): 382k arguments from four debate portals,
  each a premise with its PRO/CON stance toward a conclusion, as the authors posted it (human), extracted by
  the portal structure. Only idebate.org, debatewise.org and debatepedia.org are used (49k): their conclusion
  is the claim the argument addresses. Debate.org (333k) is left out: there the conclusion is the debate
  title and the stance is the debater's side, which often does not read off the text (first sample checked:
  a CON post arguing for the title). Kept: 40-350 words, no forfeit/acceptance boilerplate, at most
  ARGSME_PER_DEBATE arguments per debate page (1,544 pages pass: 9k rows, under the ARGSME_ROWS cap).
- CommonsenseQA 2.0 (Talmor et al. 2021; allenai/csqa2, CC-BY-4.0): 11.8k yes/no commonsense questions written
  by players to beat a model, validated by other players (human). Train + dev; the test has no answers. A
  question is kept only when a strict majority of its validations agree with the answer (no validation, a
  tie or a disagreeing majority drops it: "All races are equal" -> no had one validation, for instance).
Neither the index nor JEVal uses these datasets (the index's stance set is VAST test; JEVal has no stance or
CSQA2 source); every row still goes through the contamination filter.

Stance questions reuse the VAST builder's option sets and phrasings (stance_sarcasm.py). EZ-STANCE is 3-way
(favor / against / neutral). P-Stance and args.me have no neutral class: 70% of their questions are 2-way,
30% 3-way with neutral as a wrong option. Claim targets (6+ words) get claim phrasings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import random
import re
import zipfile

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import fetch_spec, hub_dir
from qwen_decision.v3.sources.stance_sarcasm import STANCE_GENERIC, STANCE_STATE_KEYS, STANCE_WITH_TOPIC, STANCES

RAW = common.RAW / "phase7"
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
PSTANCE_FILES = {  # Google Drive file ids in the folder the P-Stance README links (train and val only)
    "train_bernie": "11OPnWs_mBmsCl32VeK8Sb2aTwKIz4RdV", "train_biden": "1Onq6oAxe--l7aI047T3d979N_JRunbVc",
    "train_trump": "1830SzCs-AllGAQEoMW14rAKIpLH8JBq-", "val_bernie": "1068LHFvG5-zgwfoIsGpJ7v-T-xNUQo60",
    "val_biden": "1NuAM8DItJhoYdukBz9pPHW2e4EjYAd9i", "val_trump": "1xKmh65J1RbpzWb1mnIFs4i3SiAKTHT2V",
}
SHA256 = {  # files fetched on 2026-10-19
    "ezstance": "4683d124c8c8dde1b1282c3f534efa6c0b92e22335a8821249ef1d63cb26455b",
    "pstance_train_bernie": "04bf99d25de6301b8e5e69ecc1b6fdd7036b09c6a603176e330005ff16c1853f",
    "pstance_train_biden": "5fb6784caa7319751c22bb057b318901513ecb50ff221877a23868e967e6488c",
    "pstance_train_trump": "e50fe5f9be3f196f158d135e7cd9166eec2e08047100b8f873175af6a4a9372d",
    "pstance_val_bernie": "de01ddab716edd5ac468839dcb3697428c4147ec0cffeba4f0b9ba060d99a299",
    "pstance_val_biden": "9b7a16f92062ab9960cda3800ab2aa3a56237f8bebb6e3d91bfa8e8d6d4ac4e8",
    "pstance_val_trump": "7bbd23bb98eca6259d099f86082ef0f2cfea02f71c816585fa6026515294accf",
}
ARGSME_REPO, ARGSME_REVISION = "webis/args_me", "042bdf645bc980c751e9b217cc77e47624394390"
CSQA2_COMMIT = "3f57a4a6a1d51fe6a9b462b59be67c08d41fdc39"
FETCH = {
    "ezstance": ("url", DRIVE.format("1WxysM2VwjcZkHnCDMe7P9bwA8VW7IU01"), "ezstance.zip"),
    **{f"pstance_{k}": ("url", DRIVE.format(v), f"raw_{k}.csv") for k, v in PSTANCE_FILES.items()},
    "csqa2": ("github", "allenai/csqa2", CSQA2_COMMIT,
              ["LICENSE", "README.md", "dataset/CSQA2_train.json.gz", "dataset/CSQA2_dev.json.gz"]),
    "argsme": ("hub", ARGSME_REPO, ["README.md", "args-me.jsonl"]),  # pinned to ARGSME_REVISION in main()
}


def raw_file(name: str):
    path = RAW / name / FETCH[name][2]
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == SHA256[name], path
    return path


# --- stance questions ----------------------------------------------------------------------------------
CLAIM_WITH_TARGET = [
    "Does the author of this text agree with the claim \"{t}\", disagree with it, or neither?",
    "Relative to the statement \"{t}\", is the text supportive, opposed, or neutral?",
    "Claim: {t}\nWhat is the writer's position on this claim?",
    "Would the author endorse the claim \"{t}\", reject it, or take no side?",
]
TWO_WAY = [
    "Is the author for or against {t}?",
    "Does this text support or oppose \"{t}\"?",
    "Which side of \"{t}\" does the writer take?",
    "Judging from the text, does its author back {t} or argue against it?",
]
TWO_WAY_GENERIC = ["Is the text for or against the given target?", "Which side of the target does the author take?"]


def stance_row(rng: random.Random, text: str, target: str, counts: list[float], three: bool) -> tuple:
    """(state, question, target) for a stance decision; counts are over (against, favor, neutral)."""
    labels = rng.choice(STANCES)
    order = rng.sample(range(3 if three else 2), 3 if three else 2)
    criteria = {labels[i][0]: labels[i][1] for i in order}
    dist = [counts[i] for i in order]
    total = sum(dist)
    claim = len(target.split()) >= 6
    if rng.random() < .6:
        state = text if rng.random() < .6 else {rng.choice(STANCE_STATE_KEYS)[0]: text}
        pool = (CLAIM_WITH_TARGET if claim else STANCE_WITH_TOPIC) if three else TWO_WAY
        instructions = rng.choice(pool).format(t=target)
    else:
        a, b = rng.choice(STANCE_STATE_KEYS)
        state = {a: text, b: target}
        instructions = rng.choice(STANCE_GENERIC if three else TWO_WAY_GENERIC)
    question, target_dist = common.choice(state, instructions, criteria, target=[x / total for x in dist])
    return state, question, target_dist


def majority(labels: list[int]) -> int | None:
    top = collections.Counter(labels).most_common()
    return None if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


# --- EZ-STANCE ------------------------------------------------------------------------------------------
EZSTANCE = "ezstance-train"
EZ_LABELS = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
TRAILING_NUMBERS = re.compile(r"(?:\s+\d{1,4}){1,4}\s*$")


def tweet_text(original: str, cleaned: str) -> str:
    """The original tweet ('Ori Text') without the number groups the crawl appended (thread or engagement counts):
    trailing numbers are cut when the authors' cleaned 'Text' does not end with them (76% of train rows have some)."""
    tail = TRAILING_NUMBERS.search(original)
    if tail and not cleaned.rstrip().endswith(tail.group(0).strip()):
        original = original[:tail.start()]
    return " ".join((original or cleaned).split())


def ezstance(stats: dict):
    rng = random.Random(2026101931)
    pairs = collections.defaultdict(list)
    with zipfile.ZipFile(raw_file("ezstance")) as z:
        for split in ("train", "val"):
            text = z.read(f"subtaskA/mixed/raw_{split}_all_onecol.csv").decode("utf-8")
            for r in csv.DictReader(io.StringIO(text)):
                post = tweet_text(r["Ori Text"], r["Text"])
                target = " ".join(r["Target 1"].split())
                if post and target and r["Stance 1"] in EZ_LABELS:
                    pairs[(digest(post), digest(target))].append((EZ_LABELS[r["Stance 1"]], split, post, target))
    # pooled by the normalized text the row id uses, so spellings that differ only in case or punctuation share
    # one label (review 2026-10-19: 11 duplicate ids when pooled by raw text)
    for key, found in pairs.items():
        post, target = min((p_, t_) for *_, p_, t_ in found)
        label = majority([x[0] for x in found])
        if label is None:
            stats["ezstance_ties_dropped"] += 1
            continue
        counts = [float(label == i) for i in range(3)]
        state, question, dist = stance_row(rng, post, target, counts, three=True)
        stats["ezstance_target_kind"]["claim" if len(target.split()) >= 6 else "noun phrase"] += 1
        yield common.example(EZSTANCE, f"{key[0]}:{key[1]}", "stance", state, question, dist,
                             "+".join(sorted({x[1] for x in found})))


# --- P-Stance -------------------------------------------------------------------------------------------
PSTANCE = "pstance-train"


def pstance(stats: dict):
    rng = random.Random(2026101932)
    pairs = collections.defaultdict(list)
    for key in PSTANCE_FILES:
        with raw_file(f"pstance_{key}").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["Stance"] in ("FAVOR", "AGAINST"):
                    tweet, target = " ".join(r["Tweet"].split()), r["Target"].strip()
                    pairs[(digest(tweet), digest(target))].append((EZ_LABELS[r["Stance"]], key.split("_")[0], tweet, target))
    for item, found in pairs.items():  # pooled by the id's normalized text, as for EZ-STANCE
        tweet, target = min((t_, g_) for *_, t_, g_ in found)
        label = majority([x[0] for x in found])
        if label is None:
            stats["pstance_ties_dropped"] += 1
            continue
        three = rng.random() < .3
        state, question, dist = stance_row(rng, tweet, target, [float(label == i) for i in range(3)], three)
        stats["pstance_three_way"] += three
        yield common.example(PSTANCE, f"{item[0]}:{item[1]}", "stance", state, question, dist,
                             "+".join(sorted({x[1] for x in found})))


# --- args.me --------------------------------------------------------------------------------------------
ARGSME = "argsme-train"
ARGSME_PORTALS = ("idebate.org", "debatewise.org", "debatepedia.org")
ARGSME_PER_DEBATE, ARGSME_ROWS = 6, 20000
BOILERPLATE = re.compile(r"forfeit|\bi accept\b|extend (all )?my arguments|vote (pro|con)|thanks? for (the|this) debate", re.I)


def argsme(stats: dict):
    rng = random.Random(2026101933)
    debates = collections.defaultdict(list)
    with (hub_dir(ARGSME_REPO, ARGSME_REVISION) / "args-me.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            portal = re.sub(r"^https?://(www\.)?", "", r["context"].get("sourceUrl", "")).split("/")[0]
            if portal not in ARGSME_PORTALS:
                continue
            text = " ".join(p["text"] for p in r["premises"])
            text = " ".join(re.sub(r"\[\[.*?\]\]", "", text).split()).lstrip(".-–• ")
            conclusion = " ".join(r["conclusion"].split())
            stance = {p["stance"] for p in r["premises"]}
            if (len(stance) != 1 or not 40 <= len(text.split()) <= 350 or BOILERPLATE.search(text)
                    or len(conclusion.split()) < 3):
                stats["argsme_filtered"][portal] += 1
                continue
            debates[r["context"]["sourceId"]].append((r["id"], text, conclusion, stance.pop(), portal))
    picked = []
    for source_id in sorted(debates):
        args = debates[source_id]
        picked.append((source_id, rng.sample(args, min(ARGSME_PER_DEBATE, len(args)))))
    rng.shuffle(picked)
    n = 0
    for source_id, args in picked:
        if n >= ARGSME_ROWS:
            break
        for arg_id, text, conclusion, stance, portal in args:
            three = rng.random() < .3
            label = {"CON": 0, "PRO": 1}[stance]
            state, question, dist = stance_row(rng, text, conclusion, [float(label == i) for i in range(3)], three)
            stats["argsme_rows"][portal] += 1
            n += 1
            # source ids hold a timestamp with colons, so the debate is keyed by its digest
            yield common.example(ARGSME, f"{digest(source_id)}:{digest(arg_id)}", "stance", state, question, dist, "corpus")
    stats["argsme_debates_available"] = len(debates)


# --- CommonsenseQA 2.0 ----------------------------------------------------------------------------------
CSQA2 = "csqa2-train"
CSQA2_TRUE = ["Is this true?", "Based on common sense, is the statement correct?", "Answer yes or no: is this right?",
              "Is the following claim accurate?", "Using everyday knowledge, would you say yes or no?"]
CSQA2_FALSE = ["Is this statement false?", "Is the claim wrong?", "Would common sense say this is incorrect?"]
CSQA2_KEYS = ["statement", "question", "claim"]


def csqa2(stats: dict):
    rng = random.Random(2026101934)
    base = RAW / f"allenai__csqa2@{CSQA2_COMMIT[:12]}/dataset"
    for split in ("train", "dev"):
        with gzip.open(base / f"CSQA2_{split}.json.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                votes = r["validations"] or []
                if 2 * sum(v == r["answer"] for v in votes) <= len(votes) or not votes:
                    stats["csqa2"][f"{split}:dropped_unvalidated_or_disputed"] += 1
                    continue
                text, truth = " ".join(r["question"].split()), r["answer"] == "yes"
                flipped = rng.random() < .15
                ask = rng.choice(CSQA2_FALSE if flipped else CSQA2_TRUE)
                x = rng.random()
                if x < .15 and text.endswith("?") and not flipped:
                    state, ask = "", text  # the question itself is the instruction
                elif x < .6:
                    state = text
                else:
                    state = {rng.choice(CSQA2_KEYS): text}
                question, target = common.noul(ask, truth != flipped)
                stats["csqa2"][f"{split}:{'flipped' if flipped else 'plain'}"] += 1
                yield common.example(CSQA2, r["id"], "commonsense-yesno", state, question, target, split)


BUILDERS = [
    (EZSTANCE, ezstance, {"upstream": f"EZ-STANCE zip from the Google Drive link in github chenyez/EZ-STANCE README (sha256 {SHA256['ezstance'][:12]}), "
                                      "subtaskA/mixed train+val", "licence": "no licence file (tweets; research release): review before release",
                          "label_origin": "human (Cogitotech annotators: noun-phrase targets by majority of 3, claim targets written per stance)",
                          "ezstance_ties_dropped": 0, "ezstance_target_kind": collections.Counter()}),
    (PSTANCE, pstance, {"upstream": "P-Stance CSVs from the Google Drive folder in github chuchun8/PStance README (train+val; sha256 pinned)",
                        "licence": "repository MIT; tweets (Twitter terms)", "label_origin": "human (crowd annotators)",
                        "pstance_ties_dropped": 0, "pstance_three_way": 0}),
    (ARGSME, argsme, {"upstream": f"{ARGSME_REPO}@{ARGSME_REVISION[:12]} args-me.jsonl (idebate, debatewise, debatepedia only)",
                      "licence": "CC-BY-4.0", "label_origin": "human (the author's own PRO/CON side; extracted from the portal structure)",
                      "argsme_filtered": collections.Counter(), "argsme_rows": collections.Counter()}),
    (CSQA2, csqa2, {"upstream": f"github allenai/csqa2@{CSQA2_COMMIT[:8]} dataset/CSQA2_{{train,dev}}.json.gz", "licence": "CC-BY-4.0",
                    "label_origin": "human (player-written, validated by other players)", "csqa2": collections.Counter()}),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", nargs="*", default=None, help="download these (all when none given)")
    p.add_argument("names", nargs="*")
    args = p.parse_args()
    if args.fetch is not None:
        RAW.mkdir(parents=True, exist_ok=True)
        for name in args.fetch or list(FETCH):
            if name == "argsme":
                local, revision = common.fetch(ARGSME_REPO, FETCH[name][2], revision=ARGSME_REVISION)
                record = {"name": name, "repo": ARGSME_REPO, "revision": revision, "path": str(local)}
            else:
                record = fetch_spec(name, FETCH[name], RAW)
            print(json.dumps(record), flush=True)
            with (RAW / "fetch.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
    else:
        wanted = set(args.names)
        for name, build, meta in BUILDERS:
            if not wanted or name in wanted:
                common.write(name, build(meta), meta)
