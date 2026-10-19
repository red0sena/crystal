"""Fresh held-out set (2026-10-18): datasets in none of our training sources, not in JEVal, not in the public index.

  python -m qwen_decision.v3.fresh_eval build            # writes data/fresh/fresh-v1.jsonl (+ .manifest.json)
  python -m qwen_decision.v3.fresh_eval score --model <checkpoint dir> [--attention noncausal] --out runs/fresh/<name>.json

Sources (validation/test splits, at most 100 rows each, seeded sample):
- PIQA, QASC, QuaRTz, Balanced COPA, SuperGLUE CB and WiC.
- GoEmotions validation, single-label rows only. JEVal uses its test split.
- ETHICS commonsense test, short items.
- SWAG validation.
- BigBench tasks outside BBH: epistemic_reasoning, timedial, presuppositions_as_nli, analogical_similarity,
  anachronisms, fantasy_reasoning, entailed_polarity. BBH is in the index, so its tasks are left out.
Questions follow JEVal's layout: placeholder keys opt_1..opt_k with the option text as description, or noul
for yes/no judgments. Any row whose normalized state text is also a state in our training mix (v3-gold2,
train or dev) is dropped and counted. This guarantees freshness only against our data: what d3-lite was trained
on is not public. Licences differ (some non-commercial); the set is for evaluation only, never training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/fresh"
OUT = ROOT / "data/fresh/fresh-v1.jsonl"
PER_SOURCE = 100
EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
            "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
BIGBENCH = ["epistemic_reasoning", "timedial", "presuppositions_as_nli", "analogical_similarity", "anachronisms",
            "fantasy_reasoning", "entailed_polarity"]


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def table(repo: str, file: str) -> pd.DataFrame:
    p = RAW / repo.replace("/", "__") / file
    return pd.read_csv(p) if file.endswith(".csv") else pq.read_table(p).to_pandas()


def choice(source, rid, state, instructions, options, gold):
    return {"id": f"{source}:{rid}", "source": source, "state": state,
            "question": {"type": "choice", "instructions": instructions,
                         "criteria": {f"opt_{i + 1}": str(o) for i, o in enumerate(options)}},
            "label": f"opt_{gold + 1}"}


def noul(source, rid, state, instructions, truth):
    return {"id": f"{source}:{rid}", "source": source, "state": state,
            "question": {"type": "noul", "instructions": instructions}, "label": bool(truth)}


def rows():
    df = table("baber/piqa", "piqa_validation.parquet")
    for i, r in df.iterrows():
        yield choice("piqa", i, r["goal"], "Which solution achieves the goal?", [r["sol1"], r["sol2"]], int(r["label"]))
    df = table("allenai/qasc", "data/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        yield choice("qasc", r["id"], r["question"], "Which option best answers the question?", texts, labels.index(r["answerKey"]))
    df = table("allenai/quartz", "data/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        yield choice("quartz", r["id"], {"background": r["para"], "question": r["question"]},
                     "Using the background, which option answers the question?", texts, labels.index(r["answerKey"]))
    df = table("pkavumba/balanced-copa", "test.csv")
    for _, r in df.iterrows():
        ask = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        yield choice("copa", r["id"], r["premise"], ask, [r["choice1"], r["choice2"]], int(r["label"]))
    df = table("aps/super_glue", "cb/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        yield choice("cb", r["idx"], {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                     "How does the hypothesis relate to the premise?",
                     ["the premise entails the hypothesis", "the premise contradicts the hypothesis", "neither"], int(r["label"]))
    df = table("aps/super_glue", "wic/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        yield noul("wic", r["idx"], {"word": r["word"], "sentence 1": r["sentence1"], "sentence 2": r["sentence2"]},
                   f"Is the word \"{r['word']}\" used with the same meaning in both sentences?", int(r["label"]) == 1)
    df = table("google-research-datasets/go_emotions", "simplified/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        labels = list(r["labels"])
        if len(labels) == 1:
            yield choice("goemotions", r["id"], r["text"], "Which emotion does this comment express most?", EMOTIONS, int(labels[0]))
    df = table("hendrycks/ethics", "data/commonsense/test.csv")
    for i, r in df.iterrows():
        if str(r["is_short"]) == "True":
            yield noul("ethics-commonsense", i, r["input"], "By ordinary moral standards, is what the narrator did clearly wrong?",
                       int(r["label"]) == 1)
    df = table("allenai/swag", "regular/validation-00000-of-00001.parquet")
    for _, r in df.iterrows():
        yield choice("swag", r["fold-ind"], r["startphrase"], "Which ending most likely continues the scene?",
                     [r[f"ending{k}"] for k in range(4)], int(r["label"]))
    for task in BIGBENCH:
        df = table("tasksource/bigbench", f"{task}/validation-00000-of-00001.parquet")
        for _, r in df.iterrows():
            options, scores = list(r["multiple_choice_targets"]), list(r["multiple_choice_scores"])
            if sum(scores) != 1 or len(options) < 2 or len(set(options)) != len(options):
                continue
            state = re.sub(r"\n\s*A:\s*$", "", str(r["inputs"]).strip())
            yield choice(f"bigbench-{task}", r["idx"], state, "Choose the correct answer.", options, scores.index(1))


def build(args):
    trained = set()
    for f in ("data/mix/v3-gold2.jsonl", "data/mix/v3-gold2.dev.jsonl"):
        for line in (ROOT / f).open():
            r = json.loads(line)
            trained.add(hashlib.sha1(norm(json.dumps(r["state"], ensure_ascii=False)).encode()).hexdigest())
    by, dropped = {}, {}
    for r in rows():
        key = hashlib.sha1(norm(json.dumps(r["state"], ensure_ascii=False)).encode()).hexdigest()
        if key in trained:
            dropped[r["source"]] = dropped.get(r["source"], 0) + 1
            continue
        by.setdefault(r["source"], []).append(r)
    out = []
    for source, items in sorted(by.items()):
        items.sort(key=lambda r: r["id"])
        out += random.Random(f"fresh-v1:{source}").sample(items, min(PER_SOURCE, len(items)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"rows": len(out), "per_source": {s: sum(r["source"] == s for r in out) for s in by},
                "available": {s: len(v) for s, v in by.items()}, "dropped_state_in_training_mix": dropped,
                "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def score(args):
    from qwen_decision.v3.engine import Scorer

    scorer = Scorer(args.model, attention=args.attention)
    by, started = {}, time.time()
    for line in OUT.open():
        r = json.loads(line)
        p = scorer.score(r["state"], {"q": r["question"]})["q"]
        if r["question"]["type"] == "noul":
            right, chance = (p[1] >= .5) == r["label"], .5
        else:
            keys = list(r["question"]["criteria"])
            right, chance = keys[max(range(len(p)), key=p.__getitem__)] == r["label"], 1 / len(keys)
        s = by.setdefault(r["source"], [0, 0, 0.])
        s[0] += 1
        s[1] += right
        s[2] += chance
    sources = {}
    for name, (n, right, chance) in sorted(by.items()):
        acc, ch = right / n, chance / n
        sources[name] = {"n": n, "accuracy": acc, "chance": ch, "skill": max(0., (acc - ch) / (1 - ch))}
    out = {"model": args.model, "data_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "seconds": time.time() - started,
           "macro_accuracy": statistics.mean(v["accuracy"] for v in sources.values()),
           "macro_skill": statistics.mean(v["skill"] for v in sources.values()), "sources": sources}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("model", "macro_accuracy", "macro_skill", "seconds")}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--attention")
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
