"""Proxy for the Decision Index private parts: JEVal (carlosxiang/JEVal) through any kit engine.

  python -m qwen_decision.v3.proxy_eval --engine qwen_decision.v3.engine:V3Engine \
      --option model=runs/v3/<run>/final --per-source 60 --out runs/proxy/<name>
  PYTHONPATH=models/d3-lite python -m qwen_decision.v3.proxy_eval --engine d3_engine:D3Engine \
      --option model=models/d3-lite --per-source 60 --out runs/proxy/d3-lite

JEVal is evaluation only: never train on it. Each row is one question. The kit
accepts choice and noul only, so a score question (ordered level texts) is
asked as a choice over its levels in level order. A failed request counts as
wrong, as on the board.

Groups: `same` sources test a skill the index also tests (often the same
dataset); `new` sources are domains the public index does not cover.

TRAINED_ON sources are left out: their datasets' train splits are training data
from 2026-10-17 (lexglue-train, cuad-train), so they no longer measure
generalization. Pass --include-trained to score them anyway.

Protocol (runs/proxy/protocol/jeval-v2.json, made once with --make-protocol): the
kept rows are split into a dev half, used for design decisions, and a restricted
half, scored only for final candidates (--split restricted). Rows that share a
document (mix.data_groups: same passage, contract or dialogue) stay on one side.
The restricted half is restricted from now on, not a clean held-out test: JEVal
results already shaped earlier design choices (data sources, long-context work), so
it limits further selection bias rather than undoing past bias. The real held-out
test is the board's private suite. Every summary records the protocol file's
sha256, the split and the sampling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import time

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
JEVAL = ROOT / "data/jeval/data/test.parquet"
SAME = {"ai2-arc-challenge", "ai2-arc-easy", "boolq", "clinc150", "commonsense-qa", "decider", "fever", "go-emotions",
        "gpqa", "halueval-dialogue", "halueval-general", "halueval-qa", "halueval-summarization", "hellaswag",
        "jevbench", "jevforge", "mmlu-pro", "nlu-eval-68", "scifact", "truthfulqa", "winogrande", "bbq"}


PROTOCOL = ROOT / "runs/proxy/protocol/jeval-v2.json"
TRAINED_ON = {"lex-glue-case-hold", "lex-glue-ecthr-a", "lex-glue-ecthr-b", "lex-glue-ledgar", "lex-glue-scotus",
              "lex-glue-unfair-tos", "cuad"}


def group_of(source: str) -> str:
    return "same" if source in SAME else "new"


def make_protocol() -> dict:
    """Split the kept rows by document group: a group goes to dev when sha256(its first id)[0] is even."""
    from qwen_decision.v3.mix import data_groups

    if PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} exists; a protocol is made once")
    rows = [r for r in pq.read_table(JEVAL, columns=["id", "source", "state", "questions_json"]).to_pylist()
            if r["source"] not in TRAINED_ON]
    items = [{"state": r["state"], "question": next(iter(json.loads(r["questions_json"]).values()))} for r in rows]
    group = data_groups(items)
    first = {}
    for r, g in zip(rows, group):
        first[g] = min(first.get(g, r["id"]), r["id"])
    split = {"dev": [], "restricted": []}
    for r, g in zip(rows, group):
        split["dev" if hashlib.sha256(first[g].encode()).digest()[0] % 2 == 0 else "restricted"].append(r["id"])
    protocol = {"name": "jeval-v2", "jeval": str(JEVAL.relative_to(ROOT)),
                "jeval_sha256": hashlib.sha256(JEVAL.read_bytes()).hexdigest(), "excluded_sources": sorted(TRAINED_ON),
                "rule": "mix.data_groups document groups; sha256(first id of the group)[0] even -> dev, odd -> restricted",
                "groups": len(first), "dev": sorted(split["dev"]), "restricted": sorted(split["restricted"])}
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol) + "\n")
    return protocol


def protocol_ids(split: str) -> tuple[set[str] | None, str | None]:
    if split == "all":
        return None, None
    return set(json.loads(PROTOCOL.read_text())[split]), hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_rows(per_source: int | None, seed: int = 0, sources: set[str] | None = None,
              include_trained: bool = False, split: str = "all") -> list[dict]:
    ids, _ = protocol_ids(split)
    by_source = defaultdict(list)
    for r in pq.read_table(JEVAL).to_pylist():
        if sources and r["source"] not in sources:
            continue
        if r["source"] in TRAINED_ON and not include_trained:
            continue
        if ids is not None and r["id"] not in ids:
            continue
        by_source[r["source"]].append(r)
    rows = []
    for source, items in sorted(by_source.items()):
        items = sorted(items, key=lambda r: r["id"])
        if per_source and per_source < len(items):
            items = random.Random(f"{seed}:{source}").sample(items, per_source)
        rows += items
    return rows


def as_kit(question: dict) -> tuple[dict, list[str] | None]:
    """The kit question and, for score questions, the level keys in order."""
    if question["type"] == "score":
        levels = [str(i) for i in range(len(question["criteria"]))]
        return {"type": "choice", "instructions": question.get("instructions"),
                "criteria": dict(zip(levels, question["criteria"]))}, levels
    return question, None


def judge(question: dict, answer: dict | None, label) -> tuple[bool, float]:
    """(correct, chance accuracy) for one answered question; None answer is wrong."""
    if question["type"] == "noul":
        return (answer is not None and (answer["noul"] >= .5) == bool(label)), .5
    k = len(question["criteria"])
    if answer is None:
        return False, 1 / k
    if question["type"] == "score":
        return answer["choice"] == str(int(label)), 1 / k
    return answer["choice"] == label, 1 / k


def engine_from(spec: str, options: dict):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(**options)


def summarize(records: list[dict]) -> dict:
    by_source = defaultdict(lambda: {"n": 0, "correct": 0, "chance": 0., "errors": 0})
    for r in records:
        s = by_source[r["source"]]
        s["n"] += 1
        s["correct"] += int(r["correct"])
        s["chance"] += r["chance"]
        s["errors"] += int(r["error"] is not None)
    sources = {}
    for name, s in sorted(by_source.items()):
        accuracy, chance = s["correct"] / s["n"], s["chance"] / s["n"]
        sources[name] = {"n": s["n"], "accuracy": accuracy, "chance": chance, "errors": s["errors"],
                         "skill": max(0., (accuracy - chance) / (1 - chance)), "group": group_of(name)}
    out = {"sources": sources}
    for group in ("same", "new", "all"):
        chosen = [v for v in sources.values() if group == "all" or v["group"] == group]
        if chosen:
            out[group] = {"sources": len(chosen), "macro_accuracy": sum(v["accuracy"] for v in chosen) / len(chosen),
                          "macro_skill": sum(v["skill"] for v in chosen) / len(chosen)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine")
    p.add_argument("--option", action="append", default=[], help="key=value engine option")
    p.add_argument("--per-source", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", nargs="*")
    p.add_argument("--out", type=Path)
    p.add_argument("--include-trained", action="store_true", help="also score the TRAINED_ON sources")
    p.add_argument("--split", choices=("dev", "restricted", "all"), default="dev",
                   help="protocol half: dev for decisions, restricted only for final candidates, all = the pre-protocol behaviour")
    p.add_argument("--make-protocol", action="store_true", help="write the protocol file once and exit")
    args = p.parse_args()
    options = dict(o.split("=", 1) for o in args.option)
    if args.make_protocol:
        protocol = make_protocol()
        print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in protocol.items()}))
        return
    if not (args.engine and args.out):
        p.error("--engine and --out are required")
    rows = load_rows(args.per_source, args.seed, set(args.sources) if args.sources else None, args.include_trained, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    engine = engine_from(args.engine, options)
    engine.warmup()
    records, started = [], time.time()
    with (args.out / "predictions.jsonl").open("w") as f:
        for i, row in enumerate(rows):
            questions = json.loads(row["questions_json"])
            labels = json.loads(row["label_json"])
            (key, question), = questions.items()
            label = labels[key] if isinstance(labels, dict) else labels
            kit_question, _ = as_kit(question)
            answer, error = None, None
            tick = time.time()
            try:
                response, _ = engine(row["state"], {key: kit_question})
                answer = response["answers"][key]
            except Exception as e:  # unanswered counts as wrong, as on the board
                error = f"{type(e).__name__}: {e}"[:300]
            correct, chance = judge(question, answer, label)
            record = {"id": row["id"], "source": row["source"], "type": question["type"], "label": label,
                      "answer": answer, "correct": correct, "chance": chance, "error": error,
                      "seconds": time.time() - tick}
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(rows)} rows, {time.time() - started:.0f}s", flush=True)
    summary = {"engine": args.engine, "options": options, "per_source": args.per_source, "seed": args.seed,
               "split": args.split, "protocol_sha256": protocol_ids(args.split)[1], "include_trained": args.include_trained,
               "rows": len(records), "seconds": time.time() - started, **summarize(records)}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "seconds", "same", "new", "all") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
