"""Shared contract of the phase-2 builders.

An example is one question:
  {"id": "<name>:<source item id>", "source": <name>, "family": <skill family>, "kind": "choice"|"noul"|"score",
   "state": str | object, "question": {"type": "choice", "instructions": ..., "criteria": {key: description|None}}
                                     or {"type": "noul", "instructions": ...},
   "target": [p per option in format.options() order], "ordered": bool (levels keep their order), "split": <upstream split>}
format.options() order is the criteria order for choice and (false, true) for noul.

Rules every builder follows:
- Pin the upstream revision (fetch() records it) and use only the splits the plan allows.
- Never copy an index instruction verbatim: pick from several of our own phrasings (instruction()),
  so the model learns the skill rather than one benchmark's template, and the contamination filter
  does not drop the rows for sharing a template line.
- write() runs the contamination filter on every example and records what it dropped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.decontam import Filter, strings

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/phase2"
DECONTAM = ROOT / "data/decontam"


def fetch(repo: str, patterns: list[str] | None = None, repo_type: str = "dataset", revision: str | None = None) -> tuple[Path, str]:
    """Snapshot of a Hub repo at a pinned revision (the current head when none is given)."""
    revision = revision or HfApi().repo_info(repo, repo_type=repo_type).sha
    local = RAW / f"{repo.replace('/', '__')}@{revision[:12]}"
    snapshot_download(repo, repo_type=repo_type, revision=revision, allow_patterns=patterns, local_dir=local)
    return local, revision


def instruction(rng: random.Random, phrasings: list[str], **fields) -> str:
    return rng.choice(phrasings).format(**fields)


def choice(state: Any, instructions: str, options: list[str] | dict[str, Any], gold: int | None = None,
           target: list[float] | None = None) -> tuple[dict, list[float]]:
    """A choice question over `options` (texts, or {key: description}) and its target distribution."""
    criteria = options if isinstance(options, dict) else {o: None for o in options}
    if len(criteria) != len(options) or not 2 <= len(criteria) <= fmt.MAX_OPTIONS:
        raise ValueError("options must be 2-255 distinct keys")
    if target is None:
        target = [0.] * len(criteria)
        target[gold] = 1.
    return {"type": "choice", "instructions": instructions, "criteria": criteria}, target


def noul(instructions: str, truth: bool | float, criteria: dict[str, str] | None = None) -> tuple[dict, list[float]]:
    """A yes/no question (optionally with true/false descriptions); the target is [P(false), P(true)]."""
    p = float(truth)
    question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question, [1. - p, p]


def example(name: str, item_id: str, family: str, state: Any, question: dict, target: list[float],
            split: str, ordered: bool = False, kind: str | None = None) -> dict:
    return {"id": f"{name}:{item_id}", "source": name, "family": family, "kind": kind or question["type"],
            "state": state, "question": question, "target": target, "ordered": ordered, "split": split}


def write(name: str, examples: Iterable[dict], meta: dict, documents: bool = False, tables=None) -> dict:
    """Filter and write one source; returns the manifest.

    documents=True relaxes the filter for long documents (see decontam.Filter.flagged); use it only
    when the builder has itself checked document-level overlap with the test set, and say how in meta.
    tables(example) -> (state texts, table text): the table text is checked by 13-word shingles only, like
    option texts, because a generic row ("Total 1,234 5,678") matches short index lines once digits are
    masked; the rest of the state stays under the strict checks.
    """
    flt = Filter(DECONTAM)
    OUT.mkdir(parents=True, exist_ok=True)
    kept, dropped, reasons, seen = 0, 0, [], set()
    started = time.time()
    with (OUT / f"{name}.jsonl").open("w") as f:
        for e in examples:
            keys, _ = fmt.options(e["question"])
            assert len(keys) == len(e["target"]) and abs(sum(e["target"]) - 1) < 1e-6, e["id"]
            key = hashlib.sha256(json.dumps([e["state"], e["question"]], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            q = e["question"]
            texts, table = tables(e) if tables else (strings(e["state"]), [])
            why = flt.flagged(texts + strings(q.get("instructions")), documents=documents,
                              options=strings(q.get("criteria")) + table)
            if why:
                dropped += 1
                if len(reasons) < 20:
                    reasons.append({"id": e["id"], "why": why})
                continue
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            kept += 1
    manifest = {"name": name, **meta, "kept": kept, "dropped_contaminated": dropped, "drop_examples": reasons,
                "filter_mode": ("documents" if documents else "strict") + ("; table text by shingles only" if tables else ""),
                "decontam_index": json.loads((DECONTAM / "meta.json").read_text()), "seconds": time.time() - started}
    (OUT / f"{name}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("name", "kept", "dropped_contaminated")}))
    return manifest


def load(names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        with (OUT / f"{name}.jsonl").open() as f:
            rows += [json.loads(line) for line in f]
    return rows
