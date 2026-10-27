"""Fixed ordinary-CE data and new document tests for semantic matching.

Reuses an audited pool of never-scored official held-out documents, explicitly
excluding every document touched by the preceding pair experiment and all
currently recorded predictions. No contrast-pair loss or pair selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from qwen_decision.v3 import evidence_pair_data as prior_data, long_training as common
from qwen_decision.v3 import semantic_matching as semantic

PREVIOUS = Path("runs/evidence-pairs/factorial-v1")
TARGET = ("condaqa-train", "vitaminc-train", "boolq-train", "snli-train", "mnli-train")
PRIMARY = ("condaqa-train", "vitaminc-train", "retention-boolq", "retention-snli", "retention-mnli")
RETENTION = ("piqa", "qasc", "goemotions", "ethics-commonsense", "swag", "retention-hellaswag")


def lines(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def digest(value):
    return common.search.digest(value)


def balanced(rows, total, salt):
    by = defaultdict(list)
    for row in rows:
        by[row["source"]].append(row)
    for source in by:
        by[source].sort(key=lambda r: digest([salt, r["id"]]))
    result, i = [], 0
    while len(result) < total:
        changed = False
        for source in sorted(by):
            if i < len(by[source]) and len(result) < total:
                result.append(by[source][i]); changed = True
        if not changed:
            raise ValueError("not enough balanced training cases")
        i += 1
    return result


def build(out):
    renderer, _ = common.search.renderer_for(common.search.BASE)
    inputs = json.loads((PREVIOUS / "inputs.json").read_text())
    files = dict(inputs["files"])
    for path, sha in files.items():
        if common.train.sha256_file(Path(path)) != sha:
            raise ValueError(f"audited source changed: {path}")
    for name in ("manifest.json", "inputs.json", "split-audit.json", "report.json", "data-summary.json",
                 "pair-dev.jsonl", "pair-test.jsonl", "retention-test.jsonl", "retained-candidates.jsonl"):
        path = PREVIOUS / name
        files[str(path)] = common.train.sha256_file(path)
    seen_ids, seen_groups = set(), set()
    prediction_files = []
    for path in sorted(Path("runs").rglob("*.jsonl")):
        if out in path.parents:
            continue
        try:
            with path.open() as f:
                first = json.loads(next(f))
        except (ValueError, StopIteration):
            continue
        if not isinstance(first, dict) or "probabilities" not in first or "id" not in first:
            continue
        prediction_files.append(str(path))
        for r in lines(path):
            seen_ids.add(r["id"])
            if r.get("group"):
                seen_groups.add(r["group"])
        files[str(path)] = common.train.sha256_file(path)
    for name in ("pair-dev", "pair-test", "retention-test"):
        for r in lines(PREVIOUS / f"{name}.jsonl"):
            seen_groups.add(r["group"])
    by = defaultdict(lambda: defaultdict(list))
    pool = list(lines(PREVIOUS / "retained-candidates.jsonl"))
    # Propagate any newly seen example ID to its entire original document group.
    seen_groups.update(r["group"] for r in pool if r["id"] in seen_ids)
    for r in pool:
        if r["group"] in seen_groups or r["source"] not in PRIMARY + RETENTION:
            continue
        if r["source"] not in ("condaqa-train", "vitaminc-train") and not prior_data.independent_retention(r):
            continue
        by[r["source"]][r["group"]].append(r)
    dev, primary, retention, source_counts = [], [], [], {}
    for source in PRIMARY + RETENTION:
        candidates = []
        groups = by[source]
        for group in sorted(groups, key=lambda g: digest(["single-forward-doc-v1", source, g])):
            selected = None
            for row in sorted(groups[group], key=lambda r: digest(["single-forward-row-v1", r["id"]])):
                if len(common.views.view(renderer, row)["input_ids"]) <= 4096:
                    selected = row; break
            if selected is not None:
                candidates.append(selected)
            if len(candidates) >= 32 + (256 if source in PRIMARY else 128):
                break
        count = 256 if source in PRIMARY else 128
        if len(candidates) < 32 + count:
            raise ValueError(f"insufficient new documents: {source}: {len(candidates)}")
        dev += candidates[:32]
        (primary if source in PRIMARY else retention).extend(candidates[32:32 + count])
        source_counts[source] = {"dev": 32, "test": count}
        print("NEW_DOCUMENTS", source, source_counts[source], flush=True)
    full = Path("data/mix/v3-gold6.jsonl")
    files[str(full)] = common.train.sha256_file(full)
    pools = defaultdict(list)
    for r in lines(full):
        if r["source"] in TARGET and r.get("label_origin") in ("human", "programmatic"):
            pools[r["source"]].append(r)
    target_pool = []
    for source in TARGET:
        selected = []
        ids = set()
        for r in sorted(pools[source], key=lambda r: digest(["single-forward-target-v1", r["id"]])):
            if r["id"] not in ids and len(common.views.view(renderer, r)["input_ids"]) <= 1536:
                selected.append(r); ids.add(r["id"])
            if len(selected) == 820:
                break
        if len(selected) < 819:
            raise ValueError(f"not enough unique target examples: {source}")
        target_pool += selected
    targets = balanced(target_pool, 4096, "single-forward-target-v1")
    anchor_path = common.OUT / "train.jsonl"
    files[str(anchor_path)] = common.train.sha256_file(anchor_path)
    anchors = balanced([r for r in lines(anchor_path) if r["source"] not in TARGET and
                        r.get("label_origin") in ("human", "programmatic")], 4096, "single-forward-anchor-v1")
    if len({r["id"] for r in targets + anchors}) != 8192:
        raise ValueError("duplicate training ID")
    meta = prior_data.metadata()
    train_keys = {k for r in targets + anchors for k in prior_data.keys(r, meta)}
    sets = {"dev": dev, "primary-test": primary, "retention-test": retention}
    audit = {}
    eval_keys, eval_groups = set(), set()
    for name, rr in sets.items():
        keys = {k for r in rr for k in prior_data.keys(r, meta)}
        groups = {r["group"] for r in rr}
        if len(groups) != len(rr) or keys & train_keys or keys & eval_keys or groups & seen_groups or groups & eval_groups:
            raise ValueError(f"evaluation independence failed: {name}")
        audit[name] = {"rows": len(rr), "distinct_documents": len(groups), "train_overlap_keys": 0,
                       "previous_eval_overlap_keys": 0, "previously_scored_documents": 0}
        eval_keys |= keys; eval_groups |= groups
    all_rows = {"target-train": targets, "anchor-train": anchors, **sets}
    for name, rr in all_rows.items():
        path = out / f"{name}.jsonl"
        common.search.save_records(path, rr)
        files[str(path)] = common.train.sha256_file(path)
    summary = {"rows": {k: len(v) for k, v in all_rows.items()}, "training_sources": dict(Counter(r["source"] for r in targets + anchors)),
        "evaluation_sources": source_counts, "new_prediction_files_audited": len(prediction_files),
        "inherited_audit": str(PREVIOUS / "split-audit.json"), "direct_independence": audit,
        "sampling": "one hash-selected original gold case per previously unscored document; source caps fixed before scoring",
        "limits": ["original pretraining contamination unknown", "normalized exact and explicit metadata audit, not exhaustive semantic deduplication",
                   "CONDAQA groups original passages, not entire articles", "fresh MNLI validation is three-way while existing training includes binary entailment judgments",
                   "targeted five-task probe plus six retention families, not the full 100-source general benchmark"]}
    common.train.atomic_json(out / "data-summary.json", summary)
    files[str(out / "data-summary.json")] = common.train.sha256_file(out / "data-summary.json")
    common.train.atomic_json(out / "inputs.json", {"files": files, "summary": summary})
    print("DATA_READY", summary["rows"], flush=True)
    return summary


def encode(renderer, row, rng=None):
    encoded = common.views.random_view(renderer, row, rng) if rng is not None else common.views.view(renderer, row)
    return encoded | semantic.content_spans(renderer, row, encoded)
