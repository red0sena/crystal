"""Freeze independent classification evaluations before observing either model."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pyarrow.parquet as pq
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = ROOT / "data/external-classification-candidates-v1"
OUT = ROOT / "data/external-classification-eval-v1"
TRAIN = ROOT / "data/text-decisions-v1/train.jsonl"
THRESHOLDS = {"easy_min_human_accuracy": 0.8, "middle_min_human_accuracy": 0.5}


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def tokens(text):
    return re.findall(r"\w+", norm(text))


def shingles(text):
    words = tokens(text)
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def load_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "manifest.json").exists(), "Preserve frozen evaluation data"
    training_manifest = json.loads((TRAIN.parent / "manifest.json").read_text())
    assert sha256(TRAIN) == training_manifest["stats"]["train"]["file_sha256"]
    assert json.loads((TRAIN.parent / "audit.json").read_text())["verified"]
    reference_tags = defaultdict(set)
    def add_ref(text, tag):
        value = norm(text)
        if len(value) >= 20:
            reference_tags[value].add(tag)
    with TRAIN.open() as handle:
        for line in handle:
            row = json.loads(line)
            request = row["request"]
            for text in (request["state"], request["question"], request["state"].split("\n\nAnswer candidates:\n", 1)[0]):
                add_ref(text, "training_input")
    # Also exclude matches to the entire prior source families, even their
    # held-out records, so the new evaluation is more conservative than a
    # train-only exact-overlap screen.
    prior_files = [ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet"]
    for row in pq.read_table(prior_files[0], columns=["question"]).to_pylist():
        add_ref(row["question"], "original_MMLU_Pro")
    for split in ("train", "eval"):
        path = TRAIN.parent / f"raw/furonghuang-lab__Easy2Hard-Bench/E2H-AMC/{split}-00000-of-00001.parquet"
        prior_files.append(path)
        for row in pq.read_table(path, columns=["problem"]).to_pylist():
            add_ref(row["problem"], "original_E2H_AMC")
    references = list(reference_tags)
    inverted = defaultdict(list)
    print(f"Building overlap index for {len(references)} unique reference texts", flush=True)
    for i, text in enumerate(references):
        for shingle in shingles(text):
            inverted[shingle].append(i)
    print(f"Overlap index ready: {len(inverted)} unique five-word shingles", flush=True)

    def overlap(text):
        normalized = norm(text)
        if normalized in reference_tags:
            return {"kind": "exact", "reference_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                    "reference_sources": sorted(reference_tags[normalized])}
        candidate_shingles = shingles(text)
        if len(normalized) < 40 or len(candidate_shingles) < 5:
            return None
        matches = Counter(j for shingle in candidate_shingles for j in inverted.get(shingle, []))
        for j, count in matches.most_common():
            coverage = count / len(candidate_shingles)
            if coverage < 0.5:
                break
            reference = references[j]
            contained = normalized in reference
            # Conservative reuse screen: near-identical question wording or
            # >=90% five-word shingle containment in a training/source text.
            ratio = SequenceMatcher(None, normalized, reference, autojunk=False).ratio() if coverage < 0.9 and len(reference) <= 2*len(normalized) else 0
            if contained or coverage >= 0.9 or ratio >= 0.9:
                return {"kind": "contained" if contained else "near_duplicate", "five_word_coverage": coverage,
                        "sequence_ratio": ratio, "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                        "reference_sources": sorted(reference_tags[reference])}
        return None

    inspection = json.loads((SNAPSHOTS / "inspection.json").read_text())
    super_source = next(d["source"] for d in inspection["datasets"] if d["name"] == "SuperGPQA")
    kmmlu_source = json.loads((SNAPSHOTS / "kmmlu-inspection.json").read_text())["source"]
    for source in (super_source, kmmlu_source):
        for entry in source["files"]:
            assert sha256(ROOT / entry["path"]) == entry["sha256"]
    raw_super = load_lines(SNAPSHOTS / "raw/SuperGPQA/SuperGPQA-all.jsonl")
    raw_kmmlu = []
    for entry in kmmlu_source["files"]:
        path = ROOT / entry["path"]
        if path.name.endswith("-test.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                raw_kmmlu.extend({**r, "snapshot_file": entry["path"], "snapshot_row": i} for i, r in enumerate(csv.DictReader(handle)))
    assert len(raw_super) == 26529 and len(raw_kmmlu) == 35030
    datasets, exclusions = [], {}
    for name, raw, source in (("SuperGPQA", raw_super, super_source), ("KMMLU", raw_kmmlu, kmmlu_source)):
        text_key = "question"
        domain_key = "discipline" if name == "SuperGPQA" else "Category"
        labels = sorted({r[domain_key] for r in raw})
        difficulty_labels = ["easy", "middle", "hard"]
        assert len(labels) == (13 if name == "SuperGPQA" else 45)
        by_text = defaultdict(list)
        for i, row in enumerate(raw):
            by_text[norm(row[text_key])].append(i)
        removed, kept, seen = [], [], set()
        for i, row in enumerate(raw):
            normalized = norm(row[text_key])
            if normalized in seen:
                removed.append({"source_row": i, "reason": "duplicate_question"})
                continue
            seen.add(normalized)
            if len({raw[j][domain_key] for j in by_text[normalized]}) != 1:
                removed.append({"source_row": i, "reason": "conflicting_domain_labels"})
                continue
            hit = overlap(row[text_key])
            if hit:
                removed.append({"source_row": i, "reason": "prior_corpus_overlap", **hit})
                continue
            kept.append((i, row))
            if (i+1) % 5000 == 0:
                print(f"SCREENED {name}: {i+1}/{len(raw)}", flush=True)
        exclusions[name] = removed
        print(f"RETAINED {name}: {len(kept)}/{len(raw)}", flush=True)
        for kind in ("domain", "difficulty"):
            task = name + "_" + kind
            records = []
            missing = 0
            for source_row, row in kept:
                if name == "SuperGPQA":
                    state = row["question"] + "\n\nAnswer candidates:\n" + "\n".join(row["options"])
                    gold = row["discipline"] if kind == "domain" else row["difficulty"]
                    question = "Which academic discipline best matches this problem?" if kind == "domain" else "Estimate this problem's difficulty for a graduate-level student trained in its academic field."
                    annotation = {"official_discipline": row["discipline"], "official_difficulty": row["difficulty"]}
                    row_source = {"dataset": source["repo"], "revision": source["revision"], "split": "train", "row_index": source_row, "uuid": row["uuid"]}
                else:
                    state = row["question"] + "\n\nAnswer candidates:\n" + "\n".join(row[c] for c in "ABCD")
                    question = "Which exam subject best matches this problem?" if kind == "domain" else "Estimate this problem's difficulty for Korean exam takers, using human answer accuracy bands: easy >=80%, middle >=50% and <80%, hard <50%."
                    try:
                        rating = float(row["Human Accuracy"])
                    except ValueError:
                        rating = float("nan")
                    valid = 0 <= rating <= 1
                    if kind == "difficulty" and not valid:
                        missing += 1
                        continue
                    gold = row["Category"] if kind == "domain" else "easy" if rating >= .8 else "middle" if rating >= .5 else "hard"
                    annotation = {"official_subject": row["Category"], "human_accuracy": rating if valid else None}
                    row_source = {"dataset": source["repo"], "revision": source["revision"], "split": "test", "row_index": source_row,
                                  "snapshot_file": row["snapshot_file"], "snapshot_row": row["snapshot_row"]}
                options = labels if kind == "domain" else difficulty_labels
                descriptions = {"easy": "Easy", "middle": "Medium difficulty", "hard": "Hard"} if kind == "difficulty" else {label: label for label in labels}
                # Label order is fixed by content hash before either model runs.
                shuffled = list(options)
                shuffle_key = "external-classification-v1:" + task + ":" + norm(row[text_key])
                rng = random.Random(int(hashlib.sha256(shuffle_key.encode()).hexdigest(), 16))
                rng.shuffle(shuffled)
                request = {"state": state, "question": question, "options": {k: descriptions[k] for k in shuffled}}
                records.append({"id": task + ":" + str(source_row), "task": task, "source": row_source, "request": request,
                                "answer_id": gold, "gold_index": shuffled.index(gold), "option_count": len(shuffled), "annotation": annotation})
            path = OUT / (task + ".jsonl")
            with path.open("w") as handle:
                for r in records:
                    handle.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            datasets.append({"task": task, "dataset": name, "kind": kind, "source_rows": len(raw), "retained_total": len(records),
                             "corpus_overlap_or_duplicate_exclusions": len(removed), "missing_difficulty_exclusions": missing,
                             "labels": labels if kind == "domain" else difficulty_labels, "class_counts": dict(Counter(r["answer_id"] for r in records)),
                             "file": str(path), "sha256": sha256(path), "source": source})
    atomic_json(OUT / "exclusions.json", exclusions)
    atomic_json(OUT / "manifest.json", {
        "complete": True, "datasets": datasets, "total_requests_per_model": sum(s["retained_total"] for s in datasets),
        "training_data_sha256": sha256(TRAIN), "prior_corpora_sha256": {str(p): sha256(p) for p in prior_files},
        "preparation_code_sha256": sha256(Path(__file__)), "exclusions_sha256": sha256(OUT / "exclusions.json"),
        "system_prompt": training_manifest["system_prompt"], "kmmlu_difficulty_thresholds": THRESHOLDS,
        "kmmlu_difficulty_provenance": "Derived bands from native Human Accuracy, thresholds fixed before evaluation; not official three-class annotations. Test taker populations vary by exam.",
        "domain_provenance": "Native 13 SuperGPQA disciplines and 45 KMMLU subjects; no mapping to trained 14-class taxonomy",
        "overlap_protocol": "NFKC/case/whitespace exact checks plus conservative 90% five-word shingle containment / 90% sequence similarity, against training inputs and complete prior MMLU-Pro/E2H-AMC corpora. Paraphrase/pretraining exposure not ruled out.",
        "selection": "All valid retained rows; source-native labels, deterministic per-record shuffled choices; no new training or model selection"})
    print("DATA COMPLETE " + json.dumps({s["task"]: s["retained_total"] for s in datasets}), flush=True)


if __name__ == "__main__":
    main()
