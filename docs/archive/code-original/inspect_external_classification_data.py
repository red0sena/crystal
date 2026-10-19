"""Pin and inspect new domain/difficulty data; no model training or evaluation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.parse

import pyarrow.parquet as pq

from fetch_text_sources import get, download
from evaluate_mmlu_pro import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external-classification-candidates-v1"
SPECS = [
    {"name": "SuperGPQA", "repo": "m-a-p/SuperGPQA", "split": "train", "expected_rows": 26529,
     "paths": ["SuperGPQA-all.jsonl"], "text": "question", "domain": ["discipline", "field", "subfield"], "difficulty": "difficulty"},
    {"name": "Omni-MATH", "repo": "KbsdJames/Omni-MATH", "split": "test", "expected_rows": 4428,
     "paths": ["test.jsonl"], "text": "problem", "domain": ["domain"], "difficulty": "difficulty"},
    {"name": "MATH", "repo": "EleutherAI/hendrycks_math", "split": "test", "expected_rows": 5000,
     "paths": [f"{c}/test-00000-of-00001.parquet" for c in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")],
     "text": "problem", "domain": ["type"], "difficulty": "level"},
]


def normalize(s):
    return re.sub(r"\s+", " ", s).strip().casefold()


def fetch(spec):
    directory = OUT / "raw" / spec["name"]
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "manifest.json"
    if cache.exists():
        manifest = json.loads(cache.read_text())
        for f in manifest["files"]:
            assert sha256(ROOT / f["path"]) == f["sha256"]
    else:
        info = json.loads(get("https://huggingface.co/api/datasets/" + spec["repo"]))
        revision = info["sha"]
        siblings = {x["rfilename"] for x in info["siblings"]}
        assert set(spec["paths"]) <= siblings
        files = []
        for path in [*spec["paths"], *(["README.md"] if "README.md" in siblings else [])]:
            files.append(download(f"https://huggingface.co/datasets/{spec['repo']}/resolve/{revision}/{path}", directory / path))
        viewer_splits = json.loads(get("https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": spec["repo"]})))
        manifest = {**spec, "revision": revision, "license": info.get("cardData", {}).get("license"),
                    "files": files, "viewer_splits": viewer_splits}
        atomic_json(cache, manifest)
    rows = []
    for path in spec["paths"]:
        local = directory / path
        if local.suffix == ".parquet":
            rows.extend(pq.read_table(local).to_pylist())
        else:
            with local.open() as handle:
                rows.extend(json.loads(line) for line in handle)
    assert len(rows) == spec["expected_rows"]
    print(f"DOWNLOADED {spec['name']}: {len(rows)} rows", flush=True)
    return spec, manifest, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "inspection.json").exists(), "Preserve completed inspections."
    training_path = ROOT / "data/text-decisions-v1/train.jsonl"
    training_manifest = json.loads((training_path.parent / "manifest.json").read_text())
    assert sha256(training_path) == training_manifest["stats"]["train"]["file_sha256"]
    training_datasets, text_index = set(), set()
    with training_path.open() as handle:
        for line in handle:
            r = json.loads(line)
            training_datasets.add(r["source"]["dataset"])
            for text in (r["request"]["state"], r["request"]["question"], r["request"]["state"].split("\n\nAnswer candidates:\n", 1)[0]):
                if len(normalize(text)) >= 20:
                    text_index.add(normalize(text))
    # This is deliberately a preliminary exact-text check, not a claim of
    # independence from paraphrased source problems or base pretraining.
    findings = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(fetch, SPECS))
    for spec, manifest, rows in fetched:
        assert spec["repo"] not in training_datasets
        domains = {}
        for column in spec["domain"]:
            counter = Counter()
            for r in rows:
                value = r[column]
                if isinstance(value, list):
                    counter.update(value)
                else:
                    counter[value] += 1
            domains[column] = {"unique": len(counter), "counts": dict(sorted(counter.items()))}
        difficulty = Counter(str(r[spec["difficulty"]]) for r in rows)
        missing = sum(not r[spec["text"]] or not r[spec["difficulty"]] or any(not r[c] for c in spec["domain"]) for r in rows)
        seen = Counter(normalize(r[spec["text"]]) for r in rows)
        overlap = [i for i, r in enumerate(rows) if normalize(r[spec["text"]]) in text_index]
        examples = {}
        for i, row in enumerate(rows):
            label = str(row[spec["difficulty"]])
            if label not in examples:
                examples[label] = {"snapshot_row_index": i, "problem": row[spec["text"]],
                                   **{c: row[c] for c in spec["domain"]}, spec["difficulty"]: row[spec["difficulty"]]}
        findings.append({"name": spec["name"], "source": manifest, "rows": len(rows),
                         "columns": list(rows[0]), "domain_labels": domains,
                         "difficulty_counts": dict(sorted(difficulty.items())), "missing_required_labels_or_text": missing,
                         "duplicate_normalized_problem_rows": sum(n-1 for n in seen.values()),
                         "repository_used_in_finetuning": False, "preliminary_exact_training_text_matches": len(overlap),
                         "matching_snapshot_row_indices": overlap, "examples_by_difficulty": examples,
                         "text_only_diagram_marker_rows": sum("[asy]" in r[spec["text"]] for r in rows)})
        print(json.dumps({"name": spec["name"], "rows": len(rows), "difficulty": dict(difficulty),
                          "domain_counts": {c: domains[c]["unique"] for c in domains}, "exact_training_matches": len(overlap)}, ensure_ascii=False), flush=True)
    atomic_json(OUT / "inspection.json", {
        "complete": True, "datasets": findings, "training_data_sha256": sha256(training_path),
        "inspection_code_sha256": sha256(Path(__file__)), "trained_models_modified": False,
        "scope": "Research and source snapshots only; no accuracy measurement or new training",
        "overlap_check_scope": "Preliminary normalized exact matches to training request state/question and pre-candidate stems. Does not certify absence of containment, paraphrases, shared source contests or pretraining exposure.",
        "recommended_primary": "SuperGPQA: native 13-discipline and easy/middle/hard labels across graduate subjects",
        "recommended_secondary": "MATH test: native Level 1-5 and seven math subjects, after overlap and diagram filtering",
        "additional": "Omni-MATH: multi-label math domain paths and fine numeric difficulty; AoPS ratings plus GPT-4o labels, not all purely human ratings"})
    print("INSPECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
