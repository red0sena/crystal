"""Download three new evaluation splits; exclude normalized text overlaps."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.parse

import pyarrow.parquet as pq

from qwen_decision.sources.fetch_text_sources import get, download
from qwen_decision.evaluate.evaluate_mmlu_pro import sha256, atomic_json
from qwen_decision.core.scorer import validate_request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/unseen-text-v1"
TRAIN = ROOT / "data/text-decisions-v1/train.jsonl"
SPECS = [
    {"name": "HellaSwag", "repo": "Rowan/hellaswag", "config": "default", "split": "validation", "file": "data/validation-00000-of-00001.parquet", "expected_rows": 10042},
    {"name": "WinoGrande", "repo": "allenai/winogrande", "config": "winogrande_xl", "split": "validation", "file": "winogrande_xl/validation-00000-of-00001.parquet", "expected_rows": 1267},
    {"name": "OpenBookQA", "repo": "allenai/openbookqa", "config": "main", "split": "test", "file": "main/test-00000-of-00001.parquet", "expected_rows": 500},
]

def normalized(s):
    return re.sub(r"\s+", " ", s).strip().casefold()

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def download_one(spec):
    repo = spec["repo"]
    directory = OUT / "raw" / spec["name"]
    cached = directory / "manifest.json"
    if cached.exists():
        manifest = json.loads(cached.read_text())
        assert all(manifest[k] == spec[k] for k in spec)
        for entry in manifest["files"]:
            assert sha256(ROOT / entry["path"]) == entry["sha256"]
        rows = pq.read_table(directory / spec["file"]).to_pylist()
        assert len(rows) == spec["expected_rows"]
        print(f"Verified cached {spec['name']}: {len(rows)} rows", flush=True)
        return spec, rows, manifest
    info = json.loads(get("https://huggingface.co/api/datasets/" + repo))
    revision = info["sha"]
    siblings = {x["rfilename"] for x in info["siblings"]}
    assert spec["file"] in siblings
    splits = json.loads(get("https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": repo})))
    assert any(x["config"] == spec["config"] and x["split"] == spec["split"] for x in splits["splits"])
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name in [spec["file"], *(["README.md"] if "README.md" in siblings else [])]:
        files.append(download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}", directory / name))
    rows = pq.read_table(directory / spec["file"]).to_pylist()
    assert len(rows) == spec["expected_rows"]
    manifest = {**spec, "revision": revision, "license": info.get("cardData", {}).get("license"), "files": files, "rows": len(rows), "viewer_splits": splits}
    atomic_json(directory / "manifest.json", manifest)
    print(f"Downloaded {spec['name']}: {len(rows)} rows", flush=True)
    return spec, rows, manifest

def convert(spec, row, index):
    name = spec["name"]
    if name == "HellaSwag":
        state = row["ctx"]
        question = "Which continuation most plausibly follows the context?"
        descriptions = row["endings"]
        gold = int(row["label"])
        # `ind` is local to the original source collection and repeats across
        # ActivityNet/WikiHow. Snapshot row indices uniquely identify examples.
        source_id = str(index)
        texts = [row["ctx"], row["ctx_a"] + " " + row["ctx_b"], row["ctx_a"]]
        # Preserve source strings, including bracket markers; no answer-aware
        # preprocessing or activity_label hints enter the model input.
    elif name == "WinoGrande":
        state = row["sentence"]
        question = "Which option correctly fills the blank marked _ in the sentence?"
        descriptions = [row["option1"], row["option2"]]
        gold = int(row["answer"]) - 1
        source_id = str(index)
        assert state.count("_") == 1
        texts = [state, *[state.replace("_", option) for option in descriptions]]
    elif name == "OpenBookQA":
        state = ""
        question = row["question_stem"]
        descriptions = row["choices"]["text"]
        gold = row["choices"]["label"].index(row["answerKey"])
        source_id = str(row["id"])
        texts = [question]
    else:
        raise AssertionError(name)
    assert all(isinstance(x, str) and x.strip() for x in descriptions)
    assert 0 <= gold < len(descriptions)
    request = {"state": state, "question": question, "options": {f"option_{i}": x for i, x in enumerate(descriptions)}}
    validate_request(request, 1.)
    record = {"id": f"{name}:{source_id}", "source": {"dataset": spec["repo"], "config": spec["config"], "split": spec["split"], "row_id": source_id},
              "request": request, "gold_index": gold, "answer_id": f"option_{gold}", "option_count": len(descriptions)}
    if name == "HellaSwag":
        record["source"].update(original_ind=row["ind"], original_source_id=row["source_id"])
    return record, [normalized(s) for s in texts if len(normalized(s)) >= 20]

def main():
    assert not (OUT / "manifest.json").exists(), "Preserve existing data snapshots."
    OUT.mkdir(parents=True, exist_ok=True)
    training = [json.loads(line) for line in TRAIN.open()]
    # Protect against a changed training set or using an unaudited mixture.
    data_manifest = json.loads((TRAIN.parent / "manifest.json").read_text())
    assert sha256(TRAIN) == data_manifest["stats"]["train"]["file_sha256"]
    assert json.loads((TRAIN.parent / "audit.json").read_text())["verified"]
    training_datasets = sorted({r["source"]["dataset"] for r in training})
    source_families = sorted({str(x).casefold() for r in training for x in r.get("annotation", {}).get("source_family", [])})
    direct_text_index = defaultdict(list)
    base_hash_index = defaultdict(list)
    for r in training:
        base_hash_index[r["base_content_sha256"]].append(r["id"])
        req = r["request"]
        for text in (req["state"], req["question"], req["state"] + "\n" + req["question"]):
            if len(normalized(text)) >= 20:
                direct_text_index[normalized(text)].append(r["id"])
    # Search prompt-contained text too, including NI input wrapped in a task
    # definition. This checks verbatim reuse despite different task framing.
    contained_pool = [normalized(r["request"]["state"] + "\n" + r["request"]["question"]) for r in training]
    contained_pool = "\x00".join(contained_pool)
    datasets = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(download_one, SPECS))
    for spec, rows, source in downloaded:
        assert spec["repo"] not in training_datasets
        assert spec["name"].casefold() not in source_families
        records, excluded = [], []
        seen = set()
        for index, row in enumerate(rows):
            record, texts = convert(spec, row, index)
            matched = {i for text in texts for i in [*direct_text_index.get(text, []), *base_hash_index.get(digest(text), [])]}
            # To avoid accidental matches on short generic contexts, containment
            # requires at least 40 characters; exact hashes have no such cutoff.
            contained = [text for text in texts if len(text) >= 40 and text in contained_pool]
            if matched or contained:
                excluded.append({"id": record["id"], "reason": "training_text_overlap", "training_ids": sorted(matched), "contained_texts": contained})
                continue
            signature = digest(json.dumps(record["request"], sort_keys=True, ensure_ascii=False))
            if signature in seen:
                excluded.append({"id": record["id"], "reason": "within_eval_duplicate"})
                continue
            seen.add(signature)
            records.append(record)
        assert len({r["id"] for r in records}) == len(records)
        path = OUT / f"{spec['name']}.jsonl"
        with path.open("w") as handle:
            for r in records:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        atomic_json(OUT / f"{spec['name']}-excluded.json", excluded)
        datasets.append({"name": spec["name"], "source": source, "source_total": len(rows), "evaluated_total": len(records), "excluded": len(excluded),
                         "excluded_reasons": dict(Counter(r["reason"] for r in excluded)), "file": str(path), "sha256": sha256(path)})
        print(f"Prepared {spec['name']}: kept={len(records)} excluded={len(excluded)}", flush=True)
    manifest = {"datasets": datasets, "training_data_sha256": sha256(TRAIN), "training_datasets": training_datasets,
                "training_NI_source_families": source_families, "preparation_code_sha256": sha256(Path(__file__)),
                "overlap_protocol": "normalized exact base/component hashes and >=40-character verbatim containment against all fine-tuning training prompts; within-evaluation exact duplicate removal",
                "scope": "Not used in our fine-tuning or checkpoint selection. Base-model pretraining exposure and semantic near-duplicates are unknown.",
                "evaluation_protocol": "full public labeled split minus stated exclusions; one no-generation option-logit argmax pass per model; not completion log-likelihood evaluation"}
    atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "audit.json", {"verified": True, "no_direct_training_dataset": True, "training_examples_checked": len(training), "retained": sum(x["evaluated_total"] for x in datasets), "excluded": sum(x["excluded"] for x in datasets), "scope": manifest["scope"]})

if __name__ == "__main__":
    main()
