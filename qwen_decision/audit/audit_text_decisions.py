"""Independently reconcile prepared targets against downloaded source rows."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pyarrow.parquet as pq
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/text-decisions-v1"
RAW = OUT / "raw"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_lines(path):
    # JSONL uses LF; Unicode line separators may occur inside JSON strings.
    with path.open() as handle:
        return [json.loads(x) for x in handle if x.strip()]

def canonical_hash(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().casefold().encode()).hexdigest()

def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        assert sha(code_path(name)) == expected
    source_files = 0
    for entry in manifest["sources"]:
        key = "manifest" if "manifest" in entry else "local_file"
        path = ROOT / entry[key]
        assert sha(path) == entry["sha256"]
        if key == "manifest":
            for f in json.loads(path.read_text())["files"]:
                assert sha(ROOT / f["path"]) == f["sha256"]
                source_files += 1
        else:
            source_files += 1
    cache = {}
    def parquet(repo, file):
        key = (repo, file)
        if key not in cache:
            path = ROOT / file if repo == "local" else RAW / repo.replace("/", "__") / file
            cache[key] = pq.read_table(path).to_pylist()
        return cache[key]
    def id_map(repo, file, field):
        key = (repo, file, field)
        if key not in cache:
            rr = load_lines(RAW / repo.replace("/", "__") / file) if file.endswith(".jsonl") else parquet(repo, file)
            cache[key] = {str(r[field]): r for r in rr}
        return cache[key]
    with zipfile.ZipFile(ROOT / "data/choice-benchmarks/ybisk__piqa/physicaliqa-train-dev.zip") as z:
        piqa = {}
        for split, stem in (("train", "train"), ("validation", "dev")):
            member = next(n for n in z.namelist() if n.endswith(f"/{stem}.jsonl") or n == f"{stem}.jsonl")
            labels = next(n for n in z.namelist() if n.endswith(f"/{stem}-labels.lst") or n == f"{stem}-labels.lst")
            piqa[split] = ([json.loads(x) for x in z.read(member).decode().splitlines()], [int(x) for x in z.read(labels).decode().splitlines()])
    banking = {}
    for split in ("train", "test"):
        with (RAW / "banking77-original" / f"{split}.csv").open() as f:
            banking[split] = list(csv.DictReader(f))
    rubric = json.loads((OUT / "difficulty_rubric.json").read_text())
    split_groups, split_texts = {}, {}
    all_ids = set()
    summaries = {}
    training_ratings = []
    for split in ("train", "dev", "test"):
        path = OUT / f"{split}.jsonl"
        assert sha(path) == manifest["stats"][split]["file_sha256"]
        records = load_lines(path)
        split_groups[split], split_texts[split] = set(), set()
        by_task = Counter()
        quality = Counter()
        label_counts = defaultdict(Counter)
        for r in records:
            assert r["id"] not in all_ids
            all_ids.add(r["id"])
            assert r["split"] == split
            split_groups[split].add(r["group_id"])
            split_texts[split].add(r["base_content_sha256"])
            req = r["request"]
            assert set(req) == {"state", "question", "options"}
            assert isinstance(req["state"], str) and req["question"].strip()
            names = list(req["options"])
            assert 2 <= len(names) <= 26
            assert names[r["gold_index"]] == r["answer_id"]
            assert all(isinstance(x, str) and x.strip() for x in req["options"].values())
            src = r["source"]
            repo, config, native_split, row_id = (src[k] for k in ("dataset", "config", "split", "row_id"))
            actual = None
            if repo in ("allenai/ai2_arc", "tau/commonsense_qa"):
                prefix = config if repo == "allenai/ai2_arc" else "data"
                original = id_map(repo, f"{prefix}/{native_split}-00000-of-00001.parquet", "id")[row_id]
                idx = original["choices"]["label"].index(original["answerKey"])
                actual = f"option_{idx}"
                assert req["state"] == "" and req["question"] == original["question"]
                for i, text in enumerate(original["choices"]["text"]):
                    assert req["options"][f"option_{i}"] == text
            elif repo == "ybisk/piqa":
                original = piqa[native_split][0][int(row_id)]
                actual = f"option_{piqa[native_split][1][int(row_id)]}"
                assert req["question"] == original["goal"] and req["state"] == ""
                assert req["options"]["option_0"] == original["sol1"] and req["options"]["option_1"] == original["sol2"]
            elif repo == "TIGER-Lab/MMLU-Pro":
                original = id_map("local", "data/mmlu-pro/data/test-00000-of-00001.parquet", "question_id")[row_id]
                if r["task"] == "academic_domain":
                    actual = original["category"]
                    assert req["state"] == original["question"] + "\n\nAnswer candidates:\n" + "\n".join(x for x in original["options"] if x != "N/A")
                else:
                    retained = [i for i, text in enumerate(original["options"]) if text != "N/A"]
                    actual = f"option_{retained.index(original['answer_index'])}"
                    assert req["question"] == original["question"] and req["state"] == ""
                    for i, original_index in enumerate(retained):
                        assert req["options"][f"option_{i}"] == original["options"][original_index]
            elif repo == "ehovy/race":
                original = parquet(repo, f"all/{native_split}-00000-of-00001.parquet")[int(row_id)]
                actual = f"option_{ord(original['answer']) - ord('A')}"
                assert req["state"] == original["article"] and req["question"] == original["question"]
                for i, text in enumerate(original["options"]):
                    assert req["options"][f"option_{i}"] == text
            elif repo in ("fancyzhx/ag_news", "stanfordnlp/sst2", "dair-ai/emotion", "nyu-mll/glue"):
                prefix = config if repo in ("nyu-mll/glue", "dair-ai/emotion") else "data"
                file = f"{prefix}/{native_split}-00000-of-00001.parquet"
                rr = parquet(repo, file)
                if repo in ("stanfordnlp/sst2", "nyu-mll/glue"):
                    original = id_map(repo, file, "idx")[row_id]
                else:
                    original = rr[int(row_id)]
                p = RAW / repo.replace("/", "__") / file
                label_key = (repo, file, "label_names")
                if label_key not in cache:
                    cache[label_key] = json.loads(pq.read_schema(p).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
                labels = cache[label_key]
                actual = labels[original["label"]]
                if config == "rte":
                    state = "Premise:\n" + original["sentence1"] + "\n\nHypothesis:\n" + original["sentence2"]
                elif config == "mrpc":
                    state = "Sentence 1:\n" + original["sentence1"] + "\n\nSentence 2:\n" + original["sentence2"]
                else:
                    state = original.get("text", original.get("sentence"))
                assert req["state"] == state and set(names) == set(labels)
            elif repo == "PolyAI/banking77":
                original = banking[native_split][int(row_id)]
                actual = original["category"]
                assert req["state"] == original["text"] and len(names) == 16
                assert not r["annotation"]["benchmark_equivalence"]
            elif repo == "mouryat9/CogBench":
                original = id_map(repo, f"data/{native_split}.jsonl", "question_id")[row_id]
                actual = original["bloom_name"]
                assert req["state"] == original["question_text"]
                assert r["annotation"]["label_source"] == original["label_source"]
                assert r["sample_weight"] == (1. if original["label_source"] == "original_author" else .5)
                if original["label_source"] != "original_author":
                    assert original["confidence"] >= .8
            elif repo == "furonghuang-lab/Easy2Hard-Bench":
                original = parquet(repo, f"E2H-AMC/{native_split}-00000-of-00001.parquet")[int(row_id)]
                rating = original["rating"]
                idx = int(rating > rubric["cutoffs"][0]) + int(rating > rubric["cutoffs"][1])
                actual = ["low", "medium", "high"][idx]
                assert req["state"] == original["problem"] and r["annotation"]["original_rating"] == rating
                if split == "train":
                    training_ratings.append(rating)
            elif repo == "allenai/natural-instructions":
                path = RAW / "natural-instructions/tasks" / f"{config}.json"
                if str(path) not in cache:
                    cache[str(path)] = json.loads(path.read_text())
                    cache[(str(path), "instances")] = {x["id"]: x for x in cache[str(path)]["Instances"]}
                task = cache[str(path)]
                original = cache[(str(path), "instances")][row_id]
                actual = task["Categories"][0]
                assert req["state"] == "Requested operation:\n" + "\n".join(task["Definition"]) + "\n\nInput to that operation:\n" + original["input"]
            else:
                raise AssertionError(repo)
            assert actual == r["answer_id"], (r["id"], actual, r["answer_id"])
            by_task[r["task"]] += 1
            quality[r["label_provenance"]] += 1
            label_counts[r["task"]][r["answer_id"]] += 1
        assert dict(by_task) == manifest["stats"][split]["by_task"]
        summaries[split] = {"total": len(records), "by_task": dict(sorted(by_task.items())), "label_provenance": dict(quality), "label_counts": {k: dict(v) for k, v in label_counts.items()}}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert split_groups[a].isdisjoint(split_groups[b])
        assert split_texts[a].isdisjoint(split_texts[b])
    # Verify that the saved difficulty cutoffs really use retained training rows.
    np.testing.assert_allclose(np.quantile(training_ratings, [1/3, 2/3]), rubric["cutoffs"], atol=1e-12, rtol=0)
    # Historical held-out questions from ARC/CSQA/PIQA remain outside training,
    # even when used as a different task or from a second source.
    for file in sorted((ROOT / "results/choice-benchmarks-probabilities-1run").glob("*/questions.json")):
        for old in json.loads(file.read_text()):
            assert canonical_hash(old["request"]["question"]) not in split_texts["train"]
    result = {"verified": True, "checked_records": len(all_ids), "source_files_verified": source_files, "cross_split_group_overlap": 0, "cross_split_base_text_overlap": 0, "targets_reconciled_against_raw_sources": True, "historical_arc_csqa_piqa_heldout_protected": True, "difficulty_thresholds_recomputed_from_retained_train": True, "stats": summaries}
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "stats"}, indent=2))

if __name__ == "__main__":
    main()
