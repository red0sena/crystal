"""Inspect all KMMLU test subjects and availability of empirical difficulty."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import urllib.parse

from fetch_text_sources import get, download
from evaluate_mmlu_pro import atomic_json, sha256
from inspect_external_classification_data import normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external-classification-candidates-v1"
REPO = "HAERAE-HUB/KMMLU"


def main():
    assert not (OUT / "kmmlu-inspection.json").exists(), "Preserve completed inspection."
    directory = OUT / "raw/KMMLU"
    directory.mkdir(parents=True, exist_ok=True)
    info = json.loads(get("https://huggingface.co/api/datasets/" + REPO))
    revision = info["sha"]
    paths = sorted(x["rfilename"] for x in info["siblings"] if x["rfilename"].startswith("data/") and x["rfilename"].endswith("-test.csv"))
    assert len(paths) == 45
    paths.append("README.md")

    def one(path):
        return download(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}", directory / path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(one, paths))
    splits = json.loads(get("https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode({"dataset": REPO})))
    rows, subjects = [], {}
    for path in paths[:-1]:
        with (directory / path).open(encoding="utf-8-sig", newline="") as handle:
            subset = list(csv.DictReader(handle))
        subjects[path] = len(subset)
        rows.extend(subset)
    assert len(rows) == 35030
    categories = Counter(r["Category"] for r in rows)
    assert len(categories) == 45
    accuracy = []
    missing_accuracy = Counter()
    examples = []
    for r in rows:
        value = r["Human Accuracy"]
        try:
            number = float(value)
        except ValueError:
            number = float("nan")
        if math.isfinite(number) and 0 <= number <= 1:
            accuracy.append(number)
            if len(examples) < 3:
                examples.append({"question": r["question"], "Category": r["Category"], "Human Accuracy": number})
        else:
            missing_accuracy[r["Category"]] += 1
    train = ROOT / "data/text-decisions-v1/train.jsonl"
    training_index, training_datasets = set(), set()
    with train.open() as handle:
        for line in handle:
            r = json.loads(line)
            training_datasets.add(r["source"]["dataset"])
            for text in (r["request"]["state"], r["request"]["question"], r["request"]["state"].split("\n\nAnswer candidates:\n", 1)[0]):
                if len(normalize(text)) >= 20:
                    training_index.add(normalize(text))
    assert REPO not in training_datasets
    overlap = sum(normalize(r["question"]) in training_index for r in rows)
    text_counts = Counter(normalize(r["question"]) for r in rows)
    manifest = {"repo": REPO, "revision": revision, "files": files, "split": "test", "viewer_splits": splits,
                "license": info.get("cardData", {}).get("license")}
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(OUT / "kmmlu-inspection.json", {
        "complete": True, "source": manifest, "rows": len(rows), "columns": list(rows[0]),
        "subject_counts": dict(sorted(categories.items())), "subjects": len(categories),
        "human_accuracy_valid_rows": len(accuracy), "human_accuracy_missing_rows": len(rows) - len(accuracy),
        "human_accuracy_missing_by_subject": dict(missing_accuracy),
        "human_accuracy_range": [min(accuracy), max(accuracy)],
        "difficulty_label_status": "Published continuous Human Accuracy field; no native easy/medium/hard labels. Need fixed documented thresholds or a continuous/ranking evaluation.",
        "repository_used_in_finetuning": False, "preliminary_exact_training_text_matches": overlap,
        "duplicate_normalized_question_rows": sum(v-1 for v in text_counts.values()),
        "training_data_sha256": sha256(train), "inspection_code_sha256": sha256(Path(__file__)),
        "examples": examples, "trained_models_modified": False,
        "overlap_check_scope": "Normalized exact state/question and pre-candidate stems only; no semantic or translation overlap audit and no pretraining exposure claim."})
    print(json.dumps({"rows": len(rows), "subjects": len(categories), "valid_human_accuracy": len(accuracy),
                      "missing_human_accuracy": len(rows) - len(accuracy), "exact_training_matches": overlap,
                      "duplicate_question_rows": sum(v-1 for v in text_counts.values())}), flush=True)


if __name__ == "__main__":
    main()
