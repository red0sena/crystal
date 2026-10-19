"""Independently audit all source labels, choice probabilities, and metrics."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import string

from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, sha256
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/external-classification-eval-v1"
OUT = ROOT / "results/external-classification-v1"
SNAPSHOTS = ROOT / "data/external-classification-candidates-v1"


def lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main():
    data = json.loads((DATA / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert data["complete"] and summary["complete"]
    assert summary["data_manifest_sha256"] == sha256(DATA / "manifest.json")
    assert sha256(ROOT / "data/text-decisions-v1/train.jsonl") == data["training_data_sha256"]
    assert data["kmmlu_difficulty_thresholds"] == {"easy_min_human_accuracy": .8, "middle_min_human_accuracy": .5}
    assert sha256(code_path("prepare_external_classification_evaluation.py")) == data["preparation_code_sha256"]
    assert sha256(DATA / "exclusions.json") == data["exclusions_sha256"]
    super_raw = lines(SNAPSHOTS / "raw/SuperGPQA/SuperGPQA-all.jsonl")
    kmmlu_cache = {}
    rows, details, total = [], [], 0
    for spec in data["datasets"]:
        task = spec["task"]
        for entry in spec["source"]["files"]:
            assert sha256(ROOT / entry["path"]) == entry["sha256"]
        records = lines(Path(spec["file"]))
        assert sha256(Path(spec["file"])) == spec["sha256"]
        assert len(records) == spec["retained_total"]
        assert dict(Counter(r["answer_id"] for r in records)) == spec["class_counts"]
        by_id = {r["id"]: r for r in records}
        assert len(by_id) == len(records)
        for r in records:
            if spec["dataset"] == "SuperGPQA":
                original = super_raw[r["source"]["row_index"]]
                assert r["source"]["uuid"] == original["uuid"]
                expected = original["discipline"] if spec["kind"] == "domain" else original["difficulty"]
                state = original["question"] + "\n\nAnswer candidates:\n" + "\n".join(original["options"])
            else:
                name = r["source"]["snapshot_file"]
                if name not in kmmlu_cache:
                    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
                        kmmlu_cache[name] = list(csv.DictReader(handle))
                original = kmmlu_cache[name][r["source"]["snapshot_row"]]
                if spec["kind"] == "domain":
                    expected = original["Category"]
                else:
                    rating = float(original["Human Accuracy"])
                    assert math.isfinite(rating) and 0 <= rating <= 1
                    expected = "easy" if rating >= .8 else "middle" if rating >= .5 else "hard"
                state = original["question"] + "\n\nAnswer candidates:\n" + "\n".join(original[c] for c in "ABCD")
            assert r["request"]["state"] == state
            assert r["answer_id"] == expected
            assert list(r["request"]["options"])[r["gold_index"]] == expected
            assert set(r["request"]["options"]) == set(spec["labels"])
        for model in ("original", "trained"):
            directory = OUT / model / task
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["data_manifest_sha256"] == sha256(DATA / "manifest.json")
            assert manifest["dataset_file_sha256"] == spec["sha256"]
            assert manifest["codes"] == (string.ascii_uppercase + string.ascii_lowercase)[:len(spec["labels"])]
            assert len(set(manifest["option_token_ids"])) == len(spec["labels"])
            for filename, digest in manifest["code_sha256"].items():
                assert sha256(code_path(filename)) == digest
            for filename, digest in manifest["model_weights_sha256"].items():
                assert sha256(Path(manifest["model"]) / filename) == digest
            pred = lines(directory / "predictions.jsonl")
            assert len(pred) == len(records) and {p["source_id"] for p in pred} == set(by_id)
            matrix = [[0] * len(spec["labels"]) for _ in spec["labels"]]
            for p in pred:
                r = by_id[p["source_id"]]
                assert p["source_id"] == records[p["question_id"]]["id"]
                assert p["option_labels"] == list(r["request"]["options"])
                logits = p["logits"]
                assert len(logits) == len(p["probabilities"]) == r["option_count"]
                assert all(math.isfinite(z) for z in logits)
                scores = [math.exp(z - max(logits)) for z in logits]
                probs = [v / sum(scores) for v in scores]
                assert all(abs(a-b) < 2e-6 for a,b in zip(probs, p["probabilities"]))
                index = max(range(len(logits)), key=lambda i: logits[i])
                assert p["prediction_code"] == manifest["codes"][index]
                assert p["prediction"] == p["option_labels"][index]
                assert p["gold"] == r["answer_id"]
                assert p["correct"] == (index == r["gold_index"])
                assert p["output_tokens"] == 0
                matrix[spec["labels"].index(p["gold"])][spec["labels"].index(p["prediction"])] += 1
            result = summary["results"][task][model]
            assert result["confusion_matrix_true_rows_predicted_columns"] == matrix
            correct = sum(matrix[i][i] for i in range(len(matrix)))
            assert result["correct"] == correct and result["total"] == len(records)
            assert abs(result["accuracy_pct"] - 100 * correct / len(records)) < 1e-9
            assert result["inference"]["generation_calls"] == result["inference"]["output_tokens"] == 0
            assert result["inference"]["forward_calls"] == len(manifest["batch_layout"])
            assert result["inference"]["predictions_sha256"] == sha256(directory / "predictions.jsonl")
            assert sorted(i for b in manifest["batch_layout"] for i in b) == list(range(len(records)))
            for i, label in enumerate(spec["labels"]):
                tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(row[i] for row in matrix)
                measured = result["per_class"][label]
                assert measured["support"] == support and measured["correct"] == tp
                assert measured["predicted_count"] == predicted
                recall = 100 * tp / support if support else 0
                precision = 100 * tp / predicted if predicted else 0
                f1 = 200 * tp / (support + predicted) if support + predicted else 0
                assert abs(measured["recall_pct"] - recall) < 1e-9
                assert abs(measured["precision_pct"] - precision) < 1e-9
                assert abs(measured["f1_pct"] - f1) < 1e-9
                rows.append({"task": task, "model": model, "class": label, "support": support, "correct": tp,
                             "recall_pct": 100 * tp / support if support else "", "predicted_count": predicted,
                             "precision_pct": 100 * tp / predicted if predicted else 0,
                             "f1_pct": 200 * tp / (support + predicted) if support + predicted else 0})
            n = len(spec["labels"])
            assert abs(result["balanced_accuracy_pct"] - sum(100 * matrix[i][i] / sum(matrix[i]) for i in range(n)) / n) < 1e-9
            independent_f1 = sum(200 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix)) if sum(matrix[i]) + sum(row[i] for row in matrix) else 0 for i in range(n)) / n
            assert abs(result["macro_f1_pct"] - independent_f1) < 1e-9
            assert abs(result["majority_class_baseline_pct"] - 100 * max(map(sum, matrix)) / len(records)) < 1e-9
        original = json.loads((OUT / "original" / task / "manifest.json").read_text())
        trained = json.loads((OUT / "trained" / task / "manifest.json").read_text())
        for field in ("encoding_sha256", "batch_layout", "codes", "option_token_ids", "dataset_file_sha256", "system_prompt"):
            assert original[field] == trained[field]
        details.append({"task": task, "labels_checked": len(records), "predictions_checked": 2 * len(records), "matched_inputs_and_batches": True})
        total += len(records)
    assert summary["total_questions_per_model"] == total == data["total_requests_per_model"]
    with (OUT / "per-class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(OUT / "audit.json", {"verified": True, "source_labels_checked": total, "predictions_checked": 2*total,
                "tasks": details, "summary_sha256": sha256(OUT / "summary.json"),
                "audit_code_sha256": sha256(Path(__file__)), "generated_tokens": 0})
    print(json.dumps({"verified": True, "source_labels_checked": total, "predictions_checked": 2*total}), flush=True)


if __name__ == "__main__":
    main()
