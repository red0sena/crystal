"""Audit every saved classification score, label and matched evaluation input."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from evaluate_mmlu_pro import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/full-classification-v1"
DATA = ROOT / "data/text-decisions-v1"


def read_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    selection = json.loads((OUT / "manifest.json").read_text())
    assert summary["complete"] and summary["manifest_sha256"] == sha256(OUT / "manifest.json")
    assert sha256(DATA / "test.jsonl") == selection["heldout_data_sha256"]
    assert sha256(DATA / "train.jsonl") == selection["training_data_sha256"]
    all_test = read_lines(DATA / "test.jsonl")
    mmlu = {str(r["question_id"]): r for r in pq.read_table(ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet").to_pylist()}
    amc = pq.read_table(DATA / "raw/furonghuang-lab__Easy2Hard-Bench/E2H-AMC/eval-00000-of-00001.parquet").to_pylist()
    rows, report, total = [], [], 0
    for task, labels in selection["labels"].items():
        records = json.loads((OUT / "data" / task / "selected-records.json").read_text())
        assert records == [r for r in all_test if r["task"] == task]
        by_id = {r["id"]: r for r in records}
        assert len(by_id) == len(records)
        for r in records:
            if task == "academic_domain":
                source = mmlu[r["source"]["row_id"]]
                expected = source["category"]
                assert r["request"]["state"].startswith(source["question"] + "\n\nAnswer candidates:\n")
            else:
                source = amc[int(r["source"]["row_id"])]
                assert r["request"]["state"] == source["problem"]
                a, b = selection["difficulty_cutoffs"]
                expected = "low" if source["rating"] < a else "medium" if source["rating"] < b else "high"
                assert source["rating"] == r["annotation"]["original_rating"]
            assert expected == r["answer_id"]
        for model_name in ("original", "trained"):
            directory = OUT / model_name / task
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["selection_manifest_sha256"] == sha256(OUT / "manifest.json")
            assert manifest["questions_sha256"] == sha256(directory / "questions.json")
            assert manifest["selected_records_sha256"] == sha256(OUT / "data" / task / "selected-records.json")
            for name, digest in manifest["evaluation_code_sha256"].items():
                assert sha256(ROOT / name) == digest
            for name, digest in manifest["model_weights_sha256"].items():
                assert sha256(Path(manifest["model"]) / name) == digest
            predictions = read_lines(directory / "run-1.jsonl")
            assert len(predictions) == len(records)
            assert {p["source_id"] for p in predictions} == set(by_id)
            matrix = [[0] * len(labels) for _ in labels]
            for p in predictions:
                r = by_id[p["source_id"]]
                assert p["option_labels"] == list(r["request"]["options"])
                logits = p["logits"]
                assert len(logits) == len(labels) and all(math.isfinite(z) for z in logits)
                scores = [math.exp(z - max(logits)) for z in logits]
                probabilities = [v / sum(scores) for v in scores]
                assert all(abs(a-b) < 2e-6 for a, b in zip(probabilities, p["probabilities"]))
                predicted_index = max(range(len(logits)), key=lambda i: logits[i])
                assert p["prediction"] == chr(ord("A") + predicted_index)
                assert p["gold"] == chr(ord("A") + r["gold_index"])
                predicted_label = p["option_labels"][predicted_index]
                assert p["correct"] == (predicted_label == r["answer_id"])
                assert p["output_tokens"] == 0
                matrix[labels.index(r["answer_id"])][labels.index(predicted_label)] += 1
            metrics = summary["results"][task][model_name]
            assert metrics["confusion_matrix_true_rows_predicted_columns"] == matrix
            correct = sum(matrix[i][i] for i in range(len(labels)))
            assert metrics["correct"] == correct
            assert abs(metrics["accuracy_pct"] - 100 * correct / len(records)) < 1e-9
            assert metrics["inference"]["generation_calls"] == 0
            assert metrics["inference"]["forward_calls"] == len(manifest["batch_layout"])
            assert len(manifest["batch_layout"]) == len({tuple(b) for b in manifest["batch_layout"]})
            assert sorted(i for b in manifest["batch_layout"] for i in b) == list(range(len(records)))
            f1_sum = 0
            for i, label in enumerate(labels):
                support, predicted, tp = sum(matrix[i]), sum(row[i] for row in matrix), matrix[i][i]
                assert metrics["per_class"][label]["support"] == support
                assert metrics["per_class"][label]["correct"] == tp
                recall = 100 * tp / support
                precision = 100 * tp / predicted if predicted else 0
                f1 = 200 * tp / (support + predicted) if support + predicted else 0
                assert abs(metrics["per_class"][label]["recall_pct"] - recall) < 1e-9
                assert abs(metrics["per_class"][label]["f1_pct"] - f1) < 1e-9
                f1_sum += f1
                rows.append({"task": task, "class": label, "model": model_name, "support": support,
                             "correct": tp, "predicted_count": predicted, "recall_pct": recall,
                             "precision_pct": precision, "f1_pct": f1})
            assert abs(metrics["macro_f1_pct"] - f1_sum / len(labels)) < 1e-9
        assert sha256(OUT / "original" / task / "questions.json") == sha256(OUT / "trained" / task / "questions.json")
        original_manifest = json.loads((OUT / "original" / task / "manifest.json").read_text())
        trained_manifest = json.loads((OUT / "trained" / task / "manifest.json").read_text())
        assert original_manifest["batch_layout"] == trained_manifest["batch_layout"]
        assert original_manifest["option_token_ids"] == trained_manifest["option_token_ids"]
        report.append({"task": task, "source_labels_checked": len(records), "predictions_checked": 2 * len(records), "matched_prompts_and_batches": True})
        total += len(records)
    with (OUT / "per-class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert summary["total_questions_per_model"] == total and summary["total_predictions"] == 2 * total
    atomic_json(OUT / "audit.json", {"verified": True, "source_labels_checked": total, "predictions_checked": 2 * total,
                "results": report, "audit_code_sha256": sha256(Path(__file__)),
                "summary_sha256": sha256(OUT / "summary.json"), "per_class_csv_sha256": sha256(OUT / "per-class.csv")})
    print(json.dumps({"verified": True, "source_labels_checked": total, "predictions_checked": 2 * total}))


if __name__ == "__main__":
    main()
