"""Verify saved probabilities, argmax choices and accuracy against the source test split."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import string
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/mmlu-pro-0shot-5runs"

def main():
    combined = json.loads((RESULTS / "summary.json").read_text())
    manifest = combined["manifest"]
    assert combined["complete"] and manifest["limit"] is None
    assert len(combined["runs"]) == manifest["repeats"] == 5
    rows = pq.read_table(ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet",
                         columns=["question_id", "options", "answer_index", "category"]).to_pylist()
    expected = {}
    for row in rows:
        retained = [i for i, option in enumerate(row["options"]) if option != "N/A"]
        expected[row["question_id"]] = (retained.index(row["answer_index"]), len(retained), row["category"])
    assert len(expected) == manifest["question_count"]
    scores, audits, first, disagreements, category_rows = [], [], None, [], []
    for summary in combined["runs"]:
        run = summary["run"]
        path = RESULTS / f"run-{run}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == len({r["question_id"] for r in records}) == len(expected)
        assert {r["question_id"] for r in records} == set(expected)
        correct, max_error, predictions = 0, 0.0, {}
        categories = defaultdict(lambda: {"correct": 0, "total": 0})
        for record in records:
            gold, count, category = expected[record["question_id"]]
            assert record["gold"] == string.ascii_uppercase[gold] and record["category"] == category
            assert record["output_tokens"] == 0
            probs, logits = record["probabilities"], record["logits"]
            assert len(probs) == len(logits) == count
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
            assert all(math.isfinite(z) for z in logits) and abs(sum(probs) - 1) < 1e-6
            exps = [math.exp(z - max(logits)) for z in logits]
            error = max(abs(p - e / sum(exps)) for p, e in zip(probs, exps))
            assert error < 2e-7
            max_error = max(max_error, error)
            prediction = max(range(count), key=lambda i: probs[i])
            assert record["prediction"] == string.ascii_uppercase[prediction]
            assert record["correct"] == (prediction == gold)
            correct += int(prediction == gold)
            categories[category]["correct"] += int(prediction == gold)
            categories[category]["total"] += 1
            predictions[record["question_id"]] = prediction
        accuracy = 100 * correct / len(expected)
        assert correct == summary["correct"] and summary["total"] == len(expected)
        assert abs(accuracy - summary["accuracy_pct"]) < 1e-12
        assert summary["forward_calls"] == manifest["batch_count"]
        assert hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest() == summary["prediction_sha256"]
        for category, stats in sorted(categories.items()):
            reported = summary["per_category"][category]
            assert stats["correct"] == reported["correct"] and stats["total"] == reported["total"]
            category_rows.append([run, category, stats["correct"], stats["total"], 100 * stats["correct"] / stats["total"]])
        scores.append(accuracy)
        if first is None:
            first = predictions
        disagreements.append(sum(predictions[k] != first[k] for k in predictions))
        audits.append({"run": run, "correct": correct, "total": len(expected), "accuracy_pct": accuracy,
                       "max_probability_error": max_error, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert disagreements == combined["prediction_disagreements_vs_run_1"]
    assert abs(statistics.mean(scores) - combined["mean_accuracy_pct"]) < 1e-12
    assert abs(statistics.pstdev(scores) - combined["population_stddev_percentage_points"]) < 1e-12
    with (RESULTS / "accuracy-by-category.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "category", "correct", "total", "accuracy_pct"])
        writer.writerows(category_rows)
    audit = {"verified": True, "source_questions": len(expected), "runs": audits,
             "mean_accuracy_pct": statistics.mean(scores),
             "population_stddev_percentage_points": statistics.pstdev(scores),
             "prediction_disagreements_vs_run_1": disagreements}
    (RESULTS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
