"""Full held-out domain and relative-difficulty classification, no generation."""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch

import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, encode_questions, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_probabilities import batch_probabilities, run_benchmark
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v1"
OUT = ROOT / "results/full-classification-v1"
LABELS = {
    "academic_domain": ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"],
    "relative_difficulty": ["low", "medium", "high"],
}


def read_lines(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_source_labels(by_task, cutoffs):
    mmlu = pq.read_table(ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet", columns=["question_id", "question", "category", "options"]).to_pylist()
    mmlu_by_id = {str(r["question_id"]): r for r in mmlu}
    amc = pq.read_table(DATA / "raw/furonghuang-lab__Easy2Hard-Bench/E2H-AMC/eval-00000-of-00001.parquet", columns=["problem", "rating"]).to_pylist()
    for task, records in by_task.items():
        assert len({r["id"] for r in records}) == len(records)
        assert {r["answer_id"] for r in records} == set(LABELS[task])
        for r in records:
            assert set(r["request"]["options"]) == set(LABELS[task])
            assert list(r["request"]["options"])[r["gold_index"]] == r["answer_id"]
            if task == "academic_domain":
                row = mmlu_by_id[r["source"]["row_id"]]
                assert r["answer_id"] == row["category"]
                assert r["request"]["state"].startswith(row["question"] + "\n\nAnswer candidates:\n")
                assert all(o in r["request"]["state"] for o in row["options"] if o != "N/A")
            else:
                row = amc[int(r["source"]["row_id"])]
                assert r["request"]["state"] == row["problem"]
                assert row["rating"] == r["annotation"]["original_rating"]
                expected = "low" if row["rating"] < cutoffs[0] else "medium" if row["rating"] < cutoffs[1] else "high"
                assert r["answer_id"] == expected


def classification_metrics(records, predictions, labels):
    by_id = {r["id"]: r for r in records}
    assert len(predictions) == len(records)
    assert {p["source_id"] for p in predictions} == set(by_id)
    matrix = [[0 for _ in labels] for _ in labels]
    position_counts = Counter()
    for p in predictions:
        record = by_id[p["source_id"]]
        assert p["option_labels"] == list(record["request"]["options"])
        predicted = p["option_labels"][ord(p["prediction"]) - ord("A")]
        gold = record["answer_id"]
        assert p["correct"] == (predicted == gold)
        matrix[labels.index(gold)][labels.index(predicted)] += 1
        position_counts[p["prediction"]] += 1
    per_class = {}
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {"support": support, "correct": tp, "predicted_count": predicted,
                            "recall_pct": 100 * recall, "precision_pct": 100 * precision, "f1_pct": 100 * f1}
    total = len(records)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    result = {"total": total, "correct": correct, "accuracy_pct": 100 * correct / total,
              "labels": labels, "confusion_matrix_true_rows_predicted_columns": matrix,
              "per_class": per_class,
              "balanced_accuracy_pct": sum(v["recall_pct"] for v in per_class.values()) / len(labels),
              "macro_f1_pct": sum(v["f1_pct"] for v in per_class.values()) / len(labels),
              "random_choice_expected_accuracy_pct": 100 / len(labels),
              "majority_class_baseline_pct": 100 * max(v["support"] for v in per_class.values()) / total,
              "prediction_position_counts": dict(position_counts),
              "first_display_position_pct": 100 * position_counts["A"] / total,
              "largest_confusions": sorted(
                  [{"gold": labels[i], "predicted": labels[j], "count": matrix[i][j]}
                   for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i][j]],
                  key=lambda x: (-x["count"], x["gold"], x["predicted"]))[:10]}
    if labels == ["low", "medium", "high"]:
        result["ordinal_errors"] = {
            "overestimated": sum(matrix[i][j] for i in range(3) for j in range(3) if j > i),
            "underestimated": sum(matrix[i][j] for i in range(3) for j in range(3) if j < i),
            "two_level_errors": matrix[0][2] + matrix[2][0],
            "mean_absolute_band_error": sum(abs(i-j) * matrix[i][j] for i in range(3) for j in range(3)) / total,
        }
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not list(OUT.glob("*/*/run-1.jsonl")), "Preserve completed evaluations."
    dataset_manifest = json.loads((DATA / "manifest.json").read_text())
    assert json.loads((DATA / "audit.json").read_text())["verified"]
    for split in ("train", "test"):
        assert sha256(DATA / f"{split}.jsonl") == dataset_manifest["stats"][split]["file_sha256"]
    assert json.loads((ROOT / "training/text-decision-full-v1/pipeline.json").read_text())["status"] == "complete"
    training_groups, training_content = set(), set()
    with (DATA / "train.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            training_groups.add(r["group_id"])
            training_content.add(r["base_content_sha256"])
    by_task = {task: [] for task in LABELS}
    with (DATA / "test.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            if r["task"] in by_task:
                assert r["group_id"] not in training_groups
                assert r["base_content_sha256"] not in training_content
                by_task[r["task"]].append(r)
    assert [len(by_task[t]) for t in LABELS] == [1181, 2974]
    cutoffs = json.loads((DATA / "difficulty_rubric.json").read_text())["cutoffs"]
    verify_source_labels(by_task, cutoffs)
    for task, records in by_task.items():
        directory = OUT / "data" / task
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "selected-records.json", records)
    atomic_json(OUT / "manifest.json", {
        "selection": "All existing held-out test records for academic_domain and relative_difficulty; no sample selection",
        "totals": {task: len(records) for task, records in by_task.items()},
        "labels": LABELS, "difficulty_cutoffs": cutoffs,
        "training_data_sha256": sha256(DATA / "train.jsonl"), "heldout_data_sha256": sha256(DATA / "test.jsonl"),
        "source_labels_verified": True, "train_group_and_content_overlap": 0,
        "domain_scope": "Full custom reserved MMLU-Pro domain-classification test partition; source academic categories, not question answering",
        "difficulty_scope": "E2H-AMC eval split minus previous duplicate/group filtering; continuous source IRT ratings binned with unchanged training-only tertiles",
        "model_selection": "Previously completed training and development-based checkpoint averaging; no evaluation-dependent training or selection",
        "protocol": "One measured no-generation pass per model; warmup excluded; selected option-code logits softmax argmax; no CoT"})
    scorer_module.SYSTEM_PROMPT = dataset_manifest["system_prompt"]
    start = time.time()
    results = {}
    for model_name, path in (("original", ROOT / "model"), ("trained", ROOT / "training/text-decision-full-v1/model")):
        scorer = ChoiceScorer(str(path), device="cuda")
        hashes = {p.name: sha256(p) for p in sorted(path.glob("*.safetensors"))}
        for task, records in by_task.items():
            questions = [{"question_id": i, "source_id": r["id"], "category": task,
                          "request": r["request"], "gold_index": r["gold_index"],
                          "option_count": len(r["request"]["options"])} for i, r in enumerate(records)]
            ids = encode_questions(scorer, questions)
            scorer.processor.tokenizer.padding_side = "left"
            batches = make_batches(questions, 32, 16384)
            directory = OUT / model_name / task
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
            layout = [[q["question_id"] for q in b] for b in batches]
            if model_name == "trained":
                assert sha256(directory / "questions.json") == sha256(OUT / "original" / task / "questions.json")
                assert layout == json.loads((OUT / "original" / task / "manifest.json").read_text())["batch_layout"]
            run_manifest = {"benchmark": task, "split": "full reserved test partition",
                            "model": str(path), "model_name": model_name, "model_weights_sha256": hashes,
                            "selection_manifest_sha256": sha256(OUT / "manifest.json"),
                            "selected_records_sha256": sha256(OUT / "data" / task / "selected-records.json"),
                            "system_prompt": scorer_module.SYSTEM_PROMPT, "batch_size": 32, "padded_token_budget": 16384,
                            "batch_layout": layout, "option_token_ids": ids,
                            "evaluation_code_sha256": {n: sha256(code_path(n)) for n in ("evaluate_full_classification.py", "evaluate_choice_probabilities.py", "evaluate_mmlu_pro.py", "scorer.py")},
                            "questions_sha256": sha256(directory / "questions.json"),
                            "protocol": "Zero-shot classification; unchanged per-record shuffled class order; BF16; option-code softmax argmax; generation forbidden"}
            atomic_json(directory / "manifest.json", run_manifest)
            print(f"START {model_name} {task}: {len(questions)} questions; {len(batches)} batches", flush=True)
            batch_probabilities(scorer, batches[0], ids)
            measured = run_benchmark(scorer, questions, ids, batches, directory, run_manifest)
            predictions = read_lines(directory / "run-1.jsonl")
            metrics = classification_metrics(records, predictions, LABELS[task])
            assert measured["correct"] == metrics["correct"]
            metrics["inference"] = measured
            results.setdefault(task, {})[model_name] = metrics
            atomic_json(OUT / "progress.json", {"status": "evaluating", "last_model": model_name, "last_task": task, "results": results})
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    for task, result in results.items():
        result["accuracy_change_percentage_points"] = result["trained"]["accuracy_pct"] - result["original"]["accuracy_pct"]
        original = {p["source_id"]: p for p in read_lines(OUT / "original" / task / "run-1.jsonl")}
        trained = {p["source_id"]: p for p in read_lines(OUT / "trained" / task / "run-1.jsonl")}
        result["paired_changes"] = {
            "wrong_to_correct": sum(not original[k]["correct"] and trained[k]["correct"] for k in original),
            "correct_to_wrong": sum(original[k]["correct"] and not trained[k]["correct"] for k in original),
        }
    summary = {"complete": True, "results": results, "total_questions_per_model": sum(len(r) for r in by_task.values()),
               "total_predictions": 2 * sum(len(r) for r in by_task.values()), "elapsed_seconds": time.time() - start,
               "generated_tokens": 0, "repeats_per_model": 1, "manifest_sha256": sha256(OUT / "manifest.json")}
    atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "progress.json", {"status": "complete", "summary": str(OUT / "summary.json")})
    print("COMPLETE " + json.dumps({t: {m: results[t][m]["accuracy_pct"] for m in ("original", "trained")} for t in LABELS}), flush=True)


if __name__ == "__main__":
    main()
