"""Check option softmax/argmax against immutable, previously audited question mappings."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import string

from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, sha256
from qwen_decision.paths import code_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/choice-benchmarks-probabilities-1run")
    parser.add_argument("--previous", type=Path, default=ROOT / "results/choice-benchmarks-generate-1run")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    prior_audit = json.loads((args.previous / "audit.json").read_text())
    assert summary["complete"] and summary["repeats_per_benchmark"] == 1 and prior_audit["verified"]
    assert {r["benchmark"] for r in summary["results"]} == {"ARC-Challenge", "CommonsenseQA", "PIQA"}
    audited_old = {r["benchmark"]: r for r in prior_audit["benchmarks"]}
    audited = []
    for result in summary["results"]:
        name = result["benchmark"]
        directory, previous_directory = args.results / name, args.previous / name
        manifest = json.loads((directory / "manifest.json").read_text())
        previous_manifest = json.loads((previous_directory / "manifest.json").read_text())
        for key in ("model_id", "model_revision", "model_config_sha256", "model_weights_sha256", "dataset_revision",
                    "dataset_files_sha256", "questions_sha256", "system_prompt", "padding", "batch_count", "batch_size",
                    "batch_token_budget", "dtype", "split", "question_count", "few_shot_examples", "enable_thinking"):
            assert manifest[key] == previous_manifest[key], key
        assert manifest["enable_thinking"] is False and manifest["few_shot_examples"] == 0
        assert manifest["temperature_for_option_softmax"] == 1.0 and manifest["head_mode"] == "selected"
        for filename, digest in manifest["code_sha256"].items():
            assert sha256(code_path(filename)) == digest
        for filename, digest in manifest["dataset_files_sha256"].items():
            assert sha256(ROOT / "data/choice-benchmarks" / filename) == digest
        # These exact canonical mappings were independently compared to all
        # original question/choice/gold rows by audit_choice_benchmarks.py.
        assert sha256(directory / "questions.json") == sha256(previous_directory / "questions.json") == manifest["questions_sha256"]
        questions = json.loads((directory / "questions.json").read_text())
        assert sha256(previous_directory / "run-1.jsonl") == audited_old[name]["jsonl_sha256"]
        old_records = [json.loads(line) for line in (previous_directory / "run-1.jsonl").read_text().splitlines()]
        records = [json.loads(line) for line in (directory / "run-1.jsonl").read_text().splitlines()]
        assert len(records) == len(questions) == len(old_records) == manifest["question_count"] == result["total"]
        assert [r["question_id"] for r in records] == [r["question_id"] for r in old_records]
        assert len({r["question_id"] for r in records}) == len(records)
        assert set(r["question_id"] for r in records) == set(range(len(questions)))
        old_by_id = {r["question_id"]: r for r in old_records}
        correct = changed = scoring_only_correct = generation_only_correct = 0
        max_error = 0.0
        predictions, changes = {}, []
        for record in records:
            question = questions[record["question_id"]]
            labels = list(question["request"]["options"])
            assert len(labels) == question["option_count"] and labels == list(string.ascii_uppercase[:len(labels)])
            assert record["option_labels"] == labels and record["source_id"] == question["source_id"]
            assert record["gold"] == labels[question["gold_index"]] and record["category"] == name
            logits, probabilities = record["logits"], record["probabilities"]
            assert len(logits) == len(probabilities) == len(labels)
            assert all(math.isfinite(x) for x in logits + probabilities)
            assert all(0 <= x <= 1 for x in probabilities) and abs(sum(probabilities) - 1) < 2e-7
            weights = [math.exp(x - max(logits)) for x in logits]
            expected = [x / sum(weights) for x in weights]
            error = max(abs(a - b) for a, b in zip(probabilities, expected))
            assert error < 2e-7
            max_error = max(max_error, error)
            winner = max(range(len(labels)), key=lambda i: probabilities[i])
            assert winner == max(range(len(labels)), key=lambda i: logits[i])
            assert record["prediction"] == labels[winner]
            is_correct = record["prediction"] == record["gold"]
            assert record["correct"] == is_correct and record["output_tokens"] == 0
            correct += int(is_correct)
            predictions[record["question_id"]] = record["prediction"]
            old = old_by_id[record["question_id"]]
            assert record["input_tokens"] == old["input_tokens"] and record["gold"] == old["gold"]
            scoring_only_correct += int(is_correct and not old["correct"])
            generation_only_correct += int(not is_correct and old["correct"])
            if record["prediction"] != old["prediction"]:
                changed += 1
                ranked = sorted(logits, reverse=True)
                changes.append({"question_id": record["question_id"], "source_id": record["source_id"],
                                "generation_prediction": old["prediction"], "probability_prediction": record["prediction"],
                                "gold": record["gold"], "top_two_logit_gap": ranked[0] - ranked[1]})
        assert correct == result["correct"] and result["accuracy_pct"] == 100 * correct / len(questions)
        assert result["generation_calls"] == result["output_tokens"] == 0
        assert result["forward_calls"] == manifest["batch_count"]
        assert result["prediction_sha256"] == hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest()
        assert result["manifest_sha256"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        previous_score = audited_old[name]["accuracy_pct"]
        audit = {
            "benchmark": name, "split": manifest["split"], "correct": correct, "total": len(questions),
            "probability_accuracy_pct": result["accuracy_pct"], "generation_accuracy_pct": previous_score,
            "difference_percentage_points": result["accuracy_pct"] - previous_score,
            "changed_answers": changed, "probability_only_correct": scoring_only_correct,
            "generation_only_correct": generation_only_correct, "max_softmax_absolute_error": max_error,
            "prediction_label_counts": dict(Counter(predictions.values())),
            "changed_answer_top_two_gap_counts": dict(Counter(str(c["top_two_logit_gap"]) for c in changes)),
            "jsonl_sha256": sha256(directory / "run-1.jsonl"),
        }
        atomic_json(directory / "answer-differences.json", changes)
        audited.append(audit)
    combined = {"verified": True, "benchmarks": audited}
    atomic_json(args.results / "audit.json", combined)
    columns = ["benchmark", "split", "correct", "total", "probability_accuracy_pct", "generation_accuracy_pct",
               "difference_percentage_points", "changed_answers"]
    with (args.results / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in audited:
            writer.writerow({key: row[key] for key in columns})
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
