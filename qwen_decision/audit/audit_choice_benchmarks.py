"""Independent recount of the three single-pass generated-answer evaluations."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import string

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, sha256
from qwen_decision.paths import code_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/choice-benchmarks-generate-1run")
    args = parser.parse_args()
    combined = json.loads((args.results / "summary.json").read_text())
    assert combined["complete"] and combined["repeats_per_benchmark"] == 1
    assert {row["benchmark"] for row in combined["results"]} == {"ARC-Challenge", "CommonsenseQA", "PIQA"}
    data_root = ROOT / "data/choice-benchmarks"
    audits = []
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    previous = json.loads((ROOT / "results/mmlu-pro-generate-5runs/manifest.json").read_text())
    for result in combined["results"]:
        name = result["benchmark"]
        directory = args.results / name
        manifest = json.loads((directory / "manifest.json").read_text())
        for key in ("model_revision", "model_config_sha256", "model_weights_sha256", "system_prompt", "generation_config"):
            assert manifest[key] == previous[key], key
        assert manifest["repeats"] == 1 and manifest["few_shot_examples"] == 0
        assert manifest["sampling"] is False and manifest["enable_thinking"] is False
        for filename, digest in manifest["code_sha256"].items():
            assert sha256(code_path(filename)) == digest
        files = list(manifest["dataset_files_sha256"])
        for filename, digest in manifest["dataset_files_sha256"].items():
            assert sha256(data_root / filename) == digest
        if name == "PIQA":
            assert manifest["split"] == "validation"
            inputs = [json.loads(line) for line in (data_root / files[0]).read_text().splitlines()]
            labels = [int(line) for line in (data_root / files[1]).read_text().splitlines()]
            assert len(inputs) == len(labels) == 1838
            originals = [{"id": str(i), "question": row["goal"], "options": [row["sol1"], row["sol2"]],
                          "gold_index": label, "labels": ["0", "1"]} for i, (row, label) in enumerate(zip(inputs, labels))]
            assert sha256(data_root / "ybisk__piqa/physicaliqa-train-dev.zip") == manifest["archive_sha256"]
        else:
            rows = pq.read_table(data_root / files[0], columns=["id", "question", "choices", "answerKey"]).to_pylist()
            originals = []
            for row in rows:
                labels = row["choices"]["label"]
                assert row["answerKey"] in labels
                originals.append({"id": row["id"], "question": row["question"], "options": row["choices"]["text"],
                                  "gold_index": labels.index(row["answerKey"]), "labels": labels})
            assert len(originals) == (1172 if name == "ARC-Challenge" else 1221)
            assert manifest["split"] == ("test" if name == "ARC-Challenge" else "validation")
        normalized = json.loads((directory / "questions.json").read_text())
        assert sha256(directory / "questions.json") == manifest["questions_sha256"]
        assert len(normalized) == len(originals) == manifest["question_count"]
        for i, (source, question) in enumerate(zip(originals, normalized)):
            assert question["question_id"] == i and question["source_id"] == source["id"]
            assert question["original_choice_labels"] == source["labels"]
            assert question["gold_index"] == source["gold_index"]
            assert question["request"] == {
                "question": source["question"],
                "options": {string.ascii_uppercase[k]: text for k, text in enumerate(source["options"])},
            }
        records = [json.loads(line) for line in (directory / "run-1.jsonl").read_text().splitlines()]
        ids = [row["question_id"] for row in records]
        assert len(ids) == len(set(ids)) == len(originals) and set(ids) == set(range(len(originals)))
        correct = invalid = output_tokens = 0
        predictions = {}
        for record in records:
            source = originals[record["question_id"]]
            valid_letters = set(string.ascii_uppercase[:len(source["options"])])
            gold = string.ascii_uppercase[source["gold_index"]]
            assert record["gold"] == gold and record["category"] == name
            tokens = record["generated_token_ids"]
            assert 0 < len(tokens) <= manifest["generation_config"]["max_new_tokens"]
            assert record["output_tokens"] == len(tokens)
            assert record["response"] == tokenizer.decode(tokens, skip_special_tokens=True)
            raw = tokenizer.decode(tokens, skip_special_tokens=False)
            assert record["response_with_special_tokens"] == raw
            assert "<think>" not in raw and "</think>" not in raw
            eos = manifest["generation_config"]["eos_token_id"]
            assert eos not in tokens[:-1] and record["ended_with_eos"] == (tokens[-1] == eos)
            response = record["response"].strip()
            prediction = response if re.fullmatch("[A-J]", response) and response in valid_letters else None
            assert prediction == record["prediction"]
            assert record["valid_format"] == (prediction is not None)
            is_correct = prediction == gold
            assert record["correct"] == is_correct
            correct += int(is_correct)
            invalid += int(prediction is None)
            output_tokens += len(tokens)
            predictions[record["question_id"]] = prediction
        accuracy = 100 * correct / len(originals)
        assert (correct, len(originals), accuracy, invalid, output_tokens) == (
            result["correct"], result["total"], result["accuracy_pct"], result["invalid_format_count"], result["output_tokens"],
        )
        assert result["prediction_sha256"] == hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest()
        assert result["manifest_sha256"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        assert result["generation_calls"] == manifest["batch_count"] and result["forward_calls"] >= result["generation_calls"]
        audits.append({
            "benchmark": name, "split": manifest["split"], "correct": correct, "total": len(originals),
            "accuracy_pct": accuracy, "invalid_format_count": invalid,
            "uniform_random_expected_accuracy_pct": 100 * sum(1 / len(row["options"]) for row in originals) / len(originals),
            "prediction_label_counts": dict(Counter(record["prediction"] for record in records)),
            "gold_label_counts": dict(Counter(record["gold"] for record in records)),
            "generated_token_count_distribution": dict(Counter(record["output_tokens"] for record in records)),
            "jsonl_sha256": sha256(directory / "run-1.jsonl"),
        })
    audit = {"verified": True, "benchmarks": audits}
    atomic_json(args.results / "audit.json", audit)
    with (args.results / "scores.csv").open("w", newline="") as handle:
        columns = ["benchmark", "split", "correct", "total", "accuracy_pct", "invalid_format_count", "uniform_random_expected_accuracy_pct"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in audits:
            writer.writerow({key: row[key] for key in columns})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
