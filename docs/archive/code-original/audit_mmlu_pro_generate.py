"""Recount generated-answer results from raw tokens and independent dataset labels."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import string

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from evaluate_mmlu_pro import ROOT, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/mmlu-pro-generate-5runs")
    parser.add_argument("--previous", type=Path, default=ROOT / "results/mmlu-pro-0shot-5runs")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    manifest = summary["manifest"]
    assert summary["complete"] and manifest["repeats"] == 5 and manifest["limit"] is None
    previous_summary = json.loads((args.previous / "summary.json").read_text())
    previous_manifest = previous_summary["manifest"]
    for key in ("model_revision", "model_config_sha256", "model_weights_sha256", "dataset_sha256",
                "question_count", "category_counts", "system_prompt", "prompt_builder_sha256"):
        assert manifest[key] == previous_manifest[key], key
    assert manifest["enable_thinking"] is False and manifest["sampling"] is False
    for key, filename in (("evaluator_sha256", "evaluate_mmlu_pro_generate.py"),
                          ("shared_evaluator_sha256", "evaluate_mmlu_pro.py"),
                          ("prompt_builder_sha256", "scorer.py")):
        assert manifest[key] == sha256(ROOT / filename)
    config = manifest["generation_config"]
    assert config["do_sample"] is False and config["num_beams"] == 1
    for key in ("forced_bos_token_id", "forced_eos_token_id", "force_words_ids", "suppress_tokens",
                "begin_suppress_tokens", "bad_words_ids", "constraints"):
        assert config.get(key) is None, key
    source = Path(previous_manifest["model_path"]).parent / "data/mmlu-pro/data/test-00000-of-00001.parquet"
    assert sha256(source) == manifest["dataset_sha256"]
    truth = {}
    for row in pq.read_table(source, columns=["question_id", "category", "options", "answer_index"]).to_pylist():
        retained = [i for i, option in enumerate(row["options"]) if option != "N/A"]
        truth[row["question_id"]] = {
            "gold": string.ascii_uppercase[retained.index(row["answer_index"])],
            "category": row["category"], "count": len(retained),
        }
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"])
    previous = {r["question_id"]: r for r in (
        json.loads(line) for line in (args.previous / "run-1.jsonl").read_text().splitlines()
    )}
    audits, first, scores, disagreements, responses_equal = [], None, [], [], []
    first_records = None
    for run in summary["runs"]:
        path = args.results / f"run-{run['run']}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == len(truth) == run["total"]
        ids = [record["question_id"] for record in records]
        assert len(set(ids)) == len(ids) and set(ids) == set(truth)
        correct = invalid = tokens = 0
        per_category = defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0})
        predictions = {}
        for record in records:
            expected = truth[record["question_id"]]
            assert record["gold"] == expected["gold"] and record["category"] == expected["category"]
            generated = record["generated_token_ids"]
            assert 0 < len(generated) <= config["max_new_tokens"]
            assert record["output_tokens"] == len(generated)
            assert record["response"] == tokenizer.decode(generated, skip_special_tokens=True)
            assert record["response_with_special_tokens"] == tokenizer.decode(generated, skip_special_tokens=False)
            assert "<think>" not in record["response_with_special_tokens"]
            assert "</think>" not in record["response_with_special_tokens"]
            assert record["ended_with_eos"] == (generated[-1] == config["eos_token_id"])
            assert config["eos_token_id"] not in generated[:-1]
            label = record["response"].strip()
            prediction = label if re.fullmatch(r"[A-J]", label) and label in string.ascii_uppercase[:expected["count"]] else None
            is_correct = prediction == expected["gold"]
            assert record["prediction"] == prediction
            assert record["correct"] == is_correct and record["valid_format"] == (prediction is not None)
            correct += int(is_correct)
            invalid += int(prediction is None)
            tokens += len(generated)
            stats = per_category[record["category"]]
            stats["total"] += 1
            stats["correct"] += int(is_correct)
            stats["invalid"] += int(prediction is None)
            predictions[record["question_id"]] = prediction
        for stats in per_category.values():
            stats["accuracy_pct"] = 100 * stats["correct"] / stats["total"]
        score = 100 * correct / len(truth)
        assert run["correct"] == correct and run["invalid_format_count"] == invalid
        assert run["output_tokens"] == tokens and run["accuracy_pct"] == score
        assert run["per_category"] == dict(sorted(per_category.items()))
        assert run["generation_calls"] == manifest["batch_count"]
        assert run["forward_calls"] >= run["generation_calls"]
        assert run["prediction_sha256"] == hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest()
        assert run["manifest_sha256"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if first is None:
            first, first_records = predictions, records
        disagreements.append(sum(predictions[k] != first[k] for k in first))
        responses_equal.append(all(a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(records, first_records)))
        scores.append(score)
        audits.append({"run": run["run"], "correct": correct, "total": len(truth),
                       "accuracy_pct": score, "invalid_format_count": invalid,
                       "jsonl_sha256": sha256(path)})
    assert len(audits) == 5
    assert summary["mean_accuracy_pct"] == statistics.mean(scores)
    assert summary["population_stddev_percentage_points"] == statistics.pstdev(scores)
    assert summary["prediction_disagreements_vs_run_1"] == disagreements
    comparison, changed_logit_gaps = Counter(), Counter()
    for record in first_records:
        old = previous[record["question_id"]]
        assert record["input_tokens"] == old["input_tokens"] and record["gold"] == old["gold"]
        comparison["different_answers"] += int(record["prediction"] != old["prediction"])
        comparison["both_correct"] += int(record["correct"] and old["correct"])
        comparison["generation_only_correct"] += int(record["correct"] and not old["correct"])
        comparison["scoring_only_correct"] += int(not record["correct"] and old["correct"])
        comparison["both_wrong"] += int(not record["correct"] and not old["correct"])
        if record["prediction"] != old["prediction"]:
            ranked = sorted(old["logits"], reverse=True)
            changed_logit_gaps[str(ranked[0] - ranked[1])] += 1
    result = {
        "verified": True, "runs": audits, "raw_response_tokens_equal_across_runs": responses_equal,
        "comparison_with_previous_scoring": dict(comparison),
        "previous_mean_accuracy_pct": previous_summary["mean_accuracy_pct"],
        "generation_mean_accuracy_pct": statistics.mean(scores),
        "difference_percentage_points": statistics.mean(scores) - previous_summary["mean_accuracy_pct"],
        "generated_label_counts": dict(Counter(r["prediction"] for r in first_records)),
        "non_label_first_token_count": sum(
            tokenizer.decode([r["generated_token_ids"][0]]) not in set(string.ascii_uppercase[:truth[r["question_id"]]["count"]])
            for r in first_records),
        "invalid_responses": dict(Counter(r["response"] for r in first_records if not r["valid_format"])),
        "generated_token_count_distribution": dict(Counter(r["output_tokens"] for r in first_records)),
        "previous_top_two_logit_gap_for_changed_answers": dict(changed_logit_gaps),
    }
    atomic_json(args.results / "audit.json", result)
    with (args.results / "accuracy-by-category.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "total", "correct", "invalid", "accuracy_pct"])
        writer.writeheader()
        for category, stats in summary["runs"][0]["per_category"].items():
            writer.writerow({"category": category, **stats})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
