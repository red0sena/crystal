"""Audit remapped gold content, label/position counts and tie-free bias controls."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import string

from transformers import AutoTokenizer

from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, sha256
from qwen_decision.paths import code_path


def json_counts(counter):
    return {str(key): value for key, value in counter.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/choice-bias-controls")
    parser.add_argument("--crossed", action="store_true", help="Audit independently shuffled content and labels")
    args = parser.parse_args()
    combined = json.loads((args.results / "summary.json").read_text())
    assert combined["complete"] and len(combined["results"]) == (6 if args.crossed else 15)
    baseline = ROOT / "results/choice-benchmarks-probabilities-1run"
    assert json.loads((baseline / "audit.json").read_text())["verified"]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    audits, plans = [], {}
    for result in combined["results"]:
        name, case = result["benchmark"], result["case"]
        directory = args.results / name / case
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["baseline_manifest_sha256"] == sha256(baseline / name / "manifest.json")
        old_manifest = json.loads((baseline / name / "manifest.json").read_text())
        for key in ("model_id", "model_revision", "model_weights_sha256", "dataset_files_sha256", "dtype", "split"):
            assert manifest[key] == old_manifest[key]
        assert manifest["enable_thinking"] is False and manifest["training"] is False
        assert manifest["temperature"] == 1 and manifest["repeats"] == 1 and manifest["padding"] == "left"
        for filename, digest in manifest["code_sha256"].items():
            assert digest == sha256(code_path(filename))
        for filename, digest in manifest["dataset_files_sha256"].items():
            assert digest == sha256(ROOT / "data/choice-benchmarks" / filename)
        assert old_manifest["questions_sha256"] == sha256(baseline / name / "questions.json")
        originals = json.loads((baseline / name / "questions.json").read_text())
        questions = json.loads((directory / "questions.json").read_text())
        assert manifest["questions_sha256"] == sha256(directory / "questions.json")
        assert len(questions) == len(originals) == result["total"]
        plans[(name, case)] = questions
        neutral = case.startswith("neutral_")
        expected_system = old_manifest["system_prompt"].replace(
            "Reply with only its uppercase letter (for example A).", "Reply with only its option label.",
        ) if neutral else old_manifest["system_prompt"]
        assert manifest["system_prompt"] == expected_system
        for original, q in zip(originals, questions):
            assert (q["question_id"], q["source_id"], q["request"]["question"]) == (
                original["question_id"], original["source_id"], original["request"]["question"],
            )
            indices, labels = q["display_to_original_index"], list(q["request"]["options"])
            texts = list(original["request"]["options"].values())
            count = len(texts)
            assert sorted(indices) == list(range(count))
            assert list(q["request"]["options"].values()) == [texts[i] for i in indices]
            expected_labels = [str(i + 1) for i in range(count)] if case == "neutral_shuffle_numbers" else list(string.ascii_uppercase[:count])
            assert sorted(labels) == sorted(expected_labels)
            if case != "shuffle_contents" and not args.crossed:
                assert indices == list(range(count))
            if case in ("shuffle_contents", "neutral_letters"):
                assert labels == expected_labels
            gold_index = original["gold_index"]
            assert q["gold_original_index"] == gold_index
            assert q["gold_display_index"] == indices.index(gold_index)
            assert q["gold_label"] == labels[indices.index(gold_index)]
        assert [tokenizer.encode(label, add_special_tokens=False) for label in manifest["candidates"]] == [[i] for i in manifest["candidate_token_ids"]]
        records = [json.loads(line) for line in (directory / "run-1.jsonl").read_text().splitlines()]
        assert len(records) == len(questions) and {r["question_id"] for r in records} == set(range(len(questions)))
        old = {r["question_id"]: r for r in (json.loads(line) for line in (baseline / name / "run-1.jsonl").read_text().splitlines())}
        correct = ties = content_stable = 0
        labels_count, positions, gold_labels, gold_positions, exposure = Counter(), Counter(), Counter(), Counter(), Counter()
        unique_labels, unique_positions = Counter(), Counter()
        not_first = not_first_label = not_first_first = 0
        unique_not_first = unique_not_first_label = unique_not_first_first = 0
        preferred = result["preferred_label"]
        assert preferred == ("1" if case == "neutral_shuffle_numbers" else "A")
        predictions, max_softmax_error = {}, 0.0
        for r in records:
            q = questions[r["question_id"]]
            labels = list(q["request"]["options"])
            assert r["source_id"] == q["source_id"] and r["display_labels"] == labels
            assert r["display_to_original_index"] == q["display_to_original_index"]
            for key in ("gold_label", "gold_display_index", "gold_original_index"):
                assert r[key] == q[key]
            logits, probs = r["logits"], r["probabilities"]
            assert len(logits) == len(probs) == len(labels)
            assert all(math.isfinite(x) for x in logits + probs)
            weights = [math.exp(x - max(logits)) for x in logits]
            expected = [x / sum(weights) for x in weights]
            error = max(abs(a - b) for a, b in zip(expected, probs))
            assert error < 2e-7 and abs(sum(probs) - 1) < 2e-7 and all(0 <= p <= 1 for p in probs)
            max_softmax_error = max(max_softmax_error, error)
            selected = max(range(len(logits)), key=lambda i: logits[i])
            assert selected == max(range(len(probs)), key=lambda i: probs[i])
            assert r["prediction_display_index"] == selected and r["prediction_label"] == labels[selected]
            original_index = q["display_to_original_index"][selected]
            assert r["prediction_original_index"] == original_index
            is_correct = original_index == q["gold_original_index"]
            unique = logits.count(max(logits)) == 1
            assert r["correct"] == is_correct and r["unique_maximum"] == unique and r["output_tokens"] == 0
            correct += int(is_correct)
            ties += int(not unique)
            labels_count[labels[selected]] += 1
            positions[selected + 1] += 1
            gold_labels[q["gold_label"]] += 1
            gold_positions[q["gold_display_index"] + 1] += 1
            exposure[labels.index(preferred) + 1] += 1
            if unique:
                unique_labels[labels[selected]] += 1
                unique_positions[selected + 1] += 1
            if labels[0] != preferred:
                not_first += 1
                not_first_label += int(labels[selected] == preferred)
                not_first_first += int(selected == 0)
                if unique:
                    unique_not_first += 1
                    unique_not_first_label += int(labels[selected] == preferred)
                    unique_not_first_first += int(selected == 0)
            predictions[q["question_id"]] = original_index
            baseline_index = string.ascii_uppercase.index(old[q["question_id"]]["prediction"])
            content_stable += int(original_index == baseline_index)
        total, unique_total = len(records), len(records) - ties
        assert correct == result["correct"] and result["accuracy_pct"] == 100 * correct / total
        for key, counter in (("prediction_label_counts", labels_count), ("prediction_position_counts", positions),
                             ("gold_label_counts", gold_labels), ("gold_position_counts", gold_positions),
                             ("preferred_label_position_exposure", exposure)):
            assert result[key] == json_counts(counter), key
        assert result["first_position_selection_pct"] == 100 * positions[1] / total
        assert result["preferred_label_selection_pct"] == 100 * labels_count[preferred] / total
        assert result["top_logit_tie_count"] == ties and result["unique_maximum_count"] == unique_total
        assert result["first_position_selection_pct_excluding_ties"] == (100 * unique_positions[1] / unique_total if unique_total else None)
        assert result["preferred_label_selection_pct_excluding_ties"] == (100 * unique_labels[preferred] / unique_total if unique_total else None)
        assert result["generation_calls"] == result["output_tokens"] == 0 and result["forward_calls"] == manifest["batch_count"]
        assert result["prediction_sha256"] == hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest()
        assert result["manifest_sha256"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        audits.append({**result, "verified": True, "content_agreement_with_baseline_pct": 100 * content_stable / total,
            "max_softmax_error": max_softmax_error,
            "preferred_not_first_count": not_first,
            "preferred_label_selection_pct_when_not_first": 100 * not_first_label / not_first if not_first else None,
            "first_position_selection_pct_when_preferred_not_first": 100 * not_first_first / not_first if not_first else None,
            "unique_maximum_preferred_not_first_count": unique_not_first,
            "preferred_label_selection_pct_when_not_first_excluding_ties": 100 * unique_not_first_label / unique_not_first if unique_not_first else None,
            "first_position_selection_pct_when_preferred_not_first_excluding_ties": 100 * unique_not_first_first / unique_not_first if unique_not_first else None,
            "jsonl_sha256": sha256(directory / "run-1.jsonl")})
    for name in {r["benchmark"] for r in audits}:
        neutral_labels = plans[(name, "neutral_shuffle_letters")]
        numeric_labels = plans[(name, "neutral_shuffle_numbers")]
        if not args.crossed:
            original_labels = plans[(name, "shuffle_labels")]
            assert original_labels == neutral_labels
        else:
            control_root = ROOT / "results/choice-bias-controls" / name
            content_control = json.loads((control_root / "shuffle_contents/questions.json").read_text())
            label_control = json.loads((control_root / "neutral_shuffle_letters/questions.json").read_text())
            for q, content, label in zip(neutral_labels, content_control, label_control):
                assert q["display_to_original_index"] == content["display_to_original_index"]
                assert list(q["request"]["options"]) == list(label["request"]["options"])
        for letters, digits in zip(neutral_labels, numeric_labels):
            assert list(digits["request"]["options"]) == [str(string.ascii_uppercase.index(label) + 1) for label in letters["request"]["options"]]
            assert letters["display_to_original_index"] == digits["display_to_original_index"]
    atomic_json(args.results / "audit.json", {"verified": True, "results": audits})
    columns = ["benchmark", "case", "accuracy_pct", "first_position_selection_pct", "preferred_label_selection_pct",
        "first_position_selection_pct_excluding_ties", "preferred_label_selection_pct_excluding_ties",
        "top_logit_tie_count", "content_agreement_with_baseline_pct"]
    with (args.results / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in audits:
            writer.writerow({key: row[key] for key in columns})
    print(json.dumps({"verified": True, "results": [{key: row[key] for key in columns} for row in audits]}, indent=2))


if __name__ == "__main__":
    main()
