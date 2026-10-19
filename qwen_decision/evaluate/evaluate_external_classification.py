"""One no-generation pass per model on full screened SuperGPQA and KMMLU."""
from __future__ import annotations
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_probabilities import batch_probabilities
from qwen_decision.evaluate.external_classification_protocol import CODES, encode, single_probabilities
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/external-classification-eval-v1"
OUT = ROOT / "results/external-classification-v1"


def load_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def metrics(records, predictions, labels):
    matrix = [[0] * len(labels) for _ in labels]
    gold_by_id = {r["id"]: r["answer_id"] for r in records}
    assert len(predictions) == len(records) == len(gold_by_id)
    assert {p["source_id"] for p in predictions} == set(gold_by_id)
    positions = Counter()
    for p in predictions:
        gold = gold_by_id[p["source_id"]]
        assert p["correct"] == (p["prediction"] == gold)
        matrix[labels.index(gold)][labels.index(p["prediction"])] += 1
        positions[p["prediction_code"]] += 1
    per_class = {}
    for i, label in enumerate(labels):
        support, predicted, tp = sum(matrix[i]), sum(r[i] for r in matrix), matrix[i][i]
        per_class[label] = {"support": support, "correct": tp, "predicted_count": predicted,
                            "recall_pct": 100 * tp / support if support else None,
                            "precision_pct": 100 * tp / predicted if predicted else 0,
                            "f1_pct": 200 * tp / (support + predicted) if support + predicted else 0}
    n = len(records)
    result = {"total": n, "correct": sum(matrix[i][i] for i in range(len(labels))), "labels": labels,
              "per_class": per_class, "confusion_matrix_true_rows_predicted_columns": matrix,
              "macro_f1_pct": sum(v["f1_pct"] for v in per_class.values()) / len(labels),
              "balanced_accuracy_pct": sum(v["recall_pct"] for v in per_class.values() if v["support"]) / sum(bool(v["support"]) for v in per_class.values()),
              "majority_class_baseline_pct": 100 * max(v["support"] for v in per_class.values()) / n,
              "random_choice_expected_accuracy_pct": 100 / len(labels), "prediction_code_counts": dict(positions),
              "largest_confusions": sorted([{ "gold": labels[i], "predicted": labels[j], "count": matrix[i][j]}
                                            for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i][j]],
                                           key=lambda v: (-v["count"], v["gold"], v["predicted"]))[:10]}
    result["accuracy_pct"] = 100 * result["correct"] / n
    if labels == ["easy", "middle", "hard"]:
        result["ordinal_errors"] = {"overestimated": sum(matrix[i][j] for i in range(3) for j in range(3) if i < j),
                                    "underestimated": sum(matrix[i][j] for i in range(3) for j in range(3) if i > j),
                                    "two_level_errors": matrix[0][2] + matrix[2][0]}
    return result


def run(scorer, questions, token_ids, batches, directory, task):
    forwards = 0
    def hook(*args):
        nonlocal forwards
        forwards += 1
    handle = scorer.model.model.register_forward_hook(hook)
    partial = directory / "predictions.jsonl.partial"
    scorer._synchronize()
    start = last_progress = time.perf_counter()
    completed = correct = 0
    try:
        with partial.open("w") as output, patch.object(scorer.model, "generate", side_effect=AssertionError("Generation forbidden")):
            for batch in batches:
                logits, probs = batch_probabilities(scorer, batch, token_ids)
                for i, q in enumerate(batch):
                    count = q["option_count"]
                    index = int(probs[i, :count].argmax())
                    labels = list(q["request"]["options"])
                    pred = labels[index]
                    gold = labels[q["gold_index"]]
                    row = {"question_id": q["question_id"], "source_id": q["source_id"], "prediction_code": CODES[index],
                           "prediction": pred, "gold": gold, "correct": pred == gold, "option_labels": labels,
                           "logits": logits[i, :count].tolist(), "probabilities": probs[i, :count].tolist(),
                           "input_tokens": len(q["input_ids"]), "output_tokens": 0}
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    correct += int(row["correct"])
                    completed += 1
                now = time.perf_counter()
                if now - last_progress >= 25 or completed == len(questions):
                    output.flush()
                    atomic_json(directory / "progress.json", {"completed": completed, "total": len(questions), "correct": correct,
                                                               "accuracy_pct": 100 * correct / completed, "elapsed_seconds": now-start})
                    print(f"{task}: {completed}/{len(questions)}; accuracy {100*correct/completed:.2f}%", flush=True)
                    last_progress = now
    finally:
        handle.remove()
    scorer._synchronize()
    assert completed == len(questions) and forwards == len(batches)
    partial.replace(directory / "predictions.jsonl")
    return {"elapsed_seconds": time.perf_counter() - start, "forward_calls": forwards,
            "generation_calls": 0, "output_tokens": 0, "total": completed, "correct": correct,
            "predictions_sha256": sha256(directory / "predictions.jsonl")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not list(OUT.glob("*/*/predictions.jsonl")), "Preserve completed model results"
    data = json.loads((DATA / "manifest.json").read_text())
    assert data["complete"]
    assert sha256(ROOT / "data/text-decisions-v1/train.jsonl") == data["training_data_sha256"]
    scorer_module.SYSTEM_PROMPT = data["system_prompt"]
    results = {}
    start = time.time()
    torch.manual_seed(0)
    for model_name, path in (("original", ROOT / "model"), ("trained", ROOT / "training/text-decision-full-v1/model")):
        scorer = ChoiceScorer(str(path), device="cuda")
        weights = {p.name: sha256(p) for p in sorted(path.glob("*.safetensors"))}
        for spec in data["datasets"]:
            task = spec["task"]
            assert sha256(Path(spec["file"])) == spec["sha256"]
            records = load_lines(Path(spec["file"]))
            questions = [{"question_id": i, "source_id": r["id"], "request": r["request"],
                          "gold_index": r["gold_index"], "option_count": r["option_count"]} for i, r in enumerate(records)]
            ids = encode(scorer, questions)
            scorer.processor.tokenizer.padding_side = "left"
            batches = make_batches(questions, 32, 32768)
            directory = OUT / model_name / task
            directory.mkdir(parents=True, exist_ok=True)
            layout = [[q["question_id"] for q in batch] for batch in batches]
            encoding_hash = hashlib.sha256(json.dumps([q["input_ids"] for q in questions], separators=(",", ":")).encode()).hexdigest()
            manifest = {"task": task, "model": str(path), "model_name": model_name, "model_weights_sha256": weights,
                        "data_manifest_sha256": sha256(DATA / "manifest.json"), "dataset_file_sha256": spec["sha256"],
                        "batch_layout": layout, "batch_size": 32, "padded_token_budget": 32768,
                        "encoding_sha256": encoding_hash, "codes": CODES[:len(ids)], "option_token_ids": ids,
                        "code_sha256": {n: sha256(code_path(n)) for n in ("evaluate_external_classification.py", "external_classification_protocol.py", "evaluate_choice_probabilities.py", "evaluate_mmlu_pro.py", "scorer.py")},
                        "system_prompt": scorer_module.SYSTEM_PROMPT,
                        "protocol": "One measured pass; per-record deterministic shuffled options; BF16 selected LM Head rows; no generated tokens, CoT, training or checkpoint selection. Native 45-class KMMLU uses uppercase/lowercase single-token codes."}
            if model_name == "trained":
                prior = json.loads((OUT / "original" / task / "manifest.json").read_text())
                for key in ("encoding_sha256", "batch_layout", "codes", "option_token_ids", "dataset_file_sha256"):
                    assert prior[key] == manifest[key], f"Unmatched {key}"
            # Warmup and the extended-code batch/single check are outside the
            # measured pass. Both models receive exactly the same check inputs.
            print(f"START {model_name} {task}: {len(questions)} questions; {len(batches)} batches", flush=True)
            warm_logits, warm_probs = batch_probabilities(scorer, batches[0], ids)
            if spec["task"] == "KMMLU_domain":
                differences = []
                for i, q in enumerate(batches[0][:3]):
                    single_logits, single_probs = single_probabilities(scorer, q, ids)
                    differences.append({"question_id": q["question_id"], "max_probability_difference": float((single_probs[0]-warm_probs[i]).abs().max()),
                                        "argmax_equal": int(single_probs[0].argmax()) == int(warm_probs[i].argmax())})
                assert all(x["max_probability_difference"] < .05 for x in differences)
                manifest["extended_code_verification"] = differences
            atomic_json(directory / "manifest.json", manifest)
            atomic_json(OUT / "progress.json", {"status": "evaluating", "current_model": model_name, "current_task": task,
                                               "total_questions": len(questions), "results": results})
            inference = run(scorer, questions, ids, batches, directory, task)
            predictions = load_lines(directory / "predictions.jsonl")
            measured = metrics(records, predictions, spec["labels"])
            assert measured["correct"] == inference["correct"]
            measured["inference"] = inference
            results.setdefault(task, {})[model_name] = measured
            atomic_json(directory / "summary.json", measured)
            atomic_json(OUT / "partial-summary.json", results)
            del questions, records, predictions, batches
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    for task, r in results.items():
        r["accuracy_change_percentage_points"] = r["trained"]["accuracy_pct"] - r["original"]["accuracy_pct"]
    atomic_json(OUT / "summary.json", {"complete": True, "results": results,
                "total_questions_per_model": data["total_requests_per_model"], "elapsed_seconds": time.time()-start,
                "repeats_per_model": 1, "generated_tokens": 0, "data_manifest_sha256": sha256(DATA / "manifest.json"),
                "interpretation": "Unseen retained examples and changed native taxonomies/difficulty criteria; excludes exact/near matches to training and prior corpora. Does not certify base pretraining independence. KMMLU difficulty bins are derived, not native annotations."})
    atomic_json(OUT / "progress.json", {"status": "complete"})
    print("ALL COMPLETE " + json.dumps({t: {m: r[m]["accuracy_pct"] for m in ("original", "trained")} for t,r in results.items()}), flush=True)


if __name__ == "__main__":
    main()
