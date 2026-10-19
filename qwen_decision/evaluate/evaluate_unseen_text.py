"""Matched, one-pass no-generation evaluation on newly downloaded datasets."""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import time

import torch

import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, encode_questions, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_probabilities import batch_probabilities, run_benchmark
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unseen-text-v1"
OUT = ROOT / "results/unseen-text-v1"
TRAINED = ROOT / "training/text-decision-full-v1/model"

def predictions(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "summary.json").exists(), "Preserve existing completed evaluations."
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert json.loads((DATA / "audit.json").read_text())["verified"]
    assert sha256(ROOT / "data/text-decisions-v1/train.jsonl") == manifest["training_data_sha256"]
    assert json.loads((ROOT / "training/text-decision-full-v1/pipeline.json").read_text())["status"] == "complete"
    scorer_module.SYSTEM_PROMPT = json.loads((ROOT / "data/text-decisions-v1/manifest.json").read_text())["system_prompt"]
    results = {}
    by_dataset = {}
    for spec in manifest["datasets"]:
        path = Path(spec["file"])
        assert sha256(path) == spec["sha256"]
        with path.open() as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == spec["evaluated_total"]
        by_dataset[spec["name"]] = records
    # Fix models, data and all prompts before observing results. No tuning,
    # checkpoint selection, or gradient updates use these evaluation datasets.
    start = time.time()
    for model_name, path in (("original", ROOT / "model"), ("trained", TRAINED)):
        scorer = ChoiceScorer(str(path), device="cuda")
        hashes = {p.name: sha256(p) for p in sorted(path.glob("*.safetensors"))}
        for spec in manifest["datasets"]:
            name = spec["name"]
            records = by_dataset[name]
            questions = [{"question_id": i, "source_id": r["id"], "category": name,
                          "request": r["request"], "gold_index": r["gold_index"], "option_count": r["option_count"]} for i, r in enumerate(records)]
            ids = encode_questions(scorer, questions)
            scorer.processor.tokenizer.padding_side = "left"
            batches = make_batches(questions, 32, 16384)
            directory = OUT / model_name / name
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
            if model_name == "trained":
                assert sha256(directory / "questions.json") == sha256(OUT / "original" / name / "questions.json")
            run_manifest = {"benchmark": name, "split": spec["source"]["split"] + "; full labeled split minus overlap/duplicate exclusions",
                            "dataset": spec, "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
                            "model": str(path), "model_name": model_name, "model_weights_sha256": hashes,
                            "system_prompt": scorer_module.SYSTEM_PROMPT, "batch_size": 32, "padded_token_budget": 16384,
                            "batch_layout": [[q["question_id"] for q in b] for b in batches],
                            "option_token_ids": ids, "protocol": "one pass per model; zero-shot; no generated tokens or CoT; choice logits softmax argmax; BF16; canonical source option order",
                            "evaluation_code_sha256": {n: sha256(code_path(n)) for n in ("evaluate_unseen_text.py", "evaluate_choice_probabilities.py", "evaluate_mmlu_pro.py", "scorer.py")},
                            "questions_sha256": sha256(directory / "questions.json"), "not_used_for_finetuning_or_selection": True}
            atomic_json(directory / "manifest.json", run_manifest)
            print(f"START {model_name} {name}: {len(questions)} questions; {len(batches)} batches", flush=True)
            # Kernel warmup, excluded from measured pass and reported totals.
            batch_probabilities(scorer, batches[0], ids)
            result = run_benchmark(scorer, questions, ids, batches, directory, run_manifest)
            pred = predictions(directory / "run-1.jsonl")
            assert len(pred) == len(questions) and len({r["source_id"] for r in pred}) == len(questions)
            result["first_display_position_pct"] = 100 * sum(r["prediction"] == "A" for r in pred) / len(pred)
            result["random_choice_expected_accuracy_pct"] = 100 * sum(1 / r["option_count"] for r in records) / len(records)
            result["prediction_position_counts"] = dict(Counter(r["prediction"] for r in pred))
            results.setdefault(name, {})[model_name] = result
            atomic_json(OUT / "progress.json", {"status": "evaluating", "last_model": model_name, "last_dataset": name, "results": results})
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    for name, result in results.items():
        result["accuracy_change_percentage_points"] = result["trained"]["accuracy_pct"] - result["original"]["accuracy_pct"]
    summary = {"complete": True, "results": results, "total_questions_per_model": sum(len(x) for x in by_dataset.values()),
               "elapsed_seconds": time.time() - start, "generated_tokens": 0, "repeats_per_model": 1,
               "interpretation": "New datasets not used in our fine-tuning or model selection; base pretraining exposure unknown. Scores use option-code logits, not standard completion likelihood protocols.",
               "dataset_manifest_sha256": sha256(DATA / "manifest.json")}
    atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "progress.json", {"status": "complete", "summary": str(OUT / "summary.json")})
    print("EVALUATION COMPLETE " + json.dumps(results), flush=True)

if __name__ == "__main__":
    main()
