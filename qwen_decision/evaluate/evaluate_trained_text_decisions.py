"""Evaluate fixed held-out examples and position sensitivity after training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, encode_questions, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_probabilities import run_benchmark

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results/text-decisions-v1-baseline"

def read_predictions(directory):
    with (directory / "run-1.jsonl").open() as handle:
        return {r["source_id"]: r for r in map(json.loads, handle)}

def run_pass(scorer, task, records, out, shuffled, weights):
    questions = []
    for i, r in enumerate(records):
        names = list(r["request"]["options"])
        if shuffled:
            rng = random.Random(int(hashlib.sha256(("position-eval:" + r["id"]).encode()).hexdigest(), 16))
            rng.shuffle(names)
            if names == list(r["request"]["options"]):
                names = names[1:] + names[:1]
        req = {**r["request"], "options": {k: r["request"]["options"][k] for k in names}}
        questions.append({"question_id": i, "source_id": r["id"], "category": task,
                          "request": req, "gold_index": names.index(r["answer_id"]), "option_count": len(names)})
    ids = encode_questions(scorer, questions)
    scorer.processor.tokenizer.padding_side = "left"
    batches = make_batches(questions, 32, 16384)
    out.mkdir(parents=True)
    atomic_json(out / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
    manifest = {"benchmark": task, "split": "fixed 120 held-out examples from original baseline",
                "model_weights_sha256": weights, "system_prompt": scorer_module.SYSTEM_PROMPT,
                "options_shuffled": shuffled, "protocol": "no generation; selected next-token logits; softmax argmax; BF16; left padding",
                "baseline_records_sha256": sha256(BASELINE / task / "selected-records.json"),
                "evaluation_code_sha256": sha256(Path(__file__))}
    atomic_json(out / "manifest.json", manifest)
    summary = run_benchmark(scorer, questions, ids, batches, out, manifest)
    pred = read_predictions(out)
    return summary, pred

def selected_name(p):
    return p["option_labels"][ord(p["prediction"]) - ord("A")]

def metrics(pred):
    rows = list(pred.values())
    ece = 0.
    for bin_index in range(15):
        items = [r for r in rows if min(14, int(max(r["probabilities"]) * 15)) == bin_index]
        if items:
            ece += len(items) / len(rows) * abs(sum(r["correct"] for r in items) / len(items) - sum(max(r["probabilities"]) for r in items) / len(items))
    return {"accuracy_pct": 100 * sum(r["correct"] for r in rows) / len(rows),
            "first_display_position_pct": 100 * sum(r["prediction"] == "A" for r in rows) / len(rows),
            "ece_15_bins": ece, "samples": len(rows)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.output / "summary.json").exists()
    baseline_summary = json.loads((BASELINE / "summary.json").read_text())
    assert baseline_summary["complete"]
    assert json.loads((BASELINE / "audit.json").read_text())["verified"]
    scorer_module.SYSTEM_PROMPT = json.loads((ROOT / "data/text-decisions-v1/manifest.json").read_text())["system_prompt"]
    records = {r["benchmark"]: json.loads((BASELINE / r["benchmark"] / "selected-records.json").read_text()) for r in baseline_summary["results"]}
    # The baseline canonical results already exist. Add a matched shuffled pass
    # on the original weights to measure content consistency before training.
    results = {}
    for model_name, path in (("original", ROOT / "model"), ("trained", args.model)):
        scorer = ChoiceScorer(str(path), device="cuda")
        weights = {p.name: sha256(p) for p in sorted(path.glob("*.safetensors"))}
        for task, chosen in sorted(records.items()):
            canonical = read_predictions(BASELINE / task) if model_name == "original" else run_pass(scorer, task, chosen, args.output / model_name / "canonical" / task, False, weights)[1]
            if model_name == "original":
                assert weights == json.loads((BASELINE / task / "manifest.json").read_text())["model_weights_sha256"]
            shuffled = run_pass(scorer, task, chosen, args.output / model_name / "shuffled" / task, True, weights)[1]
            assert set(canonical) == set(shuffled) == {r["id"] for r in chosen}
            result = {"canonical": metrics(canonical), "shuffled": metrics(shuffled),
                      "same_content_selection_pct": 100 * sum(selected_name(p) == selected_name(shuffled[k]) for k, p in canonical.items()) / len(canonical)}
            results.setdefault(task, {})[model_name] = result
            atomic_json(args.output / "progress.json", {"status": "evaluating", "model": model_name, "last_task": task, "results": results})
        del scorer
        torch.cuda.empty_cache()
    for task, r in results.items():
        r["accuracy_change_percentage_points"] = r["trained"]["canonical"]["accuracy_pct"] - r["original"]["canonical"]["accuracy_pct"]
        r["consistency_change_percentage_points"] = r["trained"]["same_content_selection_pct"] - r["original"]["same_content_selection_pct"]
    atomic_json(args.output / "summary.json", {"complete": True, "samples_per_task": 120, "tasks": len(results),
                "heldout_total": sum(r["original"]["canonical"]["samples"] for r in results.values()),
                "results": results, "generated_tokens": 0, "test_used_for_training_or_selection": False,
                "interpretation": "Small custom held-out comparison, not full official benchmark scores. One fixed permutation per example. Raw choice probabilities are not guaranteed calibrated."})

if __name__ == "__main__":
    main()
