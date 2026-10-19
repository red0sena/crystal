"""One no-generation option-probability pass on the same three choice benchmarks."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import string
import time
from unittest.mock import patch

import torch

from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, encode_questions, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_benchmarks import DATA_ROOT, SPECS, load_benchmark
from qwen_decision.paths import code_path


@torch.inference_mode()
def batch_probabilities(scorer, batch, token_ids):
    tokenizer = scorer.processor.tokenizer
    assert tokenizer.padding_side == "left"
    inputs = tokenizer.pad([
        {"input_ids": q["input_ids"], "attention_mask": [1] * len(q["input_ids"])}
        for q in batch
    ], padding=True, return_tensors="pt").to(scorer.device)
    # The final position is real in every left-padded row, exactly as in the
    # earlier generation evaluation. Only choice rows of the LM Head are used.
    assert bool(inputs["attention_mask"][:, -1].all())
    logits = scorer.option_logits_batch(inputs, token_ids)
    valid = torch.arange(len(token_ids), device=scorer.device)[None, :] < torch.tensor(
        [q["option_count"] for q in batch], device=scorer.device,
    )[:, None]
    assert bool(torch.isfinite(logits[valid]).all())
    probabilities = logits.masked_fill(~valid, -torch.inf).softmax(dim=-1)
    return logits.cpu(), probabilities.cpu()


def run_benchmark(scorer, questions, token_ids, batches, output, manifest):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    predictions = {}
    correct = forwards = 0

    def count_forward(*args):
        nonlocal forwards
        forwards += 1

    hook = scorer.model.model.register_forward_hook(count_forward)
    scorer._synchronize()
    start = last_progress = time.perf_counter()
    partial = output / "run-1.jsonl.partial"
    try:
        with partial.open("w", encoding="utf-8") as handle, patch.object(
            scorer.model, "generate", side_effect=AssertionError("Generation is forbidden in probability evaluation"),
        ):
            for batch_index, batch in enumerate(batches):
                logits, probabilities = batch_probabilities(scorer, batch, token_ids)
                for index, question in enumerate(batch):
                    count = question["option_count"]
                    probs = probabilities[index, :count]
                    prediction = string.ascii_uppercase[int(probs.argmax())]
                    gold = string.ascii_uppercase[question["gold_index"]]
                    is_correct = prediction == gold
                    correct += int(is_correct)
                    predictions[question["question_id"]] = prediction
                    record = {
                        "question_id": question["question_id"], "source_id": question["source_id"],
                        "category": question["category"], "prediction": prediction, "gold": gold,
                        "correct": is_correct, "option_labels": list(question["request"]["options"]),
                        "probabilities": probs.tolist(), "logits": logits[index, :count].tolist(),
                        "input_tokens": len(question["input_ids"]), "output_tokens": 0,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                now = time.perf_counter()
                if now - last_progress >= 20 or batch_index + 1 == len(batches):
                    handle.flush()
                    atomic_json(output / "progress.json", {
                        "completed": len(predictions), "total": len(questions), "correct": correct,
                        "elapsed_seconds": now - start,
                    })
                    print(f"{manifest['benchmark']}: {len(predictions)}/{len(questions)}; "
                          f"acc {100 * correct / len(predictions):.2f}%", flush=True)
                    last_progress = now
    finally:
        hook.remove()
    scorer._synchronize()
    assert len(predictions) == len(questions) and forwards == len(batches)
    partial.replace(output / "run-1.jsonl")
    summary = {
        "benchmark": manifest["benchmark"], "split": manifest["split"], "run": 1, "seed": 0,
        "total": len(questions), "correct": correct, "accuracy_pct": 100 * correct / len(questions),
        "output_tokens": 0, "generation_calls": 0, "forward_calls": forwards,
        "elapsed_seconds": time.perf_counter() - start,
        "prediction_sha256": hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    atomic_json(output / "run-1.summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "results/choice-benchmarks-probabilities-1run")
    parser.add_argument("--previous", type=Path, default=ROOT / "results/choice-benchmarks-generate-1run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*/run-*.jsonl")):
        parser.error("Completed runs exist; choose a new output directory.")
    scorer = ChoiceScorer(str(args.model), device="cuda")
    weight_hashes = {p.name: sha256(p) for p in sorted(args.model.glob("*.safetensors"))}
    results = []
    for name, spec in SPECS.items():
        prior = json.loads((args.previous / name / "manifest.json").read_text())
        assert prior["model_weights_sha256"] == weight_hashes
        assert prior["model_config_sha256"] == sha256(args.model / "config.json")
        questions = load_benchmark(name)
        token_ids = encode_questions(scorer, questions)
        scorer.processor.tokenizer.padding_side = "left"
        batches = make_batches(questions, prior["batch_size"], prior["batch_token_budget"])
        output = args.output / name
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
        assert sha256(output / "questions.json") == prior["questions_sha256"]
        assert len(batches) == prior["batch_count"] and prior["dtype"] == str(scorer.model.dtype)
        for filename, digest in prior["dataset_files_sha256"].items():
            assert sha256(DATA_ROOT / filename) == digest
        manifest = {k: v for k, v in prior.items() if k not in ("generation_config", "answer_parser", "code_sha256", "interpretation")}
        manifest.update({
            "protocol": "zero-shot; identical prompts, choice order, left padding and batches; next-token option logits; softmax argmax; no generation",
            "head_mode": "selected", "temperature_for_option_softmax": 1.0, "use_cache": False,
            "option_token_ids": token_ids, "previous_results": str(args.previous),
            "code_sha256": {filename: sha256(code_path(filename)) for filename in (
                "evaluate_choice_probabilities.py", "evaluate_choice_benchmarks.py", "evaluate_mmlu_pro.py", "scorer.py",
            )},
            "batch_layout_sha256": hashlib.sha256(json.dumps([[q["question_id"] for q in batch] for batch in batches]).encode()).hexdigest(),
            "interpretation": "Choice-normalized scores, not calibrated correctness probabilities. Selected/full vocabulary BF16 projection or cache settings may affect close logits.",
        })
        atomic_json(output / "manifest.json", manifest)
        print(f"START {name}: {len(questions)} questions, {len(batches)} batches", flush=True)
        # Warmup is separate from the single measured full pass.
        batch_probabilities(scorer, batches[0], token_ids)
        result = run_benchmark(scorer, questions, token_ids, batches, output, manifest)
        results.append(result)
        atomic_json(output / "summary.json", {"manifest": manifest, "runs": [result], "complete": True})
        atomic_json(args.output / "summary.json", {
            "model_id": "Qwen/Qwen3.5-0.8B", "repeats_per_benchmark": 1,
            "protocol": manifest["protocol"], "results": results, "complete": len(results) == len(SPECS),
        })
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
