"""Original Qwen greedy generation on MMLU-Pro, without reasoning or option constraints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import string
import time

import torch
from transformers import GenerationConfig

from scorer import ChoiceScorer, SYSTEM_PROMPT
from evaluate_mmlu_pro import ROOT, atomic_json, encode_questions, load_questions, make_batches, sha256


def parse_answer(response, option_count):
    # Whitespace is harmless; any prose, punctuation or multiple labels is invalid.
    answer = response.strip()
    if re.fullmatch(r"[A-J]", answer) and answer in string.ascii_uppercase[:option_count]:
        return answer
    return None


@torch.inference_mode()
def generate_batch(scorer, batch, generation_config):
    tokenizer = scorer.processor.tokenizer
    if tokenizer.padding_side != "left":
        raise AssertionError("Decoder generation requires left padding.")
    encoded = tokenizer.pad([
        {"input_ids": q["input_ids"], "attention_mask": [1] * len(q["input_ids"])}
        for q in batch
    ], padding=True, return_tensors="pt").to(scorer.device)
    sequences = scorer.model.generate(
        **encoded, generation_config=generation_config, logits_to_keep=1,
    )
    suffixes = sequences[:, encoded["input_ids"].shape[1]:].cpu().tolist()
    eos_ids = generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    results = []
    for question, suffix in zip(batch, suffixes):
        stop = next((i for i, token in enumerate(suffix) if token in eos_ids), None)
        actual_ids = suffix if stop is None else suffix[:stop + 1]
        response = tokenizer.decode(actual_ids, skip_special_tokens=True)
        results.append({
            "question_id": question["question_id"], "category": question["category"],
            "prediction": parse_answer(response, question["option_count"]),
            "gold": string.ascii_uppercase[question["gold_index"]],
            "response": response,
            "response_with_special_tokens": tokenizer.decode(actual_ids, skip_special_tokens=False),
            "generated_token_ids": actual_ids, "output_tokens": len(actual_ids),
            "input_tokens": len(question["input_ids"]), "ended_with_eos": stop is not None,
        })
    return results


def run_pass(scorer, batches, config, run_number, output, total, manifest):
    torch.manual_seed(run_number - 1)
    torch.cuda.manual_seed_all(run_number - 1)
    predictions, subject = {}, defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0})
    correct = invalid = output_tokens = forward_calls = generation_calls = 0

    def record_forward(*args):
        nonlocal forward_calls
        forward_calls += 1

    hook = scorer.model.model.register_forward_hook(record_forward)
    scorer._synchronize()
    start = last_progress = time.perf_counter()
    partial = output / f"run-{run_number}.jsonl.partial"
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for batch_index, batch in enumerate(batches):
                records = generate_batch(scorer, batch, config)
                generation_calls += 1
                for record in records:
                    record["correct"] = record["prediction"] == record["gold"]
                    record["valid_format"] = record["prediction"] is not None
                    correct += int(record["correct"])
                    invalid += int(not record["valid_format"])
                    output_tokens += record["output_tokens"]
                    stats = subject[record["category"]]
                    stats["correct"] += int(record["correct"])
                    stats["invalid"] += int(not record["valid_format"])
                    stats["total"] += 1
                    predictions[record["question_id"]] = record["prediction"]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                now = time.perf_counter()
                if now - last_progress >= 20 or batch_index + 1 == len(batches):
                    completed, elapsed = len(predictions), now - start
                    progress = {
                        "run": run_number, "completed": completed, "total": total,
                        "correct": correct, "invalid": invalid,
                        "accuracy_so_far_pct": 100 * correct / completed,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed * (total - completed) / completed,
                    }
                    atomic_json(output / "progress.json", progress)
                    handle.flush()
                    print(f"Run {run_number}: {completed}/{total}; acc {100 * correct / completed:.2f}%; "
                          f"invalid {invalid}; elapsed {elapsed:.0f}s; ETA {progress['eta_seconds']:.0f}s", flush=True)
                    last_progress = now
    finally:
        hook.remove()
    scorer._synchronize()
    if len(predictions) != total or generation_calls != len(batches) or forward_calls < generation_calls:
        raise AssertionError("Incomplete pass or generation was not performed.")
    partial.replace(output / f"run-{run_number}.jsonl")
    for stats in subject.values():
        stats["accuracy_pct"] = 100 * stats["correct"] / stats["total"]
    summary = {
        "run": run_number, "seed": run_number - 1, "total": total, "correct": correct,
        "accuracy_pct": 100 * correct / total, "invalid_format_count": invalid,
        "elapsed_seconds": time.perf_counter() - start,
        "output_tokens": output_tokens, "generation_calls": generation_calls, "forward_calls": forward_calls,
        "per_category": dict(sorted(subject.items())),
        "prediction_sha256": hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    atomic_json(output / f"run-{run_number}.summary.json", summary)
    print(f"COMPLETED run {run_number}: {correct}/{total} = {summary['accuracy_pct']:.4f}%; invalid {invalid}", flush=True)
    return summary, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mmlu-pro-generate-5runs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-token-budget", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.repeats, args.batch_size, args.batch_token_budget, args.max_new_tokens) <= 0:
        parser.error("All limits must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("run-*.jsonl")):
        parser.error("Output already has completed runs; use a new directory.")
    questions = load_questions(args.dataset, args.limit)
    scorer = ChoiceScorer(str(args.model), device="cuda")
    encode_questions(scorer, questions)
    scorer.processor.tokenizer.padding_side = "left"
    batches = make_batches(questions, args.batch_size, args.batch_token_budget)
    config = GenerationConfig.from_model_config(scorer.model.config)
    config.do_sample = False
    config.num_beams = 1
    config.max_new_tokens = args.max_new_tokens
    config.use_cache = True
    config.pad_token_id = scorer.processor.tokenizer.pad_token_id
    # The chat tokenizer ends an assistant reply at <|im_end|>; the model
    # config's <|endoftext|> alone would generate beyond the reply boundary.
    config.eos_token_id = scorer.processor.tokenizer.eos_token_id
    manifest = {
        "model_id": "Qwen/Qwen3.5-0.8B", "model_path": str(args.model),
        "model_revision": (args.model / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0],
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": {p.name: sha256(p) for p in sorted(args.model.glob("*.safetensors"))},
        "dataset_id": "TIGER-Lab/MMLU-Pro", "split": "test",
        "dataset_revision": (args.dataset.parent.parent / ".cache/huggingface/download/data/test-00000-of-00001.parquet.metadata").read_text().splitlines()[0],
        "dataset_sha256": sha256(args.dataset), "question_count": len(questions),
        "category_counts": dict(sorted(Counter(q["category"] for q in questions).items())),
        "protocol": "zero-shot; identical prompts/options to prior scoring; unconstrained greedy full-vocabulary generation; no reasoning",
        "training": False, "few_shot_examples": 0, "enable_thinking": False,
        "sampling": False, "generation_config": config.to_dict(),
        "answer_parser": "strip whitespace; exactly one valid uppercase A-J label; other outputs count as wrong",
        "repeats": args.repeats, "batch_size": args.batch_size,
        "batch_token_budget": args.batch_token_budget, "batch_count": len(batches),
        "padding": "left for autoregressive generation", "limit": args.limit,
        "system_prompt": SYSTEM_PROMPT, "prompt_builder_sha256": sha256(ROOT / "scorer.py"),
        "shared_evaluator_sha256": sha256(ROOT / "evaluate_mmlu_pro.py"),
        "evaluator_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__, "dtype": str(scorer.model.dtype),
        "gpu": torch.cuda.get_device_name(),
        "interpretation": "Five actual deterministic passes, no cached predictions. Not official 5-shot CoT evaluation. Padding differs from prior option scoring; BF16 differences may affect close logits.",
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(f"Prepared {len(questions)} questions; {len(batches)} batches; {config.max_new_tokens} output-token cap", flush=True)
    for batch in batches[:2]:
        generate_batch(scorer, batch, config)
    summaries, first_predictions, disagreements = [], None, []
    for run_number in range(1, args.repeats + 1):
        summary, predictions = run_pass(scorer, batches, config, run_number, args.output, len(questions), manifest)
        summaries.append(summary)
        if first_predictions is None:
            first_predictions = predictions
        disagreements.append(sum(predictions[k] != first_predictions[k] for k in predictions))
        scores = [s["accuracy_pct"] for s in summaries]
        combined = {
            "manifest": manifest, "runs": summaries, "mean_accuracy_pct": statistics.mean(scores),
            "population_stddev_percentage_points": statistics.pstdev(scores),
            "prediction_disagreements_vs_run_1": disagreements, "complete": len(summaries) == args.repeats,
        }
        atomic_json(args.output / "summary.json", combined)
    print(json.dumps({k: v for k, v in combined.items() if k not in ("manifest", "runs")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
