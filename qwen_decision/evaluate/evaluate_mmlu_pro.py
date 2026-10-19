"""MMLU-Pro, zero-shot next-token option scoring, independently repeated full passes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import string
import time
from unittest.mock import patch

import pyarrow.parquet as pq
import torch

from qwen_decision.core.scorer import ChoiceScorer, request_messages, validate_request, SYSTEM_PROMPT
from qwen_decision.paths import code_path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_questions(dataset_path: Path, limit: int | None):
    # Read only these columns: cot_content cannot enter the prompt builder.
    rows = pq.read_table(dataset_path, columns=[
        "question_id", "question", "options", "answer_index", "answer", "category",
    ]).to_pylist()
    if limit is not None:
        rows = rows[:limit]
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs in evaluation set.")
    questions = []
    for row in rows:
        gold_original = int(row["answer_index"])
        if not 0 <= gold_original < len(row["options"]):
            raise ValueError(f"Invalid gold index for question {row['question_id']}.")
        if string.ascii_uppercase[gold_original] != row["answer"]:
            raise ValueError(f"Gold answer/index mismatch for question {row['question_id']}.")
        retained = [i for i, option in enumerate(row["options"]) if option != "N/A"]
        if gold_original not in retained:
            raise ValueError("Gold answer is a removed N/A option.")
        descriptions = [row["options"][i] for i in retained]
        request = {
            "question": row["question"],
            "options": {string.ascii_uppercase[i]: option for i, option in enumerate(descriptions)},
        }
        validate_request(request, 1.0)
        questions.append({
            "question_id": int(row["question_id"]), "category": row["category"],
            "gold_index": retained.index(gold_original), "request": request,
            "option_count": len(descriptions), "original_indices": retained,
        })
    return questions


def encode_questions(scorer, questions):
    tokenizer = scorer.processor.tokenizer
    tokenizer.padding_side = "right"
    token_ids = []
    for letter in string.ascii_uppercase[:max(q["option_count"] for q in questions)]:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {letter} is not one token.")
        token_ids.append(ids[0])
    for i, question in enumerate(questions):
        prompt = scorer.processor.apply_chat_template(
            request_messages(question["request"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Validate the actual response boundary for every candidate. Only the
        # fixed template suffix is needed to establish no BPE boundary merging.
        suffix = prompt[-64:]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        for label, token_id in zip(string.ascii_uppercase, token_ids):
            if tokenizer.encode(suffix + label, add_special_tokens=False) != suffix_ids + [token_id]:
                raise ValueError("Response boundary would merge with option label.")
        if len(ids) > scorer.max_input_tokens:
            raise ValueError(f"Question {question['question_id']} exceeds input limit; no truncation allowed.")
        question["input_ids"] = ids
        if (i + 1) % 2000 == 0:
            print(f"Encoded {i + 1}/{len(questions)} questions", flush=True)
    return token_ids


def make_batches(questions, batch_size, token_budget):
    # Fixed length order minimizes padding. Reuse tokens/batch layout, never predictions.
    ordered = sorted(questions, key=lambda q: (len(q["input_ids"]), q["question_id"]))
    batches, current = [], []
    for question in ordered:
        if current and (len(current) >= batch_size or
                        len(question["input_ids"]) * (len(current) + 1) > token_budget):
            batches.append(current)
            current = []
        current.append(question)
    if current:
        batches.append(current)
    return batches


def batch_logits(scorer, batch, token_ids):
    encoded = scorer.processor.tokenizer.pad(
        [{"input_ids": q["input_ids"], "attention_mask": [1] * len(q["input_ids"])} for q in batch],
        padding=True, return_tensors="pt",
    ).to(scorer.device)
    last_positions = encoded["attention_mask"].sum(dim=1) - 1
    return scorer.option_logits_batch(encoded, token_ids, last_positions=last_positions)


def verify_batch_path(scorer, questions, token_ids, output):
    # Spread through length range to catch reading the padding position,
    # incorrect masks and any mismatch with the original processor path.
    ordered = sorted(questions, key=lambda q: len(q["input_ids"]))
    selected = [ordered[int(i * (len(ordered) - 1) / 7)] for i in range(8)]
    batched = batch_logits(scorer, selected, token_ids).cpu()
    comparisons = []
    for row, question in zip(batched, selected):
        inputs, labels, _ = scorer.prepare(question["request"])
        if inputs["input_ids"][0].tolist() != question["input_ids"]:
            raise AssertionError("Benchmark tokens differ from the original scorer.")
        single = scorer.option_logits(inputs, [l["token_id"] for l in labels], head_mode="selected").cpu()
        candidate = row[:question["option_count"]]
        torch.testing.assert_close(candidate, single, atol=0.25, rtol=0)
        comparisons.append({
            "question_id": question["question_id"], "input_tokens": len(question["input_ids"]),
            "max_abs_logit_difference": float((candidate - single).abs().max()),
            "same_argmax": bool(candidate.argmax() == single.argmax()),
        })
    atomic_json(output / "batch-validation.json", comparisons)
    print(f"Batch path validated on {len(comparisons)} lengths; "
          f"{sum(c['same_argmax'] for c in comparisons)}/{len(comparisons)} same argmax", flush=True)


def run_pass(scorer, batches, token_ids, run_number, output, total, manifest):
    torch.manual_seed(run_number - 1)
    torch.cuda.manual_seed_all(run_number - 1)
    predictions = {}
    correct = 0
    subject = defaultdict(lambda: {"correct": 0, "total": 0})
    partial = output / f"run-{run_number}.jsonl.partial"
    scorer._synchronize()
    start = time.perf_counter()
    last_progress = start
    forward_calls = 0

    def record_forward(*args):
        nonlocal forward_calls
        forward_calls += 1

    hook = scorer.model.model.register_forward_hook(record_forward)
    try:
        with partial.open("w", encoding="utf-8") as handle, patch.object(
            scorer.model, "generate", side_effect=AssertionError("Generation is forbidden in this evaluation"),
        ):
            for batch_index, batch in enumerate(batches):
                logits = batch_logits(scorer, batch, token_ids)
                valid = torch.arange(len(token_ids), device=scorer.device)[None, :] < torch.tensor(
                    [q["option_count"] for q in batch], device=scorer.device,
                )[:, None]
                if not torch.isfinite(logits[valid]).all():
                    raise RuntimeError("Non-finite valid option logits.")
                probabilities = logits.masked_fill(~valid, -torch.inf).softmax(dim=-1).cpu()
                logits_cpu = logits.cpu()
                for row_index, question in enumerate(batch):
                    count = question["option_count"]
                    probs = probabilities[row_index, :count]
                    prediction = int(probs.argmax())
                    is_correct = prediction == question["gold_index"]
                    correct += int(is_correct)
                    stats = subject[question["category"]]
                    stats["correct"] += int(is_correct)
                    stats["total"] += 1
                    record = {
                        "question_id": question["question_id"], "category": question["category"],
                        "prediction": string.ascii_uppercase[prediction],
                        "gold": string.ascii_uppercase[question["gold_index"]], "correct": is_correct,
                        "probabilities": probs.tolist(), "logits": logits_cpu[row_index, :count].tolist(),
                        "input_tokens": len(question["input_ids"]), "output_tokens": 0,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions[question["question_id"]] = prediction
                now = time.perf_counter()
                if now - last_progress >= 20 or batch_index + 1 == len(batches):
                    elapsed = now - start
                    completed = len(predictions)
                    progress = {
                        "run": run_number, "completed": completed, "total": total,
                        "correct": correct, "accuracy_so_far_pct": 100 * correct / completed,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed * (total - completed) / completed,
                    }
                    atomic_json(output / "progress.json", progress)
                    handle.flush()
                    print(f"Run {run_number}: {completed}/{total}; acc {100 * correct / completed:.2f}%; "
                          f"elapsed {elapsed:.0f}s; ETA {progress['eta_seconds']:.0f}s", flush=True)
                    last_progress = now
    finally:
        hook.remove()
    scorer._synchronize()
    if len(predictions) != total or forward_calls != len(batches):
        raise AssertionError("Incomplete pass or unexpected number of forward calls.")
    partial.replace(output / f"run-{run_number}.jsonl")
    for stats in subject.values():
        stats["accuracy_pct"] = 100 * stats["correct"] / stats["total"]
    summary = {
        "run": run_number, "seed": run_number - 1,
        "correct": correct, "total": total, "accuracy_pct": 100 * correct / total,
        "elapsed_seconds": time.perf_counter() - start,
        "output_tokens": 0, "forward_calls": forward_calls,
        "per_category": dict(sorted(subject.items())),
        "prediction_sha256": hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    atomic_json(output / f"run-{run_number}.summary.json", summary)
    print(f"COMPLETED run {run_number}: {correct}/{total} = {summary['accuracy_pct']:.4f}%", flush=True)
    return summary, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mmlu-pro-0shot-5runs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-token-budget", type=int, default=32768)
    parser.add_argument("--limit", type=int, help="Development smoke subset; omitted for full test")
    args = parser.parse_args()
    if min(args.repeats, args.batch_size, args.batch_token_budget) <= 0 or (args.limit is not None and args.limit < 8):
        parser.error("Positive repeats/batch limits are required; development limit must be >=8.")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("run-*.jsonl")):
        parser.error("Output already contains completed runs; choose a new output directory.")
    questions = load_questions(args.dataset, args.limit)
    print(f"Loaded {len(questions)} test questions across {len({q['category'] for q in questions})} categories", flush=True)
    scorer = ChoiceScorer(str(args.model), device="cuda")
    token_ids = encode_questions(scorer, questions)
    batches = make_batches(questions, args.batch_size, args.batch_token_budget)
    print(f"Prepared {len(batches)} batches; tokens min/max "
          f"{min(len(q['input_ids']) for q in questions)}/{max(len(q['input_ids']) for q in questions)}", flush=True)
    manifest = {
        "model_id": "Qwen/Qwen3.5-0.8B", "model_path": str(args.model),
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": {p.name: sha256(p) for p in sorted(args.model.glob("*.safetensors"))},
        "model_revision": (args.model / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0],
        "dataset_id": "TIGER-Lab/MMLU-Pro", "split": "test",
        "dataset_revision": (args.dataset.parent.parent / ".cache/huggingface/download/data/test-00000-of-00001.parquet.metadata").read_text().splitlines()[0],
        "dataset_sha256": sha256(args.dataset), "question_count": len(questions),
        "category_counts": dict(sorted(Counter(q["category"] for q in questions).items())),
        "protocol": "zero-shot; original option order; next-token label logits; argmax softmax; no CoT or generation",
        "training": False, "few_shot_examples": 0, "enable_thinking": False,
        "temperature_for_option_softmax": 1.0, "sampling": False,
        "repeats": args.repeats, "batch_size": args.batch_size,
        "batch_token_budget": args.batch_token_budget, "batch_count": len(batches),
        "padding": "right; gather last real token", "limit": args.limit,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_builder_sha256": sha256(code_path("scorer.py")),
        "evaluator_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__, "dtype": str(scorer.model.dtype),
        "gpu": torch.cuda.get_device_name(),
        "interpretation": "Repeated deterministic passes test reproducibility, not independent stochastic trials. Not official 5-shot CoT protocol.",
    }
    atomic_json(args.output / "manifest.json", manifest)
    verify_batch_path(scorer, questions, token_ids, args.output)
    # Warm the same shape without retaining predictions for the measured runs.
    for batch in batches[:2]:
        batch_logits(scorer, batch, token_ids)
    summaries, first_predictions, disagreements = [], None, []
    for run_number in range(1, args.repeats + 1):
        summary, predictions = run_pass(scorer, batches, token_ids, run_number, args.output, len(questions), manifest)
        summaries.append(summary)
        if first_predictions is None:
            first_predictions = predictions
        disagreements.append(sum(predictions[k] != first_predictions[k] for k in predictions))
        scores = [row["accuracy_pct"] for row in summaries]
        combined = {
            "manifest": manifest, "runs": summaries,
            "mean_accuracy_pct": statistics.mean(scores),
            "population_stddev_percentage_points": statistics.pstdev(scores),
            "prediction_disagreements_vs_run_1": disagreements,
            "complete": len(summaries) == args.repeats,
        }
        atomic_json(args.output / "summary.json", combined)
    print(json.dumps({k: v for k, v in combined.items() if k not in ["manifest", "runs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
