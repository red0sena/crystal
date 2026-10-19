"""Separate choice content, displayed position, answer label, and prompt example effects."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import string
import time
from unittest.mock import patch

import torch

from qwen_decision.core.scorer import ChoiceScorer, SYSTEM_PROMPT, request_messages
from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, make_batches, sha256
from qwen_decision.evaluate.evaluate_choice_benchmarks import DATA_ROOT, SPECS, load_benchmark
from qwen_decision.paths import code_path


SEED = 42
CASES = ["shuffle_contents", "shuffle_labels", "neutral_letters", "neutral_shuffle_letters", "neutral_shuffle_numbers"]
NEUTRAL_SYSTEM = SYSTEM_PROMPT.replace(
    "Reply with only its uppercase letter (for example A).", "Reply with only its option label.",
)


def permutation(benchmark, source_id, kind, count):
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{benchmark}:{source_id}:{kind}".encode()).digest()[:8], "big")
    values = list(range(count))
    random.Random(seed).shuffle(values)
    return values


def transform(question, case):
    count = question["option_count"]
    indices = permutation(question["category"], question["source_id"], "content", count) if case == "shuffle_contents" else list(range(count))
    shuffled_labels = case in ("shuffle_labels", "neutral_shuffle_letters", "neutral_shuffle_numbers")
    label_indices = permutation(question["category"], question["source_id"], "labels", count) if shuffled_labels else list(range(count))
    numeric = case == "neutral_shuffle_numbers"
    labels = [str(i + 1) if numeric else string.ascii_uppercase[i] for i in label_indices]
    original_options = list(question["request"]["options"].values())
    gold_position = indices.index(question["gold_index"])
    return {
        "question_id": question["question_id"], "source_id": question["source_id"], "category": question["category"],
        "option_count": count, "request": {"question": question["request"]["question"], "options": {
            label: original_options[i] for label, i in zip(labels, indices)
        }},
        "display_to_original_index": indices, "gold_original_index": question["gold_index"],
        "gold_display_index": gold_position, "gold_label": labels[gold_position],
    }


def messages(question, case):
    neutral = case.startswith("neutral_")
    system = NEUTRAL_SYSTEM if neutral else SYSTEM_PROMPT
    instruction = "Answer with only the option label." if neutral else "Answer with only the option letter."
    options = "\n".join(f"{label}. {text}" for label, text in question["request"]["options"].items())
    text = f"State:\n\n\nQuestion:\n{question['request']['question']}\n\nOptions:\n{options}\n\n{instruction}"
    return [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": text}]}]


def encode(scorer, questions, case):
    tokenizer = scorer.processor.tokenizer
    candidates = sorted({label for q in questions for label in q["request"]["options"]})
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in candidates]
    assert all(len(tokens) == 1 for tokens in ids)
    token_ids = [tokens[0] for tokens in ids]
    for question in questions:
        prompt = scorer.processor.apply_chat_template(messages(question, case), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if case == "neutral_letters":
            # Content/layout control: only the two answer instructions change.
            original = request_messages(question["request"])
            assert messages(question, "shuffle_contents") == original
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(tokens) <= scorer.max_input_tokens
        suffix = prompt[-64:]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        for label, token_id in zip(candidates, token_ids):
            assert tokenizer.encode(suffix + label, add_special_tokens=False) == suffix_ids + [token_id]
        question["input_ids"] = tokens
    tokenizer.padding_side = "left"
    return candidates, token_ids


@torch.inference_mode()
def score_batch(scorer, batch, candidates, token_ids):
    inputs = scorer.processor.tokenizer.pad([
        {"input_ids": q["input_ids"], "attention_mask": [1] * len(q["input_ids"])} for q in batch
    ], padding=True, return_tensors="pt").to(scorer.device)
    assert bool(inputs["attention_mask"][:, -1].all())
    union_logits = scorer.option_logits_batch(inputs, token_ids)
    candidate_indices = {label: i for i, label in enumerate(candidates)}
    columns = [[candidate_indices[label] for label in q["request"]["options"]] for q in batch]
    gather = torch.tensor([row + [0] * (len(candidates) - len(row)) for row in columns], device=scorer.device)
    logits = union_logits.gather(1, gather)
    valid = torch.arange(len(candidates), device=scorer.device)[None, :] < torch.tensor(
        [q["option_count"] for q in batch], device=scorer.device)[:, None]
    assert bool(torch.isfinite(logits[valid]).all())
    return logits.cpu(), logits.masked_fill(~valid, -torch.inf).softmax(-1).cpu()


def run(scorer, questions, batches, candidates, token_ids, output, manifest):
    forwards = 0

    def record_forward(*args):
        nonlocal forwards
        forwards += 1

    hook = scorer.model.model.register_forward_hook(record_forward)
    correct = 0
    label_counts, position_counts, gold_label_counts, gold_position_counts = Counter(), Counter(), Counter(), Counter()
    unique_labels, unique_positions, preferred_exposure = Counter(), Counter(), Counter()
    predictions, tie_count = {}, 0
    preferred = "1" if manifest["case"] == "neutral_shuffle_numbers" else "A"
    scorer._synchronize()
    start = last_progress = time.perf_counter()
    partial = output / "run-1.jsonl.partial"
    try:
        with partial.open("w", encoding="utf-8") as handle, patch.object(scorer.model, "generate",
                side_effect=AssertionError("No generation allowed")):
            for batch in batches:
                logits, probabilities = score_batch(scorer, batch, candidates, token_ids)
                for i, q in enumerate(batch):
                    labels = list(q["request"]["options"])
                    values, probs = logits[i, :len(labels)].tolist(), probabilities[i, :len(labels)].tolist()
                    selected = max(range(len(labels)), key=lambda j: probs[j])
                    label, original_index = labels[selected], q["display_to_original_index"][selected]
                    unique = values.count(max(values)) == 1
                    is_correct = original_index == q["gold_original_index"]
                    correct += int(is_correct)
                    tie_count += int(not unique)
                    label_counts[label] += 1
                    position_counts[selected + 1] += 1
                    gold_label_counts[q["gold_label"]] += 1
                    gold_position_counts[q["gold_display_index"] + 1] += 1
                    preferred_exposure[labels.index(preferred) + 1] += 1
                    if unique:
                        unique_labels[label] += 1
                        unique_positions[selected + 1] += 1
                    record = {"question_id": q["question_id"], "source_id": q["source_id"],
                        "display_labels": labels, "display_to_original_index": q["display_to_original_index"],
                        "prediction_label": label, "prediction_display_index": selected, "prediction_original_index": original_index,
                        "gold_label": q["gold_label"], "gold_display_index": q["gold_display_index"],
                        "gold_original_index": q["gold_original_index"], "correct": is_correct,
                        "logits": values, "probabilities": probs, "unique_maximum": unique,
                        "input_tokens": len(q["input_ids"]), "output_tokens": 0}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions[q["question_id"]] = original_index
                now = time.perf_counter()
                if now - last_progress >= 20:
                    handle.flush()
                    print(f"{manifest['benchmark']} {manifest['case']}: {len(predictions)}/{len(questions)}", flush=True)
                    last_progress = now
    finally:
        hook.remove()
    scorer._synchronize()
    assert len(predictions) == len(questions) and forwards == len(batches)
    partial.replace(output / "run-1.jsonl")
    total, unique_total = len(questions), len(questions) - tie_count
    result = {
        "benchmark": manifest["benchmark"], "case": manifest["case"], "total": total, "correct": correct,
        "accuracy_pct": 100 * correct / total, "first_position_selection_pct": 100 * position_counts[1] / total,
        "preferred_label": preferred, "preferred_label_selection_pct": 100 * label_counts[preferred] / total,
        "prediction_label_counts": dict(label_counts), "prediction_position_counts": dict(position_counts),
        "gold_label_counts": dict(gold_label_counts), "gold_position_counts": dict(gold_position_counts),
        "preferred_label_position_exposure": dict(preferred_exposure),
        "top_logit_tie_count": tie_count, "unique_maximum_count": unique_total,
        "first_position_selection_pct_excluding_ties": 100 * unique_positions[1] / unique_total if unique_total else None,
        "preferred_label_selection_pct_excluding_ties": 100 * unique_labels[preferred] / unique_total if unique_total else None,
        "generation_calls": 0, "output_tokens": 0, "forward_calls": forwards,
        "elapsed_seconds": time.perf_counter() - start,
        "prediction_sha256": hashlib.sha256(json.dumps(sorted(predictions.items())).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    atomic_json(output / "summary.json", {"manifest": manifest, "result": result, "complete": True})
    print(f"DONE {manifest['benchmark']} {manifest['case']}: accuracy {result['accuracy_pct']:.2f}%, "
          f"first {result['first_position_selection_pct']:.2f}%, {preferred} {result['preferred_label_selection_pct']:.2f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/choice-bias-controls")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*/*/run-1.jsonl")):
        parser.error("Completed runs already exist; choose another output.")
    originals = {name: load_benchmark(name) for name in SPECS}
    scorer = ChoiceScorer(str(ROOT / "model"), device="cuda")
    baseline_root = ROOT / "results/choice-benchmarks-probabilities-1run"
    actual_weights = {p.name: sha256(p) for p in (ROOT / "model").glob("*.safetensors")}
    results = []
    for name, original in originals.items():
        previous = json.loads((baseline_root / name / "manifest.json").read_text())
        assert actual_weights == previous["model_weights_sha256"]
        for filename, digest in previous["dataset_files_sha256"].items():
            assert sha256(DATA_ROOT / filename) == digest
        for case in CASES:
            questions = [transform(q, case) for q in original]
            candidates, token_ids = encode(scorer, questions, case)
            batches = make_batches(questions, previous["batch_size"], previous["batch_token_budget"])
            destination = args.output / name / case
            destination.mkdir(parents=True, exist_ok=True)
            atomic_json(destination / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
            manifest = {"benchmark": name, "case": case, "seed": SEED, "repeats": 1,
                "model_id": previous["model_id"], "model_revision": previous["model_revision"],
                "model_weights_sha256": actual_weights, "dataset_files_sha256": previous["dataset_files_sha256"],
                "dataset_revision": previous["dataset_revision"], "split": previous["split"],
                "baseline_manifest_sha256": sha256(baseline_root / name / "manifest.json"),
                "questions_sha256": sha256(destination / "questions.json"),
                "system_prompt": NEUTRAL_SYSTEM if case.startswith("neutral_") else SYSTEM_PROMPT,
                "enable_thinking": False, "training": False, "temperature": 1.0, "padding": "left",
                "dtype": str(scorer.model.dtype), "candidates": candidates, "candidate_token_ids": token_ids,
                "batch_count": len(batches), "batch_size": previous["batch_size"], "batch_token_budget": previous["batch_token_budget"],
                "code_sha256": {filename: sha256(code_path(filename)) for filename in (
                    "evaluate_choice_bias.py", "evaluate_choice_benchmarks.py", "evaluate_mmlu_pro.py", "scorer.py")},
                "tie_rule": "first displayed maximum; report selection fractions with ties excluded too"}
            atomic_json(destination / "manifest.json", manifest)
            torch.manual_seed(SEED)
            batch0 = batches[0]
            score_batch(scorer, batch0, candidates, token_ids)
            result = run(scorer, questions, batches, candidates, token_ids, destination, manifest)
            results.append(result)
            atomic_json(args.output / "summary.json", {"cases": CASES, "seed": SEED, "results": results,
                "complete": len(results) == len(SPECS) * len(CASES)})


if __name__ == "__main__":
    main()
