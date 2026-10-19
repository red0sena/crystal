"""One zero-shot, no-reasoning generation pass on ARC-Challenge, CSQA and PIQA."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import string

import pyarrow.parquet as pq
import torch
from transformers import GenerationConfig

from qwen_decision.core.scorer import ChoiceScorer, SYSTEM_PROMPT, validate_request
from qwen_decision.evaluate.evaluate_mmlu_pro import ROOT, atomic_json, encode_questions, make_batches, sha256
from qwen_decision.evaluate.evaluate_mmlu_pro_generate import generate_batch, run_pass
from qwen_decision.paths import code_path


DATA_ROOT = ROOT / "data/choice-benchmarks"
SPECS = {
    "ARC-Challenge": {
        "dataset_id": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "expected_count": 1172,
        "files": ["allenai__ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"],
    },
    "CommonsenseQA": {
        "dataset_id": "tau/commonsense_qa", "config": "default", "split": "validation", "expected_count": 1221,
        "files": ["tau__commonsense_qa/data/validation-00000-of-00001.parquet"],
    },
    "PIQA": {
        "dataset_id": "ybisk/piqa", "config": "plain_text", "split": "validation", "expected_count": 1838,
        "files": ["ybisk__piqa/dev.jsonl", "ybisk__piqa/dev-labels.lst"],
    },
}


def load_benchmark(name, data_root=DATA_ROOT):
    spec = SPECS[name]
    if name == "PIQA":
        rows = [json.loads(line) for line in (data_root / spec["files"][0]).read_text().splitlines()]
        labels = [int(line) for line in (data_root / spec["files"][1]).read_text().splitlines()]
        if len(rows) != len(labels):
            raise ValueError("PIQA question/label count mismatch.")
        examples = [(str(i), row["goal"], [row["sol1"], row["sol2"]], ["0", "1"], str(label))
                    for i, (row, label) in enumerate(zip(rows, labels))]
    else:
        # Select only question/options/labels. Auxiliary concepts and answers
        # are never part of the request or model prompt.
        rows = pq.read_table(data_root / spec["files"][0], columns=["id", "question", "choices", "answerKey"]).to_pylist()
        examples = [(row["id"], row["question"], row["choices"]["text"], row["choices"]["label"], row["answerKey"])
                    for row in rows]
    if len(examples) != spec["expected_count"]:
        raise ValueError(f"Unexpected {name} size: {len(examples)}")
    if len({row[0] for row in examples}) != len(examples):
        raise ValueError("Duplicate source IDs.")
    questions = []
    for index, (source_id, question, descriptions, original_labels, answer) in enumerate(examples):
        if len(descriptions) != len(original_labels) or len(set(original_labels)) != len(original_labels):
            raise ValueError("Malformed choices.")
        if answer not in original_labels:
            raise ValueError("Missing gold answer; never evaluate unlabeled test rows.")
        request = {"question": question, "options": {
            string.ascii_uppercase[i]: text for i, text in enumerate(descriptions)
        }}
        validate_request(request, 1.0)
        questions.append({
            "question_id": index, "source_id": source_id, "category": name,
            "request": request, "gold_index": original_labels.index(answer),
            "option_count": len(descriptions), "original_choice_labels": original_labels,
        })
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "results/choice-benchmarks-generate-1run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-token-budget", type=int, default=32768)
    args = parser.parse_args()
    if min(args.batch_size, args.batch_token_budget) <= 0:
        parser.error("Batch limits must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*/run-*.jsonl")):
        parser.error("Completed results already exist; use a new output directory.")
    datasets = {name: load_benchmark(name) for name in SPECS}
    print({name: len(questions) for name, questions in datasets.items()}, flush=True)
    scorer = ChoiceScorer(str(args.model), device="cuda")
    config = GenerationConfig.from_model_config(scorer.model.config)
    config.do_sample = False
    config.num_beams = 1
    config.max_new_tokens = 8
    config.use_cache = True
    config.pad_token_id = scorer.processor.tokenizer.pad_token_id
    config.eos_token_id = scorer.processor.tokenizer.eos_token_id
    model_manifest = {
        "model_id": "Qwen/Qwen3.5-0.8B", "model_path": str(args.model),
        "model_revision": (args.model / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0],
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": {p.name: sha256(p) for p in sorted(args.model.glob("*.safetensors"))},
        "training": False, "few_shot_examples": 0, "enable_thinking": False, "sampling": False,
        "protocol": "zero-shot; original choice order; unconstrained full-vocabulary greedy generation; answer label only",
        "system_prompt": SYSTEM_PROMPT, "generation_config": config.to_dict(),
        "answer_parser": "strip whitespace; exactly one valid uppercase option label; invalid responses count as wrong",
        "padding": "left", "repeats": 1,
        "batch_size": args.batch_size, "batch_token_budget": args.batch_token_budget,
        "dtype": str(scorer.model.dtype), "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "code_sha256": {filename: sha256(code_path(filename)) for filename in (
            "evaluate_choice_benchmarks.py", "evaluate_mmlu_pro_generate.py", "evaluate_mmlu_pro.py", "scorer.py",
        )},
        "interpretation": "Custom direct-answer generation protocol; not standard completion likelihood scoring or few-shot leaderboard evaluation.",
    }
    results = []
    for name, questions in datasets.items():
        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        encode_questions(scorer, questions)
        scorer.processor.tokenizer.padding_side = "left"
        batches = make_batches(questions, args.batch_size, args.batch_token_budget)
        spec = SPECS[name]
        metadata = json.loads((DATA_ROOT / (spec["dataset_id"].replace("/", "__") + ".metadata.json")).read_text())
        # Save the normalized question/choice mapping for audit, never pass
        # this file or any gold fields to model.generate().
        atomic_json(destination / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
        manifest = {
            **model_manifest, "benchmark": name, "dataset_id": spec["dataset_id"],
            "config": spec["config"], "split": spec["split"], "dataset_revision": metadata["revision"],
            "dataset_files_sha256": {filename: sha256(DATA_ROOT / filename) for filename in spec["files"]},
            "question_count": len(questions), "batch_count": len(batches),
            "option_count_distribution": dict(sorted(Counter(q["option_count"] for q in questions).items())),
            "input_tokens_min_max": [min(len(q["input_ids"]) for q in questions), max(len(q["input_ids"]) for q in questions)],
            "questions_sha256": sha256(destination / "questions.json"),
        }
        if name == "PIQA":
            manifest["source_url"] = metadata["source_url"]
            manifest["archive_sha256"] = metadata["archive_sha256"]
        atomic_json(destination / "manifest.json", manifest)
        print(f"START {name}: {len(questions)} questions; {len(batches)} batches", flush=True)
        generate_batch(scorer, batches[0], config)
        summary, _ = run_pass(scorer, batches, config, 1, destination, len(questions), manifest)
        atomic_json(destination / "summary.json", {"manifest": manifest, "runs": [summary], "complete": True})
        results.append({"benchmark": name, "split": spec["split"], **summary})
        atomic_json(args.output / "summary.json", {
            "model_id": model_manifest["model_id"], "protocol": model_manifest["protocol"],
            "repeats_per_benchmark": 1, "results": results, "complete": len(results) == len(SPECS),
        })
    print(json.dumps([{k: r[k] for k in ("benchmark", "split", "correct", "total", "accuracy_pct", "invalid_format_count")}
                      for r in results], indent=2), flush=True)


if __name__ == "__main__":
    main()
