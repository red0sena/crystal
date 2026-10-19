"""One bounded no-generation baseline across the prepared text task mixture."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import encode_questions, make_batches, sha256, atomic_json
from qwen_decision.evaluate.evaluate_choice_probabilities import batch_probabilities, run_benchmark
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v1"
OUT = ROOT / "results/text-decisions-v1-baseline"
SAMPLES = 120

def main():
    dataset_manifest = json.loads((DATA / "manifest.json").read_text())
    assert json.loads((DATA / "audit.json").read_text())["verified"]
    assert not (OUT / "summary.json").exists(), "Preserve completed evaluations."
    assert sha256(DATA / "test.jsonl") == dataset_manifest["stats"]["test"]["file_sha256"]
    grouped = defaultdict(list)
    with (DATA / "test.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            grouped[r["task"]].append(r)
    # Neutral wording with no example of a preferred option code.
    scorer_module.SYSTEM_PROMPT = dataset_manifest["system_prompt"]
    scorer = ChoiceScorer(str(ROOT / "model"), device="cuda")
    results = []
    for task, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (r["label_provenance"] != "human_original_author" if task == "cognitive_level" else False, hashlib.sha256(("baseline:" + r["id"]).encode()).hexdigest()))
        chosen = rows[:SAMPLES]
        questions = [{
            "question_id": i, "source_id": r["id"], "category": task,
            "request": r["request"], "gold_index": r["gold_index"],
            "option_count": len(r["request"]["options"]),
        } for i, r in enumerate(chosen)]
        token_ids = encode_questions(scorer, questions)
        scorer.processor.tokenizer.padding_side = "left"
        batches = make_batches(questions, 32, 16384)
        out = OUT / task
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
        atomic_json(out / "selected-records.json", chosen)
        manifest = {
            "benchmark": task, "split": "custom held-out; deterministic subset",
            "samples": len(chosen), "sampling": "hash-based; cognitive task prioritizes human labels, remaining quota uses silver",
            "source_test_total": len(rows), "label_provenance": dict(Counter(r["label_provenance"] for r in chosen)),
            "dataset_manifest_sha256": sha256(DATA / "manifest.json"), "dataset_test_sha256": sha256(DATA / "test.jsonl"),
            "system_prompt": scorer_module.SYSTEM_PROMPT,
            "model_id": "Qwen/Qwen3.5-0.8B", "training": False, "enable_thinking": False,
            "model_weights_sha256": {p.name: sha256(p) for p in sorted((ROOT / "model").glob("*.safetensors"))},
            "protocol": "one pass; zero-shot; next-token option-code logits; choice softmax; no generated tokens; BF16; left padding",
            "option_token_ids": token_ids, "option_code_mapping": "A..Z in request insertion order",
            "code_sha256": {n: sha256(code_path(n)) for n in ("evaluate_text_decisions.py", "evaluate_mmlu_pro.py", "evaluate_choice_probabilities.py", "scorer.py")},
        }
        atomic_json(out / "manifest.json", manifest)
        print(f"START {task}: {len(chosen)} questions; {len(batches)} batches", flush=True)
        batch_probabilities(scorer, batches[0], token_ids)
        result = run_benchmark(scorer, questions, token_ids, batches, out, manifest)
        pred = [json.loads(x) for x in (out / "run-1.jsonl").read_text().splitlines()]
        chosen_by_id = {r["id"]: r for r in chosen}
        qualities = defaultdict(lambda: {"correct": 0, "total": 0})
        first = ties = 0
        for p in pred:
            provenance = chosen_by_id[p["source_id"]]["label_provenance"]
            qualities[provenance]["correct"] += p["correct"]
            qualities[provenance]["total"] += 1
            first += p["prediction"] == "A"
            ties += p["logits"].count(max(p["logits"])) > 1
        result.update({
            "random_choice_expected_accuracy_pct": 100 * sum(1 / len(r["request"]["options"]) for r in chosen) / len(chosen),
            "majority_label_baseline_pct": 100 * max(Counter(r["answer_id"] for r in chosen).values()) / len(chosen),
            "first_display_position_pct": 100 * first / len(chosen), "tied_max_count": ties,
            "by_label_provenance": dict(qualities),
        })
        atomic_json(out / "summary.json", result)
        results.append(result)
        atomic_json(OUT / "summary.json", {"complete": len(results) == len(grouped), "training": False, "samples_per_task": SAMPLES, "results": results, "interpretation": "small custom sanity baseline, not a full benchmark or a trained-model result"})
    print(json.dumps([{k: r[k] for k in ("benchmark", "correct", "total", "accuracy_pct")} for r in results], indent=2), flush=True)

if __name__ == "__main__":
    main()
