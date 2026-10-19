"""Cross independently shuffled content and labels under the same neutral instructions."""

import argparse
import hashlib
import json
from pathlib import Path

from scorer import ChoiceScorer
from evaluate_mmlu_pro import ROOT, atomic_json, make_batches, sha256
from evaluate_choice_benchmarks import DATA_ROOT, SPECS, load_benchmark
from evaluate_choice_bias import SEED, NEUTRAL_SYSTEM, transform, encode, score_batch, run


def cross_transform(original, case):
    content = transform(original, "shuffle_contents")
    label_layout = transform(original, case)
    labels = list(label_layout["request"]["options"])
    descriptions = list(content["request"]["options"].values())
    content["request"]["options"] = dict(zip(labels, descriptions))
    content["gold_label"] = labels[content["gold_display_index"]]
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/choice-bias-crossed")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*/*/run-1.jsonl")):
        parser.error("Completed results exist; choose another directory.")
    scorer = ChoiceScorer(str(ROOT / "model"), device="cuda")
    baseline = ROOT / "results/choice-benchmarks-probabilities-1run"
    weights = {p.name: sha256(p) for p in (ROOT / "model").glob("*.safetensors")}
    cases = ["neutral_shuffle_letters", "neutral_shuffle_numbers"]
    results = []
    for name in SPECS:
        originals = load_benchmark(name)
        prior = json.loads((baseline / name / "manifest.json").read_text())
        assert weights == prior["model_weights_sha256"]
        for filename, digest in prior["dataset_files_sha256"].items():
            assert sha256(DATA_ROOT / filename) == digest
        for case in cases:
            questions = [cross_transform(q, case) for q in originals]
            candidates, token_ids = encode(scorer, questions, case)
            batches = make_batches(questions, prior["batch_size"], prior["batch_token_budget"])
            output = args.output / name / case
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "questions.json", [{k: v for k, v in q.items() if k != "input_ids"} for q in questions])
            manifest = {"benchmark": name, "case": case, "experiment": "independently shuffled content and labels",
                "seed": SEED, "repeats": 1, "model_id": prior["model_id"], "model_revision": prior["model_revision"],
                "model_weights_sha256": weights, "dataset_files_sha256": prior["dataset_files_sha256"],
                "dataset_revision": prior["dataset_revision"], "split": prior["split"],
                "baseline_manifest_sha256": sha256(baseline / name / "manifest.json"),
                "questions_sha256": sha256(output / "questions.json"), "system_prompt": NEUTRAL_SYSTEM,
                "enable_thinking": False, "training": False, "temperature": 1.0, "padding": "left",
                "dtype": str(scorer.model.dtype), "candidates": candidates, "candidate_token_ids": token_ids,
                "batch_count": len(batches), "batch_size": prior["batch_size"], "batch_token_budget": prior["batch_token_budget"],
                "code_sha256": {filename: sha256(ROOT / filename) for filename in (
                    "evaluate_choice_bias_crossed.py", "evaluate_choice_bias.py", "evaluate_choice_benchmarks.py",
                    "evaluate_mmlu_pro.py", "scorer.py")},
                "tie_rule": "first displayed maximum; report selection fractions with ties excluded too"}
            atomic_json(output / "manifest.json", manifest)
            score_batch(scorer, batches[0], candidates, token_ids)
            results.append(run(scorer, questions, batches, candidates, token_ids, output, manifest))
            atomic_json(args.output / "summary.json", {"cases": cases, "seed": SEED,
                "experiment": "independently shuffled content and labels", "results": results,
                "complete": len(results) == len(SPECS) * len(cases)})


if __name__ == "__main__":
    main()
