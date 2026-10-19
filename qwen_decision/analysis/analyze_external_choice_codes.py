"""Describe choice-code effects without rerunning or changing the evaluation."""
from collections import Counter
import json
from pathlib import Path
import string

from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/external-classification-v1"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["complete"]
    training_counts = Counter()
    with (ROOT / "data/text-decisions-v1/train.jsonl").open() as handle:
        for line in handle:
            training_counts[len(json.loads(line)["request"]["options"])] += 1
    maximum = max(training_counts)
    assert maximum == 16
    codes = string.ascii_uppercase + string.ascii_lowercase
    groups = {
        "A-P: alphabetic codes used during current fine-tuning": codes[:maximum],
        "Q-Z: additional uppercase codes": codes[maximum:26],
        "a-s: additional lowercase codes": codes[26:45],
    }
    output = {}
    for model in ("original", "trained"):
        path = OUT / model / "KMMLU_domain/predictions.jsonl"
        gold = Counter()
        correct = Counter()
        predicted = Counter()
        for line in path.open():
            p = json.loads(line)
            code = codes[p["option_labels"].index(p["gold"])]
            gold[code] += 1
            correct[code] += int(p["correct"])
            predicted[p["prediction_code"]] += 1
        output[model] = {
            "predictions_sha256": sha256(path),
            "by_gold_code": {
                code: {"support": gold[code], "correct": correct[code],
                       "accuracy_pct": 100 * correct[code] / gold[code],
                       "predicted_count": predicted[code]}
                for code in codes[:45]
            },
            "by_code_group": {
                name: {
                    "support": sum(gold[c] for c in members),
                    "correct": sum(correct[c] for c in members),
                    "accuracy_pct": 100 * sum(correct[c] for c in members) / sum(gold[c] for c in members),
                    "predicted_count": sum(predicted[c] for c in members),
                    "prediction_share_pct": 100 * sum(predicted[c] for c in members) / sum(predicted.values()),
                }
                for name, members in groups.items()
            },
        }
    result = {
        "training_maximum_option_count": maximum,
        "training_option_count_distribution": dict(sorted(training_counts.items())),
        "results": output,
        "interpretation": "Descriptive analysis of the frozen single-pass evaluation; code and position are confounded. No causal attribution without a separate randomized-code evaluation. Alphabetic fine-tuning used at most A-P; numeric codes were also used for <=9 options. Base pretraining exposure is unknown.",
        "code_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUT / "code-position-analysis.json", result)
    print(json.dumps({m: r["by_code_group"] for m, r in output.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
