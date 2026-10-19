"""Audit the no-generation baseline's source selection and numerical scores."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v1"
OUT = ROOT / "results/text-decisions-v1-baseline"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["complete"] and not summary["training"]
    with (DATA / "test.jsonl").open() as f:
        test_rows = {r["id"]: r for r in map(json.loads, f)}
    totals = []
    for result in summary["results"]:
        task = result["benchmark"]
        folder = OUT / task
        manifest = json.loads((folder / "manifest.json").read_text())
        assert sha(DATA / "manifest.json") == manifest["dataset_manifest_sha256"]
        assert sha(DATA / "test.jsonl") == manifest["dataset_test_sha256"]
        for file, digest in manifest["code_sha256"].items():
            assert sha(code_path(file)) == digest
        for file, digest in manifest["model_weights_sha256"].items():
            assert sha(ROOT / "model" / file) == digest
        selected = json.loads((folder / "selected-records.json").read_text())
        questions = json.loads((folder / "questions.json").read_text())
        predictions = [json.loads(x) for x in (folder / "run-1.jsonl").read_text().splitlines()]
        assert len(selected) == len(questions) == len(predictions) == result["total"] == 120
        q_by_id = {q["question_id"]: q for q in questions}
        by_source = {r["id"]: r for r in selected}
        assert len(by_source) == 120
        for r in selected:
            assert test_rows[r["id"]] == r and r["task"] == task
        seen = set()
        correct = first = tied = 0
        quality = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in predictions:
            q = q_by_id[p["question_id"]]
            r = by_source[p["source_id"]]
            assert p["question_id"] not in seen
            seen.add(p["question_id"])
            assert q["request"] == r["request"] and q["gold_index"] == r["gold_index"]
            logits = np.asarray(p["logits"], dtype=np.float64)
            probabilities = np.asarray(p["probabilities"], dtype=np.float64)
            assert len(logits) == len(probabilities) == len(r["request"]["options"])
            expected = np.exp(logits - logits.max())
            expected /= expected.sum()
            np.testing.assert_allclose(probabilities, expected, atol=2e-7, rtol=0)
            assert abs(probabilities.sum() - 1) < 2e-7
            prediction = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(np.argmax(logits))]
            gold = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[r["gold_index"]]
            assert p["prediction"] == prediction and p["gold"] == gold
            assert p["correct"] == (prediction == gold) and p["output_tokens"] == 0
            correct += p["correct"]
            first += prediction == "A"
            tied += np.sum(logits == logits.max()) > 1
            quality[r["label_provenance"]]["correct"] += p["correct"]
            quality[r["label_provenance"]]["total"] += 1
        assert correct == result["correct"]
        assert result["output_tokens"] == result["generation_calls"] == 0
        assert result["forward_calls"] > 0
        assert abs(result["accuracy_pct"] - 100 * correct / 120) < 1e-10
        assert abs(result["first_display_position_pct"] - 100 * first / 120) < 1e-10
        assert tied == result["tied_max_count"]
        assert dict(quality) == result["by_label_provenance"]
        totals.append({"task": task, "correct": correct, "total": 120, "accuracy_pct": result["accuracy_pct"]})
    audit = {"verified": True, "tasks": len(totals), "total_scored": sum(r["total"] for r in totals), "output_tokens": 0, "generation_calls": 0, "original_model_weights_verified": True, "predictions_and_softmax_recomputed": True, "heldout_membership_verified": True, "results": totals}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
