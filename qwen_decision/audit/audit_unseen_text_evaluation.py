"""Independently check source labels, matched prompts, and saved score outputs."""
from __future__ import annotations
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, sha256
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unseen-text-v1"
OUT = ROOT / "results/unseen-text-v1"

def read_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]

def main():
    data = json.loads((DATA / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["complete"]
    total = 0
    report = []
    for spec in data["datasets"]:
        name = spec["name"]
        source = spec["source"]
        for entry in source["files"]:
            assert sha256(ROOT / entry["path"]) == entry["sha256"]
        assert sha256(Path(spec["file"])) == spec["sha256"]
        raw = pq.read_table(DATA / "raw" / name / source["file"]).to_pylist()
        assert len(raw) == spec["source_total"]
        records = read_lines(Path(spec["file"]))
        by_id = {r["id"]: r for r in records}
        assert len(by_id) == len(records)
        ob_by_id = {str(r["id"]): r for r in raw} if name == "OpenBookQA" else None
        for r in records:
            row_id = r["source"]["row_id"]
            original = ob_by_id[row_id] if ob_by_id is not None else raw[int(row_id)]
            if name == "HellaSwag":
                assert r["request"]["state"] == original["ctx"]
                assert list(r["request"]["options"].values()) == original["endings"]
                gold = int(original["label"])
            elif name == "WinoGrande":
                assert r["request"]["state"] == original["sentence"]
                assert list(r["request"]["options"].values()) == [original["option1"], original["option2"]]
                gold = {"1": 0, "2": 1}[original["answer"]]
            else:
                assert r["request"]["question"] == original["question_stem"] and r["request"]["state"] == ""
                assert list(r["request"]["options"].values()) == original["choices"]["text"]
                gold = next(i for i, code in enumerate(original["choices"]["label"]) if code == original["answerKey"])
            assert r["gold_index"] == gold
            assert list(r["request"]["options"])[gold] == r["answer_id"]
        for model_name in ("original", "trained"):
            d = OUT / model_name / name
            manifest = json.loads((d / "manifest.json").read_text())
            assert sha256(d / "questions.json") == manifest["questions_sha256"]
            for file, digest in manifest["evaluation_code_sha256"].items():
                assert sha256(code_path(file)) == digest
            for file, digest in manifest["model_weights_sha256"].items():
                assert sha256(Path(manifest["model"]) / file) == digest
            predictions = read_lines(d / "run-1.jsonl")
            assert {r["source_id"] for r in predictions} == set(by_id)
            assert len(predictions) == len(by_id)
            correct = 0
            for p in predictions:
                r = by_id[p["source_id"]]
                logits = p["logits"]
                assert len(logits) == r["option_count"]
                assert all(math.isfinite(z) for z in logits)
                scores = [math.exp(z - max(logits)) for z in logits]
                probabilities = [v / sum(scores) for v in scores]
                assert all(abs(a-b) < 2e-6 for a,b in zip(probabilities, p["probabilities"]))
                index = max(range(len(logits)), key=lambda i: logits[i])
                assert p["prediction"] == chr(ord("A") + index)
                assert p["gold"] == chr(ord("A") + r["gold_index"])
                assert p["correct"] == (index == r["gold_index"])
                assert p["output_tokens"] == 0
                correct += int(p["correct"])
            measured = summary["results"][name][model_name]
            assert measured["correct"] == correct and measured["total"] == len(records)
            assert measured["generation_calls"] == 0
            assert measured["forward_calls"] == len(manifest["batch_layout"])
            assert abs(measured["accuracy_pct"] - 100 * correct / len(records)) < 1e-9
        assert sha256(OUT / "original" / name / "questions.json") == sha256(OUT / "trained" / name / "questions.json")
        total += len(records)
        report.append({"dataset": name, "labels_checked": len(records), "predictions_checked": 2 * len(records), "matched_prompts": True, "generation_calls": 0})
    atomic_json(OUT / "audit.json", {"verified": True, "source_labels_checked": total, "predictions_checked": 2 * total,
                "results": report, "dataset_overlap_audit_sha256": sha256(DATA / "audit.json"), "audit_code_sha256": sha256(Path(__file__))})
    print(json.dumps({"verified": True, "source_labels_checked": total, "predictions_checked": 2 * total}))

if __name__ == "__main__":
    main()
