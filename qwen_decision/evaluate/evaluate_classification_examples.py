"""One fixed unseen example per official domain/difficulty class, plus broad labels."""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import torch

import qwen_decision.core.scorer as scorer_module
from qwen_decision.core.scorer import ChoiceScorer
from qwen_decision.evaluate.evaluate_mmlu_pro import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v1"
OUT = ROOT / "results/classification-examples-v1"

def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

def selection_key(r):
    return hashlib.sha256(("classification-demo-v1:" + r["id"]).encode()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "examples.json").exists(), "Preserve previous examples and results."
    dataset_manifest = json.loads((DATA / "manifest.json").read_text())
    assert sha256(DATA / "test.jsonl") == dataset_manifest["stats"]["test"]["file_sha256"]
    assert sha256(DATA / "train.jsonl") == dataset_manifest["stats"]["train"]["file_sha256"]
    assert json.loads((DATA / "audit.json").read_text())["verified"]
    train = read(DATA / "train.jsonl")
    training_groups = {r["group_id"] for r in train}
    training_content = {r["base_content_sha256"] for r in train}
    previous_ids = {r["id"] for task in ("academic_domain", "relative_difficulty") for r in json.loads((ROOT / "results/text-decisions-v1-baseline" / task / "selected-records.json").read_text())}
    grouped = defaultdict(list)
    for r in read(DATA / "test.jsonl"):
        if r["task"] in ("academic_domain", "relative_difficulty") and r["id"] not in previous_ids:
            assert r["group_id"] not in training_groups
            assert r["base_content_sha256"] not in training_content
            grouped[(r["task"], r["answer_id"])].append(r)
    examples = []
    for (task, label), rows in sorted(grouped.items()):
        # Selection is fixed before either model is loaded; no cherry-picking.
        r = min(rows, key=selection_key)
        examples.append({"id": r["id"], "task": task, "expected": label, "request": r["request"],
                         "source": r["source"], "annotation": r.get("annotation"), "label_provenance": r["label_provenance"],
                         "heldout_content_sha256": r["base_content_sha256"]})
    assert len([r for r in examples if r["task"] == "academic_domain"]) == 14
    assert len([r for r in examples if r["task"] == "relative_difficulty"]) == 3
    # Independently verify metadata against the original source snapshots.
    mmlu = pq.read_table(ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet", columns=["question_id", "question", "category"]).to_pylist()
    mmlu_by_id = {str(r["question_id"]): r for r in mmlu}
    amc = pq.read_table(DATA / "raw/furonghuang-lab__Easy2Hard-Bench/E2H-AMC/eval-00000-of-00001.parquet", columns=["problem", "rating"]).to_pylist()
    cutoffs = json.loads((DATA / "difficulty_rubric.json").read_text())["cutoffs"]
    for e in examples:
        row_id = e["source"]["row_id"]
        if e["task"] == "academic_domain":
            row = mmlu_by_id[row_id]
            assert e["expected"] == row["category"]
            assert e["request"]["state"].startswith(row["question"] + "\n\nAnswer candidates:\n")
        else:
            row = amc[int(row_id)]
            assert e["request"]["state"] == row["problem"]
            assert row["rating"] == e["annotation"]["original_rating"]
            label = "low" if row["rating"] < cutoffs[0] else "medium" if row["rating"] < cutoffs[1] else "high"
            assert e["expected"] == label
    # Requested coarse labels are a separate manually mapped demo. 'Everyday'
    # is not an official MMLU-Pro category and must not be called a gold label.
    domains = {r["expected"]: r for r in examples if r["task"] == "academic_domain"}
    hs = pq.read_table(ROOT / "data/unseen-text-v1/raw/HellaSwag/data/validation-00000-of-00001.parquet").to_pylist()
    daily_index = next(i for i, r in enumerate(hs) if r["activity_label"] == "Making a sandwich")
    daily = next(r for r in read(ROOT / "data/unseen-text-v1/HellaSwag.jsonl") if r["source"]["row_id"] == str(daily_index))
    daily_state = daily["request"]["state"] + "\n\nAnswer candidates:\n" + "\n".join(daily["request"]["options"].values())
    coarse = {"science": "Science: scientific concepts and natural phenomena.",
              "everyday": "Everyday life: household activities and ordinary practical situations.",
              "law": "Law: legal rules, rights, obligations and legal disputes."}
    for expected, state, source, provenance in (
        ("science", domains["biology"]["request"]["state"], domains["biology"]["source"], "manual mapping: official biology category to science"),
        ("law", domains["law"]["request"]["state"], domains["law"]["source"], "manual mapping: official law category to law"),
        ("everyday", daily_state, daily["source"], "manual assessment of HellaSwag sandwich-making example; not an official domain annotation"),
    ):
        examples.append({"id": "coarse-" + expected, "task": "coarse_domain_demo", "expected": expected,
                         "request": {"state": state, "question": "Which broad domain best describes this problem or situation?", "options": coarse},
                         "source": source, "label_provenance": provenance})
    atomic_json(OUT / "examples.json", examples)
    atomic_json(OUT / "manifest.json", {"selection": "one fixed hash-selected example per official class, excluding prior 120-example samples; selected before inference",
                "examples": len(examples), "source_labels_verified": 17, "coarse_labels": "3 manual broad-domain mappings, not benchmark gold labels",
                "training_data_sha256": sha256(DATA / "train.jsonl"), "heldout_data_sha256": sha256(DATA / "test.jsonl"),
                "examples_sha256": sha256(OUT / "examples.json"), "code_sha256": sha256(Path(__file__)),
                "difficulty_cutoffs": cutoffs, "difficulty_scope": "relative difficulty in E2H-AMC; not universal easy/medium/hard",
                "protocol": "one individual no-generation option-logit forward per example per model; no task execution, CoT, or generated classification text"})
    scorer_module.SYSTEM_PROMPT = dataset_manifest["system_prompt"]
    results = {r["id"]: {"example": r} for r in examples}
    for name, path in (("original", ROOT / "model"), ("trained", ROOT / "training/text-decision-full-v1/model")):
        scorer = ChoiceScorer(str(path), device="cuda")
        forwards = []
        hook = scorer.model.model.register_forward_hook(lambda *args: forwards.append(1))
        try:
            with patch.object(scorer.model, "generate", side_effect=AssertionError("Generation is forbidden")):
                for e in examples:
                    prediction = scorer.score(e["request"])
                    assert prediction["usage"]["output_tokens"] == 0
                    prediction["correct"] = prediction["choice"] == e["expected"]
                    results[e["id"]][name] = prediction
                    print(json.dumps({"model": name, "task": e["task"], "expected": e["expected"], "prediction": prediction["choice"], "top_score": prediction["probabilities"][prediction["choice"]], "correct": prediction["correct"]}), flush=True)
        finally:
            hook.remove()
        assert len(forwards) == len(examples)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    counts = {}
    for task in ("academic_domain", "relative_difficulty", "coarse_domain_demo"):
        rows = [r for r in results.values() if r["example"]["task"] == task]
        counts[task] = {"total": len(rows), **{name: sum(r[name]["correct"] for r in rows) for name in ("original", "trained")}}
    atomic_json(OUT / "results.json", {"complete": True, "results": list(results.values()), "counts": counts,
                "generated_tokens": 0, "interpretation": "Illustrative examples, not a new accuracy benchmark. Coarse-domain labels manually mapped; difficulty is relative to AMC."})
    print("COMPLETE " + json.dumps(counts), flush=True)

if __name__ == "__main__":
    main()
