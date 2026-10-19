"""Resume preserved evaluations with diagnostic BF16 checks, not an arbitrary tolerance."""
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
import scorer as scorer_module
from scorer import ChoiceScorer
from evaluate_mmlu_pro import atomic_json, make_batches, sha256
from evaluate_choice_probabilities import batch_probabilities
from external_classification_protocol import CODES, encode, single_probabilities
from evaluate_external_classification import ROOT, DATA, OUT, load_lines, metrics, run


def main():
    data = json.loads((DATA / "manifest.json").read_text())
    assert data["complete"] and not (OUT / "summary.json").exists()
    assert sha256(ROOT / "data/text-decisions-v1/train.jsonl") == data["training_data_sha256"]
    scorer_module.SYSTEM_PROMPT = data["system_prompt"]
    results = json.loads((OUT / "partial-summary.json").read_text())
    start = time.time()
    torch.manual_seed(0)
    for model_name, path in (("original", ROOT / "model"), ("trained", ROOT / "training/text-decision-full-v1/model")):
        scorer = ChoiceScorer(str(path), device="cuda")
        weights = {p.name: sha256(p) for p in sorted(path.glob("*.safetensors"))}
        for spec in data["datasets"]:
            task = spec["task"]
            directory = OUT / model_name / task
            assert sha256(Path(spec["file"])) == spec["sha256"]
            if (directory / "summary.json").exists():
                prior = json.loads((directory / "manifest.json").read_text())
                assert prior["model_weights_sha256"] == weights and prior["dataset_file_sha256"] == spec["sha256"]
                for filename, digest in prior["code_sha256"].items():
                    assert sha256(ROOT / filename) == digest
                saved = json.loads((directory / "summary.json").read_text())
                assert saved["inference"]["predictions_sha256"] == sha256(directory / "predictions.jsonl")
                results.setdefault(task, {})[model_name] = saved
                print(f"PRESERVED {model_name} {task}: {saved['accuracy_pct']:.2f}%", flush=True)
                continue
            assert not (directory / "predictions.jsonl").exists()
            records = load_lines(Path(spec["file"]))
            questions = [{"question_id": i, "source_id": r["id"], "request": r["request"],
                          "gold_index": r["gold_index"], "option_count": r["option_count"]} for i,r in enumerate(records)]
            ids = encode(scorer, questions)
            scorer.processor.tokenizer.padding_side = "left"
            batches = make_batches(questions, 32, 32768)
            directory.mkdir(parents=True, exist_ok=True)
            layout = [[q["question_id"] for q in b] for b in batches]
            encoding_hash = hashlib.sha256(json.dumps([q["input_ids"] for q in questions], separators=(",", ":")).encode()).hexdigest()
            manifest = {"task": task, "model": str(path), "model_name": model_name, "model_weights_sha256": weights,
                        "data_manifest_sha256": sha256(DATA / "manifest.json"), "dataset_file_sha256": spec["sha256"],
                        "batch_layout": layout, "batch_size": 32, "padded_token_budget": 32768,
                        "encoding_sha256": encoding_hash, "codes": CODES[:len(ids)], "option_token_ids": ids,
                        "code_sha256": {n: sha256(ROOT / n) for n in ("continue_external_classification.py", "evaluate_external_classification.py", "external_classification_protocol.py", "evaluate_choice_probabilities.py", "evaluate_mmlu_pro.py", "scorer.py")},
                        "system_prompt": scorer_module.SYSTEM_PROMPT,
                        "protocol": "One measured pass; fixed shuffled choices; BF16 selected LM Head rows; no generation, CoT, training or checkpoint selection; identical batches across models. Native 45-class KMMLU uses uppercase/lowercase single-token codes."}
            if model_name == "trained":
                prior = json.loads((OUT / "original" / task / "manifest.json").read_text())
                for key in ("encoding_sha256", "batch_layout", "codes", "option_token_ids", "dataset_file_sha256"):
                    assert prior[key] == manifest[key]
            print(f"START {model_name} {task}: {len(questions)} questions; {len(batches)} batches", flush=True)
            warm_logits, warm_probs = batch_probabilities(scorer, batches[0], ids)
            if task == "KMMLU_domain":
                repeated_logits, repeated_probs = batch_probabilities(scorer, batches[0], ids)
                assert torch.equal(warm_logits, repeated_logits) and torch.equal(warm_probs, repeated_probs)
                diagnostics = []
                for i,q in enumerate(batches[0][:3]):
                    single_logits, single_probs = single_probabilities(scorer, q, ids)
                    diagnostics.append({"question_id": q["question_id"],
                                        "max_probability_difference": float((single_probs[0]-warm_probs[i]).abs().max()),
                                        "max_logit_difference": float((single_logits[0]-warm_logits[i]).abs().max()),
                                        "argmax_equal": int(single_probs[0].argmax()) == int(warm_probs[i].argmax())})
                manifest["extended_code_verification"] = diagnostics
                manifest["batch_repeat_bitwise_identical"] = True
                manifest["scalar_batch_note"] = "BF16 probabilities can differ with GEMM/batch shapes; scalar differences are reported, not silently rounded or required to meet an arbitrary 0.05 cutoff. All measured comparisons use matched batches."
                print("CODE CHECK " + json.dumps(diagnostics), flush=True)
            atomic_json(directory / "manifest.json", manifest)
            atomic_json(OUT / "progress.json", {"status": "evaluating", "current_model": model_name, "current_task": task,
                                               "total_questions": len(questions), "results": results})
            inference = run(scorer, questions, ids, batches, directory, task)
            predictions = load_lines(directory / "predictions.jsonl")
            measured = metrics(records, predictions, spec["labels"])
            assert measured["correct"] == inference["correct"]
            measured["inference"] = inference
            results.setdefault(task, {})[model_name] = measured
            atomic_json(directory / "summary.json", measured)
            atomic_json(OUT / "partial-summary.json", results)
            del records, questions, batches, predictions
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    for task,r in results.items():
        r["accuracy_change_percentage_points"] = r["trained"]["accuracy_pct"] - r["original"]["accuracy_pct"]
    atomic_json(OUT / "summary.json", {"complete": True, "results": results, "total_questions_per_model": data["total_requests_per_model"],
                "continuation_elapsed_seconds": time.time()-start, "measured_inference_elapsed_seconds": sum(r[m]["inference"]["elapsed_seconds"] for r in results.values() for m in ("original", "trained")),
                "repeats_per_model": 1, "generated_tokens": 0, "data_manifest_sha256": sha256(DATA / "manifest.json"),
                "interpretation": "Source-native categories; SuperGPQA native three-class difficulty; KMMLU fixed human-accuracy derived bands. Screened exact/near input overlap; no proof of base pretraining independence. BF16 results depend on batch shape; paired evaluations match encodings, option orders and batches."})
    atomic_json(OUT / "progress.json", {"status": "complete"})
    print("ALL COMPLETE " + json.dumps({t:{m:r[m]["accuracy_pct"] for m in ("original", "trained")} for t,r in results.items()}), flush=True)


if __name__ == "__main__":
    main()
