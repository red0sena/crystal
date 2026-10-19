"""Diagnose scalar/batch BF16 differences before running 45-class evaluation."""
import json
from pathlib import Path
import torch
import scorer as scorer_module
from scorer import ChoiceScorer
from external_classification_protocol import encode, single_probabilities
from evaluate_choice_probabilities import batch_probabilities
from evaluate_mmlu_pro import make_batches, atomic_json

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "data/external-classification-eval-v1/manifest.json").read_text())
    spec = next(s for s in data["datasets"] if s["task"] == "KMMLU_domain")
    scorer_module.SYSTEM_PROMPT = data["system_prompt"]
    scorer = ChoiceScorer(str(ROOT / "model"), device="cuda")
    with Path(spec["file"]).open() as handle:
        records = [json.loads(line) for line in handle]
    questions = [{"question_id": i, "source_id": r["id"], "request": r["request"], "gold_index": r["gold_index"], "option_count": r["option_count"]} for i,r in enumerate(records)]
    ids = encode(scorer, questions)
    scorer.processor.tokenizer.padding_side = "left"
    batch = make_batches(questions, 32, 32768)[0]
    logits, probs = batch_probabilities(scorer, batch, ids)
    again_logits, again_probs = batch_probabilities(scorer, batch, ids)
    rows = []
    for i,q in enumerate(batch[:3]):
        single_logits, single_probs = single_probabilities(scorer, q, ids)
        equal_length_batch = [q] * 32
        same_logits, same_probs = batch_probabilities(scorer, equal_length_batch, ids)
        item = {"question_id": q["question_id"], "input_tokens": len(q["input_ids"]),
                "single_logits": single_logits[0].tolist(), "batch_logits": logits[i].tolist(),
                "single_probabilities": single_probs[0].tolist(), "batch_probabilities": probs[i].tolist(),
                "max_probability_difference": float((single_probs[0] - probs[i]).abs().max()),
                "max_logit_difference": float((single_logits[0] - logits[i]).abs().max()),
                "equal_length_batch_vs_single_probability_difference": float((same_probs[0] - single_probs[0]).abs().max()),
                "equal_length_vs_padded_batch_probability_difference": float((same_probs[0] - probs[i]).abs().max()),
                "single_argmax": int(single_probs[0].argmax()), "batch_argmax": int(probs[i].argmax()),
                "repeat_batch_max_difference": float((again_probs[i]-probs[i]).abs().max())}
        rows.append(item)
        print(json.dumps({k:v for k,v in item.items() if not isinstance(v,list)}), flush=True)
    out = ROOT / "results/external-classification-v1"
    atomic_json(out / "batching-diagnostic.json", {"model": "original", "dtype": str(scorer.model.dtype), "rows": rows})


if __name__ == "__main__":
    main()
