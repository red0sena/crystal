"""Extend the existing no-generation code scorer to 45 native KMMLU classes."""
from __future__ import annotations
import string
import torch
import scorer as scorer_module

CODES = string.ascii_uppercase + string.ascii_lowercase


def messages(request):
    # Preserve the existing prompt exactly for <=26-class tasks. The extension
    # only introduces lowercase single-token codes for native 45-class scoring.
    if len(request["options"]) <= 26:
        return scorer_module.request_messages(request)
    assert 2 <= len(request["options"]) <= len(CODES)
    options = "\n".join(f"{CODES[i]}. {v}" for i, v in enumerate(request["options"].values()))
    content = [{"type": "text", "text": f"State:\n{request['state']}\n\nQuestion:\n{request['question']}\n\nOptions:\n{options}\n\nAnswer with only the option code."}]
    return [{"role": "system", "content": scorer_module.SYSTEM_PROMPT}, {"role": "user", "content": content}]


def encode(scorer, questions):
    tokenizer = scorer.processor.tokenizer
    tokenizer.padding_side = "right"
    codes = CODES[:max(q["option_count"] for q in questions)]
    ids = []
    for code in codes:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        assert len(encoded) == 1, f"Code {code} is not a single token"
        ids.append(encoded[0])
    assert len(set(ids)) == len(ids)
    for i, q in enumerate(questions):
        prompt = scorer.processor.apply_chat_template(messages(q["request"]), tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        suffix = prompt[-64:]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        for code, token_id in zip(codes, ids):
            assert tokenizer.encode(suffix + code, add_special_tokens=False) == suffix_ids + [token_id]
        q["input_ids"] = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(q["input_ids"]) <= scorer.max_input_tokens, "No input truncation allowed"
        if (i+1) % 5000 == 0:
            print(f"ENCODED {i+1}/{len(questions)}", flush=True)
    return ids


@torch.inference_mode()
def single_probabilities(scorer, question, token_ids):
    ids = torch.tensor([question["input_ids"]], device=scorer.device)
    logits = scorer.option_logits_batch({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, token_ids)
    return logits.cpu(), logits.softmax(-1).cpu()
