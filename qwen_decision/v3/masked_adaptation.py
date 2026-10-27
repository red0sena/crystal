"""Leak-free masked next-token adaptation for the padded v3 backbone.

Only natural-language content is eligible, never the chat scaffold or choice
labels. Every target is replaced, and h[i-1] predicts original token x[i].
The full vocabulary head is checkpointed in chunks to bound training memory.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from qwen_decision.v3 import format as fmt


def content_positions(renderer, raw, encoded):
    """Map content character spans to exact existing prompt token positions."""
    if renderer.args.layout != "d3":
        raise ValueError("this experiment fixes the original d3 layout")
    question = raw["question"]
    if question["type"] == "choice":
        keys = list(question["criteria"])
        question = {**question, "criteria": {keys[i]: question["criteria"][keys[i]]
                                             for i in encoded["order"]}}
    codes = dict(zip(renderer.code_ids, renderer.codes))
    labels = [codes[t] for t in encoded["code_ids"]]
    user, spans = fmt._user_prompt_spans(raw["state"], question, labels, "d3")
    state = fmt.describe(raw["state"]) if raw["state"] not in (None, "") else "(empty)"
    asked = fmt.describe(question.get("instructions") or fmt.DEFAULT_INSTRUCTIONS)
    prefix = "State:\n" + state + "\n\nQuestion:\n" + asked + "\n\nOptions:\n"
    if not user.startswith(prefix):
        raise ValueError("unexpected prompt structure")
    state_start = len("State:\n")
    question_start = state_start + len(state) + len("\n\nQuestion:\n")
    spans = [(state_start, state_start + len(state)),
             (question_start, question_start + len(asked))] + spans
    text = fmt.render(renderer.tokenizer, raw["state"], question, labels, "d3")
    start = text.rfind(user)
    if start < 0:
        raise ValueError("template did not preserve user content")
    tokenized = renderer.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if tokenized["input_ids"] != encoded["input_ids"]:
        raise ValueError("content mapping changed the matched input")
    spans = [(start + a, start + b) for a, b in spans]
    special = set(renderer.tokenizer.all_special_ids)
    positions = [i for i, ((a, b), token) in enumerate(zip(tokenized["offset_mapping"], encoded["input_ids"]))
                 if i > 0 and b > a and token not in special
                 and any(lo <= a and b <= hi for lo, hi in spans)]
    if not positions:
        raise ValueError("no natural-language tokens to mask")
    return positions


def masked_view(row, *, mask_id, seed, rate=.2):
    """Deterministic, nonempty selection; all targets genuinely hidden."""
    if not 0 < rate <= 1:
        raise ValueError("mask rate must be in (0,1]")
    ids = row["input_ids"]
    eligible = row["eligible_positions"]
    if len(set(eligible)) != len(eligible) or any(i <= 0 or i >= len(ids) for i in eligible):
        raise ValueError("invalid eligible token positions")
    eligible = [i for i in eligible if ids[i] != mask_id]
    if not eligible:
        raise ValueError("no tokens differ from mask")
    rng = random.Random(seed)
    selected = sorted(rng.sample(eligible, min(len(eligible), max(1, round(rate * len(eligible))))))
    masked = list(ids)
    for i in selected:
        masked[i] = mask_id
    return {"input_ids": masked, "target_positions": selected,
            "read_positions": [i - 1 for i in selected], "labels": [ids[i] for i in selected]}


def collate(views, pad_id, device):
    width = max(len(v["input_ids"]) for v in views)
    ids = torch.tensor([v["input_ids"] + [pad_id] * (width - len(v["input_ids"])) for v in views],
                       dtype=torch.long, device=device)
    mask = torch.tensor([[1] * len(v["input_ids"]) + [0] * (width - len(v["input_ids"])) for v in views],
                        dtype=torch.long, device=device)
    row_indices, positions, labels, weights = [], [], [], []
    for j, v in enumerate(views):
        n = len(v["labels"])
        if not n or len(v["read_positions"]) != n:
            raise ValueError("empty or malformed masked labels")
        row_indices.extend([j] * n)
        positions.extend(v["read_positions"])
        labels.extend(v["labels"])
        weights.extend([1 / n] * n)
    return ids, mask, *(torch.tensor(x, device=device, dtype=dtype) for x, dtype in
                       [(row_indices, torch.long), (positions, torch.long),
                        (labels, torch.long), (weights, torch.float32)])


def chunked_loss(hidden, weight, labels, weights, *, chunk_size=64, recompute=True):
    """Sum of per-example mean token CE, with an exact full-vocabulary softmax.

    BF16 projection followed by FP32 cross entropy, as in standard LM training.
    Checkpointing the head avoids retaining every chunk's vocabulary logits.
    """
    if chunk_size < 1 or len(hidden) != len(labels) or len(labels) != len(weights):
        raise ValueError("bad chunk or target shape")

    def project(h, w, y, scale):
        return (F.cross_entropy(F.linear(h, w).float(), y, reduction="none") * scale).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), chunk_size):
        args = (hidden[start:start + chunk_size], weight, labels[start:start + chunk_size],
                weights[start:start + chunk_size])
        total = total + (checkpoint(project, *args, use_reentrant=False)
                         if recompute and torch.is_grad_enabled() else project(*args))
    return total


def loss(model, views, pad_id, *, chunk_size=64, recompute=True):
    device = model.get_output_embeddings().weight.device
    ids, mask, row, pos, labels, weights = collate(views, pad_id, device)
    hidden = model.model.language_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return chunked_loss(hidden[row, pos], model.get_output_embeddings().weight, labels, weights,
                        chunk_size=chunk_size, recompute=recompute)
