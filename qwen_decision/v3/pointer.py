"""Experimental v3 readouts. Both consume ONE unchanged noncausal backbone forward.

Pointer: projected final prompt vector dotted with each projected option-end vector.
Code: independent FP32 copies of the checkpoint's 255 LM-head rows (a trainable control).
Neither head generates text. This module has no Transformers/FLA dependency.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MASK = -1e9


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int = 128):
        super().__init__()
        if projection_size < 1:
            raise ValueError("projection_size must be positive")
        self.query = nn.Linear(hidden_size, projection_size, bias=False)
        self.key = nn.Linear(hidden_size, projection_size, bias=False)
        self.scale = math.sqrt(projection_size)

    def forward(self, decision, options, columns, valid):
        q = self.query(decision.float())
        k = self.key(options.float())
        return (torch.einsum("bd,bkd->bk", q, k) / self.scale).masked_fill(~valid, MASK)


class CodeHead(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().float().clone())

    def forward(self, decision, options, columns, valid):
        scores = F.linear(decision.float(), self.weight)
        return scores.gather(1, columns).masked_fill(~valid, MASK)


def collate(rows, pad_id: int, device):
    """Right-padded inputs, option positions, code-row indices, mask and soft gold targets."""
    width = max(len(r["input_ids"]) for r in rows)
    kmax = max(len(r["option_positions"]) for r in rows)
    ids, masks, positions, columns, valid, targets = [], [], [], [], [], []
    for r in rows:
        k, n = len(r["option_positions"]), len(r["input_ids"])
        if not (k == len(r["code_columns"]) == len(r["target"]) and k >= 2):
            raise ValueError("option positions, code columns and targets must agree (at least 2)")
        if any(p < 0 or p >= n for p in r["option_positions"]):
            raise ValueError("option position outside real input")
        ids.append(r["input_ids"] + [pad_id] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        positions.append(r["option_positions"] + [0] * (kmax - k))
        columns.append(r["code_columns"] + [0] * (kmax - k))
        valid.append([True] * k + [False] * (kmax - k))
        targets.append(r["target"] + [0.] * (kmax - k))
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(masks, device=device),
            "positions": torch.tensor(positions, device=device), "columns": torch.tensor(columns, device=device),
            "valid": torch.tensor(valid, device=device), "targets": torch.tensor(targets, device=device)}


def forward(backbone, head, batch):
    hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      use_cache=False).last_hidden_state
    rows = torch.arange(len(hidden), device=hidden.device)
    last = batch["attention_mask"].sum(-1) - 1
    decision = hidden[rows, last]
    options = hidden[rows[:, None], batch["positions"]]
    return head(decision, options, batch["columns"], batch["valid"])


def soft_ce(logits, targets):
    return -(targets * logits.log_softmax(-1)).sum(-1)
