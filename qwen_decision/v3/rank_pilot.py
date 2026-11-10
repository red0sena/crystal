"""Pilot: does a listwise ranking term over per-candidate yes/no logits help retrieval-shaped questions?

Arms (same base Qwen + all186 LoRA r16, same ToolRet-format groups and order, one group = one update):
  ce    mean CE over the group's yes/no questions (the production loss)
  rank  ce + LAMBDA * listwise CE: softmax over the group's relevance margins s_j = logit_yes - logit_no,
        target = the relevant candidate (FIRST / learning-to-rank style; one forward per candidate, as served)
Rows are rendered without augmentation (codes A/B in stored order: criteria no, yes), so index 1 is "yes".
Metric: nDCG@10 over held-out dev groups ranked by s, plus dev accuracy / yes-rate. Pilot only: one task, small.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from qwen_decision.v3 import champion_full_train as fit, train, vision_train as vt

ROOT = Path("runs/champion-v1/rank-pilot-v1")
DATA = Path("runs/champion-v1/format-rebuild-v1")
LAMBDA = 1.0
LR, WARMUP = 2e-4, 60
BUDGET = 4096


def groups(path):
    out = {}
    for line in path.open():
        r = json.loads(line)
        out.setdefault(r["group"], []).append(r)
    return [out[k] for k in sorted(out)]


def margins(model, enc, rows, grad):
    """Relevance margins (logit_yes - logit_no) and code logits for rows, in row order."""
    image_token = enc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    encoded = [enc.row(r, None) for r in rows]
    chunks, cur, size = [], [], 0
    for e in encoded:
        n = len(e["input_ids"])
        if cur and size + n > BUDGET:
            chunks.append(cur); cur, size = [], 0
        cur.append(e); size += n
    chunks.append(cur)
    logits = []
    with torch.set_grad_enabled(grad):
        for c in chunks:
            batch = vt.to_device(vt.pack(c, image_token, enc.vision_config if hasattr(enc, "vision_config") else VCFG))
            train.set_checkpointing(model, grad)
            logits.append(vt.packed_logits(model, batch))
    logits = torch.cat(logits)
    return logits[:, 1] - logits[:, 0], logits


def run(arm, seed=41, epochs=1):
    global VCFG
    out = ROOT / f"{arm}-seed{seed}"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fit.bind()
    VCFG = json.loads((vt.PROCESSOR / "config.json").read_text())["vision_config"]
    train.seed_all(seed)
    tr, dev = groups(DATA / "toolret-format-train.jsonl"), groups(DATA / "toolret-format-dev.jsonl")
    model = fit.legacy.load_model()
    model.train()
    opt = train.MasterAdamW(model, LR)
    enc = vt.Encoder()
    order = [g for _ in range(epochs) for g in random.Random(seed).sample(tr, len(tr))]
    steps, log = len(order), []
    t0 = time.time()
    for step, g in enumerate(order, 1):
        f = step / WARMUP if step <= WARMUP else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - WARMUP) / (steps - WARMUP)))
        for pg in opt.param_groups:
            pg["lr"] = LR * f * pg["lr_scale"]
        opt.zero_grad()
        s, logits = margins(model, enc, g, True)
        gold = torch.tensor([r["target"][1] for r in g], device="cuda")
        targets = torch.stack([1 - gold, gold], 1)
        ce = train.soft_ce(logits, targets).mean()
        loss = ce
        rank = torch.zeros((), device="cuda")
        if arm == "rank" and gold.sum() > 0:
            rank = -(gold / gold.sum() * s.float().log_softmax(0)).sum()
            loss = ce + LAMBDA * rank
        loss.backward()
        norm = opt.clip_grad_norm(1.)
        opt.step()
        if step % 50 == 0 or step == steps:
            rec = {"step": step, "ce": float(ce), "rank": float(rank), "grad_norm": norm, "seconds": time.time() - t0}
            log.append(rec); print(arm, rec, flush=True)
    model.eval()
    ndcg, correct, n, yes = [], 0, 0, 0
    for g in dev:
        s, logits = margins(model, enc, g, False)
        gold = np.array([r["target"][1] for r in g])
        order_ = np.argsort(-s.float().cpu().numpy(), kind="stable")
        rel = gold[order_][:10]
        ndcg.append(sum(rel[i] / math.log2(i + 2) for i in range(len(rel))))
        pred = logits.argmax(1).cpu().numpy()
        correct += int((pred == gold).sum()); n += len(g); yes += int(pred.sum())
    result = {"arm": arm, "seed": seed, "lambda": LAMBDA if arm == "rank" else 0, "train_groups": len(tr), "dev_groups": len(dev),
              "steps": steps, "ndcg10": float(np.mean(ndcg)), "ndcg10_per_group": [float(x) for x in ndcg],
              "dev_accuracy": correct / n, "dev_pred_yes_rate": yes / n, "dev_gold_yes_rate": sum(r["target"][1] for g in dev for r in g) / n,
              "log": log, "train_seconds": time.time() - t0}
    train.atomic_json(out / "result.json", result)
    print("RESULT", {k: v for k, v in result.items() if k not in ("log", "ndcg10_per_group")}, flush=True)
    return result


if __name__ == "__main__":
    for arm in sys.argv[1].split(","):
        run(arm, int(sys.argv[2]) if len(sys.argv) > 2 else 41)
