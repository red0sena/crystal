"""Training-step benchmark on fixed representative batches, with the time split by phase.

  python -m qwen_decision.v3.bench_train --variant new [--token-budget 8192 --checkpoint-above 8192]

The same 32-row batches (from the first rows of the gold mix, seeded augmentation) are run for every
variant: --warmup batches first (compilation, autotuning), then --steps measured ones. Each phase is
closed by a device synchronize, so the numbers are wall time on the GPU's clock, not launch time:
prep (render, tokenize, microbatch), forward+backward (packing and the copy to the GPU included), optimizer (gradient copy,
clip, AdamW, weight write-back), and tokens/s over the whole step. `train.py`'s step_seconds starts after
rendering and is not synchronized, so use this for speed decisions.

Variants: new (CPU packing) and legacy (GPU packing from padded tensors); native GQA is not switchable;
--foreach toggles MasterAdamW's multi-tensor copies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
from transformers import AutoTokenizer

from qwen_decision.v3 import modeling
from qwen_decision.v3 import train as T

ROOT = Path(__file__).resolve().parents[2]


def legacy_logits(model, rows, pad_id):
    input_ids, mask, codes, code_mask, targets = T.collate(rows, pad_id)
    return modeling.code_logits(model, input_ids, mask, codes, code_mask, packed=True), targets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=("new", "legacy"), default="new")
    p.add_argument("--mix", type=Path, default=ROOT / "data/mix/v3-gold.jsonl")
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--checkpoint-above", type=int, default=16384)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--foreach", action="store_true", help="MasterAdamW multi-tensor copies (off in train.py)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    a = argparse.Namespace(base=ROOT / "models/qwen3.5-0.8b", attention="noncausal", deltanet="causal", layout="d3",
                           kernel="varlen", pad_probability=.35, noul_choice_probability=.2, code_shuffle_probability=.2,
                           code_sample_probability=.2, micro_batch=16,
                           token_budget=args.token_budget, max_tokens=65536, pack=True, compile=True,
                           checkpoint_above=args.checkpoint_above)
    tokenizer = AutoTokenizer.from_pretrained(a.base)
    n = 32 * (args.warmup + args.steps)
    rows = [json.loads(line) for _, line in zip(range(n), args.mix.open())]
    rows = T.gold_targets(rows)
    pool = sorted({o for e in rows if e["question"]["type"] == "choice" and not e["ordered"] for o in e["question"]["criteria"]})
    renderer = T.Renderer(tokenizer, a, pool)
    model = T.model_load(a)
    model.train()
    optimizer = T.MasterAdamW(model, 1e-5, foreach=args.foreach)
    pad = tokenizer.pad_token_id
    rng = random.Random(0)
    timing = {"prep": [], "fwd_bwd": [], "optimizer": [], "step": [], "tokens": []}
    for i in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoded = [renderer(r, rng) for r in rows[i * 32:(i + 1) * 32]]
        groups = list(T.microbatches(encoded, a))
        t1 = time.perf_counter()
        optimizer.zero_grad()
        for g in groups:
            T.set_checkpointing(model, sum(len(r["input_ids"]) for r in g) > a.checkpoint_above)
            if args.variant == "new":
                logits, targets = T.logits_for(model, g, pad, True)
            else:
                logits, targets = legacy_logits(model, g, pad)
            (T.soft_ce(logits, targets).sum() / len(encoded)).backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optimizer.clip_grad_norm(1.)
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= args.warmup:
            timing["prep"].append(t1 - t0)
            timing["fwd_bwd"].append(t2 - t1)
            timing["optimizer"].append(t3 - t2)
            timing["step"].append(t3 - t0)
            timing["tokens"].append(sum(len(e["input_ids"]) for e in encoded))
    result = {"variant": args.variant, "foreach": args.foreach, "token_budget": args.token_budget, "checkpoint_above": args.checkpoint_above,
              **{f"{k}_median_s": round(statistics.median(v), 4) for k, v in timing.items() if k != "tokens"},
              "tokens_per_s": round(sum(timing["tokens"]) / sum(timing["step"])),
              "tokens_per_step": round(statistics.mean(timing["tokens"])),
              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    print(json.dumps(result), flush=True)
    if args.out:
        with args.out.open("a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
