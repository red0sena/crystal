"""Full-FT (g7 recipe) memory and speed probe on this GPU. Probe updates are discarded; nothing is saved but the report.

Trains the whole language backbone + tied embedding (vision frozen, off the GPU) with FP32-master AdamW,
exactly as v3/train.py does for g7, on real v3 text rows rendered by the current joint encoder.
1) the longest real row in each length band (single-row updates); 2) timed 32-row random updates for a
few token-budget settings, from which a token-weighted time estimate for 2 epochs is computed.
"""
from collections import Counter
import gc
import json
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from qwen_decision.v3 import champion_full_inputs as data, champion_full_train as fit
from qwen_decision.v3 import train, vision_train as vt, modeling

OUT = data.ROOT.parent / "fullft-probe-v1" / "report.json"
MAX_TOKENS = 65536
EFFECTIVE_BATCH = 32
SETTINGS = [(16384, 16384), (8192, 8192), (4096, 4096)]  # (token_budget, checkpoint_above); g7 used 16384/16384


def load():
    modeling.compile_elementwise()
    model = modeling.load(data.BASE, attention="noncausal", deltanet="causal", dtype=torch.bfloat16, kernel="varlen")
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.language_model.parameters():
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def update(model, opt, rows, budget, ckpt):
    """One discarded optimizer update; returns (gpu_seconds, encode_seconds, tokens)."""
    t0 = time.monotonic()
    cfg = SimpleNamespace(**(vars(fit.CFG) | {"token_budget": budget}))
    groups = vt.MixSteps([rows], cfg, [])[0]
    t1 = time.monotonic()
    opt.zero_grad()
    for cpu in groups:
        batch = vt.to_device(cpu)
        train.set_checkpointing(model, len(cpu["ids"]) > ckpt)
        loss = train.soft_ce(vt.packed_logits(model, batch), batch["targets"]).sum() / len(rows)
        loss.backward()
        del batch, loss
    norm = opt.clip_grad_norm(1.)
    assert np.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    return time.monotonic() - t1, t1 - t0, sum(len(c["ids"]) for c in groups)


def save(report):
    train.atomic_json(OUT, report)


def run(timed_steps=24, warm=4, settings=None, lengths_test=True):
    global OUT
    if settings:
        OUT = OUT.with_name("report-" + "-".join(f"{a}_{b}" for a, b in settings) + ".json")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    files = [data.MIX / "text-train.jsonl", data.MIX / "phishing-url-train.jsonl"]
    indices = [data.ROOT / "indices" / f"text-train-{i}.npy" for i in range(2)]
    texts = data.IndexedRows(files, indices)
    lengths = np.concatenate([np.load(p, mmap_mode="r")[:, 2] for p in indices]).astype(np.int64)
    usable = np.flatnonzero(lengths <= MAX_TOKENS)
    free, total = torch.cuda.mem_get_info()
    train.seed_all(20261108)
    model = load()
    opt = train.MasterAdamW(model, 3e-6)
    report = {"gpu": torch.cuda.get_device_name(), "device_total_gib": total / 2**30,
              "device_free_before_gib": free / 2**30,
              "static_allocated_gib": torch.cuda.memory_allocated() / 2**30,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "recipe": "g7: full LM backbone + tied embedding, vision frozen, BF16 compute, FP32 master AdamW(.9,.95) lr 3e-6",
              "pool_rows": int(len(lengths)), "usable_rows": int(len(usable)),
              "pool_tokens_usable": int(lengths[usable].sum()), "length_trials": [], "speed": [],
              "probe_updates_discarded": True}
    save(report)
    # 1) longest real row per band
    for low, high in ([(4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536)] if lengths_test else []):
        found = np.flatnonzero((lengths > low) & (lengths <= high))
        if not len(found):
            continue
        i = int(found[np.argmax(lengths[found])])
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {"band": high, "tokens": int(lengths[i]), "source": texts[i]["source"]}
        try:
            g, e, _ = update(model, opt, [(texts[i], f"probe:{i}")], 4096, 4096)
            rec.update(passed=True, seconds=g)
        except torch.OutOfMemoryError as err:
            rec.update(passed=False, error=str(err)[:300]); opt.zero_grad()
        rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["length_trials"].append(rec); save(report); print("LENGTH", rec, flush=True)
    # 2) timed random 32-row updates, same batches for every setting
    rng = np.random.default_rng(7)
    batches = [rng.choice(usable, EFFECTIVE_BATCH, replace=False) for _ in range(warm + timed_steps)]
    for budget, ckpt in settings or SETTINGS:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {"token_budget": budget, "checkpoint_above": ckpt, "steps": []}
        try:
            for k, b in enumerate(batches):
                g, e, tok = update(model, opt, [(texts[int(i)], f"probe:{k}:{int(i)}") for i in b], budget, ckpt)
                rec["steps"].append({"gpu_seconds": g, "encode_seconds": e, "tokens": tok, "warmup": k < warm})
                print("SPEED", budget, k, round(g, 2), tok, flush=True)
            timed = [s for s in rec["steps"] if not s["warmup"]]
            sec, tok = sum(s["gpu_seconds"] for s in timed), sum(s["tokens"] for s in timed)
            per_epoch_tokens = report["pool_tokens_usable"]
            rec.update(passed=True, tokens_per_second=tok / sec, mean_step_seconds=sec / len(timed),
                       mean_step_tokens=tok / len(timed),
                       pool_mean_32row_tokens=float(lengths[usable].mean() * EFFECTIVE_BATCH),
                       estimated_hours_per_epoch=per_epoch_tokens / (tok / sec) / 3600,
                       estimated_hours_2_epochs=2 * per_epoch_tokens / (tok / sec) / 3600)
        except torch.OutOfMemoryError as err:
            rec.update(passed=False, error=str(err)[:300]); opt.zero_grad()
        rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["speed"].append(rec); save(report)
        print("SETTING", {k: v for k, v in rec.items() if k != "steps"}, flush=True)
    report["complete"] = True
    save(report)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        run(settings=[(int(sys.argv[1]), int(sys.argv[2]))], lengths_test=False)
    else:
        run()
