"""Full fine-tuning with the g7 recipe on the data-scale-v1 text mix (~2.07M rows), 3 epochs, resumable.

User request (2026-11-08): "풀파인튜닝으로 3에폭까지". Recipe = g7 (v3/train.py): whole language backbone + tied
embedding/LM head trainable, vision frozen; noncausal attention, causal DeltaNet; BF16 compute with FP32-master
AdamW(.9,.95, wd .01), lr 3e-6, effective batch 32 rows, 5% linear warmup then cosine to 10%; gold CE on the shown
codes; augmentations noul->choice .2, code shuffle .2, code sample .2, no padding (current encoder TEXT_ARGS).
Throughput setting from the 3090 probe (fullft-probe-v1): 4k-token microbatches, checkpointing above 4k.

Selection: gold dev macro-source accuracy on full-g7-v3/mix/text-dev.jsonl (as g7), evaluated at step 0, every
EVAL_EVERY updates and at each epoch end; never public scores. Exports: best/, epoch-1/, epoch-2/, last/.
Resume: FP32 masters + AdamW moments + RNG every SAVE_EVERY updates (last two kept); batches are a pure function of
(seed, step), so a resumed run sees exactly the rows and augmentations it would have seen.

Long rows (11-09 resume after an OOM at update 8,992): every row up to MAX_TOKENS trains (16k-50k-token rows run as
single-row microbatches; the 3090 probe passed 49,552 tokens). Rows above it are skipped, never truncated, as g7's
train.py --max-tokens did: in this mix they are 9 rows, three broken phishing emails x 3 variants (115k, 239k and
387k tokens: a mailing-list dump and a spam with ~1M recipient addresses; 387k exceeds the model's 262,144 positions).
An update that still runs out of memory is dropped (no optimizer step) and logged.
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer

from qwen_decision.v3 import champion_full_inputs as data, champion_full_train as fit
from qwen_decision.v3 import format as fmt, modeling, train, vision_train as vt
from qwen_decision.v3.champion_joint_data import MODEL_ALLOWED

ROOT = Path("runs/champion-v1/fullft-scale-v1")
MIX = Path("runs/champion-v1/data-scale-v1/mix/text-train.jsonl")
DEV = Path("runs/champion-v1/full-g7-v3/mix/text-dev.jsonl")
SEED = 20261108
EPOCHS, LR, BATCH, WARMUP, FLOOR = 3, 3e-6, 32, .05, .1
EVAL_EVERY, SAVE_EVERY, KEEP = 16384, 2048, 2
MAX_TOKENS = 65536
CFG = SimpleNamespace(**(vars(fit.CFG) | {"workers": 6, "token_budget": 4096, "checkpoint_above": 4096}))


def sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs

def build_index(root):
    """(offset, bytes, 0) per line of MIX, after validating every row; cached by the mix file's sha256."""
    dest, proof = root / "index.npy", root / "index.json"
    digest = sha(MIX)
    if proof.exists() and json.loads(proof.read_text())["mix_sha256"] == digest:
        return dest
    offsets = []
    with MIX.open("rb") as f:
        while True:
            at = f.tell()
            raw = f.readline()
            if not raw:
                break
            r = json.loads(raw)
            # gold labels only; the g7 pool's model-labelled tool sources stay under their historical allowance
            assert r["modality"] == "text" and (r["label_origin"] in ("human", "programmatic") or r["source"] in MODEL_ALLOWED), r["id"]
            k = len(fmt.options(r["question"])[0])
            assert 2 <= k <= fmt.MAX_OPTIONS and len(r["target"]) == k, r["id"]
            assert abs(sum(r["target"]) - 1) < 1e-5 and min(r["target"]) >= 0, r["id"]
            offsets.append((at, len(raw), 0))
    root.mkdir(parents=True, exist_ok=True)
    np.save(dest, np.array(offsets, dtype=np.uint64))
    train.atomic_json(proof, {"mix": str(MIX), "mix_sha256": digest, "rows": len(offsets)})
    return dest


class Schedule:
    """Every row once per epoch (fresh permutation per epoch, final partial batch kept); 32 rows per update."""
    def __init__(self, rows, epochs=EPOCHS, seed=SEED):
        self.rows, self.epochs, self.seed = rows, epochs, seed
        self.per_epoch = math.ceil(len(rows) / BATCH)
        self.orders = {}

    def __len__(self):
        return self.per_epoch * self.epochs

    def __getitem__(self, index):
        epoch, step = divmod(index, self.per_epoch)
        if epoch not in self.orders:
            s = int(hashlib.sha256(f"{self.seed}:epoch:{epoch}".encode()).hexdigest()[:16], 16)
            self.orders = {epoch: np.random.default_rng(s).permutation(len(self.rows))}
        picked = self.orders[epoch][step * BATCH:(step + 1) * BATCH]
        return [(self.rows[int(i)], f"fullft-scale:{self.seed}:{index}:{j}") for j, i in enumerate(picked)]


# ------------------------------------------------------------------ model / optimizer state

def load_model():
    modeling.compile_elementwise()
    model = modeling.load(data.BASE, attention="noncausal", deltanet="causal", dtype=torch.bfloat16, kernel="varlen")
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.language_model.parameters():
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def save_checkpoint(root, model, opt, step, extra):
    dest = root / "checkpoints" / f"step-{step:06}"
    dest.mkdir(parents=True, exist_ok=True)
    names = trainable_names(model)
    assert len(names) == len(opt.pairs)
    save_file({n: m.detach().cpu().contiguous() for n, (_, m) in zip(names, opt.pairs)}, str(dest / "masters.safetensors"))
    torch.save({"optimizer": opt.optimizer.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "step": step, **extra}, dest / "state.pt")
    train.atomic_json(root / "last-checkpoint.json", {"directory": str(dest), "step": step})
    old = sorted((root / "checkpoints").glob("step-*"))[:-KEEP]
    for d in old:
        shutil.rmtree(d)


def restore_checkpoint(root, model, opt):
    marker = root / "last-checkpoint.json"
    if not marker.exists():
        return 0, {}
    dest = Path(json.loads(marker.read_text())["directory"])
    masters = load_file(str(dest / "masters.safetensors"), device="cuda")
    with torch.no_grad():
        for n, (p, m) in zip(trainable_names(model), opt.pairs):
            m.copy_(masters[n])
            p.copy_(m)
    state = torch.load(dest / "state.pt", map_location="cpu", weights_only=False)
    opt.optimizer.load_state_dict(state.pop("optimizer"))
    torch.set_rng_state(state.pop("rng"))
    torch.cuda.set_rng_state_all(state.pop("cuda_rng"))
    return state.pop("step"), state


def export(model, tokenizer, dest, info):
    codes, ids = fmt.answer_codes(tokenizer)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    modeling.save(model, tokenizer, tmp, {"attention": "noncausal", "layout": "d3", "deltanet": "causal", "codes": codes,
                                          "token_ids": ids, "trained_from": str(data.BASE), **info})
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)


# ------------------------------------------------------------------ evaluation

def dev_rows(limit=None):
    rows = [json.loads(l) for l in DEV.open()]
    if limit:
        rows = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:limit]
    return vt.gold_text(rows)


def evaluate(model, rows):
    pred = vt.predict(model, [("mix", r) for r in rows], workers=CFG.workers, token_budget=CFG.token_budget)
    by = defaultdict(list)
    for r, p in zip(rows, pred):
        by[r["source"]].append(p["correct"])
    model.train()
    return {"rows": len(rows), "macro_source_accuracy": statistics.mean(statistics.mean(v) for v in by.values()),
            "all_accuracy": statistics.mean(c for v in by.values() for c in v),
            "sources": {s: statistics.mean(v) for s, v in sorted(by.items())}}


# ------------------------------------------------------------------ training

def run(root=ROOT, max_steps=None, dev_limit=None, save_every=SAVE_EVERY, eval_every=EVAL_EVERY, stop_after=None):
    root.mkdir(parents=True, exist_ok=True)
    fit.bind()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    rows = data.IndexedRows([MIX], [build_index(root)])
    schedule = Schedule(rows)
    steps = max_steps or len(schedule)
    warm = max(1, math.ceil(len(schedule) * WARMUP))
    plan = {"experiment": "fullft-scale-v1", "base": str(data.BASE), "mix": str(MIX), "mix_rows": len(rows),
            "mix_manifest": str(MIX.parent / "manifest.json"), "dev": str(DEV), "seed": SEED, "epochs": EPOCHS,
            "steps_per_epoch": schedule.per_epoch, "steps": len(schedule), "run_steps": steps, "warmup_steps": warm,
            "learning_rate": LR, "lr_schedule": f"linear warmup {WARMUP:.0%}, cosine to {FLOOR:.0%}", "effective_batch": BATCH,
            "optimizer": "FP32 master AdamW(.9,.95), wd .01 (matrices), clip 1", "trainable": "language backbone + tied embedding",
            "attention": "noncausal", "deltanet": "causal", "config": vars(CFG), "max_tokens_skip": MAX_TOKENS,
            "text_args": vars(vt.TEXT_ARGS), "selection": "gold dev macro source accuracy; no public-score selection",
            "code_sha256": {Path(m.__file__).name: sha(m.__file__) for m in (sys.modules[__name__], data, fit, fmt, modeling, train, vt)}}
    if (root / "plan.json").exists():
        old = json.loads((root / "plan.json").read_text())
        for k in ("mix_rows", "steps", "seed", "learning_rate", "warmup_steps", "run_steps"):
            assert old[k] == plan[k], (k, old[k], plan[k])
    train.atomic_json(root / "plan.json", plan)
    train.seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(data.BASE, local_files_only=True)
    model = load_model()
    opt = train.MasterAdamW(model, LR)
    start, state = restore_checkpoint(root, model, opt)
    tokens, best, best_step = state.get("tokens", 0), state.get("best", -1.), state.get("best_step", -1)
    model.train()
    dev = dev_rows(dev_limit)
    history_path = root / "eval.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}

    def checkpoint_eval(step):
        nonlocal best, best_step
        result = evaluate(model, dev)
        history[f"step-{step}"] = result
        train.atomic_json(history_path, history)
        print("EVAL", step, round(result["macro_source_accuracy"], 4), round(result["all_accuracy"], 4), flush=True)
        if result["macro_source_accuracy"] > best:
            best, best_step = result["macro_source_accuracy"], step
            export(model, tokenizer, root / "best", {"step": step, "selection": "gold dev macro source accuracy",
                                                     "dev_macro": best})
            train.atomic_json(root / "best.json", {"step": step, **result})

    if start == 0 and "step-0" not in history:
        checkpoint_eval(0)
    dataset = vt.MixSteps(schedule, CFG, [])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(start, steps)), batch_size=None,
                                         num_workers=CFG.workers, pin_memory=True, prefetch_factor=4,
                                         worker_init_fn=fit.worker_init, multiprocessing_context="spawn")
    log_path = root / "train.jsonl"
    if log_path.exists():
        kept = [l for l in log_path.open() if json.loads(l)["step"] <= start]
        log_path.write_text("".join(kept))
    recent, began = deque(maxlen=512), time.monotonic()
    previous = began
    with log_path.open("a", buffering=1) as log:
        for step, groups in enumerate(loader, start + 1):
            opt.zero_grad()
            long = [b for b in groups if len(b["ids"]) > MAX_TOKENS]
            groups = [b for b in groups if len(b["ids"]) <= MAX_TOKENS]
            skipped = sum(int(b["targets"].shape[0]) for b in long)
            n = sum(int(b["targets"].shape[0]) for b in groups)
            loss_sum, right, batch_tokens, dropped = 0., 0, 0, False
            try:
                for cpu in groups:
                    batch = vt.to_device(cpu)
                    train.set_checkpointing(model, len(cpu["ids"]) > CFG.checkpoint_above)
                    logits = vt.packed_logits(model, batch)
                    ce = train.soft_ce(logits, batch["targets"])
                    (ce.sum() / n).backward()
                    loss_sum += float(ce.sum().detach())
                    right += int((batch["targets"].gather(1, logits.argmax(-1, keepdim=True))[:, 0] > 0).sum())
                    batch_tokens += len(cpu["ids"])
            except torch.OutOfMemoryError:
                batch = logits = ce = None
                opt.zero_grad()
                gc.collect(); torch.cuda.empty_cache()
                lengths = [len(b["ids"]) for b in groups]
                log.write(json.dumps({"step": step, "oom_dropped_update": True, "microbatch_tokens": lengths}) + "\n")
                print("OOM_DROPPED_UPDATE", step, lengths, flush=True)
                dropped = True
            if not dropped:
                if not math.isfinite(loss_sum):
                    raise RuntimeError(f"nonfinite loss at step {step}")
                norm = opt.clip_grad_norm(1.)
                if not math.isfinite(norm):
                    raise RuntimeError(f"nonfinite gradient norm at step {step}")
                factor = step / warm if step <= warm else FLOOR + (1 - FLOOR) * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, len(schedule) - warm)))
                for g in opt.param_groups:
                    g["lr"] = LR * factor * g["lr_scale"]
                opt.step()
                tokens += batch_tokens
                now = time.monotonic()
                record = {"step": step, "steps": steps, "epoch": step / schedule.per_epoch, "loss": loss_sum / n,
                          "batch_accuracy": right / n, "lr": opt.param_groups[0]["lr"], "grad_norm": norm, "rows": n,
                          "tokens": batch_tokens, "total_tokens": tokens, "step_seconds": now - previous}
                if skipped:
                    record["skipped_over_max_tokens"] = skipped
                previous = now
                if step > start + 16:
                    recent.append(record["step_seconds"])
                log.write(json.dumps(record) + "\n")
                if step % 50 == 0 or step <= start + 3:
                    status = {"stage": "training", **record}
                    if len(recent) >= 32:
                        mean = statistics.mean(recent)
                        end = datetime.now(timezone.utc) + timedelta(seconds=(steps - step) * mean)
                        status["eta"] = {"seconds_per_update": mean, "estimated_end_kst": end.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
                                         "excludes": "dev evaluations and exports"}
                    train.atomic_json(root / "status.json", status)
                    print("TRAIN", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}, flush=True)
            epoch_end = step % schedule.per_epoch == 0
            if step % eval_every == 0 or epoch_end or step == steps:
                checkpoint_eval(step)
            if epoch_end:
                export(model, tokenizer, root / f"epoch-{step // schedule.per_epoch}", {"step": step, "epoch": step // schedule.per_epoch})
            if step % save_every == 0 or epoch_end or step == steps:
                save_checkpoint(root, model, opt, step, {"tokens": tokens, "best": best, "best_step": best_step})
            if stop_after and step >= stop_after:
                print("STOP_AFTER", step, flush=True)
                return
    export(model, tokenizer, root / "last", {"step": steps})
    train.atomic_json(root / "complete.json", {"complete": True, "steps": steps, "total_tokens": tokens, "best_step": best_step,
                                               "best_dev_macro": best, "selection": "gold dev macro source accuracy",
                                               "training_seconds_last_process": time.monotonic() - began})
    train.atomic_json(root / "status.json", {"stage": "complete", "step": steps, "best_step": best_step})
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "smoke", "smoke-resume"))
    a = p.parse_args()
    smoke = ROOT.with_name("fullft-scale-smoke")
    try:
        if a.command == "run":
            run()
        elif a.command == "smoke":  # 6 updates, checkpoint at 4, stop; then smoke-resume continues to 8 with a dev eval
            run(smoke, max_steps=8, dev_limit=400, save_every=4, eval_every=8, stop_after=6)
        else:
            run(smoke, max_steps=8, dev_limit=400, save_every=4, eval_every=8)
    except Exception:
        train.atomic_json((smoke if a.command != "run" else ROOT) / "failure.json", {"command": a.command, "traceback": traceback.format_exc()})
        raise
