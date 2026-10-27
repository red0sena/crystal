"""Structure comparison from the untrained base model: every arm is fully fine-tuned on the same data.

All arms start from base Qwen3.5-0.8B (models/qwen3.5-0.8b, never trained on decisions) and see the same
65,536 v3-gold6 rows (all 104 sources, none truncated: data/longctx/train-nocap104-64k.jsonl) in the same
order, with the same option permutations and code draws, for the same 4,096 updates. Only the structure differs:
  causal     the stock model: all 24 layers read left to right.
  noncausal  d3: the 6 full-attention layers see the whole prompt (the "native" of every earlier experiment).
  hop        noncausal plus one re-read at the decision point. The first read's final hidden state at the
             decision token, rescaled to the embedding norm, is appended to the prompt as one more input
             vector and the model reads the whole prompt again; the answer is read at that position.
             Each read gets half of the gold CE and the second read takes the first read's state without
             gradient, so peak memory stays that of one read. No new parameters; about twice the compute.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment pool
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment check
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment smoke
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment freeze
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment run --seeds 41,73,101
  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.from_base_experiment report

Evaluation is long_context_experiment's: selection (4,147 short items) and the long test sets, with a
prompt too long for the context or the GPU counted as wrong. The decision rule is fixed in plan.json by
`freeze`, before any fit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface

from qwen_decision.v3 import architecture_search as search, long_context_experiment as lce, long_eval
from qwen_decision.v3 import long_training as lt, modeling, readout_experiment as records
from qwen_decision.v3 import robust_adaptation as views, train

OUT = Path("runs/from-base/structures-v1")
BASE = Path("models/qwen3.5-0.8b")
POOL = long_eval.OUT / "train-nocap104-64k.jsonl"
ARMS = ("causal", "noncausal", "hop")
SEEDS = (41, 73, 101)
STEPS, BATCH_SIZE, BODY_LR, WARMUP = 4096, 16, 1e-5, 128
COMPARISONS = {"noncausal-causal": ("causal", "noncausal"), "hop-noncausal": ("noncausal", "hop"),
               "hop-causal": ("causal", "hop")}
CAUSAL_VARLEN = "v3_varlen_causal"


def read(path):
    return json.loads(Path(path).read_text())


def varlen_causal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Stock causal attention over packed rows (modeling.varlen_noncausal_attention with is_causal=True)."""
    from torch.nn.attention.varlen import varlen_attn

    if attention_mask is not None or kwargs.get("cu_seq_lens_q") is None or dropout or query.shape[0] != 1:
        raise ValueError("causal varlen attention takes one packed row, no mask and no dropout")
    starts = kwargs["cu_seq_lens_q"].to(torch.int32)
    longest = int(kwargs["max_length_q"])
    q, k, v = (x[0].transpose(0, 1).to(torch.bfloat16) for x in (query, key, value))
    out = varlen_attn(q, k, v, starts, starts, longest, longest, is_causal=True)
    return out[None].to(query.dtype), None


AttentionInterface.register(CAUSAL_VARLEN, varlen_causal_attention)


def enable_causal_packed(text_model) -> None:
    """Packed rows for the stock causal model: the boundaries go to the varlen kernels, as in enable_noncausal."""
    text_model.config._attn_implementation = CAUSAL_VARLEN

    def boundaries(module, args, kwargs):
        if args or kwargs.get("cu_seq_lens_q") is None or kwargs.get("attention_mask") is not None:
            raise ValueError("the causal arm runs packed rows only")
        kwargs["attention_mask"] = {"full_attention": None, "linear_attention": None}
        return args, kwargs

    text_model.register_forward_pre_hook(boundaries, with_kwargs=True)


def load_model(arm: str, path=BASE):
    modeling.compile_elementwise()
    model = modeling.load(path, attention="causal" if arm == "causal" else "noncausal", kernel="varlen",
                          dtype=torch.bfloat16)
    if arm == "causal":
        enable_causal_packed(model.model.language_model)
    model.requires_grad_(False)
    model.model.language_model.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def embedding_rms(model) -> float:
    """Mean RMS of the input embedding rows: the norm a re-read vector is given."""
    weight = model.model.language_model.embed_tokens.weight.detach().float()
    return float(weight.pow(2).mean(-1).sqrt().mean())


def hop_layout(rows) -> dict:
    """The packed second read: each row's prompt followed by its re-read slot, positions continuing."""
    prompt, hop, starts, positions = [], [], [0], []
    for r in rows:
        n, at = len(r["input_ids"]), starts[-1]
        prompt += range(at, at + n)
        hop.append(at + n)
        positions += range(n + 1)
        starts.append(at + n + 1)
    host = torch.tensor(prompt + hop + starts + positions, dtype=torch.long).pin_memory()
    p, h, cu, pos = torch.split(host.to("cuda", non_blocking=True), [len(prompt), len(hop), len(starts), len(positions)])
    return {"prompt": p, "hop": h, "cu": cu, "cu_cpu": torch.tensor(starts), "positions": pos,
            "longest": max(len(r["input_ids"]) for r in rows) + 1}


def readout(model, batch, h):
    weight = model.get_output_embeddings().weight.index_select(0, batch["union"])
    scores = F.linear(h.float(), weight.float())
    return scores.gather(1, batch["columns"]).masked_fill(~batch["code_mask"], modeling.MASK)


def first_read(model, batch):
    """Final hidden state [B, H] at each row's decision token (modeling.packed_code_logits before the readout)."""
    text = model.model.language_model
    hidden = text(input_ids=batch["ids"][None], position_ids=batch["positions"][None], cu_seq_lens_q=batch["cu"],
                  cu_seq_lens_k=batch["cu"], max_length_q=batch["longest"], max_length_k=batch["longest"],
                  cu_seqlens_cpu=batch["cu_cpu"], use_cache=False).last_hidden_state
    return hidden[0, batch["last"]]


def second_read(model, batch, layout, h, scale: float):
    text = model.model.language_model
    vector = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) * scale
    embedded = text.embed_tokens(batch["ids"])
    inputs = embedded.new_zeros(int(layout["cu_cpu"][-1]), embedded.shape[-1])
    inputs = inputs.index_copy(0, layout["prompt"], embedded).index_copy(0, layout["hop"], vector.to(embedded.dtype))
    hidden = text(inputs_embeds=inputs[None], position_ids=layout["positions"][None], cu_seq_lens_q=layout["cu"],
                  cu_seq_lens_k=layout["cu"], max_length_q=layout["longest"], max_length_k=layout["longest"],
                  cu_seqlens_cpu=layout["cu_cpu"], use_cache=False).last_hidden_state
    return hidden[0, layout["hop"]]


def packed(rows):
    batch = modeling.pack_rows([r["input_ids"] for r in rows], [r["code_ids"] for r in rows], "cuda")
    kmax = batch["columns"].shape[1]
    targets = torch.tensor([r["target"] + [0.] * (kmax - len(r["target"])) for r in rows]).pin_memory()
    return batch, targets.to("cuda", non_blocking=True)


def logits(model, arm, rows, scale) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(answer logits, first-read logits for hop or None), no gradient bookkeeping."""
    batch, _ = packed(rows)
    if arm != "hop":
        return modeling.packed_code_logits(model, batch), None
    h = first_read(model, batch)
    return readout(model, batch, second_read(model, batch, hop_layout(rows), h, scale)), readout(model, batch, h)


def update(model, opt, arm, encoded, step, scale):
    model.train()
    opt.zero_grad()
    loss_sum, first_sum = 0., 0.
    for group in train.microbatches(encoded, lce.TRAIN_ARGS):
        train.set_checkpointing(model, sum(len(r["input_ids"]) for r in group) > lce.CHECKPOINT_ABOVE)
        batch, targets = packed(group)
        if arm == "hop":
            h = first_read(model, batch)
            first = train.soft_ce(readout(model, batch, h), targets).sum() / len(encoded)
            (first / 2).backward()
            first_sum += first.item()
            z = readout(model, batch, second_read(model, batch, hop_layout(group), h.detach(), scale))
        else:
            z = modeling.packed_code_logits(model, batch)
        loss = train.soft_ce(z, targets).sum() / len(encoded)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        (loss / 2 if arm == "hop" else loss).backward()
        loss_sum += loss.item()
    norm = opt.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError(f"nonfinite gradients at step {step}")
    lr = min(step / WARMUP, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - WARMUP) / (STEPS - WARMUP))))
    for g in opt.param_groups:
        g["lr"] = BODY_LR * g["lr_scale"] * lr
    opt.step()
    return loss_sum, (first_sum if arm == "hop" else None), norm


@torch.no_grad()
def evaluate(model, arm, renderer, rows, scale):
    """lce.evaluate for any arm; hop records also keep the first read's probabilities."""
    model.eval()
    train.set_checkpointing(model, False)
    encoded = [views.view(renderer, r) for r in rows]
    probabilities, first, unsupported = {}, {}, {}
    for r in encoded:
        if len(r["input_ids"]) + (arm == "hop") > long_eval.MAX_CONTEXT:
            unsupported[r["id"]] = "longer than the model context"
    fits = [r for r in encoded if r["id"] not in unsupported]

    def run(group):
        z, z1 = logits(model, arm, group, scale)
        for i, r in enumerate(group):
            probabilities[r["id"]] = z[i].float().softmax(-1)[:len(r["target"])].tolist()
            if z1 is not None:
                first[r["id"]] = z1[i].float().softmax(-1)[:len(r["target"])].tolist()

    for group in train.microbatches(fits, lce.EVAL_ARGS):
        try:
            run(group)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for r in group:
                try:
                    run([r])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    unsupported[r["id"]] = "GPU memory"
    result = []
    for r, e in zip(rows, encoded):
        p = probabilities.get(r["id"])
        pred = None if p is None else max(range(len(p)), key=p.__getitem__)
        record = {"id": r["id"], "source": r["source"], "group": r.get("group", r["id"]), "tokens": len(e["input_ids"]),
                  "prediction": pred, "correct": pred is not None and r["target"][pred] >= max(r["target"]) - 1e-6,
                  "probabilities": p, "unsupported": unsupported.get(r["id"])}
        if arm == "hop":
            q = first.get(r["id"])
            record["first_probabilities"] = q
            record["first_correct"] = q is not None and r["target"][max(range(len(q)), key=q.__getitem__)] >= max(r["target"]) - 1e-6
        result.append(record)
    return search.metrics(result), result


def evaluate_all(model, arm, renderer, dest: Path, scale) -> dict:
    summary = {}
    for name, rows in lce.eval_sets().items():
        started = time.monotonic()
        metric, result = evaluate(model, arm, renderer, rows, scale)
        search.save_records(dest / f"eval-{name}.jsonl", result)
        summary[name] = {"macro": metric["macro"], "accuracy": metric["accuracy"], "n": metric["n"],
                         "unsupported": sum(r["unsupported"] is not None for r in result),
                         "seconds": time.monotonic() - started, "sources": metric["sources"]}
        if arm == "hop":
            summary[name]["first_read_macro"] = search.metrics(first_records(result))["macro"]
        print("EVAL", dest.name, name, round(metric["macro"], 5), "unsupported", summary[name]["unsupported"],
              round(summary[name]["seconds"]), flush=True)
    summary["long_score"] = float(np.mean([summary[name]["macro"] for name in lce.LONG_SETS]))
    return summary


def first_records(result):
    return [{**r, "correct": r["first_correct"]} for r in result]


def pool() -> None:
    """long_eval.pool's sampler over every gold6 source.

    The earlier 100-source list (train-nocap-64k) came from a sampler capped at 1,536 tokens and dropped the
    four sources with no short rows: QuALITY, MuSiQue (multi-hop QA), kv-lookup and API-Bank.
    """
    gold6 = Path("data/mix/v3-gold6.jsonl")
    by = defaultdict(list)
    for r in lt.prior.pilot.read_rows(gold6):
        if lt.prior.old.allowed_origin(r, {"human", "programmatic"}):
            by[r["source"]].append(r)
    pools = {s: sorted(v, key=lambda r: search.digest(["nocap104-v1", r["id"]]))[:1024] for s, v in by.items()}
    quota = dict.fromkeys(sorted(pools), 0)
    while sum(quota.values()) < 65536:
        changed = False
        for s in quota:
            if quota[s] < len(pools[s]) and sum(quota.values()) < 65536:
                quota[s] += 1
                changed = True
        if not changed:
            raise ValueError("insufficient eligible examples")
    rows = sorted([r for s in quota for r in pools[s][:quota[s]]], key=lambda r: r["id"])
    selection = records.rows(lt.OUT / "selection.jsonl")
    lt.prior.pilot.check_disjoint({"train": rows, "selection": selection})
    sha = long_eval.write(POOL, rows)
    POOL.with_suffix(".manifest.json").write_text(json.dumps({
        "rows": len(rows), "sha256": sha, "gold6_sha256": train.sha256_file(gold6),
        "sampler": "per source the first 1,024 rows by sha256(['nocap104-v1', id]), then round-robin quotas to 65,536; "
                   "all gold6 sources with human or programmatic labels; no prompt-length filter",
        "disjoint_from": str(lt.OUT / "selection.jsonl"), "sources": dict(Counter(r["source"] for r in rows))},
        indent=1) + "\n")
    print("POOL", len(rows), len(quota), flush=True)


def check(out: Path) -> None:
    """Kernel checks on base weights before any fit: the causal packed path against stock unpacked attention,
    row isolation under packing, and the hop arm's first read against the noncausal arm."""
    out.mkdir(parents=True, exist_ok=True)
    renderer = lce.renderers()[0]["native"]
    rows = [views.view(renderer, r) for r in records.rows(lt.OUT / "selection.jsonl")[:64:4]]
    pad = renderer.tokenizer.pad_token_id
    report = {}
    with torch.no_grad():
        stock = modeling.load(BASE, attention="causal", dtype=torch.bfloat16).eval()
        reference, _ = train.logits_for(stock, rows, pad, packed=False)
        del stock
        torch.cuda.empty_cache()
        for arm in ARMS:
            model = load_model(arm).eval()
            train.set_checkpointing(model, False)
            scale = embedding_rms(model)
            z, z1 = logits(model, arm, rows, scale)
            altered = [{**rows[1], "input_ids": rows[1]["input_ids"][:-8] + rows[1]["input_ids"][-4:]}] + rows[2:]
            isolated, _ = logits(model, arm, [rows[0]] + altered, scale)
            entry = {"row_isolation_max_diff": float((isolated[0] - z[0]).abs().max())}
            if arm == "causal":
                real = reference > modeling.MASK / 2
                entry["vs_stock_unpacked_max_diff"] = float((z - reference)[real].abs().max())
                entry["vs_stock_argmax_agree"] = float((z.argmax(-1) == reference.argmax(-1)).float().mean())
            if arm == "noncausal":
                noncausal = z
            if arm == "hop":
                entry["first_read_vs_noncausal_max_diff"] = float((z1 - noncausal).abs().max())
                entry["embedding_rms"] = scale
            report[arm] = entry
            print("CHECK", arm, json.dumps(entry), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    train.atomic_json(out / "check.json", report)


def smoke(out: Path) -> None:
    """Matched views across arms, timing of the first real updates, and the longest rows through one update."""
    rows = records.rows(POOL)
    renderer = lce.renderers()[0]["native"]
    schedule = list(train.batches(len(rows), BATCH_SIZE, 16, random.Random(41)))
    encoded_all = [views.view(renderer, r) for r in rows]
    longest = sorted(range(len(rows)), key=lambda i: len(encoded_all[i]["input_ids"]))[-3:]
    report = {"pool_tokens": {"max": max(len(e["input_ids"]) for e in encoded_all),
                              "mean": float(np.mean([len(e["input_ids"]) for e in encoded_all])),
                              "over_1536": sum(len(e["input_ids"]) > 1536 for e in encoded_all)}}
    del encoded_all
    for arm in ARMS:
        train.seed_all(41)
        model = load_model(arm)
        scale = embedding_rms(model)
        opt = train.MasterAdamW(model, BODY_LR)
        digests, times = [], []
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = lce.step_rows(renderer, rows, indices, 41, step)
            digests.append(lce.view_digest(encoded))
            tick = time.monotonic()
            loss, first, _ = update(model, opt, arm, encoded, step, scale)
            torch.cuda.synchronize()
            times.append({"seconds": time.monotonic() - tick, "tokens": sum(len(r["input_ids"]) for r in encoded),
                          "loss": loss, "first_loss": first})
        encoded = [views.view(renderer, rows[i]) for i in longest]
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        update(model, opt, arm, encoded, 17, scale)
        torch.cuda.synchronize()
        long_update = {"tokens": [len(r["input_ids"]) for r in encoded], "seconds": time.monotonic() - tick,
                       "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        probe = [r for r in lce.eval_sets()["babilong"] if r["source"] == "babilong-128k"][:2]
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        _, result = evaluate(model, arm, renderer, probe, scale)
        eval_probe = {"rows": len(probe), "seconds": time.monotonic() - tick, "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
                      "unsupported": sum(r["unsupported"] is not None for r in result)}
        median = float(np.median([t["seconds"] for t in times[4:]]))
        report[arm] = {"view_digests": digests, "updates": times, "median_update_s": median,
                       "estimated_fit_hours": median * STEPS / 3600, "long_update": long_update, "eval_128k": eval_probe}
        print("SMOKE", arm, json.dumps({k: v for k, v in report[arm].items() if k not in ("view_digests", "updates")}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report["matched_views"] = all(report[arm]["view_digests"] == report["causal"]["view_digests"] for arm in ARMS)
    train.atomic_json(out / "smoke.json", report)
    if not report["matched_views"]:
        raise ValueError("arms see different rows or views")


def implementation() -> dict:
    modules = [modeling, train, views, lce, long_eval, search, records, lt]
    paths = [Path(m.__file__) for m in modules] + [Path(__file__), Path("qwen_decision/v3/format.py")]
    return {str(p): train.sha256_file(p) for p in paths}


def freeze(out: Path) -> dict:
    """The preregistered plan, written once before any fit."""
    path = out / "plan.json"
    if path.exists():
        raise ValueError("plan already frozen")
    checked, smoked = read(out / "check.json"), read(out / "smoke.json")
    if not smoked["matched_views"]:
        raise ValueError("smoke did not match views")
    plan = {"experiment": "from-base-structures-v1", "base": str(BASE),
            "base_sha256": train.sha256_file(BASE / "model.safetensors-00001-of-00001.safetensors"),
            "pool": str(POOL), "pool_sha256": train.sha256_file(POOL),
            "test_manifest_sha256": train.sha256_file(long_eval.TEST / "manifest.json"),
            "audit_sha256": train.sha256_file(long_eval.OUT / "audit.json"),
            "arms": {"causal": "stock Qwen3.5, every layer left to right",
                     "noncausal": "d3: the 6 full-attention layers bidirectional, DeltaNet causal",
                     "hop": "noncausal + one decision-point re-read (module docstring); answer from the second read"},
            "hop_embedding_rms": checked["hop"]["embedding_rms"],
            "seeds": list(SEEDS), "order": "seed-major: every arm of a seed before the next seed",
            "steps": STEPS, "batch_size": BATCH_SIZE, "body_lr": BODY_LR, "warmup": WARMUP,
            "schedule": "linear warmup then cosine to 10%", "optimizer": "FP32-master AdamW betas(.9,.95), WD.01, clip 1",
            "loss": "gold soft CE over shown codes; hop: half on each read, second read takes the first read's state detached",
            "trainable": "whole language backbone incl. tied embedding/LM head; vision tower unused",
            "packing": "varlen packed rows, no truncation", "checkpoint_above_tokens": lce.CHECKPOINT_ABOVE,
            "views": "lce.step_rows (robust_adaptation.random_view), identical across arms, checked per step",
            "metrics": {"selection": "macro accuracy over sources, 4,147 items",
                        "long": "mean of macro accuracy over " + ", ".join(lce.LONG_SETS)},
            "comparisons": COMPARISONS,
            "decision": "for each comparison and each metric separately: B beats A if all 3 seed deltas are positive "
                        "and the hierarchical seed x item bootstrap 95% interval (lce.hierarchical) excludes 0. "
                        "Six tests, no multiplicity correction; claims name the metric.",
            "reported_only": "per set, BABILong by task and length, LongBench v2 short, hop first read vs second read",
            "implementation": implementation(), "check": checked,
            "smoke": {arm: {k: smoked[arm][k] for k in ("median_update_s", "estimated_fit_hours", "long_update", "eval_128k")}
                      for arm in ARMS} | {"pool_tokens": smoked["pool_tokens"]}}
    train.atomic_json(path, plan)
    return plan


def fit(out: Path, arm: str, seed: int) -> None:
    plan = read(out / "plan.json")
    if implementation() != plan["implementation"]:
        raise ValueError("frozen implementation changed")
    dest = out / f"{arm}-seed{seed}"
    if (dest / "result.json").exists():
        print("SKIP_COMPLETE", arm, seed, flush=True)
        return
    if dest.exists():
        dest.rename(out / f"{dest.name}.incomplete-{time.time_ns()}")
    dest.mkdir()
    rows = records.rows(POOL)
    if train.sha256_file(POOL) != plan["pool_sha256"]:
        raise ValueError("training pool changed")
    renderer = lce.renderers()[0]["native"]
    reference = None if arm == ARMS[0] else records.rows(out / f"{ARMS[0]}-seed{seed}/train.jsonl")
    scale = plan["hop_embedding_rms"]
    train.seed_all(seed)
    model = load_model(arm)
    opt = train.MasterAdamW(model, BODY_LR)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    schedule = train.batches(len(rows), BATCH_SIZE, STEPS, random.Random(seed))
    with (dest / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = lce.step_rows(renderer, rows, indices, seed, step)
            digest = lce.view_digest(encoded)
            if reference is not None and digest != reference[step - 1]["view_sha256"]:
                raise ValueError(f"step {step}: rows or views differ from {ARMS[0]}")
            loss, first, norm = update(model, opt, arm, encoded, step, scale)
            entry = {"step": step, "loss": loss, "first_read_loss": first, "grad_norm": norm, "view_sha256": digest,
                     "tokens": sum(len(r["input_ids"]) for r in encoded), "seconds": time.monotonic() - started}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 32 == 0:
                print("FIT", arm, seed, step, round(loss, 4), round(entry["seconds"]), flush=True)
                train.atomic_json(out / "status.json", {"state": "training", "arm": arm, "seed": seed, "step": step,
                                                        "steps": STEPS, "seconds": entry["seconds"]})
    trained_seconds = time.monotonic() - started
    _, config = lce.renderers()
    modeling.save(model, renderer.tokenizer, dest / "model",
                  config | {"attention": "causal" if arm == "causal" else "noncausal", "layout": "d3", "steps": STEPS,
                            "trained_from": str(BASE), "experiment": plan["experiment"], "arm": arm, "seed": seed,
                            "hop_embedding_rms": scale if arm == "hop" else None})
    peak = torch.cuda.max_memory_allocated() / 1e9
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    train.atomic_json(out / "status.json", {"state": "evaluating", "arm": arm, "seed": seed})
    summary = evaluate_all(model, arm, renderer, dest, scale)
    train.atomic_json(dest / "result.json", {"arm": arm, "seed": seed, "train_seconds": trained_seconds,
                                             "peak_cuda_gb": peak, "eval": summary})
    print("RESULT", arm, seed, round(summary["long_score"], 5), round(summary["selection"]["macro"], 5), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report(out: Path) -> None:
    plan = read(out / "plan.json")
    names = list(lce.eval_sets())
    done = [s for s in plan["seeds"] if all((out / f"{arm}-seed{s}/result.json").exists() for arm in ARMS)]
    load = lambda arm, seed: {name: records.rows(out / f"{arm}-seed{seed}/eval-{name}.jsonl") for name in names}
    evals = {arm: {s: load(arm, s) for s in done} for arm in ARMS}
    result = {"plan_sha256": train.sha256_file(out / "plan.json"), "completed_seeds": done,
              "scores": {arm: {s: read(out / f"{arm}-seed{s}/result.json")["eval"] for s in done} for arm in ARMS},
              "comparisons": {}}
    for label, (a, b) in COMPARISONS.items():
        if not done:
            break
        entry = {"selection": lce.hierarchical(evals[a], evals[b], ["selection"]),
                 "long": lce.hierarchical(evals[a], evals[b], lce.LONG_SETS),
                 "per_set": {name: lce.hierarchical(evals[a], evals[b], [name], draws=2000) for name in names}}
        for metric in ("selection", "long"):
            entry[f"{metric}_better"] = (len(done) == len(plan["seeds"]) and entry[metric]["all_seeds_positive"]
                                         and entry[metric]["ci95"][0] > 0)
        result["comparisons"][label] = entry
        print("COMPARE", label, json.dumps({m: entry[m] for m in ("selection", "long")}), flush=True)
    if done:
        first = {s: {name: first_records(v) for name, v in evals["hop"][s].items()} for s in done}
        result["hop_second_vs_first_read"] = {"selection": lce.hierarchical(first, evals["hop"], ["selection"]),
                                              "long": lce.hierarchical(first, evals["hop"], lce.LONG_SETS)}
        result["babilong_by_task"] = {arm: {s: {task: search.metrics([r for r in evals[arm][s]["babilong"]
                                                                      if r["group"].split(":")[1] == task])["accuracy"]
                                                 for task in sorted({r["group"].split(":")[1] for r in evals[arm][s]["babilong"]})}
                                            for s in done} for arm in ARMS}
    train.atomic_json(out / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pool", "check", "smoke", "freeze", "run", "report"])
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "pool":
        pool()
    elif args.stage == "check":
        check(args.output)
    elif args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "report":
        report(args.output)
    else:
        for seed in map(int, args.seeds.split(",")):
            for arm in ARMS:
                fit(args.output, arm, seed)
            report(args.output)


if __name__ == "__main__":
    main()
