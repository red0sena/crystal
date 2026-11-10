"""R1/R2: the v2 recipe (16,384 updates, LR 2e-4, warmup 512, cosine to 10%) on the full v3 pool.

Only the text source mixture differs between the arms:
  r1  previous v2 source shares (b_s) for the 112 old sources; the new URL source keeps its natural share.
  r2  natural row shares of the full pool (p_s), i.e. the v3 mixture.
Everything else (seed, LoRA init, images, augmentation, optimizer, schedule) is identical.
R2 vs v3 epoch1 isolates the schedule; R1 vs R2 isolates the mixture; v2 is the fixed reference.
Evaluation uses the unmerged BF16 adapter on the immutable v3 dev files; no public checkpoint choice.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from qwen_decision.v3 import champion_full_inputs as data, champion_full_train as fit
from qwen_decision.v3 import champion_audit as audit, train, vision_train as vt

ROOT = Path("runs/champion-v1/mix-ablation-v1")
V3 = data.ROOT
V2 = Path("runs/champion-v1/joint-g7-v2")
AUDIT = V3 / "cause-investigation/data-and-optimization-audit.json"
ARMS = ("r1", "r2")
SEED = data.SEED
STEPS, WARMUP, LR = 16384, 512, 2e-4
TEXT_TAKE, IMAGE_TAKE = 16, 4
URL = "phishing-url-train"
CHECKPOINT_EVERY = 4096
CFG = fit.CFG


def arm_root(arm):
    assert arm in ARMS
    return ROOT / arm


def paused():
    return (ROOT / "pause-request.json").exists()


# ---------------------------------------------------------------- mixture

def target_shares(arm, full_rows, old_rows):
    """q_s over the full-pool sources. r1 keeps URL at its natural share and rescales v2 shares for the rest."""
    n = sum(full_rows.values())
    natural = {s: c / n for s, c in full_rows.items() if c}
    if arm == "r2":
        return natural
    assert arm == "r1"
    url = natural.get(URL, 0.)
    old = {s: old_rows.get(s, 0) for s in natural if s != URL}
    missing = [s for s, c in old_rows.items() if c and s not in natural]
    assert not missing, f"v2 sources absent from the full pool: {missing}"
    total = sum(old.values())
    q = {s: (1 - url) * c / total for s, c in old.items() if c}
    if url:
        q[URL] = url
    return q


def quotas(total, shares):
    """Largest-remainder integer draws per source; sums exactly to total."""
    norm = sum(shares.values())
    raw = {s: total * w / norm for s, w in shares.items()}
    out = {s: int(math.floor(v)) for s, v in raw.items()}
    left = total - sum(out.values())
    for s in sorted(raw, key=lambda s: (-(raw[s] - out[s]), s))[:left]:
        out[s] += 1
    assert sum(out.values()) == total
    return out


def _rng(*key):
    return np.random.default_rng(int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:16], 16))


def text_draws(sources, names, arm_quotas, seed=SEED):
    """Global row indices. Within a source rows come from shuffled passes without replacement, so a source
    repeats only after all its rows were used; then all draws are shuffled together."""
    by = {}
    for code, name in enumerate(names):
        by[name] = np.flatnonzero(sources == code)
    picks = []
    for name in sorted(arm_quotas):
        need, rows = arm_quotas[name], by[name]
        assert need == 0 or len(rows), name
        passes, got = 0, []
        while need > 0:
            perm = rows[_rng(seed, "source", name, passes).permutation(len(rows))]
            got.append(perm[:need])
            need -= len(perm[:need])
            passes += 1
        picks += got
    draws = np.concatenate(picks) if picks else np.zeros(0, dtype=np.int64)
    return draws[_rng(seed, "text-order").permutation(len(draws))]


class MixSchedule:
    """Fixed-length 16 text + 4 image updates; image cycles match the v3 schedule's independent shuffles."""
    def __init__(self, texts, images, draws, steps=STEPS, seed=SEED, text_take=TEXT_TAKE, image_take=IMAGE_TAKE):
        assert len(draws) == steps * text_take and len(images)
        self.texts, self.images, self.draws = texts, images, draws
        self.steps, self.seed, self.text_take, self.image_take = steps, seed, text_take, image_take
        self.per_epoch = steps  # train.jsonl "epoch" field = fraction of this schedule
        self.orders = {}

    def __len__(self):
        return self.steps

    def image_order(self, cycle):
        if cycle not in self.orders:
            self.orders[cycle] = _rng(self.seed, "image", cycle).permutation(len(self.images))
        return self.orders[cycle]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows = [self.texts[int(i)] for i in self.draws[index*self.text_take:(index+1)*self.text_take]]
        for j in range(self.image_take):
            cycle, at = divmod(index*self.image_take+j, len(self.images))
            rows.append(self.images[int(self.image_order(cycle)[at])])
        random.Random(f"{self.seed}:batch:{index}").shuffle(rows)
        return [(r, f"mix-view:{self.seed}:{index}:{j}") for j, r in enumerate(rows)]


def source_index():
    """Per-row source code for the exact v3 training text parts (cached, verified against the cause audit)."""
    out, meta = ROOT / "text-sources.npy", ROOT / "text-sources.json"
    p = fit.prepared_inputs()
    files = [r["path"] for r in p["text_parts"]]
    signature = {f: audit.sha(f) for f in files}
    if out.exists() and meta.exists():
        m = audit.read(meta)
        if m["files"] == signature and m["array_sha256"] == audit.sha(out):
            return np.load(out), m["names"]
    loads = json.loads
    codes, names, labels = {}, [], []
    for f in files:
        with open(f, "rb") as handle:
            for line in handle:
                s = loads(line)["source"]
                if s not in codes:
                    codes[s] = len(names)
                    names.append(s)
                labels.append(codes[s])
    sources = np.asarray(labels, dtype=np.int16)
    assert len(sources) == p["text_rows"]
    expected = {s: v["full_rows"] for s, v in audit.read(AUDIT)["sources"].items() if v["full_rows"]}
    counted = {names[c]: int(n) for c, n in zip(*np.unique(sources, return_counts=True))}
    assert counted == expected, "Source counts differ from the cause-investigation audit"
    ROOT.mkdir(parents=True, exist_ok=True)
    np.save(out, sources)
    train.atomic_json(meta, {"files": signature, "names": names, "rows": len(sources), "counts": counted,
                             "array_sha256": audit.sha(out)})
    return sources, names


def mixture(arm):
    sources, names = source_index()
    a = audit.read(AUDIT)["sources"]
    full = {s: v["full_rows"] for s, v in a.items()}
    old = {s: v["old_rows"] for s, v in a.items()}
    q = target_shares(arm, full, old)
    need = quotas(STEPS * TEXT_TAKE, q)
    return sources, names, q, need


# ---------------------------------------------------------------- plan

def implementation():
    result = fit.implementation()
    result[str(Path(__file__))] = audit.sha(__file__)
    return result


def freeze(arm):
    root = arm_root(arm)
    plan_path = root / "plan.json"
    if plan_path.exists():
        return verify(arm)
    root.mkdir(parents=True, exist_ok=True)
    v3 = fit.verify()  # v3 inputs, base and frozen v3 training code unchanged
    sources, names, q, need = mixture(arm)
    texts, images = fit.stores()
    draws = text_draws(sources, names, need)
    np.save(root / "text-draws.npy", draws)
    full = Counter(names[int(c)] for c in sources)
    exposure = {s: {"target_share": q.get(s, 0.), "draws": need.get(s, 0), "rows": full[s],
                    "passes": need.get(s, 0) / full[s]} for s in sorted(full)}
    plan = {"experiment": f"mix-ablation-v1-{arm}", "arm": arm, "base": str(data.BASE), "base_sha256": audit.BASE_SHA,
            "seed": SEED, "steps": STEPS, "warmup_steps": WARMUP, "learning_rate": LR,
            "lr_schedule": "linear warmup, cosine decay to10% (v2 formula)", "text_per_update": TEXT_TAKE,
            "image_per_update": IMAGE_TAKE, "text_draws": int(len(draws)), "unique_text_rows": int(len(np.unique(draws))),
            "image_presentations": STEPS * IMAGE_TAKE, "image_passes": STEPS * IMAGE_TAKE / len(images),
            "text_pool_rows": len(texts), "image_pool_rows": len(images),
            "mixture": "previous v2 source shares; URL natural share" if arm == "r1" else "natural full-pool row shares",
            "source_exposure": exposure, "text_draws_sha256": audit.sha(root / "text-draws.npy"),
            "text_sources_sha256": audit.sha(ROOT / "text-sources.npy"),
            "trainable": v3["trainable"], "optimizer": v3["optimizer"], "config": vars(CFG),
            "inputs": v3["inputs"], "image_hash_manifest": v3["image_hash_manifest"], "v3_plan_sha256": audit.sha(fit.PLAN),
            "implementation": implementation(),
            "checkpointing": f"every{CHECKPOINT_EVERY} and final; adapters+FP32 masters+optimizer+RNG",
            "not_replicated_from_v2": ["within-source class balancing", "presentation rewrites",
                                       "v2 text padding-option pool (v3 pipeline disables padding)", "v2 image pool"],
            "selection": "fixed final update; no public checkpoint choice", "official_rank": None}
    train.atomic_json(plan_path, plan)
    snapshot = root / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    return verify(arm)


def verify(arm):
    root = arm_root(arm)
    p = audit.read(root / "plan.json")
    assert p["implementation"] == implementation(), "Training source changed after freeze"
    fit.verify()
    assert audit.sha(root / "text-draws.npy") == p["text_draws_sha256"]
    assert audit.sha(ROOT / "text-sources.npy") == p["text_sources_sha256"]
    return p


# ---------------------------------------------------------------- training

def worker_init(_):
    fit.bind()
    torch.set_num_threads(1)


def lr_factor(step, warm=WARMUP, n=STEPS):
    return min(step/warm, 1.)*(.1+.45*(1+math.cos(math.pi*max(0, step-warm)/max(1, n-warm))))


def run(arm):
    fit.bind()
    root = arm_root(arm)
    plan = verify(arm)
    if (root / "report.json").exists():
        return
    fit.checkpoint.ROOT = root  # save/restore checkpoints under this arm
    train.seed_all(SEED)
    model = fit.legacy.load_model()
    opt = train.MasterAdamW(model, plan["learning_rate"])
    start_step, tokens = fit.checkpoint.restore_checkpoint(model, opt)
    texts, images = fit.stores()
    schedule = MixSchedule(texts, images, np.load(root / "text-draws.npy"))
    dataset = vt.MixSteps(schedule, CFG, [])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(start_step, len(dataset))), batch_size=None,
        num_workers=CFG.workers, pin_memory=True, prefetch_factor=2, worker_init_fn=worker_init,
        multiprocessing_context="spawn", generator=torch.Generator().manual_seed(SEED))
    log_path = root / "train.jsonl"
    if log_path.exists():
        prior = [r for r in audit.rows(log_path) if r["step"] <= start_step]
        log_path.write_text("".join(json.dumps(r)+"\n" for r in prior))
    model.train()
    recent, started = deque(maxlen=256), time.monotonic()
    previous = started
    with log_path.open("a", buffering=1) as log:
        for step, groups in enumerate(loader, start_step+1):
            if paused():
                train.atomic_json(root / "status.json", {"stage": "paused_by_request", "step": step-1})
                raise SystemExit("pause-request.json present; stopping before update " + str(step))
            opt.zero_grad()
            loss_sum, batch_tokens = 0., 0
            nrows = sum(int(b["targets"].shape[0]) for b in groups)
            for cpu in groups:
                batch = vt.to_device(cpu)
                train.set_checkpointing(model, len(cpu["ids"]) > CFG.checkpoint_above)
                loss = train.soft_ce(vt.packed_logits(model, batch), batch["targets"]).sum()/nrows
                assert torch.isfinite(loss)
                loss.backward()
                loss_sum += float(loss.detach())
                batch_tokens += len(cpu["ids"])
            norm = opt.clip_grad_norm(1.)
            assert math.isfinite(norm)
            for group in opt.param_groups:
                group["lr"] = plan["learning_rate"]*group["lr_scale"]*lr_factor(step)
            opt.step()
            tokens += batch_tokens
            now = time.monotonic()
            record = {"step": step, "steps": STEPS, "loss": loss_sum, "grad_norm": norm, "rows": nrows,
                      "tokens": batch_tokens, "total_tokens": tokens, "lr": plan["learning_rate"]*lr_factor(step),
                      "step_seconds": now-previous}
            previous = now
            if step > start_step+16:
                recent.append(record["step_seconds"])
            log.write(json.dumps(record)+"\n")
            if step % 16 == 0:
                state = {"stage": "training", "arm": arm, **record}
                if len(recent) >= 32:
                    state["eta"] = fit.eta_state(record, recent)
                train.atomic_json(root / "status.json", state)
                print("TRAIN", arm, record, flush=True)
            if step == 64 or step % CHECKPOINT_EVERY == 0 or step == STEPS:
                assert fit.checkpoint.ROOT == root, "checkpoint target must stay inside this arm"
                fit.checkpoint.save_checkpoint(model, opt, step, tokens)
    final = audit.read(root / "last-checkpoint.json")
    assert final["step"] == STEPS
    train.atomic_json(root / "report.json", {"complete": True, "arm": arm, "steps": STEPS, "total_train_tokens": tokens,
                      "final_checkpoint": final, "seconds_this_process": time.monotonic()-started})
    train.atomic_json(root / "status.json", {"stage": "trained", "arm": arm, "steps": STEPS})


# ---------------------------------------------------------------- evaluation

def adapter_for(arm):
    if arm == "v2":
        path = V2 / "adapter.safetensors"
        return path, audit.sha(path)
    final = audit.read(arm_root(arm) / "report.json")["final_checkpoint"]
    path = Path(final["directory"]) / "adapter.safetensors"
    assert audit.sha(path) == final["files"]["adapter.safetensors"]
    return path, final["files"]["adapter.safetensors"]


def load_adapter_model(arm):
    from safetensors.torch import load_file
    model = fit.legacy.load_model()
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert len(params) == 372
    path, _ = adapter_for(arm)
    stored = {n: t for n, t in load_file(str(path), device="cpu").items() if not n.startswith("master.")}
    assert set(stored) == set(params)
    with torch.no_grad():
        for n, p in params.items():
            assert p.shape == stored[n].shape and p.dtype == stored[n].dtype
            p.copy_(stored[n])
    train.set_checkpointing(model, False)
    model.requires_grad_(False)
    model.eval()
    return model


def evaluate_all():
    from qwen_decision.v3 import champion_epoch1_eval as ev
    fit.bind()
    eval_root = ROOT / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    ev.ROOT = eval_root  # predictions under ROOT/eval/<arm>/<dataset>.jsonl; same scorer as the epoch1 comparison
    prepared = audit.read(V3 / "prepared.json")
    sets = {name: ev.dev_rows(path) for name, path in prepared["evaluation"].items()}
    results = {}
    for arm in ("v2", *ARMS):
        if paused():
            raise SystemExit("pause-request.json present")
        model = load_adapter_model(arm)
        results[arm] = {"adapter_sha256": adapter_for(arm)[1]}
        for name, rows in sets.items():
            results[arm][name] = {k: v for k, v in ev.evaluate(model, arm, name, rows).items() if k != "sources"}
            train.atomic_json(eval_root / "results.json", results)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    compare(sets)


def v2_overlap(sets):
    """Rows of the v3 dev files that v2 could have trained on (text by item/id, image by file name or id)."""
    import os
    items, ids = set(), set()
    for r in audit.rows(V2 / "mix/text-train.jsonl"):
        items.add(r.get("item"))
        ids.update((r["id"], r.get("original_id")))
    names, image_ids = set(), set()
    for r in audit.rows(V2 / "mix/image-train.jsonl"):
        names.update(os.path.basename(p) for p in r["images"])
        image_ids.add(r["id"])
    items.discard(None)
    ids.discard(None)

    def seen(r):
        if r["modality"] == "text":
            return r.get("item") in items or r["id"] in ids
        return r["id"] in image_ids or any(os.path.basename(p) in names for p in r["images"])
    return {name: sum(map(seen, rows)) for name, rows in sets.items()}


def compare(sets=None):
    """Row and source-macro accuracy per arm on the identical immutable v3 dev files."""
    eval_root = ROOT / "eval"
    if sets is None:
        from qwen_decision.v3 import champion_epoch1_eval as ev
        sets = {name: ev.dev_rows(path) for name, path in audit.read(V3 / "prepared.json")["evaluation"].items()}
    arms = {"v2": eval_root / "v2", "r1": eval_root / "r1", "r2": eval_root / "r2",
            "v3-epoch1": V3 / "epoch1-eval/epoch1"}
    out = {"dev_rows": {n: len(r) for n, r in sets.items()}, "dev_rows_possibly_in_v2_training": v2_overlap(sets),
           "arms": {}}
    for arm, folder in arms.items():
        out["arms"][arm] = {}
        for name in sets:
            path = folder / (name + ".jsonl")
            if not path.exists():
                continue
            preds = list(audit.rows(path))
            if len(preds) != len(sets[name]):
                continue
            entry = {"rows": len(preds), "accuracy": sum(p["correct"] for p in preds)/len(preds)}
            by = {}
            for p in preds:
                by.setdefault(p["source"], []).append(p["correct"])
            entry["macro_accuracy"] = sum(sum(v)/len(v) for v in by.values())/len(by)
            entry["source_accuracy"] = {s: sum(v)/len(v) for s, v in sorted(by.items())}
            out["arms"][arm][name] = entry
    out["scope"] = ("internal dev, unmerged BF16 adapters, same scorer; v2-overlap check is exact id/item/file-name only, "
                    "not near-duplicate proof; not public or official index")
    train.atomic_json(eval_root / "comparison.json", out)
    print("COMPARISON", json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sources", "freeze", "run", "eval", "compare"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    try:
        if args.command == "sources":
            source_index()
        elif args.command == "freeze":
            for arm in ([args.arm] if args.arm else ARMS):
                freeze(arm)
        elif args.command == "run":
            run(args.arm)
        elif args.command == "eval":
            evaluate_all()
        else:
            compare()
    except BaseException:
        ROOT.mkdir(parents=True, exist_ok=True)
        train.atomic_json(ROOT / "failure.json", {"stage": args.command, "arm": args.arm, "traceback": traceback.format_exc()})
        raise
