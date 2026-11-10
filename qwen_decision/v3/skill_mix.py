"""R3 = R1 (v2 source shares, 16,384-update v2 schedule) + two programmatic skill sources at fixed shares.

Only addition versus R1: home-policy-synth and symbolic-chord-synth (qwen_decision/v3/skill_synth.py), each SHARE of
text draws; the other sources keep R1's relative shares. Seed, LoRA init, images, optimizer and schedule are R1's.
R3 vs R1 measures the skill sources. Held-out synthetic dev uses a different generator seed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from qwen_decision.v3 import champion_full_inputs as data, champion_full_train as fit, mix_ablation as m
from qwen_decision.v3 import champion_audit as audit, skill_synth as synth, train, vision_train as vt, format as fmt

ROOT = Path("runs/champion-v1/skill-mix-v1/r3")
SHARE = {"home-policy-synth": .02, "symbolic-chord-synth": .02}
FILES = {"home-policy-synth": synth.ROOT / "home-policy-train.jsonl",
         "symbolic-chord-synth": synth.ROOT / "symbolic-chord-train.jsonl"}
DEV = {"home-policy-synth": synth.ROOT / "home-policy-dev.jsonl",
       "symbolic-chord-synth": synth.ROOT / "symbolic-chord-dev.jsonl"}
DEV_SEED = 777001


def generate():
    """Train and dev files (dev from another seed); skip when present with recorded hashes."""
    proof = synth.ROOT / "generation.json"
    if proof.exists():
        p = audit.read(proof)
        assert all(audit.sha(f) == h for f, h in p["files"].items())
        return p
    synth.write(FILES["home-policy-synth"], synth.home_rows(3000))
    synth.write(FILES["symbolic-chord-synth"], synth.chord_rows(12000))
    synth.write(DEV["home-policy-synth"], synth.home_rows(200, seed=DEV_SEED))
    synth.write(DEV["symbolic-chord-synth"], synth.chord_rows(1000, seed=DEV_SEED))
    train_items = {json.loads(l)["state"].__repr__() for f in FILES.values() for l in open(f)}
    dev_items = {json.loads(l)["state"].__repr__() for f in DEV.values() for l in open(f)}
    files = {str(f): audit.sha(f) for f in [*FILES.values(), *DEV.values()]}
    p = {"files": files, "generator_sha256": audit.sha(synth.__file__), "seed": synth.SEED, "dev_seed": DEV_SEED,
         "train_dev_identical_states": len(train_items & dev_items),
         "rows": {str(f): sum(1 for _ in open(f)) for f in [*FILES.values(), *DEV.values()]}}
    assert p["train_dev_identical_states"] == 0
    train.atomic_json(proof, p)
    return p


def build_index(path):
    """(offset, bytes, rendered prompt tokens) per row, written next to the file; no truncation."""
    from transformers import AutoTokenizer
    dest = Path(str(path).replace(".jsonl", ".index.npy"))
    proof = dest.with_suffix(".json")
    if proof.exists():
        p = audit.read(proof)
        if p["file_sha256"] == audit.sha(path) and p["index_sha256"] == audit.sha(dest):
            return str(dest)
    tok = AutoTokenizer.from_pretrained(data.BASE, local_files_only=True)
    codes, _ = fmt.answer_codes(tok)
    out, longest = [], 0
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            r = json.loads(raw)
            n = len(tok.encode(fmt.render(tok, r["state"], r["question"], codes[:len(r["target"])]), add_special_tokens=False))
            longest = max(longest, n)
            out.append((offset, len(raw), n))
    np.save(dest, np.asarray(out, dtype=np.uint64))
    train.atomic_json(proof, {"file_sha256": audit.sha(path), "index_sha256": audit.sha(dest), "rows": len(out),
                              "max_tokens": longest})
    return str(dest)


def stores():
    p = fit.prepared_inputs()
    files = [r["path"] for r in p["text_parts"]] + [str(FILES[s]) for s in sorted(FILES)]
    indices = [r["index"] for r in p["text_parts"]] + [build_index(FILES[s]) for s in sorted(FILES)]
    texts = data.IndexedRows(files, indices)
    _, images = fit.stores(p)
    return texts, images


def sources():
    base, names = m.source_index()
    names = list(names)
    parts = [base]
    for s in sorted(FILES):
        names.append(s)
        n = sum(1 for _ in open(FILES[s]))
        parts.append(np.full(n, len(names)-1, dtype=np.int16))
    return np.concatenate(parts), names


def shares():
    a = audit.read(m.AUDIT)["sources"]
    q = m.target_shares("r1", {s: v["full_rows"] for s, v in a.items()}, {s: v["old_rows"] for s, v in a.items()})
    rest = 1 - sum(SHARE.values())
    return {**{s: w*rest for s, w in q.items()}, **SHARE}


def implementation():
    result = fit.implementation()
    for mod in (m, synth):
        result[str(Path(mod.__file__))] = audit.sha(mod.__file__)
    result[str(Path(__file__))] = audit.sha(__file__)
    return result


def freeze():
    plan_path = ROOT / "plan.json"
    if plan_path.exists():
        return verify()
    ROOT.mkdir(parents=True, exist_ok=True)
    v3 = fit.verify()
    gen = generate()
    texts, images = stores()
    src, names = sources()
    assert len(src) == len(texts)
    q = shares()
    need = m.quotas(m.STEPS * m.TEXT_TAKE, q)
    draws = m.text_draws(src, names, need)
    np.save(ROOT / "text-draws.npy", draws)
    full = Counter(names[int(c)] for c in src)
    plan = {"experiment": "skill-mix-v1-r3", "base": str(data.BASE), "base_sha256": audit.BASE_SHA, "seed": m.SEED,
            "steps": m.STEPS, "warmup_steps": m.WARMUP, "learning_rate": m.LR, "lr_schedule": "R1/v2 cosine to 10%",
            "text_draws": int(len(draws)), "text_pool_rows": len(texts), "image_pool_rows": len(images),
            "skill_shares": SHARE, "generation": gen,
            "source_exposure": {s: {"target_share": q.get(s, 0.), "draws": need.get(s, 0), "rows": full[s]} for s in sorted(full)},
            "text_draws_sha256": audit.sha(ROOT / "text-draws.npy"), "inputs": v3["inputs"],
            "trainable": v3["trainable"], "optimizer": v3["optimizer"], "config": vars(m.CFG),
            "implementation": implementation(), "selection": "fixed final update", "official_rank": None}
    train.atomic_json(plan_path, plan)
    (ROOT / "source-snapshot").mkdir(exist_ok=True)
    for f in (__file__, synth.__file__):
        shutil.copyfile(f, ROOT / "source-snapshot" / Path(f).name)
    return verify()


def verify():
    p = audit.read(ROOT / "plan.json")
    assert p["implementation"] == implementation(), "Training source changed after freeze"
    fit.verify()
    assert audit.sha(ROOT / "text-draws.npy") == p["text_draws_sha256"]
    for f, h in p["generation"]["files"].items():
        assert audit.sha(f) == h, f
    return p


def run():
    fit.bind()
    plan = verify()
    if (ROOT / "report.json").exists():
        return
    fit.checkpoint.ROOT = ROOT
    train.seed_all(m.SEED)
    model = fit.legacy.load_model()
    opt = train.MasterAdamW(model, plan["learning_rate"])
    start_step, tokens = fit.checkpoint.restore_checkpoint(model, opt)
    texts, images = stores()
    schedule = m.MixSchedule(texts, images, np.load(ROOT / "text-draws.npy"))
    dataset = vt.MixSteps(schedule, m.CFG, [])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(start_step, len(dataset))), batch_size=None,
        num_workers=m.CFG.workers, pin_memory=True, prefetch_factor=2, worker_init_fn=m.worker_init,
        multiprocessing_context="spawn", generator=torch.Generator().manual_seed(m.SEED))
    log_path = ROOT / "train.jsonl"
    if log_path.exists():
        prior = [r for r in audit.rows(log_path) if r["step"] <= start_step]
        log_path.write_text("".join(json.dumps(r)+"\n" for r in prior))
    model.train()
    recent, started = deque(maxlen=256), time.monotonic()
    previous = started
    with log_path.open("a", buffering=1) as log:
        for step, groups in enumerate(loader, start_step+1):
            if (ROOT.parent / "pause-request.json").exists():
                train.atomic_json(ROOT / "status.json", {"stage": "paused_by_request", "step": step-1})
                raise SystemExit("pause-request.json present")
            opt.zero_grad()
            loss_sum, batch_tokens = 0., 0
            nrows = sum(int(b["targets"].shape[0]) for b in groups)
            for cpu in groups:
                batch = vt.to_device(cpu)
                train.set_checkpointing(model, len(cpu["ids"]) > m.CFG.checkpoint_above)
                loss = train.soft_ce(vt.packed_logits(model, batch), batch["targets"]).sum()/nrows
                assert torch.isfinite(loss)
                loss.backward()
                loss_sum += float(loss.detach())
                batch_tokens += len(cpu["ids"])
            norm = opt.clip_grad_norm(1.)
            assert math.isfinite(norm)
            for group in opt.param_groups:
                group["lr"] = plan["learning_rate"]*group["lr_scale"]*m.lr_factor(step)
            opt.step()
            tokens += batch_tokens
            now = time.monotonic()
            record = {"step": step, "steps": m.STEPS, "loss": loss_sum, "grad_norm": norm, "rows": nrows,
                      "tokens": batch_tokens, "total_tokens": tokens, "lr": plan["learning_rate"]*m.lr_factor(step),
                      "step_seconds": now-previous}
            previous = now
            if step > start_step+16:
                recent.append(record["step_seconds"])
            log.write(json.dumps(record)+"\n")
            if step % 16 == 0:
                state = {"stage": "training", "arm": "r3", **record}
                if len(recent) >= 32:
                    state["eta"] = fit.eta_state(record, recent)
                train.atomic_json(ROOT / "status.json", state)
                print("TRAIN r3", record, flush=True)
            if step == 64 or step % m.CHECKPOINT_EVERY == 0 or step == m.STEPS:
                assert fit.checkpoint.ROOT == ROOT
                fit.checkpoint.save_checkpoint(model, opt, step, tokens)
    final = audit.read(ROOT / "last-checkpoint.json")
    assert final["step"] == m.STEPS
    train.atomic_json(ROOT / "report.json", {"complete": True, "arm": "r3", "steps": m.STEPS, "final_checkpoint": final,
                      "total_train_tokens": tokens, "seconds_this_process": time.monotonic()-started})
    train.atomic_json(ROOT / "status.json", {"stage": "trained", "arm": "r3"})


def adapter():
    final = audit.read(ROOT / "report.json")["final_checkpoint"]
    path = Path(final["directory"]) / "adapter.safetensors"
    assert audit.sha(path) == final["files"]["adapter.safetensors"]
    return path, final["files"]["adapter.safetensors"]


def evaluate():
    """v3 dev files + synthetic held-out dev, for r3 and r1 (r1 reuses its v3-dev predictions)."""
    from qwen_decision.v3 import champion_epoch1_eval as ev
    import gc
    fit.bind()
    ev.ROOT = ROOT / "eval"
    ev.ROOT.mkdir(parents=True, exist_ok=True)
    sets = {name: ev.dev_rows(path) for name, path in audit.read(m.V3 / "prepared.json")["evaluation"].items()}
    synth_sets = {f"{s}-dev": ev.dev_rows(str(p)) for s, p in DEV.items()}
    results = {}
    for arm in ("r3", "r1"):
        model = load(arm)
        todo = {**(sets if arm == "r3" else {}), **synth_sets}
        results[arm] = {name: {k: v for k, v in ev.evaluate(model, arm, name, rows).items() if k != "sources"}
                        for name, rows in todo.items()}
        train.atomic_json(ROOT / "eval/results.json", results)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def load(arm):
    from safetensors.torch import load_file
    model = fit.legacy.load_model()
    params = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    path = adapter()[0] if arm == "r3" else m.adapter_for(arm)[0]
    stored = {n: t for n, t in load_file(str(path), device="cpu").items() if not n.startswith("master.")}
    assert set(stored) == set(params)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(stored[n])
    train.set_checkpointing(model, False)
    model.requires_grad_(False)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "freeze", "run", "eval"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    fit.bind()
    try:
        {"generate": generate, "freeze": freeze, "run": run, "eval": evaluate}[args.command]()
    except BaseException:
        ROOT.mkdir(parents=True, exist_ok=True)
        train.atomic_json(ROOT / "failure.json", {"stage": args.command, "traceback": traceback.format_exc()})
        raise
