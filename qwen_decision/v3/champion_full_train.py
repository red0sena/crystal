"""Full enriched-data LoRA fit with resumable indexed joint batches and both public evaluations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gc
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor

from qwen_decision.v3 import champion_full_inputs as data, champion_joint_train as legacy
from qwen_decision.v3 import champion_scale_train as checkpoint
from qwen_decision.v3 import champion_audit as audit, train, vision_train as vt, modeling
from qwen_decision.v3 import format as fmt, lora_experiment as lora, lora_placement_experiment as place

ROOT, PLAN = data.ROOT, data.ROOT / "plan.json"
EPOCHS = 2
CFG = SimpleNamespace(**(vars(legacy.CFG) | {"workers": 4, "token_budget": 4096, "checkpoint_above": 4096}))


def bind():
    data.configure()
    legacy.data, legacy.ROOT, legacy.SEED, legacy.CFG = data, ROOT, data.SEED, CFG
    vt.LM_TARGETS = place.ARMS["all"][0]
    checkpoint.ROOT = ROOT


def worker_init(_):
    bind()
    torch.set_num_threads(1)


def implementation():
    modules = [data, legacy, checkpoint, train, vt, modeling, fmt, lora, place]
    result = {str(Path(m.__file__)): audit.sha(m.__file__) for m in modules}
    result[str(Path(__file__))] = audit.sha(__file__)
    result["qwen_decision/v3/champion_input_review.py"] = audit.sha("qwen_decision/v3/champion_input_review.py")
    result["qwen_decision/v3/champion_full_service.py"] = audit.sha("qwen_decision/v3/champion_full_service.py")
    for p in Path("external/decision-index/decision_index").rglob("*"):
        if p.suffix in (".py", ".json"):
            result[str(p)] = audit.sha(p)
    for p in ("qwen_decision/v3/public_eval_engines.py", "qwen_decision/v3/engine.py", "qwen_decision/v3/vision_engine.py"):
        result[p] = audit.sha(p)
    return result


def prepared_inputs():
    p = audit.read(ROOT/"prepared.json")
    review = audit.read(ROOT/"input-review.json")
    assert review["complete"] and review["g7_core_unchanged"]
    assert review["original_file"] == p["text_parts"][0]["path"]
    assert p["reports"]["text-train-0"]["signature"]["sha256"] == review["original_sha256"]
    p["text_parts"][0] = review["text_part"]
    p["reports"]["text-train-0"] = review["report"]
    p["text_rows"] -= review["removed_rows"]
    return p


def stores(prepared=None):
    p = prepared or prepared_inputs()
    texts = data.IndexedRows([r["path"] for r in p["text_parts"]], [r["index"] for r in p["text_parts"]])
    images = data.IndexedRows([p["image_part"]["path"]], [p["image_part"]["index"]])
    return texts, images


class Schedule:
    """All text rows once per epoch, no dropped tail; image cycles keep 16:4 joint batches."""
    def __init__(self, texts, images, epochs=EPOCHS, seed=data.SEED, text_take=16, image_take=4):
        assert len(texts) and len(images)
        self.texts, self.images = texts, images
        self.epochs, self.seed, self.text_take, self.image_take = epochs, seed, text_take, image_take
        self.per_epoch = math.ceil(len(texts) / text_take)
        self.orders = {}

    def __len__(self):
        return self.per_epoch * self.epochs

    def order(self, kind, cycle, n):
        key = kind, cycle
        if key not in self.orders:
            seed = int(__import__("hashlib").sha256(f"{self.seed}:{kind}:{cycle}".encode()).hexdigest()[:16], 16)
            self.orders[key] = np.random.default_rng(seed).permutation(n)
        return self.orders[key]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch, step = divmod(index, self.per_epoch)
        order = self.order("text", epoch, len(self.texts))
        rows = [self.texts[int(i)] for i in order[step*self.text_take:(step+1)*self.text_take]]
        for j in range(self.image_take):
            cycle, at = divmod(index*self.image_take+j, len(self.images))
            image = self.order("image", cycle, len(self.images))[at]
            rows.append(self.images[int(image)])
        random.Random(f"{self.seed}:batch:{index}").shuffle(rows)
        return [(r, f"full-view:{self.seed}:{index}:{j}") for j, r in enumerate(rows)]


def verify(plan=None):
    p = plan or audit.read(PLAN)
    assert p["implementation"] == implementation(), "Training source changed after freeze"
    assert audit.sha(data.BASE / "model.safetensors-00001-of-00001.safetensors") == audit.BASE_SHA
    for f, sha in p["inputs"].items():
        assert audit.sha(f) == sha, f
    for f, sha in audit.read(p["image_hash_manifest"]).items():
        assert audit.sha(f) == sha, f
    return p


def check():
    bind()
    p = prepared_inputs()
    assert p["complete"] and p["source_sha256"] == audit.sha(data.__file__)
    assert audit.sha(data.BASE / "model.safetensors-00001-of-00001.safetensors") == audit.BASE_SHA
    texts, images = stores(p)
    schedule = Schedule(texts, images)
    # The actual maximum of every rendered text part, not a character-length approximation.
    candidates, offset = [], 0
    for i, part in enumerate(p["text_parts"]):
        report = p["reports"]["text-train-" + str(i)]
        candidates += [(n, offset + j) for n, j in report["longest"]]
        offset += report["rows"]
    longest_n, longest_i = max(candidates)
    native_limit = audit.read(data.BASE/"config.json")["text_config"]["max_position_embeddings"]
    assert longest_n+32 <= native_limit, "Training prompt plus augmentation exceeds native context"
    longest_image = images[p["reports"]["image-train"]["longest"][0][1]]
    url = texts[len(texts)-1]
    mind = images[len(images)-1]
    assert url["source"] == "phishing-url-train" and mind["source"] == "mind2web"
    regular = [r for r, _ in schedule[0]]
    text_sample = next(r for r in regular if r["modality"] == "text")
    sample = [text_sample, mind]
    train.seed_all(data.SEED)
    free_before, total_device = torch.cuda.mem_get_info()
    model = legacy.load_model().eval()
    model_memory = {"allocated_gib": torch.cuda.memory_allocated()/2**30,
                    "reserved_gib": torch.cuda.memory_reserved()/2**30}
    encoder = vt.Encoder(max_pixels=CFG.max_pixels)
    b = legacy.encoded_batch(sample, encoder)
    with torch.no_grad():
        initial = vt.packed_logits(model, b)
        saved = []
        for name, module in list(model.named_modules()):
            if isinstance(module, lora.LoRALinear):
                parent, _, leaf = name.rpartition(".")
                owner = model.get_submodule(parent)
                saved.append((owner, leaf, module))
                setattr(owner, leaf, module.base)
        native = vt.packed_logits(model, b)
        for owner, leaf, module in saved:
            setattr(owner, leaf, module)
    error = float((initial-native)[b["code_mask"]].abs().max())
    assert error == 0
    frozen = legacy.frozen_digest(model)
    opt = train.MasterAdamW(model, 2e-4)
    model.train()
    trials = [("joint_update", regular)]
    # Real complete rows near useful length buckets; never cut a row to fit a bucket.
    for low, high in ((2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768)):
        found, offset = [], 0
        for part in p["text_parts"]:
            index = np.load(part["index"], mmap_mode="r")
            lengths = index[:, 2]
            eligible = np.flatnonzero((lengths > low) & (lengths <= high))
            if len(eligible):
                i = int(eligible[np.argmax(lengths[eligible])])
                found.append((int(lengths[i]), offset+i))
            offset += len(index)
        if found:
            _, i = max(found)
            trials.append((f"text_at_most_{high}", [texts[i]]))
    trials += [("maximum_text", [texts[longest_i]]), ("vision_url_mind2web", [longest_image, url, mind])]
    updates = []
    for step, (name, rows) in enumerate(trials):
        dataset = vt.MixSteps([[(r, f"full-smoke:{step}:{j}") for j, r in enumerate(rows)]], CFG, [])
        opt.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss_sum, tokens = 0., 0
        for cpu in dataset[0]:
            batch = vt.to_device(cpu)
            train.set_checkpointing(model, len(cpu["ids"]) > CFG.checkpoint_above)
            loss = train.soft_ce(vt.packed_logits(model, batch), batch["targets"]).sum()/len(rows)
            assert torch.isfinite(loss)
            loss.backward()
            loss_sum += float(loss.detach())
            tokens += len(cpu["ids"])
        norm = opt.clip_grad_norm(1.)
        assert math.isfinite(norm)
        opt.step()
        torch.cuda.synchronize()
        measurement = {"trial": name, "rows": len(rows), "tokens": tokens, "loss": loss_sum,
                       "grad_norm": norm, "seconds": time.monotonic()-start,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                       "device_free_after_gib": torch.cuda.mem_get_info()[0]/2**30}
        updates.append(measurement)
        train.atomic_json(ROOT/"memory-profile-progress.json", {"trials": updates})
        print("MEMORY_TRIAL", measurement, flush=True)
    assert legacy.frozen_digest(model) == frozen
    assert all(bool((t != 0).any()) for n, t in model.named_parameters() if n.endswith("lora_b"))
    model.eval()
    train.set_checkpointing(model, False)
    native_checks = []
    with torch.no_grad():
        for row in sample:
            encoded = encoder.row(row, None)
            batch = legacy.encoded_batch([row], encoder)
            packed = vt.packed_logits(model, batch)[0, batch["code_mask"][0]]
            if row["modality"] == "text":
                ids = torch.tensor([encoded["input_ids"]], device="cuda")
                codes = torch.tensor([encoded["code_ids"]], device="cuda")
                raw = modeling.code_logits(model, ids, torch.ones_like(ids), codes, torch.ones_like(codes, dtype=torch.bool))[0]
            else:
                question, _, target = vt.item_view(row, None)
                pictures = encoder.pictures(row, root=data.MIX)
                prompt = fmt.render(encoder.tokenizer, vt.IMAGE*len(pictures), question, encoder.codes[:len(target)])
                args = encoder.processor(text=[prompt], images=pictures, return_tensors="pt").to("cuda")
                hidden = model.model(**args, use_cache=False).last_hidden_state[0, -1]
                weights = model.get_output_embeddings().weight[torch.tensor(encoded["code_ids"], device="cuda")]
                raw = weights.float() @ hidden.float()
            diff = float((packed-raw).abs().max())
            assert diff < .5 and int(packed.argmax()) == int(raw.argmax())
            native_checks.append({"modality": row["modality"], "max_logit_error": diff, "same_prediction": True})
    report = {"passed": True, "zero_lora_error": error, "frozen_base_unchanged": True,
              "max_text_tokens": longest_n, "updates": updates, "native_comparison": native_checks,
              "peak_cuda_gb": max(r["peak_allocated_gib"] for r in updates), "smoke_parameters_discarded": True,
              "gpu": torch.cuda.get_device_name(), "total_device_gib": total_device/2**30,
              "device_free_before_model_gib": free_before/2**30, "model_loaded_memory": model_memory,
              "implementation": implementation()}
    train.atomic_json(ROOT / "check.json", report)
    print("GPU_PREFLIGHT_PASSED", report, flush=True)


def freeze():
    if PLAN.exists():
        return verify()
    p = prepared_inputs()
    checked = audit.read(ROOT / "check.json")
    assert checked["passed"] and checked["implementation"] == implementation()
    texts, images = stores(p)
    schedule = Schedule(texts, images)
    inputs = {str(ROOT/"prepared.json"): audit.sha(ROOT/"prepared.json"),
              str(ROOT/"input-review.json"): audit.sha(ROOT/"input-review.json"),
              str(data.DATA/"input-manifest.json"): audit.sha(data.DATA/"input-manifest.json"),
              str(data.DATA/"new-source-registry.json"): audit.sha(data.DATA/"new-source-registry.json")}
    for part in [*p["text_parts"], p["image_part"]]:
        for file in part.values():
            inputs[file] = audit.sha(file)
    for file in p["evaluation"].values():
        inputs[file] = audit.sha(file)
    image_manifest = data.DATA / "enriched-vision-image-hashes.json"
    inputs[str(image_manifest)] = audit.sha(image_manifest)
    for f in data.BASE.glob("*"):
        if f.suffix in (".json", ".jinja", ".txt"):
            inputs[str(f)] = audit.sha(f)
    public = Path("runs/champion-v1/public-eval-v1")
    for file in (public/"text-counted-rows.jsonl", public/"vision-suite/vision-rows.jsonl"):
        inputs[str(file)] = audit.sha(file)
    for file, sha in audit.read(public/"joint-vision/plan.json")["files"].items():
        if file.startswith(str(public/"vision-suite")):
            inputs[file] = sha
    plan = {"experiment": "full-g7-enriched-lora-v3", "base": str(data.BASE), "base_sha256": audit.BASE_SHA,
            "seed": data.SEED, "epochs": EPOCHS, "text_rows": len(texts), "image_rows": len(images),
            "steps_per_epoch": schedule.per_epoch, "steps": len(schedule), "text_per_update": 16, "image_per_update": 4,
            "text_coverage": "Each retained row exactly once per epoch, including final partial batch; no quota or truncation",
            "image_coverage": "Continuous independently shuffled cycles; repeats preserve 20% image contribution",
            "image_presentations": len(schedule)*4, "image_passes": len(schedule)*4/len(images),
            "learning_rate": 2e-4, "warmup_steps": math.ceil(len(schedule)*.02),
            "lr_schedule": "linear warmup, cosine decay to10%", "optimizer": "FP32 master AdamW(.9,.95), wd.01, clip1",
            "trainable": "all186 LM LoRA r16 alpha32 dropout.05; frozen original base, tied embeddings and ViT",
            "config": vars(CFG), "inputs": inputs, "image_hash_manifest": str(image_manifest),
            "implementation": implementation(), "checkpointing": "First at64, every4096, every text epoch boundary, final; adapters+FP32 masters+optimizer+RNG",
            "evaluation": list(p["evaluation"]), "public_evaluation": "All counted37 text and11 vision benchmarks after export",
            "selection": "fixed final update; no public checkpoint choice", "alignment": "not part of CE fit; separate later experiment",
            "resources": "current GPU; persistent user service, resumable on failure; no upload or submission"}
    train.atomic_json(PLAN, plan)
    snapshot = ROOT / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for i, file in enumerate(plan["implementation"]):
        shutil.copyfile(file, snapshot / f"{i:03}-{Path(file).name}")
    verify(plan)
    return plan


def eta_state(record, recent):
    from datetime import datetime, timezone, timedelta
    from zoneinfo import ZoneInfo
    mean = statistics.mean(recent)
    remaining = (record["steps"]-record["step"])*mean
    now = datetime.now(timezone.utc)
    return {"measured_steps": len(recent), "seconds_per_update": mean, "remaining_training_seconds": remaining,
            "estimated_training_end_kst": (now+timedelta(seconds=remaining)).astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
            "updated_utc": now.isoformat(), "excludes": "internal evaluation, export verification and full public evaluations",
            "scope": "Rolling measured throughput; early estimate and GPU availability can change."}


def run():
    bind()
    plan = verify()
    if (ROOT/"report.json").exists():
        assert audit.read(ROOT/"report.json")["export_replay_exact"]
        return
    train.seed_all(data.SEED)
    model = legacy.load_model()
    opt = train.MasterAdamW(model, plan["learning_rate"])
    start_step, tokens = checkpoint.restore_checkpoint(model, opt)
    texts, images = stores()
    schedule = Schedule(texts, images)
    assert len(schedule) == plan["steps"]
    dataset = vt.MixSteps(schedule, CFG, [])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(start_step, len(dataset))), batch_size=None,
        num_workers=CFG.workers, pin_memory=True, prefetch_factor=2, worker_init_fn=worker_init,
        multiprocessing_context="spawn", generator=torch.Generator().manual_seed(data.SEED))
    log_path = ROOT/"train.jsonl"
    if log_path.exists():
        prior = [r for r in audit.rows(log_path) if r["step"] <= start_step]
        log_path.write_text("".join(json.dumps(r)+"\n" for r in prior))
    model.train()
    recent = deque(maxlen=256)
    started = time.monotonic()
    previous = started
    train.atomic_json(ROOT/"status.json", {"stage": "loading_training_batches", "training_started": False, "resume_step": start_step})
    with log_path.open("a", buffering=1) as log:
        for step, groups in enumerate(loader, start_step+1):
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
            warm, n = plan["warmup_steps"], plan["steps"]
            factor = min(step/warm, 1.)*(.1+.45*(1+math.cos(math.pi*max(0, step-warm)/max(1, n-warm))))
            for group in opt.param_groups:
                group["lr"] = plan["learning_rate"]*group["lr_scale"]*factor
            opt.step()
            tokens += batch_tokens
            now = time.monotonic()
            record = {"step": step, "steps": n, "epoch": step/schedule.per_epoch, "loss": loss_sum, "grad_norm": norm,
                      "rows": nrows, "tokens": batch_tokens, "total_tokens": tokens,
                      "step_seconds": now-previous, "seconds_this_process": now-started}
            previous = now
            if step > start_step+16:
                recent.append(record["step_seconds"])
            log.write(json.dumps(record)+"\n")
            if step % 16 == 0:
                state = {"stage": "training", "training_started": True, **record}
                if len(recent) >= 32:
                    state["eta"] = eta_state(record, recent)
                train.atomic_json(ROOT/"status.json", state)
                print("TRAIN", record, flush=True)
            if step == 64 or step % 4096 == 0 or step % schedule.per_epoch == 0 or step == n:
                checkpoint.save_checkpoint(model, opt, step, tokens)
    train.atomic_json(ROOT/"status.json", {"stage": "exporting", "training_started": True, "step": plan["steps"]})
    legacy.merge_and_save(model, opt)
    cfg = audit.read(ROOT/"model/v3_config.json")
    cfg.update(experiment=plan["experiment"], steps=plan["steps"], seed=data.SEED)
    train.atomic_json(ROOT/"model/v3_config.json", cfg)
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    sample = [texts[0], images[len(images)-1]]
    encoder = vt.Encoder(max_pixels=CFG.max_pixels)
    batch = legacy.encoded_batch(sample, encoder)
    model.eval()
    train.set_checkpointing(model, False)
    with torch.inference_mode():
        expected = vt.packed_logits(model, batch).cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    saved = modeling.load(ROOT/"model", attention="noncausal", deltanet="causal", dtype=torch.bfloat16, kernel="varlen")
    saved.model.visual.to("cuda")
    saved.eval()
    with torch.inference_mode():
        actual = vt.packed_logits(saved, batch).cpu()
    assert torch.equal(expected, actual)
    report = {"complete": True, "export_replay_exact": True, "steps": plan["steps"],
              "total_train_tokens": tokens, "seed": data.SEED, "official_full_rank": None,
              "training_seconds_this_process": time.monotonic()-started}
    train.atomic_json(ROOT/"report.json", report)
    verify()
    train.atomic_json(ROOT/"status.json", {"stage": "trained", "training_started": True, "steps": plan["steps"], "verified": True})


def evaluate():
    bind()
    verify()
    modeling.compile_elementwise()
    vt.use_vision_attention("varlen")
    model = modeling.load(ROOT/"model", attention="noncausal", deltanet="causal", dtype=torch.bfloat16, kernel="varlen")
    model.model.visual.to("cuda")
    model.eval()
    result = {}
    for name, path in audit.read(ROOT/"prepared.json")["evaluation"].items():
        rows = vt.gold_text(list(audit.rows(path)))
        # vt.predict's spawned workers use their module defaults. Absolute image
        # references preserve this run's immutable mix root in those workers.
        rows = [{**r, "images": [str((data.MIX/p).resolve()) for p in r["images"]]}
                if r["modality"] == "image" else r for r in rows]
        pred = vt.predict(model, [("mix", r) for r in rows], workers=CFG.workers, token_budget=CFG.token_budget)
        by = defaultdict(list)
        with (ROOT/f"exported-{name}.jsonl").open("w") as out:
            for row, p in zip(rows, pred):
                by[row["source"]].append(p["correct"])
                out.write(json.dumps({"id": row["id"], "source": row["source"], **p})+"\n")
        scores = {s: {"rows": len(v), "accuracy": statistics.mean(v)} for s, v in by.items()}
        result[name] = {"rows": len(rows), "macro_accuracy": statistics.mean(v["accuracy"] for v in scores.values()), "sources": scores}
        train.atomic_json(ROOT/"internal-evaluation.json", result)


def public_eval(modality):
    verify()
    assert audit.read(ROOT/"report.json")["export_replay_exact"]
    public = Path("runs/champion-v1/public-eval-v1")
    dest = ROOT/("public-"+modality)
    if (dest/"completion.json").exists():
        assert audit.read(dest/"completion.json")["complete"]
        return
    hashes = {str(p): audit.sha(p) for p in (ROOT/"model").glob("*") if p.is_file()}
    if modality == "text":
        from decision_index.runner import run as runner
        from decision_index.pipeline import score_run
        from decision_index.suite.io import Suite
        runner("qwen_decision.v3.public_eval_engines:JointText", {"model": str(ROOT/"model")},
               public/"text-counted-rows.jsonl", dest, compact=True,
               corpus_sha256=audit.sha(public/"text-counted-rows.jsonl"))
        score = score_run(Suite(public/"text-suite", "0.3"), dest/"results.jsonl", "full-g7-enriched-lora-v3", dest)
    else:
        from decision_index.vision.run import run as runner
        from decision_index.vision.score import score_run
        runner("qwen_decision.v3.vision_engine:VisionEngine", {"model": str(ROOT/"model"), "prompt": "decide"},
               public/"vision-suite", dest, workers=1, chunk=16, compact=True)
        score = score_run(public/"vision-suite/vision-rows.jsonl", dest/"results.jsonl", dest)
    for f, sha in hashes.items():
        assert audit.sha(f) == sha
    rows = list(audit.rows(dest/"results.jsonl"))
    counts = Counter(r["status"] for r in rows)
    expected = 110087 if modality == "text" else 12210
    assert len({r["run_id"] for r in rows}) == expected and counts["ok"] == expected
    train.atomic_json(dest/"completion.json", {"complete": True, "counts": dict(counts), "visited_requests": expected,
                      "public_index": score.get("decision_index", score.get("public_vision_index")), "model_hashes": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "freeze", "run", "eval", "eval-text", "eval-vision"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    bind()
    try:
        if args.command == "check":
            check()
        elif args.command == "freeze":
            freeze()
        elif args.command == "run":
            run()
        elif args.command == "eval":
            evaluate()
        else:
            public_eval(args.command.removeprefix("eval-"))
    except Exception:
        train.atomic_json(ROOT/"failure.json", {"stage": args.command, "traceback": traceback.format_exc()})
        raise
