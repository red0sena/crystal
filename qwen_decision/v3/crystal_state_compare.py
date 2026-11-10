"""Matched-data causal/Crystal full-FT screen and question-free state transfer.

All worlds, labels, semantic holdouts and downstream settings are inherited
verbatim from state_rotation v2. Only text/seen records supervise the LM.
Crystal here means noncausal full-attention + full language FT + code CE,
not the previously published checkpoint or its g6/g7 weight soup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from qwen_decision.v3 import format as fmt, modeling, state_rotation as sr, train

BASE = Path("models/qwen3.5-0.8b")
PARENT = Path("runs/rotation-state/plain-qwen-v2")
OUT = Path("runs/rotation-state/crystal-method-v1")
ARMS = ("causal", "crystal")
SEEDS = sr.HEAD_SEEDS
EPOCHS, BATCH, LR = 1, 16, 1e-5
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


def read(path):
    return json.loads(Path(path).read_text())


def dataset(parent=PARENT):
    rows, specs = read(parent / "scenes.json"), read(parent / "query-specs.json")
    assert rows == sr.make_scenes() and specs == sr.query_specs()
    all_items = sr.examples(rows, specs)
    splits = {s: [e for e in all_items if rows[e["scene"]]["split"] == s and (s == "test" or not e["unseen"])]
              for s in ("train", "validation", "test")}
    assert [len(splits[s]) for s in splits] == [8922, 2251, 3072]
    assert not any(e["unseen"] for s in ("train", "validation") for e in splits[s])
    assert splits["test"] == read(parent / "test-items.json")
    return rows, specs, splits


def training_example(item, rows, specs):
    spec = specs[item["query"]]
    answers = [sr.ANSWER_NAMES[a] for a in spec["answers"]]
    return {"id": f"{rows[item['scene']]['id']}:{spec['key']}", "kind": "choice", "source": spec["task"],
            "ordered": False, "state": rows[item["scene"]]["caption"],
            "question": {"type": "choice", "instructions": spec["text"], "criteria": dict.fromkeys(answers)},
            "target": [float(a == item["gold"]) for a in spec["answers"]]}


def renderer(tokenizer):
    # No external/distractor examples or new gold; just the original candidate set.
    args = SimpleNamespace(layout="d3", pointer_positions=False, pad_probability=0., noul_choice_probability=0.,
                           code_shuffle_probability=.2, code_sample_probability=.2)
    return train.Renderer(tokenizer, args, [])


def schedule(size, seed):
    rng = random.Random(seed)
    for epoch in range(1, EPOCHS + 1):
        order = list(range(size)); rng.shuffle(order)
        for start in range(0, size, BATCH):
            yield epoch, order[start:start + BATCH]


def encode_step(render, examples, ids, seed, step):
    rng = random.Random((seed << 32) + step)
    return [render(examples[i], rng) for i in ids]


def digest_rows(rows):
    return hashlib.sha256(json.dumps([{k: r[k] for k in ("id", "input_ids", "code_ids", "target")}
                                     for r in rows], separators=(",", ":")).encode()).hexdigest()


def load_training(arm):
    # Reuse the original validated full-FT path (packed rows, FLA, FP32 masters).
    from qwen_decision.v3 import from_base_experiment as fb
    return fb.load_model("noncausal" if arm == "crystal" else "causal", path=BASE)


def packed_scores(model, encoded):
    batch = modeling.pack_rows([r["input_ids"] for r in encoded], [r["code_ids"] for r in encoded], "cuda")
    return modeling.packed_code_logits(model, batch)


def macro_score(logits, encoded):
    gold = torch.tensor([r["target"].index(1.) for r in encoded], device=logits.device)
    losses = F.cross_entropy(logits, gold, reduction="none").detach().cpu().numpy()
    correct = (logits.argmax(-1) == gold).detach().cpu().numpy()
    return losses, correct


@torch.no_grad()
def validate(model, encoded):
    model.eval(); train.set_checkpointing(model, False)
    losses, correct = [], []
    for start in range(0, len(encoded), BATCH):
        group = encoded[start:start + BATCH]
        a, b = macro_score(packed_scores(model, group), group)
        losses.extend(a); correct.extend(b)
    return {"macro_nll": float(np.mean([np.mean([v for v, r in zip(losses, encoded) if r["source"] == task])
                                        for task in sr.TASKS])),
            "macro_accuracy": float(np.mean([np.mean([v for v, r in zip(correct, encoded) if r["source"] == task])
                                             for task in sr.TASKS]))}


def step_update(model, optimizer, encoded, weights, step, steps):
    model.train(); optimizer.zero_grad()
    train.set_checkpointing(model, False)
    # Microbatch eight, effective batch 16, exactly the same groups in both arms.
    loss_sum = 0.
    for start in range(0, len(encoded), 8):
        group = encoded[start:start + 8]
        logits = packed_scores(model, group)
        targets = torch.tensor([r["target"].index(1.) for r in group], device="cuda")
        w = torch.tensor([weights[r["source"]] for r in group], device="cuda")
        loss = (F.cross_entropy(logits, targets, reduction="none") * w).sum() / len(encoded)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward(); loss_sum += loss.item()
    norm = optimizer.clip_grad_norm(1.)
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradients")
    warmup = math.ceil(.06 * steps)
    scale = min(step / warmup, 1.) * (.1 + .45 * (1 + math.cos(math.pi * max(0, step - warmup) / (steps - warmup))))
    for group in optimizer.param_groups:
        group["lr"] = LR * scale
    optimizer.step()
    return loss_sum, norm


def freeze(out):
    rows, specs, splits = dataset()
    out.mkdir(parents=True, exist_ok=False)
    files = [PARENT / name for name in ("scenes.json", "query-specs.json", "test-items.json", "features.pt", "report.json")]
    sources = [Path(__file__), Path(sr.__file__), Path(modeling.__file__), Path(train.__file__),
               Path("qwen_decision/v3/from_base_experiment.py"), Path(fmt.__file__)]
    plan = {"experiment": "matched-data-crystal-state-transfer-v1", "parent": str(PARENT),
            "inherited_files_sha256": {str(p): train.sha256_file(p) for p in files},
            "source_sha256": {str(p): train.sha256_file(p) for p in sources},
            "base": str(BASE), "base_revision": sr.REVISION,
            "base_weights_sha256": {str(p): train.sha256_file(p) for p in BASE.glob("*.safetensors")},
            "arms": {"plain": "untouched frozen Qwen, existing state_rotation v2 results",
                     "causal": "full language FT, stock causal attention, shown-code gold CE",
                     "crystal": "full language FT, noncausal 6 softmax layers, causal 18 DeltaNet layers, shown-code gold CE"},
            "training_data": "the exact 8922 text/seen rows; no image gold or unseen query supervision",
            "splits": {s: len(v) for s, v in splits.items()}, "world_counts": sr.COUNTS,
            "heldout_queries": sorted(sr.HOLDOUT), "lm_seeds": SEEDS, "head_seeds": SEEDS,
            "lm_fit": {"epochs": EPOCHS, "batch": BATCH, "micro_batch": 8, "lr": LR,
                       "optimizer": "existing FP32-master AdamW, beta .9/.95, wd .01 for matrices, clip1",
                       "schedule": "6% linear warmup then cosine to 10%",
                       "loss": "task-balanced hard gold CE over displayed codes",
                       "augmentation": "shuffle same options; .2 code-shuffle/.2 code-sample; no candidate padding",
                       "selection": "one complete pass, final epoch; text/seen validation only monitors the fit"},
            "state_evaluation": "after freezing each fitted encoder: COMMON CAUSAL inference, identical question-free content pooling, train-only PCA64, Procrustes, text-only MLP160epochs and 20 wrong-pair controls as parent; only LM training method differs",
            "native_evaluation": "same test scenes/questions/candidates, native tied code readout, no generation; separate from state-transfer score",
            "primary_contrasts": ["crystal minus causal state image-rotation seen macro accuracy", "crystal minus plain state image-rotation seen macro accuracy"],
            "uncertainty": "paired crossed resampling of 3 LM seeds, 3 shared MLP seeds, and 256 whole scenes; plain encoder is not independently replicated; fixed split; 5000 draws",
            "interpretation": "same previously viewed test set: exploratory follow-up, not a fresh confirmatory test; state transfer fixes causal inference, native scores use each recipe's attention; matched one-pass screen, not converged g6/g7 reproduction; no original g6/g7 data, pretrained Crystal checkpoint, or weight soup"}
    train.atomic_json(out / "plan.json", plan)
    sr.dump(out / "scenes.json", rows); sr.dump(out / "query-specs.json", specs)
    print("PLAN FROZEN", out, flush=True)


def check_plan(out):
    plan = read(out / "plan.json")
    for key in ("inherited_files_sha256", "source_sha256", "base_weights_sha256"):
        for path, expected in plan[key].items():
            assert train.sha256_file(Path(path)) == expected, path


def fit(arm, seed, out, smoke=False):
    check_plan(out)
    rows, specs, splits = dataset()
    examples = [training_example(e, rows, specs) for e in splits["train"]]
    counts = Counter(r["source"] for r in examples)
    weights = {t: len(examples) / (3 * counts[t]) for t in sr.TASKS}
    directory = out / f"{arm}-{seed}"
    if not smoke and (directory / "fit.json").exists():
        print("FIT EXISTS", directory, flush=True); return
    directory.mkdir(parents=True, exist_ok=True)
    train.seed_all(seed)
    model = load_training(arm)
    render = renderer(AutoProcessor.from_pretrained(BASE, local_files_only=True).tokenizer)
    validation = [render(training_example(e, rows, specs)) for e in splits["validation"]]
    optimizer = train.MasterAdamW(model, LR)
    steps = EPOCHS * math.ceil(len(examples) / BATCH)
    best, history, digests, times = float("inf"), [], [], []
    started = time.monotonic()
    best_epoch = 0
    for step, (epoch, ids) in enumerate(schedule(len(examples), seed), 1):
        encoded = encode_step(render, examples, ids, seed, step)
        digests.append(digest_rows(encoded))
        tick = time.monotonic()
        loss, norm = step_update(model, optimizer, encoded, weights, step, steps)
        torch.cuda.synchronize(); times.append(time.monotonic() - tick)
        if step <= 3 or step % 25 == 0:
            progress = {"arm": arm, "seed": seed, "epoch": epoch, "step": step, "steps": steps,
                        "loss": loss, "gradient_norm": norm, "elapsed_seconds": time.monotonic() - started,
                        "recent_seconds_per_step": float(np.median(times[-20:])),
                        "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9}
            print("FIT", json.dumps(progress), flush=True)
            train.atomic_json(directory / "progress.json", progress)
        if smoke and step == 6:
            train.atomic_json(out / f"smoke-{arm}.json", {"steps": len(times), "seconds_per_step": times,
                              "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9, "digests": digests,
                              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
            break
        if step % math.ceil(len(examples) / BATCH) == 0:
            score = validate(model, validation)
            history.append({"epoch": epoch, **score})
            print("VALIDATION", arm, seed, epoch, score, flush=True)
            if score["macro_nll"] < best:
                best, best_epoch = score["macro_nll"], epoch
                save_file(train.language_state(model), str(directory / "best.safetensors"))
            train.atomic_json(directory / "history.json", history)
    if not smoke:
        train.atomic_json(directory / "step-input-digests.json", digests)
        train.atomic_json(directory / "fit.json", {"arm": arm, "seed": seed, "epochs": EPOCHS,
                          "steps": len(times), "records_per_epoch": len(examples), "best_epoch": best_epoch,
                          "best_validation_macro_nll": best, "history": history,
                          "elapsed_seconds": time.monotonic() - started,
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "checkpoint_sha256": train.sha256_file(directory / "best.safetensors")})
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()


def inference_model(arm, checkpoint=None):
    model = modeling.load(BASE, attention="noncausal" if arm == "crystal" else "causal", kernel="sdpa")
    if checkpoint is not None:
        model.model.language_model.load_state_dict(load_file(str(checkpoint)))
    model.model.visual.to("cuda")
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def extract_in_memory(model, processor, rows, specs, directory):
    processor.tokenizer.padding_side = "left"
    collected, counts = {"image": [], "text": []}, {"image": [], "text": []}
    started = time.monotonic()
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        for modality in collected:
            contents = [IMAGE if modality == "image" else r["caption"] for r in group]
            prompts = [sr.prompt(processor.tokenizer, c) for c in contents]
            inputs = processor(text=prompts, images=[sr.draw_scene(r) for r in group] if modality == "image" else None,
                               padding=True, return_tensors="pt").to("cuda")
            mask = inputs["input_ids"] == model.config.image_token_id if modality == "image" else sr.content_mask(
                processor.tokenizer, prompts, contents, inputs["input_ids"]).to("cuda")
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            collected[modality].append(sr.masked_mean(hidden, mask).cpu())
            counts[modality].extend(mask.sum(-1).cpu().tolist())
        if start % 256 == 0:
            print("EXTRACT", directory.name, start, len(rows), flush=True)
    query = []
    for start in range(0, len(specs), 4):
        contents = [s["text"] for s in specs[start:start + 4]]
        prompts = [sr.prompt(processor.tokenizer, c, "Question") for c in contents]
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
        mask = sr.content_mask(processor.tokenizer, prompts, contents, inputs["input_ids"]).to("cuda")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        query.append(sr.masked_mean(hidden, mask).cpu())
    result = {k: torch.cat(v) for k, v in collected.items()}
    result.update({"query": torch.cat(query), "pooling_token_counts": counts,
                   "extraction_seconds": time.monotonic() - started, "qwen_trainable_parameters": 0})
    torch.save(result, directory / "features.pt")
    return result


@torch.inference_mode()
def native_records(model, processor, rows, specs, testing, modality):
    codes, code_ids = fmt.answer_codes(processor.tokenizer)
    array = np.zeros((256, 2, 3, 3), dtype=np.float64)
    per_item = []
    for start in range(0, len(testing), 4):
        group = testing[start:start + 4]
        prompts, widths = [], []
        for item in group:
            spec, row = specs[item["query"]], rows[item["scene"]]
            question = training_example(item, rows, specs)["question"]
            width = len(spec["answers"]); widths.append(width)
            prompts.append(fmt.render(processor.tokenizer, IMAGE if modality == "image" else row["caption"],
                                      question, codes[:width], "d3"))
        inputs = processor(text=prompts, images=[sr.draw_scene(rows[e["scene"]]) for e in group] if modality == "image" else None,
                           padding=True, return_tensors="pt").to("cuda")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        h = hidden[torch.arange(len(group), device="cuda"), modeling.last_positions(inputs["attention_mask"])]
        logits = F.linear(h.float(), model.get_output_embeddings().weight[code_ids[:max(widths)]].float())
        for item, z, width in zip(group, logits, widths):
            spec = specs[item["query"]]
            gold = spec["answers"].index(item["gold"])
            z = z[:width]
            correct = int(z.argmax().item() == gold)
            nll = -z.log_softmax(-1)[gold].item()
            scene = int(rows[item["scene"]]["id"].split("-")[1])
            array[scene, int(item["unseen"]), item["task"]] += [correct, nll, 1]
            per_item.append({**item, "logits": z.cpu().tolist(), "prediction": int(z.argmax()), "local_gold": gold})
    return array, per_item


def evaluate(arm, seed, out):
    check_plan(out)
    directory = out / f"{arm}-{seed}"
    if (directory / "evaluation-complete.json").exists():
        print("EVAL EXISTS", directory, flush=True); return
    rows, specs, splits = dataset()
    # Hold the entire question-free evaluation architecture fixed across arms.
    model = inference_model("causal", directory / "best.safetensors")
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    features = extract_in_memory(model, processor, rows, specs, directory)
    if arm == "crystal":
        modeling.enable_noncausal(model.model.language_model, "sdpa")
    native = {}
    for modality in ("text", "image"):
        arr, records = native_records(model, processor, rows, specs, splits["test"], modality)
        np.save(directory / f"native-{modality}.npy", arr)
        sr.dump(directory / f"native-{modality}-items.json", records)
        native[modality] = sr.summarize(arr[None])
        print("NATIVE", arm, seed, modality, native[modality], flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    # Exact existing downstream pipeline, including all 3 paired head seeds.
    state_report = sr.analyze(rows, specs, features, directory)
    sr.dump(directory / "evaluation-complete.json", {"native": native, "state_report": "report.json"})


def evaluate_plain_native(out):
    directory = out / "plain"
    directory.mkdir(exist_ok=True)
    if (directory / "native.json").exists():
        return
    rows, specs, splits = dataset()
    model = inference_model("causal")
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    summary = {}
    for modality in ("text", "image"):
        array, records = native_records(model, processor, rows, specs, splits["test"], modality)
        np.save(directory / f"native-{modality}.npy", array)
        sr.dump(directory / f"native-{modality}-items.json", records)
        summary[modality] = sr.summarize(array[None])
    sr.dump(directory / "native.json", summary)
    del model
    gc.collect(); torch.cuda.empty_cache()


def all_results(out):
    arrays = {}
    names = ("image_mean_only", "image_rotation", "text_reference")
    for arm in ("plain", *ARMS):
        dirs = [PARENT] * len(SEEDS) if arm == "plain" else [out / f"{arm}-{seed}" for seed in SEEDS]
        arrays[arm] = {name: np.stack([np.load(d / "test-scene-records.npz")[name] for d in dirs]) for name in names}
    summary = {arm: {name: sr.summarize(value.reshape(-1, *value.shape[2:])) for name, value in variants.items()}
               for arm, variants in arrays.items()}
    comparisons = {}
    for ref in ("plain", "causal"):
        for name in names:
            rng = np.random.default_rng(sr.SEED + 500)
            samples = {g: {m: [] for m in ("accuracy", "nll")} for g in ("seen", "unseen")}
            for _ in range(5000):
                lm = rng.integers(3, size=3)
                hs = rng.integers(3, size=3)
                scenes = rng.integers(256, size=256)
                selected = {}
                for arm in ("crystal", ref):
                    value = arrays[arm][name][lm[:, None], hs[None, :]][:, :, scenes]
                    selected[arm] = sr.summarize(value.reshape(-1, *value.shape[2:]))
                for g in samples:
                    for m in samples[g]:
                        samples[g][m].append(selected["crystal"][g]["macro_" + m] - selected[ref][g]["macro_" + m])
            comparisons[f"crystal-minus-{ref}:{name}"] = {
                g: {m: {"delta": summary["crystal"][name][g]["macro_" + m] - summary[ref][name][g]["macro_" + m],
                         "ci95": np.quantile(v, [.025, .975]).tolist()} for m, v in metrics.items()}
                for g, metrics in samples.items()}
    input_checks = {}
    for seed in SEEDS:
        a, b = [read(out / f"{arm}-{seed}/step-input-digests.json") for arm in ARMS]
        assert a == b
        input_checks[str(seed)] = {"all_step_inputs_identical": True, "steps": len(a)}
    native, fits = {"plain": read(out / "plain/native.json")}, {}
    for arm in ARMS:
        native[arm] = {modality: sr.summarize(np.stack([np.load(out / f"{arm}-{seed}/native-{modality}.npy")
                                                     for seed in SEEDS])) for modality in ("text", "image")}
        fits[arm] = [read(out / f"{arm}-{seed}/fit.json") for seed in SEEDS]
    retrieval = {arm: {name: float(np.mean([read(out / f"{arm}-{seed}/report.json")["alignment_test"][name]["retrieval_R1"]
                                          for seed in SEEDS])) for name in ("image_mean_only", "image_rotation")}
                 for arm in ARMS}
    retrieval["plain"] = {name: read(PARENT / "report.json")["alignment_test"][name]["retrieval_R1"]
                           for name in ("image_mean_only", "image_rotation")}
    report = {"experiment": "matched-data-crystal-state-transfer-v1", "complete": True,
              "state_transfer": summary, "retrieval_R1": retrieval, "native_code_head": native,
              "paired_hierarchical_bootstrap_5000": comparisons, "input_checks": input_checks, "fits": fits,
              "warning": "Repeated use of previously viewed test; exploratory one-pass screen. State extraction uses common causal inference; native Crystal uses noncausal inference. Plain repeats the same frozen encoder, not 3 independent encoders. Crossed LM-seed/head-seed/scene bootstrap is conditional on fixed split and 3 seeds."}
    sr.dump(out / "report.json", report)
    print("COMPLETE", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "smoke", "fit", "evaluate", "run", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--arm", choices=ARMS, default="crystal")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "freeze": freeze(args.out)
    elif args.command == "smoke": fit(args.arm, args.seed, args.out, smoke=True)
    elif args.command == "fit": fit(args.arm, args.seed, args.out)
    elif args.command == "evaluate": evaluate(args.arm, args.seed, args.out)
    elif args.command == "report": all_results(args.out)
    elif args.command == "run":
        for seed in SEEDS:
            for arm in ARMS:
                fit(arm, seed, args.out)
        evaluate_plain_native(args.out)
        for seed in SEEDS:
            for arm in ARMS:
                evaluate(arm, seed, args.out)
        all_results(args.out)


if __name__ == "__main__":
    main()
