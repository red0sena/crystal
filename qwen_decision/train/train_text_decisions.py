"""Local full-text-backbone decision training inspired by public d1 practices.

The d1 loss/data recipe is not public. This implements our own weighted
choice-restricted CE, shuffled content and code positions, mixed tasks, long
reading contexts, and independent runs followed by checkpoint averaging.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v1"
DEFAULT_OUT = ROOT / "training/text-decision-full-v1"
SYSTEM = "Use the provided state to answer the question. Select exactly one supplied option. Reply with only its option code."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)

def read_records(name):
    with (DATA / f"{name}.jsonl").open() as f:
        return [json.loads(line) for line in f]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 재현 설정: FP32 파라미터/Adam 상태를 유지하고 forward 행렬 연산만 BF16으로 수행한다.
# 언어 backbone과 tied embedding/LM Head 전체를 학습한다. LoRA나 새 분류 head는 없다.
# vision은 고정하며 CPU에 둔다. generate()는 실수로 호출하면 즉시 실패하게 한다.
def model_load():
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        ROOT / "model", dtype=torch.float32, attn_implementation="sdpa",
    )
    # Keep FP32 master parameters and Adam moments. Only forward matmuls use BF16.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.language_model.parameters():
        p.requires_grad_(True)
    for p in model.get_output_embeddings().parameters():
        p.requires_grad_(True)
    model.model.language_model.to("cuda")
    model.lm_head.to("cuda")
    model.model.visual.to("cpu")
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    def forbidden(*args, **kwargs):
        raise AssertionError("Generation is forbidden during decision training")
    model.generate = forbidden
    return model

def render(tokenizer, r, rng=None, long_pool=None, long_probability=0.):
    req = r["request"]
    # options의 이름은 의미상 정답 ID다. 출력 코드(A/B/숫자)와 표시 위치는 별도로 섞는다.
    # target은 섞인 names에서 answer_id의 위치로 다시 계산하므로 항상 같은 내용을 가리킨다.
    names = list(req["options"])
    codes = list(LETTERS[:len(names)])
    if rng is not None:
        rng.shuffle(names)
        if len(names) <= 9 and rng.random() < .25:
            codes = [str(i + 1) for i in range(len(names))]
        # Code assignment and option-content order are independent.
        rng.shuffle(codes)
    state = req["state"]
    augmentation = None
    if rng is not None and long_pool and r["task"] == "reading_comprehension" and rng.random() < long_probability:
        selected = rng.sample(long_pool, min(3, len(long_pool)))
        passages = [("Background passage", text, record_id) for text, record_id in selected if text != state]
        position = rng.randrange(len(passages) + 1)
        passages.insert(position, ("Relevant passage", state, r["id"]))
        state = "Answer using only the passage marked Relevant passage. Other passages are unrelated background.\n\n" + "\n\n".join(f"{title}:\n{text}" for title, text, _ in passages)
        augmentation = {"relevant_position": position, "background_training_ids": [x[2] for x in passages if x[0] != "Relevant passage"]}
    user = f"State:\n{state}\n\nQuestion:\n{req['question']}\n\nOptions:\n" + "\n".join(f"{code}. {req['options'][name]}" for code, name in zip(codes, names)) + "\n\nAnswer with only the option code."
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = []
    suffix = prompt[-64:]
    prefix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    for code in codes:
        token_ids = tokenizer.encode(code, add_special_tokens=False)
        assert len(token_ids) == 1
        assert tokenizer.encode(suffix + code, add_special_tokens=False) == prefix_ids + token_ids
        candidate_ids.append(token_ids[0])
    return {"input_ids": ids, "candidate_ids": candidate_ids, "target": names.index(r["answer_id"]), "weight": r["sample_weight"], "id": r["id"], "task": r["task"], "augmentation": augmentation}

def collate(tokenizer, rows):
    tokenizer.padding_side = "right"
    inputs = tokenizer.pad([{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"])} for r in rows], padding=True, return_tensors="pt").to("cuda")
    all_candidates = sorted({tid for r in rows for tid in r["candidate_ids"]})
    lookup = {v: i for i, v in enumerate(all_candidates)}
    width = max(len(r["candidate_ids"]) for r in rows)
    columns = [[lookup[v] for v in r["candidate_ids"]] + [0] * (width - len(r["candidate_ids"])) for r in rows]
    mask = [[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"])) for r in rows]
    return inputs, torch.tensor(all_candidates, device="cuda"), torch.tensor(columns, device="cuda"), torch.tensor(mask, device="cuda")

# 입력 전체를 읽되 loss는 마지막 실제 입력 토큰의 next-token 점수에서만 계산한다.
# [batch, sequence, vocab] 전체 logits 대신 후보 코드에 해당하는 LM Head 행만 사용한다.
# 각 문제에 없는 후보는 mask 처리하므로 다른 문제의 선택지가 softmax 분모에 섞이지 않는다.
def logits_for(model, tokenizer, rows):
    inputs, ids, columns, mask = collate(tokenizer, rows)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Same text backbone and head used at inference, without creating full
        # vocabulary logits at every sequence position.
        hidden = model.model.language_model(**inputs, use_cache=False).last_hidden_state
        last = inputs["attention_mask"].sum(-1) - 1
        h = hidden[torch.arange(len(rows), device="cuda"), last]
        head = model.get_output_embeddings()
        all_scores = F.linear(h, head.weight.index_select(0, ids), None)
    return all_scores.float().gather(1, columns).masked_fill(~mask, float("-inf"))

def optimizer_for(model, lr):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}], lr=lr, betas=(.9, .95), eps=1e-8, fused=True)

def microbatches(rows, args):
    group = []
    for r in sorted(rows, key=lambda x: len(x["input_ids"])):
        if group and (len(group) >= args.micro_batch or len(r["input_ids"]) * (len(group) + 1) > args.token_budget):
            yield group
            group = []
        group.append(r)
    if group:
        yield group

@torch.no_grad()
def evaluate(model, tokenizer, records, args):
    model.eval()
    results = defaultdict(lambda: {"correct": 0, "total": 0, "weighted_loss_sum": 0., "weight_sum": 0.})
    encoded = [render(tokenizer, r) for r in records]
    for batch in microbatches(encoded, args):
        logits = logits_for(model, tokenizer, batch)
        targets = torch.tensor([r["target"] for r in batch], device="cuda")
        losses = F.cross_entropy(logits, targets, reduction="none").tolist()
        correct = (logits.argmax(-1) == targets).tolist()
        for r, loss, ok in zip(batch, losses, correct):
            d = results[r["task"]]
            d["correct"] += int(ok)
            d["total"] += 1
            d["weighted_loss_sum"] += loss * r["weight"]
            d["weight_sum"] += r["weight"]
    for d in results.values():
        d["accuracy"] = d["correct"] / d["total"]
        d["weighted_loss"] = d["weighted_loss_sum"] / d["weight_sum"]
    return {"macro_accuracy": sum(d["accuracy"] for d in results.values()) / len(results), "macro_weighted_loss": sum(d["weighted_loss"] for d in results.values()) / len(results), "by_task": dict(results)}

def checkpoint(model, optimizer, directory, state):
    stage = directory / "checkpoint-staging"
    if stage.exists():
        raise RuntimeError(f"Unfinished checkpoint exists: {stage}")
    stage.mkdir(parents=True)
    # Save trained language weights separately. Original vision weights remain
    # fixed and are recovered from the original local model on load/export.
    save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()}, str(stage / "language.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "state": state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, stage / "training_state.pt")
    atomic_json(stage / "state.json", state)
    old = directory / "checkpoint-previous"
    latest = directory / "latest"
    if latest.exists():
        latest.rename(old)
    stage.rename(latest)
    if old.exists():
        shutil.rmtree(old)

def export_model(model, processor, directory):
    directory.mkdir(parents=True, exist_ok=True)
    # save_pretrained handles shared embedding/head weights. Preserve the
    # original multimodal architecture and frozen vision encoder.
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(directory)

def preflight(model, tokenizer, train, args, out):
    rng = random.Random(123)
    selected = sorted(train, key=lambda r: r["id"])[:4]
    encoded = [render(tokenizer, r, rng) for r in selected]
    # Check that bypassing the multimodal wrapper gives the same text scores.
    model.eval()
    with torch.no_grad():
        actual = logits_for(model, tokenizer, encoded)
        inputs, ids, columns, mask = collate(tokenizer, encoded)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            last = inputs["attention_mask"].sum(-1) - 1
            h = hidden[torch.arange(len(encoded), device="cuda"), last]
            expected = F.linear(h, model.get_output_embeddings().weight.index_select(0, ids)).float()
        expected = expected.gather(1, columns).masked_fill(~mask, float("-inf"))
        assert torch.allclose(actual[mask], expected[mask], atol=.04, rtol=.01)
        wrapper_max_difference = (actual[mask] - expected[mask]).abs().max().item()
    del actual, expected, hidden, h, inputs
    model.train()
    optimizer = optimizer_for(model, args.lr)
    tracked = model.model.language_model.layers[-1].mlp.down_proj.weight
    before = tracked.detach().clone()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(model, tokenizer, encoded)
        targets = torch.tensor([r["target"] for r in encoded], device="cuda")
        loss = F.cross_entropy(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(loss.item())
    assert not torch.equal(before, tracked)
    # A real long augmented reading example must also have finite backward.
    reading = [r for r in train if r["task"] == "reading_comprehension"]
    pool = [(r["request"]["state"], r["id"]) for r in reading]
    long_example = None
    for r in sorted(reading, key=lambda r: -len(r["request"]["state"])):
        e = render(tokenizer, r, random.Random(7), pool, 1.)
        if 2048 <= len(e["input_ids"]) <= args.max_tokens:
            long_example = e
            break
    assert long_example is not None
    optimizer.zero_grad(set_to_none=True)
    logits = logits_for(model, tokenizer, [long_example])
    loss = F.cross_entropy(logits, torch.tensor([long_example["target"]], device="cuda"))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(tracked.grad).all()
    result = {"passed": True, "short_training_losses": losses, "master_weights_changed": True,
              "finite_gradients": True, "long_input_tokens": len(long_example["input_ids"]),
              "long_backward_loss": loss.item(), "vision_frozen": all(not p.requires_grad for p in model.model.visual.parameters()),
              "fp32_master_parameters": tracked.dtype == torch.float32, "generated_tokens": 0,
              "tied_embedding_head": True, "text_wrapper_max_logit_difference": wrapper_max_difference,
              "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
    atomic_json(out / "preflight.json", result)
    print(json.dumps(result), flush=True)

def train_run(model, processor, train, dev, args, out, seed, mixture):
    seed_all(seed)
    tokenizer = processor.tokenizer
    directory = out / f"seed-{seed}"
    directory.mkdir()
    by_task = defaultdict(list)
    for r in train:
        by_task[r["task"]].append(r)
    rng = random.Random(seed)
    if mixture == "natural":
        selected = list(train)
        rng.shuffle(selected)
    else:
        task_names = sorted(by_task)
        probabilities = [math.sqrt(len(by_task[t])) for t in task_names]
        selected = [rng.choice(by_task[t]) for t in rng.choices(task_names, weights=probabilities, k=len(train))]
    # Local length sorting reduces padding while randomizing bucket order and
    # the examples inside a bucket. Content/code permutations still change.
    selected = [r for start in range(0, len(selected), 256) for r in sorted(selected[start:start + 256], key=lambda r: len(r["request"]["state"]) + len(r["request"]["question"]))]
    pool = [(r["request"]["state"], r["id"]) for r in train if r["task"] == "reading_comprehension"]
    optimizer = optimizer_for(model, args.lr)
    params = [p for p in model.parameters() if p.requires_grad]
    steps = math.ceil(len(selected) / args.effective_batch)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    baseline = evaluate(model, tokenizer, dev, args)
    atomic_json(directory / "dev-before.json", baseline)
    start = time.time()
    processed = skipped = augmentations = 0
    length_histogram = Counter()
    best_score = baseline["macro_accuracy"]
    last_dev = baseline
    atomic_json(directory / "plan.json", {"seed": seed, "mixture": mixture, "steps": steps, "draws": len(selected), "task_draw_counts": dict(Counter(r["task"] for r in selected))})
    with (directory / "metrics.jsonl").open("w", buffering=1) as log:
        for step in range(1, steps + 1):
            model.train()
            examples = selected[(step - 1) * args.effective_batch: step * args.effective_batch]
            encoded = []
            for r in examples:
                e = render(tokenizer, r, rng, pool, args.long_probability)
                if len(e["input_ids"]) > args.max_tokens and e["augmentation"]:
                    # Revert augmentation, not the underlying source input.
                    e = render(tokenizer, r, rng)
                if len(e["input_ids"]) > args.max_tokens:
                    skipped += 1
                    continue
                encoded.append(e)
                length_histogram[str(2 ** math.ceil(math.log2(max(1, len(e["input_ids"])))))] += 1
                augmentations += int(e["augmentation"] is not None)
            if not encoded:
                continue
            warmup = max(1, math.ceil(steps * .05))
            factor = step / warmup if step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            # 전체 effective batch의 가중치 합으로 각 microbatch를 나눈 뒤 gradient를 누적한다.
            # microbatch마다 평균을 내면 길이 때문에 배치가 작아진 샘플의 영향이 커질 수 있다.
            weight_sum = sum(r["weight"] for r in encoded)
            loss_sum, correct = 0., 0
            tick = time.time()
            for batch in microbatches(encoded, args):
                logits = logits_for(model, tokenizer, batch)
                targets = torch.tensor([r["target"] for r in batch], device="cuda")
                weights = torch.tensor([r["weight"] for r in batch], device="cuda")
                # 문제별 CE = -log softmax(logits)[정답 인덱스]. 정답 토큰이나 풀이를 입력에 넣지 않는다.
                # silver 인지 라벨 등 신뢰도가 낮은 샘플은 sample_weight로 영향을 줄인다.
                weighted_sum = (F.cross_entropy(logits, targets, reduction="none") * weights).sum()
                loss = weighted_sum / weight_sum
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at {seed}:{step}")
                loss.backward()
                loss_sum += weighted_sum.item()
                correct += int((logits.argmax(-1) == targets).sum())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient norm at {seed}:{step}")
            optimizer.step()
            processed += len(encoded)
            metric = {"seed": seed, "step": step, "planned_steps": steps, "examples_seen": processed,
                      "loss": loss_sum / weight_sum, "batch_accuracy": correct / len(encoded),
                      "lr": optimizer.param_groups[0]["lr"], "gradient_norm": grad_norm,
                      "optimizer_step_seconds": time.time() - tick, "elapsed_seconds": time.time() - start,
                      "oversize_source_rows_skipped": skipped, "long_augmented_examples_seen": augmentations,
                      "input_length_buckets": dict(length_histogram), "generated_tokens": 0}
            log.write(json.dumps(metric) + "\n")
            atomic_json(out / "progress.json", {"status": "training", "mixture": mixture, **metric})
            if step <= 5 or step % 10 == 0:
                print(f"seed={seed} step={step}/{steps} loss={metric['loss']:.4f} examples={processed} step_s={metric['optimizer_step_seconds']:.2f}", flush=True)
            if step % args.eval_every == 0 or step == steps:
                last_dev = evaluate(model, tokenizer, dev, args)
                atomic_json(directory / f"dev-step-{step}.json", last_dev)
                # Use development data for selection only. Test stays untouched.
                if last_dev["macro_accuracy"] > best_score:
                    best_score = last_dev["macro_accuracy"]
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()}, str(directory / "best-language.safetensors"))
                    atomic_json(directory / "best.json", {"step": step, "dev": last_dev})
            if step == 5 or step % args.save_every == 0 or step == steps:
                checkpoint(model, optimizer, directory, {**metric, "seed": seed, "mixture": mixture, "sampler_python_rng": rng.getstate(), "dev": last_dev})
        save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()}, str(directory / "final-language.safetensors"))
        atomic_json(directory / "complete.json", {"complete": True, "seed": seed, "mixture": mixture, "steps": steps, "examples_seen": processed, "dev": last_dev})
    return directory

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--long-probability", type=float, default=.25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dev-per-task", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    assert json.loads((DATA / "audit.json").read_text())["verified"]
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    for split in ("train", "dev", "test"):
        assert sha(DATA / f"{split}.jsonl") == data_manifest["stats"][split]["file_sha256"]
    processor = AutoProcessor.from_pretrained(ROOT / "model")
    train = read_records("train")
    by_task = defaultdict(list)
    for r in read_records("dev"):
        by_task[r["task"]].append(r)
    dev = [r for _, rr in sorted(by_task.items()) for r in sorted(rr, key=lambda x: x["id"])[:args.dev_per_task]]
    seed_all(41)
    model = model_load()
    if args.preflight:
        preflight(model, processor.tokenizer, train, args, args.output)
        return
    assert not (args.output / "manifest.json").exists(), "Preserve existing runs; use a new output directory."
    manifest = {"base_model": "Qwen/Qwen3.5-0.8B", "training": "full language-backbone and tied LM-head fine-tuning; vision frozen",
                "dtype": "FP32 master weights and moments; BF16 autocast", "loss": "sample-weighted cross-entropy over supplied option-code logits only",
                "data_manifest_sha256": sha(DATA / "manifest.json"), "train_examples": len(train), "dev_examples": len(dev),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runs": [{"seed": 41, "mixture": "natural"}, {"seed": 73, "mixture": "sqrt_task_counts"}],
                "source": "https://www.liquid.ai/blog/d1-open", "recipe_limit": "Inspired by disclosed practices; exact d1 data mixtures/loss are not published. No incompatible LFM weights are averaged into Qwen.",
                "augmentations": "independent content/code permutations; 25% numeric codes for <=9 candidates; 25% RACE uses up to 3 train-only background passages, relevant passage position random; no source input truncation",
                "generated_tokens": 0, "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_files_sha256": {p.name: sha(p) for p in (ROOT / "model").glob("*.safetensors")},
                "code_sha256": sha(Path(__file__)), "started_at_unix": time.time()}
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(args.output / "development-records.json", dev)
    # 두 run 모두 원본 Qwen에서 시작한다. seed 73은 seed 41을 이어 학습하는 2차 epoch가 아니다.
    # natural은 실제 개수 비율, sqrt_task_counts는 작은 과제의 상대적 비중을 높이는 혼합이다.
    first = train_run(model, processor, train, dev, args, args.output, 41, "natural")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = model_load()
    second = train_run(model, processor, train, dev, args, args.output, 73, "sqrt_task_counts")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Average corresponding FP32 parameters of independently tuned Qwen runs.
    a = load_file(str(first / "final-language.safetensors"))
    b = load_file(str(second / "final-language.safetensors"))
    assert a.keys() == b.keys()
    merged = {k: (a[k] + b[k]) * .5 if a[k].is_floating_point() else a[k] for k in a}
    merge_path = args.output / "averaged-language.safetensors"
    save_file(merged, str(merge_path))
    del a, b, merged
    gc.collect()
    candidates = [("seed-41-final", first / "final-language.safetensors"), ("seed-73-final", second / "final-language.safetensors"), ("averaged", merge_path)]
    for name, directory in (("seed-41", first), ("seed-73", second)):
        if (directory / "best-language.safetensors").exists():
            candidates.append((name + "-best", directory / "best-language.safetensors"))
    model = model_load()
    comparisons = []
    for name, path in candidates:
        state = load_file(str(path))
        model.model.language_model.load_state_dict(state, strict=True)
        del state
        result = evaluate(model, processor.tokenizer, dev, args)
        comparisons.append({"name": name, "path": str(path), "dev": result})
    # 최종/개발 최고/두 최종 가중치 평균 중 개발 표본만으로 선택한다. test를 보지 않는다.
    # 이 규칙과 개발 표본 크기를 바꾸면 선택 모델과 test 점수가 달라지므로 반드시 기록한다.
    winner = max(comparisons, key=lambda r: (r["dev"]["macro_accuracy"], -r["dev"]["macro_weighted_loss"]))
    state = load_file(winner["path"])
    model.model.language_model.load_state_dict(state, strict=True)
    del state
    export_model(model, processor, args.output / "model")
    atomic_json(args.output / "selection.json", {"selected": winner, "candidates": comparisons, "selection_partition": "development only"})
    atomic_json(args.output / "progress.json", {"status": "training_complete", "selected": winner["name"], "dev_macro_accuracy": winner["dev"]["macro_accuracy"], "model": str(args.output / "model")})
    print("TRAINING COMPLETE " + json.dumps(winner), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve partial metrics and checkpoints; do not silently claim success.
        import sys
        output = DEFAULT_OUT
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
        atomic_json(output / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error), "time": time.time()})
        raise
