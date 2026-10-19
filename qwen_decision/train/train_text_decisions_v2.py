"""v2 decision training: 2-64 options, three question types, soft targets,
long documents and robustness augmentation.

Same optimizer, schedule, two-run averaging and development-only selection as
v1 (train_text_decisions.py). Every prompt is rendered by decision_format.py,
the module inference uses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from safetensors.torch import load_file, save_file

import qwen_decision.core.decision_format as fmt
from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/text-decisions-v2"
DEFAULT_OUT = ROOT / "training/text-decision-v2"
MASK = -1e9  # finite, so 0 * log p stays 0 for padded candidate columns
CODE_MODES = {"choice": (("sequential", .6), ("shuffled", .2), ("subset", .2)),
              "noul": (("sequential", .6), ("shuffled", .2), ("subset", .2)),
              "score": (("sequential", .8), ("subset", .2))}


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
    model = Qwen3_5ForConditionalGeneration.from_pretrained(ROOT / "model", dtype=torch.float32, attn_implementation="sdpa")
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


def pick(rng, weighted):
    x, acc = rng.random(), 0.
    for value, weight in weighted:
        acc += weight
        if x < acc:
            return value
    return weighted[-1][0]


def sample_k(rng, max_k):
    buckets = [(2, 16), (17, 32), (33, fmt.MAX_OPTIONS)]
    lo, hi = rng.choice([b for b in buckets if b[0] <= max_k])
    return rng.randint(lo, min(hi, max_k))


def inject(rng, state, code):
    line = rng.choice(fmt.INJECTIONS).format(code=code)
    parts = state.split("\n")
    parts.insert(rng.randrange(len(parts) + 1), line)
    return "\n".join(parts)


def render(tokenizer, r, rng=None, long_pool=None, args=None):
    """Return model inputs for one record. rng=None renders the fixed inference
    format (stored option order, sequential codes, no augmentation)."""
    req, qtype = r["request"], r["qtype"]
    options = dict(req["options"])
    names = list(options)
    gold = r["answer_id"]
    target = dict(r.get("target") or {gold: 1.})
    info = {"escape": None, "injected": False, "mode": "sequential", "long": None}
    if rng is not None:
        escape = (qtype == "choice" and "target" not in r and rng.random() < args.escape_probability)
        if "option_sampling" in r:
            k = sample_k(rng, min(r["option_sampling"]["max_k"], len(names)))
            k = min(k, fmt.MAX_OPTIONS - int(escape))
            names = [gold, *rng.sample([x for x in names if x != gold], k - 1)]
        if qtype != "score":
            rng.shuffle(names)
        # 기존 v2 재현을 위해 그대로 보존한 증강. 정답 제거 여부에 따라 최종 선택지 수가
        # k 또는 k+1로 달라지는 누출 가능성이 발견됐다. 수정 실험은 새 버전/manifest로 분리한다.
        if escape:
            remove_gold = rng.random() < .5 and len(names) > 2
            if remove_gold:
                names.remove(gold)
                target = {fmt.ESCAPE_NAME: 1.}
            names.insert(rng.randrange(len(names) + 1), fmt.ESCAPE_NAME)
            options[fmt.ESCAPE_NAME] = fmt.ESCAPE_TEXT
            info["escape"] = "gold_removed" if remove_gold else "gold_kept"
        info["mode"] = pick(rng, CODE_MODES[qtype])
    if len(names) > fmt.MAX_OPTIONS:
        raise ValueError(f"{r['id']} renders {len(names)} options without sampling.")
    codes = fmt.assign_codes(len(names), info["mode"], rng)
    state = req["state"]
    if rng is not None and long_pool and r["task"] == "reading_comprehension" and rng.random() < args.long_probability:
        # Kept from v1 for comparability: unrelated train-only passages, relevant one marked.
        selected = rng.sample(long_pool, min(3, len(long_pool)))
        passages = [("Background passage", text) for text, _ in selected if text != state]
        position = rng.randrange(len(passages) + 1)
        passages.insert(position, ("Relevant passage", state))
        state = ("Answer using only the passage marked Relevant passage. Other passages are unrelated background.\n\n"
                 + "\n\n".join(f"{title}:\n{text}" for title, text in passages))
        info["long"] = position
    if rng is not None and rng.random() < args.injection_probability:
        state = inject(rng, state, rng.choice(codes))
        info["injected"] = True
    prompt = fmt.prompt_text(tokenizer, state, req["question"], codes, [options[n] for n in names], qtype)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    # target 분포를 실제 표시 순서로 재배열한다. 일반 분류는 one-hot, STS-B는 이웃 점수에
    # 나뉜 soft target이다. score 선택지의 낮음→높음 순서는 유지한다.
    probs = [target.get(n, 0.) for n in names]
    assert abs(sum(probs) - 1) < 1e-6, (r["id"], names, target)
    hard = max(range(len(names)), key=lambda i: probs[i])
    return {"input_ids": ids, "candidate_ids": fmt.code_ids_for(tokenizer, codes), "target": probs, "hard": hard,
            "codes": codes, "weight": r["sample_weight"], "id": r["id"], "task": r["task"], "qtype": qtype, **info}


def collate(tokenizer, rows):
    tokenizer.padding_side = "right"
    inputs = tokenizer.pad([{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"])} for r in rows],
                           padding=True, return_tensors="pt").to("cuda")
    all_candidates = sorted({tid for r in rows for tid in r["candidate_ids"]})
    lookup = {v: i for i, v in enumerate(all_candidates)}
    width = max(len(r["candidate_ids"]) for r in rows)
    columns = [[lookup[v] for v in r["candidate_ids"]] + [0] * (width - len(r["candidate_ids"])) for r in rows]
    mask = [[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"])) for r in rows]
    targets = [r["target"] + [0.] * (width - len(r["target"])) for r in rows]
    return (inputs, torch.tensor(all_candidates, device="cuda"), torch.tensor(columns, device="cuda"),
            torch.tensor(mask, device="cuda"), torch.tensor(targets, device="cuda", dtype=torch.float32))


# 입력 전체를 읽되 loss는 마지막 실제 입력 토큰의 next-token 점수에서만 계산한다.
# [batch, sequence, vocab] 전체 logits 대신 후보 코드에 해당하는 LM Head 행만 사용한다.
# 각 문제에 없는 후보는 mask 처리하므로 다른 문제의 선택지가 softmax 분모에 섞이지 않는다.
def logits_for(model, tokenizer, rows):
    inputs, ids, columns, mask, targets = collate(tokenizer, rows)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model.language_model(**inputs, use_cache=False).last_hidden_state
        last = inputs["attention_mask"].sum(-1) - 1
        h = hidden[torch.arange(len(rows), device="cuda"), last]
        all_scores = F.linear(h, model.get_output_embeddings().weight.index_select(0, ids), None)
    return all_scores.float().gather(1, columns).masked_fill(~mask, MASK), targets


# 샘플별 soft CE: -sum_i q_i log p_i. q가 one-hot이면 v1의 -log p_gold와 같다.
# 가변 선택지의 padding target은 0이며 logits mask는 큰 음수다(MASK 상수 참조).
# 유한 mask를 써서 0 * (-inf)가 NaN이 되는 것을 방지한다.
def soft_ce(logits, targets):
    return -(targets * logits.log_softmax(-1)).sum(-1)


def optimizer_for(model, lr):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}],
                             lr=lr, betas=(.9, .95), eps=1e-8, fused=True)


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
        logits, targets = logits_for(model, tokenizer, batch)
        losses = soft_ce(logits, targets).tolist()
        correct = (logits.argmax(-1) == torch.tensor([r["hard"] for r in batch], device="cuda")).tolist()
        for r, loss, ok in zip(batch, losses, correct):
            d = results[r["task"]]
            d["correct"] += int(ok)
            d["total"] += 1
            d["weighted_loss_sum"] += loss * r["weight"]
            d["weight_sum"] += r["weight"]
    for d in results.values():
        d["accuracy"] = d["correct"] / d["total"]
        d["weighted_loss"] = d["weighted_loss_sum"] / d["weight_sum"]
    return {"macro_accuracy": sum(d["accuracy"] for d in results.values()) / len(results),
            "macro_weighted_loss": sum(d["weighted_loss"] for d in results.values()) / len(results),
            "by_task": dict(results)}


def checkpoint(model, optimizer, directory, state):
    stage = directory / "checkpoint-staging"
    if stage.exists():
        raise RuntimeError(f"Unfinished checkpoint exists: {stage}")
    stage.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()},
              str(stage / "language.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "state": state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, stage / "training_state.pt")
    atomic_json(stage / "state.json", state)
    old, latest = directory / "checkpoint-previous", directory / "latest"
    if latest.exists():
        latest.rename(old)
    stage.rename(latest)
    if old.exists():
        shutil.rmtree(old)


def render_stats(encoded):
    """Coverage of option counts and target codes actually seen in training."""
    return {"option_count": dict(sorted(Counter(len(e["codes"]) for e in encoded).items())),
            "target_code": dict(Counter(e["codes"][e["hard"]] for e in encoded)),
            "mode": dict(Counter(e["mode"] for e in encoded)), "escape": dict(Counter(str(e["escape"]) for e in encoded)),
            "injected": sum(e["injected"] for e in encoded), "qtype": dict(Counter(e["qtype"] for e in encoded))}


def preflight(model, tokenizer, train, args, out):
    rng = random.Random(123)
    by_task = defaultdict(list)
    for r in train:
        by_task[r["task"]].append(r)
    sample = [rr[0] for _, rr in sorted(by_task.items())]
    encoded = [render(tokenizer, r, rng, None, args) for r in sample]
    # Rendering coverage on a larger sample: every code must be a reachable target.
    big = [render(tokenizer, r, random.Random(i), None, args) for i, r in enumerate(random.Random(5).sample(train, 20000))]
    stats = render_stats(big)
    missing = [c for c in fmt.CODES if c not in stats["target_code"]]
    model.eval()
    with torch.no_grad():
        actual, _ = logits_for(model, tokenizer, encoded[:4])
        inputs, ids, columns, mask, _ = collate(tokenizer, encoded[:4])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            h = hidden[torch.arange(4, device="cuda"), inputs["attention_mask"].sum(-1) - 1]
            expected = F.linear(h, model.get_output_embeddings().weight.index_select(0, ids)).float()
        expected = expected.gather(1, columns).masked_fill(~mask, MASK)
        wrapper_max_difference = (actual[mask] - expected[mask]).abs().max().item()
        assert wrapper_max_difference < .1
    model.train()
    optimizer = optimizer_for(model, args.lr)
    tracked = model.model.language_model.layers[-1].mlp.down_proj.weight
    before = tracked.detach().clone()
    losses, step_seconds = [], []
    for _ in range(3):
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch in microbatches(encoded, args):
            logits, targets = logits_for(model, tokenizer, batch)
            loss = soft_ce(logits, targets).mean()
            assert torch.isfinite(loss)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(loss.item())
        step_seconds.append(time.time() - tick)
    assert not torch.equal(before, tracked)
    longest = max((render(tokenizer, r) for r in train if r["task"] == "long_reading"), key=lambda e: len(e["input_ids"]))
    optimizer.zero_grad(set_to_none=True)
    tick = time.time()
    logits, targets = logits_for(model, tokenizer, [longest])
    loss = soft_ce(logits, targets).mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(tracked.grad).all()
    result = {"passed": not missing, "codes_never_targeted_in_20000_draws": missing, "render_stats_20000": stats,
              "short_training_losses": losses, "step_seconds_26_examples": step_seconds,
              "long_input_tokens": len(longest["input_ids"]), "long_forward_backward_seconds": time.time() - tick,
              "text_wrapper_max_logit_difference": wrapper_max_difference,
              "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9, "generated_tokens": 0}
    atomic_json(out / "preflight.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "render_stats_20000"}), flush=True)


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
    selected = [r for start in range(0, len(selected), 256)
                for r in sorted(selected[start:start + 256], key=lambda r: len(r["request"]["state"]) + len(r["request"]["question"]))]
    pool = [(r["request"]["state"], r["id"]) for r in train if r["task"] == "reading_comprehension"]
    optimizer = optimizer_for(model, args.lr)
    params = [p for p in model.parameters() if p.requires_grad]
    steps = math.ceil(len(selected) / args.effective_batch)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    baseline = evaluate(model, tokenizer, dev, args)
    atomic_json(directory / "dev-before.json", baseline)
    start = time.time()
    processed = skipped = 0
    seen = Counter()
    coverage = {"option_count": Counter(), "target_code": Counter(), "mode": Counter(), "escape": Counter(),
                "injected": 0, "qtype": Counter(), "input_length_buckets": Counter()}
    best_score = baseline["macro_accuracy"]
    last_dev = baseline
    atomic_json(directory / "plan.json", {"seed": seed, "mixture": mixture, "steps": steps, "draws": len(selected),
                                          "task_draw_counts": dict(Counter(r["task"] for r in selected))})
    with (directory / "metrics.jsonl").open("w", buffering=1) as log:
        for step in range(1, steps + 1):
            model.train()
            encoded = []
            for r in selected[(step - 1) * args.effective_batch: step * args.effective_batch]:
                e = render(tokenizer, r, rng, pool, args)
                if len(e["input_ids"]) > args.max_tokens and e["long"] is not None:
                    e = render(tokenizer, r, rng, None, args)
                if len(e["input_ids"]) > args.max_tokens:
                    skipped += 1
                    seen[f"skipped:{r['task']}"] += 1
                    continue
                encoded.append(e)
                coverage["option_count"][len(e["codes"])] += 1
                coverage["target_code"][e["codes"][e["hard"]]] += 1
                coverage["mode"][e["mode"]] += 1
                coverage["escape"][str(e["escape"])] += 1
                coverage["injected"] += int(e["injected"])
                coverage["qtype"][e["qtype"]] += 1
                coverage["input_length_buckets"][str(2 ** math.ceil(math.log2(max(1, len(e["input_ids"])))))] += 1
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
                logits, targets = logits_for(model, tokenizer, batch)
                weights = torch.tensor([r["weight"] for r in batch], device="cuda")
                weighted_sum = (soft_ce(logits, targets) * weights).sum()
                loss = weighted_sum / weight_sum
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at {seed}:{step}")
                loss.backward()
                loss_sum += weighted_sum.item()
                correct += int((logits.argmax(-1) == torch.tensor([r["hard"] for r in batch], device="cuda")).sum())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient norm at {seed}:{step}")
            optimizer.step()
            processed += len(encoded)
            metric = {"seed": seed, "step": step, "planned_steps": steps, "examples_seen": processed,
                      "loss": loss_sum / weight_sum, "batch_accuracy": correct / len(encoded),
                      "lr": optimizer.param_groups[0]["lr"], "gradient_norm": grad_norm,
                      "optimizer_step_seconds": time.time() - tick, "elapsed_seconds": time.time() - start,
                      "oversize_rows_skipped": skipped, "generated_tokens": 0}
            log.write(json.dumps(metric) + "\n")
            atomic_json(out / "progress.json", {"status": "training", "mixture": mixture, **metric})
            if step <= 5 or step % 10 == 0:
                print(f"seed={seed} step={step}/{steps} loss={metric['loss']:.4f} examples={processed} "
                      f"step_s={metric['optimizer_step_seconds']:.2f}", flush=True)
            if step % args.eval_every == 0 or step == steps:
                last_dev = evaluate(model, tokenizer, dev, args)
                atomic_json(directory / f"dev-step-{step}.json", last_dev)
                atomic_json(directory / "coverage.json", {k: dict(v) if isinstance(v, Counter) else v
                                                          for k, v in coverage.items()} | {"skipped": dict(seen)})
                if last_dev["macro_accuracy"] > best_score:
                    best_score = last_dev["macro_accuracy"]
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()},
                              str(directory / "best-language.safetensors"))
                    atomic_json(directory / "best.json", {"step": step, "dev": last_dev})
            if step % args.save_every == 0 or step == steps:
                checkpoint(model, optimizer, directory, {**metric, "seed": seed, "mixture": mixture,
                                                          "sampler_python_rng": rng.getstate(), "dev": last_dev})
        save_file({k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()},
                  str(directory / "final-language.safetensors"))
        atomic_json(directory / "complete.json", {"complete": True, "seed": seed, "mixture": mixture, "steps": steps,
                                                  "examples_seen": processed, "dev": last_dev})
    # The optimizer state is only needed to resume this run; it is ~9 GB.
    shutil.rmtree(directory / "latest", ignore_errors=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=12288)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--long-probability", type=float, default=.25)
    parser.add_argument("--escape-probability", type=float, default=.05)
    parser.add_argument("--injection-probability", type=float, default=.03)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--dev-per-task", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    for split in ("train", "dev", "test"):
        assert sha(DATA / f"{split}.jsonl") == data_manifest["stats"][split]["file_sha256"]
    processor = AutoProcessor.from_pretrained(ROOT / "model")
    fmt.code_token_ids(processor.tokenizer)
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
                "dtype": "FP32 master weights and moments; BF16 autocast",
                "loss": "sample-weighted soft cross-entropy over supplied option-code logits (one-hot targets reduce to CE)",
                "data_manifest_sha256": sha(DATA / "manifest.json"), "train_examples": len(train), "dev_examples": len(dev),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runs": [{"seed": 41, "mixture": "natural"}, {"seed": 73, "mixture": "sqrt_task_counts"}],
                "format": {"codes": fmt.CODES, "system": fmt.SYSTEM, "code_modes": CODE_MODES,
                           "module_sha256": sha(code_path("decision_format.py"))},
                "augmentations": "per-draw K sampling for wide tasks; content shuffle (choice/noul); code modes; escape option; injection lines; v1 RACE background passages",
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
    a = load_file(str(first / "final-language.safetensors"))
    b = load_file(str(second / "final-language.safetensors"))
    assert a.keys() == b.keys()
    merge_path = args.output / "averaged-language.safetensors"
    save_file({k: (a[k] + b[k]) * .5 if a[k].is_floating_point() else a[k] for k in a}, str(merge_path))
    del a, b
    gc.collect()
    candidates = [("seed-41-final", first / "final-language.safetensors"),
                  ("seed-73-final", second / "final-language.safetensors"), ("averaged", merge_path)]
    for name, directory in (("seed-41", first), ("seed-73", second)):
        if (directory / "best-language.safetensors").exists():
            candidates.append((name + "-best", directory / "best-language.safetensors"))
    model = model_load()
    comparisons = []
    for name, path in candidates:
        model.model.language_model.load_state_dict(load_file(str(path)), strict=True)
        comparisons.append({"name": name, "path": str(path), "dev": evaluate(model, processor.tokenizer, dev, args)})
    # 최종/개발 최고/두 최종 가중치 평균 중 개발 표본만으로 선택한다. test를 보지 않는다.
    # 이 규칙과 개발 표본 크기를 바꾸면 선택 모델과 test 점수가 달라지므로 반드시 기록한다.
    winner = max(comparisons, key=lambda r: (r["dev"]["macro_accuracy"], -r["dev"]["macro_weighted_loss"]))
    model.model.language_model.load_state_dict(load_file(winner["path"]), strict=True)
    (args.output / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "model", safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(args.output / "model")
    atomic_json(args.output / "selection.json", {"selected": winner, "candidates": comparisons, "selection_partition": "development only"})
    atomic_json(args.output / "progress.json", {"status": "training_complete", "selected": winner["name"],
                                                "dev_macro_accuracy": winner["dev"]["macro_accuracy"], "model": str(args.output / "model")})
    print("TRAINING COMPLETE " + json.dumps({k: v for k, v in winner.items() if k != "dev"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import sys
        output = DEFAULT_OUT
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
        atomic_json(output / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error), "time": time.time()})
        raise
