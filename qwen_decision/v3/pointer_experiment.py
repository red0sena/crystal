"""Matched code/Pointer pilot on an existing v3 checkpoint; no benchmark training.

python -m qwen_decision.v3.pointer_experiment --base runs/v3/<run>/best \
    --mix data/mix/v3-gold5.jsonl --output runs/pointer/pilot

Each arm starts from the SAME backbone and gets the SAME augmented examples and
updates. First train only the readout; optionally continue from its best head with
the last N backbone layers unfrozen. Select on grouped gold dev, never on test.
This deliberately uses padded SDPA to keep the first experiment simple.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from qwen_decision.v3 import format as fmt, modeling, pointer
from qwen_decision.v3 import train as training
from qwen_decision.v3.mix import CORPUS_EXCLUDED, allowed_origin


def read_rows(path, limit=0):
    with path.open() as f:
        raw = [json.loads(line) for line in itertools.islice(f, limit or None)]
    if any(r["source"] in CORPUS_EXCLUDED for r in raw):
        raise ValueError(f"excluded teacher corpus in {path}")
    rows = [r for r in raw if allowed_origin(r, {"human", "programmatic"})]
    if not rows:
        raise ValueError(f"no eligible gold rows in {path}")
    counts = Counter(r["id"] for r in rows)
    # Some mix files retain an upstream id for both choice and noul variants.
    # Give distinct prompts stable evaluation ids, while retaining the upstream id for split checks.
    rows = [{**r, "upstream_id": r["id"], "id": (r["id"] + "#" + hashlib.sha256(
        json.dumps([r["state"], r["question"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        if counts[r["id"]] > 1 else r["id"])} for r in rows]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"duplicate id with the same prompt in {path}")
    return training.gold_targets(rows)


def check_disjoint(splits):
    """Check original grouped splits by upstream ids, document keys and exact normalized content.

    Do not rerun mix.data_groups on a small sample: its corpus-frequency boilerplate filter
    then treats common classification templates as documents and links unrelated short states.
    This does not replace the full-corpus near-duplicate audit used to create the mix.
    """
    seen = {}
    for name, examples in splits.items():
        for r in examples:
            content = r["state"] if r["state"] not in (None, "", {}) else r["question"]
            fingerprint = hashlib.sha256(re.sub(r"\s+", " ", json.dumps(
                content, sort_keys=True, ensure_ascii=False)).strip().encode()).hexdigest()
            keys = [("id", r.get("upstream_id", r["id"])), ("text", fingerprint)]
            if r.get("item"):
                keys.append(("item", r["item"]))
            keys += [("document", d) for d in r.get("documents") or []]
            for key in keys:
                if key in seen and seen[key] != name:
                    raise ValueError(f"train/dev/test overlap: {r['id']} ({seen[key]}, {name}; {key[0]})")
                seen[key] = name


class Renderer(training.Renderer):
    def __call__(self, example, rng=None):
        row = super().__call__(example, rng)
        lookup = {t: i for i, t in enumerate(self.code_ids)}
        return {**row, "code_columns": [lookup[t] for t in row["code_ids"]]}


def evaluation_rows(examples, renderer):
    rows = []
    for e in examples:
        k = len(e["target"])
        row = renderer(e)
        rows.append({**row, "order": list(range(k)), "variant": "original"})
        if e["ordered"] or renderer.positional(e):
            continue
        question = e["question"]
        if question["type"] == "noul":
            _, texts = fmt.options(question)
            if len(set(texts)) != k:
                continue
            question = {**question, "type": "choice", "criteria": dict.fromkeys(texts[::-1])}
        else:
            question = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
        shown = {**e, "question": question, "target": e["target"][::-1]}
        rows.append({**renderer(shown), "order": list(range(k))[::-1], "variant": "reversed"})
    return rows


@torch.no_grad()
def evaluate(model, head, rows, args, pad_id, output):
    model.eval()
    head.eval()
    records = []
    # Exclude first-call kernel startup from recorded latency and memory.
    warm = pointer.collate(rows[:1], pad_id, "cuda")
    pointer.forward(model.model.language_model, head, warm)
    del warm
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for group in training.microbatches(rows, args):
        batch = pointer.collate(group, pad_id, "cuda")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        logits = pointer.forward(model.model.language_model, head, batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        probabilities = logits.softmax(-1).tolist()
        for row, probs in zip(group, probabilities):
            k = len(row["target"])
            probs = probs[:k]
            pred = max(range(k), key=probs.__getitem__)
            original_probs = [0.] * k
            for i, index in enumerate(row["order"]):
                original_probs[index] = probs[i]
            records.append({"id": row["id"], "source": row["source"], "variant": row["variant"],
                            "prediction": row["order"][pred], "probabilities": original_probs,
                            "correct": row["target"][pred] >= max(row["target"]) - 1e-6,
                            "batch_forward_seconds": seconds, "batch_size": len(group),
                            "tokens": len(row["input_ids"])})
    by_source, by_id = defaultdict(list), defaultdict(dict)
    for r in records:
        by_id[r["id"]][r["variant"]] = r
        if r["variant"] == "original":
            by_source[r["source"]].append(r["correct"])
    sources = {s: {"n": len(v), "accuracy": sum(v) / len(v)} for s, v in sorted(by_source.items())}
    pairs = [v for v in by_id.values() if "reversed" in v]
    originals = [r for r in records if r["variant"] == "original"]
    result = {"n": len(originals), "accuracy": sum(r["correct"] for r in originals) / len(originals),
              "macro_source_accuracy": sum(v["accuracy"] for v in sources.values()) / len(sources),
              "sources": sources, "order_pairs": len(pairs),
              "order_agreement": (sum(v["original"]["prediction"] == v["reversed"]["prediction"]
                                      for v in pairs) / len(pairs) if pairs else None),
              "reversed_accuracy": sum(v["reversed"]["correct"] for v in pairs) / len(pairs) if pairs else None,
              "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "forward_peak_extra_gb": (torch.cuda.max_memory_allocated() - resident) / 1e9,
              "inference_parameters_gb": sum(p.numel() * p.element_size() for module in (model, head)
                                              for p in module.parameters() if p.is_cuda) / 1e9,
              "memory_scope": "peak total includes resident optimizer state; forward extra excludes it; allocated bytes, not reserved",
              "amortized_forward_ms": 1000 * sum(r["batch_forward_seconds"] / r["batch_size"]
                                                  for r in records) / len(records),
              "latency_scope": "warm padded-batch backbone+head only, amortized per question; excludes rendering/H2D"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return result


def configure_trainable(model, head, last_layers):
    model.requires_grad_(False)
    head.requires_grad_(True)
    if last_layers:
        layers = model.model.language_model.layers
        if not 0 < last_layers <= len(layers):
            raise ValueError("--unfreeze-last-layers exceeds the backbone depth")
        for layer in layers[-last_layers:]:
            layer.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def train_phase(model, head, renderer, train, dev, args, out, steps, last_layers):
    out.mkdir(parents=True, exist_ok=True)
    pad_id = renderer.tokenizer.pad_token_id
    configure_trainable(model, head, last_layers)
    container = torch.nn.ModuleDict({"model": model, "head": head})
    optimizer = training.MasterAdamW(container, args.backbone_lr if last_layers else args.head_lr)
    # The head always gets its own LR, including during limited backbone fine-tuning.
    masters = {id(p): m for p, m in optimizer.pairs}
    head_masters = {id(masters[id(p)]) for p in head.parameters()}
    for group in list(optimizer.param_groups):
        head_params = [p for p in group["params"] if id(p) in head_masters]
        group["params"] = [p for p in group["params"] if id(p) not in head_masters]
        if head_params:
            optimizer.optimizer.add_param_group({**group, "params": head_params, "lr": args.head_lr})
    history = {"0": evaluate(model, head, dev, args, pad_id, out / "dev-step-0.jsonl")}
    best, best_step = history["0"]["macro_source_accuracy"], 0

    def snapshot():
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(out / "head.safetensors"))
        if last_layers:
            save_file(training.language_state(model), str(out / "language.safetensors"))

    snapshot()
    schedule = training.batches(len(train), args.effective_batch, steps, random.Random(args.seed))
    rng = random.Random(args.seed + 1)
    seen = 0
    with (out / "train.jsonl").open("w") as log:
        for step, (indices, _) in enumerate(schedule, 1):
            encoded = [renderer(train[i], rng) for i in indices]
            if any(len(r["input_ids"]) > args.max_tokens for r in encoded):
                raise ValueError("an augmented prompt exceeds --max-tokens; increase the limit (never truncated)")
            # Frozen backbone remains in eval mode; both heads have no dropout.
            model.train(bool(last_layers))
            head.train()
            optimizer.zero_grad()
            loss_sum = 0.
            for group in training.microbatches(encoded, args):
                batch = pointer.collate(group, pad_id, "cuda")
                logits = pointer.forward(model.model.language_model, head, batch)
                loss = pointer.soft_ce(logits, batch["targets"]).sum() / len(encoded)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                loss_sum += loss.item()
            norm = optimizer.clip_grad_norm(1.)
            if not math.isfinite(norm):
                raise RuntimeError("nonfinite gradient norm")
            optimizer.step()
            seen += len(encoded)
            log.write(json.dumps({"step": step, "loss": loss_sum, "rows_seen": seen, "grad_norm": norm}) + "\n")
            if step % args.eval_every == 0 or step == steps:
                result = evaluate(model, head, dev, args, pad_id, out / f"dev-step-{step}.jsonl")
                history[str(step)] = result
                if result["macro_source_accuracy"] > best:
                    best, best_step = result["macro_source_accuracy"], step
                    snapshot()
                training.atomic_json(out / "history.json", history)
                print(f"{out.name} {step}/{steps}: loss={loss_sum:.4f} dev macro={result['macro_source_accuracy']:.4f}", flush=True)
    head.load_state_dict(load_file(str(out / "head.safetensors")))
    if last_layers:
        model.model.language_model.load_state_dict(load_file(str(out / "language.safetensors")))
    result = {"best_step": best_step, "updates": steps, "rows_seen": seen,
              "trainable_parameters": sum(p.numel() for p in container.parameters() if p.requires_grad),
              "initial_dev": history["0"],
              "dev_selected": history[str(best_step)]}
    training.atomic_json(out / "result.json", result)
    return result


def load_run(path: Path, device="cuda"):
    """Reload an arm directory without merging experimental heads into the stock LM checkpoint."""
    config = json.loads((path / "readout_config.json").read_text())
    model = modeling.load(Path(config["base"]), attention=config["attention"], deltanet=config["deltanet"], device=device)
    selected = path / config["selected"]
    if (selected / "language.safetensors").exists():
        model.model.language_model.load_state_dict(load_file(str(selected / "language.safetensors")))
    state = load_file(str(selected / "head.safetensors"))
    if config["arm"] == "code":
        head = pointer.CodeHead(state["weight"])
    else:
        head = pointer.PointerHead(state["query.weight"].shape[1], config["projection_size"])
    head.load_state_dict(state)
    return model.eval(), head.to(device).eval(), config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--mix", type=Path, required=True)
    p.add_argument("--dev", type=Path, help="default: <mix>.dev.jsonl")
    p.add_argument("--test", type=Path, help="optional independent gold mix-format set, evaluated only after selection")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-rows", type=int, default=2048)
    p.add_argument("--dev-per-source", type=int, default=32)
    p.add_argument("--head-steps", type=int, default=100)
    p.add_argument("--tune-steps", type=int, default=0)
    p.add_argument("--unfreeze-last-layers", type=int, default=2)
    p.add_argument("--projection-size", type=int, default=128)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=3e-6)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--token-budget", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()
    for name in ("head_steps", "effective_batch", "micro_batch", "token_budget", "max_tokens", "eval_every", "projection_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.train_rows, args.dev_per_source, args.tune_steps) < 0 or (args.tune_steps and args.unfreeze_last_layers < 1):
        p.error("row/step limits must be nonnegative; tuning needs at least one unfrozen layer")
    if args.head_lr <= 0 or args.backbone_lr <= 0:
        p.error("learning rates must be positive")
    if not args.base.is_dir() or not args.mix.is_file():
        p.error("restore --base checkpoint and --mix from Hugging Face first")
    args.dev = args.dev or args.mix.with_suffix(".dev.jsonl")
    if not args.dev.is_file() or (args.test and not args.test.is_file()):
        p.error("dev/test file does not exist")
    if args.output.exists():
        p.error("preserve existing runs: use a new output directory")
    config = modeling.read_config(args.base)
    if config["attention"] != "noncausal":
        p.error("this experiment needs an existing v3 noncausal checkpoint")
    if (args.base / modeling.READOUT_NAME).exists():
        p.error("use our v3 checkpoint with an LM head; separate d3 readouts are outside this comparison")
    if not torch.cuda.is_available():
        p.error("the pilot runner needs CUDA (the unit tests also cover CPU)")
    args.pack = False
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    render_args = SimpleNamespace(layout=config["layout"], pointer_positions=True, pad_probability=0.,
                                  noul_choice_probability=.2, code_shuffle_probability=.2, code_sample_probability=.2)
    renderer = Renderer(tokenizer, render_args, [])
    splits = {"train": read_rows(args.mix, args.train_rows), "dev": read_rows(args.dev)}
    if args.dev_per_source:
        counts = Counter()
        kept = []
        for r in sorted(splits["dev"], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
            if counts[r["source"]] < args.dev_per_source:
                kept.append(r)
                counts[r["source"]] += 1
        splits["dev"] = kept
    if args.test:
        splits["test"] = read_rows(args.test)
    check_disjoint(splits)
    skipped = {}
    for name, examples in splits.items():
        accepted, skipped[name] = [], []
        for e in examples:
            variants = [renderer(e)] if name == "train" else evaluation_rows([e], renderer)
            if max(len(r["input_ids"]) for r in variants) <= args.max_tokens:
                accepted.append(e)
            else:
                skipped[name].append(e["id"])
        if not accepted:
            p.error(f"no {name} rows under --max-tokens")
        splits[name] = accepted
    args.output.mkdir(parents=True)
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "base_config": config, "split_ids": {s: [e["id"] for e in rows] for s, rows in splits.items()},
                "sha256": {str(f): training.sha256_file(f) for f in (args.mix, args.dev, args.test) if f},
                "base_files_sha256": {f.name: training.sha256_file(f) for f in sorted(args.base.iterdir())
                                      if f.is_file() and f.suffix in (".safetensors", ".json")},
                "skipped_over_max_tokens": skipped, "versions": {"torch": torch.__version__,
                    "transformers": __import__("transformers").__version__, "fla": __import__("fla").__version__},
                "implementation_sha256": {f.name: training.sha256_file(f) for f in
                    (Path(__file__), Path(pointer.__file__), Path(fmt.__file__), Path(training.__file__), Path(modeling.__file__))},
                "split_check": "upstream ids, item/document keys and exact normalized full state (question when state empty); "
                               "reuse the mix's full-corpus grouped split; no new near-duplicate guarantee",
                "policy": "same v3 input, padded SDPA, same draws/updates; gold-dev selection; no held-out claim for dev"}
    training.atomic_json(args.output / "manifest.json", manifest)
    dev = evaluation_rows(splits["dev"], renderer)
    report = {"interpretation": "dev is used for selection; independent test required for accuracy claims", "arms": {}}
    for arm in ("code", "pointer"):
        training.seed_all(args.seed)
        model = modeling.load(args.base, attention=config["attention"], deltanet=config["deltanet"], kernel="sdpa")
        code_weights = model.get_output_embeddings().weight[renderer.code_ids]
        head = (pointer.CodeHead(code_weights) if arm == "code" else
                pointer.PointerHead(code_weights.shape[1], args.projection_size)).to("cuda")
        del code_weights
        out = args.output / arm
        result = {"frozen": train_phase(model, head, renderer, splits["train"], dev, args, out / "frozen", args.head_steps, 0)}
        selected = out / "frozen"
        if args.tune_steps:
            training.seed_all(args.seed)
            result["tuned"] = train_phase(model, head, renderer, splits["train"], dev, args, out / "tuned",
                                          args.tune_steps, args.unfreeze_last_layers)
            selected = out / "tuned"
        if args.test:
            result["test"] = evaluate(model, head, evaluation_rows(splits["test"], renderer), args,
                                      tokenizer.pad_token_id, selected / "test-predictions.jsonl")
        training.atomic_json(out / "readout_config.json", {"arm": arm, "projection_size": args.projection_size,
            "base": str(args.base.resolve()), "selected": selected.name, "code_ids": renderer.code_ids,
            "layout": config["layout"], "attention": config["attention"], "deltanet": config["deltanet"]})
        report["arms"][arm] = result
        training.atomic_json(args.output / "summary.json", report)
        del head, model
        gc.collect()
        torch.cuda.empty_cache()
    for phase in ("frozen", "tuned"):
        if phase in report["arms"]["code"]:
            a, b = (report["arms"][arm][phase]["dev_selected"] for arm in ("code", "pointer"))
            report[f"{phase}_dev_macro_delta_pointer_minus_code"] = b["macro_source_accuracy"] - a["macro_source_accuracy"]
    report["initial_checkpoint_dev"] = report["arms"]["code"]["frozen"]["initial_dev"]
    training.atomic_json(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
