"""v3 training: gold-label cross-entropy onto the 255-code readout. No teacher model.

  python -m qwen_decision.v3.train --mix data/mix/v3-gold.jsonl --output runs/v3/<run> [--train-rows N] [--steps N]

Full language backbone and tied embedding/LM head, FP32 master weights and moments, vision frozen,
cross-entropy over the shown answer codes against each row's gold (gold_targets), with the v3 prompt
(format.py) and attention mode (modeling.py); forward and backward run in BF16 as at inference
(MasterAdamW). Training reads only a mix.py file (gold labels, label_origin/target_kind fields); the model
is selected on the mix's grouped gold dev set by accuracy averaged over sources, and best/ and last/ are
both exported (final -> best). --steps fixes the number of updates independently of the data size.

Augmentations (training draws only; evaluation renders the stored order):
- choice options shuffled (score levels keep their order);
- with --pad-probability, a choice question is padded with unrelated options
  from other rows (target 0) up to 17-255 options, so every answer code is a
  reachable target;
- with --noul-choice-probability, a noul question is shown as the 2-option
  choice the engine uses for reversed option orders.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from qwen_decision.v3 import format as fmt
from qwen_decision.v3 import modeling

ROOT = Path(__file__).resolve().parents[2]
PAD_BUCKETS = ((17, 64), (65, 160), (161, 255))


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    import fla
    import transformers
    import triton

    return {"torch": torch.__version__, "transformers": transformers.__version__, "fla": fla.__version__,
            "triton": triton.__version__, "python": __import__("sys").version.split()[0]}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_load(args):
    """BF16 model (FP32 masters live in MasterAdamW); language backbone and tied embedding/LM head trainable."""
    if args.compile:
        modeling.compile_elementwise()
    model = modeling.load(args.base, attention=args.attention, deltanet=args.deltanet, dtype=torch.bfloat16,
                          kernel=args.kernel)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight, "training reads the tied LM head"
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.language_model.parameters():
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


class Renderer:
    def __init__(self, tokenizer, args, pool: list[str]):
        self.tokenizer, self.args, self.pool = tokenizer, args, pool
        self.codes, self.code_ids = fmt.answer_codes(tokenizer)
        self._positional = {}

    def positional(self, example: dict) -> bool:
        from qwen_decision.v3.mix import positional_options
        if example["id"] not in self._positional:
            self._positional[example["id"]] = positional_options(example["question"])
        return self._positional[example["id"]]

    def __call__(self, example: dict, rng: random.Random | None = None) -> dict:
        """Model inputs for one example; rng=None renders the stored order with no augmentation."""
        question, target = example["question"], list(example["target"])
        keys, texts = fmt.options(question)
        info = {"padded": False, "noul_as_choice": False}
        if rng is not None and question["type"] == "noul" and rng.random() < self.args.noul_choice_probability:
            question = {"type": "choice", "instructions": question.get("instructions"),
                        "criteria": {text: None for text in texts}}
            keys = texts
            info["noul_as_choice"] = True
        order = list(range(len(keys)))
        fixed = example["ordered"] or self.positional(example)  # options refer to each other by position
        info["positional"] = fixed and not example["ordered"]
        if rng is not None and question["type"] == "choice" and not fixed:
            criteria = dict(question["criteria"])
            buckets = [b for b in PAD_BUCKETS if b[1] > len(keys)]  # only sizes above the question's own
            if buckets and rng.random() < self.args.pad_probability:
                lo, hi = rng.choice(buckets)
                # The top bucket leans toward 255 options: the last codes are targets only in nearly full lists.
                want = (max(lo, hi - int(rng.expovariate(1 / 25))) if hi == 255
                        else rng.randint(max(lo, len(keys) + 1), hi))
                want = max(want, len(keys) + 1)
                taken = set(criteria)
                while len(criteria) < want:
                    extra = rng.choice(self.pool)
                    if extra not in taken:
                        taken.add(extra)
                        criteria[extra] = None
                        target.append(0.)
                question = {**question, "criteria": criteria}
                keys = list(criteria)
                info["padded"] = True
            order = list(range(len(keys)))
            rng.shuffle(order)
        if question["type"] == "choice":
            question = {**question, "criteria": {keys[i]: question["criteria"][keys[i]] for i in order}}
        target = [target[i] for i in order]
        k = len(target)
        # Codes per displayed position. Inference always shows A, B, C, ... in order, so most rows keep that; the
        # rest decouple code from position (v2 plan, docs/v2_plan_ko.md B): the first k codes shuffled, or k codes
        # drawn from all 255 and listed in code order, so every code can be a target even in short lists.
        assigned, info["code_assignment"] = list(range(k)), "sequential"
        if rng is not None and question["type"] == "choice" and not fixed:
            draw = rng.random()
            if draw < self.args.code_shuffle_probability:
                rng.shuffle(assigned)
                info["code_assignment"] = "shuffled"
            elif draw < self.args.code_shuffle_probability + self.args.code_sample_probability:
                assigned = sorted(rng.sample(range(len(self.codes)), k))
                info["code_assignment"] = "sampled"
        text = fmt.render(self.tokenizer, example["state"], question, [self.codes[c] for c in assigned], self.args.layout)
        encoded = (fmt.encode_options(self.tokenizer, example["state"], question,
                                      [self.codes[c] for c in assigned], self.args.layout)
                   if getattr(self.args, "pointer_positions", False)
                   else {"input_ids": self.tokenizer.encode(text, add_special_tokens=False)})
        hard = max(range(k), key=target.__getitem__)
        return {**encoded,
                "code_ids": [self.code_ids[c] for c in assigned], "target": target, "hard": hard,
                "hard_code": self.codes[assigned[hard]], "id": example["id"], "kind": example["kind"],
                "source": example["source"], **info}


def collate(rows, pad_id):
    width = max(len(r["input_ids"]) for r in rows)
    kmax = max(len(r["code_ids"]) for r in rows)
    input_ids = torch.tensor([r["input_ids"] + [pad_id] * (width - len(r["input_ids"])) for r in rows], device="cuda")
    mask = torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows], device="cuda")
    codes = torch.tensor([r["code_ids"] + [r["code_ids"][0]] * (kmax - len(r["code_ids"])) for r in rows], device="cuda")
    code_mask = torch.tensor([[True] * len(r["code_ids"]) + [False] * (kmax - len(r["code_ids"])) for r in rows], device="cuda")
    targets = torch.tensor([r["target"] + [0.] * (kmax - len(r["target"])) for r in rows], device="cuda")
    return input_ids, mask, codes, code_mask, targets


def set_checkpointing(model, enabled: bool) -> None:
    """Turn activation checkpointing on or off for the next forward (the decoder layers read the flag per call)."""
    for layer in model.model.language_model.layers:
        layer.gradient_checkpointing = enabled


def logits_for(model, rows, pad_id, packed=False):
    if packed:  # built on the CPU and sent in one copy (modeling.pack_rows): no host syncs in the forward
        batch = modeling.pack_rows([r["input_ids"] for r in rows], [r["code_ids"] for r in rows], "cuda")
        kmax = batch["columns"].shape[1]
        targets = torch.tensor([r["target"] + [0.] * (kmax - len(r["target"])) for r in rows]).pin_memory()
        return modeling.packed_code_logits(model, batch), targets.to("cuda", non_blocking=True)
    input_ids, mask, codes, code_mask, targets = collate(rows, pad_id)
    logits = modeling.code_logits(model, input_ids, mask, codes, code_mask)
    return logits, targets


def soft_ce(logits, targets):
    """-sum_i q_i log p_i per row; equals KL(q||p) plus the teacher entropy."""
    return -(targets * logits.log_softmax(-1)).sum(-1)


def entropy(targets):
    return -(targets * targets.clamp_min(1e-12).log()).sum(-1)


def microbatches(rows, args):
    """Length-sorted groups under --token-budget: real tokens when packed, else padded width x rows (at most --micro-batch rows)."""
    group = []
    for r in sorted(rows, key=lambda x: len(x["input_ids"])):
        longest = max([len(g["input_ids"]) for g in group] + [len(r["input_ids"])])
        cost = (sum(len(g["input_ids"]) for g in group) + len(r["input_ids"]) if args.pack
                else longest * (len(group) + 1))
        if group and ((not args.pack and len(group) >= args.micro_batch) or cost > args.token_budget):
            yield group
            group = []
        group.append(r)
    if group:
        yield group


@torch.no_grad()
def evaluate(model, renderer, examples, args, pad_id):
    """Overall and by kind/source: agreement with the target's argmax, accuracy, and KL(target||model).

    accuracy counts a prediction right when its option has the target's top probability, so every
    tied gold of a multi-gold row is right; "macro_source_accuracy" averages it over sources (each
    source weighs the same, as benchmarks do on the board) and is what gold-dev model selection reads.
    """
    model.eval()
    groups = defaultdict(lambda: {"n": 0, "agree": 0, "correct": 0, "kl_sum": 0.})
    encoded = [renderer(e) for e in examples]
    for batch in microbatches(encoded, args):
        set_checkpointing(model, False)
        logits, targets = logits_for(model, batch, pad_id, args.pack)
        kl = (soft_ce(logits, targets) - entropy(targets)).tolist()
        pred = logits.argmax(-1)
        agree = (pred == targets.argmax(-1)).tolist()
        correct = (targets.gather(1, pred[:, None])[:, 0] >= targets.max(-1).values - 1e-6).tolist()
        for r, k, a, c in zip(batch, kl, agree, correct):
            for name in ("all", f"kind:{r['kind']}", f"source:{r['source']}"):
                g = groups[name]
                g["n"] += 1
                g["agree"] += int(a)
                g["correct"] += int(c)
                g["kl_sum"] += k
    model.train()
    out = {name: {"n": g["n"], "agreement": g["agree"] / g["n"], "accuracy": g["correct"] / g["n"], "kl": g["kl_sum"] / g["n"]}
           for name, g in sorted(groups.items())}
    sources = [v["accuracy"] for name, v in out.items() if name.startswith("source:")]
    out["macro_source_accuracy"] = sum(sources) / len(sources)
    return out


class MasterAdamW:
    """AdamW on FP32 master copies of the BF16 model's trainable parameters.

    The model computes in BF16 with no autocast: FP32 weights under autocast kept the residual stream in
    FP32 and re-cast activations at every layer, ~24% of a step's GPU time (torch profiler). An update at
    lr 1e-5 is far below BF16 resolution, so the masters accumulate it and the BF16 weights are rewritten
    from them after each step. Zero-initialized DeltaNet gates get their own larger rate (lr_scale) and no decay.

    anchor > 0 is L2-SP (Li et al. 2018, "Explicit inductive bias for transfer learning") in decoupled form:
    instead of AdamW's weight decay toward 0, every step moves each master lr * anchor of the way back to its
    starting (base) value, so a parameter forgets its own drift with a horizon of about 1 / (lr * anchor)
    steps (anchor 100 at lr 1e-5: ~1,000 steps). A penalty on ||theta - theta_0||^2 added to the loss would
    instead be rescaled per coordinate by Adam. The base copy costs 4 bytes per trainable parameter (~3 GB).
    """

    def __init__(self, model, lr, gate_lr=None, foreach=False, anchor=0.):
        self.pairs = [(p, p.detach().float().clone()) for p in model.parameters() if p.requires_grad]
        self.anchor = anchor
        self.foreach = foreach
        # foreach=True keeps persistent FP32 gradient buffers and does the BF16 -> FP32 gradient copy and the
        # FP32 -> BF16 write-back as single multi-tensor launches. Measured (bench_train, 2026-10-17): the
        # optimizer phase stayed 85 ms of a 2.3 s step while peak memory rose 2.8 GB, so it is off by default.
        self.buffers = [torch.zeros_like(m) for _, m in self.pairs] if foreach else None
        master = {id(p): m for p, m in self.pairs}
        decay, no_decay, gates = [], [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (gates if name.endswith("backward_gate") else decay if p.ndim >= 2 else no_decay).append(master[id(p)])
        groups = [{"params": decay, "weight_decay": 0. if anchor else .01, "lr_scale": 1.},
                  {"params": no_decay, "weight_decay": 0., "lr_scale": 1.}]
        if gates:
            groups.append({"params": gates, "weight_decay": 0., "lr_scale": (gate_lr or lr) / lr})
        self.optimizer = torch.optim.AdamW(groups, lr=lr, betas=(.9, .95), eps=1e-8, fused=True)
        self.param_groups = self.optimizer.param_groups
        # Base values of the anchored masters (the gates start at 0 and are not anchored).
        self.anchored = ([(group["params"], [m.clone() for m in group["params"]]) for group in self.param_groups[:2]]
                         if anchor else [])

    def zero_grad(self, set_to_none=True):
        for p, m in self.pairs:
            p.grad = m.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> float:
        """Move the BF16 gradients to the masters and clip them there; returns the total norm."""
        if self.foreach and all(p.grad is not None for p, _ in self.pairs):
            torch._foreach_copy_(self.buffers, [p.grad for p, _ in self.pairs])
            for (p, m), buffer in zip(self.pairs, self.buffers):
                m.grad, p.grad = buffer, None
        else:  # a parameter without a gradient keeps grad None (no decay, no moment update), as before
            for p, m in self.pairs:
                m.grad = None if p.grad is None else p.grad.float()
                p.grad = None
        return float(torch.nn.utils.clip_grad_norm_([m for _, m in self.pairs], max_norm, foreach=True))

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        if self.anchor:
            weight = self.param_groups[0]["lr"] * self.anchor
            assert weight < 1, f"lr * anchor = {weight} must stay below 1"
            for masters, base in self.anchored:
                torch._foreach_lerp_(masters, base, weight)
        if self.foreach:
            torch._foreach_copy_([p for p, _ in self.pairs], [m for _, m in self.pairs])
        else:
            for p, m in self.pairs:
                p.copy_(m)


def language_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.model.language_model.state_dict().items()}


def preflight(model, renderer, train, args, pad_id, out: Path):
    """Rendering coverage, readout and attention-mode checks, and three real optimizer steps."""
    rng = random.Random(123)
    draws = [renderer(e, random.Random(i)) for i, e in enumerate(rng.sample(train, 20000))]
    targeted = Counter(d["hard_code"] for d in draws)
    stats = {"option_count_max": max(len(d["code_ids"]) for d in draws),
             "padded_share": sum(d["padded"] for d in draws) / len(draws),
             "noul_as_choice_share": sum(d["noul_as_choice"] for d in draws) / len(draws),
             "code_assignment": dict(Counter(d["code_assignment"] for d in draws)),
             "codes_never_targeted": [c for c in renderer.codes if c not in targeted],
             "first_never_targeted_position": next((i + 1 for i, c in enumerate(renderer.codes) if c not in targeted), None),
             "tokens_p50_p99_max": [int(np.percentile([len(d["input_ids"]) for d in draws], q)) for q in (50, 99, 100)]}
    model.eval()
    sample = draws[:4]
    with torch.no_grad():
        logits, _ = logits_for(model, sample, pad_id)
        input_ids, mask, codes, code_mask, _ = collate(sample, pad_id)
        hidden = model.model.language_model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
        full = model.lm_head(hidden[torch.arange(4, device="cuda"), modeling.last_positions(mask)]).float()
        expected = full.gather(1, codes).masked_fill(~code_mask, modeling.MASK)
        readout_max_difference = (logits - expected)[code_mask].abs().max().item()
        # A later token must change an earlier hidden state only in noncausal mode.
        ids = input_ids[:1, :int(mask[0].sum())]
        changed = ids.clone()
        changed[0, -2] = (changed[0, -2] + 1) % model.config.text_config.vocab_size
        a = model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[0, 5]
        b = model.model.language_model(input_ids=changed, use_cache=False).last_hidden_state[0, 5]
        future_visible = bool((a - b).abs().max().item() > 0)
    assert readout_max_difference < .1, readout_max_difference
    assert future_visible == (args.attention == "noncausal"), (future_visible, args.attention)
    model.train()
    optimizer = MasterAdamW(model, args.lr, args.gate_lr, anchor=args.anchor)  # as in main, anchor copy included
    tracked = model.model.language_model.layers[-1].mlp.down_proj.weight
    before = tracked.detach().clone()
    steps, losses = [], []
    batch = draws[4:4 + args.effective_batch]
    for _ in range(3):
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for group in microbatches(batch, args):
            set_checkpointing(model, sum(len(r["input_ids"]) for r in group) > args.checkpoint_above)
            logits, targets = logits_for(model, group, pad_id, args.pack)
            loss = soft_ce(logits, targets).sum() / len(batch)
            assert torch.isfinite(loss)
            loss.backward()
            total += loss.item()
        optimizer.clip_grad_norm(1.)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(total)
        steps.append(time.time() - tick)
    assert not torch.equal(before, tracked)
    # The longest prompt of the whole training set (not just the 20,000 draws): render the 30 longest by characters.
    by_chars = sorted(train, key=lambda e: len(json.dumps(e["state"], ensure_ascii=False)) + len(json.dumps(e["question"])))[-30:]
    longest = max([renderer(e, random.Random(0)) for e in by_chars] + draws, key=lambda d: len(d["input_ids"]))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    set_checkpointing(model, len(longest["input_ids"]) > args.checkpoint_above)
    logits, targets = logits_for(model, [longest], pad_id, args.pack)
    soft_ce(logits, targets).sum().backward()
    result = {"render": stats, "readout_max_logit_difference": readout_max_difference,
              "future_token_changes_earlier_state": future_visible, "losses_3_steps": losses,
              "step_seconds": steps, "longest_tokens": len(longest["input_ids"]),
              "longest_peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
              # Every code up to 160 options must be a target in 20,000 draws (the public suite tops out at 151).
              "passed": (stats["first_never_targeted_position"] or 999) > 160}
    atomic_json(out / "preflight.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("preflight failed: some answer codes up to 160 options are never a target")


def gold_targets(rows: list[dict]) -> list[dict]:
    """The training/eval target of each row from its gold (mix.annotate fields), never from a model's spread.

    single: one-hot on the gold. valid_set (several correct options): even over them, and any of them
    counts as right. vote_distribution (annotators disagreed): even over the plurality option(s); the
    minority votes are not a target.
    """
    from qwen_decision.v3.mix import gold_set

    out = []
    for r in rows:
        if "label_origin" not in r or "target_kind" not in r:
            raise ValueError(f"{r['id']}: rows need mix.py's label_origin/target_kind")
        golds = gold_set(r)
        out.append({**r, "target": [1 / len(golds) if i in golds else 0. for i in range(len(r["target"]))]})
    return out


def batches(n_rows: int, size: int, steps: int, rng: random.Random):
    """`steps` batches of row indices; the order is reshuffled at every pass, so a small set is repeated."""
    order, at, epoch = [], 0, -1
    for _ in range(steps):
        batch = []
        while len(batch) < size:
            if at == len(order):
                order, at, epoch = list(range(n_rows)), 0, epoch + 1
                rng.shuffle(order)
            take = min(size - len(batch), len(order) - at)
            batch += order[at:at + take]
            at += take
        yield batch, epoch


def main():
    from qwen_decision.v3.mix import CORPUS_EXCLUDED, allowed_origin

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=ROOT / "models/qwen3.5-0.8b")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attention", choices=modeling.ATTENTION_MODES, default="noncausal")
    p.add_argument("--layout", choices=fmt.LAYOUTS, default="d3")
    p.add_argument("--deltanet", choices=modeling.DELTANET_MODES, default="causal")
    p.add_argument("--kernel", choices=("sdpa", "varlen"), default="varlen", help="noncausal attention kernel")
    p.add_argument("--no-pack", dest="pack", action="store_false",
                   help="pad microbatch rows instead of packing them (packing needs noncausal, varlen, causal DeltaNet)")
    p.add_argument("--gate-lr", type=float, default=1e-3, help="learning rate of the zero-initialized DeltaNet gates")
    p.add_argument("--mix", type=Path, required=True, help="a data/mix/<name>.jsonl from mix.py (gold labels only)")
    p.add_argument("--train-rows", type=int, default=0, help="read the first N rows of --mix (0 = all); prefixes are proportional")
    p.add_argument("--epochs", type=int, default=1,
                   help="passes over the rows when --steps is 0; with more than one, the model after every pass is "
                        "saved as epoch-<k>/ (scored on the gold dev set at that step) next to best/ and last/")
    p.add_argument("--steps", type=int, default=0,
                   help="optimizer updates; 0 = one pass over the rows. More steps than one pass repeat the rows, "
                        "reshuffled each pass, so data size and update count can be varied separately")
    p.add_argument("--label-origins", default="human,programmatic",
                   help="keep rows whose label_origin (mix.py) is listed; model-made golds only for mix.MODEL_ORIGIN_ALLOWED")
    p.add_argument("--gold-dev", type=Path, help="gold dev set for model selection (default: <mix>.dev.jsonl)")
    p.add_argument("--gold-dev-per-source", type=int, default=0,
                   help="0 = the whole gold dev set; N = a fixed random sample (sha256 of the id) of N rows per source")
    p.add_argument("--train-probe-per-source", type=int, default=25,
                   help="fixed training rows per source (sha256 of the id) scored unaugmented at each eval: "
                        "separates not-learned (wrong on seen rows) from not-generalized (right on seen, wrong on dev)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--anchor", type=float, default=0.,
                   help="L2-SP toward the base weights (MasterAdamW): each step moves the weights lr*anchor of the way "
                        "back; drift horizon ~1/(lr*anchor) steps. 0 = off (AdamW weight decay 0.01 toward 0)")
    p.add_argument("--effective-batch", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=16, help="row cap of an unpacked microbatch")
    p.add_argument("--token-budget", type=int, default=16384, help="tokens per microbatch (padded width x rows unpacked)")
    p.add_argument("--checkpoint-above", type=int, default=16384,
                   help="activation checkpointing only for microbatches with more real tokens than this")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="leave the element-wise blocks uncompiled (modeling.compile_elementwise)")
    p.add_argument("--max-tokens", type=int, default=65536, help="skip (never truncate) longer training prompts")
    p.add_argument("--pad-probability", type=float, default=.35)
    p.add_argument("--noul-choice-probability", type=float, default=.2)
    p.add_argument("--code-shuffle-probability", type=float, default=.2,
                   help="unordered choice rows whose first k codes are shuffled over the displayed options")
    p.add_argument("--code-sample-probability", type=float, default=.2,
                   help="unordered choice rows shown k codes drawn from all 255, in code order (rest: A, B, C, ...)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    if args.pack and (args.attention, args.kernel, args.deltanet) != ("noncausal", "varlen", "causal"):
        p.error("packing needs --attention noncausal --kernel varlen --deltanet causal; pass --no-pack")
    args.gold_dev = args.gold_dev or args.mix.with_suffix(".dev.jsonl")
    if not args.gold_dev.exists():
        p.error(f"no gold dev set at {args.gold_dev}: model selection needs one (mix.py writes it)")
    out = args.output
    if not args.preflight:
        assert not (out / "manifest.json").exists(), "Preserve existing runs; use a new output directory."
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    pad_id = tokenizer.pad_token_id
    with args.mix.open() as f:  # prefixes of a mix file are proportional subsets (mix.py)
        rows = [json.loads(line) for i, line in enumerate(f) if not args.train_rows or i < args.train_rows]
    allowed = set(args.label_origins.split(","))
    teacher = [r["id"] for r in rows if r["source"] in CORPUS_EXCLUDED]
    if teacher:
        raise SystemExit(f"{len(teacher)} rows from excluded corpus sources in {args.mix} (e.g. {teacher[0]})")
    filtered = Counter(r["label_origin"] for r in rows if not allowed_origin(r, allowed))
    train = gold_targets([r for r in rows if allowed_origin(r, allowed)])
    gold_dev = [r for r in map(json.loads, args.gold_dev.open()) if allowed_origin(r, allowed)]
    if args.gold_dev_per_source:  # a fixed sample, not the file's first rows
        by = defaultdict(list)
        for r in sorted(gold_dev, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
            by[r["source"]].append(r)
        gold_dev = [r for rows_ in by.values() for r in rows_[:args.gold_dev_per_source]]
    per = Counter(r["source"] for r in gold_dev)
    gold_dev = gold_targets(gold_dev)
    by = defaultdict(list)
    for r in sorted(train, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        by[r["source"]].append(r)
    train_probe = [r for rows_ in by.values() for r in rows_[:args.train_probe_per_source]]
    pool = sorted({o for e in train if e["question"]["type"] == "choice" and not e["ordered"]
                   for o in e["question"]["criteria"]})
    renderer = Renderer(tokenizer, args, pool)
    model = model_load(args)
    if args.preflight:
        preflight(model, renderer, train, args, pad_id, out)
        return
    per_epoch = math.ceil(len(train) / args.effective_batch)
    steps = args.steps or args.epochs * per_epoch
    mix_manifest = args.mix.with_suffix(".manifest.json")
    manifest = {"base": str(args.base), "attention": args.attention, "layout": args.layout, "deltanet": args.deltanet,
                "training": "full language backbone and tied embedding/LM head; vision frozen; BF16 compute, FP32 master weights and moments",
                "loss": "cross-entropy over the shown answer codes against gold targets (gold_targets); no teacher model",
                "data": {"mix": str(args.mix), "mix_sha256": sha256_file(args.mix), "rows_read": len(rows),
                         "train_rows": len(train), "label_origin_filtered_out": dict(filtered),
                         "train_sources": dict(Counter(e["source"] for e in train)),
                         "label_origin": dict(Counter(r["label_origin"] for r in train)),
                         "target_kind": dict(Counter(r["target_kind"] for r in train)),
                         "gold_dev": str(args.gold_dev), "gold_dev_sha256": sha256_file(args.gold_dev),
                         "gold_dev_rows": len(gold_dev), "gold_dev_sources": len(per), "train_probe_rows": len(train_probe),
                         "mix_manifest": json.loads(mix_manifest.read_text()) if mix_manifest.exists() else None},
                "steps": steps, "steps_per_epoch": per_epoch, "passes": steps * args.effective_batch / len(train),
                "versions": versions(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("format.py", "modeling.py", "train.py", "mix.py")},
                "gpu": torch.cuda.get_device_name(), "started_unix": time.time()}
    atomic_json(out / "manifest.json", manifest)
    config = {"attention": args.attention, "layout": args.layout, "deltanet": args.deltanet,
              "codes": renderer.codes, "token_ids": renderer.code_ids, "trained_from": str(args.base)}
    rng = random.Random(args.seed)
    optimizer = MasterAdamW(model, args.lr, args.gate_lr, anchor=args.anchor)

    def run_eval():
        result = {"gold": evaluate(model, renderer, gold_dev, args, pad_id)}
        if train_probe:
            result["train_probe"] = evaluate(model, renderer, train_probe, args, pad_id)
        return result

    def selection(result) -> float:
        """Gold-dev accuracy averaged over sources (each source weighs the same)."""
        return result["gold"]["macro_source_accuracy"]

    history = {"step-0": run_eval()}
    atomic_json(out / "eval.json", history)
    best, best_step = selection(history["step-0"]), 0
    save_file(language_state(model), str(out / "best-language.safetensors"))
    start, skipped = time.time(), 0
    seen = Counter()  # rows, tokens, updates, passes
    coverage = Counter()
    warmup = max(1, math.ceil(steps * .05))
    step = 0  # optimizer updates done; batches with no trainable row do not count
    with (out / "metrics.jsonl").open("w", buffering=1) as log:
        for indices, epoch in batches(len(train), args.effective_batch, 10 * steps + 10, rng):
            if step == steps:
                break
            began = time.time()
            encoded = []
            for i in indices:
                e = renderer(train[i], rng)
                if len(e["input_ids"]) > args.max_tokens:
                    skipped += 1
                    continue
                encoded.append(e)
                coverage[f"k:{min(len(e['code_ids']), 999) // 32 * 32}"] += 1
                coverage[f"kind:{e['kind']}"] += 1
            if not encoded:
                seen.update(empty_batches=1)
                continue
            step += 1
            factor = step / warmup if step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor * group["lr_scale"]
            optimizer.zero_grad(set_to_none=True)
            tick = time.time()  # rendering done: prep_seconds = tick - began
            sums = torch.zeros(2, device="cuda")  # loss, accuracy; read once per step (no per-microbatch sync)
            for group in microbatches(encoded, args):
                # Checkpoint only microbatches above --checkpoint-above tokens: recomputation cost ~25% of a step.
                set_checkpointing(model, sum(len(r["input_ids"]) for r in group) > args.checkpoint_above)
                logits, targets = logits_for(model, group, pad_id, args.pack)
                ce = soft_ce(logits, targets)
                (ce.sum() / len(encoded)).backward()
                right = targets.gather(1, logits.argmax(-1, keepdim=True))[:, 0] > 0
                sums += torch.stack([ce.sum(), right.sum()]).detach().float()
            loss_sum, right = sums.tolist()
            if not math.isfinite(loss_sum):
                raise RuntimeError(f"nonfinite loss at step {step}")
            grad_norm = optimizer.clip_grad_norm(1.)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite gradient norm at step {step}")
            optimizer.step()
            torch.cuda.synchronize()  # step_seconds is device time of the whole step, rendering included
            seen.update(rows=len(encoded), tokens=sum(len(e["input_ids"]) for e in encoded), updates=1)
            metric = {"step": step, "steps": steps, "pass": epoch, "loss": loss_sum / len(encoded),
                      "batch_accuracy": right / len(encoded), "lr": optimizer.param_groups[0]["lr"],
                      "grad_norm": grad_norm, "step_seconds": time.time() - began, "prep_seconds": tick - began,
                      "elapsed": time.time() - start,
                      "rows_seen": seen["rows"], "tokens_seen": seen["tokens"], "skipped_over_max_tokens": skipped}
            gates = modeling.gate_state(model.model.language_model)
            if gates:
                metric["gate_abs_mean"] = float(torch.stack([g.abs().mean() for g in gates.values()]).mean())
            log.write(json.dumps(metric) + "\n")
            if step <= 5 or step % 50 == 0:
                print(f"step {step}/{steps} pass {epoch} loss={metric['loss']:.4f} acc={metric['batch_accuracy']:.3f} "
                      f"s/step={metric['step_seconds']:.2f}", flush=True)
            epoch_end = args.epochs > 1 and not args.steps and step % per_epoch == 0
            if step % args.eval_every == 0 or step == steps or epoch_end:
                result = run_eval()
                history[f"step-{step}"] = result
                atomic_json(out / "eval.json", history)
                atomic_json(out / "coverage.json", dict(coverage))
                print(f"eval step {step}: gold macro={selection(result):.4f} all={result['gold']['all']['accuracy']:.4f}", flush=True)
                if selection(result) > best:
                    best, best_step = selection(result), step
                    save_file(language_state(model), str(out / "best-language.safetensors"))
                    atomic_json(out / "best.json", {"step": step, **result})
            if epoch_end and step < steps:  # the last epoch is last/ (linked below)
                modeling.save(model, tokenizer, out / f"epoch-{step // per_epoch}",
                              {**config, "steps": steps, "step": step, "epoch": step // per_epoch})
                print(f"saved epoch-{step // per_epoch} at step {step}", flush=True)
    if step != steps:
        raise RuntimeError(f"only {step} of {steps} updates: too many rows over --max-tokens")
    # last/ is the final step; best/ the selected step (step 0 = the base model won); final -> best for the eval scripts.
    modeling.save(model, tokenizer, out / "last", {**config, "steps": steps, "step": steps})
    if args.epochs > 1 and not args.steps:
        (out / f"epoch-{args.epochs}").symlink_to("last")
    if best_step != steps:
        missing, unexpected = model.model.language_model.load_state_dict(
            load_file(str(out / "best-language.safetensors")), strict=False)
        assert not unexpected and not missing, (missing[:3], unexpected[:3])
    modeling.save(model, tokenizer, out / "best", {**config, "steps": steps, "step": best_step,
                                                    "selection": "gold dev macro source accuracy"})
    (out / "best-language.safetensors").unlink()
    if (out / "final").is_symlink() or (out / "final").exists():
        (out / "final").unlink()
    (out / "final").symlink_to("best")
    atomic_json(out / "complete.json", {"complete": True, "steps": steps, "elapsed": time.time() - start,
                                        "rows_seen": seen["rows"], "tokens_seen": seen["tokens"], "updates": seen["updates"],
                                        "empty_batches_skipped": seen["empty_batches"],
                                        "passes": seen["rows"] / len(train), "skipped_over_max_tokens": skipped,
                                        "best_step": best_step, "best_selection": best,
                                        "selection": "gold dev macro source accuracy",
                                        "last": history[f"step-{steps}"], "best": history[f"step-{best_step}"]})
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
