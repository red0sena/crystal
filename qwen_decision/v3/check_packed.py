"""Regression checks for the fast training path (GPU): run before trusting a change to modeling/train.

  python -m qwen_decision.v3.check_packed [--mix data/mix/v3-main.jsonl]

Exits non-zero if any check fails. Each check compares the fast path with the reference path on the same
inputs in FP32 (where differences come only from BF16 attention inside varlen_attn and TF32 dots in FLA):
1. conv: FLA's causal conv (fla_causal_conv1d) vs transformers' PyTorch fallback, outputs and gradients.
2. compiled blocks: compile_elementwise RMSNorm / gated RMSNorm / SwiGLU vs eager, seeded identical inputs.
3. packed vs padded: code logits and gradients of a real microbatch, packed=True vs the padded varlen path.
4. independence: each row packed together with others vs the same row alone (no leakage across rows).
Thresholds come from the measurements of 2026-10-16/17 (wiki/project/v3-log.md) with a margin.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from qwen_decision.v3 import modeling
from qwen_decision.v3 import train as T

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/qwen3.5-0.8b"


def cos(a, b) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def rel(a, b) -> float:
    return ((a.float() - b.float()).abs().max() / a.float().abs().max().clamp_min(1e-12)).item()


def torch_causal_conv1d(hidden_states, weight, bias=None, activation=None):
    """transformers' PyTorch fallback for Qwen3.5's short convolution (modeling_qwen3_5.causal_conv1d_fn)."""
    seq_len = hidden_states.shape[-1]
    out = F.conv1d(hidden_states.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1,
                   groups=hidden_states.shape[1])[:, :, :seq_len]
    return (F.silu(out) if activation == "silu" else out).to(hidden_states.dtype)


def check_conv(results):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, 6144, 700, device="cuda", generator=g).requires_grad_(True)
    w = (torch.randn(6144, 4, device="cuda", generator=g) * .3).requires_grad_(True)
    b = (torch.randn(6144, device="cuda", generator=g) * .1).requires_grad_(True)
    up = torch.randn(2, 6144, 700, device="cuda", generator=g)
    outs = []
    for fn in (torch_causal_conv1d, modeling.fla_causal_conv1d):
        y = fn(x, w, b, activation="silu")
        (y * up).sum().backward()
        outs.append([y.detach(), x.grad.clone(), w.grad.clone()])
        x.grad = w.grad = b.grad = None
    worst = max(rel(a, c) for a, c in zip(*outs))
    results.append(("conv FLA vs PyTorch (FP32, rel max)", worst, worst < 1e-4))


def check_compiled(results):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    cfg = AutoConfig.from_pretrained(BASE).text_config
    eager = {n: getattr(qwen, n).forward for n in ("Qwen3_5RMSNorm", "Qwen3_5RMSNormGated", "Qwen3_5MLP")}
    torch.manual_seed(0)
    modules = {"rms": qwen.Qwen3_5RMSNorm(1024), "gated": qwen.Qwen3_5RMSNormGated(128), "mlp": qwen.Qwen3_5MLP(cfg, 3584)}
    with torch.no_grad():
        for m in modules.values():
            for p in m.parameters():
                p.normal_(0, .05)

    def run(name, m):
        m = copy.deepcopy(m).cuda()
        g = torch.Generator(device="cuda").manual_seed(1)
        dim = 128 if name == "gated" else 1024
        x = torch.randn(3001, dim, device="cuda", generator=g, requires_grad=True)
        z = torch.randn(3001, dim, device="cuda", generator=g)
        up = torch.randn(3001, dim, device="cuda", generator=g)
        y = m(x, z) if name == "gated" else m(x)
        (y * up).sum().backward()
        return [y.detach(), x.grad] + [p.grad for p in m.parameters()]

    for n, f in eager.items():
        getattr(qwen, n).forward = f
    qwen._v3_compiled = False
    reference = {k: run(k, m) for k, m in modules.items()}
    modeling.compile_elementwise()
    for k, m in modules.items():
        worst = max(rel(a, c) for a, c in zip(reference[k], run(k, m)))
        results.append((f"compiled {k} vs eager (FP32, rel max)", worst, worst < 1e-5))


def batch_rows(mix: Path, n: int):
    rows = []
    with mix.open() as f:
        for line in f:
            r = json.loads(line)
            if r["question"]["type"] == "choice" and len(json.dumps(r["state"])) < 8000:
                rows.append(r)
            if len(rows) >= 400:
                break
    random.Random(0).shuffle(rows)
    return rows[:n]


def check_packed(results, mix: Path):
    args = argparse.Namespace(layout="d3", pad_probability=.35, noul_choice_probability=.2,
                              code_shuffle_probability=.2, code_sample_probability=.2)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    pool = sorted({o for r in batch_rows(mix, 400) for o in r["question"]["criteria"]})
    renderer = T.Renderer(tokenizer, args, pool)
    rng = random.Random(0)
    encoded = sorted([renderer(r, rng) for r in batch_rows(mix, 12)], key=lambda e: len(e["input_ids"]))
    batch = T.collate(encoded, tokenizer.pad_token_id)
    model = modeling.load(BASE, attention="noncausal", dtype=torch.float32, kernel="varlen")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # FP32 activations
    model.train()
    weights = [model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight,
               model.model.language_model.layers[3].self_attn.q_proj.weight]
    out = {}
    for packed in (False, True):
        model.zero_grad()
        logits = modeling.code_logits(model, *batch[:4], packed=packed)
        T.soft_ce(logits, batch[4]).sum().backward()
        out[packed] = (logits.detach(), [w.grad.clone() for w in weights])
    model.zero_grad()  # the CPU-built pack (train.logits_for) must match the GPU-built one
    logits, targets = T.logits_for(model, encoded, tokenizer.pad_token_id, True)
    T.soft_ce(logits, targets).sum().backward()
    cpu_pack = (logits.detach(), [w.grad.clone() for w in weights])
    diff_cpu = (cpu_pack[0] - out[True][0])[out[True][0] > modeling.MASK / 2].abs().max().item()
    results.append(("CPU-built pack vs GPU-built pack logits (FP32, max abs)", diff_cpu, diff_cpu < 1e-3))
    real = out[False][0] > modeling.MASK / 2
    diff = (out[False][0] - out[True][0])[real].abs().max().item()
    agree = bool((out[False][0].argmax(-1) == out[True][0].argmax(-1)).all())
    grad = min(cos(a, b) for a, b in zip(out[False][1], out[True][1]))
    results.append(("packed vs padded logits (FP32, max abs)", diff, diff < .15 and agree))
    results.append(("packed vs padded gradients (FP32, min cosine)", grad, grad > .9995))
    model.eval()
    with torch.no_grad():
        # Leakage would make a row depend on its neighbours. Batch composition alone already moves logits a
        # little (BF16 attention inside varlen_attn, TF32 in FLA), so the packed path is compared with the
        # padded path, where rows cannot see each other by construction: same rows, same check.
        def spread(packed: bool) -> tuple[float, bool]:
            together = T.logits_for(model, encoded, tokenizer.pad_token_id, packed)[0]
            worst, agree = 0., True
            for i, e in enumerate(encoded):
                alone = T.logits_for(model, [e], tokenizer.pad_token_id, packed)[0][0]
                k = len(e["code_ids"])
                worst = max(worst, (together[i, :k] - alone[:k]).abs().max().item())
                agree &= bool(together[i, :k].argmax() == alone[:k].argmax())
            return worst, agree
        packed_worst, packed_agree = spread(True)
        padded_worst, _ = spread(False)
    results.append((f"row with others vs alone, packed (padded reference {padded_worst:.3f}; FP32 max abs, info)",
                    packed_worst, packed_agree))
    # Structural test, free of numerical noise: the gradient of one row's logits with respect to the input
    # embeddings of every other packed row must be exactly zero (no attention, conv or scan path between rows).
    model.train()
    captured = {}

    def keep(module, inputs, output):
        output.retain_grad()
        captured["embeddings"] = output
    hook = model.model.language_model.embed_tokens.register_forward_hook(keep)
    lengths = [len(e["input_ids"]) for e in encoded]
    starts = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
    worst_cross = 0.
    for i in (0, len(encoded) // 2, len(encoded) - 1):
        model.zero_grad()
        logits, _ = T.logits_for(model, encoded, tokenizer.pad_token_id, True)
        logits[i, :len(encoded[i]["code_ids"])].sum().backward()
        grad = captured["embeddings"].grad[0]
        outside = torch.cat([grad[:starts[i]], grad[starts[i + 1]:]])
        inside = grad[starts[i]:starts[i + 1]].abs().max().item()
        worst_cross = max(worst_cross, outside.abs().max().item())
        assert inside > 0, "no gradient reached the row's own tokens"
    hook.remove()
    results.append(("gradient of a row's logits w.r.t. other rows' embeddings (must be exactly 0)", worst_cross,
                    worst_cross == 0.))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", type=Path, default=ROOT / "data/mix/v3-gold.jsonl")
    args = p.parse_args()
    results = []
    check_conv(results)
    check_compiled(results)
    check_packed(results, args.mix)
    failed = 0
    for name, value, ok in results:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3g}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
