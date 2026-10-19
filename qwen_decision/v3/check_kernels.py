"""Check the Gated DeltaNet kernels on this GPU: FLA versus the PyTorch reference.

flash-linear-attention 0.5.2's backward was reported wrong on some GPU/Triton
combinations, so gradients are compared before training. Run one process per
path (the kernel is bound when transformers is imported), then compare:

  python -m qwen_decision.v3.check_kernels --path fla   --out runs/v3/kernels/fla-fp32.pt
  python -m qwen_decision.v3.check_kernels --path torch --out runs/v3/kernels/torch-fp32.pt
  python -m qwen_decision.v3.check_kernels --compare runs/v3/kernels/fla-fp32.pt runs/v3/kernels/torch-fp32.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACKED = ("layers.0.linear_attn.in_proj_qkv.weight", "layers.0.linear_attn.A_log", "layers.1.linear_attn.out_proj.weight",
           "layers.3.self_attn.q_proj.weight", "layers.12.linear_attn.in_proj_qkv.weight", "layers.23.mlp.down_proj.weight")


def run(path: str, out: Path, precision: str) -> None:
    if path == "torch":
        sys.modules["fla"] = None  # makes `import fla` fail, so transformers binds its PyTorch reference
    import torch
    from transformers import AutoTokenizer

    from qwen_decision.v3 import format as fmt
    from qwen_decision.v3 import modeling
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    bound = m.torch_chunk_gated_delta_rule
    implementation = getattr(bound, "__wrapped__", bound)
    torch.manual_seed(0)
    base = ROOT / "models/qwen3.5-0.8b"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = modeling.load(base, attention="noncausal", dtype=torch.float32)
    text = model.model.language_model
    for p in model.parameters():
        p.requires_grad_(False)
    for name in TRACKED:
        text.get_parameter(name).requires_grad_(True)
    codes, ids = fmt.answer_codes(tokenizer)
    paragraph = ("The quarterly audit found that the warehouse inventory system double-counted returned items, "
                 "while the shipping team reported delays caused by a vendor outage. ")
    rows = []
    for repeats, k in ((60, 4), (25, 12)):
        question = {"type": "choice", "instructions": "Which team should own the fix?",
                    "criteria": {f"team_{i}": f"Team number {i}" for i in range(k)}}
        rows.append((tokenizer.encode(fmt.render(tokenizer, paragraph * repeats, question, codes[:k]),
                                      add_special_tokens=False), ids[:k]))
    width = max(len(r) for r, _ in rows)
    kmax = max(len(c) for _, c in rows)
    input_ids = torch.tensor([r + [tokenizer.pad_token_id] * (width - len(r)) for r, _ in rows], device="cuda")
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r, _ in rows], device="cuda")
    code_ids = torch.tensor([c + [c[0]] * (kmax - len(c)) for _, c in rows], device="cuda")
    code_mask = torch.tensor([[True] * len(c) + [False] * (kmax - len(c)) for _, c in rows], device="cuda")
    targets = torch.zeros(len(rows), kmax, device="cuda")
    targets[0, 1], targets[1, 7] = 1., 1.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = modeling.code_logits(model, input_ids, mask, code_ids, code_mask)
    loss = -(targets * logits.log_softmax(-1)).sum()
    loss.backward()
    result = {"path": path, "precision": precision, "implementation": f"{implementation.__module__}.{implementation.__qualname__}",
              "fla_imported": sys.modules.get("fla") is not None, "tokens": [len(r) for r, _ in rows],
              "loss": loss.item(), "logits": logits.detach().cpu(),
              "grads": {name: text.get_parameter(name).grad.detach().float().cpu() for name in TRACKED}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("grads", "logits")}))


def compare(a_path: Path, b_path: Path) -> None:
    import torch

    a, b = torch.load(a_path), torch.load(b_path)
    report = {"paths": [a["path"], b["path"]], "precision": [a["precision"], b["precision"]],
              "fla_imported": [a["fla_imported"], b["fla_imported"]],
              "loss": [a["loss"], b["loss"]], "max_logit_difference": (a["logits"] - b["logits"]).abs().max().item(),
              "grads": {}}
    for name in TRACKED:
        x, y = a["grads"][name].flatten(), b["grads"][name].flatten()
        report["grads"][name] = {"cosine": torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
                                 "relative_error": ((x - y).norm() / y.norm().clamp_min(1e-30)).item()}
    worst = min(g["cosine"] for g in report["grads"].values())
    # Only an FP32 comparison isolates the kernel; BF16 differences include rounding of two different algorithms.
    report["passed"] = worst > .9999 and a["fla_imported"] != b["fla_imported"] and a["precision"] == b["precision"] == "fp32"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", choices=("fla", "torch"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.path, args.out, args.precision)
