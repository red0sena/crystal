"""WiSE-FT weight interpolation between the base model and a fine-tuned v3 checkpoint.

  python -m qwen_decision.v3.interpolate --tuned runs/v3/<run>/final --alpha 0.6 --out runs/v3/<run>-wise60

theta = (1 - alpha) * base + alpha * tuned for every tensor (Wortsman et al. 2022, "Robust fine-tuning of
zero-shot models"). Recovers base-model knowledge lost in fine-tuning at no training cost. The output keeps
the tuned checkpoint's files (config, tokenizer, v3_config.json) with merged weights.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]


def tensors(path: Path) -> dict[str, torch.Tensor]:
    state = {}
    for f in sorted(glob.glob(str(path / "*.safetensors"))):
        if Path(f).name not in ("backward_gates.safetensors", "readout.safetensors"):
            state.update(load_file(f))
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=ROOT / "models/qwen3.5-0.8b")
    p.add_argument("--tuned", type=Path, required=True)
    p.add_argument("--alpha", type=float, required=True, help="weight of the tuned checkpoint")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    base, tuned = tensors(args.base), tensors(args.tuned)
    missing = sorted(set(tuned) - set(base))
    if missing:
        raise SystemExit(f"{len(missing)} tuned tensors have no base counterpart, e.g. {missing[:3]}")
    merged = {}
    for k, v in tuned.items():
        b = base[k].to(torch.float32)
        merged[k] = ((1 - args.alpha) * b + args.alpha * v.to(torch.float32)).to(v.dtype).contiguous()
    shutil.copytree(args.tuned, args.out, ignore=shutil.ignore_patterns("*.safetensors"))
    for gates in glob.glob(str(args.tuned / "backward_gates.safetensors")):  # gates have no base value: scale toward 0
        save_file({k: v * args.alpha for k, v in load_file(gates).items()}, str(args.out / Path(gates).name))
    save_file(merged, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    (args.out / "interpolation.json").write_text(json.dumps(
        {"base": str(args.base), "tuned": str(args.tuned), "alpha": args.alpha, "tensors": len(merged)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
