"""Weight-space merges of v3 checkpoints with the base model, without training.

  # WiSE-FT (Wortsman et al.): every tensor keeps the same share of the update
  python -m qwen_decision.v3.merge --tuned runs/v3/<run>/final --alpha 0.5 --out runs/v3/<name>
  # LiNeS (Wang et al., ICLR 2025): the update of decoder layer l (1..L) is scaled by alpha + beta*(l-1)/(L-1)
  python -m qwen_decision.v3.merge --tuned runs/v3/<run>/final --alpha 0.3 --beta 0.4 --out runs/v3/<name>
  # checkpoint average or soup: the updates of several checkpoints of the same base, weighted
  python -m qwen_decision.v3.merge --tuned runs/v3/<run>/epoch-2 runs/v3/<run>/epoch-3 --alpha 1 --out ...
  # Model Stock (Jang et al., ECCV 2024): the share of the averaged update set per tensor by the angle
  python -m qwen_decision.v3.merge --tuned runs/v3/<a>/final runs/v3/<b>/final --alpha 1 --stock --out ...
  # TIES-Merging (Yadav et al., NeurIPS 2023): trim each update to its largest --ties share, elect a sign per
  # entry, average only the updates that agree with it
  python -m qwen_decision.v3.merge --tuned <a> <b> <c> --alpha 1 --ties 0.2 --out ...
  # DARE (Yu et al., ICML 2024): drop each update entry with probability --dare and rescale the rest by 1/(1-p);
  # with --ties as well it is DARE-TIES (drop, then elect signs and average the agreeing entries)
  python -m qwen_decision.v3.merge --tuned <a> <b> --alpha 1 --dare 0.7 [--ties 1] --out ...

theta = base + sum_i w_i * lambda(tensor) * (theta_i - base), computed in FP32 and stored in the tuned dtype.
lambda is alpha for every tensor outside the decoder layers. In LiNeS's pseudocode, the parameters outside
the residual blocks use alpha. Here that includes the tied input embedding, which is also the code readout.
Use --outside to give them their own share. beta < 0 keeps more of the shallow layers than the deep ones,
the profile d3-lite's own drift has (1.6-2.4% below layer 16, ~1% above). Vision tensors are frozen in
training, so they equal the base and are unchanged by any merge. --alpha 0.5 alone reproduces interpolate.py
exactly (checked 2026-10-17).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import torch
from safetensors.torch import save_file

from qwen_decision.v3.interpolate import tensors

ROOT = Path(__file__).resolve().parents[2]
LAYER = re.compile(r"language_model\.layers\.(\d+)\.")


def ties(deltas: list[torch.Tensor], weights: list[float], keep: float) -> torch.Tensor:
    """TIES-Merging for one tensor: trim, elect sign, disjoint (weighted) mean."""
    trimmed = []
    for d in deltas:
        if keep < 1 and d.numel() > 1:
            k = max(1, int(round(d.numel() * (1 - keep))))  # entries at or below the k-th smallest |d| are dropped
            threshold = d.abs().flatten().kthvalue(k).values
            d = torch.where(d.abs() > threshold, d, torch.zeros_like(d))
        trimmed.append(d)
    elected = torch.sign(sum(w * d for w, d in zip(weights, trimmed)))
    numerator = torch.zeros_like(trimmed[0])
    denominator = torch.zeros_like(trimmed[0])
    for w, d in zip(weights, trimmed):
        agree = (torch.sign(d) == elected) & (d != 0)
        numerator += torch.where(agree, w * d, torch.zeros_like(d))
        denominator += agree * w
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), torch.zeros_like(numerator))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=ROOT / "models/qwen3.5-0.8b")
    p.add_argument("--tuned", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", help="per --tuned checkpoint (default: equal, summing to 1)")
    p.add_argument("--alpha", type=float, required=True, help="share of the update kept (first decoder layer, and outside layers)")
    p.add_argument("--beta", type=float, default=0., help="LiNeS: added share at the last decoder layer (0 = WiSE-FT)")
    p.add_argument("--outside", type=float, help="share for tensors outside the decoder layers (default: alpha)")
    p.add_argument("--stock", action="store_true",
                   help="Model Stock (Jang et al., ECCV 2024): per tensor, share t = N cos / (1 + (N-1) cos) of the "
                        "averaged update, cos = mean pairwise cosine of the checkpoints' updates (needs 2+ --tuned; "
                        "--alpha/--beta then multiply t, so use --alpha 1)")
    p.add_argument("--ties", type=float, help="TIES: keep this share of each update's entries (largest |delta| per tensor), "
                                              "elect signs, disjoint weighted mean; 1 = no trimming")
    p.add_argument("--dare", type=float, help="DARE: drop probability of each update entry (rescale the rest by 1/(1-p))")
    p.add_argument("--seed", type=int, default=20261019, help="DARE drop masks")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.stock and len(args.tuned) < 2:
        raise SystemExit("--stock needs at least two --tuned checkpoints of the same base")
    weights = args.weights or [1 / len(args.tuned)] * len(args.tuned)
    if len(weights) != len(args.tuned):
        raise SystemExit("one --weights value per --tuned checkpoint")
    base = tensors(args.base)
    tuned = [tensors(t) for t in args.tuned]
    keys = set(tuned[0])
    for t, path in zip(tuned, args.tuned):
        if set(t) != keys:
            raise SystemExit(f"{path} has other tensors than {args.tuned[0]}")
    missing = sorted(keys - set(base))
    if missing:
        raise SystemExit(f"{len(missing)} tuned tensors have no base counterpart, e.g. {missing[:3]}")
    depth = 1 + max(int(m.group(1)) for k in keys if (m := LAYER.search(k)))
    outside = args.alpha if args.outside is None else args.outside

    def share(key: str) -> float:
        m = LAYER.search(key)
        if not m:
            return outside
        return args.alpha + args.beta * int(m.group(1)) / max(1, depth - 1)

    merged, stock = {}, {}
    generator = torch.Generator().manual_seed(args.seed)
    for k in sorted(keys):
        b = base[k].to(torch.float32)
        deltas = [t[k].to(torch.float32) - b for t in tuned]
        if args.dare:
            deltas = [d * (torch.rand(d.shape, generator=generator) >= args.dare) / (1 - args.dare) for d in deltas]
        if args.ties is not None:
            update = ties(deltas, weights, args.ties)
        else:
            update = sum(w * d for w, d in zip(weights, deltas))
        scale = share(k)
        if args.stock:
            pairs = [(i, j) for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
            norms = [d.norm() for d in deltas]
            if all(n > 0 for n in norms):
                cos = sum(float((deltas[i] * deltas[j]).sum() / (norms[i] * norms[j])) for i, j in pairs) / len(pairs)
                cos = min(max(cos, 0.), 1.)
                n = len(deltas)
                t = n * cos / (1 + (n - 1) * cos)
                stock[k] = round(t, 4)
                scale *= t
        merged[k] = (b + scale * update).to(tuned[0][k].dtype).contiguous()
    shutil.copytree(args.tuned[0], args.out, ignore=shutil.ignore_patterns("*.safetensors"))
    save_file(merged, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    recipe = {"base": str(args.base), "tuned": [str(t) for t in args.tuned], "weights": weights, "alpha": args.alpha,
              "beta": args.beta, "outside": outside, "decoder_layers": depth,
              "layer_shares": [round(args.alpha + args.beta * i / max(1, depth - 1), 4) for i in range(depth)],
              "rule": "theta = base + sum_i w_i * share * (theta_i - base); share = alpha + beta*(l-1)/(L-1) per decoder layer"
                      + ("; times Model Stock t per tensor" if args.stock else "")
                      + (f"; DARE drop p={args.dare} (seed {args.seed}), rescaled by 1/(1-p)" if args.dare else "")
                      + (f"; TIES: top {args.ties} of |delta| per tensor, elected sign, weighted mean of agreeing entries"
                         if args.ties is not None else ""),
              "ties": args.ties, "dare": args.dare, "seed": args.seed if args.dare else None,
              "stock_t": stock or None}
    (args.out / "merge.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps({k: recipe[k] for k in ("alpha", "beta", "outside", "weights")}))


if __name__ == "__main__":
    main()
