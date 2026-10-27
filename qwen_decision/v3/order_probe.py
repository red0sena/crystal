"""Training-free bound on what option isolation could buy: score each choice question in 4 option orders.

Isolating the options from each other (no cross-option attention, positions reset per option) makes
a model order-invariant. Averaging a trained model's probabilities over option orders is a cheap
stand-in: its gain over a single order, and how often the chosen option changes with the order, bound
the accuracy that order invariance alone could add. It says nothing about other effects of isolation.

  PYTHONPATH=external/decision-index .venv/bin/python -m qwen_decision.v3.order_probe

Models: runs/long-context/echo-v1/native-seed{41,73,101}/model. Sets: selection (4,147), JEVal long (684)
and BABILong up to 16k (600). noul questions and options that refer to each other keep one order.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import json
from pathlib import Path
import statistics

import torch

from qwen_decision.v3 import architecture_search as search, engine, long_context_experiment as lce
from qwen_decision.v3 import robust_adaptation as views, train

OUT = Path("runs/long-context/order-probe-v1")
SEEDS = (41, 73, 101)
ORDERS = 4
BABILONG_MAX = {"babilong-0k", "babilong-1k", "babilong-2k", "babilong-4k", "babilong-8k", "babilong-16k"}


def sets() -> dict[str, list[dict]]:
    all_sets = lce.eval_sets()
    return {"selection": all_sets["selection"], "jeval-long": all_sets["jeval-long"],
            "babilong-16k": [r for r in all_sets["babilong"] if r["source"] in BABILONG_MAX]}


def orders_for(renderer, row) -> list[list[int]]:
    k = len(row["target"])
    fixed = row["ordered"] or renderer.positional(row) or row["question"]["type"] != "choice"
    return [list(range(k))] if fixed else engine.option_orders(k, ORDERS)


@torch.no_grad()
def score(model, renderer, rows) -> dict[str, list[list[float]]]:
    """Per row, one probability list per order, mapped back to the original option order."""
    model.eval()
    train.set_checkpointing(model, False)
    pad = renderer.tokenizer.pad_token_id
    encoded = []
    for r in rows:
        for j, order in enumerate(orders_for(renderer, r)):
            encoded.append({**views.view(renderer, r, order), "id": f"{r['id']}\t{j}"})
    probabilities = {}
    for group in train.microbatches(encoded, lce.EVAL_ARGS):
        z, _ = train.logits_for(model, group, pad, packed=True)
        for e, p in zip(group, z.float().softmax(-1).cpu()):
            shown = p[:len(e["target"])].tolist()
            original = [0.] * len(shown)
            for position, option in enumerate(e["order"]):
                original[option] = shown[position]
            probabilities[e["id"]] = original
    out = defaultdict(list)
    for e in encoded:
        row_id, j = e["id"].split("\t")
        out[row_id].append(probabilities[e["id"]])
    return out


def summarize(rows, scored) -> dict:
    def correct(r, p):
        pred = max(range(len(p)), key=p.__getitem__)
        return r["target"][pred] >= max(r["target"]) - 1e-6

    by = defaultdict(lambda: {"base": [], "single": [], "averaged": [], "flip": [], "permuted": 0})
    for r in rows:
        ps = scored[r["id"]]
        s = by[r["source"]]
        s["base"].append(correct(r, ps[0]))
        s["single"].append(statistics.mean(correct(r, p) for p in ps))
        s["averaged"].append(correct(r, [sum(x) / len(ps) for x in zip(*ps)]))
        if len(ps) > 1:
            s["permuted"] += 1
            s["flip"].append(len({max(range(len(p)), key=p.__getitem__) for p in ps}) > 1)
    macro = lambda key: 100 * statistics.mean(statistics.mean(v[key]) for v in by.values())
    flips = [f for v in by.values() for f in v["flip"]]
    return {"base_macro": macro("base"), "single_order_mean_macro": macro("single"), "averaged_macro": macro("averaged"),
            "averaged_minus_base": macro("averaged") - macro("base"),
            "permuted_items": sum(v["permuted"] for v in by.values()), "items": len(rows),
            "flip_rate_permuted": 100 * statistics.mean(flips) if flips else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    renderer = lce.renderers()[0]["native"]
    data = sets()
    result = {"orders": ORDERS, "seeds": {}}
    for seed in SEEDS:
        model = lce.load_model(Path(f"runs/long-context/echo-v1/native-seed{seed}/model"))
        result["seeds"][seed] = {}
        for name, rows in data.items():
            scored = score(model, renderer, rows)
            search.save_records(OUT / f"seed{seed}-{name}.jsonl", [{"id": k, "probabilities": v} for k, v in scored.items()])
            result["seeds"][seed][name] = summarize(rows, scored)
            print("PROBE", seed, name, json.dumps(result["seeds"][seed][name]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result["mean"] = {name: {key: statistics.mean(result["seeds"][s][name][key] for s in SEEDS)
                             for key in ("base_macro", "averaged_macro", "averaged_minus_base", "flip_rate_permuted")}
                      for name in data}
    train.atomic_json(OUT / "result.json", result)
    print("MEAN", json.dumps(result["mean"]), flush=True)


if __name__ == "__main__":
    main()
