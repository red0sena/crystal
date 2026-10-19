"""Evaluate one model in the v2 format and store per-item option logits.

Every model (original, v1, v2) is scored with the same decision_format prompt,
sequential codes and the stored option order, so differences come from the
weights. Raw logits are kept; calibration and statistics are computed later by
report_v2.py.

Passes:
  test         every record of data/text-decisions-v2/test.jsonl
  unseen       HellaSwag, WinoGrande, OpenBookQA (data/unseen-text-v1)
  external     SuperGPQA and KMMLU domain/difficulty (data/external-classification-eval-v1)
  permutation  up to 200 choice items per test task, three extra option orders
  robustness   up to 200 items per test task with an injected instruction, and
               with the gold removed and "None of the above" added
  calibration  development rows outside the model-selection subset (for fitting)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

import decision_format as fmt
from scorer_v2 import DecisionScorer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/text-decisions-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(x) for x in f]


def item(record, *, task=None, names=None, extra_options=None, state=None, gold=None, tag=None):
    options = dict(record["request"]["options"])
    if extra_options:
        options.update(extra_options)
    names = names or list(options)
    return {"id": record["id"] + (f"#{tag}" if tag else ""), "source_id": record["id"], "task": task or record["task"],
            "qtype": record.get("qtype", "choice"), "state": state if state is not None else record["request"]["state"],
            "question": record["request"]["question"], "names": names, "texts": [options[n] for n in names],
            "gold": names.index(gold or record["answer_id"]), "target": record.get("target"), "tag": tag}


def run(scorer, items, out_path, label):
    tokenizer = scorer.tokenizer
    tick = time.time()
    sequences = []
    for it in items:
        codes = fmt.assign_codes(len(it["names"]))
        text = fmt.prompt_text(tokenizer, it["state"], it["question"], codes, it["texts"], it["qtype"])
        sequences.append(tokenizer.encode(text, add_special_tokens=False))
    too_long = [i for i, s in enumerate(sequences) if len(s) > scorer.max_input_tokens]
    assert not too_long, f"{len(too_long)} inputs exceed the limit; inputs are never truncated"
    hidden = scorer.last_hidden(sequences)
    correct = 0
    with out_path.open("w") as f:
        for it, h, seq in zip(items, hidden, sequences):
            logits = scorer.option_logits(h, fmt.assign_codes(len(it["names"])))
            pred = int(logits.argmax())
            correct += pred == it["gold"]
            f.write(json.dumps({"id": it["id"], "source_id": it["source_id"], "task": it["task"], "qtype": it["qtype"],
                                "tag": it["tag"], "n_options": len(it["names"]), "gold": it["gold"], "pred": pred,
                                "correct": pred == it["gold"], "names": it["names"], "target": it["target"],
                                "input_tokens": len(seq), "logits": [round(x, 4) for x in logits.tolist()]}) + "\n")
    seconds = time.time() - tick
    print(f"{label}: {len(items)} items, accuracy {correct / len(items):.4f}, {seconds:.0f}s", flush=True)
    return {"items": len(items), "accuracy": correct / len(items), "seconds": seconds, "file": str(out_path)}


def permutation_items(test, per_task=200):
    by = defaultdict(list)
    for r in test:
        if r["qtype"] == "choice" and len(r["request"]["options"]) >= 2:
            by[r["task"]].append(r)
    items = []
    for task, rr in sorted(by.items()):
        rr = sorted(rr, key=lambda r: hashlib.sha256(("perm:" + r["id"]).encode()).hexdigest())[:per_task]
        for r in rr:
            names = list(r["request"]["options"])
            items.append(item(r, tag="perm0"))
            for k in (1, 2, 3):
                rng = random.Random(int(hashlib.sha256(f"perm{k}:{r['id']}".encode()).hexdigest(), 16))
                shuffled = names[:]
                rng.shuffle(shuffled)
                if shuffled == names:
                    shuffled = names[1:] + names[:1]
                items.append(item(r, names=shuffled, tag=f"perm{k}"))
    return items


def robustness_items(test, per_task=200):
    by = defaultdict(list)
    for r in test:
        by[r["task"]].append(r)
    items = []
    for task, rr in sorted(by.items()):
        rr = sorted(rr, key=lambda r: hashlib.sha256(("robust:" + r["id"]).encode()).hexdigest())[:per_task]
        for r in rr:
            names = list(r["request"]["options"])
            rng = random.Random(int(hashlib.sha256(("robust:" + r["id"]).encode()).hexdigest(), 16))
            gold = names.index(r["answer_id"])
            wrong = rng.choice([i for i in range(len(names)) if i != gold])
            code = fmt.CODES[wrong]
            line = fmt.INJECTIONS[0].format(code=code)
            items.append(item(r, state=r["request"]["state"] + "\n" + line, tag=f"inject:{wrong}"))
            if r["qtype"] == "choice" and 3 <= len(names) < fmt.MAX_OPTIONS:
                kept = [n for n in names if n != r["answer_id"]]
                kept.insert(rng.randrange(len(kept) + 1), fmt.ESCAPE_NAME)
                items.append(item(r, names=kept, extra_options={fmt.ESCAPE_NAME: fmt.ESCAPE_TEXT},
                                  gold=fmt.ESCAPE_NAME, tag="escape_gold_removed"))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", default="test,unseen,external,permutation,robustness,calibration")
    parser.add_argument("--calibration-per-task", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: keep every n-th item only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = DecisionScorer(args.model, calibration=None)
    original_generate = getattr(scorer.lm, "generate", None)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": str(args.model), "model_files_sha256": {p.name: sha(p) for p in sorted(args.model.glob("*.safetensors"))},
        "data_manifest_sha256": sha(DATA / "manifest.json"), "format_sha256": sha(ROOT / "decision_format.py"),
        "evaluator_sha256": sha(Path(__file__)), "torch": torch.__version__, "dtype": "bfloat16",
        "protocol": "decision_format prompt, sequential codes, stored option order, right padding, last real token, "
                    "LM-head code rows, raw logits saved; zero generated tokens", "passes": {}}
    test = read_jsonl(DATA / "test.jsonl")
    for name in args.passes.split(","):
        if name in manifest["passes"]:
            print(f"skip completed pass {name}", flush=True)
            continue
        if name == "test":
            items = [item(r) for r in test]
        elif name == "unseen":
            items = [item(r, task=b) for b in ("HellaSwag", "WinoGrande", "OpenBookQA")
                     for r in read_jsonl(ROOT / f"data/unseen-text-v1/{b}.jsonl")]
        elif name == "external":
            items = [item(r) for t in ("SuperGPQA_domain", "SuperGPQA_difficulty", "KMMLU_domain", "KMMLU_difficulty")
                     for r in read_jsonl(ROOT / f"data/external-classification-eval-v1/{t}.jsonl")]
        elif name == "permutation":
            items = permutation_items(test)
        elif name == "robustness":
            items = robustness_items(test)
        elif name == "calibration":
            selection = {r["id"] for r in json.loads((ROOT / "training/text-decision-v2/development-records.json").read_text())}
            by = defaultdict(list)
            for r in read_jsonl(DATA / "dev.jsonl"):
                if r["id"] not in selection:
                    by[r["task"]].append(r)
            items = [item(r) for task, rr in sorted(by.items())
                     for r in sorted(rr, key=lambda r: hashlib.sha256(("calib:" + r["id"]).encode()).hexdigest())[:args.calibration_per_task]]
        else:
            raise ValueError(name)
        if args.limit:
            items = items[::args.limit]
        manifest["passes"][name] = run(scorer, items, args.output / f"{name}.jsonl", name)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    assert getattr(scorer.lm, "generate", None) is original_generate


if __name__ == "__main__":
    main()
