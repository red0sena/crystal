"""Frozen plain-Qwen image/text readout alignment on controlled, disjoint scenes.

Run: .venv/bin/python -m qwen_decision.v3.rotation_pilot
Only an orthogonal map is fitted; no model weights, answer labels, or test
features enter fitting or map selection. This is a synthetic feasibility probe,
not a reproduction of the JEPA theorem or a natural-image benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from qwen_decision.v3 import format as fmt

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
COLORS = {"red": "#df3030", "blue": "#2459df", "green": "#198341",
          "yellow": "#f1cd20", "purple": "#913bbc", "orange": "#ef8220"}
SHAPES = ("circle", "square", "triangle", "diamond")
RELATIONS = ("left", "right", "above", "below")
PHRASES = ("to the left of", "to the right of", "above", "below")
TASKS = ("color", "shape", "relation")
SEED = 20261104
RIDGES = (.001, .01, .1, 1.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def scenes(counts=(384, 96, 192)):
    """Canonical object ordering excludes reversed descriptions of one world."""
    objects = list(itertools.product(COLORS, SHAPES))
    worlds = [(a, b, r) for a, b in itertools.combinations(objects, 2)
              if a[0] != b[0] and a[1] != b[1] for r in range(4)]
    rng = random.Random(SEED)
    rng.shuffle(worlds)
    if sum(counts) > len(worlds) or any(n % 3 for n in counts):
        raise ValueError("need unique worlds and equal task counts")
    rows = []
    cursor = 0
    for split, n in zip(("train", "validation", "test"), counts):
        for j, (a, b, relation) in enumerate(worlds[cursor:cursor + n]):
            kind, target = TASKS[j % 3], rng.randrange(2)
            obj, other = (a, b) if target == 0 else (b, a)
            relative = relation if target == 0 else relation ^ 1
            if kind == "color":
                question = f"What color is the {obj[1]}?"
                choices, gold = list(COLORS), list(COLORS).index(obj[0])
            elif kind == "shape":
                question = f"What shape is the {obj[0]} object?"
                choices, gold = list(SHAPES), SHAPES.index(obj[1])
            else:
                question = f"Where is the {obj[1]} relative to the {other[1]}?"
                choices, gold = ["left", "right", "above", "below"], relative
            rows.append({"id": f"{split}-{j:04d}", "split": split, "task": kind,
                         "world": [list(a), list(b), relation], "drawing_seed": rng.randrange(2**32),
                         "caption": f"A {a[0]} {a[1]} is {PHRASES[relation]} a {b[0]} {b[1]}. "
                                    "There are exactly two objects on a white background.",
                         "question": {"type": "choice", "instructions": question,
                                      "criteria": dict.fromkeys(choices)}, "gold": gold})
        cursor += n
    return rows


def draw_scene(row):
    a, b, relation = row["world"]
    rng = random.Random(row["drawing_seed"])
    positions = [((65, 128), (191, 128)), ((191, 128), (65, 128)),
                 ((128, 65), (128, 191)), ((128, 191), (128, 65))][relation]
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    for (color, shape), (x, y) in zip((a, b), positions):
        x, y, size = x + rng.randint(-8, 8), y + rng.randint(-8, 8), rng.randint(25, 33)
        box, fill = (x - size, y - size, x + size, y + size), COLORS[color]
        if shape == "circle":
            draw.ellipse(box, fill=fill)
        elif shape == "square":
            draw.rectangle(box, fill=fill)
        elif shape == "triangle":
            draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)], fill=fill)
        else:
            draw.polygon([(x, y - size), (x + size, y), (x, y + size), (x - size, y)], fill=fill)
    return image


def fit_rotation(x, y, ridge_ratio):
    """Row vectors: min ||normalize(x) R-normalize(y)||² + n*ratio*||R-I||².

    A positive identity penalty resolves the underdetermined full-space fit
    conservatively (n < hidden dimension); this cannot identify a unique true
    1024-dimensional cross-modal rotation from these few pairs.
    """
    if x.shape != y.shape or x.ndim != 2 or ridge_ratio <= 0:
        raise ValueError("matched matrices and a positive identity penalty required")
    x, y = F.normalize(x.double(), dim=-1), F.normalize(y.double(), dim=-1)
    cross = x.T @ y + len(x) * ridge_ratio * torch.eye(x.shape[1], dtype=torch.float64)
    u, _, vh = torch.linalg.svd(cross)
    return (u @ vh).float()


def normalized_mse(x, y):
    return ((F.normalize(x.float(), dim=-1) - F.normalize(y.float(), dim=-1)) ** 2).sum(-1)


def shuffled_targets(rows):
    """Wrong scene correspondence, while preserving each exact question/options."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[json.dumps(row["question"], sort_keys=True)].append(i)
    rng, order = random.Random(SEED + 1), list(range(len(rows)))
    for indices in groups.values():
        rng.shuffle(indices)
        for a, b in zip(indices, indices[1:] + indices[:1]):
            order[a] = b
    return order


@torch.inference_mode()
def extract(model_path, rows, out, batch_size):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    codes, code_ids = fmt.answer_codes(processor.tokenizer)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.requires_grad_(False).eval()
    assert not any(p.requires_grad for p in model.parameters())
    model.generate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("generation forbidden"))
    collected, lengths = {"image": [], "text": []}, {"image": [], "text": []}
    started = time.monotonic()
    for begin in range(0, len(rows), batch_size):
        group = rows[begin:begin + batch_size]
        for modality in collected:
            prompts = [fmt.render(processor.tokenizer,
                       "<|vision_start|><|image_pad|><|vision_end|>" if modality == "image" else r["caption"],
                       r["question"], codes[:len(r["question"]["criteria"])]) for r in group]
            inputs = processor(text=prompts, images=[draw_scene(r) for r in group] if modality == "image" else None,
                               padding=True, return_tensors="pt").to("cuda")
            assert inputs["attention_mask"][:, -1].all()
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
            collected[modality].append(hidden.float().cpu())
            lengths[modality].extend(inputs["attention_mask"].sum(-1).cpu().tolist())
        if begin == 0 or (begin + batch_size) % 64 == 0:
            print(f"extracted {min(begin + batch_size, len(rows))}/{len(rows)} paired inputs, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
            write_json(out / "progress.json", {"paired_inputs": min(begin + batch_size, len(rows)),
                                              "total": len(rows), "seconds": time.monotonic() - started})
    result = {k: torch.cat(v) for k, v in collected.items()}
    result["head"] = model.get_output_embeddings().weight[code_ids[:6]].float().cpu()
    result["token_lengths"] = lengths
    result["seconds"] = time.monotonic() - started
    result["frozen_parameters"] = sum(p.numel() for p in model.parameters())
    result["trainable_model_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.save(result, out / "features.pt")
    return result


def scores(features, rows, head):
    logits = features @ head.T
    for i, row in enumerate(rows):
        logits[i, len(row["question"]["criteria"]):] = -1e9
    gold = torch.tensor([r["gold"] for r in rows])
    losses = F.cross_entropy(logits, gold, reduction="none")
    correct = logits.argmax(-1) == gold
    return {"accuracy": correct.float().mean().item(), "nll": losses.mean().item(),
            "by_task": {task: {"n": sum(r["task"] == task for r in rows),
                        "accuracy": correct[torch.tensor([r["task"] == task for r in rows])].float().mean().item(),
                        "nll": losses[torch.tensor([r["task"] == task for r in rows])].mean().item()}
                        for task in TASKS}}, logits, correct, losses


def analyze(rows, features, out):
    x, y, head = features["image"], features["text"], features["head"]
    indices = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s])
               for s in ("train", "validation", "test")}
    tr, va, te = [indices[s] for s in ("train", "validation", "test")]
    candidates, rotations = [], []
    for ratio in RIDGES:
        rotation = fit_rotation(x[tr], y[tr], ratio)
        rotations.append(rotation)
        candidates.append({"ridge_ratio": ratio,
                           "validation_normalized_mse": normalized_mse(x[va] @ rotation, y[va]).mean().item()})
    chosen = min(range(len(candidates)), key=lambda i: candidates[i]["validation_normalized_mse"])
    rotation = rotations[chosen]
    train_rows, test_rows = [[rows[i] for i in ids.tolist()] for ids in (tr, te)]
    shuffled = shuffled_targets(train_rows)
    wrong_rotation = fit_rotation(x[tr], y[tr][shuffled], RIDGES[chosen])
    arms = {"image_identity": x[te], "image_rotation": x[te] @ rotation,
            "image_shuffled_pairs_rotation": x[te] @ wrong_rotation, "text_reference": y[te]}
    metrics, outputs, row_results = {}, {}, []
    for name, h in arms.items():
        result, logits, correct, losses = scores(h, test_rows, head)
        result["normalized_alignment_mse"] = normalized_mse(h, y[te]).mean().item()
        result["mean_cosine_to_text"] = F.cosine_similarity(h, y[te]).mean().item()
        metrics[name], outputs[name] = result, (logits, correct, losses)
    generator = torch.Generator().manual_seed(SEED + 2)
    resamples = torch.randint(len(te), (5000, len(te)), generator=generator)
    contrasts = {}
    for comparison in ("image_identity", "image_shuffled_pairs_rotation"):
        acc = outputs["image_rotation"][1].float() - outputs[comparison][1].float()
        nll = outputs["image_rotation"][2] - outputs[comparison][2]
        contrasts["image_rotation-minus-" + comparison] = {
            "accuracy_delta": acc.mean().item(), "accuracy_delta_ci95": torch.quantile(acc[resamples].mean(-1), torch.tensor([.025, .975])).tolist(),
            "nll_delta": nll.mean().item(), "nll_delta_ci95": torch.quantile(nll[resamples].mean(-1), torch.tensor([.025, .975])).tolist()}
    for j, row in enumerate(test_rows):
        row_results.append({"id": row["id"], "task": row["task"], "gold": row["gold"],
                            "arms": {name: {"logits": values[0][j, :len(row["question"]["criteria"])].tolist(),
                                     "correct": values[1][j].item(), "nll": values[2][j].item()}
                                     for name, values in outputs.items()}})
    orthogonal_error = (rotation.T @ rotation - torch.eye(rotation.shape[0])).abs().max().item()
    norm_error = ((x[te] @ rotation).norm(dim=-1) - x[te].norm(dim=-1)).abs().max().item()
    assert orthogonal_error < 1e-5 and norm_error < 1e-3
    report = {"experiment": "plain-qwen-rotation-pilot-v1", "model_revision": REVISION,
              "counts": {s: len(ids) for s, ids in indices.items()}, "hidden_dim": x.shape[1],
              "selection": "minimum validation normalized alignment MSE; no answer labels",
              "candidates": candidates, "selected_ridge_ratio": RIDGES[chosen],
              "metrics": metrics, "paired_scene_bootstrap_5000": contrasts,
              "checks": {"max_orthogonality_error": orthogonal_error, "max_norm_error": norm_error,
                         "shuffled_training_pairs_changed": sum(i != j for i, j in enumerate(shuffled)),
                         "trainable_model_parameters": features["trainable_model_parameters"],
                         "frozen_model_parameters": features["frozen_parameters"]},
              "feature_extraction_seconds": features["seconds"],
              "limitations": ["Synthetic two-object scenes and one prompt family only; not natural-image transfer.",
                              "Shared frozen VLM final readouts, not two independently trained JEPA encoders.",
                              "384 pairs < 1024 dimensions: identity-regularized orthogonal fit; full rotation not identifiable.",
                              "Confidence intervals cover sampled test scenes, not model seeds or new domains.",
                              "Pairs and captions are programmatically constructed; no teacher or generated labels."]}
    write_json(out / "report.json", report)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in row_results))
    torch.save({"rotation": rotation, "shuffled_rotation": wrong_rotation}, out / "rotation.pt")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--out", type=Path, default=Path("runs/rotation-pilot/plain-qwen-v1"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    rows = scenes()
    if args.analyze_only:
        stored = [json.loads(line) for line in (args.out / "scenes.jsonl").read_text().splitlines()]
        if stored != rows:
            raise ValueError("saved scenes differ from deterministic plan")
        analyze(rows, torch.load(args.out / "features.pt", weights_only=True), args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"model": str(args.model.resolve()), "revision": REVISION, "attention": "untouched causal",
            "model_parameters": "all frozen", "dataset": "canonical distinct synthetic two-object scenes",
            "counts": [384, 96, 192], "image_size": [256, 256], "seed": SEED,
            "ridges": RIDGES, "rotation": "uncentered orthogonal Procrustes with identity penalty",
            "selection": "validation normalized alignment MSE, never answer labels or test features",
            "primary_comparison": "rotation vs identity on held-out scene accuracy and NLL",
            "negative_control": "wrong scene pairs within exact same question and option text",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.out / "plan.json", plan)
    (args.out / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows[:6]:
        draw_scene(row).save(args.out / (row["id"] + ".png"))
    features = extract(args.model, rows, args.out, args.batch_size)
    analyze(rows, features, args.out)


if __name__ == "__main__":
    main()
