"""Question-free state alignment and text-only conditional decision transfer.

The untouched Qwen3.5-0.8B is a frozen feature extractor. A shared PCA and
centered orthogonal map use training states only. A small decision MLP is
trained on text states and seen queries only, then frozen for image transfer.
This is a controlled synthetic application, not independent-JEPA training.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from qwen_decision.v3.rotation_pilot import COLORS, SHAPES, REVISION

SEED = 202611051
COUNTS = (1024, 256, 256)
DIM = 64
HEAD_SEEDS = (271, 283, 307)
SHUFFLES = 20
ANSWER_NAMES = list(COLORS) + list(SHAPES) + ["left", "right", "above", "below"]
TASKS = ("color", "shape", "relation")
HOLDOUT = {"color:diamond", "shape:purple", "relation:circle:triangle", "relation:triangle:circle",
           "relation:square:diamond", "relation:diamond:square"}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def make_scenes():
    worlds = []
    for colors in itertools.combinations(COLORS, 3):
        for shapes in itertools.permutations(SHAPES, 3):
            objects = list(zip(colors, shapes))
            for order in itertools.permutations(range(3)):
                for axis in ("horizontal", "vertical"):
                    worlds.append({"objects": [list(objects[i]) for i in order], "axis": axis})
    rng = random.Random(SEED)
    rng.shuffle(worlds)
    rows, cursor = [], 0
    for split, count in zip(("train", "validation", "test"), COUNTS):
        for j, world in enumerate(worlds[cursor:cursor + count]):
            names = [f"a {c} {s}" for c, s in world["objects"]]
            direction = "left to right" if world["axis"] == "horizontal" else "top to bottom"
            caption = (f"Three objects are arranged {world['axis']}ly on a white background. "
                       f"From {direction}: {names[0]}, {names[1]}, and {names[2]}.")
            # Correct the English adverb without changing spatial facts.
            caption = caption.replace("verticallyly", "vertically")
            rows.append({"id": f"{split}-{j:04d}", "split": split, "world": world,
                         "drawing_seed": rng.randrange(2**32), "caption": caption})
        cursor += count
    return rows


def draw_scene(row):
    rng = random.Random(row["drawing_seed"])
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    horizontal = row["world"]["axis"] == "horizontal"
    for (color, shape), position in zip(row["world"]["objects"], (48, 128, 208)):
        x, y = (position, 128) if horizontal else (128, position)
        x, y, radius = x + rng.randint(-5, 5), y + rng.randint(-5, 5), rng.randint(19, 25)
        fill, box = COLORS[color], (x - radius, y - radius, x + radius, y + radius)
        if shape == "circle":
            draw.ellipse(box, fill=fill)
        elif shape == "square":
            draw.rectangle(box, fill=fill)
        elif shape == "triangle":
            draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=fill)
        else:
            draw.polygon([(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)], fill=fill)
    return image


def query_specs():
    result = []
    for shape in SHAPES:
        result.append({"key": f"color:{shape}", "text": f"What color is the {shape}?", "task": "color",
                       "reference": shape, "answers": list(range(6))})
    for color in COLORS:
        result.append({"key": f"shape:{color}", "text": f"What shape is the {color} object?", "task": "shape",
                       "reference": color, "answers": list(range(6, 10))})
    for a, b in itertools.permutations(SHAPES, 2):
        result.append({"key": f"relation:{a}:{b}", "text": f"Where is the {a} relative to the {b}?", "task": "relation",
                       "reference": a, "other": b, "answers": list(range(10, 14))})
    return result


def examples(rows, specs):
    result = []
    for i, row in enumerate(rows):
        objects = row["world"]["objects"]
        by_shape = {s: (j, c) for j, (c, s) in enumerate(objects)}
        by_color = {c: s for c, s in objects}
        for q, spec in enumerate(specs):
            if spec["task"] == "color":
                if spec["reference"] not in by_shape:
                    continue
                answer = by_shape[spec["reference"]][1]
            elif spec["task"] == "shape":
                if spec["reference"] not in by_color:
                    continue
                answer = by_color[spec["reference"]]
            else:
                if spec["reference"] not in by_shape or spec["other"] not in by_shape:
                    continue
                before = by_shape[spec["reference"]][0] < by_shape[spec["other"]][0]
                answer = ("left" if before else "right") if row["world"]["axis"] == "horizontal" else ("above" if before else "below")
            result.append({"scene": i, "query": q, "task": TASKS.index(spec["task"]), "gold": ANSWER_NAMES.index(answer),
                           "unseen": spec["key"] in HOLDOUT})
    return result


def prompt(tokenizer, content, label="Scene"):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": label + ":\n" + content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def content_mask(tokenizer, prompts, contents, input_ids):
    encoded = tokenizer(prompts, padding=True, return_tensors="pt", return_offsets_mapping=True)
    assert torch.equal(encoded["input_ids"], input_ids.cpu())
    masks = []
    for text, content, offsets in zip(prompts, contents, encoded["offset_mapping"]):
        start = text.index(content)
        masks.append((offsets[:, 0] < start + len(content)) & (offsets[:, 1] > start))
    return torch.stack(masks)


def masked_mean(hidden, mask):
    assert mask.any(-1).all()
    return (hidden.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)


@torch.inference_mode()
def extract(model_path, rows, specs, out, batch_size):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    model.generate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("generation forbidden"))
    collected, token_counts = {"image": [], "text": []}, {"image": [], "text": []}
    started = time.monotonic()
    for begin in range(0, len(rows), batch_size):
        group = rows[begin:begin + batch_size]
        for modality in collected:
            contents = ["<|vision_start|><|image_pad|><|vision_end|>" if modality == "image" else row["caption"] for row in group]
            prompts = [prompt(processor.tokenizer, text) for text in contents]
            inputs = processor(text=prompts, images=[draw_scene(row) for row in group] if modality == "image" else None,
                               padding=True, return_tensors="pt").to("cuda")
            mask = (inputs["input_ids"] == model.config.image_token_id) if modality == "image" else content_mask(
                processor.tokenizer, prompts, contents, inputs["input_ids"]).to("cuda")
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            collected[modality].append(masked_mean(hidden, mask).cpu())
            token_counts[modality].extend(mask.sum(-1).cpu().tolist())
        if begin == 0 or (begin + batch_size) % 128 == 0:
            done = min(begin + batch_size, len(rows))
            print(f"state pairs {done}/{len(rows)}, {time.monotonic() - started:.1f}s", flush=True)
            dump(out / "progress.json", {"stage": "state_features", "done": done, "total": len(rows)})
    query_features = []
    for begin in range(0, len(specs), batch_size):
        contents = [s["text"] for s in specs[begin:begin + batch_size]]
        prompts = [prompt(processor.tokenizer, text, "Question") for text in contents]
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
        mask = content_mask(processor.tokenizer, prompts, contents, inputs["input_ids"]).to("cuda")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        query_features.append(masked_mean(hidden, mask).cpu())
    result = {k: torch.cat(v) for k, v in collected.items()}
    result.update({"query": torch.cat(query_features), "pooling_token_counts": token_counts,
                   "extraction_seconds": time.monotonic() - started,
                   "qwen_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    torch.save(result, out / "features.pt")
    return result


def orthogonal(x, y):
    u, _, vh = torch.linalg.svd(x.double().T @ y.double())
    return (u @ vh).float()


def state_space(x, y, train_indices, dim=DIM):
    a, b = x.double(), y.double()
    mu_a, mu_b = a[train_indices].mean(0), b[train_indices].mean(0)
    _, singular, vh = torch.linalg.svd(torch.cat((a[train_indices] - mu_a, b[train_indices] - mu_b)), full_matrices=False)
    basis = vh[:dim].T
    normalize = lambda z: F.normalize(z.float(), dim=-1)
    return {"image": normalize((a - mu_a) @ basis), "text": normalize((b - mu_b) @ basis),
            "raw_image": normalize((a - mu_b) @ basis), "basis": basis.float(),
            "mu_image": mu_a.float(), "mu_text": mu_b.float(),
            "pca_explained_energy": float(singular[:dim].square().sum() / singular.square().sum())}


class Decision(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * DIM, 256), nn.GELU(), nn.Dropout(.05),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(ANSWER_NAMES)))

    def forward(self, state, question, answer_mask):
        return self.net(torch.cat((state, question), dim=-1)).masked_fill(~answer_mask, -1e9)


def tensors(items, specs, device="cuda"):
    masks = torch.zeros((len(specs), len(ANSWER_NAMES)), dtype=torch.bool, device=device)
    for q, spec in enumerate(specs):
        masks[q, spec["answers"]] = True
    return {"state": torch.tensor([r["scene"] for r in items], device=device),
            "query": torch.tensor([r["query"] for r in items], device=device),
            "gold": torch.tensor([r["gold"] for r in items], device=device),
            "task": torch.tensor([r["task"] for r in items], device=device), "masks": masks}


def fit_decision(seed, text_states, query_vectors, training, validation, specs, out):
    torch.manual_seed(seed)
    model = Decision().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    train, val = tensors(training, specs), tensors(validation, specs)
    states, questions = text_states.to("cuda"), query_vectors.to("cuda")
    counts = torch.bincount(train["task"], minlength=3).float()
    weights = (len(training) / (3 * counts))[train["task"]]
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    started = time.monotonic()
    for epoch in range(1, 161):
        model.train()
        order = torch.randperm(len(training), device="cuda")
        for ids in order.split(512):
            q = train["query"][ids]
            logits = model(states[train["state"][ids]], questions[q], train["masks"][q])
            loss = (F.cross_entropy(logits, train["gold"][ids], reduction="none") * weights[ids]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            q = val["query"]
            z = model(states[val["state"]], questions[q], val["masks"][q])
            losses = F.cross_entropy(z, val["gold"], reduction="none")
            score = torch.stack([losses[val["task"] == t].mean() for t in range(3)]).mean().item()
            accuracy = (z.argmax(-1) == val["gold"]).float().mean().item()
        history.append({"epoch": epoch, "validation_seen_macro_nll": score, "validation_seen_accuracy": accuracy})
        if score < best - 1e-5:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 20 == 0:
            print(f"text-only head seed={seed} epoch={epoch} valNLL={score:.4f} valAcc={accuracy:.3f}", flush=True)
        if stale >= 20:
            break
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    torch.save(best_state, out / f"text-head-{seed}.pt")
    dump(out / f"text-head-{seed}.json", {"seed": seed, "best_epoch": best_epoch, "best_validation_nll": best,
                                         "seconds": time.monotonic() - started, "history": history})
    return model


@torch.inference_mode()
def decision_records(model, state_vectors, query_vectors, items, specs):
    data = tensors(items, specs)
    states, questions = state_vectors.to("cuda"), query_vectors.to("cuda")
    correct, losses = [], []
    for ids in torch.arange(len(items), device="cuda").split(1024):
        q = data["query"][ids]
        z = model(states[data["state"][ids]], questions[q], data["masks"][q])
        correct.extend((z.argmax(-1) == data["gold"][ids]).cpu().tolist())
        losses.extend(F.cross_entropy(z, data["gold"][ids], reduction="none").cpu().tolist())
    # Aggregate within scene, retaining question counts for clustered resampling.
    scenes = sorted({r["scene"] for r in items})
    lookup = {s: i for i, s in enumerate(scenes)}
    array = np.zeros((len(scenes), 2, 3, 3), dtype=np.float64)
    for row, right, nll in zip(items, correct, losses):
        slot = array[lookup[row["scene"]], int(row["unseen"]), row["task"]]
        slot += [right, nll, 1]
    return array


def summarize(array):
    # Input [heads, scenes, seen/unseen, tasks, correct/nll/count].
    summed = array.sum(axis=1).mean(axis=0)
    result = {}
    for group, values in zip(("seen", "unseen"), summed):
        rates = values[:, :2] / values[:, 2:3]
        result[group] = {"macro_accuracy": float(rates[:, 0].mean()), "macro_nll": float(rates[:, 1].mean()),
                         "questions": int(values[:, 2].sum()),
                         "by_task": {task: {"accuracy": float(rates[i, 0]), "nll": float(rates[i, 1]),
                                            "questions": int(values[i, 2])} for i, task in enumerate(TASKS)}}
    return result


def alignment_metrics(a, b, indices):
    x, y = F.normalize(a[indices].double(), dim=-1), F.normalize(b[indices].double(), dim=-1)
    sim = x @ y.T
    ranks = (sim >= sim.diag()[:, None]).sum(-1) - 1
    return {"paired_cosine": (x * y).sum(-1).mean().item(),
            "retrieval_R1": (ranks < 1).double().mean().item(), "retrieval_R5": (ranks < 5).double().mean().item()}


def analyze(rows, specs, features, out):
    split = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "validation", "test")}
    tr, va, te = split["train"], split["validation"], split["test"]
    space = state_space(features["image"], features["text"], tr)
    rotation = orthogonal(space["image"][tr], space["text"][tr])
    # Independent fixed projection of query features; only seen queries center it.
    seen_queries = [i for i, q in enumerate(specs) if q["key"] not in HOLDOUT]
    generator = torch.Generator().manual_seed(SEED + 4)
    projection = torch.randn(features["query"].shape[1], DIM, generator=generator) / DIM**.5
    query_mean = features["query"][seen_queries].mean(0)
    query_vectors = F.normalize((features["query"] - query_mean) @ projection, dim=-1)
    variants = {"image_raw": space["raw_image"], "image_mean_only": space["image"],
                "image_rotation": space["image"] @ rotation, "text_reference": space["text"]}
    rotations = []
    for seed in range(SHUFFLES):
        rng = random.Random(SEED + 100 + seed)
        order = list(range(len(tr))); rng.shuffle(order)
        permutation = list(range(len(tr)))
        for a, b in zip(order, order[1:] + order[:1]):
            permutation[a] = b
        wrong = orthogonal(space["image"][tr], space["text"][tr][permutation])
        rotations.append(wrong)
        variants[f"shuffle_{seed:02d}"] = space["image"] @ wrong
    all_items = examples(rows, specs)
    training = [e for e in all_items if e["scene"] in set(tr) and not e["unseen"]]
    validation = [e for e in all_items if e["scene"] in set(va) and not e["unseen"]]
    testing = [e for e in all_items if e["scene"] in set(te)]
    assert not any(specs[e["query"]]["key"] in HOLDOUT for e in training + validation)
    arrays = {name: [] for name in variants}
    for seed in HEAD_SEEDS:
        model = fit_decision(seed, space["text"], query_vectors, training, validation, specs, out)
        for name, state in variants.items():
            arrays[name].append(decision_records(model, state, query_vectors, testing, specs))
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    results = {name: summarize(array) for name, array in arrays.items()}
    align = {name: alignment_metrics(state, space["text"], te) for name, state in variants.items() if name != "text_reference"}
    rng = np.random.default_rng(SEED + 200)
    comparisons = {}
    for baseline in ("image_raw", "image_mean_only"):
        samples = {group: {key: [] for key in ("accuracy", "nll")} for group in ("seen", "unseen")}
        for _ in range(5000):
            seeds = rng.integers(len(HEAD_SEEDS), size=len(HEAD_SEEDS))
            scenes = rng.integers(len(te), size=len(te))
            a = summarize(arrays["image_rotation"][seeds][:, scenes])
            b = summarize(arrays[baseline][seeds][:, scenes])
            for group in samples:
                samples[group]["accuracy"].append(a[group]["macro_accuracy"] - b[group]["macro_accuracy"])
                samples[group]["nll"].append(a[group]["macro_nll"] - b[group]["macro_nll"])
        comparisons[baseline] = {group: {key + "_delta_ci95": np.quantile(values, [.025, .975]).tolist()
                                         for key, values in metric.items()} for group, metric in samples.items()}
    shuffle_summary = {}
    for group in ("seen", "unseen"):
        shuffle_summary[group] = {}
        for metric in ("macro_accuracy", "macro_nll"):
            vals = [results[f"shuffle_{s:02d}"][group][metric] for s in range(SHUFFLES)]
            shuffle_summary[group][metric] = {"mean": float(np.mean(vals)), "min": min(vals), "max": max(vals)}
    singular = torch.linalg.svdvals(space["image"][tr].double())
    report = {"experiment": "plain-qwen-question-free-state-rotation-v2", "counts": {k: len(v) for k, v in split.items()},
              "queries": {"total_signatures": len(specs), "seen_signatures": len(seen_queries), "heldout_signatures": sorted(HOLDOUT),
                          "train_records": len(training), "validation_records": len(validation), "test_records": len(testing)},
              "representation": {"source_dimension": features["image"].shape[1], "aligned_dimension": DIM,
                                 "shared_train_PCA_energy": space["pca_explained_energy"],
                                 "image_train_min_singular_value": singular[-1].item(),
                                 "qwen_trainable_parameters": features["qwen_trainable_parameters"],
                                 "state_input_contains_question_or_options": False},
              "alignment_test": align, "decision_test": {k: v for k, v in results.items() if not k.startswith("shuffle_")},
              "shuffle_decision_summary_20": shuffle_summary, "paired_head_seed_scene_bootstrap_5000": comparisons,
              "text_reference_gate": {"seen_at_least_80_percent": results["text_reference"]["seen"]["macro_accuracy"] >= .8,
                                      "unseen_at_least_70_percent": results["text_reference"]["unseen"]["macro_accuracy"] >= .7},
              "extraction_seconds": features["extraction_seconds"],
              "checks": {"orthogonality_max_error": (rotation.T @ rotation - torch.eye(DIM)).abs().max().item(),
                         "distinct_worlds": len({json.dumps(r["world"]) for r in rows}),
                         "train_state_pairs": len(tr), "unique_state_vectors_reused_for_test_questions": len(te)},
              "limitations": ["Synthetic three-object scenes; no natural-image claim.",
                              "Same jointly pretrained frozen VLM on both sides, not independent JEPA training.",
                              "Shared PCA reduces to 64 dimensions before rotation; not full 1024-dimensional equivalence.",
                              "Small text-only MLP is newly trained; this is not the native Qwen code head.",
                              "Unseen query signatures are fixed semantic combinations, not arbitrary new tasks.",
                              "Bootstrap conditions on one state split/PCA/rotation; head seeds and test scenes are resampled.",
                              "If the text-only unseen gate fails, image unseen failure cannot be attributed solely to the rotation."]}
    torch.save({**{k: space[k] for k in ("basis", "mu_image", "mu_text")}, "rotation": rotation,
                "shuffled_rotations": torch.stack(rotations), "query_projection": projection, "query_mean": query_mean}, out / "maps.pt")
    np.savez_compressed(out / "test-scene-records.npz", **arrays)
    dump(out / "test-items.json", testing)
    dump(out / "all-shuffle-results.json", results)
    dump(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--out", type=Path, default=Path("runs/rotation-state/plain-qwen-v2"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    rows, specs = make_scenes(), query_specs()
    if args.analyze_only:
        assert json.loads((args.out / "scenes.json").read_text()) == rows
        analyze(rows, specs, torch.load(args.out / "features.pt", weights_only=True), args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"experiment": "plain-qwen-question-free-state-rotation-v2", "revision": REVISION,
            "seed": SEED, "counts": COUNTS, "state_pooling": "mean of final-layer content tokens; image pads or caption offsets only",
            "state_prompt": "helpful-assistant system; user Scene: <image-or-caption>; no question/options",
            "state_split": "new three-object worlds, disjoint semantic states; train/validation/test",
            "space": "separate modality means and shared 64D PCA from training states; per-state unit normalization",
            "rotation": "unregularized square orthogonal Procrustes, 1024 pairs for 64 dimensions; no map hyperparameter selection",
            "controls": ["raw image centered by text mean", "separate mean only", "20 global deranged pairings"],
            "decision": "text states + separately encoded question; 128-256-256-14 MLP, dropout .05, AdamW lr .001 wd .001",
            "head_seeds": HEAD_SEEDS, "head_training": "text gold only, seen questions only, task-balanced CE, batch512, max160 epochs, validation-seen NLL best, patience20",
            "heldout_query_keys": sorted(HOLDOUT), "question_vectors": "frozen Qwen content mean, center by seen-query mean; fixed random 1024x64 projection",
            "evaluation": "freeze heads; query all test states; task-macro accuracy/NLL, seen and heldout queries separately; text reference gate .80/.70",
            "uncertainty": "5000 paired head-seed x whole-scene resamples; all questions of a scene stay together",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(args.out / "plan.json", plan)
    dump(args.out / "scenes.json", rows); dump(args.out / "query-specs.json", specs)
    for row in rows[:6]:
        draw_scene(row).save(args.out / (row["id"] + ".png"))
    features = extract(args.model, rows, specs, args.out, args.batch_size)
    analyze(rows, specs, features, args.out)


if __name__ == "__main__":
    main()
