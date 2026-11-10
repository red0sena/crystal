"""Actual state-bottleneck Decision training: CE vs same/between-observation LeJEPA.

Frozen Qwen token features -> learned question-free 64D state -> learned
factored Decision decoder. Both losses act on that very state. No post-hoc
PCA/probe, no cross-modal prediction loss, no image answer supervision.
This is a shared-encoder synthetic adaptation, not the independent-encoder theorem.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from qwen_decision.v3 import state_rotation as sr

ROOT = Path("runs/rotation-state/jepa-decision-v1")
PARENT = Path("runs/rotation-state/plain-qwen-v2")
BASE = Path("models/qwen3.5-0.8b")
SEEDS = (401, 419, 443)
ARMS = ("ce", "bad", "good")
DIM, WIDTH, LENGTH = 64, 128, 128
EPOCHS, BATCH, AUX_WEIGHT = 200, 128, .1
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for b in iter(lambda: handle.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def data():
    rows, specs = read(PARENT / "scenes.json"), read(PARENT / "query-specs.json")
    assert rows == sr.make_scenes() and specs == sr.query_specs()
    return rows, specs


def caption_observation(row, observation):
    objs = [f"a {c} {s}" for c, s in row["world"]["objects"]]
    vertical = row["world"]["axis"] == "vertical"
    direction = "top to bottom" if vertical else "left to right"
    locations = ("top", "middle", "bottom") if vertical else ("left", "middle", "right")
    style = (row["drawing_seed"] + observation * 3) % 6
    if style == 0:
        return f"From {direction}, the objects are {objs[0]}, {objs[1]}, and {objs[2]}."
    if style == 1:
        return f"There is {objs[0]} at the {locations[0]}, {objs[1]} in the middle, and {objs[2]} at the {locations[2]}."
    if style == 2:
        return f"The arrangement goes from {direction}: {objs[0]}; {objs[1]}; {objs[2]}."
    if style == 3:
        return f"The {locations[2]} object is {objs[2]}. The {locations[0]} object is {objs[0]}. The middle object is {objs[1]}."
    if style == 4:
        return f"Reading from {direction}, one finds {objs[0]}, then {objs[1]}, then {objs[2]}."
    return f"In the middle is {objs[1]}, with {objs[0]} at the {locations[0]} and {objs[2]} at the {locations[2]}."


def text_view(row, observation, augmentation):
    words = caption_observation(row, observation).split()
    # Delete optional articles only: never mask a color, shape, order, or location.
    rng = random.Random(row["drawing_seed"] + 104729 * observation + 8191 * augmentation)
    candidates = [i for i, word in enumerate(words) if word.lower() in ("a", "the")]
    dropped = {i for i in candidates if rng.random() < .45}
    if not dropped and candidates:
        dropped.add(candidates[augmentation % len(candidates)])
    return " ".join(w for i, w in enumerate(words) if i not in dropped)


def image_view(row, observation, augmentation):
    changed = {**row, "drawing_seed": (row["drawing_seed"] + 104729 * (observation + 1)) % 2**32}
    pixels = np.asarray(sr.draw_scene(changed), dtype=np.int16)
    rng = np.random.default_rng(row["drawing_seed"] + 8191 * (augmentation + 1))
    # Mild sensor noise on the same base observation. No crop/flip/color swap.
    noise = rng.integers(-2, 3, size=pixels.shape, dtype=np.int16)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def pair_indices(seed, step, size, arm, device):
    generator = torch.Generator(device=device).manual_seed(seed * 1000003 + step)
    obs = torch.randint(2, (size,), generator=generator, device=device)
    a = torch.randint(2, (size,), generator=generator, device=device)
    b = torch.randint(2, (size,), generator=generator, device=device)
    first = 2 * obs + a
    second = 2 * (1 - obs) + b if arm == "good" else 2 * obs + (1 - a)
    return first, second


@torch.inference_mode()
def prepare(out):
    from transformers import AutoProcessor
    from qwen_decision.v3 import modeling
    rows, specs = data()
    out.mkdir(parents=True, exist_ok=False)
    sr.dump(out / "view-plan.json", {"source_sha256": sha(__file__), "worlds": sr.COUNTS,
            "parent_scenes_sha256": sha(PARENT / "scenes.json"), "token_feature_limit": LENGTH,
            "frozen_frontend": "original Qwen, noncausal, final content tokens, no questions/options",
            "views": "2 independently rendered/captioned observations x 2 semantics-preserving corruptions",
            "good": "different observations", "bad": "same observation, different corruptions"})
    sr.dump(out / "scenes.json", rows); sr.dump(out / "query-specs.json", specs)
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    modeling.compile_elementwise()
    model = modeling.load(BASE, attention="noncausal").eval().requires_grad_(False)
    model.model.visual.to("cuda")
    banks, counts = {}, {}
    for modality in ("text", "image"):
        jobs = [(i, -1) for i in range(len(rows))] + [(i, v) for i in range(1024) for v in range(4)]
        bank = torch.zeros((len(jobs), LENGTH, 1024), dtype=torch.float16)
        lens = torch.zeros(len(jobs), dtype=torch.long)
        for start in range(0, len(jobs), 4):
            group = jobs[start:start + 4]
            contents, pictures = [], []
            for index, view in group:
                row = rows[index]
                contents.append(IMAGE if modality == "image" else row["caption"] if view < 0 else text_view(row, view // 2, view % 2))
                if modality == "image":
                    pictures.append(sr.draw_scene(row) if view < 0 else image_view(row, view // 2, view % 2))
            prompts = [sr.prompt(processor.tokenizer, text) for text in contents]
            inputs = processor(text=prompts, images=pictures or None, padding=True, return_tensors="pt").to("cuda")
            mask = inputs["input_ids"] == model.config.image_token_id if modality == "image" else sr.content_mask(
                processor.tokenizer, prompts, contents, inputs["input_ids"]).to("cuda")
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            for j in range(len(group)):
                value = hidden[j, mask[j]]
                assert 0 < len(value) <= LENGTH, len(value)
                bank[start + j, :len(value)] = value.half().cpu(); lens[start + j] = len(value)
            if start % 512 == 0:
                print("FEATURES", modality, start, len(jobs), flush=True)
        banks[modality], counts[modality] = bank, lens
        torch.save({"values": bank, "lengths": lens}, out / f"{modality}-tokens.pt")
    print("FEATURES COMPLETE", flush=True)


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, WIDTH)
        self.ff = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 256), nn.GELU(), nn.Linear(256, WIDTH))
        self.context = nn.TransformerEncoderLayer(WIDTH, 4, 256, dropout=0., batch_first=True, norm_first=True)
        self.pool = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.slots = nn.Parameter(torch.randn(8, WIDTH) * .5)
        self.out = nn.Sequential(nn.LayerNorm(8 * WIDTH), nn.Linear(8 * WIDTH, DIM))
        p, f = torch.arange(LENGTH).float()[:, None], torch.exp(torch.arange(0, WIDTH, 2).float() * (-math.log(10000) / WIDTH))
        pe = torch.zeros(LENGTH, WIDTH); pe[:, 0::2] = torch.sin(p * f); pe[:, 1::2] = torch.cos(p * f)
        self.register_buffer("positions", pe)

    def forward(self, tokens, lengths):
        h = self.proj(F.layer_norm(tokens.float(), (1024,))) + .1 * self.positions[None]
        h = h + self.ff(h)
        padding = torch.arange(LENGTH, device=h.device)[None] >= lengths[:, None]
        h = self.context(h, src_key_padding_mask=padding)
        pooled, _ = self.pool(self.slots[None].expand(len(h), -1, -1), h, h, key_padding_mask=padding, need_weights=False)
        return self.out(pooled.flatten(1))


class FactoredDecision(nn.Module):
    """Shared differentiable object binding for this dataset's structured queries.

    No object/axis gold is used: the 32 intermediate logits train only through
    the original seen answer CE. Query grammar is given, not learned language.
    """
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(nn.Linear(DIM, 256), nn.GELU(), nn.Linear(256, 32))

    @staticmethod
    def answer_logits(scene_logits, specs):
        colors = scene_logits[:, :18].reshape(-1, 3, 6).softmax(-1)
        shapes = scene_logits[:, 18:30].reshape(-1, 3, 4).softmax(-1)
        axis = scene_logits[:, 30:32].softmax(-1)
        result = scene_logits.new_full((len(scene_logits), len(specs), 14), -1e9)
        upper = torch.triu(torch.ones(3, 3, device=scene_logits.device), diagonal=1)
        for q, spec in enumerate(specs):
            if spec["task"] == "color":
                weight = shapes[:, :, list(sr.SHAPES).index(spec["reference"])]
                p = (weight[:, :, None] * colors).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-8)
            elif spec["task"] == "shape":
                weight = colors[:, :, list(sr.COLORS).index(spec["reference"])]
                p = (weight[:, :, None] * shapes).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-8)
            else:
                a, b = [list(sr.SHAPES).index(spec[k]) for k in ("reference", "other")]
                joint = shapes[:, :, a, None] * shapes[:, None, :, b]
                before = (joint * upper).sum((1, 2)); after = (joint * upper.T).sum((1, 2))
                norm = (before + after).clamp_min(1e-8)
                p = torch.stack((axis[:, 0] * before / norm, axis[:, 0] * after / norm,
                                 axis[:, 1] * before / norm, axis[:, 1] * after / norm), -1)
            result[:, q, spec["answers"]] = p.clamp_min(1e-8).log()
        return result

    def forward(self, state, specs):
        return self.answer_logits(self.decode(state), specs)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decision = StateEncoder(), FactoredDecision()

    def forward(self, tokens, lengths, specs):
        z = self.encoder(tokens, lengths)
        return self.decision(z, specs), z


def sigreg(x, step):
    # Epps-Pulley equation from stable-pretraining ab836bf, methods/lejepa.py.
    # 128 directions instead of 1024; same batch-scaled EP and 17-point quadrature.
    generator = torch.Generator(device=x.device).manual_seed(step)
    a = F.normalize(torch.randn(DIM, 128, generator=generator, device=x.device), dim=0)
    t = torch.linspace(0, 3, 17, device=x.device); phi = (-.5 * t.square()).exp()
    w = torch.full_like(t, 2 * 3 / 16); w[[0, -1]] *= .5
    angle = (x @ a)[:, :, None] * t
    error = (angle.cos().mean(0) - phi).square() + angle.sin().mean(0).square()
    return (error @ (w * phi)).mean() * len(x)


def jepa(z, step):
    center = z.mean(0, keepdim=True)
    inv = (z - center).square().mean()
    ep = sigreg(z.flatten(0, 1), step)
    return inv + .02 * ep, inv, ep


def geometry(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(0), y - y.mean(0)
    cka = (x.T @ y).square().sum() / ((x.T @ x).norm() * (y.T @ y).norm()).clamp_min(1e-15)
    neighbors = []
    for v in (x, y):
        v = F.normalize(v, dim=-1); sim = v @ v.T
        sim.fill_diagonal_(-float("inf")); neighbors.append(sim.topk(10, dim=-1).indices)
    overlap = (neighbors[0][:, :, None] == neighbors[1][:, None, :]).any(-1).double().mean()
    return {"biased_linear_CKA": float(cka), "centered_cosine_mutual10NN": float(overlap)}


def load_banks(out):
    return {m: {k: v.to("cuda") for k, v in torch.load(out / f"{m}-tokens.pt", weights_only=True).items()}
            for m in ("text", "image")}


def labels(rows, specs):
    gold = torch.full((len(rows), len(specs)), -1, dtype=torch.long, device="cuda")
    for item in sr.examples(rows, specs):
        gold[item["scene"], item["query"]] = item["gold"]
    tasks = torch.tensor([sr.TASKS.index(s["task"]) for s in specs], device="cuda")
    seen = torch.tensor([s["key"] not in sr.HOLDOUT for s in specs], device="cuda")
    return gold, tasks, seen


def score(logits, gold, tasks, seen, group="seen"):
    valid = (gold >= 0) & (seen if group == "seen" else ~seen)[None]
    loss = -logits.log_softmax(-1).gather(-1, gold.clamp_min(0)[..., None])[..., 0]
    correct = logits.argmax(-1) == gold
    nll = torch.stack([loss[valid & (tasks == t)[None]].mean() for t in range(3)]).mean()
    acc = torch.stack([correct[valid & (tasks == t)[None]].float().mean() for t in range(3)]).mean()
    return nll, acc


def fit(out, arm, seed, epochs=EPOCHS, pilot=False):
    rows, specs = data(); banks = load_banks(out); gold, tasks, seen = labels(rows, specs)
    torch.manual_seed(seed); model = Model().cuda()
    initial = hashlib.sha256()
    for key, value in model.state_dict().items():
        initial.update(key.encode()); initial.update(value.detach().cpu().numpy().tobytes())
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    directory = out / ("pilot" if pilot else f"{arm}-{seed}"); directory.mkdir(exist_ok=True)
    best, best_state, history = float("inf"), None, []
    started = time.monotonic(); digests = []; step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator(device="cuda").manual_seed(seed * 1000 + epoch)
        order = torch.randperm(1024, generator=generator, device="cuda")
        sums = np.zeros(4)
        for ids in order.split(BATCH):
            step += 1
            digests.append(hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
            main = banks["text"]
            z = model.encoder(main["values"][ids], main["lengths"][ids])
            ce, _ = score(model.decision(z, specs), gold[ids], tasks, seen)
            auxiliary, inv_sum, ep_sum = 0., 0., 0.
            for m in ("text", "image"):
                indices = len(rows) + ids[:, None] * 4 + torch.arange(4, device="cuda")[None]
                b = banks[m]
                all_z = model.encoder(b["values"][indices.flatten()], b["lengths"][indices.flatten()]).reshape(len(ids), 4, DIM)
                i, j = pair_indices(seed, step, len(ids), "good" if arm == "good" else "bad", "cuda")
                views = torch.stack((all_z[torch.arange(len(ids), device="cuda"), i], all_z[torch.arange(len(ids), device="cuda"), j]))
                aux, inv, ep = jepa(views, step * 2 + int(m == "image"))
                auxiliary = auxiliary + aux / 2; inv_sum += float(inv.detach()) / 2; ep_sum += float(ep.detach()) / 2
            loss = ce + (0. if arm == "ce" else AUX_WEIGHT) * auxiliary
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            sums += [float(ce.detach()), float(auxiliary.detach()), inv_sum, ep_sum]
        model.eval()
        with torch.no_grad():
            ids = torch.arange(1024, 1280, device="cuda")
            logits, _ = model(banks["text"]["values"][ids], banks["text"]["lengths"][ids], specs)
            val_nll, val_acc = score(logits, gold[ids], tasks, seen)
        entry = {"epoch": epoch, "validation_seen_nll": float(val_nll), "validation_seen_accuracy": float(val_acc),
                 "train_ce": sums[0] / 8, "auxiliary": sums[1] / 8, "invariance": sums[2] / 8, "sigreg": sums[3] / 8}
        history.append(entry)
        if float(val_nll) < best:
            best = float(val_nll); best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0:
            print("FIT", arm, seed, entry, "seconds",round(time.monotonic() - started, 1),flush=True)
            sr.dump(directory / "progress.json", entry)
    torch.save(best_state, directory / "model.pt")
    sr.dump(directory / "fit.json", {"arm": arm, "seed": seed, "best_epoch": best_epoch,
             "best_validation_nll": best, "history": history, "seconds": time.monotonic() - started,
             "train_step_digests": digests, "trainable_parameters": sum(p.numel() for p in model.parameters()),
             "initialization_sha256": initial.hexdigest(), "source_sha256": sha(__file__),
             "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "gradient_clip_norm": 1.}})
    del model, banks, opt
    torch.cuda.empty_cache()


def freeze(out):
    assert (out / "pilot/fit.json").exists()
    pilot = read(out / "pilot/fit.json")
    selected = pilot["history"][pilot["best_epoch"] - 1]
    assert selected["validation_seen_accuracy"] >= .95, "Text capability gate not met; no image transfer claim"
    original = ast.parse((out / "prepare-source.py").read_text())
    current = ast.parse(Path(__file__).read_text())
    names = ("prepare", "data", "caption_observation", "text_view", "image_view")
    for name in names:
        a = next(n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        b = next(n for n in current.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        assert ast.dump(a) == ast.dump(b), f"Frontend changed: {name}"
    assert sha(out / "prepare-source.py") == read(out / "view-plan.json")["source_sha256"]
    assert sha(out / "scenes.json") == sha(PARENT / "scenes.json")
    assert sha(out / "query-specs.json") == sha(PARENT / "query-specs.json")
    Path(out / "experiment-source.py").write_text(Path(__file__).read_text())
    sr.dump(out / "plan.json", {"source_sha256": sha(__file__), "seeds": SEEDS, "arms": ARMS,
            "epochs": EPOCHS, "batch_worlds": BATCH, "aux_weight": AUX_WEIGHT, "state_dim": DIM,
            "data_sha256": {str(PARENT / n): sha(PARENT / n) for n in ("scenes.json", "query-specs.json", "test-items.json")},
            "banks_sha256": {m: sha(out / f"{m}-tokens.pt") for m in ("text", "image")},
            "frontend_sources_sha256": {p: sha(p) for p in ("qwen_decision/v3/modeling.py", "qwen_decision/v3/state_rotation.py", str(out / "prepare-source.py"))},
            "qwen_files_sha256": {p.name: sha(p) for p in BASE.iterdir() if p.is_file()},
            "frontend_ast_unchanged": list(names), "pilot_selected_validation_accuracy": selected["validation_seen_accuracy"],
            "official_sigreg": "https://github.com/galilai-group/stable-pretraining/blob/ab836bf/stable_pretraining/methods/lejepa.py",
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(), "float32_matmul_precision": torch.get_float32_matmul_precision()},
            "architecture": "frozen original noncausal Qwen token frontend; shared learned attention state encoder64; factored decoder; NO state bypass, posthoc PCA or new probe",
            "query": "known synthetic grammar via query metadata; shared differentiable object binding; no object-property gold loss; not open-ended natural-language Decision",
            "ce": "same 8922 text/seen records; all gold only through actual state bottleneck",
            "jepa": "within-modality mean-of-2-view MSE + .02 batch-scaled EP; 128 directions/17nodes; no direct image-text loss",
            "controls": "all arms encode identical 4 views per modality, even CE-only; only loss weights/pairing differ",
            "selection": "best validation seen text NLL; no image/unseen/test selection",
            "evaluation": "freeze same trained decoder; train-only centered orthogonal Procrustes; raw/mean-only/R/20 wrong pairs; secondary train-only global scale with R; scene x seed bootstrap5000",
            "gates": "seen text>=.95, unseen text>=.90; report all arms regardless; no attribution of unseen image failure if text gate fails",
            "limitations": ["shared adapter on frozen jointly pretrained Qwen, not independently pretrained modality encoders", "no eigengap or population optimum established", "few deterministic synthetic views; all arms same augmented world dataset", "previously viewed scene split; exploratory", "structured synthetic question interface"]})


@torch.inference_mode()
def analyze(out):
    rows, specs = data(); banks = load_banks(out); gold, tasks, seen = labels(rows, specs)
    arrays, alignment, diagnostics = {}, {}, {}
    test_items = read(PARENT / "test-items.json")
    for arm in ARMS:
        arrays[arm] = {k: [] for k in ("text", "raw", "mean", "rotation", "similarity", "shuffle")}
        alignment[arm], diagnostics[arm] = [], []
        for seed in SEEDS:
            directory = out / f"{arm}-{seed}"
            model = Model().cuda(); model.load_state_dict(torch.load(directory / "model.pt", weights_only=True)); model.eval().requires_grad_(False)
            z = {}
            for m in ("text", "image"):
                z[m] = torch.cat([model.encoder(banks[m]["values"][i:i + 128], banks[m]["lengths"][i:i + 128])
                                  for i in range(0, len(rows), 128)])
            mi, mt = z["image"][:1024].mean(0), z["text"][:1024].mean(0)
            r = sr.orthogonal((z["image"][:1024] - mi).cpu(), (z["text"][:1024] - mt).cpu()).cuda()
            xr, yt = (z["image"][:1024] - mi) @ r, z["text"][:1024] - mt
            scale = (xr.double() * yt.double()).sum() / xr.double().square().sum()
            variants = {"text": z["text"], "raw": z["image"], "mean": z["image"] - mi + mt,
                        "rotation": (z["image"] - mi) @ r + mt,
                        "similarity": (z["image"] - mi) @ r * scale.float() + mt}
            record = {}
            for name, state in variants.items():
                logits = model.decision(state[1280:], specs)
                array = scene_records(logits, test_items, gold[1280:])
                arrays[arm][name].append(array); record[name] = array
            shuffled = []
            for n in range(20):
                generator = torch.Generator().manual_seed(9001 + n)
                order = torch.randperm(1024, generator=generator).cuda()
                perm = torch.empty_like(order); perm[order] = order.roll(1)
                wrong = sr.orthogonal((z["image"][:1024] - mi).cpu(), (z["text"][perm] - mt).cpu()).cuda()
                logits = model.decision(((z["image"] - mi) @ wrong + mt)[1280:], specs)
                shuffled.append(scene_records(logits, test_items, gold[1280:]))
            arrays[arm]["shuffle"].append(np.mean(shuffled, axis=0)); record["all_shuffles"] = np.stack(shuffled)
            centered_image, centered_text = z["image"] - mi, z["text"] - mt
            metrics = sr.alignment_metrics((centered_image @ r).cpu(), centered_text.cpu(), list(range(1280, 1536)))
            metrics.update(geometry(z["image"][1280:], z["text"][1280:]))
            metrics["train_only_global_scale"] = float(scale)
            alignment[arm].append(metrics)
            diag = {}
            for m in z:
                c = z[m][:1024].double(); singular = torch.linalg.svdvals(c - c.mean(0)); v = singular.square(); p = v / v.sum()
                diag[m] = {"rms": float(c.square().mean().sqrt()), "mean_coordinate_std": float(c.std(0).mean()),
                           "effective_rank": float((-p * p.clamp_min(1e-15).log()).sum().exp())}
            diagnostics[arm].append(diag)
            torch.save({"states": {m: v.cpu() for m, v in z.items()}, "rotation": r.cpu(), "image_mean": mi.cpu(), "text_mean": mt.cpu()}, directory / "evaluation-states.pt")
            np.savez_compressed(directory / "test-records.npz", **record)
            print("EVALUATED",arm,seed,metrics,flush=True)
    arrays = {a: {k: np.stack(v) for k, v in variants.items()} for a, variants in arrays.items()}
    scores = {a: {k: sr.summarize(v) for k, v in variants.items()} for a, variants in arrays.items()}
    comparisons = {}
    for ref in ("ce", "bad"):
        for variant in ("text", "rotation"):
            rng = np.random.default_rng(8153); values = []
            for _ in range(5000):
                seed_ids = rng.integers(3, size=3); scenes = rng.integers(256, size=256)
                a = sr.summarize(arrays["good"][variant][seed_ids][:, scenes]); b = sr.summarize(arrays[ref][variant][seed_ids][:, scenes])
                values.append([[a[g][m] - b[g][m] for m in ("macro_accuracy", "macro_nll")] for g in ("seen", "unseen")])
            comparisons[f"good-minus-{ref}:{variant}"] = {"point": [[scores["good"][variant][g][m] - scores[ref][variant][g][m] for m in ("macro_accuracy", "macro_nll")] for g in ("seen", "unseen")],
                "ci95_group_metric": np.quantile(values, [.025, .975], axis=0).tolist()}
    for seed in SEEDS:
        fits = [read(out / f"{arm}-{seed}/fit.json") for arm in ARMS]
        digests = [fit["train_step_digests"] for fit in fits]
        assert digests[0] == digests[1] == digests[2]
        assert len({fit["initialization_sha256"] for fit in fits}) == 1
    sr.dump(out / "report.json", {"complete": True, "decision": scores, "alignment": alignment,
             "diagnostics": diagnostics, "comparisons": comparisons, "matched_step_orders": True,
             "per_seed_decision": {a: {str(seed): {k: sr.summarize(v[i:i + 1]) for k, v in variants.items()}
                                      for i, seed in enumerate(SEEDS)} for a, variants in arrays.items()},
             "text_capability_gates": {a: {"seen95": scores[a]["text"]["seen"]["macro_accuracy"] >= .95,
                                          "unseen90": scores[a]["text"]["unseen"]["macro_accuracy"] >= .9} for a in ARMS}})


def scene_records(logits, items, gold):
    logp = logits.double().log_softmax(-1).cpu().numpy(); pred = logits.argmax(-1).cpu().numpy()
    expected = gold.cpu().numpy()
    array = np.zeros((256, 2, 3, 3), dtype=np.float64)
    for item in items:
        s, q = item["scene"] - 1280, item["query"]
        assert expected[s, q] == item["gold"]
        array[s, int(item["unseen"]), item["task"]] += [int(pred[s, q] == item["gold"]), -logp[s, q, item["gold"]], 1]
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "freeze", "run", "report"))
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_float32_matmul_precision("high")
    if args.command == "prepare": prepare(args.out)
    elif args.command == "pilot": fit(args.out, "ce", 379, pilot=True)
    elif args.command == "freeze": freeze(args.out)
    elif args.command == "run":
        assert read(args.out / "plan.json")["source_sha256"] == sha(__file__)
        for seed in SEEDS:
            for arm in ARMS:
                if not (args.out / f"{arm}-{seed}/fit.json").exists(): fit(args.out, arm, seed)
        analyze(args.out)
    else: analyze(args.out)


if __name__ == "__main__": main()
