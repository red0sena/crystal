"""Frozen-state Decision refit, with paired head seeds and calibrated noise.

No encoder training, image answer loss, test selection, or seed exclusion.
Three conditions separate head refitting, unit normalization, and noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from qwen_decision.v3 import jepa_decision as exp
from qwen_decision.v3 import jepa_decision_verify as audit

PARENT = Path("runs/rotation-state/jepa-decision-seeds10-v1")
ROOT = Path("runs/rotation-state/jepa-head-refit-v1")
SEEDS = (401, 419, 443, 461, 479, 503, 521, 541, 563, 587)
HEAD_SEEDS = (701, 719, 733)
CONDITIONS = ("raw_clean", "unit_clean", "unit_noise")
EPOCHS, BATCH, BOOTSTRAPS = 400, 256, 5000
COMBINATIONS = [(c, h) for c in CONDITIONS for h in HEAD_SEEDS]


def digest_state(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Heads(nn.Module):
    """Independent heads in one batch; no parameters or optimizer moments shared."""
    def __init__(self, seeds=HEAD_SEEDS, conditions=CONDITIONS):
        super().__init__()
        original = []
        for seed in seeds:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                original.append(exp.FactoredDecision().state_dict())
        weights = [x for _ in conditions for x in original]
        self.w1 = nn.Parameter(torch.stack([x["decode.0.weight"] for x in weights]))
        self.b1 = nn.Parameter(torch.stack([x["decode.0.bias"] for x in weights]))
        self.w2 = nn.Parameter(torch.stack([x["decode.2.weight"] for x in weights]))
        self.b2 = nn.Parameter(torch.stack([x["decode.2.bias"] for x in weights]))

    def forward(self, states):
        hidden = F.gelu(states @ self.w1.transpose(-1, -2) + self.b1[:, None])
        return answer_logits(hidden @ self.w2.transpose(-1, -2) + self.b2[:, None])

    def single_state(self, index):
        return {"decode.0.weight": self.w1[index].detach().cpu().clone(),
                "decode.0.bias": self.b1[index].detach().cpu().clone(),
                "decode.2.weight": self.w2[index].detach().cpu().clone(),
                "decode.2.bias": self.b2[index].detach().cpu().clone()}


def answer_logits(raw):
    # Same factored algebra as the original decoder, vectorized over queries.
    colors = raw[..., :18].reshape(*raw.shape[:-1], 3, 6).softmax(-1)
    shapes = raw[..., 18:30].reshape(*raw.shape[:-1], 3, 4).softmax(-1)
    axis = raw[..., 30:32].softmax(-1)
    pc = shapes.transpose(-1, -2) @ colors / shapes.sum(-2)[..., :, None].clamp_min(1e-8)
    ps = colors.transpose(-1, -2) @ shapes / colors.sum(-2)[..., :, None].clamp_min(1e-8)
    upper = torch.triu(torch.ones(3, 3, device=raw.device, dtype=raw.dtype), diagonal=1)
    before = shapes.transpose(-1, -2) @ upper @ shapes
    after = before.transpose(-1, -2)
    pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
    a, b = zip(*pairs)
    before, after = before[..., list(a), list(b)], after[..., list(a), list(b)]
    norm = (before + after).clamp_min(1e-8)
    pr = torch.stack((axis[..., 0, None] * before / norm, axis[..., 0, None] * after / norm,
                      axis[..., 1, None] * before / norm, axis[..., 1, None] * after / norm), -1)
    pc, ps, pr = [p.clamp_min(1e-8).log() for p in (pc, ps, pr)]
    return torch.cat((F.pad(pc, (0, 8), value=-1e9), F.pad(ps, (6, 4), value=-1e9),
                      F.pad(pr, (10, 0), value=-1e9)), -2)


def losses(logits, gold, specs, seen=True):
    valid = gold >= 0
    task = torch.tensor([exp.sr.TASKS.index(s["task"]) for s in specs], device=gold.device)
    allowed = torch.tensor([(s["key"] not in exp.sr.HOLDOUT) == seen for s in specs], device=gold.device)
    nll = -logits.log_softmax(-1).gather(-1, gold.clamp_min(0)[None, ..., None].expand(len(logits), -1, -1, 1))[..., 0]
    return torch.stack([nll[:, valid & allowed & (task == t)].mean(-1) for t in range(3)]).mean(0)


def clip_heads(model, maximum=1.):
    norm = torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    scale = (maximum / (norm + 1e-6)).clamp(max=1.)
    for p in model.parameters():
        p.grad.mul_(scale.reshape(-1, *([1] * (p.ndim - 1))))
    return norm


def gold_table(rows, specs, device):
    # Use the independent world-to-answer implementation from the prior audit.
    gold = torch.full((len(rows), len(specs)), -1, device=device, dtype=torch.long)
    for index, row in enumerate(rows):
        shapes = {s for c, s in row["world"]["objects"]}
        colors = {c for c, s in row["world"]["objects"]}
        for q, spec in enumerate(specs):
            valid = (spec["reference"] in shapes if spec["task"] == "color" else
                     spec["reference"] in colors if spec["task"] == "shape" else
                     spec["reference"] in shapes and spec["other"] in shapes)
            if valid:
                gold[index, q] = audit.gold_from_world(row, spec)
    return gold


def transformed_states(saved):
    text, image = saved["states"]["text"].float(), saved["states"]["image"].float()
    mt, mi = text[:1024].mean(0), image[:1024].mean(0)
    torch.testing.assert_close(mt, saved["text_mean"]); torch.testing.assert_close(mi, saved["image_mean"])
    rotation = saved["rotation"].float()
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(exp.DIM), atol=2e-5, rtol=0)
    raw_image = (image - mi) @ rotation + mt
    unit_text = F.normalize(text - mt, dim=-1)
    unit_image = F.normalize((image - mi) @ rotation, dim=-1)
    # First half of validation is calibration only; second half selects heads.
    cosine = float((unit_text[1024:1152] * unit_image[1024:1152]).sum(-1).mean())
    assert 0 < cosine <= 1, ("Noise rule requires positive calibration cosine", cosine)
    sigma = math.sqrt(cosine ** -2 - 1)
    return {"text": text, "rotation": raw_image}, {"text": unit_text, "rotation": unit_image}, cosine, sigma


def training_inputs(raw, unit, ids, step, sigma, device):
    states = torch.stack([raw[ids] if c == "raw_clean" else unit[ids] for c, _ in COMBINATIONS])
    for j, (condition, head_seed) in enumerate(COMBINATIONS):
        if condition == "unit_noise":
            generator = torch.Generator(device=device).manual_seed(head_seed * 1000003 + step)
            noise = torch.randn(states[j].shape, generator=generator, device=device)
            states[j] += noise * (sigma / math.sqrt(exp.DIM))
    return states


def prepare(root):
    parent_plan = exp.read(PARENT / "plan.json")
    assert tuple(parent_plan["seeds"]) == SEEDS
    assert exp.read(PARENT / "verification.json")["complete"]
    root.mkdir(parents=True, exist_ok=False)
    inputs = {str(PARENT / name): exp.sha(PARENT / name) for name in ("plan.json", "report.json", "verification.json", "scenes.json", "query-specs.json")}
    for arm in exp.ARMS:
        for seed in SEEDS:
            for name in ("model.pt", "evaluation-states.pt", "fit.json", "test-records.npz"):
                path = PARENT / f"{arm}-{seed}" / name
                inputs[str(path)] = exp.sha(path)
    shutil.copyfile(__file__, root / "experiment-source.py")
    plan = {"source_sha256": exp.sha(__file__), "inputs_sha256": inputs, "encoder_seeds": list(SEEDS),
            "head_seeds": list(HEAD_SEEDS), "arms": list(exp.ARMS), "conditions": list(CONDITIONS),
            "head_count": len(SEEDS) * len(HEAD_SEEDS) * len(exp.ARMS) * len(CONDITIONS),
            "epochs": EPOCHS, "batch_worlds": BATCH, "steps": EPOCHS * (1024 // BATCH),
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "per_head_clip": 1.},
            "training": "same 1024 worlds / 8922 seen text questions; task macro CE; frozen cached states; no image gold",
            "initialization": "three fixed head seeds shared across every arm/encoder/condition; each head independently optimized",
            "selection": "minimum clean seen text NLL on validation worlds1152:1280; no image/test/unseen selection",
            "calibration": "mean cosine on validation worlds1024:1152, labels unused; rotation and centering fixed from train1024",
            "noise": "unit-centered state + N(0,sigma^2/64 I); sigma=sqrt(c^-2-1); relative RMS norm, no post-noise normalization",
            "control": "raw_clean isolates head refit; unit_clean vs raw_clean adds normalization; unit_noise vs unit_clean adds noise",
            "inference": "clean inputs for text and train-rotation image, same head; no ensemble, average of individual model scores",
            "stop": "all 30 encoder checkpoints x three head seeds x three conditions, 400 epochs, retain failures, no exclusions",
            "statistics": "5000 paired crossed bootstrap draws: encoder seeds, common head seeds and common test worlds independently resampled; preserves shared head initialization across encoders",
            "scope": "post-hoc diagnostic on an already examined synthetic split; existing encoders were jointly CE+JEPA trained; not independent JEPA theorem reproduction",
            "paper": "https://arxiv.org/pdf/2610.39302v1#page=13"}
    exp.sr.dump(root / "plan.json", plan)
    exp.sr.dump(root / "status.json", {"state": "prepared", "heads_total": plan["head_count"]})


def fit_group(root, arm, seed):
    directory = root / f"{arm}-{seed}"; directory.mkdir(exist_ok=True)
    assert not (directory / "fit.json").exists()
    rows, specs = exp.data()
    saved = torch.load(PARENT / f"{arm}-{seed}/evaluation-states.pt", map_location="cpu", weights_only=True)
    raw, unit, cosine, sigma = transformed_states(saved)
    device = torch.device("cuda")
    raw, unit = [{k: v.to(device) for k, v in x.items()} for x in (raw, unit)]
    gold = gold_table(rows, specs, device)
    model = Heads().to(device); initial = [digest_state(model.single_state(i)) for i in range(len(COMBINATIONS))]
    assert initial[:3] == initial[3:6] == initial[6:9]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best = np.full(len(COMBINATIONS), np.inf); best_epoch = np.zeros(len(COMBINATIONS), dtype=int)
    best_states = [None] * len(COMBINATIONS); history = []; orders = []; step = 0
    started = time.monotonic()
    validation = torch.stack([raw["text"][1152:1280] if c == "raw_clean" else unit["text"][1152:1280] for c, _ in COMBINATIONS])
    for epoch in range(1, EPOCHS + 1):
        # The same world order is used across all encoder arms and head conditions.
        order = torch.randperm(1024, generator=torch.Generator().manual_seed(991 * epoch)).to(device)
        orders.append(hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest())
        train = torch.zeros(len(COMBINATIONS), device=device)
        for ids in order.split(BATCH):
            step += 1
            inputs = training_inputs(raw["text"], unit["text"], ids, step, sigma, device)
            loss = losses(model(inputs), gold[ids], specs)
            assert bool(torch.isfinite(loss).all())
            optimizer.zero_grad(set_to_none=True); loss.sum().backward(); clip_heads(model); optimizer.step()
            train += loss.detach() / (1024 // BATCH)
        with torch.no_grad():
            nll = losses(model(validation), gold[1152:1280], specs).cpu().numpy()
        improved = nll < best
        for i in np.flatnonzero(improved):
            best[i], best_epoch[i], best_states[i] = nll[i], epoch, model.single_state(i)
        history.append({"epoch": epoch, "train_ce": train.cpu().tolist(), "validation_seen_nll": nll.tolist()})
        if epoch == 1 or epoch % 50 == 0:
            print("FIT", arm, seed, epoch, "best", np.round(best, 4).tolist(), "seconds", round(time.monotonic() - started, 1), flush=True)
    torch.save(best_states, directory / "heads.pt")
    exp.sr.dump(directory / "fit.json", {"arm": arm, "encoder_seed": seed, "conditions_and_head_seeds": COMBINATIONS,
               "initialization_sha256": initial, "best_epoch": best_epoch.tolist(), "best_nll": best.tolist(),
               "calibration_cosine": cosine, "sigma": sigma, "history": history, "epoch_orders_sha256": orders,
               "seconds": time.monotonic() - started, "source_sha256": exp.sha(__file__)})
    with torch.no_grad():
        for i, state in enumerate(best_states):
            model.w1[i].copy_(state["decode.0.weight"]); model.b1[i].copy_(state["decode.0.bias"])
            model.w2[i].copy_(state["decode.2.weight"]); model.b2[i].copy_(state["decode.2.bias"])
        logits = {v: model(torch.stack([raw[v][1280:] if c == "raw_clean" else unit[v][1280:] for c, _ in COMBINATIONS])).cpu()
                  for v in ("text", "rotation")}
    torch.save(logits, directory / "test-logits.pt")
    records = {v: records_from_logits(x, gold[1280:].cpu(), specs) for v, x in logits.items()}
    np.savez_compressed(directory / "test-records.npz", **records)


def records_from_logits(logits, gold, specs):
    if logits.ndim == 3: logits = logits[None]
    logp = logits.double().log_softmax(-1).numpy(); pred = logits.argmax(-1).numpy(); expected = gold.cpu().numpy()
    result = np.zeros((len(logits), 256, 2, 3, 3))
    for q, spec in enumerate(specs):
        group, task = int(spec["key"] in exp.sr.HOLDOUT), exp.sr.TASKS.index(spec["task"])
        valid = np.flatnonzero(expected[:, q] >= 0)
        answer = expected[valid, q]
        result[:, valid, group, task, 0] += pred[:, valid, q] == answer
        result[:, valid, group, task, 1] -= logp[:, valid, q, answer]
        result[:, valid, group, task, 2] += 1
    return result


def arrays(root):
    result = {}
    for arm in exp.ARMS:
        for variant in ("text", "rotation"):
            for ci, condition in enumerate(CONDITIONS):
                result[arm, condition, variant] = np.stack([np.load(root / f"{arm}-{s}/test-records.npz")[variant][ci * 3:(ci + 1) * 3] for s in SEEDS])
            original = np.stack([np.load(PARENT / f"{arm}-{s}/test-records.npz")[variant] for s in SEEDS])
            result[arm, "original", variant] = np.repeat(original[:, None], len(HEAD_SEEDS), axis=1)
    return result


def bootstrap_weights(draws=BOOTSTRAPS):
    rng = np.random.default_rng(202611041)
    wh, ws = np.zeros((draws, len(SEEDS), len(HEAD_SEEDS))), np.zeros((draws, 256))
    for b in range(draws):
        encoders = np.bincount(rng.integers(len(SEEDS), size=len(SEEDS)), minlength=len(SEEDS))
        heads = np.bincount(rng.integers(len(HEAD_SEEDS), size=len(HEAD_SEEDS)), minlength=len(HEAD_SEEDS))
        wh[b] = encoders[:, None] * heads[None]
        ws[b] = np.bincount(rng.integers(256, size=256), minlength=256)
    return wh, ws


def bootstrap_scores(array, wh, ws):
    # First sum the common test-scene draw, then the crossed encoder/head draw.
    by_scene = ws @ array.transpose(2, 0, 1, 3, 4, 5).reshape(256, -1)
    by_scene = by_scene.reshape(len(ws), len(SEEDS), len(HEAD_SEEDS), 2, 3, 3)
    sums = (by_scene * wh[..., None, None, None]).sum((1, 2))
    return (sums[..., :2] / sums[..., 2:3]).mean(-2)


def analyze(root):
    data = arrays(root); wh, ws = bootstrap_weights()
    scores = {":".join(k): exp.sr.summarize(x.reshape(-1, *x.shape[2:])) for k, x in data.items()}
    pairs = []
    for arm in exp.ARMS:
        pairs.extend([((arm, "raw_clean"), (arm, "original")), ((arm, "unit_clean"), (arm, "raw_clean")),
                      ((arm, "unit_noise"), (arm, "unit_clean"))])
    for condition in CONDITIONS:
        pairs.extend([(("good", condition), ("ce", condition)), (("good", condition), ("bad", condition))])
    boot = {k: bootstrap_scores(x, wh, ws) for k, x in data.items()}
    comparisons = {}
    for left, right in pairs:
        for variant in ("text", "rotation"):
            lk, rk = (*left, variant), (*right, variant)
            delta = boot[lk] - boot[rk]
            point = [[scores[":".join(lk)][g][m] - scores[":".join(rk)][g][m]
                      for m in ("macro_accuracy", "macro_nll")] for g in ("seen", "unseen")]
            comparisons[f'{":".join(left)}-minus-{":".join(right)}:{variant}'] = {
                "point": point, "ci95": np.quantile(delta, [.025, .975], axis=0).tolist()}
    per_encoder = {":".join(k): {str(seed): exp.sr.summarize(x[i]) for i, seed in enumerate(SEEDS)} for k, x in data.items()}
    gates = {key: {g: sum(value[g]["macro_accuracy"] >= threshold for value in per.values())
                   for g, threshold in (("seen", .95), ("unseen", .90))}
             for key, per in per_encoder.items() if key.endswith(":text")}
    exp.sr.dump(root / "report.json", {"complete": True, "scores": scores, "per_encoder": per_encoder,
                "comparisons": comparisons, "encoder_mean_text_gate_counts": gates,
                "encoder_seeds": list(SEEDS), "head_seeds": list(HEAD_SEEDS),
                "calibration": {f"{a}-{s}": {k: exp.read(root / f"{a}-{s}/fit.json")[k] for k in ("calibration_cosine", "sigma")}
                                for a in exp.ARMS for s in SEEDS},
                "bootstrap_draws": BOOTSTRAPS, "scope": exp.read(root / "plan.json")["scope"]})


def verify(root):
    plan = exp.read(root / "plan.json"); assert plan["source_sha256"] == exp.sha(__file__)
    for path, value in plan["inputs_sha256"].items(): assert exp.sha(path) == value, path
    rows, specs = exp.data(); expected = gold_table(rows[1280:], specs, "cpu")
    checks = {}; max_error = 0.; tied = 0; records_checked = 0
    for arm in exp.ARMS:
        for seed in SEEDS:
            directory = root / f"{arm}-{seed}"; fit = exp.read(directory / "fit.json")
            assert fit["source_sha256"] == plan["source_sha256"] and len(fit["history"]) == EPOCHS
            assert len(fit["epoch_orders_sha256"]) == EPOCHS
            assert fit["initialization_sha256"][:3] == fit["initialization_sha256"][3:6] == fit["initialization_sha256"][6:9]
            for i in range(len(COMBINATIONS)):
                vals = [h["validation_seen_nll"][i] for h in fit["history"]]
                assert fit["best_epoch"][i] == int(np.argmin(vals)) + 1 and fit["best_nll"][i] == min(vals)
            saved = torch.load(PARENT / f"{arm}-{seed}/evaluation-states.pt", map_location="cpu", weights_only=True)
            raw, unit, cosine, sigma = transformed_states(saved)
            assert abs(cosine - fit["calibration_cosine"]) < 1e-7 and abs(sigma - fit["sigma"]) < 1e-6
            # Independently refit R with NumPy on train rows only.
            x, y = [saved["states"][m][:1024].numpy().astype(float) for m in ("image", "text")]
            x -= x.mean(0); y -= y.mean(0); u, _, vt = np.linalg.svd(x.T @ y)
            assert np.max(np.abs(u @ vt - saved["rotation"].numpy())) < 2e-4
            heads = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
            logits = torch.load(directory / "test-logits.pt", map_location="cpu", weights_only=True)
            records = np.load(directory / "test-records.npz")
            group_error = 0.
            for i, (condition, head_seed) in enumerate(COMBINATIONS):
                head = exp.FactoredDecision().cuda().eval(); head.load_state_dict(heads[i])
                for variant in ("text", "rotation"):
                    state = (raw if condition == "raw_clean" else unit)[variant][1280:]
                    with torch.no_grad(): ref = head(state.cuda(), specs).double().log_softmax(-1).cpu()
                    actual = logits[variant][i].double().log_softmax(-1)
                    for q, spec in enumerate(specs):
                        valid = expected[:, q] >= 0; answers = spec["answers"]
                        error = float((ref[valid, q][:, answers] - actual[valid, q][:, answers]).abs().max())
                        group_error = max(group_error, error); assert error < 1e-4, (arm, seed, i, variant, error)
                        old, new = actual[valid, q].argmax(-1), ref[valid, q].argmax(-1)
                        changed = old != new
                        if bool(changed.any()):
                            margins = ref[valid, q].max(-1).values - ref[valid, q].gather(-1, old[:, None])[:, 0]
                            assert float(margins[changed].max()) < 1e-5
                            tied += int(changed.sum())
                    # Gold and aggregation are reconstructed from all valid world/query pairs.
                    independent = np.zeros((256, 2, 3, 3))
                    for scene in range(256):
                        for q, spec in enumerate(specs):
                            answer = int(expected[scene, q])
                            if answer < 0: continue
                            g, t = int(spec["key"] in exp.sr.HOLDOUT), exp.sr.TASKS.index(spec["task"])
                            independent[scene, g, t] += [int(actual[scene, q].argmax()) == answer,
                                                        -float(ref[scene, q, answer]), 1]
                            records_checked += 1
                    np.testing.assert_array_equal(independent[..., [0, 2]], records[variant][i, ..., [0, 2]].transpose(1, 2, 3, 0))
                    assert np.max(np.abs(independent[..., 1] - records[variant][i, ..., 1])) < .001
            checks[f"{arm}-{seed}"] = {"heads": 9, "max_candidate_logp_error": group_error}
            max_error = max(max_error, group_error)
    # Independent aggregate via seed/head averaging of task-wise counts.
    report = exp.read(root / "report.json")
    for key, array in arrays(root).items():
        sums = array.sum(2); rates = sums[..., :2] / sums[..., 2:3]
        means = rates.mean((0, 1, 3))
        for g, group in enumerate(("seen", "unseen")):
            for m, metric in enumerate(("macro_accuracy", "macro_nll")):
                assert abs(means[g, m] - report["scores"][":".join(key)][group][metric]) < 1e-10
    exp.sr.dump(root / "verification.json", {"complete": True, "checks": checks, "questions_replayed": records_checked,
                "max_candidate_logp_error": max_error, "near_tie_argmax_differences": tied,
                "report_sha256": exp.sha(root / "report.json"), "source_sha256": exp.sha(__file__),
                "scope": "all frozen input hashes; train-only NumPy R; calibration; selection epochs; all head probabilities and independent world gold/counts/NLL; aggregate means",
                "limits": "training not independently repeated; bootstrap computed by saved source, not independently replayed"})


def publish(root):
    project = Path("wiki/project"); report = exp.read(root / "report.json")
    for src, dst in (("report.json", "jepa-head-refit-results.json"), ("verification.json", "jepa-head-refit-verification.json")):
        shutil.copyfile(root / src, project / dst)
    page = project / "jepa-head-refit.md"; content = page.read_text()
    marker = "<!-- REFIT_RESULTS -->"; assert marker in content
    lines = ["## 결과", "", "정확도는10개 상태 인코더 seed×3개 판정기 seed의 개별 점수 평균이다. 앙상블이 아니다.", "",
             "| 표현 학습 | 판정기 조건 | text seen | image R seen | text unseen | image R unseen |", "|---|---|---:|---:|---:|---:|"]
    for arm in exp.ARMS:
        for condition in ("original", *CONDITIONS):
            values = [report["scores"][f"{arm}:{condition}:{v}"][g]["macro_accuracy"] * 100 for g in ("seen", "unseen") for v in ("text", "rotation")]
            lines.append(f"| {arm} | {condition} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += ["", "### 사전 고정한 비교의 차이와95% 구간", "", "정확도는%p, NLL은 원 단위다. 아래 구간은 개별95% 구간이며 다중 비교 보정은 없다.", "",
              "| 비교 | 입력 | 질문 | 정확도 차이 [95% 구간] | NLL 차이 [95% 구간] |", "|---|---|---|---:|---:|"]
    for key, comp in report["comparisons"].items():
        for g, group in enumerate(("seen", "unseen")):
            p, ci = np.array(comp["point"])[g], np.array(comp["ci95"])[:, g]
            lines.append(f"| {key.rsplit(':', 1)[0]} | {key.rsplit(':', 1)[1]} | {group} | {p[0]*100:+.2f} [{ci[0,0]*100:+.2f},{ci[1,0]*100:+.2f}] | {p[1]:+.3f} [{ci[0,1]:+.3f},{ci[1,1]:+.3f}] |")
    verification = exp.read(root / "verification.json")
    lines += ["", "### 검산과 해석 범위", "", f"{verification['questions_replayed']:,}개 질문의 판정을 원래 decoder로 대조했고 장면·과제별 정답 개수와 분모가 일치했다.",
              f"유효 후보 log 확률 최대 오차는 {verification['max_candidate_logp_error']:.3g}였다. 근접 동점의 argmax 차이는 {verification['near_tie_argmax_differences']}건이며 원 실행의 선택을 유지했다.",
              "학습을 독립 재실행하지 않았고 bootstrap 구간도 별도 구현으로 재검산하지 않았다. 모든 입력 파일의 해시와 train-only 회전·calibration·checkpoint 선택·평균은 검산했다.",
              "텍스트가 낮은 seed도 모두 포함한다. text seen95%/unseen90% 통과 수와 과제별 점수는 전체 결과 JSON에 기록했다.",
              "정규화·노이즈 비교는 이번 판정기의 진단이며 독립 인코더의 직교 동치나 일반 자연어·이미지 성능 향상을 입증하지 않는다."]
    content = content.replace("**실행 준비 중.**", "**270개 판정기 학습·평가·검산 완료.**")
    content = content.replace("**실행 중.**", "**270개 판정기 학습·평가·검산 완료.**").replace(marker, "\n".join(lines))
    page.write_text(content)


def run(root):
    plan = exp.read(root / "plan.json"); assert plan["source_sha256"] == exp.sha(__file__)
    for path, value in plan["inputs_sha256"].items(): assert exp.sha(path) == value, path
    started = time.monotonic()
    for arm in exp.ARMS:
        for seed in SEEDS:
            directory = root / f"{arm}-{seed}"
            if not (directory / "test-records.npz").exists():
                assert not (directory / "fit.json").exists(), "Partial group: retain attempt and inspect before retry"
                fit_group(root, arm, seed)
            complete = sum((root / f"{a}-{s}/test-records.npz").exists() for a in exp.ARMS for s in SEEDS)
            exp.sr.dump(root / "status.json", {"state": "training", "groups_complete": complete, "groups_total": 30,
                        "heads_complete": complete * 9, "heads_total": plan["head_count"], "seconds": time.monotonic() - started})
    for phase, action in (("analyzing", analyze), ("verifying", verify), ("saving_wiki", publish)):
        exp.sr.dump(root / "status.json", {"state": phase, "heads_complete": 270, "heads_total": 270}); action(root)
    manifest = {str(p.relative_to(root)): exp.sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("checksums.json", "status.json") and p.suffix != ".log"}
    exp.sr.dump(root / "checksums.json", manifest)
    exp.sr.dump(root / "status.json", {"state": "complete", "heads_complete": 270, "heads_total": 270, "verified": True,
                "wiki": "wiki/project/jepa-head-refit.md", "seconds": time.monotonic() - started})
    print("COMPLETE", root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze", "verify", "publish"))
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    try:
        globals()[args.command](args.out)
    except Exception:
        if args.out.exists(): exp.sr.dump(args.out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
