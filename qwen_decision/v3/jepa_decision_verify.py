"""Artifact replay with independent gold, decoder algebra and aggregation.

StateEncoder checks saved activations against token banks. Gold, FP64 decoder
probability algebra, aggregation, NumPy Procrustes and bootstrap are independent.
The original FP32 head is additionally replayed to audit rounding ties and
preserve execution's argmax; every valid candidate log-probability is compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ANSWERS = ["red", "blue", "green", "yellow", "purple", "orange", "circle", "square", "triangle", "diamond",
           "left", "right", "above", "below"]
ARMS, SEEDS = ("ce", "bad", "good"), (401, 419, 443)
VARIANTS = ("text", "raw", "mean", "rotation", "similarity")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def headline(array):
    counts = array.reshape(-1, 2, 3, 3).sum(0)
    rates = counts[..., :2] / counts[..., 2:3]
    return {group: {metric: float(rates[g, :, m].mean()) for m, metric in enumerate(("macro_accuracy", "macro_nll"))}
            for g, group in enumerate(("seen", "unseen"))}


def gold_from_world(row, spec):
    objects = row["world"]["objects"]
    if spec["task"] == "color":
        answer = next(c for c, s in objects if s == spec["reference"])
    elif spec["task"] == "shape":
        answer = next(s for c, s in objects if c == spec["reference"])
    else:
        order = [s for c, s in objects]
        before = order.index(spec["reference"]) < order.index(spec["other"])
        answer = (("left", "right") if row["world"]["axis"] == "horizontal" else ("above", "below"))[not before]
    return ANSWERS.index(answer)


def decode(states, params, specs):
    # FP32 CUDA linear layers match execution precision. Probability algebra
    # below is independently expressed, in FP64, using explicit object pairs.
    z = states.to("cuda")
    w = {k: v.to("cuda") for k, v in params.items() if k.startswith("decision.")}
    h = F.gelu(F.linear(z, w["decision.decode.0.weight"], w["decision.decode.0.bias"]))
    raw = F.linear(h, w["decision.decode.2.weight"], w["decision.decode.2.bias"]).double().cpu()
    color = raw[:, :18].reshape(-1, 3, 6).softmax(-1)
    shape = raw[:, 18:30].reshape(-1, 3, 4).softmax(-1)
    axis = raw[:, 30:].softmax(-1)
    output = []
    for spec in specs:
        if spec["task"] == "color":
            weights = shape[:, :, ANSWERS.index(spec["reference"]) - 6]
            p = torch.einsum("ni,nij->nj", weights, color) / weights.sum(-1)[:, None].clamp_min(1e-8)
        elif spec["task"] == "shape":
            weights = color[:, :, ANSWERS.index(spec["reference"])]
            p = torch.einsum("ni,nij->nj", weights, shape) / weights.sum(-1)[:, None].clamp_min(1e-8)
        else:
            a, b = (ANSWERS.index(spec[k]) - 6 for k in ("reference", "other"))
            before, after = torch.zeros(len(z), dtype=torch.float64), torch.zeros(len(z), dtype=torch.float64)
            for i in range(3):
                for j in range(3):
                    if i < j:
                        before += shape[:, i, a] * shape[:, j, b]
                    elif i > j:
                        after += shape[:, i, a] * shape[:, j, b]
            p = torch.stack([axis[:, ax] * value / (before + after).clamp_min(1e-8)
                             for ax in range(2) for value in (before, after)], -1)
        p = p.clamp_min(1e-8); p /= p.sum(-1, keepdim=True)
        output.append(p.log().numpy())
    return output


def records(logprobs, rows, specs, items, predictions=None):
    result = np.zeros((256, 2, 3, 3))
    for item in items:
        scene, query = item["scene"], item["query"]
        spec = specs[query]
        gold = gold_from_world(rows[scene], spec)
        assert gold == item["gold"]
        local_gold = spec["answers"].index(gold)
        lp = logprobs[query][scene - 1280]
        predicted = np.argmax(lp) if predictions is None else predictions[query][scene - 1280]
        result[scene - 1280, int(item["unseen"]), item["task"]] += [int(predicted == local_gold), -lp[local_gold], 1]
    return result


def audited_decode(states, params, specs, items, execution_head):
    independent = decode(states, params, specs)
    execution = execution_head(states.to("cuda"), specs).double().log_softmax(-1).cpu().numpy()
    predictions, flips, max_error, max_margin = [], 0, 0., 0.
    for q, spec in enumerate(specs):
        reference = execution[:, q, spec["answers"]]
        predictions.append(reference.argmax(-1))
        valid = np.array([item["scene"] - 1280 for item in items if item["query"] == q])
        if not len(valid): continue
        a, b = independent[q][valid], reference[valid]
        error = float(np.abs(a - b).max()); max_error = max(max_error, error)
        assert error < 1e-5, (q, error)
        different = a.argmax(-1) != b.argmax(-1)
        if different.any():
            margin = a.max(-1) - a[np.arange(len(a)), b.argmax(-1)]
            maximum = float(margin[different].max())
            assert maximum < 1e-6, (q, maximum)
            max_margin = max(max_margin, maximum); flips += int(different.sum())
    return independent, predictions, {"fp64_fp32_argmax_differences": flips,
        "max_candidate_logp_error": max_error, "max_changed_argmax_margin": max_margin,
        "accuracy_convention": "original FP32 execution argmax, all differing FP64 choices audited within 1e-6 log-probability"}


def rotation(x, y):
    u, _, vt = np.linalg.svd(x.double().numpy().T @ y.double().numpy())
    return torch.from_numpy((u @ vt).astype(np.float32))


@torch.inference_mode()
def verify(root):
    from qwen_decision.v3.jepa_decision import StateEncoder, FactoredDecision
    plan, report = read(root / "plan.json"), read(root / "report.json")
    seeds = tuple(plan["seeds"])
    assert len(seeds) == len(set(seeds)) and len(seeds) > 1
    assert plan["source_sha256"] == sha(root / "experiment-source.py") == sha("qwen_decision/v3/jepa_decision.py")
    hashes = {}
    if "extension" in plan:
        extension = plan["extension"]
        assert sha(root / "extension-source.py") == extension["source_sha256"]
        assert sha(root / "evaluation-source.py") == extension["evaluation_source_sha256"]
        parent = Path(extension["parent"])
        assert sha(parent / "report.json") == extension["parent_report_sha256"]
        for relative, expected in extension["reused_artifacts_sha256"].items():
            assert sha(parent / relative) == expected
            if Path(relative).name in ("fit.json", "model.pt"):
                assert sha(root / relative) == expected
    for section in ("data_sha256", "frontend_sources_sha256"):
        for path, expected in plan[section].items():
            assert sha(path) == expected
            hashes[path] = expected
    for m, expected in plan["banks_sha256"].items():
        path = root / f"{m}-tokens.pt"
        assert sha(path) == expected
        hashes[str(path)] = expected
    for name, expected in plan["qwen_files_sha256"].items():
        path = Path("models/qwen3.5-0.8b") / name
        assert sha(path) == expected
        hashes[str(path)] = expected
    rows, specs = read(root / "scenes.json"), read(root / "query-specs.json")
    items = read(Path("runs/rotation-state/plain-qwen-v2/test-items.json"))
    holdouts = {"color:diamond", "shape:purple", "relation:circle:triangle", "relation:triangle:circle",
                "relation:square:diamond", "relation:diamond:square"}
    assert len(items) == 3072 and len({(x["scene"], x["query"]) for x in items}) == len(items)
    for item in items:
        assert item["unseen"] == (specs[item["query"]]["key"] in holdouts)
        assert item["task"] == ("color", "shape", "relation").index(specs[item["query"]]["task"])
    banks = {m: torch.load(root / f"{m}-tokens.pt", weights_only=True) for m in ("text", "image")}
    for bank in banks.values():
        assert bank["values"].shape == (5632, 128, 1024)
        assert bool(((bank["lengths"] > 0) & (bank["lengths"] <= 128)).all())
        assert bool(torch.isfinite(bank["values"]).all())
    arrays, checks = {}, {}
    for arm in ARMS:
        arrays[arm], checks[arm] = {}, {}
        for seed in seeds:
            directory = root / f"{arm}-{seed}"
            fit = read(directory / "fit.json")
            assert fit["source_sha256"] == plan["source_sha256"]
            assert len(fit["history"]) == 200 and len(fit["train_step_digests"]) == 1600
            best = min(fit["history"], key=lambda row: row["validation_seen_nll"])
            assert best["epoch"] == fit["best_epoch"]
            params = torch.load(directory / "model.pt", weights_only=True)
            execution_head = FactoredDecision().cuda().eval()
            execution_head.load_state_dict({k.removeprefix("decision."): v for k, v in params.items() if k.startswith("decision.")})
            saved = torch.load(directory / "evaluation-states.pt", weights_only=True)
            encoder = StateEncoder().cuda().eval()
            encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in params.items() if k.startswith("encoder.")})
            state_error = 0.
            for m, bank in banks.items():
                replay = torch.cat([encoder(bank["values"][i:i + 128].cuda(), bank["lengths"][i:i + 128].cuda()).cpu()
                                    for i in range(0, 1536, 128)])
                state_error = max(state_error, float((replay - saved["states"][m]).abs().max()))
                torch.testing.assert_close(replay, saved["states"][m], atol=2e-5, rtol=2e-5)
            x, y = saved["states"]["image"], saved["states"]["text"]
            mx, my = saved["image_mean"], saved["text_mean"]
            torch.testing.assert_close(mx, x[:1024].mean(0), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(my, y[:1024].mean(0), atol=2e-6, rtol=2e-6)
            r = rotation(x[:1024] - mx, y[:1024] - my)
            torch.testing.assert_close(r, saved["rotation"], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(r.T @ r, torch.eye(64), atol=1e-6, rtol=1e-6)
            # Use execution's stored transform with CUDA matmul for exact
            # precision; its train-only fit was independently checked above.
            x, y, mx, my, r = [v.cuda() for v in (x, y, mx, my, saved["rotation"])]
            xr, yt = (x[:1024] - mx) @ r, y[:1024] - my
            scale = (xr.double() * yt.double()).sum() / xr.double().square().sum()
            states = {"text": y, "raw": x, "mean": x - mx + my, "rotation": (x - mx) @ r + my,
                      "similarity": (x - mx) @ r * scale.float() + my}
            metric = report["alignment"][arm][seeds.index(seed)]
            # Rotation-invariant geometry, independently via Gram matrices.
            xc, yc = [v[1280:].double().cpu().numpy() for v in (x, y)]
            xc -= xc.mean(0); yc -= yc.mean(0)
            kx, ky = xc @ xc.T, yc @ yc.T
            cka = np.sum(kx * ky) / np.sqrt(np.sum(kx * kx) * np.sum(ky * ky))
            assert abs(cka - metric["biased_linear_CKA"]) < 1e-10
            sets = []
            for v in (xc, yc):
                u = v / np.linalg.norm(v, axis=1, keepdims=True)
                sim = u @ u.T; np.fill_diagonal(sim, -np.inf)
                sets.append(np.argsort(-sim, axis=1)[:, :10])
            overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(*sets)])
            assert abs(overlap - metric["centered_cosine_mutual10NN"]) < 1e-12
            pair = []
            for v in (((x - mx) @ r)[1280:], (y - my)[1280:]):
                a = v.double().cpu().numpy(); pair.append(a / np.linalg.norm(a, axis=1, keepdims=True))
            sim = pair[0] @ pair[1].T
            rank = np.sum(sim >= np.diag(sim)[:, None], axis=1) - 1
            for key, value in {"retrieval_R1": np.mean(rank < 1), "retrieval_R5": np.mean(rank < 5),
                               "paired_cosine": np.diag(sim).mean(), "train_only_global_scale": float(scale)}.items():
                assert abs(value - metric[key]) < 1e-8, (key, value, metric[key])
            stored = np.load(directory / "test-records.npz")
            result, max_error, rounding = {}, 0., {}
            for name, value in states.items():
                lp, predictions, rounding[name] = audited_decode(value[1280:], params, specs, items, execution_head)
                actual = records(lp, rows, specs, items, predictions)
                np.testing.assert_array_equal(actual[..., [0, 2]], stored[name][..., [0, 2]])
                error = float(np.abs(actual[..., 1] - stored[name][..., 1]).max())
                assert error < .002, (arm, seed, name, error)
                max_error = max(max_error, error); result[name] = stored[name]
            shuffles = []
            for n in range(20):
                order = torch.randperm(1024, generator=torch.Generator().manual_seed(9001 + n))
                perm = torch.empty_like(order); perm[order] = order.roll(1)
                assert bool((perm != torch.arange(1024)).all())
                wrong = rotation((x[:1024] - mx).cpu(), (y[perm.cuda()] - my).cpu()).cuda()
                state = ((x - mx) @ wrong + my)[1280:]
                lp, predictions, rounding[f"shuffle-{n}"] = audited_decode(state, params, specs, items, execution_head)
                actual = records(lp, rows, specs, items, predictions)
                expected = stored["all_shuffles"][n]
                np.testing.assert_array_equal(actual[..., [0, 2]], expected[..., [0, 2]])
                error = float(np.abs(actual[..., 1] - expected[..., 1]).max())
                assert error < .002, (arm, seed, n, error)
                max_error = max(max_error, error); shuffles.append(expected)
            result["shuffle"] = np.mean(shuffles, axis=0)
            for name, value in result.items(): arrays[arm].setdefault(name, []).append(value)
            checks[arm][str(seed)] = {"replayed_decisions": len(items) * 25, "correct_count_mismatches": 0,
                                    "max_scene_task_nll_sum_error": max_error, "max_saved_state_error": state_error,
                                    "independent_biased_CKA": float(cka), "independent_mutual10NN": float(overlap),
                                    "rounding_audit": rounding,
                                    "selected_validation_seen_accuracy": best["validation_seen_accuracy"]}
            print("VERIFIED", arm, seed, checks[arm][str(seed)], flush=True)
            del encoder
    arrays = {a: {k: np.stack(v) for k, v in variants.items()} for a, variants in arrays.items()}
    scores = {a: {k: headline(v) for k, v in variants.items()} for a, variants in arrays.items()}
    for arm, variants in scores.items():
        for name, groups in variants.items():
            for group, metrics in groups.items():
                for metric, value in metrics.items():
                    assert abs(value - report["decision"][arm][name][group][metric]) < 1e-12
    for arm, variants in arrays.items():
        for i, seed in enumerate(seeds):
            for name, records_ in variants.items():
                for group, metrics in headline(records_[i:i + 1]).items():
                    for metric, value in metrics.items():
                        assert abs(value - report["per_seed_decision"][arm][str(seed)][name][group][metric]) < 1e-12
    if "paired_seed_accuracy_deltas" in report:
        for reference in ("ce", "bad"):
            for name in ("text", "rotation"):
                for group in ("seen", "unseen"):
                    delta = [headline(arrays["good"][name][i:i + 1])[group]["macro_accuracy"] -
                             headline(arrays[reference][name][i:i + 1])[group]["macro_accuracy"] for i in range(len(seeds))]
                    np.testing.assert_allclose(delta, report["paired_seed_accuracy_deltas"][reference][name][group], atol=1e-12)
        for arm in ARMS:
            for group, threshold in (("seen", .95), ("unseen", .9)):
                count = sum(headline(arrays[arm]["text"][i:i + 1])[group]["macro_accuracy"] >= threshold for i in range(len(seeds)))
                assert count == report["capability_pass_counts"][arm][group]
    cohort_scores = {}
    if "cohort_decision" in report:
        for name, subset in (("original3", plan["extension"]["original_seeds"]),
                             ("additional7", plan["extension"]["additional_seeds"])):
            positions = [seeds.index(s) for s in subset]
            cohort_scores[name] = {a: {k: headline(v[positions]) for k, v in variants.items()} for a, variants in arrays.items()}
            for a, variants in cohort_scores[name].items():
                for k, groups in variants.items():
                    for group, metrics in groups.items():
                        for metric, value in metrics.items():
                            assert abs(value - report["cohort_decision"][name][a][k][group][metric]) < 1e-12
    for seed in seeds:
        fits = [read(root / f"{arm}-{seed}/fit.json") for arm in ARMS]
        assert len({f["initialization_sha256"] for f in fits}) == 1
        assert fits[0]["train_step_digests"] == fits[1]["train_step_digests"] == fits[2]["train_step_digests"]
    comparisons = {}
    for reference in ("ce", "bad"):
        for name in ("text", "rotation"):
            rng = np.random.default_rng(8153); values = []
            for _ in range(5000):
                seed_counts = np.bincount(rng.integers(len(seeds), size=len(seeds)), minlength=len(seeds))
                scene_counts = np.bincount(rng.integers(256, size=256), minlength=256)
                delta = []
                for arm in ("good", reference):
                    total = np.einsum("i,j,ijklm->klm", seed_counts, scene_counts, arrays[arm][name])
                    delta.append((total[..., :2] / total[..., 2:3]).mean(1))
                values.append(delta[0] - delta[1])
            ci = np.quantile(values, [.025, .975], axis=0)
            key = f"good-minus-{reference}:{name}"
            np.testing.assert_allclose(ci, report["comparisons"][key]["ci95_group_metric"], atol=1e-12)
            comparisons[key] = {"ci95_group_metric": ci.tolist(), "bootstrap_draws_recomputed": 5000,
                                "comparison_scope": "recomputed CI endpoints match saved CI; primary per-draw values were not saved"}
    result = {"complete": True, "checks": checks, "headlines": scores, "comparisons": comparisons,
              "source_and_input_hashes": hashes, "verifier_source_sha256": sha(__file__), "seeds": list(seeds),
              "cohort_headlines": cohort_scores,
              "report_sha256": sha(root / "report.json"),
              "official_loss_check": read(root / "official-loss-check.json"),
              "input_audit": read(root / "input-audit.json"),
              "oracle_preflight": read(root / "verifier-preflight.json"),
              "scope": "all saved canonical states, train-only maps, all test gold, all 25 decision conditions/run, all headline scores and 5000 paired seed x scene bootstrap draws/comparison",
              "not_independently_repeated": "frozen Qwen frontend feature-bank generation; bank, frontend sources and base weights are checked by hashes"}
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/rotation-state/jepa-decision-v1"))
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_float32_matmul_precision("high")
    verify(args.out)


if __name__ == "__main__": main()
