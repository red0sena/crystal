"""Independent numerical replay of Crystal state comparison artifacts.

No use of Decision.forward, decision_records, summarize, or the gold helper.
"""
from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("runs/rotation-state/crystal-method-v1")
PARENT = Path("runs/rotation-state/plain-qwen-v2")
ANSWERS = ["red", "blue", "green", "yellow", "purple", "orange", "circle", "square", "triangle", "diamond",
           "left", "right", "above", "below"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 24), b""):
            h.update(part)
    return h.hexdigest()


def headline(array):
    # Sum numerators and denominators across seeds and scenes, average tasks.
    total = array.reshape(-1, 2, 3, 3).sum(0)
    rates = total[..., :2] / total[..., 2:3]
    return {name: {"macro_accuracy": float(rates[i, :, 0].mean()), "macro_nll": float(rates[i, :, 1].mean())}
            for i, name in enumerate(("seen", "unseen"))}


def gold_from_world(row, query):
    objects = row["world"]["objects"]
    if query["task"] == "color":
        answer = next(color for color, shape in objects if shape == query["reference"])
    elif query["task"] == "shape":
        answer = next(shape for color, shape in objects if color == query["reference"])
    else:
        order = [shape for color, shape in objects]
        difference = order.index(query["reference"]) - order.index(query["other"])
        answer = ("left" if difference < 0 else "right") if row["world"]["axis"] == "horizontal" else ("above" if difference < 0 else "below")
    return ANSWERS.index(answer)


def verify_native(directory, rows, specs, items):
    results = {}
    for modality in ("text", "image"):
        path = directory / f"native-{modality}-items.json"
        if not path.exists():
            continue
        records = read(path)
        array = np.zeros((256, 2, 3, 3), dtype=np.float64)
        assert len(records) == len(items)
        for record, expected in zip(records, items):
            assert all(record[k] == v for k, v in expected.items())
            query = specs[record["query"]]
            gold = gold_from_world(rows[record["scene"]], query)
            assert record["local_gold"] == query["answers"].index(gold)
            z = np.asarray(record["logits"], dtype=np.float64)
            nll = np.log(np.exp(z - z.max()).sum()) + z.max() - z[record["local_gold"]]
            correct = int(np.argmax(z) == record["local_gold"])
            scene = int(rows[record["scene"]]["id"].split("-")[1])
            array[scene, int(record["unseen"]), record["task"]] += [correct, nll, 1]
        saved = np.load(directory / f"native-{modality}.npy")
        assert np.array_equal(array[..., [0, 2]], saved[..., [0, 2]])
        error = float(np.abs(array[..., 1] - saved[..., 1]).max())
        assert error < .001
        results[modality] = {"questions_verified": len(records), "correct_count_mismatches": 0,
                             "max_scene_task_nll_sum_error": error, "independent_headline": headline(array)}
    return results


@torch.inference_mode()
def verify_states(directory, rows, specs, items):
    features = torch.load(directory / "features.pt", weights_only=True)
    maps = torch.load(directory / "maps.pt", weights_only=True)
    # The original state space SVD used float64 before casting states to FP32.
    # Reconstruct it independently from training feature rows, not saved activations.
    x, y = features["image"].double(), features["text"].double()
    training = [i for i, row in enumerate(rows) if row["split"] == "train"]
    mx, my = x[training].mean(0), y[training].mean(0)
    _, _, vh = torch.linalg.svd(torch.cat((x[training] - mx, y[training] - my)), full_matrices=False)
    basis = vh[:64].T
    np.testing.assert_allclose(mx.numpy(), maps["mu_image"].double().numpy(), atol=1e-5)
    np.testing.assert_allclose(my.numpy(), maps["mu_text"].double().numpy(), atol=1e-5)
    torch.testing.assert_close(basis.float(), maps["basis"], atol=1e-5, rtol=1e-4)
    image = F.normalize(((x - mx) @ basis).float(), dim=-1)
    text = F.normalize(((y - my) @ basis).float(), dim=-1)
    variants = {"image_raw": F.normalize(((x - my) @ basis).float(), dim=-1),
                "image_mean_only": image, "image_rotation": image @ maps["rotation"], "text_reference": text}
    query = F.normalize((features["query"] - maps["query_mean"]) @ maps["query_projection"], dim=-1)
    scene_indices = torch.tensor([e["scene"] for e in items])
    query_indices = torch.tensor([e["query"] for e in items])
    gold = torch.tensor([gold_from_world(rows[e["scene"]], specs[e["query"]]) for e in items])
    masks = torch.zeros((len(items), 14), dtype=torch.bool)
    for i, e in enumerate(items):
        masks[i, specs[e["query"]]["answers"]] = True
    saved = np.load(directory / "test-scene-records.npz")
    max_nll, mismatch = 0., 0
    independent = {name: [] for name in variants}
    for seed_index, seed in enumerate((271, 283, 307)):
        weights = torch.load(directory / f"text-head-{seed}.pt", weights_only=True)
        for name, states in variants.items():
            h = torch.cat((states[scene_indices], query[query_indices]), dim=-1).double()
            for index in (0, 3, 5):
                h = h @ weights[f"net.{index}.weight"].double().T + weights[f"net.{index}.bias"].double()
                if index != 5:
                    # Exact GELU formula, independent of the saved module.
                    h = .5 * h * (1. + torch.erf(h / 2.**.5))
            z = h.masked_fill(~masks, -float("inf"))
            correct = (z.argmax(-1) == gold).numpy()
            nll = (torch.logsumexp(z, dim=-1) - z[torch.arange(len(items)), gold]).numpy()
            array = np.zeros((256, 2, 3, 3), dtype=np.float64)
            for item, right, loss in zip(items, correct, nll):
                scene = int(rows[item["scene"]]["id"].split("-")[1])
                array[scene, int(item["unseen"]), item["task"]] += [int(right), loss, 1]
            mismatch += int(np.count_nonzero(array[..., 0] != saved[name][seed_index, ..., 0]))
            max_nll = max(max_nll, float(np.abs(array[..., 1] - saved[name][seed_index, ..., 1]).max()))
            independent[name].append(array)
    assert mismatch == 0, (directory, mismatch)
    assert max_nll < .01, (directory, max_nll)
    result = {"heads_replayed": 3, "variants_replayed": len(variants), "correct_count_mismatches": mismatch,
              "max_scene_task_nll_sum_error": max_nll,
              "independent_headlines": {k: headline(np.stack(v)) for k, v in independent.items()},
              "rotation_orthogonality_max_error": float((maps["rotation"].double().T @ maps["rotation"].double() - torch.eye(64)).abs().max())}
    test = [i for i, row in enumerate(rows) if row["split"] == "test"]
    similarity = F.normalize((image @ maps["rotation"])[test].double(), dim=-1) @ F.normalize(text[test].double(), dim=-1).T
    result["independent_retrieval_R1"] = float((similarity.argmax(-1) == torch.arange(256)).double().mean())
    expected = read(directory / "report.json")["alignment_test"]["image_rotation"]["retrieval_R1"]
    assert result["independent_retrieval_R1"] == expected
    expected_decisions = read(directory / "report.json")["decision_test"]
    for name, groups in result["independent_headlines"].items():
        for group, metrics in groups.items():
            assert metrics["macro_accuracy"] == expected_decisions[name][group]["macro_accuracy"] or abs(metrics["macro_accuracy"] - expected_decisions[name][group]["macro_accuracy"]) < 1e-14
            assert abs(metrics["macro_nll"] - expected_decisions[name][group]["macro_nll"]) < 1e-4
    return result


def verify_overall():
    report = read(ROOT / "report.json")
    arrays = {}
    for arm in ("plain", "causal", "crystal"):
        directories = [ROOT / "noncausal-control/untrained-causal"] * 3 if arm == "plain" else [ROOT / f"{arm}-{seed}" for seed in (271, 283, 307)]
        arrays[arm] = {name: np.stack([np.load(d / "test-scene-records.npz")[name] for d in directories])
                       for name in ("image_mean_only", "image_rotation", "text_reference")}
        for name, array in arrays[arm].items():
            metrics = headline(array)
            for group in metrics:
                for metric, value in metrics[group].items():
                    assert abs(value - report["state_transfer"][arm][name][group][metric]) < 1e-12
    supplement = read(ROOT / "noncausal-control/report.json")
    nc_root = ROOT / "noncausal-control"
    nc_candidate = np.stack([np.load(nc_root / f"crystal-{seed}/test-scene-records.npz")["image_rotation"] for seed in (271, 283, 307)])
    nc_reference = np.stack([np.load(nc_root / "untrained-noncausal/test-scene-records.npz")["image_rotation"]] * 3)
    cases = {f"crystal-minus-{reference}": (arrays["crystal"]["image_rotation"], arrays[reference]["image_rotation"],
              report["paired_hierarchical_bootstrap_5000"][f"crystal-minus-{reference}:image_rotation"])
             for reference in ("plain", "causal")}
    cases["noncausal-crystal-minus-untrained"] = (nc_candidate, nc_reference, supplement["crystal_minus_untrained_ci95"]["image_rotation"])
    results = {}
    for contrast, (candidate, reference, expected) in cases.items():
        rng = np.random.default_rng(202611051 + 500)
        d = candidate - reference
        counts = reference[..., 2]
        samples = []
        for _ in range(5000):
            # Multiplicity weights are independent of the original fancy-index implementation.
            lw = np.bincount(rng.integers(3, size=3), minlength=3)
            hw = np.bincount(rng.integers(3, size=3), minlength=3)
            sw = np.bincount(rng.integers(256, size=256), minlength=256)
            numerator = np.einsum("i,j,k,ijkgtm->gtm", lw, hw, sw, d, optimize=True)
            denominator = np.einsum("i,j,k,ijkgt->gt", lw, hw, sw, counts, optimize=True)
            samples.append((numerator[..., :2] / denominator[..., None]).mean(axis=1))
        ci = np.quantile(np.asarray(samples), [.025, .975], axis=0)
        for g, group in enumerate(("seen", "unseen")):
            for m, metric in enumerate(("accuracy", "nll")):
                np.testing.assert_allclose(ci[:, g, m], expected[group][metric]["ci95"], atol=1e-12)
        results[contrast] = {"independent_ci95_seen_accuracy": ci[:, 0, 0].tolist(),
                                                 "all_seen_unseen_accuracy_nll_intervals_match": True}
    return results


def native_comparisons():
    """Exploratory native-head intervals, separate from the predeclared primary contrasts."""
    arrays = {}
    for arm in ("plain", "causal", "crystal", "plain_noncausal"):
        if arm == "plain":
            directories = [ROOT / "plain"] * 3
        elif arm == "plain_noncausal":
            directories = [ROOT / "noncausal-control/untrained-noncausal"] * 3
        else:
            directories = [ROOT / f"{arm}-{seed}" for seed in (271, 283, 307)]
        arrays[arm] = {m: np.stack([np.load(d / f"native-{m}.npy") for d in directories]) for m in ("text", "image")}
    result = {}
    for reference in ("plain", "causal", "plain_noncausal"):
        rng = np.random.default_rng(202611051 + 900)
        samples = {m: [] for m in ("text", "image")}
        for _ in range(5000):
            lw = np.bincount(rng.integers(3, size=3), minlength=3)
            sw = np.bincount(rng.integers(256, size=256), minlength=256)
            for modality in samples:
                difference = arrays["crystal"][modality] - arrays[reference][modality]
                numerator = np.einsum("i,k,ikgtm->gtm", lw, sw, difference, optimize=True)
                denominator = np.einsum("i,k,ikgt->gt", lw, sw, arrays[reference][modality][..., 2], optimize=True)
                samples[modality].append((numerator[..., :2] / denominator[..., None]).mean(axis=1))
        result[f"crystal-minus-{reference}"] = {}
        for modality, values in samples.items():
            ci = np.quantile(values, [.025, .975], axis=0)
            a, b = headline(arrays["crystal"][modality]), headline(arrays[reference][modality])
            result[f"crystal-minus-{reference}"][modality] = {
                group: {metric: {"delta": a[group]["macro_" + metric] - b[group]["macro_" + metric],
                                 "ci95": ci[:, g, m].tolist()} for m, metric in enumerate(("accuracy", "nll"))}
                for g, group in enumerate(("seen", "unseen"))}
    return result


def main():
    torch.set_num_threads(8)
    rows, specs = read(PARENT / "scenes.json"), read(PARENT / "query-specs.json")
    items = read(PARENT / "test-items.json")
    for item in items:
        assert item["gold"] == gold_from_world(rows[item["scene"]], specs[item["query"]])
    plan = read(ROOT / "plan.json")
    for name in ("source_sha256", "base_weights_sha256", "inherited_files_sha256"):
        for file, expected in plan[name].items():
            assert sha(Path(file)) == expected, file
    extra = ROOT / "noncausal-control"
    supplement = read(extra / "plan.json")
    control_source = Path("qwen_decision/v3/crystal_state_attention_control.py")
    maintenance = None
    if sha(control_source) != supplement["source_sha256"]:
        maintenance = read(extra / "runner-maintenance.json")
        snapshot = extra / maintenance["executed_source_snapshot"]
        assert sha(snapshot) == supplement["source_sha256"] == maintenance["executed_source_sha256"]
        assert sha(control_source) == maintenance["current_source_sha256"]
        trees = [ast.parse(p.read_text()) for p in (snapshot, control_source)]
        for tree in trees:
            tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != "freeze"]
        assert ast.dump(trees[0]) == ast.dump(trees[1])
    if supplement["prior_plan_sha256"] is not None:
        assert sha(extra / "plan-initial.json") == supplement["prior_plan_sha256"]
        assert sha(extra / "source-initial.py") == read(extra / "plan-initial.json")["source_sha256"]
    input_checks = {}
    for seed in (271, 283, 307):
        a, b = [read(ROOT / f"{arm}-{seed}/step-input-digests.json") for arm in ("causal", "crystal")]
        assert a == b and len(a) == 558
        input_checks[str(seed)] = len(a)
        for arm in ("causal", "crystal"):
            directory = ROOT / f"{arm}-{seed}"
            assert sha(directory / "best.safetensors") == read(directory / "fit.json")["checkpoint_sha256"]
    directories = [ROOT / f"{arm}-{seed}" for arm in ("causal", "crystal") for seed in (271, 283, 307)]
    directories += [ROOT / "noncausal-control" / name for name in ("untrained-causal", "untrained-noncausal", "crystal-271", "crystal-283", "crystal-307")]
    result = {"all_3072_test_gold_verified_from_worlds": True, "immutable_input_source_base_hashes_verified": True,
              "paired_step_input_digests_identical": input_checks, "control_runner_freeze_only_maintenance": maintenance,
              "directories": {}}
    for directory in directories:
        print("VERIFY", directory, flush=True)
        result["directories"][str(directory.relative_to(ROOT))] = {
            "state": verify_states(directory, rows, specs, items), "native": verify_native(directory, rows, specs, items)}
    result["directories"]["plain"] = {"native": verify_native(ROOT / "plain", rows, specs, items)}
    result["overall_and_bootstrap"] = verify_overall()
    result["native_comparisons_posthoc_5000_seed_scene_bootstrap"] = native_comparisons()
    original_features = torch.load(PARENT / "features.pt", weights_only=True)
    matched_features = torch.load(extra / "untrained-causal/features.pt", weights_only=True)
    result["baseline_kernel_drift"] = {}
    for name in ("image", "text", "query"):
        a, b = original_features[name].double(), matched_features[name].double()
        result["baseline_kernel_drift"][name] = {"relative_frobenius_error": float((a - b).norm() / a.norm()),
                                                 "mean_paired_cosine": float((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).mean())}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    checksums = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob("*"))
                 if p.is_file() and p.name != "checksums.json" and p.suffix != ".log"}
    (ROOT / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print("ALL INDEPENDENT CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
