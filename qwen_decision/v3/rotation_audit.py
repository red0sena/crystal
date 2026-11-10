"""Post-hoc audit of the frozen rotation pilot; never edits its evidence.

Recompute decisions independently from saved logits, verify input/artifact
integrity, and distinguish geometry from readout correction. New mean-only and
centered analyses are exploratory, not additional preregistered confirmations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F


def mean_direction_rotation(x, y):
    """One proper rotation of the two training mean directions, no pairing used."""
    u, v = F.normalize(x.double().mean(0), dim=0), F.normalize(y.double().mean(0), dim=0)
    cosine = (u @ v).clamp(-1, 1)
    sine = torch.sqrt(1 - cosine.square())
    if sine < 1e-10:
        raise ValueError("degenerate mean directions require a separate convention")
    b = (v - cosine * u) / sine
    identity = torch.eye(len(u), dtype=torch.float64)
    return identity + (cosine - 1) * (torch.outer(u, u) + torch.outer(b, b)) + sine * (
        torch.outer(u, b) - torch.outer(b, u))


def linear_cka(x, y):
    """Biased, centered linear CKA; not the paper's unbiased HSIC estimator."""
    x, y = x - x.mean(0), y - y.mean(0)
    k, ell = x @ x.T, y @ y.T
    return ((k * ell).sum() / torch.sqrt(k.square().sum() * ell.square().sum())).item()


def neighbors(x, k=10):
    z = F.normalize(x, dim=-1)
    similarities = z @ z.T
    similarities.fill_diagonal_(-torch.inf)
    return similarities.topk(k, dim=-1).indices


def knn_overlap(x, y, k=10):
    a, b = neighbors(x, k), neighbors(y, k)
    return (a[:, :, None] == b[:, None, :]).any(-1).double().mean().item()


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum(), float(z.max() + math.log(math.fsum(exp)))


def evaluate_readout(h, rows, head):
    z = (h @ head.T).numpy()
    right, loss = [], []
    for i, row in enumerate(rows):
        values = z[i, :len(row["question"]["criteria"])]
        p, _ = probabilities(values)
        right.append(int(values.argmax()) == row["gold"])
        loss.append(-math.log(p[row["gold"]]))
    return {"accuracy": float(np.mean(right)), "nll": float(np.mean(loss))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/rotation-pilot/plain-qwen-v1"))
    parser.add_argument("--out", type=Path, default=Path("wiki/project/rotation-alignment-audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.run
    checksums = json.loads((root / "checksums.json").read_text())
    integrity = {}
    for path, expected in checksums.items():
        with Path(path).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        integrity[path] = actual == expected
    assert all(integrity.values()), integrity
    rows = [json.loads(line) for line in (root / "scenes.jsonl").read_text().splitlines()]
    records = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    report = json.loads((root / "report.json").read_text())
    features = torch.load(root / "features.pt", weights_only=True)
    maps = torch.load(root / "rotation.pt", weights_only=True)
    indices = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "validation", "test")}
    tr, te = indices["train"], indices["test"]
    test_rows = [rows[i] for i in te]
    assert [r["id"] for r in records] == [r["id"] for r in test_rows]
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({json.dumps(r["world"]) for r in rows}) == len(rows)
    image_hashes = []
    # Reuse rendering only for image-byte duplicate detection, not metric validation.
    from qwen_decision.v3.rotation_pilot import draw_scene, shuffled_targets
    for row in rows:
        picture = draw_scene(row)
        image_hashes.append(hashlib.sha256(picture.tobytes()).hexdigest())
        a, b, relation = row["world"]
        question = row["question"]["instructions"]
        options = list(row["question"]["criteria"])
        if row["task"] == "color":
            shape = re.fullmatch(r"What color is the (\w+)\?", question)[1]
            expected = next(color for color, name in (a, b) if name == shape)
        elif row["task"] == "shape":
            color = re.fullmatch(r"What shape is the (\w+) object\?", question)[1]
            expected = next(shape for name, shape in (a, b) if name == color)
        else:
            subject, other = re.fullmatch(r"Where is the (\w+) relative to the (\w+)\?", question).groups()
            assert {subject, other} == {a[1], b[1]}
            expected = ("left", "right", "above", "below")[relation if subject == a[1] else relation ^ 1]
        assert options[row["gold"]] == expected, row["id"]
    assert len(set(image_hashes)) == len(rows)
    independent, values = {}, {}
    for arm in records[0]["arms"]:
        right, nll, brier = [], [], []
        for row, saved in zip(test_rows, records):
            z = np.asarray(saved["arms"][arm]["logits"], dtype=np.float64)
            p, _ = probabilities(z)
            correct = int(z.argmax()) == row["gold"]
            loss = -math.log(p[row["gold"]])
            assert correct == saved["arms"][arm]["correct"]
            assert abs(loss - saved["arms"][arm]["nll"]) < 2e-6
            target = np.zeros(len(z)); target[row["gold"]] = 1
            right.append(correct); nll.append(loss); brier.append(float(((p - target) ** 2).sum()))
        right, nll = np.asarray(right), np.asarray(nll)
        independent[arm] = {"correct": int(right.sum()), "n": len(right), "accuracy": float(right.mean()),
                            "nll": float(nll.mean()), "brier_sum_over_options": float(np.mean(brier)),
                            "by_task": {task: {"correct": int(right[[r["task"] == task for r in test_rows]].sum()),
                                                "n": sum(r["task"] == task for r in test_rows),
                                                "nll": float(nll[[r["task"] == task for r in test_rows]].mean())}
                                        for task in ("color", "shape", "relation")}}
        for key in ("accuracy", "nll"):
            assert abs(independent[arm][key] - report["metrics"][arm][key]) < 2e-6
        values[arm] = (right, nll)
    bootstrap, transitions = {}, {}
    task_slots = [np.array([i for i, r in enumerate(test_rows) if r["task"] == task])
                  for task in ("color", "shape", "relation")]
    rng = np.random.default_rng(20261105)
    samples = np.concatenate([rng.choice(slots, (20000, len(slots)), replace=True) for slots in task_slots], axis=1)
    for arm in ("image_identity", "image_shuffled_pairs_rotation"):
        original, rotated = values[arm], values["image_rotation"]
        gains = int((~original[0] & rotated[0]).sum()); losses = int((original[0] & ~rotated[0]).sum())
        discordant = gains + losses
        exact_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant)
        transitions[arm] = {"wrong_to_correct": gains, "correct_to_wrong": losses,
                            "both_correct": int((original[0] & rotated[0]).sum()),
                            "both_wrong": int((~original[0] & ~rotated[0]).sum()),
                            "mcnemar_exact_two_sided_p": exact_p}
        delta_acc = rotated[0].astype(float) - original[0].astype(float)
        delta_nll = rotated[1] - original[1]
        bootstrap[arm] = {"accuracy_delta_ci95": np.quantile(delta_acc[samples].mean(1), [.025, .975]).tolist(),
                          "nll_delta_ci95": np.quantile(delta_nll[samples].mean(1), [.025, .975]).tolist()}
    x, y = features["image"].double(), features["text"].double()
    head, rotation = features["head"].double(), maps["rotation"].double()
    mean_map = mean_direction_rotation(x[tr], y[tr])
    variants = {"identity": torch.eye(x.shape[1], dtype=torch.float64), "paired": rotation,
                "shuffled": maps["shuffled_rotation"].double(), "mean_direction_only": mean_map}
    xn, yn = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    prompt_groups = defaultdict(list)
    for i, row in enumerate(test_rows):
        prompt_groups[json.dumps(row["question"], sort_keys=True)].append(i)
    geometry = {}
    reference_gram = xn[te] @ xn[te].T
    for name, transform in variants.items():
        a, b = xn[te] @ transform, yn[te]
        centered_a, centered_b = (xn[te] - xn[tr].mean(0)) @ transform, yn[te] - yn[tr].mean(0)
        similarities = F.normalize(a, dim=-1) @ b.T
        within_hits = []
        for group in prompt_groups.values():
            for i in group:
                within_hits.append(int(group[similarities[i, group].argmax()]) == i)
        geometry[name] = {"raw_paired_cosine": F.cosine_similarity(a, b).mean().item(),
                          "train_centered_unit_feature_cosine": F.cosine_similarity(centered_a, centered_b).mean().item(),
                          "mutual10nn_no_clipping": knn_overlap(a, b),
                          "biased_centered_linear_CKA": linear_cka(a, b),
                          "max_normalized_image_gram_change": ((a @ a.T) - reference_gram).abs().max().item(),
                          "cross_modal_retrieval_top1_all_192": (similarities.argmax(-1) == torch.arange(len(te))).double().mean().item(),
                          "cross_modal_retrieval_top1_same_prompt": float(np.mean(within_hits)),
                          "readout": evaluate_readout(x[te] @ transform, test_rows, head)}
    effective_head = head @ rotation.T
    head_equivalence_error = ((x[te] @ rotation) @ head.T - x[te] @ effective_head.T).abs().max().item()
    training_rows = [rows[i] for i in tr]
    shuffled = shuffled_targets(training_rows)
    # Same-state-only positives do not imply unique spectral solutions.
    example_a = torch.tensor([1., -1., 0.], dtype=torch.float64) * math.sqrt(1.5)
    example_b = torch.tensor([1., 1., -2.], dtype=torch.float64) / math.sqrt(2)
    counterexample = {"mean_a": example_a.mean().item(), "mean_b": example_b.mean().item(),
                      "second_moment_a": example_a.square().mean().item(), "second_moment_b": example_b.square().mean().item(),
                      "best_O1_error": min((example_a - example_b).square().mean().item(),
                                           (example_a + example_b).square().mean().item()),
                      "positive_pair_kernel": "diag(1/3,1/3,1/3); K=I; all eigenvalues 1; k=1 has no eigengap"}
    forward_path = root / "forward-spotcheck.json"
    forward = json.loads(forward_path.read_text()) if forward_path.exists() else None
    summary = {"audit_date": "2026-11-04", "scope": "post-hoc audit and exploratory diagnostics; maps are not reselected; separate forward spot check if available",
               "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "audit_runtime": {"torch": torch.__version__, "numpy": np.__version__},
               "source_run": str(root), "artifact_checksums_verified": integrity,
               "data_checks": {"unique_ids": len(rows), "unique_worlds": len(rows), "unique_rendered_images": len(set(image_hashes)),
                               "gold_labels_independently_checked": len(rows),
                               "counts": {k: len(v) for k, v in indices.items()},
                               "gold_counts_by_split_task": {s: {t: dict(Counter(str(rows[i]["gold"]) for i in ids if rows[i]["task"] == t))
                                                                for t in ("color", "shape", "relation")} for s, ids in indices.items()},
                               "shuffled_pairs_same_gold_fraction": sum(r["gold"] == training_rows[j]["gold"] for r, j in zip(training_rows, shuffled)) / len(tr),
                               "test_distinct_exact_prompts": len(prompt_groups),
                               "within_prompt_retrieval_chance": len(prompt_groups) / len(te)},
               "independent_recomputation": independent, "transitions": transitions,
               "posthoc_task_stratified_bootstrap_20000": bootstrap,
               "posthoc_geometry": geometry,
               "mean_direction_rotation_orthogonality_error": (mean_map.T @ mean_map - torch.eye(len(mean_map))).abs().max().item(),
               "effective_head_equivalence_max_logit_error": head_equivalence_error,
               "forward_spotcheck": None if forward is None else {
                   "path": str(forward_path), "n": forward["n"], "same_predictions": forward["same_predictions"],
                   "min_hidden_cosine": min(r["hidden_cosine"] for r in forward["records"]),
                   "max_logit_abs_diff": max(r["max_logit_abs_diff"] for r in forward["records"]),
                   "scope": forward["scope"]},
               "unit_feature_mean_energy_fraction_test": {"image": xn[te].mean(0).square().sum().item(),
                                                         "text": yn[te].mean(0).square().sum().item()},
               "same_state_only_counterexample": counterexample,
               "scope_caveats": ["Diagnostics chosen after seeing the original test; not new held-out confirmation.",
                                 "One fixed mapping split, one negative-control shuffle, one checkpoint.",
                                 "Full rotation and negative control share ratio selected on true-pair validation.",
                                 "Biased centered CKA and no-clipping kNN differ from the paper's full PRH protocol.",
                                 "Confidence intervals condition on the fixed fitted map; no refit or domain uncertainty."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("transitions", "posthoc_geometry", "unit_feature_mean_energy_fraction_test")}, indent=2))


if __name__ == "__main__":
    main()
