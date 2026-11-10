"""Ten-seed extension using the immutable original fit and evaluator.

The original three-seed run is preserved. Only the evaluator's seed list and
bootstrap sample count change. Text and image are both evaluated as before.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from qwen_decision.v3 import jepa_decision as exp

PARENT = exp.ROOT
ROOT = Path("runs/rotation-state/jepa-decision-seeds10-v1")
SEEDS = (401, 419, 443, 461, 479, 503, 521, 541, 563, 587)
ADDED = SEEDS[3:]
WORKERS = 2
MODULE = "qwen_decision.v3.jepa_decision_seed_extension"


def evaluator_source(seeds):
    original = Path(exp.__file__).read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    start = min([node.lineno] + [n.lineno for n in node.decorator_list])
    function = "\n".join(original.splitlines()[start - 1:node.end_lineno]) + "\n"
    pattern = "rng.integers(3, size=3)"
    assert function.count(pattern) == 1
    function = function.replace(pattern, "rng.integers(len(SEEDS), size=len(SEEDS))")
    return "from qwen_decision.v3.jepa_decision import *\nSEEDS = " + repr(tuple(seeds)) + "\n\n" + function


def prepare(root):
    old = exp.read(PARENT / "plan.json")
    assert tuple(old["seeds"]) == SEEDS[:3]
    assert exp.sha(exp.__file__) == old["source_sha256"] == exp.sha(PARENT / "experiment-source.py")
    root.mkdir(parents=True, exist_ok=False)
    for m in ("text", "image"):
        assert exp.sha(PARENT / f"{m}-tokens.pt") == old["banks_sha256"][m]
        os.link(PARENT / f"{m}-tokens.pt", root / f"{m}-tokens.pt")
    for name in ("scenes.json", "query-specs.json", "view-plan.json", "prepare-source.py", "experiment-source.py",
                 "official-loss-check.json", "input-audit.json", "verifier-preflight.json", "environment.json"):
        shutil.copy2(PARENT / name, root / name)
    reused = {}
    for seed in SEEDS[:3]:
        for arm in exp.ARMS:
            source, target = PARENT / f"{arm}-{seed}", root / f"{arm}-{seed}"
            shutil.copytree(source, target)
            for path in source.iterdir():
                if path.is_file():
                    reused[str(path.relative_to(PARENT))] = exp.sha(path)
    (root / "evaluation-source.py").write_text(evaluator_source(SEEDS))
    shutil.copy2(__file__, root / "extension-source.py")
    old.update({"seeds": list(SEEDS), "extension": {
        "parent": str(PARENT), "parent_report_sha256": exp.sha(PARENT / "report.json"),
        "parent_plan_sha256": exp.sha(PARENT / "plan.json"), "original_seeds": list(SEEDS[:3]),
        "additional_seeds": list(ADDED), "fixed_total_seeds": 10, "workers": WORKERS,
        "source_sha256": exp.sha(__file__), "evaluation_source_sha256": exp.sha(root / "evaluation-source.py"),
        "reused_artifacts_sha256": reused,
        "stopping_rule": "run all seven additional seeds, all three arms, all 200 epochs; no optional stopping or seed exclusions",
        "selection": "unchanged best text/seen validation NLL; no new loss or architecture tuning",
        "scope": "same previously examined synthetic split; pooled ten seeds and added-seven means both reported",
        "execution": "two isolated Python worker processes, original fit function unchanged; failed attempts retained and retried with same configuration"
    }})
    exp.sr.dump(root / "plan.json", old)
    print("FROZEN", root, "seeds", SEEDS, flush=True)


def check_plan(root):
    plan = exp.read(root / "plan.json")
    assert tuple(plan["seeds"]) == SEEDS
    assert plan["source_sha256"] == exp.sha(exp.__file__)
    assert plan["extension"]["source_sha256"] == exp.sha(__file__)
    assert plan["extension"]["evaluation_source_sha256"] == exp.sha(root / "evaluation-source.py")
    assert plan["extension"]["parent_report_sha256"] == exp.sha(PARENT / "report.json")
    return plan


def worker(root, arm, seed):
    check_plan(root)
    assert seed in ADDED and arm in exp.ARMS
    assert not (root / f"{arm}-{seed}/fit.json").exists()
    exp.fit(root, arm, seed)


def train(root):
    check_plan(root)
    jobs = [(seed, arm) for seed in ADDED for arm in exp.ARMS if not (root / f"{arm}-{seed}/fit.json").exists()]
    def launch(job):
        seed, arm = job
        attempt = 1
        while (root / f"fit-{arm}-{seed}-attempt{attempt}.log").exists(): attempt += 1
        path = root / f"fit-{arm}-{seed}-attempt{attempt}.log"
        print("START", arm, seed, "attempt", attempt, flush=True)
        with path.open("x") as handle:
            process = subprocess.run([sys.executable, "-u", "-m", MODULE, "worker", "--out", str(root),
                                      "--arm", arm, "--seed", str(seed)], stdout=handle, stderr=subprocess.STDOUT)
        print("END", arm, seed, "exit", process.returncode, flush=True)
        return process.returncode
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(launch, jobs))
    assert not any(results), "A worker failed; inspect per-attempt logs, preserve failed attempts and resume unchanged"
    for seed in SEEDS:
        fits = [exp.read(root / f"{arm}-{seed}/fit.json") for arm in exp.ARMS]
        assert all(f["source_sha256"] == exp.sha(exp.__file__) and len(f["history"]) == 200 for f in fits)
        assert len({f["initialization_sha256"] for f in fits}) == 1
        assert fits[0]["train_step_digests"] == fits[1]["train_step_digests"] == fits[2]["train_step_digests"]


def cohort_scores(root, seeds):
    result = {}
    for arm in exp.ARMS:
        records = [np.load(root / f"{arm}-{seed}/test-records.npz") for seed in seeds]
        result[arm] = {v: exp.sr.summarize(np.stack([r[v] if v != "shuffle" else r["all_shuffles"].mean(0)
                                                   for r in records]))
                       for v in ("text", "raw", "mean", "rotation", "similarity", "shuffle")}
        for record in records: record.close()
    return result


def analyze(root):
    check_plan(root)
    spec = importlib.util.spec_from_file_location("jepa_ten_seed_evaluator", root / "evaluation-source.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.analyze(root)
    report = exp.read(root / "report.json")
    report["seeds"] = list(SEEDS)
    report["cohort_decision"] = {"original3": cohort_scores(root, SEEDS[:3]), "additional7": cohort_scores(root, ADDED)}
    original = exp.read(PARENT / "report.json")["decision"]
    for arm in exp.ARMS:
        for variant in original[arm]:
            for group in ("seen", "unseen"):
                for metric in ("macro_accuracy", "macro_nll"):
                    assert abs(report["cohort_decision"]["original3"][arm][variant][group][metric] - original[arm][variant][group][metric]) < 1e-12
    report["paired_seed_accuracy_deltas"] = {ref: {v: {g: [
        report["per_seed_decision"]["good"][str(s)][v][g]["macro_accuracy"] - report["per_seed_decision"][ref][str(s)][v][g]["macro_accuracy"]
        for s in SEEDS] for g in ("seen", "unseen")} for v in ("text", "rotation")} for ref in ("ce", "bad")}
    report["capability_pass_counts"] = {a: {g: sum(report["per_seed_decision"][a][str(s)]["text"][g]["macro_accuracy"] >= threshold for s in SEEDS)
                                            for g, threshold in (("seen", .95), ("unseen", .90))} for a in exp.ARMS}
    exp.sr.dump(root / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "analyze"))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=exp.ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_float32_matmul_precision("high")
    if args.command == "prepare": prepare(args.out)
    elif args.command == "worker": worker(args.out, args.arm, args.seed)
    elif args.command == "run": train(args.out); analyze(args.out)
    else: analyze(args.out)


if __name__ == "__main__": main()
