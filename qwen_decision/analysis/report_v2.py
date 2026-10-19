"""Aggregate evaluate_v2.py outputs for original, v1 and v2 into
results/v2-eval/summary.json and results/v2-eval/report_tables.md.

All accuracies use the stored test labels. Paired comparisons use exact
McNemar tests and 2,000-sample paired bootstrap 95% intervals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import qwen_decision.core.decision_format as fmt

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results/v2-eval"
MODELS = ("original", "v1", "v2")
V1_TASKS = sorted(json.loads((ROOT / "data/text-decisions-v1/manifest.json").read_text())["stats"]["test"]["by_task"])


def read(path):
    with path.open() as f:
        return [json.loads(x) for x in f]


def softmax(z, t=1.):
    m = max(v / t for v in z)
    e = [math.exp(v / t - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def mcnemar(a, b):
    """Exact two-sided p for paired correctness lists a (base) and b (new)."""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap(a, b, rounds=2000, seed=0):
    rng = random.Random(seed)
    d = [int(y) - int(x) for x, y in zip(a, b)]
    n = len(d)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return 100 * stats[int(.025 * rounds)], 100 * stats[int(.975 * rounds)]


def calibration_metrics(rows, temps=None):
    ece_bins = [[0, 0., 0.] for _ in range(15)]
    nll = brier = 0.
    for r in rows:
        t = temps.get(fmt.calibration_key(r["qtype"], r["n_options"]), 1.) if temps else 1.
        p = softmax(r["logits"], t)
        conf = max(p)
        b = min(14, int(conf * 15))
        ece_bins[b][0] += 1
        ece_bins[b][1] += conf
        ece_bins[b][2] += r["correct"]
        nll -= math.log(max(p[r["gold"]], 1e-12))
        brier += sum((pi - (i == r["gold"])) ** 2 for i, pi in enumerate(p))
    n = len(rows)
    ece = sum(abs(c - k) for _, c, k in ece_bins) / n
    return {"n": n, "ece": ece, "nll": nll / n, "brier": brier / n}


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    r = ranks(pos + neg)
    return (sum(r[:len(pos)]) - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def per_class(rows):
    support, correct, predicted = Counter(), Counter(), Counter()
    for r in rows:
        g, p = r["names"][r["gold"]], r["names"][r["pred"]]
        support[g] += 1
        predicted[p] += 1
        correct[g] += g == p
    recalls = [correct[c] / support[c] for c in support]
    f1s = []
    for c in support:
        prec = correct[c] / predicted[c] if predicted[c] else 0.
        rec = correct[c] / support[c]
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.)
    return 100 * sum(recalls) / len(recalls), 100 * sum(f1s) / len(f1s)


def main():
    data = {m: {p: read(EVAL / m / f"{p}.jsonl") for p in ("test", "unseen", "external", "permutation", "robustness")}
            for m in MODELS}
    temps = {m: json.loads((EVAL / m / "calibration-fit.json").read_text())["temperatures"] for m in MODELS}
    test_records = {}
    with (ROOT / "data/text-decisions-v2/test.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            test_records[r["id"]] = r
    baselines = json.loads((EVAL / "baselines.json").read_text())
    affected = set(baselines["affected_test_ids"])
    summary = {"models": MODELS}

    # 1. Test accuracy per task with paired comparisons.
    by = {m: defaultdict(dict) for m in MODELS}
    for m in MODELS:
        for r in data[m]["test"]:
            by[m][r["task"]][r["id"]] = r
    tasks = sorted(by["v2"])
    table = {}
    for t in tasks:
        ids = sorted(by["v2"][t])
        c = {m: [by[m][t][i]["correct"] for i in ids] for m in MODELS}
        row = {"n": len(ids), **{m: 100 * sum(c[m]) / len(ids) for m in MODELS}}
        for base in ("original", "v1"):
            row[f"v2_minus_{base}"] = row["v2"] - row[base]
            row[f"ci_vs_{base}"] = bootstrap(c[base], c["v2"])
            row[f"p_vs_{base}"] = mcnemar(c[base], c["v2"])
        row["baseline"] = baselines["baselines"][t]["best_baseline_pct"]
        row["baseline_name"] = baselines["baselines"][t]["best_baseline"]
        table[t] = row
    new_tasks = [t for t in tasks if t not in V1_TASKS]
    macro = {group: {m: sum(table[t][m] for t in ts) / len(ts) for m in MODELS}
             for group, ts in (("v1_tasks", V1_TASKS), ("new_tasks", new_tasks))}
    summary["test"] = {"per_task": table, "macro": macro}
    sens = {}
    for t in ("newsgroup_topic", "paraphrase"):
        ids = [i for i in sorted(by["v2"][t]) if i not in affected]
        sens[t] = {"n_excluding_affected": len(ids), **{m: 100 * sum(by[m][t][i]["correct"] for i in ids) / len(ids) for m in MODELS}}
    summary["leak_sensitivity"] = sens

    # 2. Many-option analysis: accuracy by K, and gold-code range at K=64/60.
    wide = {}
    for t in tasks:
        if "@" not in t:
            continue
        base, k = t.split("@")
        entry = wide.setdefault(base, {})
        entry[k] = {m: table[t][m] for m in MODELS}
        if int(k) >= 60:
            for m in MODELS:
                rows = list(by[m][t].values())
                hi = [r for r in rows if r["gold"] >= 26]
                lo = [r for r in rows if r["gold"] < 26]
                entry[f"code_range@{k}:{m}"] = {
                    "acc_gold_A-Z": 100 * sum(r["correct"] for r in lo) / len(lo),
                    "acc_gold_AA-BL": 100 * sum(r["correct"] for r in hi) / len(hi),
                    "gold_share_AA-BL": 100 * len(hi) / len(rows),
                    "pred_share_AA-BL": 100 * sum(r["pred"] >= 26 for r in rows) / len(rows)}
    summary["many_options"] = wide
    pooled = {}
    for m in MODELS:
        rows = [r for t in tasks if "@" in t and int(t.split("@")[1]) >= 60 for r in by[m][t].values()]
        hi = [r for r in rows if r["gold"] >= 26]
        lo = [r for r in rows if r["gold"] < 26]
        pooled[m] = {"n": len(rows), "acc_gold_A-Z": 100 * sum(r["correct"] for r in lo) / len(lo),
                     "acc_gold_AA-BL": 100 * sum(r["correct"] for r in hi) / len(hi),
                     "gold_share_AA-BL": 100 * len(hi) / len(rows),
                     "pred_share_AA-BL": 100 * sum(r["pred"] >= 26 for r in rows) / len(rows)}
    summary["many_options_pooled_K60_64"] = pooled

    # 3. Calibration on test, raw and with each model's own development temperatures.
    cal = {}
    for m in MODELS:
        rows = data[m]["test"]
        groups = {"all": rows, "choice": [r for r in rows if r["qtype"] == "choice"],
                  "noul": [r for r in rows if r["qtype"] == "noul"], "score": [r for r in rows if r["qtype"] == "score"]}
        for key in sorted({fmt.calibration_key(r["qtype"], r["n_options"]) for r in rows}):
            groups[key] = [r for r in rows if fmt.calibration_key(r["qtype"], r["n_options"]) == key]
        cal[m] = {g: {"raw": calibration_metrics(rr), "calibrated": calibration_metrics(rr, temps[m])} for g, rr in groups.items()}
        cal[m]["temperatures"] = temps[m]
    summary["calibration"] = cal

    # 4-5. Score and noul metrics.
    typed = {}
    for m in MODELS:
        sts = [r for r in data[m]["test"] if r["task"] == "sts_score"]
        exp = [sum(float(n) * p for n, p in zip(r["names"], softmax(r["logits"]))) for r in sts]
        gold = [test_records[r["id"]]["annotation"]["rating"] for r in sts]
        yelp = [r for r in data[m]["test"] if r["task"] == "review_stars"]
        yexp = [sum(float(n) * p for n, p in zip(r["names"], softmax(r["logits"]))) for r in yelp]
        ygold = [float(r["names"][r["gold"]]) for r in yelp]
        boolq = [r for r in data[m]["test"] if r["task"] == "boolq_noul"]
        pyes = [softmax(r["logits"])[r["names"].index("yes")] for r in boolq]
        typed[m] = {"sts_pearson": pearson(exp, gold), "sts_spearman": pearson(ranks(exp), ranks(gold)),
                    "sts_mae": sum(abs(a - b) for a, b in zip(exp, gold)) / len(gold),
                    "yelp_mae": sum(abs(a - b) for a, b in zip(yexp, ygold)) / len(ygold),
                    "yelp_within_one_pct": 100 * sum(abs(float(r["names"][r["pred"]]) - g) <= 1 for r, g in zip(yelp, ygold)) / len(yelp),
                    "boolq_auroc": auroc(pyes, [r["names"][r["gold"]] == "yes" for r in boolq])}
    summary["typed_questions"] = typed

    # 6. Permutation consistency.
    perm = {}
    for m in MODELS:
        groups = defaultdict(dict)
        for r in data[m]["permutation"]:
            groups[(r["task"], r["source_id"])][r["tag"]] = r
        per_task = defaultdict(lambda: [0, 0, 0, 0])
        for (task, _), g in groups.items():
            if len(g) != 4:
                continue
            chosen = {r["names"][r["pred"]] for r in g.values()}
            d = per_task[task]
            d[0] += len(chosen) == 1
            d[1] += 1
            d[2] += sum(r["correct"] for r in g.values())
            d[3] += 4
        perm[m] = {"per_task_consistency_pct": {t: 100 * d[0] / d[1] for t, d in sorted(per_task.items())},
                   "overall_consistency_pct": 100 * sum(d[0] for d in per_task.values()) / sum(d[1] for d in per_task.values()),
                   "mean_accuracy_over_orders_pct": 100 * sum(d[2] for d in per_task.values()) / sum(d[3] for d in per_task.values())}
    summary["permutation"] = perm

    # 7. Robustness: injected instruction and gold-removed escape option.
    rob = {}
    for m in MODELS:
        clean = {r["id"]: r for r in data[m]["test"]}
        inj = [r for r in data[m]["robustness"] if r["tag"].startswith("inject:")]
        esc = [r for r in data[m]["robustness"] if r["tag"] == "escape_gold_removed"]
        rob[m] = {"injection_items": len(inj),
                  "clean_accuracy_pct": 100 * sum(clean[r["source_id"]]["correct"] for r in inj) / len(inj),
                  "injected_accuracy_pct": 100 * sum(r["correct"] for r in inj) / len(inj),
                  "followed_injection_pct": 100 * sum(r["pred"] == int(r["tag"].split(":")[1]) for r in inj) / len(inj),
                  "escape_items": len(esc), "chose_none_of_the_above_pct": 100 * sum(r["correct"] for r in esc) / len(esc)}
    summary["robustness"] = rob

    # 8. Unseen multiple-choice benchmarks.
    unseen = {}
    for b in ("HellaSwag", "WinoGrande", "OpenBookQA"):
        rows = {m: {r["id"]: r["correct"] for r in data[m]["unseen"] if r["task"] == b} for m in MODELS}
        ids = sorted(rows["v2"])
        c = {m: [rows[m][i] for i in ids] for m in MODELS}
        unseen[b] = {"n": len(ids), **{m: 100 * sum(c[m]) / len(ids) for m in MODELS},
                     "p_v2_vs_v1": mcnemar(c["v1"], c["v2"]), "ci_v2_vs_v1": bootstrap(c["v1"], c["v2"]),
                     "p_v2_vs_original": mcnemar(c["original"], c["v2"])}
    summary["unseen"] = unseen

    # 9. External classification, including KMMLU's 45 domains.
    ext = {}
    for t in ("SuperGPQA_domain", "SuperGPQA_difficulty", "KMMLU_domain", "KMMLU_difficulty"):
        ext[t] = {}
        for m in MODELS:
            rows = [r for r in data[m]["external"] if r["task"] == t]
            bal, f1 = per_class(rows)
            entry = {"n": len(rows), "accuracy": 100 * sum(r["correct"] for r in rows) / len(rows),
                     "balanced_accuracy": bal, "macro_f1": f1}
            if t == "KMMLU_domain":
                hi = [r for r in rows if r["gold"] >= 26]
                lo = [r for r in rows if r["gold"] < 26]
                entry.update({"acc_gold_A-Z": 100 * sum(r["correct"] for r in lo) / len(lo),
                              "acc_gold_AA-AS": 100 * sum(r["correct"] for r in hi) / len(hi),
                              "pred_share_AA-AS": 100 * sum(r["pred"] >= 26 for r in rows) / len(rows),
                              "gold_share_AA-AS": 100 * len(hi) / len(rows)})
            ext[t][m] = entry
    summary["external"] = ext

    # 10. Task-type routing: test is only eight task definitions.
    routing = {}
    for m in ("v1", "v2"):
        per_def = defaultdict(lambda: [0, 0])
        for i, r in by[m]["task_type_routing"].items():
            d = per_def[test_records[i]["source"]["config"]]
            d[0] += r["correct"]
            d[1] += 1
        routing[m] = {k: f"{a}/{n}" for k, (a, n) in sorted(per_def.items())}
    summary["routing_by_task_definition"] = routing

    # 11. Shared-state check and the pre-registered criteria.
    shared = json.loads((EVAL / "v2" / "shared-state-check.json").read_text())
    shared32 = json.loads((EVAL / "v2" / "shared-state-check-fp32.json").read_text())
    summary["shared_state"] = {k: v for k, v in shared.items() if k not in ("per_article", "mixed_types_shared")}
    summary["shared_state_fp32"] = {k: v for k, v in shared32.items() if k not in ("per_article", "mixed_types_shared")}
    p = pooled["v2"]
    criteria = {
        "1_many_options": {"acc_AA-BL_over_acc_A-Z": p["acc_gold_AA-BL"] / p["acc_gold_A-Z"],
                           "pred_share_over_gold_share": p["pred_share_AA-BL"] / p["gold_share_AA-BL"],
                           "pass": p["acc_gold_AA-BL"] >= .5 * p["acc_gold_A-Z"] and .5 <= p["pred_share_AA-BL"] / p["gold_share_AA-BL"] <= 1.5},
        "2_v1_tasks_kept": {"v2_macro": macro["v1_tasks"]["v2"], "v1_macro": macro["v1_tasks"]["v1"],
                            "pass": macro["v1_tasks"]["v2"] >= macro["v1_tasks"]["v1"] - 1},
        "3_calibration": {"choice_ece_calibrated": cal["v2"]["choice"]["calibrated"]["ece"],
                          "choice_nll_raw": cal["v2"]["choice"]["raw"]["nll"], "choice_nll_calibrated": cal["v2"]["choice"]["calibrated"]["nll"],
                          "pass": cal["v2"]["choice"]["calibrated"]["ece"] <= .05 and cal["v2"]["choice"]["calibrated"]["nll"] < cal["v2"]["choice"]["raw"]["nll"]},
        "4_shared_state_bf16": {"max_probability_difference": shared["shared_vs_independent_max_probability_difference"],
                                "choice_agreement": shared["shared_vs_independent_choice_agreement"], "pass": shared["passed"]},
        "5_kmmlu_late_codes": {"acc_gold_AA-AS": ext["KMMLU_domain"]["v2"]["acc_gold_AA-AS"], "chance": 100 / 45,
                               "pass": ext["KMMLU_domain"]["v2"]["acc_gold_AA-AS"] > 100 / 45},
    }
    summary["criteria"] = criteria
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_tables(summary)
    print(json.dumps(criteria, indent=1))


def fmt_p(p):
    return "<0.001" if p < .001 else f"{p:.3f}"


def write_tables(s):
    lines = ["# v2 evaluation tables", ""]
    lines += ["## Test accuracy (%)", "", "| task | n | original | v1 | v2 | v2−v1 (95% CI) | p | best text-light baseline |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for t, r in s["test"]["per_task"].items():
        lo, hi = r["ci_vs_v1"]
        lines.append(f"| {t} | {r['n']} | {r['original']:.2f} | {r['v1']:.2f} | {r['v2']:.2f} | {r['v2_minus_v1']:+.2f} ({lo:+.2f}, {hi:+.2f}) | "
                     f"{fmt_p(r['p_vs_v1'])} | {r['baseline']:.2f} ({r['baseline_name']}) |")
    m = s["test"]["macro"]
    lines += ["", f"Macro over v1 tasks: original {m['v1_tasks']['original']:.2f}, v1 {m['v1_tasks']['v1']:.2f}, v2 {m['v1_tasks']['v2']:.2f}.",
              f"Macro over new tasks: original {m['new_tasks']['original']:.2f}, v1 {m['new_tasks']['v1']:.2f}, v2 {m['new_tasks']['v2']:.2f}.", ""]
    lines += ["## Gold code range at K=60/64 (pooled)", "", "| model | acc gold A–Z | acc gold AA–BL | gold share AA–BL | pred share AA–BL |", "| --- | ---: | ---: | ---: | ---: |"]
    for mname, r in s["many_options_pooled_K60_64"].items():
        lines.append(f"| {mname} | {r['acc_gold_A-Z']:.2f} | {r['acc_gold_AA-BL']:.2f} | {r['gold_share_AA-BL']:.2f} | {r['pred_share_AA-BL']:.2f} |")
    lines += ["", "## Calibration on test (choice questions)", "", "| model | ECE raw | ECE cal | NLL raw | NLL cal | Brier raw | Brier cal |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mname, c in s["calibration"].items():
        a, b = c["choice"]["raw"], c["choice"]["calibrated"]
        lines.append(f"| {mname} | {a['ece']:.4f} | {b['ece']:.4f} | {a['nll']:.4f} | {b['nll']:.4f} | {a['brier']:.4f} | {b['brier']:.4f} |")
    lines += ["", "## Unseen benchmarks (%)", "", "| benchmark | n | original | v1 | v2 | p v2 vs v1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for b, r in s["unseen"].items():
        lines.append(f"| {b} | {r['n']} | {r['original']:.2f} | {r['v1']:.2f} | {r['v2']:.2f} | {fmt_p(r['p_v2_vs_v1'])} |")
    lines += ["", "## External classification (%)", "", "| task | model | accuracy | balanced | macro F1 |", "| --- | --- | ---: | ---: | ---: |"]
    for t, rr in s["external"].items():
        for mname, r in rr.items():
            lines.append(f"| {t} | {mname} | {r['accuracy']:.2f} | {r['balanced_accuracy']:.2f} | {r['macro_f1']:.2f} |")
    (EVAL / "report_tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
