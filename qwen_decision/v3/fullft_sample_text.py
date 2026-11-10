"""Small public-text spot check (user 11-10: "작게 평가셋 일부만"): 40 seeded rows per counted panel.
best/ (fullft-scale-v1 step 81,920) runs with R1's public engine policy (d3, orders 1, noncausal, 8k batch tokens);
GLiClass v4 edge (author-published predictions), v2 (joint-g7-v2) and R1 are read from their stored full runs.
Metric: top-choice accuracy per question (noul: p>.5), questions without a gold skipped; macro over panels.
A development observation only; not the official index and never used for training or selection."""
import collections, json, random, sys
from pathlib import Path

OUT = Path("runs/champion-v1/fullft-scale-v1/public-sample")
ROWS = Path("runs/champion-v1/public-eval-v1/text-counted-rows.jsonl")
OTHERS = {"gliclass-v4-edge (1위)": "runs/champion-v1/public-eval-v1/references/knowledgator__decision-index-results/runs/gliclass-v4-edge/results.jsonl",
          "v2 joint-g7-v2": "runs/champion-v1/joint-g7-v2/public-text/results.jsonl",
          "R1 LoRA": "runs/champion-v1/mix-ablation-v1/r1/public/text/results.jsonl"}
PER = 40


def sample():
    by = collections.defaultdict(list)
    for line in ROWS.open():
        r = json.loads(line)
        if any(v is not None for v in r["expected"].values()):
            by[r["_evaluation"]["catalog_id"]].append(line)
    rng = random.Random(20261110)
    picked = [l for c in sorted(by) for l in rng.sample(by[c], min(PER, len(by[c])))]
    (OUT / "rows.jsonl").write_text("".join(picked))
    return len(picked), len(by)


def right(answer, gold, qtype):
    if qtype == "noul":
        return (answer["noul"] > .5) == bool(gold)
    return answer["choice"] == gold


def accuracy(path, rows):
    res = {}
    for line in open(path):
        r = json.loads(line)
        if r["run_id"] in rows:
            res[r["run_id"]] = r
    per = collections.defaultdict(list)
    missing = 0
    for rid, row in rows.items():
        r = res.get(rid)
        for q, gold in row["expected"].items():
            if gold is None:
                continue
            key = (row["_evaluation"]["catalog_id"], row["_evaluation"]["track"])
            if not r or r.get("status") != "ok" or q not in r["response"]["answers"]:
                per[key].append(0.); missing += 1
            else:
                per[key].append(float(right(r["response"]["answers"][q], gold, row["questions"][q]["type"])))
    return {k: sum(v) / len(v) for k, v in per.items()}, missing


def compare():
    rows = {json.loads(l)["_evaluation"]["run_id"]: json.loads(l) for l in (OUT / "rows.jsonl").open()}
    models = {"best (fullft 81,920)": OUT / "best/results.jsonl", **OTHERS}
    table = {m: accuracy(p, rows) for m, p in models.items()}
    keys = sorted(next(iter(table.values()))[0])
    report = {"rows": len(rows), "panels": len(keys), "metric": "top-choice accuracy, macro over panels",
              "macro": {m: sum(t[0][k] for k in keys) / len(keys) for m, t in table.items()},
              "missing_or_error_questions": {m: t[1] for m, t in table.items()},
              "panels_detail": {f"{c}:{name}": {m: round(t[0][(c, name)], 4) for m, t in table.items()} for c, name in keys}}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["macro"], indent=1, ensure_ascii=False)); print(report["missing_or_error_questions"])


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        print(sample())
    elif sys.argv[1] == "run":
        from decision_index.runner import run
        run("qwen_decision.v3.engine:V3Engine",
            {"model": "runs/champion-v1/fullft-scale-v1/best", "attention": "noncausal", "deltanet": "causal",
             "layout": "d3", "batch_tokens": 8192, "max_batch": 8, "orders": 1}, OUT / "rows.jsonl", OUT / "best", compact=True)
    elif sys.argv[1] == "crystal":  # deployed Crystal (HF cd943da), native tokenizer, same policy
        from decision_index.runner import run
        run("qwen_decision.v3.engine:V3Engine",
            {"model": "models/crystal-0.8b-cd943da", "attention": "noncausal", "deltanet": "causal",
             "layout": "d3", "batch_tokens": 8192, "max_batch": 8, "orders": 1}, OUT / "rows.jsonl", OUT / "crystal", compact=True)
    else:
        compare()
