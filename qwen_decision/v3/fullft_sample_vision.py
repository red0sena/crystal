"""Small public-vision spot check (user 11-10): 60 seeded rows per benchmark (11 benchmarks, 660 requests).
best/ (text full FT, vision tower = base) and deployed Crystal (cd943da) run with R1's public vision policy
(VisionEngine, prompt "decide", noncausal, 1,638,400 px); QEV-0.8B (vision #1), v2 and R1 come from stored full runs.
Metric: top-choice accuracy, macro over benchmarks. Development observation only."""
import collections, json, random, sys
from pathlib import Path

OUT = Path("runs/champion-v1/fullft-scale-v1/public-sample")
SUITE = Path("runs/champion-v1/public-eval-v1/vision-suite")
BASE = "models/qwen3.5-0.8b"
STORED = {"QEV-0.8B (이미지 1위)": "runs/champion-v1/public-eval-v1/qev-vision/results.jsonl",
          "v2 joint-g7-v2": "runs/champion-v1/joint-g7-v2/public-vision/results.jsonl",
          "R1 LoRA": "runs/champion-v1/mix-ablation-v1/r1/public/vision/results.jsonl"}
RUN = {"best": ("runs/champion-v1/fullft-scale-v1/best", None),
       "crystal": ("models/crystal-0.8b-cd943da", "models/crystal-0.8b-cd943da")}
PER = 60


def sample():
    by = collections.defaultdict(list)
    for line in (SUITE / "vision-rows.jsonl").open():
        by[json.loads(line)["benchmark"]].append(line)
    rng = random.Random(20261110)
    picked = [l for b in sorted(by) for l in rng.sample(by[b], min(PER, len(by[b])))]
    (OUT / "vision-rows.jsonl").write_text("".join(picked))
    print(len(picked))


def run(name):
    from qwen_decision.v3 import champion_full_inputs as data
    from decision_index.vision.run import run as runner
    model, processor = RUN[name]
    runner("qwen_decision.v3.vision_engine:VisionEngine",
           {"model": model, "attention": "noncausal", "prompt": "decide", "processor": processor or str(data.BASE),
            "max_pixels": 1638400}, SUITE, OUT / f"vision-{name}", rows_path=OUT / "vision-rows.jsonl",
           workers=1, chunk=16, compact=True)


def compare():
    rows = {json.loads(l)["_evaluation"]["run_id"]: json.loads(l) for l in (OUT / "vision-rows.jsonl").open()}
    paths = {"best": OUT / "vision-best/results.jsonl", "crystal": OUT / "vision-crystal/results.jsonl", **STORED}
    res = {m: {json.loads(l)["run_id"]: json.loads(l) for l in open(p) if json.loads(l)["run_id"] in rows} for m, p in paths.items()}
    recs = []
    for rid, row in rows.items():
        v = {}
        for m in paths:
            r = res[m].get(rid)
            ok = r is not None and r["status"] == "ok"
            v[m] = sum(ok and q in r["response"]["answers"] and r["response"]["answers"][q].get("choice") == g
                       for q, g in row["expected"].items()) / len(row["expected"])
        recs.append((row["benchmark"], v))
    def macro(rs, m):
        by = collections.defaultdict(list)
        for b, v in rs: by[b].append(v[m])
        return sum(sum(x) / len(x) for x in by.values()) / len(by)
    by = collections.defaultdict(list)
    for r in recs: by[r[0]].append(r)
    rng = random.Random(0); boots = {m: [] for m in paths if m != "best"}
    for _ in range(2000):
        bs = [x for b in by for x in [rng.choice(by[b]) for _ in by[b]]]
        bb = macro(bs, "best")
        for m in boots: boots[m].append(bb - macro(bs, m))
    report = {"rows": len(rows), "metric": "top-choice accuracy, macro over 11 benchmarks",
              "missing": {m: sum(1 for rid in rows if rid not in res[m] or res[m][rid]["status"] != "ok") for m in paths},
              "macro": {m: macro(recs, m) for m in paths},
              "bootstrap_best_minus": {m: [sum(d) / len(d), sorted(d)[50], sorted(d)[1949]] for m, d in boots.items()},
              "per_benchmark": {b: {m: sum(r[1][m] for r in by[b]) / len(by[b]) for m in paths} for b in sorted(by)}}
    (OUT / "vision-comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("missing", "macro", "bootstrap_best_minus")}, indent=1, ensure_ascii=False))
    ms = list(paths)
    print("benchmark".ljust(28) + "".join(m[:9].rjust(10) for m in ms))
    for b, v in report["per_benchmark"].items():
        print(b[:28].ljust(28) + "".join(f"{v[m]:10.3f}" for m in ms))


if __name__ == "__main__":
    {"sample": sample, "compare": compare}.get(sys.argv[1], lambda: run(sys.argv[1]))()
