"""Public-suite sample: check the rebuilt rows against the board, then score any run with the kit's own scorer.

  # 1. A sample suite directory: N linked groups per benchmark, payloads checked against the board's d3-lite run.
  python -m qwen_decision.v3.board_check sample --suite-dir external/decision-index/suite-0.3 \
      --per-benchmark 40 --out runs/board/sample-40
  # 2. Run an engine over it and score it exactly as the board does (kit CLI, nothing reimplemented):
  python -m decision_index run --suite-dir runs/board/sample-40 --no-verify --engine <module:Class> \
      --option ... --out runs/board/<name>
  python -m qwen_decision.v3.board_check score --suite-dir runs/board/sample-40 \
      --results runs/board/<name>/results.jsonl --out runs/board/score-<name>
  # 3. Decision agreement between two runs on the sample, per benchmark:
  python -m qwen_decision.v3.board_check compare --suite-dir runs/board/sample-40 \
      --results runs/board/<name>/results.jsonl --reference runs/board-d3-lite/results.jsonl.gz

Scores on a sample are estimates of the public index, not the board's numbers; the
private parts are not covered here (see proxy_eval.py). `score` is the kit's own
scorer; the only change is that panel benchmarks with no rows in the suite
directory (e.g. HLE before its gated download) leave the aggregate, which also
shifts the area weights slightly. Compare models on the same sample only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from decision_index import editions
from decision_index.suite.io import Suite, dumps, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "runs/board-d3-lite/results.jsonl.gz"


def sample(args) -> None:
    suite = Suite(args.suite_dir)
    e = suite.edition
    added_ids = set()
    if suite.added_path:
        added_ids = {r["_evaluation"]["catalog_id"] for r in read_jsonl(suite.added_path)}
    groups = defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        x = r["_evaluation"]
        groups[(x["catalog_id"], x["group_id"])].append(r)
    by_benchmark = defaultdict(list)
    for (n, gid), rows in groups.items():
        by_benchmark[n].append((hashlib.sha256(f"{args.seed}:{n}:{gid}".encode()).hexdigest(), rows))
    chosen = [r for n in sorted(by_benchmark) for _, rows in sorted(by_benchmark[n])[:args.per_benchmark] for r in rows]
    board = {}
    with gzip.open(BOARD, "rt") as f:
        for line in f:
            b = json.loads(line)
            board[b["run_id"]] = b["payload_sha256"]
    check = Counter()
    for r in chosen:
        x = r["_evaluation"]
        found = board.get(x["run_id"])
        check["missing_on_board" if found is None else "match" if found == x["payload_sha256"] else "mismatch"] += 1
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    files = {editions.ROWS_FILE: [], editions.ADDED_FILE: [], editions.GSM8K_FILE: []}
    for r in chosen:
        x = r["_evaluation"]
        name = (editions.GSM8K_FILE if suite.gsm8k_path and x["catalog_id"] == 30 else
                editions.ADDED_FILE if x["catalog_id"] in added_ids else editions.ROWS_FILE)
        files[name].append(r)
    for name, rows in files.items():
        if name == editions.ROWS_FILE or (name == editions.ADDED_FILE and suite.added_path) or (name == editions.GSM8K_FILE and suite.gsm8k_path):
            with gzip.open(out / name, "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(dumps(r) + "\n")
    for name in (editions.EXCLUSIONS_FILE, editions.MANIFEST_FILE):
        if (suite.directory / name).exists():
            shutil.copyfile(suite.directory / name, out / name)
    report = {"edition": e["id"], "source": str(suite.directory), "per_benchmark_groups": args.per_benchmark,
              "seed": args.seed, "rows": len(chosen), "benchmarks": len(by_benchmark),
              "rows_by_benchmark": dict(Counter(r["_evaluation"]["catalog_id"] for r in chosen)),
              "payload_vs_board_d3_lite": dict(check)}
    (out / "sample.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "benchmarks", "payload_vs_board_d3_lite")}))


def read_results(path: Path) -> dict[str, dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return {r["run_id"]: r for r in map(json.loads, f)}


def decisions(row: dict) -> dict | None:
    if row.get("status") != "ok":
        return None
    out = {}
    for key, a in row["response"]["answers"].items():
        out[key] = a.get("choice") if a["type"] == "choice" else a["noul"] >= .5
    return out


def compare(args) -> None:
    suite = Suite(args.suite_dir)
    ours, reference = read_results(args.results), read_results(args.reference)
    stats = defaultdict(Counter)
    for r in suite.rows(apply_exclusions=True):
        x = r["_evaluation"]
        a, b = ours.get(x["run_id"]), reference.get(x["run_id"])
        if a is None or b is None:
            stats[x["dataset"]]["unpaired"] += 1
            continue
        da, db = decisions(a), decisions(b)
        for key in r["questions"]:
            s = stats[x["dataset"]]
            s["fields"] += 1
            s["agree"] += int(da is not None and db is not None and da.get(key) == db.get(key))
    total = Counter()
    for s in stats.values():
        total.update(s)
    table = {name: {"fields": s["fields"], "agreement": s["agree"] / s["fields"] if s["fields"] else None,
                    "unpaired_rows": s["unpaired"]} for name, s in sorted(stats.items())}
    print(json.dumps({"overall_field_agreement": total["agree"] / max(1, total["fields"]), "fields": total["fields"],
                      "by_benchmark": table}, indent=2))


def score(args) -> None:
    from decision_index import pipeline
    from decision_index.scoring import index02

    suite = Suite(args.suite_dir)
    present = {r["_evaluation"]["catalog_id"] for r in suite.rows(apply_exclusions=True)}
    original = index02.spec

    def without_missing(edition="0.2"):
        s = copy.deepcopy(original(edition))
        for area in s["areas"]:
            area["benchmarks"] = [n for n in area["benchmarks"] if n in present]
        return s

    index02.spec = without_missing
    args.out.mkdir(parents=True, exist_ok=True)
    scores = pipeline.score_run(suite, args.results, args.results.parent.name, args.out)
    missing = sorted({n for a in original(suite.edition["id"])["areas"] for n in a["benchmarks"]} - present)
    brief = {"public_index_skill": scores["scores"]["balanced_skill"], "raw": scores["scores"]["balanced_raw"],
             "areas": {a["id"]: round(a["skill"], 4) for a in scores["areas"]}, "missing_from_aggregate": missing}
    (args.out / "brief.json").write_text(json.dumps(brief, indent=2) + "\n")
    print(json.dumps(brief))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--suite-dir", type=Path, required=True)
    s.add_argument("--per-benchmark", type=int, default=40, help="linked groups per benchmark")
    s.add_argument("--seed", type=int, default=20261016)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--suite-dir", type=Path, required=True)
    c.add_argument("--results", type=Path, required=True)
    c.add_argument("--reference", type=Path, default=BOARD)
    sc = sub.add_parser("score")
    sc.add_argument("--suite-dir", type=Path, required=True)
    sc.add_argument("--results", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"sample": sample, "compare": compare, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
