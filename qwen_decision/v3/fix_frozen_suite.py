"""Make a rebuilt 0.3 suite byte-identical to the lab's frozen file.

`finish_suite.py` leaves RouterBench (6) and SGD (10) different from the frozen rows (all 10,000 and 2,500 payload
hashes). Two causes, both outside the kit's pinned sources:

- SGD: the frozen rows predate kit commit 2d7791c ("SGD builder: copy the dialogue history per row"). Before it,
  every row kept a reference to the growing history list, so each row holds the whole dialogue. The board keeps
  SGD out of the index for this reason, but the frozen hash includes it.
- RouterBench: each row carries calibration profiles that are float means from `sum()`. Python 3.12 changed
  `sum()` of floats to compensated summation, and the frozen rows were built with it. Under 3.11 every mean differs
  in the last digits. pandas 2 vs 3 makes no difference.

This rebuilds both benchmarks from the raw sources already in the kit's work dir, then rewrites their lines in the
selected-rows file, recomputing payload_sha256 and proxy_tokens exactly as the kit's freeze step does. Every other
line is copied unchanged. Run it with Python >= 3.12 and the kit on the path. The result must then pass the strict
import:

  PYTHONPATH=external/decision-index python3.12 qwen_decision/v3/fix_frozen_suite.py \\
      --work external/decision-index/work --rows external/decision-index/suite-0.3/selected-rows.jsonl.gz \\
      --scratch /tmp/suite-fix --out /tmp/suite-fix/selected-rows.jsonl
  python -m decision_index suite import --edition 0.3 --dir runs/suite-0.3-frozen --rows /tmp/suite-fix/selected-rows.jsonl \\
      --added-rows external/decision-index/suite-0.3/added-rows.jsonl.gz --gsm8k-rows external/decision-index/suite-0.3/gsm8k-rows.jsonl.gz

Checked 2026-10-23: uncompressed sha256 b2b56d6f… (the edition's rows_sha256), and the normalized RouterBench and SGD
files match the sha256 values in the kit's hub/manifest.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from decision_index import editions
from decision_index.suite.build import adapters_mechanical, adapters_scored
from decision_index.suite.build.layout import REFERENCE_MODEL, Layout
from decision_index.suite.io import dumps, read_jsonl

TRACKS = ("RouterBench-0shot", "RouterBench-5shot", "SGD-service-given-intent")


def rebuild(work, scratch):
    """Run the two builders into scratch/, reading raw sources from the kit's work dir."""
    suite = scratch / "artifacts/benchmark-suite"
    suite.mkdir(parents=True, exist_ok=True)
    raw = suite / "raw"
    if not raw.exists():
        raw.symlink_to((work / "artifacts/benchmark-suite/raw").resolve())
    layout = Layout(scratch)
    adapters_mechanical.list = lambda x: x  # the SGD builder before 2d7791c: rows share the growing history list
    try:
        adapters_mechanical.build_sgd(layout)
    finally:
        del adapters_mechanical.list
    adapters_scored.routerbench(layout)
    return layout.normalized


def splice(rows_gz, normalized, out):
    new = {(t, r["id"]): r for t in TRACKS for r in read_jsonl(normalized / f"{t}.jsonl")}
    replaced = 0
    with gzip.open(rows_gz, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            old = json.loads(line)
            e = old["_evaluation"]
            if e["catalog_id"] in (6, 10):
                r = new[(e["track"], old["id"])]
                payload = dumps({"model": REFERENCE_MODEL, "state": r["state"], "questions": r["questions"]})
                r["_evaluation"] = {**e, "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                                    "proxy_tokens": max(1, len(payload) // 4)}
                line = dumps(r) + "\n"
                replaced += 1
            g.write(line)
    return replaced


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True, help="the kit's rebuild work dir (raw sources)")
    p.add_argument("--rows", type=Path, required=True, help="the rebuilt selected-rows.jsonl.gz to fix")
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="fixed selected-rows.jsonl (uncompressed)")
    args = p.parse_args()
    if sys.version_info < (3, 12):
        raise SystemExit("RouterBench needs Python >= 3.12 (compensated float sum); see the module docstring")
    normalized = rebuild(args.work, args.scratch)
    manifest = json.loads((Path(adapters_scored.__file__).parents[3] / "hub/manifest.json").read_text())
    expected = {Path(s["path"]).stem: s["sha256"] for b in manifest["benchmarks"] if b["catalog_id"] in (6, 10)
                for s in b["sources"]}
    tracks = {t: hashlib.sha256((normalized / f"{t}.jsonl").read_bytes()).hexdigest() == expected[t] for t in TRACKS}
    replaced = splice(args.rows, normalized, args.out)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"normalized_match": tracks, "replaced_lines": replaced, "sha256": sha,
                      "match": sha == editions.get("0.3")["rows_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
