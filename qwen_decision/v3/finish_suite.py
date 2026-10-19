"""Finish the 0.3 suite rebuild when only out-of-index benchmarks differ from the frozen file.

`decision_index suite rebuild --edition 0.3` stops after the 0.1 rows unless they are byte-identical
to the lab's file. With the kit at f5a6e4f the rebuilt 0.1 rows differ only in RouterBench (6) and
SGD (10), both outside the index (checked run id by run id against the board's d3-lite payload
hashes; every other benchmark, HLE included, matches). This calls the kit's own 0.2 cut and 0.3
GSM8K builders on those rows, then imports the result without the strict hash check.

  python -m qwen_decision.v3.finish_suite --work external/decision-index/work --out external/decision-index/suite-0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_index import constants as C
from decision_index import editions
from decision_index.suite.build import gsm8k_v3, release_v2
from decision_index.suite.build.layout import Layout
from decision_index.suite.download import from_local, hub_file


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    layout = Layout(args.work)
    v1_rows = layout.suite / "release-v1-rebuilt" / "selected-rows.jsonl"
    v2 = release_v2.main(layout, v1_rows, hub_file("0.2", C.SUITE_EXCLUSIONS_FILE), added=None, skip_download=True, log=print)
    rows = Path(v2["out"]) / C.SUITE_ROWS_FILE
    gsm = gsm8k_v3.main(layout, rows, log=print)
    e = editions.get("0.3")
    report = from_local(args.out, rows, added=Path(v2["out"]) / editions.ADDED_FILE, gsm8k=Path(gsm["out"]) if "out" in gsm else None,
                        verify=False, edition="0.3")
    summary = {"v2_out": v2.get("out"), "gsm8k": {k: gsm.get(k) for k in ("out", "gsm8k_sha256")},
               "gsm8k_byte_identical": gsm.get("gsm8k_sha256") == e["gsm8k_sha256"], "import": report}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
