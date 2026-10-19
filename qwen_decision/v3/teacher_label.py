"""Teacher distributions for phase-2 examples, from any checkpoint the v3 Scorer can load.

  python -m qwen_decision.v3.teacher_label --teacher models/d3-flash --name d3-flash \
      --sources clinc-train,ragtruth-train,... [--limit-per-source 20000]

The Scorer reproduces d3-family checkpoints (prompt and noncausal attention as in d3_runtime;
checked against d3-lite: 98.6% decision agreement, see runs/proxy/ours-on-d3weights). Each example
is scored with its stored option order and the probabilities are written per example id to
data/teacher/<name>/<source>.jsonl. Resumable: ids already written are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from qwen_decision.v3.engine import Scorer
from qwen_decision.v3.sources import common

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--limit-per-source", type=int, default=0)
    p.add_argument("--attention", default=None)
    p.add_argument("--layout", default=None)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--max-batch", type=int, default=16)
    args = p.parse_args()
    out = ROOT / "data/teacher" / args.name
    out.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args.teacher, attention=args.attention, layout=args.layout,
                    batch_tokens=args.batch_tokens, max_batch=args.max_batch)
    for source in args.sources.split(","):
        rows = common.load([source])
        if args.limit_per_source:
            rows = rows[:args.limit_per_source]
        path = out / f"{source}.jsonl"
        done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        started = time.time()
        with path.open("a") as f:
            # 256 examples per call; score_many batches their prompts under the token budget.
            for i in range(0, len(todo), 256):
                chunk = todo[i:i + 256]
                scored = scorer.score_many([(r["state"], {"q": r["question"]}) for r in chunk])
                for r, s in zip(chunk, scored):
                    f.write(json.dumps({"id": r["id"], "teacher": s["q"]}) + "\n")
                if (i // 256) % 10 == 0:
                    rate = (i + len(chunk)) / max(1e-9, time.time() - started)
                    print(f"{source}: {len(done) + i + len(chunk)}/{len(rows)} ({rate:.1f}/s)", flush=True)
        print(f"{source}: done {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
