"""Persistent local training, followed by held-out evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def status(out, value):
    tmp = out / "pipeline.json.tmp"
    tmp.write_text(json.dumps({**value, "updated_at_unix": time.time()}, indent=2) + "\n")
    tmp.replace(out / "pipeline.json")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "training/text-decision-full-v1")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "pipeline.json").exists(), "Preserve existing runs."
    commands = [
        ("training", [sys.executable, "-u", str(ROOT / "train_text_decisions.py"), "--output", str(args.output), "--max-steps", str(args.max_steps)]),
        ("heldout_evaluation", [sys.executable, "-u", str(ROOT / "evaluate_trained_text_decisions.py"), "--model", str(args.output / "model"), "--output", str(args.output / "evaluation")]),
    ]
    for phase, command in commands:
        status(args.output, {"status": "running", "phase": phase, "command": command})
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            status(args.output, {"status": "failed", "phase": phase, "exit_code": result.returncode})
            return result.returncode
    status(args.output, {"status": "complete", "model": str(args.output / "model"), "evaluation": str(args.output / "evaluation/summary.json")})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
