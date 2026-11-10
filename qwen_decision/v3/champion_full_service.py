"""Persistent sequential supervisor for the full enriched LoRA run."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback

from qwen_decision.v3 import champion_full_inputs as data, champion_audit as audit, train

ROOT = data.ROOT


def state(stage, **extra):
    train.atomic_json(ROOT/"pipeline.json", {"stage": stage, "updated_unix": time.time(), **extra})
    print("PIPELINE", stage, extra, flush=True)


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    with (ROOT/"pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            deadline = time.monotonic()+21600
            while not (ROOT/"prepared.json").exists():
                p = ROOT/"input-status.json"
                if p.exists() and audit.read(p)["stage"] == "failed":
                    raise RuntimeError("Full CPU input preflight failed; inspect input-status.json")
                if time.monotonic() > deadline:
                    raise RuntimeError("Input preparation exceeded six hours")
                state("input_preflight")
                time.sleep(30)
            state("review_extreme_inputs")
            with (ROOT/"input-review.log").open("a") as log:
                subprocess.run([sys.executable, "-u", "-m", "qwen_decision.v3.champion_input_review"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for stage in ("check", "freeze", "run", "eval", "eval-text", "eval-vision"):
                if stage == "check" and (ROOT/"plan.json").exists():
                    continue
                state(stage)
                with (ROOT/(stage+".log")).open("a") as log:
                    subprocess.run([sys.executable, "-u", "-m", "qwen_decision.v3.champion_full_train", stage],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            assert audit.read(ROOT/"report.json")["export_replay_exact"]
            assert all(audit.read(ROOT/("public-"+m)/"completion.json")["complete"] for m in ("text", "vision"))
            state("complete", training_complete=True, both_public_evaluations_complete=True)
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
