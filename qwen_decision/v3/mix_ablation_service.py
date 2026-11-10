"""Sequential supervisor: train r1, train r2, then evaluate v2/r1/r2 on the v3 dev files and compare.

Each stage is resumable (checkpoints / per-dataset summaries). A pause-request.json in the run root stops it.
"""
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback

from qwen_decision.v3 import mix_ablation as m, train

ROOT = m.ROOT
STAGES = (("run", "r1"), ("run", "r2"), ("eval", None))


def state(stage, **extra):
    train.atomic_json(ROOT / "pipeline.json", {"stage": stage, "updated_unix": time.time(), **extra})
    print("PIPELINE", stage, extra, flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for command, arm in STAGES:
                if m.paused():
                    state("paused_by_request")
                    return
                name = command + (f"-{arm}" if arm else "")
                state(name)
                args = [sys.executable, "-u", "-m", "qwen_decision.v3.mix_ablation", command] + (["--arm", arm] if arm else [])
                with (ROOT / (name + ".log")).open("a") as log:
                    subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            state("complete")
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
