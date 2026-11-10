"""Waits for the R1 public evaluation unit to finish, then: R3 train -> internal eval -> public text/vision."""
import fcntl
import subprocess
import sys
import time
import traceback

from qwen_decision.v3 import skill_mix as sm, train

ROOT = sm.ROOT
WAIT_FOR = "mix-ablation-public-r1"


def state(stage, **extra):
    train.atomic_json(ROOT / "pipeline.json", {"stage": stage, "updated_unix": time.time(), **extra})
    print("PIPELINE", stage, extra, flush=True)


def busy():
    out = subprocess.run(["systemctl", "--user", "is-active", WAIT_FOR], capture_output=True, text=True).stdout.strip()
    return out in ("active", "activating")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while busy():
                state("waiting_for_gpu", unit=WAIT_FOR)
                time.sleep(60)
            steps = [("run", ["qwen_decision.v3.skill_mix", "run"]), ("eval", ["qwen_decision.v3.skill_mix", "eval"]),
                     ("check-text", ["qwen_decision.v3.skill_mix_public", "check", "text"]),
                     ("check-vision", ["qwen_decision.v3.skill_mix_public", "check", "vision"]),
                     ("public-text", ["qwen_decision.v3.skill_mix_public", "run", "text"]),
                     ("public-vision", ["qwen_decision.v3.skill_mix_public", "run", "vision"])]
            for name, args in steps:
                if (ROOT.parent / "pause-request.json").exists():
                    state("paused_by_request")
                    return
                state(name)
                with (ROOT / (name + ".log")).open("a") as log:
                    subprocess.run([sys.executable, "-u", "-m", *args], stdout=log, stderr=subprocess.STDOUT, check=True)
            state("complete")
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
