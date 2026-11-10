"""Continue preparation -> runtime check -> immutable plan -> joint training without live monitoring."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/champion-v1/joint-g7-v1"


def status(stage, **extra):
    payload = {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(), **extra}
    path = ROOT / "pipeline.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-pid", type=int, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert not (ROOT / "plan.json").exists(), "An immutable plan already exists; inspect before restarting"
            status("waiting_for_prepared_inputs", preparation_pid=args.preparation_pid)
            deadline = time.monotonic() + 7200
            while not (ROOT / "prepared.json").exists():
                command = Path(f"/proc/{args.preparation_pid}/cmdline")
                assert command.exists() and b"champion_joint_data" in command.read_bytes(), "Preparation exited before writing verified inputs; inspect prepare.log"
                assert time.monotonic() < deadline, "Preparation exceeded the fixed 2-hour wait"
                time.sleep(10)
            env = dict(os.environ, PYTHONPATH=f"{REPO}:{REPO/'external/decision-index'}",
                       OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            for stage in ("check", "freeze", "run"):
                status(stage)
                with (ROOT / f"{stage}.log").open("x") as log:
                    subprocess.run([sys.executable, "-u", "-m", "qwen_decision.v3.champion_joint_train", stage],
                                   cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((ROOT / "report.json").read_text())
            assert result["complete"] and result["export_replay_exact"]
            assert set(result["trained"]) == {"text-dev", "image-dev", "text-diagnostic", "vision-diagnostic"}
            status("complete", verified=True, training_complete=True)
        except Exception:
            status("failed", error=traceback.format_exc())
            raise


if __name__ == "__main__": main()
