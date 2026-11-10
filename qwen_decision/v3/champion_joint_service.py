"""Run the already frozen pilot under a user service, independent of the tool session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path("runs/champion-v1/joint-g7-v1")


def status(state, **extra):
    path = ROOT / "service.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"state": state, "pid": os.getpid(), "updated_unix": time.time(), **extra}, indent=2)+"\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=3)
    parser.add_argument("--prepare-plan", action="store_true")
    args = parser.parse_args()
    assert not (ROOT / "train.jsonl").exists(), "This restart is only valid before any actual training"
    assert not (ROOT / "report.json").exists()
    try:
        commands = ("check", "freeze", "run") if args.prepare_plan else ("run",)
        for command in commands:
            logfile = f"{command}-attempt{args.attempt}.log"
            status("starting", command=command, log=logfile)
            with (ROOT / logfile).open("x") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "qwen_decision.v3.champion_joint_train", command],
                                         stdout=log, stderr=subprocess.STDOUT)
                while True:
                    status("running", command=command, child_pid=child.pid, log=logfile)
                    try:
                        child.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            if child.returncode:
                raise RuntimeError(f"{command} exited with status {child.returncode}; inspect {logfile}")
        result = json.loads((ROOT / "report.json").read_text())
        assert result["complete"] and result["export_replay_exact"]
        status("complete", verified=True)
    except BaseException:
        status("failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__": main()
