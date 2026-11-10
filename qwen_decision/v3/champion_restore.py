"""Restore only the pinned inputs of the joint g7 text/vision pilot."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from huggingface_hub import HfApi, hf_hub_download

from qwen_decision.v3.champion_audit import DATA_REPO, DATA_REVISION, sha
from qwen_decision.v3.train import atomic_json

ROOT = Path("runs/champion-v1/joint-g7-v1")


def run():
    started = time.monotonic()
    plan = json.loads((ROOT / "vision-download-plan.json").read_text())
    targets = [{**r, "directory": str(ROOT / "raw/cauldron")} for r in plan["files"]]
    names = [f"data/mix/v3-gold7{suffix}" for suffix in (".jsonl", ".dev.jsonl", ".manifest.json")]
    names += ["data/decontam/" + s for s in ("meta.json", "shingles.npy", "short.txt.gz")]
    info = HfApi().dataset_info(DATA_REPO, revision=DATA_REVISION, files_metadata=True)
    for f in info.siblings:
        if f.rfilename in names:
            targets.append({"repo": DATA_REPO, "revision": DATA_REVISION, "file": f.rfilename,
                            "bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None,
                            "directory": str(ROOT / "restored")})
    assert len(targets) == len(plan["files"]) + len(names)
    atomic_json(ROOT / "restore-plan.json", {"files": targets, "total_bytes": sum(r["bytes"] for r in targets)})
    atomic_json(ROOT / "status.json", {"state": "restoring", "training_started": False})
    def one(row):
        p = Path(hf_hub_download(row["repo"], row["file"], revision=row["revision"], repo_type="dataset",
                                local_dir=row["directory"]))
        digest = sha(p)
        assert p.stat().st_size == row["bytes"] and (row["sha256"] is None or digest == row["sha256"]), row["file"]
        print("RESTORED", row["file"], p.stat().st_size, flush=True)
        return {**row, "path": str(p), "sha256": digest}
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(one, targets))
    atomic_json(ROOT / "restoration.json", {"complete": True, "files": done, "seconds": time.monotonic() - started,
                                           "source_sha256": sha(__file__)})
    atomic_json(ROOT / "status.json", {"state": "inputs_restored", "training_started": False,
                                      "seconds": time.monotonic() - started})


if __name__ == "__main__":
    try:
        run()
    except Exception:
        atomic_json(ROOT / "status.json", {"state": "failed", "stage": "restore", "traceback": traceback.format_exc()})
        raise
