"""Download public data files without executing dataset loading scripts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/text-decisions-v1/raw"
SPECS = {
    "allenai/ai2_arc": [f"{c}/{s}-00000-of-00001.parquet" for c in ("ARC-Challenge", "ARC-Easy") for s in ("train", "validation", "test")],
    "tau/commonsense_qa": [f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation")],
    "stanfordnlp/sst2": [f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation")],
    "fancyzhx/ag_news": [f"data/{s}-00000-of-00001.parquet" for s in ("train", "test")],
    "dair-ai/emotion": [f"split/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    "nyu-mll/glue": [f"{c}/{s}-00000-of-00001.parquet" for c in ("rte", "mrpc", "cola") for s in ("train", "validation")],
    "ehovy/race": [f"all/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    "mouryat9/CogBench": [f"data/{s}.jsonl" for s in ("train", "validation", "test", "human_annotated")],
    "furonghuang-lab/Easy2Hard-Bench": [f"E2H-AMC/{s}-00000-of-00001.parquet" for s in ("train", "eval")],
    "PolyAI/banking77": ["banking77.py", "dataset_infos.json"],
}

def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "text-decision-data-preparation/1"}), timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
    raise AssertionError()

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload = get(url)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {"url": url, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}

def repo_one(item):
    repo, paths = item
    directory = OUT / repo.replace("/", "__")
    cached = directory / "manifest.json"
    if cached.exists():
        data = json.loads(cached.read_text())
        for entry in data["files"]:
            assert sha(ROOT / entry["path"]) == entry["sha256"]
        return data
    info = json.loads(get("https://huggingface.co/api/datasets/" + repo))
    revision = info["sha"]
    siblings = {x["rfilename"] for x in info["siblings"]}
    assert set(paths) <= siblings
    files = []
    if "README.md" in siblings:
        paths = ["README.md", *paths]
    for name in paths:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        files.append(download(url, directory / name))
    try:
        splits = json.loads(get("https://datasets-server.huggingface.co/splits?dataset=" + urllib.parse.quote(repo, safe="")))
    except Exception as error:
        splits = {"unavailable": str(error)}
    data = {"repo_id": repo, "revision": revision, "license": info.get("cardData", {}).get("license"), "splits": splits, "files": files}
    write_json(cached, data)
    print(f"Downloaded {repo}: {sum(f['bytes'] for f in files):,} bytes", flush=True)
    return data

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        data = list(pool.map(repo_one, SPECS.items()))
    write_json(OUT / "sources.json", data)
    print("All Hub source files downloaded", flush=True)

if __name__ == "__main__":
    main()
