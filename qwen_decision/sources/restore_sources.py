"""Restore the recorded model/data sources; verify their original SHA256 hashes.

This uses only Python's standard library. It does not query a moving Hub main
branch, train a model, restore old prediction rows, or delete anything.
Run --list to inspect the plan, --verify-only to check existing files, or select
--groups base,v1,v2,unseen,external to download the inputs you need.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs/download_catalog.json"
ARCHIVE = ROOT / "docs/archive"
GROUPS = {"base", "v1", "v2", "unseen", "external"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group(path: str) -> str:
    if path.startswith("data/text-decisions-v1/raw/"):
        return "v1"
    if path.startswith("data/text-decisions-v2/raw/"):
        return "v2"
    if path.startswith("data/unseen-text-v1/raw/"):
        return "unseen"
    if path.startswith("data/external-classification-candidates-v1/raw/"):
        return "external"
    return "base"


def destination(relative: str) -> Path:
    # Archived paths are data, not shell commands. Reject traversal and paths
    # outside this project, including any intervening symlink to another tree.
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in {"model", "data"}:
        raise ValueError(f"Invalid source path: {relative}")
    path = ROOT / rel
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Source path escapes the project: {relative}")
    return path


def check(entry: dict) -> bool:
    path = destination(entry["path"])
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]


def download(entry: dict) -> None:
    path = destination(entry["path"])
    if path.exists():
        if not check(entry):
            raise ValueError(f"Existing file differs from recorded source: {entry['path']}")
        return
    if not entry["url"].startswith("https://"):
        raise ValueError("Only HTTPS sources are supported.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".restore-partial")
    for attempt in range(4):
        try:
            req = urllib.request.Request(entry["url"], headers={"User-Agent": "qwen-decision-reproduction/1"})
            with urllib.request.urlopen(req, timeout=90) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size != entry["bytes"] or sha(partial) != entry["sha256"]:
                raise ValueError(f"Downloaded source changed: {entry['path']}")
            partial.replace(path)
            print(f"restored {entry['path']}", flush=True)
            return
        except ValueError:
            partial.unlink(missing_ok=True)
            raise  # A hash mismatch must not silently fall back to current data.
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def restore_metadata(selected: set[str]) -> None:
    # Source manifests carry pinned commits and the exact chosen NI task list.
    # Only restore source metadata, never a stale prepared-dataset manifest or
    # completed evaluation manifest that would make a new run skip real work.
    for p in sorted((ARCHIVE / "data").rglob("*")):
        if not p.is_file():
            continue
        relative = p.relative_to(ARCHIVE).as_posix()
        source_metadata = ("/raw/" in relative and p.name in {"manifest.json", "sources.json", "tree.json", "dataset_infos.json"})
        inspection = (relative.startswith("data/external-classification-candidates-v1/")
                      and p.name in {"inspection.json", "kmmlu-inspection.json", "source-overlap-review.json"})
        benchmark_metadata = relative.startswith("data/choice-benchmarks/") and p.name.endswith(".metadata.json")
        bucket = "external" if inspection else "base" if benchmark_metadata else group(relative)
        if bucket not in selected or not (source_metadata or inspection or benchmark_metadata):
            continue
        target = destination(relative)
        if target.exists() and sha(target) != sha(p):
            raise ValueError(f"Existing metadata differs: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="base,v1,v2,unseen,external")
    parser.add_argument("--list", action="store_true", help="Print counts and URLs; download nothing")
    parser.add_argument("--verify-only", action="store_true", help="Check existing sources; download nothing")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = set(args.groups.split(","))
    if not selected or not selected <= GROUPS or args.workers < 1:
        parser.error("Choose known groups and at least one worker.")
    entries = [e for e in json.loads(CATALOG.read_text())["files"] if group(e["path"]) in selected]
    for entry in entries:
        destination(entry["path"])
    print(f"{len(entries)} source files; {sum(e['bytes'] for e in entries) / 2**30:.3f} GiB", flush=True)
    if args.list:
        for e in entries:
            print(f"{group(e['path'])}: {e['path']} <- {e['url']}")
        return
    if args.verify_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(check, entries))
        missing = [e["path"] for e, ok in zip(entries, results) if not ok]
        if missing:
            raise SystemExit("Missing or changed sources:\n" + "\n".join(missing))
        print("All recorded source sizes and SHA256 hashes match.")
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, entries))
    restore_metadata(selected)
    print("Sources and source metadata restored. Prepared JSONL files still need rebuilding.")


if __name__ == "__main__":
    main()
