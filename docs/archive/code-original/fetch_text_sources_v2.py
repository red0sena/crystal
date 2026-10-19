"""Download the additional v2 data files without executing dataset loading scripts."""
from __future__ import annotations

import json
import urllib.parse

from fetch_text_sources import ROOT, download, get, sha, write_json

OUT = ROOT / "data/text-decisions-v2/raw"
SPECS = {
    "clinc/clinc_oos": [f"plus/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    "SetFit/20_newsgroups": ["train.jsonl", "test.jsonl"],
    "google-research-datasets/go_emotions": [f"simplified/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")],
    "nyu-mll/glue": [f"stsb/{s}-00000-of-00001.parquet" for s in ("train", "validation")],
    "Yelp/yelp_review_full": [f"yelp_review_full/{s}-00000-of-00001.parquet" for s in ("train", "test")],
    "google/boolq": [f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation")],
    "emozilla/quality": ["data/train-00000-of-00001-0bffe0bf790d89eb.parquet",
                         "data/validation-00000-of-00001-77baeb9538209706.parquet"],
}
# MASSIVE is only published as an auto-converted parquet branch.
PARQUET_BRANCH = {"mteb/amazon_massive_intent": ("en", ("train", "validation", "test"))}


def fetch_repo(repo, paths):
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
    assert set(paths) <= siblings, set(paths) - siblings
    files = [download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}", directory / name)
             for name in (["README.md"] if "README.md" in siblings else []) + list(paths)]
    data = {"repo_id": repo, "revision": revision, "license": info.get("cardData", {}).get("license"), "files": files}
    write_json(cached, data)
    print(f"Downloaded {repo}: {sum(f['bytes'] for f in files):,} bytes", flush=True)
    return data


def fetch_parquet_branch(repo, config, splits):
    directory = OUT / repo.replace("/", "__")
    cached = directory / "manifest.json"
    if cached.exists():
        return json.loads(cached.read_text())
    info = json.loads(get("https://huggingface.co/api/datasets/" + repo))
    listing = json.loads(get(f"https://huggingface.co/api/datasets/{urllib.parse.quote(repo, safe='/')}/parquet"))
    files = []
    for split in splits:
        urls = listing[config][split]
        assert len(urls) == 1, urls
        files.append(download(urls[0], directory / config / f"{split}.parquet"))
    data = {"repo_id": repo, "revision": info["sha"], "license": info.get("cardData", {}).get("license"),
            "source": "refs/convert/parquet via the Hub parquet API", "config": config, "files": files}
    write_json(cached, data)
    print(f"Downloaded {repo}/{config}: {sum(f['bytes'] for f in files):,} bytes", flush=True)
    return data


def main():
    manifests = [fetch_repo(repo, paths) for repo, paths in SPECS.items()]
    manifests += [fetch_parquet_branch(repo, config, splits) for repo, (config, splits) in PARQUET_BRANCH.items()]
    write_json(OUT / "sources.json", manifests)


if __name__ == "__main__":
    main()
