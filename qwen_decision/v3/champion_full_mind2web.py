"""Prepare every eligible Mind2Web train action for the next full-g7 run.

This is input preparation, not a training launcher. It never changes the frozen
v1/v2 experiments, uses no evaluation labels, and imposes no sample quota.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import pyarrow.parquet as pq

from qwen_decision.v3 import champion_audit as audit, champion_joint_data as old
from qwen_decision.v3 import train, vision_mix as vm

ROOT = Path("runs/champion-v1/full-g7-v3")
MIX = ROOT / "mix"
RAW = ROOT / "raw/mind2web"
PUBLIC = Path("runs/champion-v1/public-eval-v1/vision-suite")
FONT = Path("runs/champion-v1/public-eval-v1/font/Arialbd.TTF")
REPO = "osunlp/Multimodal-Mind2Web"
REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"
SEED = 20261105
COLS = ["action_uid", "annotation_id", "operation", "pos_candidates", "neg_candidates",
        "domain", "confirmed_task", "screenshot", "action_reprs", "target_action_index"]


def state(stage, **extra):
    train.atomic_json(ROOT / "mind2web-status.json", {"stage": stage, "training_started": False, **extra})


def restore():
    from huggingface_hub import HfApi, hf_hub_download

    state("restoring")
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    files = [{"file": f.rfilename, "bytes": f.size, "sha256": f.lfs.sha256}
             for f in info.siblings
             if f.rfilename.startswith("data/train-") and f.rfilename.endswith(".parquet")]
    assert len(files) == 27 and all(f["sha256"] for f in files)
    train.atomic_json(ROOT / "mind2web-download-plan.json",
                      {"repo": REPO, "revision": REVISION, "split": "train", "files": files})

    def one(entry):
        path = Path(hf_hub_download(REPO, entry["file"], repo_type="dataset",
                                   revision=REVISION, local_dir=RAW))
        assert path.stat().st_size == entry["bytes"] and audit.sha(path) == entry["sha256"]
        print("RESTORED", path.name, flush=True)
        return {**entry, "local": str(path)}

    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(one, files))
    train.atomic_json(ROOT / "mind2web-restoration.json",
                      {"complete": True, "repo": REPO, "revision": REVISION, "files": done})
    return done


def partition(rows, forbidden_actions=(), forbidden_tasks=(), forbidden_visuals=()):
    """Hold out whole tasks and identical raw screens together, not individual actions."""
    forbidden_actions, forbidden_tasks, forbidden_visuals = map(
        set, (forbidden_actions, forbidden_tasks, forbidden_visuals))
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, row in enumerate(rows):
        for key in ("task:" + row["annotation_id"], "sha:" + row["raw_sha256"],
                    "dhash:" + str(row["raw_dhash"])):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    training, dev, dropped = [], [], Counter()
    for members in groups.values():
        if any(r["action_uid"] in forbidden_actions or r["annotation_id"] in forbidden_tasks for r in members):
            dropped["evaluation_action_or_task_group"] += len(members)
            continue
        if any(r["id"] in forbidden_visuals for r in members):
            dropped["evaluation_or_prior_dev_visual_group"] += len(members)
            continue
        group = old.digest("full-mind2web-task-group", min(r["annotation_id"] for r in members))
        destination = dev if int(group[:8], 16) % 5 == 0 else training
        destination.extend({**r, "group": group, "modality": "image"} for r in members)
    return training, dev, dict(dropped)


def _convert_shard(path):
    from qwen_decision.v3 import vision_convert_screen as screen

    vm.MIX, vm.IMG = MIX, MIX / "img"
    screen.FONT = str(FONT)
    saver = vm.Saver("mind2web")
    rows, counts = [], Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4, columns=COLS):
        for row in batch.to_pylist():
            counts["raw_actions"] += 1
            result = screen.m2w_item(saver, row, random.Random(f"full:{SEED}:{row['action_uid']}"))
            if result is None:
                counts["invalid_or_insufficient_visible_candidates"] += 1
                continue
            image_path = MIX / result["images"][0]
            raw = row["screenshot"]["bytes"]
            rows.append({**result, "action_uid": row["action_uid"], "annotation_id": row["annotation_id"],
                         "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_dhash": vm.dhash(raw),
                         "marked_sha256": audit.sha(image_path), "marked_dhash": vm.dhash(image_path.read_bytes()),
                         "upstream_split": "train", "upstream_revision": REVISION})
    return rows, dict(counts)


def _hash_image(path):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw).hexdigest(), vm.dhash(raw)


def prior_evaluation_images():
    paths = {str(p) for p in PUBLIC.glob("full/*/img/*") if p.is_file()}
    # Requests and unique image files have different counts (KIE reuses images).
    # Verify the pinned source manifests instead of assuming one file per request.
    from decision_index.vision.build import images_digest
    verified = audit.read("runs/champion-v1/public-eval-v1/vision-verification.json")["sets"]
    for name, checks in verified.items():
        assert all(checks.values()), name
        items_path = PUBLIC / "full" / name / "items.jsonl"
        source = audit.read(items_path.with_name("source.json"))
        items = list(audit.rows(items_path))
        assert len(items) == source["n"] and audit.sha(items_path) == source["items_sha256"], name
        assert images_digest(PUBLIC, items) == source["images_sha256"], name
        for item in items:
            assert all(str(PUBLIC / p) in paths for p in item["images"]), item["id"]
    for r in audit.rows("runs/inference-policy-v1/examples.jsonl"):
        if r["modality"] == "vision":
            paths.update(str(Path(p).resolve()) for p in r["images"])
    for root in [Path("runs/champion-v1/joint-g7-v1"), Path("runs/champion-v1/joint-g7-v2")]:
        for r in audit.rows(root / "mix/image-dev.jsonl"):
            paths.update(str(root / "mix" / p) for p in r["images"])
    return sorted(paths)


def prepare():
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    completion = ROOT / "mind2web-prepared.json"
    if completion.exists():
        done = audit.read(completion)
        assert done["source_sha256"] == audit.sha(__file__)
        for path, sha in done["files"].items():
            assert audit.sha(path) == sha
        print("ALREADY_PREPARED", done["train_rows"], done["dev_rows"], flush=True)
        return
    restored = restore()
    assert FONT.is_file()
    state("converting", raw_shards=len(restored))
    candidates, counts = [], Counter()
    with ProcessPoolExecutor(4) as pool:
        for rows, report in pool.map(_convert_shard, [r["local"] for r in restored]):
            candidates.extend(rows)
            counts.update(report)
            state("converting", **dict(counts), candidates=len(candidates))
    assert counts["raw_actions"] == 7775
    assert len({r["action_uid"] for r in candidates}) == len(candidates)
    state("deduplicating", candidates=len(candidates))
    official = list(audit.rows(PUBLIC / "full/full_mind2web/items.jsonl"))
    actions = {r["id"].split(":", 1)[1] for r in official}
    tasks = {r["annotation_id"] for r in official}
    evalpaths = prior_evaluation_images()
    with ProcessPoolExecutor(4) as pool:
        hashed = list(pool.map(_hash_image, evalpaths, chunksize=32))
    evalsha, evaldh = {s for s, _ in hashed}, [d for _, d in hashed]
    # Both raw screens and marked crops are checked; task IDs also prevent crop-dependent leakage.
    raw_close = vm.near([r["raw_dhash"] for r in candidates], evaldh, 4)
    marked_close = vm.near([r["marked_dhash"] for r in candidates], evaldh, 4)
    bad = {r["id"] for i, r in enumerate(candidates)
           if i in raw_close | marked_close or r["raw_sha256"] in evalsha or r["marked_sha256"] in evalsha}
    training, dev, dropped = partition(candidates, actions, tasks, bad)
    # A similar screenshot in another task must not straddle train and dev either.
    devdh = [r[key] for r in dev for key in ("raw_dhash", "marked_dhash")]
    near_dev = vm.near([r["raw_dhash"] for r in training], devdh, 4)
    near_dev |= vm.near([r["marked_dhash"] for r in training], devdh, 4)
    bad_groups = {r["group"] for i, r in enumerate(training) if i in near_dev}
    kept = [r for r in training if r["group"] not in bad_groups]
    dropped["near_new_dev_group"] = len(training) - len(kept)
    training = kept
    assert training and dev
    assert not {r["annotation_id"] for r in training} & {r["annotation_id"] for r in dev}
    assert not {r["group"] for r in training} & {r["group"] for r in dev}
    # Atomic publication allows retry after an interrupted preparation.
    paths = []
    for name, rows in [("mind2web-train", training), ("mind2web-dev", dev)]:
        path = MIX / (name + ".jsonl")
        tmp = path.with_suffix(".partial")
        if tmp.exists():
            tmp.unlink()
        old.write(tmp, rows)
        tmp.replace(path)
        paths.append(path)
    hashes = {str(MIX / image): audit.sha(MIX / image)
              for row in training + dev for image in row["images"]}
    train.atomic_json(ROOT / "mind2web-image-hashes.json", hashes)
    paths.append(ROOT / "mind2web-image-hashes.json")
    report = {"complete": True, "repo": REPO, "revision": REVISION, "upstream_split": "train",
              "sample_cap": None, **dict(counts), "converted_rows": len(candidates),
              "train_rows": len(training), "dev_rows": len(dev), "dropped": dropped,
              "train_tasks": len({r["annotation_id"] for r in training}),
              "dev_tasks": len({r["annotation_id"] for r in dev}),
              "operations": dict(Counter(r["op"] for r in training)),
              "gold_positions": dict(Counter(r["gold"] for r in training)),
              "evaluation_and_prior_dev_images": len(evalpaths), "dhash_distance": 4,
              "font": str(FONT), "font_sha256": audit.sha(FONT),
              "source_sha256": audit.sha(__file__),
              "converter_sha256": audit.sha("qwen_decision/v3/vision_convert_screen.py"),
              "files": {str(p): audit.sha(p) for p in paths}, "seconds": time.monotonic() - started,
              "limitations": "Train screenshots may have rendering issues. SHA256 + 64-bit dHash <=4 is an approximate image filter, not proof against all edits/crops. No training launched."}
    train.atomic_json(completion, report)
    state("prepared", train_rows=len(training), dev_rows=len(dev))
    print("PREPARED", json.dumps(report), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        prepare()
    except Exception:
        state("failed", traceback=traceback.format_exc())
        raise
