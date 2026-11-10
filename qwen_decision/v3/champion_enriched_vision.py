"""Use all eligible cached vision candidates plus the full prepared Mind2Web train.

Existing Cauldron pools were bounded upstream; this removes the later 32,768-row
selection quota, not those upstream bounds. Preserve every previously observed dev.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time
import traceback

from qwen_decision.v3 import champion_audit as audit, champion_joint_data as old
from qwen_decision.v3 import champion_full_mind2web as m2w, train, vision_mix as vm

ROOT, MIX = m2w.ROOT, m2w.MIX
PREVIOUS = Path("runs/champion-v1/joint-g7-v2")


def status(stage, **extra):
    train.atomic_json(ROOT / "enriched-vision-status.json", {"stage": stage, "training_started": False, **extra})


def hashed(path):
    raw = Path(path).read_bytes()
    import hashlib
    return path, hashlib.sha256(raw).hexdigest(), vm.dhash(raw)


def prepare():
    completion = ROOT / "enriched-vision-prepared.json"
    if completion.exists():
        done = audit.read(completion)
        assert done["source_sha256"] == audit.sha(__file__)
        for file, sha in done["files"].items():
            assert audit.sha(file) == sha
        return
    started = time.monotonic()
    deadline = started + 7200
    while not (ROOT / "mind2web-prepared.json").exists():
        s = audit.read(ROOT / "mind2web-status.json")
        if s["stage"] == "failed":
            raise RuntimeError("Mind2Web preparation failed; inspect mind2web-status.json")
        if time.monotonic() > deadline:
            raise RuntimeError("Mind2Web input wait exceeded two hours")
        status("waiting_for_mind2web")
        time.sleep(20)
    proof = audit.read(ROOT / "mind2web-prepared.json")
    for file, sha in proof["files"].items():
        assert audit.sha(file) == sha
    for file, sha in audit.read(ROOT / "mind2web-image-hashes.json").items():
        assert audit.sha(file) == sha
    status("hashing_and_deduplicating")
    pools = sorted((PREVIOUS / "pools").glob("*.jsonl"))
    assert len(pools) == 14
    candidates = [r for p in pools for r in audit.rows(p)]
    for r in candidates:
        assert r["label"] in ("human", "program")
        assert 0 <= r["gold"] < len(r["options"]) and len(set(r["options"])) == len(r["options"])
        for image in r["images"]:
            dest = MIX / image
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(PREVIOUS / "mix" / image, dest)
    paths = sorted({str(MIX / p) for r in candidates for p in r["images"]})
    with ProcessPoolExecutor(4) as pool:
        hashes = {str(Path(p).relative_to(MIX)): (s, d)
                  for p, s, d in pool.map(hashed, paths, chunksize=32)}
        evalhashes = list(pool.map(hashed, m2w.prior_evaluation_images(), chunksize=32))
    evalsha = {s for _, s, _ in evalhashes}
    evaldh = [d for _, _, d in evalhashes]
    order = list(hashes)
    bad = {order[i] for i in vm.near([hashes[p][1] for p in order], evaldh, 4)}
    bad |= {p for p, (s, _) in hashes.items() if s in evalsha}
    grouped = old.image_groups(candidates, hashes)
    forbidden_groups = {r["group"] for r in grouped if bad.intersection(r["images"])}
    eligible = [r for r in grouped if r["group"] not in forbidden_groups]
    dev = [r | {"modality": "image"} for r in eligible if int(r["group"][:8], 16) % 5 == 0]
    training = [r | {"modality": "image"} for r in eligible if int(r["group"][:8], 16) % 5 != 0]
    mind_train = list(audit.rows(MIX / "mind2web-train.jsonl"))
    mind_dev = list(audit.rows(MIX / "mind2web-dev.jsonl"))
    devdh = [hashes[p][1] for r in dev for p in r["images"]]
    devdh += [r[key] for r in mind_dev for key in ("raw_dhash", "marked_dhash")]
    close = vm.near([hashes[p][1] for r in training for p in r["images"]], devdh, 4)
    index, near_groups = 0, set()
    for r in training:
        if any(i in close for i in range(index, index + len(r["images"]))):
            near_groups.add(r["group"])
        index += len(r["images"])
    training = [r for r in training if r["group"] not in near_groups]
    # Mind2Web was already separated against previous dev; additionally compare
    # against all newly reserved cached-pool images, including their unselected rows.
    close = vm.near([r["marked_dhash"] for r in mind_train], devdh, 4)
    close |= vm.near([r["raw_dhash"] for r in mind_train], devdh, 4)
    mind_bad = {r["group"] for i, r in enumerate(mind_train) if i in close}
    mind_train = [r for r in mind_train if r["group"] not in mind_bad]
    training += mind_train
    dev += mind_dev
    assert mind_train and training
    assert not {p for r in training for p in r["images"]} & {p for r in dev for p in r["images"]}
    assert not {r["group"] for r in training} & {r["group"] for r in dev}
    files = []
    for name, rows in [("image-train", training), ("image-dev", dev)]:
        dest = MIX / (name + ".jsonl")
        tmp = dest.with_suffix(".partial")
        if tmp.exists():
            tmp.unlink()
        old.write(tmp, rows)
        tmp.replace(dest)
        files.append(dest)
    selected = {p for r in training + dev for p in r["images"]}
    image_proof = ROOT / "enriched-vision-image-hashes.json"
    train.atomic_json(image_proof, {str(MIX / p): audit.sha(MIX / p) for p in sorted(selected)})
    files.append(image_proof)
    report = {"complete": True, "training_started": False, "train_rows": len(training), "dev_rows": len(dev),
              "train_sources": dict(Counter(r["source"] for r in training)),
              "dev_sources": dict(Counter(r["source"] for r in dev)), "sample_cap_after_cached_pool": None,
              "cached_pool_rows": len(candidates), "cached_pool_hashes": {str(p): audit.sha(p) for p in pools},
              "excluded_visual_groups": len(forbidden_groups), "near_dev_excluded_groups": len(near_groups),
              "additional_mind2web_excluded_groups": len(mind_bad), "source_sha256": audit.sha(__file__),
              "files": {str(p): audit.sha(p) for p in files}, "seconds": time.monotonic() - started,
              "scope": "All eligible previously cached 14 source pools, plus Mind2Web train. Cached pools have upstream conversion limits; this is not every possible image training row. No new teacher labels. No GPU training launched."}
    train.atomic_json(completion, report)
    status("prepared", train_rows=len(training), mind2web_train_rows=len(mind_train))
    print("VISION_PREPARED", len(training), report["train_sources"], flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except Exception:
        status("failed", traceback=traceback.format_exc())
        raise
