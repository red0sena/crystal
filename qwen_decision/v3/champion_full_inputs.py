"""Indexed, untruncated full training inputs and CPU rendering preflight."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from qwen_decision.v3 import champion_audit as audit, champion_joint_data as old
from qwen_decision.v3 import format as fmt, train, vision_train as vt
from qwen_decision.v3.champion_enriched_data import bounded_map

ROOT = Path("runs/champion-v1/full-g7-lora-v3")
DATA = Path("runs/champion-v1/full-g7-v3")
MIX = DATA / "mix"
BASE = audit.BASE
SEED = 20261106
_TOKENIZER = _CODES = _ENCODER = None


def configure():
    vt.MIX, vt.PROCESSOR = MIX, BASE
    from types import SimpleNamespace
    vt.TEXT_ARGS = SimpleNamespace(**(vars(vt.TEXT_ARGS) | {"pad_probability": 0.}))


class IndexedRows:
    """Picklable file/index references; every spawned worker opens its own mmap and file."""
    def __init__(self, files, indices):
        self.files, self.indices = [str(p) for p in files], [str(p) for p in indices]
        self.counts = [int(np.load(p, mmap_mode="r").shape[0]) for p in self.indices]
        self.ends = np.cumsum(self.counts).tolist()
        self.handles, self.maps = {}, {}

    def __len__(self):
        return sum(self.counts)

    def __getstate__(self):
        return self.__dict__ | {"handles": {}, "maps": {}}

    def __getitem__(self, i):
        import bisect
        if not 0 <= i < len(self):
            raise IndexError(i)
        part = bisect.bisect_right(self.ends, i)
        at = i - (self.ends[part - 1] if part else 0)
        if part not in self.maps:
            self.maps[part] = np.load(self.indices[part], mmap_mode="r")
            self.handles[part] = open(self.files[part], "rb")
        offset, size, _ = self.maps[part][at]
        file = self.handles[part]
        file.seek(int(offset))
        row = json.loads(file.read(int(size)))
        return train.gold_targets([row])[0] if row["modality"] == "text" else row


def validate_row(r):
    assert r["id"] and r["source"] and isinstance(r["ordered"], bool)
    if r["modality"] == "text":
        if r["source"] == "phishing-url-train":
            assert r["label_origin"] == "programmatic" and r["target_kind"] == "single"
            assert r["upstream"] == "pirocheto/phishing-url"
            assert r["upstream_revision"] == "1375d450d4540205ac11781b807249c9509739cd"
            assert r["upstream_split"] == "train"
            assert set(r["state"]) == {"url", "hostname"}
            assert r["documents"] == ["phishing-url-train:host:" + r["state"]["hostname"]]
            assert r["item"] == r["original_id"]
        else:
            assert old.eligible_origin(r), r["id"]
        k = len(fmt.options(r["question"])[0])
        assert isinstance(r["kind"], str) and 2 <= k <= fmt.MAX_OPTIONS
        assert len(r["target"]) == k and all(math.isfinite(v) and v >= 0 for v in r["target"])
        assert abs(sum(r["target"]) - 1.) < 1e-5
    else:
        assert r["modality"] == "image" and r["label"] in ("human", "program")
        assert 0 <= r["gold"] < len(r["options"]) and r["images"]
        assert all((MIX / p).is_file() for p in r["images"])


def init_worker():
    import torch
    torch.set_num_threads(1)
    configure()
    global _TOKENIZER, _CODES
    _TOKENIZER = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    _CODES, _ = fmt.answer_codes(_TOKENIZER)


def inspect_batch(batch):
    global _ENCODER
    rows = [json.loads(raw) for _, raw in batch]
    for row in rows:
        validate_row(row)
    if rows[0]["modality"] == "text":
        prompts = [fmt.render(_TOKENIZER, r["state"], r["question"], _CODES[:len(r["target"])]) for r in rows]
        encoded = _TOKENIZER(prompts, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
        lengths = [len(ids) for ids in encoded]
    else:
        if _ENCODER is None:
            _ENCODER = vt.Encoder(max_pixels=1638400)
        lengths = [len(_ENCODER.row(r, None)["input_ids"]) for r in rows]
    return [(offset, len(raw), n) for (offset, raw), n in zip(batch, lengths)]


def raw_batches(path, size):
    with Path(path).open("rb") as handle:
        while True:
            batch = []
            for _ in range(size):
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                batch.append((offset, raw))
            if not batch:
                return
            yield batch


def build_index(path, count, name, workers=4):
    dest = ROOT / "indices" / (name + ".npy")
    proof = dest.with_suffix(".json")
    signature = {"path": str(path), "sha256": audit.sha(path), "rows": count,
                 "source_sha256": audit.sha(__file__),
                 "renderer_sha256": audit.sha(fmt.__file__), "encoder_sha256": audit.sha(vt.__file__)}
    if proof.exists():
        cached = audit.read(proof)
        assert cached["signature"] == signature
        assert audit.sha(dest) == cached["index_sha256"]
        return cached
    dest.parent.mkdir(exist_ok=True, parents=True)
    partial = dest.with_suffix(".partial.npy")
    index = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint64, shape=(count, 3))
    at, total, longest = 0, 0, []
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        for batch in bounded_map(pool, inspect_batch, raw_batches(path, 128 if "text" in name else 4), limit=12):
            assert at + len(batch) <= count
            index[at:at + len(batch)] = batch
            total += sum(r[2] for r in batch)
            longest += [(r[2], at + i) for i, r in enumerate(batch)]
            longest = sorted(longest, reverse=True)[:16]
            at += len(batch)
            if at % 16384 < len(batch):
                train.atomic_json(ROOT / "input-status.json", {"stage": "rendering", "file": name, "rows": at, "total": count})
                print("INPUT_ROWS", name, at, count, flush=True)
    assert at == count
    index.flush()
    del index
    partial.replace(dest)
    result = {"signature": signature, "index": str(dest), "index_sha256": audit.sha(dest),
              "rows": count, "tokens": total, "longest": longest, "max_tokens": longest[0][0],
              "all_rows_rendered": True, "truncated_rows": 0}
    train.atomic_json(proof, result)
    print("INDEX_READY", name, count, result["max_tokens"], flush=True)
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = audit.read(DATA / "input-manifest.json")
    assert manifest["input_preparation_complete"]
    for p in manifest["text_training_parts"]:
        assert audit.sha(p["path"]) == p["sha256"]
    assert audit.sha(manifest["vision_train"]["path"]) == manifest["vision_train"]["sha256"]
    for p, sha in manifest["evaluation_parts"].items():
        assert audit.sha(p) == sha
    for p, sha in manifest["data_proofs"].items():
        assert audit.sha(p) == sha
    image_proof = DATA / "enriched-vision-image-hashes.json"
    for p, sha in audit.read(image_proof).items():
        assert audit.sha(p) == sha
    reports = {}
    text_parts = []
    for i, part in enumerate(manifest["text_training_parts"]):
        key = "text-train-" + str(i)
        reports[key] = build_index(Path(part["path"]), part["rows"], key)
        text_parts.append({"path": part["path"], "index": reports[key]["index"]})
    part = manifest["vision_train"]
    reports["image-train"] = build_index(Path(part["path"]), part["rows"], "image-train")
    # Actual held-out input files are preserved; render all of them ahead of final evaluation.
    evals = {}
    for name in ("text-dev", "phishing-url-dev", "image-dev"):
        p = MIX / (name + ".jsonl")
        count = sum(1 for _ in p.open())
        reports[name] = build_index(p, count, "text-" + name if name == "phishing-url-dev" else name)
        evals[name] = str(p)
    prepared = {"complete": True, "text_parts": text_parts,
                "image_part": {"path": part["path"], "index": reports["image-train"]["index"]},
                "text_rows": manifest["text_rows"], "image_rows": part["rows"], "reports": reports,
                "evaluation": evals, "truncation": False,
                "input_manifest_sha256": audit.sha(DATA / "input-manifest.json"),
                "source_sha256": audit.sha(__file__)}
    train.atomic_json(ROOT / "prepared.json", prepared)
    train.atomic_json(ROOT / "input-status.json", {"stage": "ready", "training_started": False})


if __name__ == "__main__":
    import traceback
    try:
        prepare()
    except Exception:
        train.atomic_json(ROOT / "input-status.json", {"stage": "failed", "traceback": traceback.format_exc()})
        raise
