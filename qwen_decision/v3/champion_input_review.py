"""Quarantine only extreme added email views; retain the full audited g7 core unchanged."""
from pathlib import Path
import json

import numpy as np

from qwen_decision.v3 import champion_full_inputs as data, champion_audit as audit, train

ROOT = data.ROOT


def run():
    done = ROOT/"input-review.json"
    if done.exists():
        proof = audit.read(done)
        for p, sha in proof["files"].items():
            assert audit.sha(p) == sha
        return
    original = data.MIX/"text-train.jsonl"
    old_index = ROOT/"indices/text-train-0.npy"
    old_report = audit.read(old_index.with_suffix(".json"))
    assert audit.sha(original) == old_report["signature"]["sha256"]
    index = np.load(old_index, mmap_mode="r")
    positions = np.flatnonzero(index[:, 2] > 65536)
    reader = data.IndexedRows([original], [old_index])
    withheld, items = [], set()
    for i in positions:
        row = reader[int(i)]
        assert row["source"] == "phishing" and row["training_component"] == "targeted-supplement", row["id"]
        items.add(row["item"])
        withheld.append({"index": int(i), "id": row["original_id"], "item": row["item"],
                         "tokens": int(index[i, 2]), "body_chars": len(row["state"]["body"])})
    assert len(positions) == 9 and len(items) == 3, "Re-review if the actual extreme source groups change"
    # All views of a withheld fact must stay together. Existing stored g7 views
    # of the same email are not altered or discarded by this augmentation review.
    skip = set(int(i) for i in positions)
    dest = ROOT/"inputs/text-train-reviewed.jsonl"
    dest.parent.mkdir(exist_ok=True)
    new_index = ROOT/"indices/text-train-reviewed.npy"
    arr = np.lib.format.open_memmap(new_index.with_suffix(".partial.npy"), mode="w+", dtype=np.uint64,
                                  shape=(len(index)-len(skip), 3))
    outpath = dest.with_suffix(".partial")
    quarantine = ROOT/"inputs/withheld-email-views.jsonl"
    at = 0
    with original.open("rb") as source, outpath.open("wb") as out, quarantine.open("wb") as held:
        for i, raw in enumerate(source):
            if i in skip:
                held.write(raw)
                continue
            arr[at] = (out.tell(), len(raw), index[i, 2])
            out.write(raw)
            at += 1
    assert at == len(index)-9
    arr.flush()
    del arr
    new_index.with_suffix(".partial.npy").replace(new_index)
    outpath.replace(dest)
    values = np.load(new_index, mmap_mode="r")[:, 2]
    top = np.argpartition(values, -16)[-16:]
    longest = sorted([(int(values[i]), int(i)) for i in top], reverse=True)
    report = {**old_report, "signature": {**old_report["signature"], "path": str(dest), "sha256": audit.sha(dest), "rows": at},
              "index": str(new_index), "index_sha256": audit.sha(new_index), "rows": at,
              "tokens": int(values.sum()), "max_tokens": int(values.max()), "longest": longest,
              "review_source_sha256": audit.sha(__file__)}
    train.atomic_json(new_index.with_suffix(".json"), report)
    proof = {"complete": True, "original_file": str(original), "original_sha256": audit.sha(original),
             "removed_rows": 9, "withheld_upstream_items": 3, "withheld": withheld,
             "reason": "Newly expanded raw email payloads above65,536 tokens; massive recipient/thread payloads, including one above the native262,144-token context. Quarantined whole views for review; no right truncation or false-label chunking.",
             "g7_core_unchanged": True, "remaining_rows": at, "maximum_remaining_tokens": int(values.max()),
             "text_part": {"path": str(dest), "index": str(new_index)}, "report": report,
             "files": {str(p): audit.sha(p) for p in (dest, new_index, new_index.with_suffix(".json"), quarantine)},
             "source_sha256": audit.sha(__file__)}
    train.atomic_json(done, proof)
    print("INPUT_REVIEW_COMPLETE", at, report["max_tokens"], flush=True)


if __name__ == "__main__":
    run()
