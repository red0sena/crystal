"""Frozen g7 + real-image inputs for a joint, from-original-weights pilot.

Uses the existing converters, gold renderer and contamination filter. No evaluation
labels select training rows. Full-board perceptual deduplication is a remaining
release gate: this pilot checks archived exact board hashes and the restored eight
diagnostic benchmarks perceptually; it is not a submission-ready training clearance.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from transformers import AutoTokenizer

from qwen_decision.v3 import champion_audit as audit, decontam, format as fmt, mix, train
from qwen_decision.v3 import vision_mix as vm, vision_convert_cauldron as cc

ROOT = Path("runs/champion-v1/joint-g7-v1")
MIX = ROOT / "mix"
BASE = audit.BASE
SEED = 20261104
N_TEXT = N_IMAGE = 16384
ADDED_ORIGINS = {"aegis-prompts": "human", "argsme-train": "human", "csqa2-train": "human",
                 "ezstance-train": "human", "minicheck-synth": "model", "nvd-cve": "human",
                 "poem-pairs-gen": "programmatic", "pstance-train": "human", "scruples-train": "human"}
# The owned, pinned g7 source manifest explicitly records the user's 2026-10-19 source-specific permission.
MODEL_ALLOWED = mix.MODEL_ORIGIN_ALLOWED | {"minicheck-synth"}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def eligible_origin(row):
    origin = ADDED_ORIGINS.get(row["source"])
    if origin is None: origin = mix.label_origin(row)
    return row["label_origin"] == origin and (origin in ("human", "programmatic") or row["source"] in MODEL_ALLOWED)


def keys(row):
    return {row["item"], *row.get("documents", [])}


def take_balanced(by, count):
    result = []; index = 0
    while len(result) < count:
        added = [by[s][index] for s in sorted(by) if len(by[s]) > index]
        if not added: raise ValueError(f"only {len(result)} eligible rows for quota {count}")
        result.extend(added[:count - len(result)]); index += 1
    return result


def text_data():
    raw = ROOT / "restored/data/mix"
    manifest = audit.read(raw / "v3-gold7.manifest.json")
    expected = audit.read("runs/champion-v1/g7-review/review.json")["hub_files_match_g7_training_manifest"]
    for file, value in expected.items(): assert audit.sha(ROOT / "restored" / file) == value["sha256"]
    dev = list(audit.rows(raw / "v3-gold7.dev.jsonl"))
    dev_keys = set().union(*(keys(r) for r in dev))
    dev_inputs = {audit.input_key(r) for r in dev}
    fresh = {audit.state_key(r["state"]) for r in audit.rows("data/fresh/fresh-v1.jsonl")}
    heaps = defaultdict(list); counts = Counter(); dropped = Counter(); origins = Counter()
    for i, row in enumerate(audit.rows(raw / "v3-gold7.jsonl"), 1):
        assert eligible_origin(row), (row["source"], row["label_origin"])
        assert row["source"] not in mix.CORPUS_EXCLUDED
        assert len(fmt.options(row["question"])[0]) == len(row["target"])
        assert abs(sum(row["target"]) - 1.) < 1e-5 and min(row["target"]) >= 0
        counts[row["source"]] += 1; origins[row["label_origin"]] += 1
        if keys(row) & dev_keys: dropped["dev_item_document"] += 1; continue
        if audit.input_key(row) in dev_inputs: dropped["dev_full_input"] += 1; continue
        if audit.has_context(row) and audit.state_key(row["state"]) in fresh: dropped["fresh_state"] += 1; continue
        # Keep bounded, deterministic candidates per source; ordinal resolves duplicate legacy ids.
        rank = int(digest("g7-joint-pool", row), 16)
        entry = (-rank, i, row); heap = heaps[row["source"]]
        if len(heap) < 512: heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)
        if i % 200000 == 0: print("G7_ROWS_CHECKED", i, flush=True)
    assert sum(counts.values()) == 999961 and dict(counts) == manifest["counts"]
    filter_ = decontam.Filter(ROOT / "restored/data/decontam")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True); codes, _ = fmt.answer_codes(tokenizer)
    by = defaultdict(list); seen = set()
    for source in sorted(heaps):
        for _, _, row in sorted(heaps[source], key=lambda e: (-e[0], e[1])):
            identity = digest({"state": row["state"], "question": row["question"], "target": audit.target_signature(row)})
            if identity in seen: dropped["identical_training_variant"] += 1; continue
            reason = filter_.flagged(decontam.row_texts(row["state"], {"q": row["question"]}))
            if reason: dropped["strict_public_jeval_filter"] += 1; continue
            text = fmt.render(tokenizer, row["state"], row["question"], codes[:len(row["target"])])
            if len(tokenizer.encode(text, add_special_tokens=False)) > 7680:
                dropped["pilot_training_length_over_7680"] += 1; continue
            seen.add(identity)
            by[source].append({**row, "modality": "text", "original_id": row["id"],
                               "id": row["id"] + ":variant:" + identity[:16]})
    selected = take_balanced(by, N_TEXT)
    dev_by = defaultdict(list)
    for row in dev:
        assert eligible_origin(row)
        dev_by[row["source"]].append(row)
    evaluation = []
    for source in sorted(dev_by):
        chosen = sorted(dev_by[source], key=lambda r: digest("joint-g7-dev", r))[:12]
        evaluation += [{**r, "modality": "text"} for r in chosen]
    train.atomic_json(ROOT / "text-audit.json", {"checked_rows": sum(counts.values()), "raw_sources": dict(counts),
                      "raw_origins": dict(origins), "dropped": dict(dropped), "candidate_sources": {s: len(v) for s, v in by.items()},
                      "train_sources": dict(Counter(r["source"] for r in selected)), "dev_rows": len(evaluation),
                      "model_origin_allowed": sorted(MODEL_ALLOWED),
                      "scope": "all raw origin/shape/counts; selected candidates rechecked against pinned public+JEVal filter; no new labels/teachers; pilot training-length exclusion, no truncation"})
    return train.gold_targets(selected), train.gold_targets(evaluation)


def image_hash(path):
    raw = Path(path).read_bytes()
    return path, hashlib.sha256(raw).hexdigest(), vm.dhash(raw)


def image_groups(items, hashes):
    parent = list(range(len(items))); first = {}
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, row in enumerate(items):
        for path in row["images"]:
            sha, perceptual = hashes[path]
            for key in ("sha:" + sha, "dhash:" + str(perceptual)):
                if key in first: parent[find(i)] = find(first[key])
                else: first[key] = i
    members = defaultdict(list)
    for i, row in enumerate(items): members[find(i)].append(row["id"])
    group = {k: digest("vision-group", min(v)) for k, v in members.items()}
    return [{**r, "group": group[find(i)]} for i, r in enumerate(items)]


def subset_options(row, seed):
    """Only remove verified wrong alternatives during training; never invent a new negative."""
    if row["ordered"] or len(row["options"]) < 4: return row
    rng = random.Random(seed); n = rng.randint(3, len(row["options"]))
    chosen = [row["gold"], *rng.sample([i for i in range(len(row["options"])) if i != row["gold"]], n - 1)]
    rng.shuffle(chosen)
    return {**row, "options": [row["options"][i] for i in chosen], "gold": chosen.index(row["gold"])}


def text_example(row):
    """Diagnostic inputs use the same complete schema expected by train.Renderer."""
    k = len(fmt.options(row["question"])[0])
    return {"modality": "text", "id": row["id"], "source": row["source"], "state": row["state"],
            "question": row["question"], "target": [float(i == row["gold"]) for i in range(k)],
            "ordered": False, "kind": row["question"]["type"]}


def vision_example(row):
    # Kit criteria already use A/B/... keys; don't add those keys twice to option descriptions.
    options = [key if value is None else fmt.describe(value) for key, value in row["question"]["criteria"].items()]
    return {"modality": "image", "id": row["id"], "source": row["source"],
            "images": [str(Path(p).resolve()) for p in row["images"]], "options": options,
            "question": fmt.describe(row["question"]["instructions"]), "gold": row["gold"], "ordered": True,
            "label": "human", "group": row["group"]}


def image_data():
    vm.MIX = MIX; vm.IMG = MIX / "img"; cc.CAULDRON = ROOT / "raw/cauldron"
    plan = audit.read(ROOT / "vision-download-plan.json")
    candidates = []
    for subset in plan["subsets"]:
        saver = vm.Saver("cauldron/" + subset)
        raw = [r for r in cc.convert(cc.SPECS[subset][1], saver, 4096, random.Random(f"joint:{SEED}:{subset}")) if r]
        candidates += raw
        print("VISION_POOL", subset, len(raw), flush=True)
    paths = sorted({str(MIX / p) for r in candidates for p in r["images"]})
    with ProcessPoolExecutor(max_workers=6) as pool: full = list(pool.map(image_hash, paths, chunksize=64))
    hashes = {str(Path(p).relative_to(MIX)): (s, d) for p, s, d in full}
    # Preserve exact archived board exclusions even though the raw board images were cleaned up.
    board_sha = {s for r in audit.rows("runs/vision/wise80-decide/results.jsonl") for s in r["image_sha256"]}
    examples = list(audit.rows("runs/inference-policy-v1/examples.jsonl"))
    evaluation_paths = sorted({p for r in examples if r["modality"] == "vision" for p in r["images"]})
    with ProcessPoolExecutor(max_workers=6) as pool: eval_hashes = list(pool.map(image_hash, evaluation_paths, chunksize=64))
    eval_sha = board_sha | {s for _, s, _ in eval_hashes}; eval_dhash = [d for _, _, d in eval_hashes]
    path_order = list(hashes)
    close = {path_order[i] for i in vm.near([hashes[p][1] for p in path_order], eval_dhash, 4)}
    forbidden = {p for p, (s, _) in hashes.items() if s in eval_sha} | close
    clean = [r for r in candidates if not forbidden.intersection(r["images"])]
    grouped = image_groups(clean, hashes)
    reserved = [r for r in grouped if int(r["group"][:8], 16) % 5 == 0]
    training = [r for r in grouped if int(r["group"][:8], 16) % 5 != 0]
    reserved_paths = {p for r in reserved for p in r["images"]}
    train_paths = sorted({p for r in training for p in r["images"]})
    near_dev = {train_paths[i] for i in vm.near([hashes[p][1] for p in train_paths],
                                             [hashes[p][1] for p in reserved_paths], 4)}
    training = [r for r in training if not near_dev.intersection(r["images"])]
    by = defaultdict(list)
    for row in training: by[row["source"]].append(row)
    for s in by: by[s].sort(key=lambda r: digest("vision-train", r["id"]))
    selected = [subset_options(r, f"joint-options:{r['id']}") | {"modality": "image"}
                for r in take_balanced(by, N_IMAGE)]
    dev_by = defaultdict(list)
    for row in reserved: dev_by[row["source"]].append(row)
    evaluation = [r | {"modality": "image"} for s in sorted(dev_by)
                  for r in sorted(dev_by[s], key=lambda r: digest("vision-dev", r["id"]))[:64]]
    assert not {p for r in selected for p in r["images"]} & {p for r in reserved for p in r["images"]}
    pairs = {}; pair_rows = []
    for row in reserved:
        if row["source"] == "cauldron/nlvr2" and row["options"][row["gold"]].lower() in ("yes", "true"):
            pairs.setdefault(row["group"], row)
    for group, row in sorted(pairs.items(), key=lambda kv: digest("alignment-pair", kv[0]))[:256]:
        pair_rows.append({"id": row["id"], "group": group, "images": row["images"],
                          "description": row["question"], "split": "evaluation" if int(group[-8:], 16) % 4 == 0 else "fit",
                          "provenance": "NLVR2 human-written true statement, one per reserved image component; no teacher caption"})
    write(MIX / "alignment-pairs.jsonl", pair_rows)
    train.atomic_json(ROOT / "vision-audit.json", {"candidates": len(candidates), "after_eval_filter": len(clean),
                      "reserved_rows": len(reserved), "train_after_reserved_near_filter": len(training),
                      "train_sources": dict(Counter(r["source"] for r in selected)),
                      "train_option_counts": dict(Counter(len(r["options"]) for r in selected)),
                      "archived_board_exact_hashes": len(board_sha), "perceptual_eval_images": len(eval_hashes),
                      "near_distance": 4, "reserved_image_near_filter": True, "alignment_pairs": len(pair_rows),
                      "full_board_perceptual_audit_complete": False,
                      "scope": "pilot only; all converter labels human/program; upstream train subsets; no test rows for training; evaluation hashes use archived exact board and restored 8-benchmark pixel hashes; full-board near-duplicate audit required before final training/submission"})
    train.atomic_json(ROOT / "training-image-hashes.json", hashes)
    return selected, evaluation


def run():
    assert (ROOT / "restoration.json").exists()
    assert not (MIX / "train.jsonl").exists(), "Preserve prepared inputs"
    started = time.monotonic(); MIX.mkdir(parents=True, exist_ok=True)
    train.atomic_json(ROOT / "status.json", {"state": "preparing_g7_and_images", "training_started": False})
    texts, text_dev = text_data(); images, image_dev = image_data()
    combined = texts + images; random.Random(SEED).shuffle(combined)
    write(MIX / "train.jsonl", combined); write(MIX / "text-dev.jsonl", text_dev); write(MIX / "image-dev.jsonl", image_dev)
    # Existing diagnostics are retained with their known, previously examined scope.
    old = list(audit.rows("runs/inference-policy-v1/examples.jsonl"))
    text_ood = []
    for r in old:
        if r["modality"] != "text": continue
        text_ood.append(text_example(r))
    write(MIX / "text-diagnostic.jsonl", text_ood)
    vision_ood = []
    for r in old:
        if r["modality"] != "vision": continue
        vision_ood.append(vision_example(r))
    write(MIX / "vision-diagnostic.jsonl", vision_ood)
    inputs = {str(p): audit.sha(p) for p in MIX.glob("*.jsonl")}
    train.atomic_json(ROOT / "prepared.json", {"complete": True, "rows": len(combined), "text_rows": len(texts),
                      "image_rows": len(images), "files": inputs, "seconds": time.monotonic() - started,
                      "base_sha256": audit.BASE_SHA, "source_sha256": audit.sha(__file__),
                      "scope": "single-seed joint pilot; original base weights; previous diagnostics are not blind tests; full-board image clearance is incomplete"})
    train.atomic_json(ROOT / "status.json", {"state": "prepared", "training_started": False, "rows": len(combined)})


if __name__ == "__main__": run()
