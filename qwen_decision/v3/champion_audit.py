"""Evidence and input audit for a new <=1B text/vision campaign; never trains a model.

Existing results are evidence, not initialization checkpoints. The restore/audit
commands cover historical g6 only. Latest g7 is documented separately in g7-review;
its added sources and historical label policy need review before a dedicated audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import traceback

from qwen_decision.v3 import format as fmt, mix, train

ROOT = Path("runs/champion-v1")
BASE = Path("models/qwen3.5-0.8b")
BASE_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
DATA_REPO = "Kyunghwan-kim/kkh-decision-data"
DATA_REVISION = "c15b054aafa3013cb421597031aee87ee150ad4b"
FILES = ("data/mix/v3-gold6.jsonl", "data/mix/v3-gold6.dev.jsonl", "data/mix/v3-gold6.manifest.json",
         "data/jeval/data/test.parquet")
EVIDENCE = (
    "qwen_decision/v3/champion_audit.py",
    "wiki/project/insights.md", "wiki/project/lora-placement.md", "wiki/project/loss-screen.md",
    "wiki/project/vision-train-screen.md", "wiki/project/hub-g7-merges.md", "wiki/project/jev-index.md",
    "wiki/papers/gliclass.md", "wiki/papers/vlm-modality-finetuning.md",
    "qwen_decision/v3/mix.py", "qwen_decision/v3/lora_placement_experiment.py",
    "qwen_decision/v3/vision_train.py", "qwen_decision/v3/proxy_eval.py",
    "runs/from-base/lora-placement-v1/report.json", "runs/from-base/loss-v1/report.json",
    "runs/vision-train/screen-report.json", "runs/inference-policy-v1/report.json",
    "runs/vision-question-repeat-v1/report.json",
    "wiki/project/champion-v1-g7-review.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def sha(path):
    return train.sha256_file(Path(path))


def small_text(models):
    return sorted((m for m in models if 0 < ((m.get("meta") or {}).get("served_params") or 0) <= 10**9),
                  key=lambda m: m["scores"]["balanced_skill"], reverse=True)


def evidence(root):
    board = root / "board"
    text, vision, method = (read(board / name) for name in ("index.json", "vision.json", "methodology.json"))
    small = small_text(text["models"])
    vl = sorted((m for m in vision["entrants"] if 0 < (m.get("params") or 0) <= 10**9),
                key=lambda m: m["full"], reverse=True)
    ours = next(m for m in small if m["engine"] == "crystal-0.8b")
    placement = read("runs/from-base/lora-placement-v1/report.json")
    losses = read("runs/from-base/loss-v1/report.json")
    assert placement["complete"] and losses["complete"]
    means = {arm: {name: {metric: sum(v[name][metric] for v in seeds.values()) / len(seeds)
                         for metric in ("correct", "nll")} for name in ("selection", "long")}
             for arm, seeds in placement["scores"].items()}
    panel_ids = {str(b["id"]) for area in method["index"]["areas"] for b in area["panel"]}
    metrics = dict(Counter(ours["results"][key]["metric"] for key in panel_ids))
    text_rows = [{"engine": m["engine"], "parameters": m["meta"]["served_params"],
                  "full": m["scores"]["balanced_skill"], "public": m["scores"]["public_skill"],
                  "same": m["v03"]["same_skills"], "new": m["v03"]["new_domains"]} for m in small]
    vision_rows = [{k: m[k] for k in ("engine", "params", "pub", "priv", "full")} for m in vl]
    hashes = {p: sha(p) for p in EVIDENCE}
    hashes.update({str(p): sha(p) for p in sorted(board.glob("*.json"))})
    hashes.update({str(p): sha(p) for p in sorted((root / "sources").rglob("*"))
                   if p.is_file() and p.suffix in (".md", ".json") and ".cache" not in p.parts})
    out = {"checked_date_kst": "2026-11-04", "source_sha256": hashes,
           "space_revision": read(board / "metadata.json")["revision"],
           "data_generated_utc": {"text": text["generated_utc"], "vision": vision["generated_utc"]},
           "text_small_models": text_rows, "vision_small_models": vision_rows,
           "strict_numeric_parameter_filter": "0 < served_params (text) or params (vision) <= 1,000,000,000; unknown sizes excluded; names ignored",
           "current_text_ordinal_position": 1 + [m["engine"] for m in small].index("crystal-0.8b"),
           "rank_caveat": "ordinal score order, not the board's statistical tie rank",
           "aspirational_official_targets": {"text": max(36.3, text_rows[0]["full"] + 5),
                                              "vision": max(50., vision_rows[0]["full"] + 5)},
           "score_policy": {"text": "60% private same +40% private new, equated; public not included",
                            "vision": vision["weights"]},
           "text_public_panel_metric_counts": metrics,
           "placement_5_seed_means": means,
           "placement_tests": placement["tests"],
           "focal_minus_ce_selection_accuracy": losses["tests"]["focal-all"]["selection correct"],
           "vision_screen": read("runs/vision-train/screen-report.json"),
           "initialization": "only the official original Qwen3.5-0.8B weights; no Crystal/QEV/GLiClass checkpoint",
           "resources": "current host GPU only, per latest user instruction",
           "canonical_latest_text_data": {"name": "v3-gold7", "train_rows": 999961, "train_sources": 113,
                                          "review": "wiki/project/champion-v1-g7-review.json"},
           "existing_input_audit_scope": "historical g6 only; not a g7 audit",
           "new_training_started": False}
    train.atomic_json(root / "evidence.json", out)
    print("EVIDENCE", {"text_leader": text_rows[0], "vision_leader": vision_rows[0], "metrics": metrics}, flush=True)


def restore(root):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(DATA_REPO, revision=DATA_REVISION, files_metadata=True)
    entries = {f.rfilename: f for f in info.siblings}
    def download(name):
        path = Path(hf_hub_download(DATA_REPO, name, repo_type="dataset", revision=DATA_REVISION,
                                    local_dir=str(root / "restored")))
        entry = entries[name]
        assert path.stat().st_size == entry.size
        digest = sha(path)
        if entry.lfs:
            assert digest == entry.lfs.sha256, name
        return {"file": name, "path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                "hub_lfs_sha256_checked": bool(entry.lfs)}
    with ThreadPoolExecutor(max_workers=3) as executor:
        restored = list(executor.map(download, FILES))
    train.atomic_json(root / "restoration.json", {"repository": DATA_REPO, "revision": DATA_REVISION, "files": restored,
                      "purpose": "candidate data/provenance audit; restoration is not approval for training all rows"})


def state_key(state):
    # Same normalization as historical fresh-state decontamination, but with sorted JSON object keys.
    value = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(re.sub(r"[\W_]+", " ", value.lower()).strip().encode()).hexdigest()


def identity_kind(row, seen):
    """Repeated ids require inspection even when every serialized field is identical."""
    digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    prior = seen.setdefault(row["id"], digest)
    return "identical_row" if prior == digest else "different_row_content"


def input_key(row):
    """Question/criteria may contain the entire task when state is empty; ignore option dictionary order."""
    value = {"state": row["state"], "question": row["question"]}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def target_signature(row):
    # Canonical input hashes ignore criteria order, so align target probabilities by semantic option key too.
    keyed = dict(zip(fmt.options(row["question"])[0], row["target"]))
    return json.dumps(keyed, ensure_ascii=False, sort_keys=True)


def has_context(row):
    return row["state"] not in (None, "", {}, []) and bool(str(row["state"]).strip())


def validate_row(row):
    keys = fmt.options(row["question"])[0]
    target = row["target"]
    assert 2 <= len(keys) <= fmt.MAX_OPTIONS
    assert len(keys) == len(target) and all(math.isfinite(x) and x >= 0 for x in target)
    assert abs(sum(target) - 1.) < 1e-5, row["id"]
    assert row["source"] not in mix.CORPUS_EXCLUDED
    assert mix.allowed_origin(row, {"human", "programmatic"})
    assert mix.label_origin(row) == row["label_origin"]
    assert mix.item_key(row) == row["item"]
    assert mix.target_kind(row) == row["target_kind"]


def audit_inputs(root):
    assert sha(BASE / "model.safetensors-00001-of-00001.safetensors") == BASE_SHA
    raw = root / "restored/data/mix"
    # Do not relabel any previously examined set as blind, or reset its partition.
    fresh = {state_key(r["state"]) for r in rows("data/fresh/fresh-v1.jsonl")}
    development = list(rows(raw / "v3-gold6.dev.jsonl"))
    dev_keys = set(); dev_states = set(); dev_inputs = set()
    for row in development:
        validate_row(row)
        dev_keys.update([row["item"], *row.get("documents", [])]); dev_inputs.add(input_key(row))
        if has_context(row): dev_states.add(state_key(row["state"]))
    stats = defaultdict(lambda: {"rows": 0, "types": Counter(), "origins": Counter(), "options": Counter(),
                                "noul_targets": Counter(), "target_kinds": Counter()})
    leaked, state_overlap, fresh_overlap, full_overlap = Counter(), Counter(), Counter(), Counter()
    input_targets = {}; multiple_targets = Counter(); input_repeats = Counter(); empty = Counter()
    ids = {}; total = 0; duplicates = defaultdict(Counter); duplicate_examples = []
    for row in rows(raw / "v3-gold6.jsonl"):
        validate_row(row)
        repeated = row["id"] in ids
        kind = identity_kind(row, ids)
        if repeated:
            duplicates[row["source"]][kind] += 1
            if len(duplicate_examples) < 20: duplicate_examples.append({"id": row["id"], "kind": kind})
        total += 1
        s = stats[row["source"]]; s["rows"] += 1
        s["types"][row["question"]["type"]] += 1; s["origins"][row["label_origin"]] += 1
        s["options"][len(row["target"])] += 1; s["target_kinds"][row["target_kind"]] += 1
        if row["question"]["type"] == "noul": s["noul_targets"][str(int(row["target"][1] >= .5))] += 1
        if dev_keys.intersection([row["item"], *row.get("documents", [])]): leaked[row["source"]] += 1
        whole_input = input_key(row); signature = target_signature(row)
        if whole_input in dev_inputs: full_overlap[row["source"]] += 1
        if whole_input in input_targets:
            input_repeats[row["source"]] += 1
            if input_targets[whole_input] != signature: multiple_targets[row["source"]] += 1
        else: input_targets[whole_input] = signature
        if not has_context(row): empty[row["source"]] += 1
        key = state_key(row["state"])
        if has_context(row) and key in dev_states: state_overlap[row["source"]] += 1
        if key in fresh: fresh_overlap[row["source"]] += 1
        if total % 100000 == 0: print("AUDIT_ROWS", total, flush=True)
    manifest = read(raw / "v3-gold6.manifest.json")
    assert total == manifest["rows"] and len(development) == manifest["dev"]["rows"]
    assert {s: v["rows"] for s, v in stats.items()} == manifest["counts"]
    # A failed gate leaves a report; never silently repair historical inputs or claim they're ready.
    checks = {"duplicate_row_ids_zero": not duplicates,
              "upstream_item_or_document_overlap_zero": not leaked,
              "canonical_full_input_train_dev_overlap_zero": not full_overlap,
              "same_full_input_multiple_targets_zero": not multiple_targets,
              "normalized_full_state_fresh_overlap_zero": not fresh_overlap}
    report = {"complete": True, "checked_rows": total, "dev_rows": len(development), "sources": dict(stats),
              "unique_training_ids": len(ids), "duplicate_id_extra_rows": dict(duplicates),
              "duplicate_id_examples": duplicate_examples,
              "canonical_full_input_train_dev_overlap": dict(full_overlap),
              "repeated_full_input_extra_rows": dict(input_repeats),
              "same_full_input_multiple_targets_extra_rows": dict(multiple_targets),
              "empty_state_rows": dict(empty),
              "checks": checks, "all_overlap_checks_pass": all(checks.values()),
              "upstream_item_or_document_train_dev_overlap": dict(leaked),
              "normalized_full_state_train_dev_overlap": dict(state_overlap),
              "state_overlap_caveat": "empty states excluded; nonempty state overlap alone is descriptive, not a full-input leakage verdict; compare state+question+criteria",
              "normalized_full_state_train_fresh_overlap": dict(fresh_overlap),
              "base_official_lfs_sha256_checked": BASE_SHA,
              "model_origin_exception": sorted(mix.MODEL_ORIGIN_ALLOWED),
              "neural_training_started": False, "corpus_ready_for_new_training": False,
              "auditor_sha256": sha(__file__),
              "remaining_gates": ["split/provenance and licence review for the selected new subset",
                                  "official benchmark contamination filter and semantic near-duplicates",
                                  "preserve existing JEVal protocol; exclude all evaluation items from training",
                                  "metric-compatible evaluation and frozen sampling/budget/selection plan",
                                  "vision inputs restored and audited separately"],
              "limits": "label provenance cross-checked against the existing registry, not re-annotated; duplicate ids classified by complete row content, not a semantic label-conflict diagnosis; multiple targets can reflect annotation disagreement and require review; canonical full-input overlap ignores JSON key/option-dictionary order but is not semantic decontamination; no token-length, image, licence, or all-benchmark contamination clearance"}
    train.atomic_json(root / "input-audit.json", report)
    print("INPUT_AUDIT", {"rows": total, "dev": len(development), "checks": checks}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evidence", "restore-audit", "audit"))
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "evidence":
        evidence(args.out); return
    started = time.monotonic()
    try:
        shutil.copyfile(__file__, args.out / f"audit-source-{sha(__file__)[:12]}.py")
        if args.command == "restore-audit":
            train.atomic_json(args.out / "input-status.json", {"state": "restoring", "training_started": False})
            restore(args.out)
        train.atomic_json(args.out / "input-status.json", {"state": "auditing", "training_started": False})
        audit_inputs(args.out)
        train.atomic_json(args.out / "input-status.json", {"state": "audit_complete", "training_started": False,
                          "seconds": time.monotonic() - started,
                          "all_overlap_checks_pass": read(args.out / "input-audit.json")["all_overlap_checks_pass"]})
    except Exception:
        train.atomic_json(args.out / "input-status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
