"""Full g7 core plus audited, task-specific training supplements; no source quotas.

Produces text inputs only. Mind2Web preparation runs separately. No GPU training
starts here: token-length preflight and a complete modality schedule remain gates.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import fnmatch
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import time
import traceback
from urllib.parse import urlsplit
import urllib.request

from qwen_decision.v3 import champion_audit as audit, champion_joint_data as old
from qwen_decision.v3 import decontam, format as fmt, mix, train
from qwen_decision.v3.sources import common, commonsense, nli, phishing, stance_sarcasm

ROOT = Path("runs/champion-v1/full-g7-v3")
RAW = ROOT / "raw/enrichment"
MIX = ROOT / "mix"
G7 = old.ROOT / "restored/data/mix"
DECONTAM = old.ROOT / "restored/data/decontam"
SEED = 20261105


def status(stage, **extra):
    train.atomic_json(ROOT / "enrichment-status.json", {"stage": stage, "training_started": False, **extra})


def restore():
    from huggingface_hub import HfApi, hf_hub_download

    hub = [(phishing.ZENODO[0], phishing.ZENODO[1], ["Nazario_5.csv", "Nigerian_5.csv"]),
           (nli.REPO, nli.REVISION, ["train.jsonl"]),
           (commonsense.HELLASWAG_REPO, commonsense.HELLASWAG_REVISION, ["data/train-*", "data/validation-*"]),
           (commonsense.WINOGRANDE_REPO, commonsense.WINOGRANDE_REVISION, ["winogrande_xl/train-*"])]
    plans = []
    for repo, revision, patterns in hub:
        info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision
        selected = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)]
        assert selected
        for f in selected:
            plans.append({"repo": repo, "revision": revision, "file": f.rfilename,
                          "directory": str(RAW / repo.replace("/", "__")), "bytes": f.size,
                          "sha256": f.lfs.sha256 if f.lfs else None})
    train.atomic_json(ROOT / "enrichment-download-plan.json", {"hub_files": plans})

    def one(p):
        file = Path(hf_hub_download(p["repo"], p["file"], revision=p["revision"],
                                    repo_type="dataset", local_dir=p["directory"]))
        digest = audit.sha(file)
        assert file.stat().st_size == p["bytes"] and (p["sha256"] is None or digest == p["sha256"])
        print("RESTORED", p["repo"], p["file"], flush=True)
        return {**p, "local": str(file), "sha256": digest}

    with ThreadPoolExecutor(4) as pool:
        files = list(pool.map(one, plans))
    git = [("iabufarha/iSarcasmEval", stance_sarcasm.ISARCASM_COMMIT,
            "train/train.En.csv", RAW / "isarcasm/train/train.En.csv"),
           *[("emilyallaway/zero-shot-stance", stance_sarcasm.VAST_COMMIT,
              f"data/VAST/vast_{split}.csv", RAW / f"vast/data/VAST/vast_{split}.csv") for split in ("train", "dev")]]
    for repo, revision, filename, path in git:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as source, tmp.open("wb") as out:
                import shutil
                shutil.copyfileobj(source, out)
            tmp.replace(path)
        files.append({"repo": repo, "revision": revision, "file": filename,
                      "url": url, "local": str(path), "sha256": audit.sha(path), "bytes": path.stat().st_size})
    train.atomic_json(ROOT / "enrichment-restoration.json", {"complete": True, "files": files})
    return files


def bound_fetch(repo, patterns=None, repo_type="dataset", revision=None):
    """Existing converters may read only files already declared in this restoration."""
    plan = audit.read(ROOT / "enrichment-restoration.json")["files"]
    found = [p for p in plan if p["repo"] == repo and p["revision"] == revision and "directory" in p]
    assert found and repo_type == "dataset", (repo, revision)
    for pattern in patterns or ["*"]:
        assert any(fnmatch.fnmatch(p["file"], pattern) for p in found), (repo, pattern)
    return Path(found[0]["directory"]), revision


def binary_views(base, positive, positive_name, negative_name, subject, instruction, evidence=None):
    """Three views of the same annotated fact, preserving false/true and option meaning."""
    base = mix.annotate(base)
    original = base["id"]
    rng = random.Random(f"{SEED}:{original}")
    names = [positive_name, negative_name]
    rng.shuffle(names)
    state = base["state"] if evidence is None else evidence
    question, target = common.choice(state, instruction, names,
                                     names.index(positive_name if positive else negative_name))
    views = [("choice", question, target),
             ("positive", *common.noul(f"{subject} {positive_name}.", positive)),
             ("negative", *common.noul(f"{subject} {negative_name}.", not positive))]
    for view, question, target in views:
        yield {**base, "id": original + ":coverage:" + view, "state": state, "question": question,
               "target": target, "kind": question["type"], "target_kind": "single",
               "coverage_family": positive_name, "coverage_label": bool(positive),
               "augmentation": "three supervised views of the same upstream annotation"}


def phishing_views():
    # These two positive corpora annotate phishing/419 fraud. Do not relabel
    # SpamAssassin's general unsolicited spam as phishing to fill the quota.
    urls, hosts = set(), set()
    email_count = 0
    for row in audit.rows("runs/champion-v1/public-eval-v1/text-counted-rows.jsonl"):
        if row["_evaluation"]["catalog_id"] != 56:
            continue
        email_count += 1
        observed = row["state"]["link_url"].strip()
        assert urlsplit(observed).hostname, row["id"]
        urls.add(observed.rstrip("/").lower())
        hosts.add(urlsplit(observed).hostname.lower())
        # Exclusion only: never use an evaluation answer as a training label.
        for url in phishing.links(" ".join(decontam.strings(row["state"]))):
            urls.add(url.rstrip("/").lower())
            host = (urlsplit(url).hostname or "").lower()
            if host:
                hosts.add(host)
    assert email_count == 2000 and urls and hosts
    seen = set()
    for filename in ("Nazario_5.csv", "Nigerian_5.csv"):
        path = RAW / phishing.ZENODO[0].replace("/", "__") / filename
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                if r["label"] not in ("0", "1"):
                    continue
                body = r["body"].strip()
                if len(body) < 40:
                    continue
                key = hashlib.sha1(" ".join(body.lower().split()).encode()).hexdigest()
                if key in seen:
                    continue
                links = phishing.links(body)
                if any(u.rstrip("/").lower() in urls or (urlsplit(u).hostname or "").lower() in hosts for u in links):
                    continue
                seen.add(key)
                sender = phishing.SENDER.match(r["sender"])
                state = {"sender": sender["name"].strip() if sender else "",
                         "from": sender["addr"].strip() if sender else r["sender"].strip(),
                         "subject": r["subject"].strip(), "body": body}
                if links:
                    state["observed_links"] = [{"url": u, "hostname": urlsplit(u).hostname or ""} for u in links]
                # Original raw-body digest keeps the original g7 item identity.
                raw_hash = hashlib.sha1(r["body"].encode()).hexdigest()[:12]
                base = common.example("phishing", f"zenodo:{filename}:{r['label']}:{raw_hash}",
                                      "phishing", state, {}, [1., 0.], "train", kind="choice")
                for view in binary_views(base, r["label"] == "1", "fraudulent or phishing", "legitimate",
                                         "This email is", "Classify this email using its sender, content and observed links."):
                    yield {**view, "upstream": phishing.ZENODO[0], "upstream_revision": phishing.ZENODO[1],
                           "upstream_file": filename, "input_features": "Link hostnames are parsed from observed URLs; no model-generated evidence"}


def sarcasm_views():
    path = RAW / "isarcasm/train/train.En.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for i, r in enumerate(csv.DictReader(handle)):
            text = (r["tweet"] or "").strip()
            if not text or r["sarcastic"] not in ("0", "1"):
                continue
            base = common.example("isarcasm-train", f"A:En:{i}", "sarcasm", {"text": text}, {}, [1., 0.], "train", kind="choice")
            for view in binary_views(base, r["sarcastic"] == "1", "sarcastic", "not sarcastic",
                                     "This text is", "Does the author intend this text sarcastically or literally?"):
                yield {**view, "upstream": "iabufarha/iSarcasmEval", "upstream_revision": stance_sarcasm.ISARCASM_COMMIT,
                       "upstream_file": "train/train.En.csv", "coverage_task": "English task A; not Arabic or category-B negatives"}


def wanli_all():
    path = RAW / nli.REPO.replace("/", "__") / "train.jsonl"
    for r in audit.rows(path):
        rng = random.Random(f"{SEED}:{r['id']}")
        labels = rng.choice(nli.LABELS)
        order = list(range(3))
        rng.shuffle(order)
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        question, target = common.choice(state, rng.choice(nli.INSTRUCTIONS),
                                         {labels[i][0]: labels[i][1] for i in order}, order.index(nli.GOLD[r["gold"]]))
        out = mix.annotate(common.example(nli.NAME, str(r["id"]), "nli", state, question, target, "train"))
        yield {**out, "upstream": nli.REPO, "upstream_revision": nli.REVISION,
               "coverage_task": "Full human-annotated WANLI train, no sample cap"}


def supplements():
    common.fetch = bound_fetch
    stance_sarcasm.RAW = RAW
    csv.field_size_limit(1 << 28)
    yield from phishing_views()
    yield from sarcasm_views()
    yield from wanli_all()
    for name, fn in [(stance_sarcasm.VAST, stance_sarcasm.vast),
                     (commonsense.HELLASWAG, commonsense.hellaswag),
                     (commonsense.WINOGRANDE, commonsense.winogrande)]:
        counts = {}
        for r in fn(counts):
            yield {**mix.annotate(r), "coverage_task": "Full allowed upstream pool, no sample cap"}
        print("SOURCE_CONVERTER", name, json.dumps(counts), flush=True)


def _norm_hash(text):
    return hashlib.sha256(re.sub(r"[\W_]+", " ", text.lower()).strip().encode()).hexdigest()


_FILTER = _DEV_KEYS = _DEV_INPUTS = _FRESH = _HELDOUT_FIELDS = None


def worker_init():
    global _FILTER, _DEV_KEYS, _DEV_INPUTS, _FRESH, _HELDOUT_FIELDS
    _FILTER = decontam.Filter(DECONTAM)
    dev = list(audit.rows(G7 / "v3-gold7.dev.jsonl"))
    _DEV_KEYS = set().union(*(old.keys(r) for r in dev))
    _DEV_INPUTS = {audit.input_key(r) for r in dev}
    _FRESH = {audit.state_key(r["state"]) for r in audit.rows("data/fresh/fresh-v1.jsonl")}
    # Protect shared posts/documents even when a new cast changes state field names.
    _HELDOUT_FIELDS = {_norm_hash(s) for r in itertools.chain(dev, audit.rows("data/fresh/fresh-v1.jsonl"))
                       for s in decontam.strings(r["state"]) if len(s) >= 120}


def filter_batch(batch):
    kept, dropped = [], Counter()
    for row in batch:
        assert old.eligible_origin(row), row["id"]
        options, _ = fmt.options(row["question"])
        assert len(options) == len(row["target"]) and 2 <= len(options) <= fmt.MAX_OPTIONS
        assert all(math.isfinite(v) and v >= 0 for v in row["target"]) and abs(sum(row["target"]) - 1.) < 1e-5
        if old.keys(row) & _DEV_KEYS:
            dropped["heldout_item_or_document"] += 1
            continue
        if audit.input_key(row) in _DEV_INPUTS:
            dropped["heldout_full_input"] += 1
            continue
        if audit.has_context(row) and audit.state_key(row["state"]) in _FRESH:
            dropped["fresh_state"] += 1
            continue
        if any(_norm_hash(s) in _HELDOUT_FIELDS for s in decontam.strings(row["state"]) if len(s) >= 120):
            dropped["heldout_content_field"] += 1
            continue
        if _FILTER.flagged(decontam.row_texts(row["state"], {"q": row["question"]})):
            dropped["public_jeval_overlap"] += 1
            continue
        kept.append(row)
    return kept, dict(dropped)


def chunks(iterator, size=256):
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def bounded_map(pool, fn, iterable, limit=12):
    iterator = iter(iterable)
    pending = deque(pool.submit(fn, arg) for arg in itertools.islice(iterator, limit))
    while pending:
        yield pending.popleft().result()
        try:
            arg = next(iterator)
        except StopIteration:
            continue
        pending.append(pool.submit(fn, arg))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    MIX.mkdir(exist_ok=True)
    if (ROOT / "enriched-text-prepared.json").exists():
        done = audit.read(ROOT / "enriched-text-prepared.json")
        assert done["source_sha256"] == audit.sha(__file__)
        assert audit.sha(done["file"]) == done["sha256"]
        return
    started = time.monotonic()
    status("restoring")
    restored = restore()
    review = audit.read("runs/champion-v1/g7-review/review.json")["hub_files_match_g7_training_manifest"]
    for path, entry in review.items():
        assert audit.sha(old.ROOT / "restored" / path) == entry["sha256"]
    # Materialize new casts with upstream identity before filtering, for a complete audit trail.
    supplement_path = ROOT / "enrichment-candidates.jsonl"
    temporary = supplement_path.with_suffix(".partial")
    status("building_supplements")
    with temporary.open("w") as out:
        for row in supplements():
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(supplement_path)
    status("filtering_full_core_and_supplements")
    raw_counts, dropped, counts, added_counts, families = Counter(), Counter(), Counter(), Counter(), Counter()
    seen, conflict = {}, set()
    candidate = MIX / "text-enriched.preconflict.jsonl"
    visited = 0

    def source_rows():
        for r in audit.rows(G7 / "v3-gold7.jsonl"):
            raw_counts[r["source"]] += 1
            yield {**r, "training_component": "g7-core"}
        for r in audit.rows(supplement_path):
            yield {**r, "training_component": "targeted-supplement"}

    with ProcessPoolExecutor(4, initializer=worker_init) as pool, candidate.open("w") as out:
        # Bound outstanding batches; do not eagerly load the million-row input.
        for kept, rejected in bounded_map(pool, filter_batch, chunks(source_rows())):
            dropped.update(rejected)
            visited += len(kept) + sum(rejected.values())
            for row in kept:
                key, signature = audit.input_key(row), old.digest(audit.target_signature(row))
                if key in seen:
                    if seen[key] != signature:
                        conflict.add(key)
                        dropped["conflicting_duplicate_input"] += 1
                    else:
                        dropped["identical_semantic_input_and_target"] += 1
                    continue
                seen[key] = signature
                out.write(json.dumps({**row, "modality": "text", "original_id": row["id"],
                                      "id": row["id"] + ":variant:" + key[:16]}, ensure_ascii=False) + "\n")
            if visited // 16384 != (visited - len(kept) - sum(rejected.values())) // 16384:
                status("filtering_full_core_and_supplements", examined=visited, unique_candidates=len(seen), dropped=dict(dropped))
                print("FILTERED", visited, len(seen), flush=True)
    manifest = audit.read(G7 / "v3-gold7.manifest.json")
    assert sum(raw_counts.values()) == 999961 and dict(raw_counts) == manifest["counts"]
    destination = MIX / "text-train.jsonl"
    tmp = destination.with_suffix(".partial")
    items, new_items, core_items = set(), set(), set()
    with tmp.open("w") as out:
        for row in audit.rows(candidate):
            if audit.input_key(row) in conflict:
                dropped["original_of_conflicting_input"] += 1
                continue
            counts[row["source"]] += 1
            items.add(row["item"])
            if row["training_component"] == "targeted-supplement":
                added_counts[row["source"]] += 1
                new_items.add(row["item"])
                families[(row["source"], row.get("coverage_task", row.get("coverage_family", "full-pool")))] += 1
            else:
                core_items.add(row["item"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(destination)
    report = {"complete": True, "training_started": False, "raw_g7_rows": 999961,
              "rows": sum(counts.values()), "distinct_upstream_items": len(items), "source_counts": dict(counts),
              "targeted_supplement_rows": dict(added_counts), "distinct_supplement_items": len(new_items),
              "supplement_items_absent_from_retained_g7": len(new_items - core_items),
              "targeted_tasks": {str(k): v for k, v in families.items()}, "dropped": dict(dropped),
              "conflicting_input_groups_quarantined": len(conflict), "sample_cap": None,
              "file": str(destination), "sha256": audit.sha(destination), "source_sha256": audit.sha(__file__),
              "raw_inputs": restored, "seconds": time.monotonic() - started,
              "scope": "All eligible g7 core + targeted human-annotated pools and supervised format views. No new teacher labels or ranking loss. Raw row counts are not distinct items. Token-length preflight, expanded vision mix and training schedule remain before GPU launch."}
    train.atomic_json(ROOT / "enriched-text-prepared.json", report)
    status("text_prepared", rows=report["rows"], supplement_rows=sum(added_counts.values()),
           new_supplement_items=report["supplement_items_absent_from_retained_g7"])
    print("TEXT_PREPARED", report["rows"], dict(added_counts), flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except Exception:
        status("failed", traceback=traceback.format_exc())
        raise
