"""Build gold pairs and an unused document holdback before any model scoring."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from qwen_decision.v3 import evidence_pairs as pairs, long_training as common
from qwen_decision.v3 import fresh_eval as fresh
from qwen_decision.v3.reference_comparison import as_example
from qwen_decision.v3.fresh_holdback import state_key

PUBLIC = Path("runs/evidence-pairs/public-v1")
TRAIN = Path("data/mix/v3-gold6.jsonl")


def lines(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def strings(value):
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                yield from strings(json.loads(value))
                return
            except ValueError:
                pass
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def metadata():
    pages, originals = {}, {}
    vc = PUBLIC / "tals__vitaminc"
    for name in ("train.jsonl", "test.jsonl"):
        for r in lines(vc / name):
            if r["revision_type"] == "real" and not str(r.get("FEVER_id") or ""):
                pages[f"vitaminc-train:{r['case_id']}"] = state_key(r["page"])
    for split in ("train", "dev", "test"):
        for r in lines(PUBLIC / "lasha-nlp__CONDAQA" / f"{split}.json"):
            originals[f"condaqa-train:{split}{r['PassageID']}"] = state_key(r["original passage"])
    return pages, originals


def keys(row, meta):
    content = row["state"] if row["state"] not in (None, "", {}) else row["question"]
    content_hash = state_key(content)
    result = [("id", row.get("upstream_id", row["id"])), ("canonical_state", content_hash)]
    if row.get("item"):
        result.append(("item", row["item"]))
    result += [("document", d) for d in row.get("documents") or []]
    for text in strings(row["state"]):
        if len(text) >= 80:
            result.append(("long_content", content_hash if text == content else state_key(text)))
    pages, originals = meta
    item = row.get("item")
    if item in pages:
        result.append(("vitaminc_article", pages[item]))
    if item in originals:
        result.append(("condaqa_original_passage", originals[item]))
    return result


def general_candidates():
    """Unused rows from official held-out splits; no training use."""
    for r in fresh.rows():
        r = as_example(r)
        yield r | {"item": f"retention:{r['source']}:{state_key(r['state'])}", "documents": [], "label_origin": "dataset_gold"}
    import pyarrow.parquet as pq
    specs = [("google__boolq/data/validation-00000-of-00001.parquet", "boolq"),
             ("stanfordnlp__snli/plain_text/validation-00000-of-00001.parquet", "snli"),
             ("nyu-mll__multi_nli/data/validation_matched-00000-of-00001.parquet", "mnli"),
             ("Rowan__hellaswag/data/validation-00000-of-00001.parquet", "hellaswag")]
    for filename, name in specs:
        for i, r in enumerate(pq.read_table(PUBLIC / filename).to_pylist()):
            if name == "boolq":
                state = {"passage": r["passage"], "question": r["question"]}
                options, label = ["no", "yes"], int(r["answer"])
                asked = "Answer the question using the passage."
            elif name in ("snli", "mnli"):
                if int(r["label"]) < 0:
                    continue
                state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
                options, label = ["entailment", "neutral", "contradiction"], int(r["label"])
                asked = "How does the hypothesis relate to the premise?"
            else:
                state, options, label = r["ctx"], list(r["endings"]), int(r["label"])
                asked = "Which ending most likely continues the scene?"
            source = f"retention-{name}"
            ident = f"{source}:validation:{r.get('pairID', r.get('idx', i))}"
            yield {"id": ident, "source": source, "state": state,
                   "question": {"type": "choice", "instructions": asked,
                                "criteria": {f"option_{j+1}": t for j, t in enumerate(options)}},
                   "target": [float(j == label) for j in range(len(options))], "ordered": False,
                   "item": f"{source}:{state_key(state)}", "documents": [], "label_origin": "dataset_gold"}


def public_test():
    labels = {"YES": 0, "NO": 1, "DON'T KNOW": 2}
    for r in lines(PUBLIC / "lasha-nlp__CONDAQA/test.json"):
        if r["label"] not in labels:
            continue
        item = f"condaqa-train:test{r['PassageID']}"
        ident = f"{item}:{r['PassageEditID']}:{r['QuestionID']}:{r['SampleID']}"
        yield {"id": ident, "upstream_id": ident, "source": "condaqa-train", "family": "negation-reading",
            "kind": "choice", "state": {"passage": r["sentence1"], "question": r["sentence2"]},
            "question": {"type": "choice", "instructions": pairs.INSTRUCTIONS["condaqa-train"],
                         "criteria": dict.fromkeys(pairs.LABELS["condaqa-train"])},
            "target": [float(i == labels[r["label"]]) for i in range(3)], "ordered": False,
            "split": "test", "label_origin": "human", "target_kind": "single", "item": item, "documents": []}
    for r in lines(PUBLIC / "tals__vitaminc/test.jsonl"):
        if r["revision_type"] != "real" or str(r.get("FEVER_id") or ""):
            continue
        item = f"vitaminc-train:{r['case_id']}"
        ident = f"{item}:{r['unique_id']}"
        yield {"id": ident, "upstream_id": ident, "source": "vitaminc-train", "family": "fact-verification",
            "kind": "choice", "state": {"evidence": r["evidence"], "claim": r["claim"]},
            "question": {"type": "choice", "instructions": pairs.INSTRUCTIONS["vitaminc-train"],
                         "criteria": dict.fromkeys(pairs.LABELS["vitaminc-train"])},
            "target": [float(k == r["label"]) for k in pairs.LABELS["vitaminc-train"]], "ordered": False,
            "split": "test", "label_origin": "human", "target_kind": "single", "item": item, "documents": [],
            "wiki_page": r["page"], "wiki_revision_id": r["wiki_revision_id"], "revision_type": "real"}


def audit_candidates(candidates, meta, out):
    """Union candidate document components, then exclude entire touched groups."""
    parent = list(range(len(candidates)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(candidates):
        for key in keys(row, meta):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    blocked = set()
    audit, files = {}, {}
    def mark(row):
        for key in keys(row, meta):
            if key in seen:
                blocked.add(find(seen[key]))

    corpora = [Path("data/mix") / f"v3-gold{s}.jsonl" for s in (2, 3, 6)]
    prior_raw = [Path("data/mix/v3-gold3.dev.jsonl"), Path("data/fresh/fresh-v1.jsonl")]
    prediction_files = []
    for path in sorted(Path("runs").rglob("*.jsonl")):
        if "evidence-pairs" in path.parts:
            continue
        try:
            with path.open() as f:
                first = json.loads(next(f))
        except (ValueError, StopIteration):
            continue
        if not isinstance(first, dict):
            continue
        if "probabilities" in first and "id" in first:
            prediction_files.append(path)
        elif "state" in first and "question" in first and any(w in path.name for w in ("selection", "confirm", "example")):
            prior_raw.append(path)
    for path in corpora + sorted(set(prior_raw)):
        before = len(blocked)
        for count, r in enumerate(lines(path), 1):
            mark(r)
            if count % 200000 == 0:
                print("AUDIT", path, count, len(blocked), flush=True)
        files[str(path)] = common.train.sha256_file(path)
        audit[str(path)] = {"rows": count, "new_blocked_documents": len(blocked) - before}
    for path in prediction_files:
        for r in lines(path):
            for key in (("id", r["id"]), ("item", r.get("group")), ("document", r.get("group"))):
                if key in seen:
                    blocked.add(find(seen[key]))
        files[str(path)] = common.train.sha256_file(path)
    import pyarrow.parquet as pq
    jeval = Path("data/jeval/data/test.parquet")
    for batch in pq.ParquetFile(jeval).iter_batches(columns=["state"]):
        for i, r in enumerate(batch.to_pylist()):
            mark({"id": f"jeval-audit:{i}", "state": r["state"], "question": {}})
    files[str(jeval)] = common.train.sha256_file(jeval)
    retained = []
    for i, row in enumerate(candidates):
        group = find(i)
        if group not in blocked:
            retained.append(row | {"group": common.search.digest(["pair-document-v1", candidates[group]["id"]])})
    common.search.save_records(out / "retained-candidates.jsonl", retained)
    result = {"candidate_rows": len(candidates), "retained_rows": len(retained),
              "retained_sources": dict(Counter(r["source"] for r in retained)),
              "retained_documents": len({r["group"] for r in retained}),
              "blocked_documents": len(blocked), "past_prediction_files": len(prediction_files), "corpus_audit": audit,
              "keys": ["upstream id", "explicit item/document", "canonical complete state", "long normalized state strings",
                       "VitaminC Wikipedia article from official metadata", "CONDAQA original passage across edits"],
              "limits": ["does not audit unknown backbone pretraining", "CONDAQA grouping is original passage, not Wikipedia article",
                         "exact normalized fingerprints and explicit metadata, not exhaustive semantic near-duplicate detection"]}
    common.train.atomic_json(out / "split-audit.json", result)
    print("AUDIT_RETAINED", result["retained_sources"], flush=True)
    return retained, files


def balanced_anchors():
    by = defaultdict(list)
    for r in lines(common.OUT / "train.jsonl"):
        if r["source"] not in pairs.SOURCES and r.get("label_origin") in ("human", "programmatic"):
            by[r["source"]].append(r)
    for s in by:
        by[s].sort(key=lambda r: common.search.digest(["pair-anchors-v1", r["id"]]))
    selected = []
    at = 0
    while len(selected) < 4096:
        for source in sorted(by):
            if at < len(by[source]) and len(selected) < 4096:
                selected.append(by[source][at])
        at += 1
    return selected


def independent_retention(row):
    """Structured contexts must be long enough for the cross-corpus key audit.

    Plain string states are already covered by complete-state fingerprints.
    A short NLI premise with a different hypothesis must not slip through as
    an apparently new document just because its complete state is different.
    """
    state = row["state"]
    if not isinstance(state, dict):
        return True
    contexts = [state[k] for k in ("premise", "passage", "evidence", "background", "sentence 1", "sentence 2")
                if k in state and isinstance(state[k], str)]
    return bool(contexts) and all(len(s) >= 80 for s in contexts)


def build(out):
    """All choices below are deterministic and precede any model evaluation."""
    renderer, _ = common.search.renderer_for(common.search.BASE)
    public_manifest = common.read(PUBLIC / "manifest.json")
    files = {str(PUBLIC / "manifest.json"): common.train.sha256_file(PUBLIC / "manifest.json")}
    for source in public_manifest["sources"]:
        for name, sha in source["files"].items():
            if common.train.sha256_file(Path(name)) != sha:
                raise ValueError("public snapshot changed")
            files[name] = sha
    for path in sorted(Path("data/raw/fresh").rglob("*")):
        if path.is_file():
            files[str(path)] = common.train.sha256_file(path)
    meta = metadata()
    candidate = list(public_test())
    candidate += [r for r in lines(Path("data/mix/v3-gold6.dev.jsonl")) if r["source"] not in pairs.SOURCES]
    candidate += list(general_candidates())
    counts = Counter(r["id"] for r in candidate)
    collided = {r["item"] for r in candidate if counts[r["id"]] > 1}
    candidate = [r for r in candidate if r["item"] not in collided]
    retained, audited = audit_candidates(candidate, meta, out)
    files.update(audited)
    files["data/mix/v3-gold6.dev.jsonl"] = common.train.sha256_file(Path("data/mix/v3-gold6.dev.jsonl"))
    held = []
    for r in retained:
        if len(common.views.view(renderer, r)["input_ids"]) <= 4096:
            held.append(r)
    held_pairs, held_rejected = pairs.make_pairs([r for r in held if r["source"] in pairs.SOURCES])
    development, final = [], []
    for dataset, dev in ((development, True), (final, False)):
        for source in pairs.SOURCES:
            for kind in ("flip", "invariant"):
                pool = [p for p in held_pairs if p["source"] == source and p["kind"] == kind
                        and ((int(p["group"][:8], 16) % 5 == 0) == dev)]
                cap = (32 if kind == "flip" else 16) if dev else (256 if kind == "flip" else 128)
                dataset += sorted(pool, key=lambda p: p["id"])[:cap]
    if sum(p["kind"] == "flip" for p in development) < 20 or sum(p["kind"] == "flip" for p in final) < 100:
        raise ValueError("insufficient genuinely unused document pairs")
    general = []
    by = defaultdict(dict)
    for r in held:
        if r["source"] not in pairs.SOURCES and independent_retention(r):
            by[r["source"]].setdefault(r["group"], r)
    for source, groups in sorted(by.items()):
        if len(groups) >= 12:
            general += sorted(groups.values(), key=lambda r: common.search.digest(["pair-retention-v1", r["id"]]))[:64]
    if len(general) < 300:
        raise ValueError("insufficient fresh general-retention cases")
    train_rows = []
    for r in lines(TRAIN):
        if r["source"] not in pairs.SOURCES:
            continue
        c = pairs.canonical(r)
        if c is not None and len(common.views.view(renderer, c)["input_ids"]) <= 1536:
            doc = [k for k in keys(c, meta) if k[0] in ("vitaminc_article", "condaqa_original_passage")]
            c["group"] = common.search.digest(doc or [("item", c["item"])])
            train_rows.append(c)
    pool, train_rejected = pairs.make_pairs(train_rows)
    training = []
    for source in pairs.SOURCES:
        pp = sorted((p for p in pool if p["source"] == source and p["kind"] == "flip"), key=lambda p: p["id"])
        if len(pp) < 128:
            raise ValueError(f"insufficient training pairs: {source}")
        training += pp[:512]
    anchors = balanced_anchors()
    data = {"train-pairs": training, "anchor-train": anchors, "pair-dev": development,
            "pair-test": final, "retention-test": general}
    for name, rr in data.items():
        path = out / f"{name}.jsonl"
        common.search.save_records(path, rr)
        files[str(path)] = common.train.sha256_file(path)
    summary = {"rows": {k: len(v) for k, v in data.items()},
               "duplicate_id_documents_dropped": sorted(collided),
               "pairs": {k: dict(Counter(f"{p['source']}:{p['kind']}" for p in data[k])) for k in ("train-pairs", "pair-dev", "pair-test")},
               "pair_documents": {k: len({p["group"] for p in data[k]}) for k in ("train-pairs", "pair-dev", "pair-test")},
               "retention_sources": dict(Counter(r["source"] for r in general)),
               "rejections": {"train": train_rejected, "test": held_rejected},
               "vitaminc_filter": "official real revisions only, no FEVER examples or synthetic revisions",
               "canonicalization": "generic instruction and equivalent choice rendering/order; evidence, question/claim and gold label unchanged"}
    common.train.atomic_json(out / "data-summary.json", summary)
    print("DATA_READY", summary["rows"], summary["pairs"], flush=True)
    return files, summary
