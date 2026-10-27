"""Long-input test sets from public data (evaluation only) and an uncapped training pool.

  build  data/longctx/test/<set>.jsonl from the pinned raw downloads in data/raw/longtest/
  pool   data/longctx/train-nocap-64k.jsonl: the long-training sampler over v3-gold6 without its
         1,536-token filter, so long rows enter training whole
  pool-long  data/longctx/train-long25-32k.jsonl: 32,768 gold6 rows, 25% over 1,536 tokens
  audit  prompt token lengths of every row, then overlap (also its own stage): exact states and
         non-template 13-word shingles shared between the test sets and every training mix

Nothing is truncated anywhere. A test prompt longer than the model's context (262,144 tokens) cannot
be scored whole; audit counts it as unsupported, which the board scores as wrong.

Test sets (never trained on):
  babilong      RMT-team/babilong qa1-qa5 at 0k-128k (100 items each), choice over the task's closed
                answer set; the same bAbI item at every length, so items pair across lengths
  qasper        allenai QASPER v0.3 test: yes/no answers and answerability, as noul, where every
                annotator agrees; answerability balanced to the number of unanswerable questions
  longbench-v2  THUDM/zai-org LongBench v2, all 503 four-option questions
  docnli-long   DocNLI test premises of at least 5,000 characters, entailment balanced 600/600
  jeval-long    the JEVal dev half of its seven long sources (LongBench v1 retrieval/TREC/LSHT, HELMET)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/longtest"
OUT = ROOT / "data/longctx"
TEST = OUT / "test"
POOL = OUT / "train-nocap-64k.jsonl"
REVISIONS = {"babilong": "ee0d588794c7ac098062ee0d247c733d62e94fe2",
             "longbench-v2": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
             "doc-nli": "9b32389b6e21e232b8114f259f762f91fb486aff",
             "qasper": "qasper-test-and-evaluator-v0.3.tgz (qasper-dataset.s3.us-west-2.amazonaws.com)"}
BABILONG_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k")
JEVAL_LONG = {"longbench-retrieval-en", "longbench-retrieval-en-e", "longbench-retrieval-zh", "longbench-trec",
              "longbench-trec-e", "longbench-lsht", "helmet"}
DOCNLI_MIN_CHARS = 5000  # DocNLI premises stop near 8,000 characters; at 6,000 only 120 are entailed
DOCNLI_PER_LABEL = 600
MAX_CONTEXT = 262144  # Qwen3.5-0.8B max_position_embeddings
# A test row is excluded when at least this share of its kept shingles is non-template text of a
# training row (e.g. DocNLI premises that are RAGTruth's CNN/DailyMail articles); below it, the
# shared shingles are scattered phrases and the row is kept.
EXCLUDE_SHARE = .01


def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def row(id_, source, group, state, question, target, origin, **meta) -> dict:
    return {"id": id_, "source": source, "group": group, "state": state, "question": question, "target": target,
            "ordered": False, "label_origin": origin, "target_kind": "single", **meta}


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


def babilong() -> list[dict]:
    base = RAW / f"babilong@{REVISIONS['babilong']}/data"
    out = []
    for task in range(1, 6):
        data = {length: json.loads((base / f"qa{task}/{length}.json").read_text()) for length in BABILONG_LENGTHS}
        answers = sorted({r["target"] for items in data.values() for r in items})
        for length, items in data.items():
            for i, r in enumerate(items):
                question = {"type": "choice", "instructions": r["question"], "criteria": {a: None for a in answers}}
                out.append(row(f"babilong:qa{task}:{length}:{i}", f"babilong-{length}", f"babilong:qa{task}:{i}",
                               r["input"], question, [float(a == r["target"]) for a in answers], "programmatic",
                               task=f"qa{task}", length=length))
    return out


def qasper() -> list[dict]:
    papers = json.loads((RAW / "qasper-test-v0.3/qasper-test-v0.3.json").read_text())
    out = []
    for paper_id, paper in sorted(papers.items()):
        parts = [f"Title: {paper['title']}", f"Abstract: {paper['abstract']}"]
        for section in paper["full_text"]:
            parts.append("\n\n".join([f"## {section['section_name']}" if section["section_name"] else ""]
                                     + [p for p in section["paragraphs"] or [] if p]).strip())
        captions = [f"- {f['caption']}" for f in paper.get("figures_and_tables") or [] if f.get("caption")]
        if captions:
            parts.append("Figures and tables:\n" + "\n".join(captions))
        state = "\n\n".join(p for p in parts if p)
        for qa in paper["qas"]:
            answers = [a["answer"] for a in qa["answers"]]
            if not answers:
                continue
            unanswerable = {a["unanswerable"] for a in answers}
            if len(unanswerable) == 1:
                none = unanswerable.pop()
                out.append(row(f"qasper:{qa['question_id']}:answerable", "qasper-answerable", f"qasper:{paper_id}", state,
                               noul(f"Does the paper above contain the information needed to answer this question: "
                                    f"{qa['question']}", "Yes, the paper answers it",
                                    "No, the paper does not answer it"),
                               [1., 0.] if none else [0., 1.], "human", paper=paper_id))
            yes_no = {a["yes_no"] for a in answers}
            if len(yes_no) == 1 and None not in yes_no and not any(a["unanswerable"] for a in answers):
                out.append(row(f"qasper:{qa['question_id']}:yesno", "qasper-yesno", f"qasper:{paper_id}", state,
                               noul(f"Based on the paper above, answer with yes or no: {qa['question']}", "Yes", "No"),
                               [0., 1.] if yes_no.pop() else [1., 0.], "human", paper=paper_id))
    # 94% of the agreed answerability questions are answerable: keep as many as there are unanswerable ones.
    answerable = [r for r in out if r["source"] == "qasper-answerable"]
    unanswerable = [r for r in answerable if r["target"][0]]
    kept = sorted((r for r in answerable if r["target"][1]), key=lambda r: digest("qasper-balance-v1", r["id"]))
    drop = {r["id"] for r in kept[len(unanswerable):]}
    return [r for r in out if r["id"] not in drop]


def longbench_v2() -> list[dict]:
    items = json.loads((RAW / f"longbench-v2@{REVISIONS['longbench-v2']}/data.json").read_text())
    out = []
    for r in items:
        letters = "ABCD"
        question = {"type": "choice", "instructions": r["question"],
                    "criteria": {f"opt_{i + 1}": r[f"choice_{c}"] for i, c in enumerate(letters)}}
        out.append(row(f"longbench-v2:{r['_id']}", f"longbench-v2-{r['length']}", f"longbench-v2:{r['_id']}",
                       r["context"], question, [float(c == r["answer"]) for c in letters], "human",
                       domain=r["domain"], sub_domain=r["sub_domain"], difficulty=r["difficulty"], length=r["length"]))
    return out


def docnli_long() -> list[dict]:
    import pyarrow.parquet as pq

    base = RAW / f"doc-nli@{REVISIONS['doc-nli']}/data"
    rows = [r for f in sorted(base.glob("test-*.parquet")) for r in pq.read_table(f).to_pylist()
            if len(r["premise"]) >= DOCNLI_MIN_CHARS]
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: digest("docnli-long-v1", r["premise"], r["hypothesis"])):
        by[r["label"]].append(r)
    out = []
    for label in ("entailment", "not_entailment"):
        for r in by[label][:DOCNLI_PER_LABEL]:
            key = digest("docnli", r["premise"], r["hypothesis"])[:16]
            out.append(row(f"docnli-long:{key}", "docnli-long", f"docnli:{digest(r['premise'])[:16]}", r["premise"],
                           noul(f"Is the following statement supported by the document above?\n\nStatement: {r['hypothesis']}",
                                "Supported", "Not supported"),
                           [0., 1.] if label == "entailment" else [1., 0.], "programmatic"))
    return out


def jeval_long() -> list[dict]:
    from qwen_decision.v3 import proxy_eval

    out = []
    for r in proxy_eval.load_rows(None, sources=JEVAL_LONG, split="dev"):
        question, levels = proxy_eval.as_kit(next(iter(json.loads(r["questions_json"]).values())))
        label = json.loads(r["label_json"])
        if question["type"] == "noul":
            target = [0., 1.] if label else [1., 0.]
        else:
            keys = levels or list(question["criteria"])
            want = str(int(label)) if levels else label
            target = [float(k == want) for k in keys]
        if sum(target) != 1:
            raise ValueError(f"{r['id']}: label not among the options")
        out.append(row(r["id"], r["source"], r["id"], r["state"], question, target, "human"))
    return out


SETS = {"babilong": babilong, "qasper": qasper, "longbench-v2": longbench_v2, "docnli-long": docnli_long,
        "jeval-long": jeval_long}


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path.name}: duplicate ids")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    manifest = {"revisions": REVISIONS, "sets": {}}
    for name, make in SETS.items():
        rows = make()
        manifest["sets"][name] = {"rows": len(rows), "sha256": write(TEST / f"{name}.jsonl", rows),
                                  "sources": dict(Counter(r["source"] for r in rows)),
                                  "positive_or_first_gold": dict(Counter(f"{r['source']}:{r['target'].index(1.)}"
                                                                         for r in rows))}
        print("BUILT", name, len(rows), flush=True)
    (TEST / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


def pool() -> None:
    """long_training.prepare's per-source sampler over v3-gold6, with no prompt-length filter."""
    from qwen_decision.v3 import architecture_search as search, long_training as lt, readout_experiment as records, train

    gold6 =ROOT / "data/mix/v3-gold6.jsonl"
    sources = sorted({r["source"] for r in records.rows(lt.prior.OUT / "large.jsonl")})
    by = defaultdict(list)
    for r in lt.prior.pilot.read_rows(gold6):
        if r["source"] in sources and lt.prior.old.allowed_origin(r, {"human", "programmatic"}):
            by[r["source"]].append(r)
    pools = {s: sorted(by[s], key=lambda r: search.digest(["long-nocap-v1", r["id"]]))[:1024] for s in sources}
    quota = dict.fromkeys(sources, 0)
    while sum(quota.values()) < 65536:
        changed = False
        for s in sources:
            if quota[s] < len(pools[s]) and sum(quota.values()) < 65536:
                quota[s] += 1
                changed = True
        if not changed:
            raise ValueError("insufficient eligible examples")
    rows = sorted([r for s in sources for r in pools[s][:quota[s]]], key=lambda r: r["id"])
    selection = records.rows(lt.OUT / "selection.jsonl")
    lt.prior.pilot.check_disjoint({"train": rows, "selection": selection})
    sha = write(POOL, rows)
    (OUT / "train-nocap-64k.manifest.json").write_text(json.dumps({
        "rows": len(rows), "sha256": sha, "gold6_sha256": train.sha256_file(gold6),
        "sampler": "per source the first 1,024 rows by sha256(['long-nocap-v1', id]), then round-robin quotas to 65,536; "
                   "no prompt-length filter (long_training used <= 1,536 tokens)",
        "disjoint_from": str(lt.OUT / "selection.jsonl"), "sources": dict(Counter(r["source"] for r in rows))},
        indent=1) + "\n")
    print("POOL", len(rows), len(sources), flush=True)


LONG25 = OUT / "train-long25-32k.jsonl"
LONG_TOKENS = 1536


def pool_long() -> None:
    """32,768 v3-gold6 rows, a quarter of them longer than 1,536 prompt tokens, none truncated.

    Long rows: every eligible gold6 row (all 104 sources) over 4,000 characters is tokenized; those over 1,536 tokens are
    taken round-robin over sources (sha256 order within a source) up to 8,192. Short rows: the uncapped
    pool's rows of at most 1,536 tokens, round-robin over sources up to 24,576. Both stay disjoint from the
    exposed selection set.
    """
    from qwen_decision.v3 import architecture_search as search, long_training as lt, readout_experiment as records, train
    from qwen_decision.v3 import format as fmt

    renderer, _ = search.renderer_for(search.BASE)
    tokenizer, codes = renderer.tokenizer, renderer.codes

    def tokens(r):
        text = fmt.render(tokenizer, r["state"], r["question"], codes[:len(r["target"])], "d3")
        return len(tokenizer.encode(text, add_special_tokens=False))

    def round_robin(by: dict, want: int) -> list[dict]:
        taken, out = dict.fromkeys(by, 0), []
        while len(out) < want:
            moved = False
            for s in sorted(by):
                if taken[s] < len(by[s]) and len(out) < want:
                    out.append(by[s][taken[s]])
                    taken[s] += 1
                    moved = True
            if not moved:
                raise ValueError(f"only {len(out)} of {want} rows available")
        return out

    gold6 = ROOT / "data/mix/v3-gold6.jsonl"
    # Every gold6 source, not the 100 of the earlier pools: their sampler (<= 1,536 tokens) had dropped the
    # four sources with no short rows (QuALITY, MuSiQue, kv-lookup, API-Bank).
    long_by = defaultdict(list)
    for r in lt.prior.pilot.read_rows(gold6):
        if (lt.prior.old.allowed_origin(r, {"human", "programmatic"})
                and len(json.dumps(r["state"], ensure_ascii=False)) > 4000 and tokens(r) > LONG_TOKENS):
            long_by[r["source"]].append(r)
    for s in long_by:
        long_by[s].sort(key=lambda r: search.digest(["long25-v1", r["id"]]))
    short_by = defaultdict(list)
    for r in sorted(records.rows(POOL), key=lambda r: search.digest(["long25-short-v1", r["id"]])):
        if tokens(r) <= LONG_TOKENS:
            short_by[r["source"]].append(r)
    long_rows, short_rows = round_robin(long_by, 8192), round_robin(short_by, 24576)
    rows = sorted(long_rows + short_rows, key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("a row is both long and short")
    lt.prior.pilot.check_disjoint({"train": rows, "selection": records.rows(lt.OUT / "selection.jsonl")})
    sha = write(LONG25, rows)
    (OUT / "train-long25-32k.manifest.json").write_text(json.dumps({
        "rows": len(rows), "sha256": sha, "gold6_sha256": train.sha256_file(gold6), "long_tokens_over": LONG_TOKENS,
        "long_rows": len(long_rows), "short_rows": len(short_rows),
        "long_sources": dict(Counter(r["source"] for r in long_rows)),
        "long_available": {s: len(v) for s, v in sorted(long_by.items())},
        "short_sources": len({r["source"] for r in short_rows}), "disjoint_from": str(lt.OUT / "selection.jsonl")},
        indent=1) + "\n")
    print("POOL_LONG25", len(rows), dict(Counter(r["source"] for r in long_rows)), flush=True)


def audit() -> None:
    import numpy as np

    from qwen_decision.v3 import architecture_search as search, decontam, fresh_holdback, readout_experiment as records
    from qwen_decision.v3 import format as fmt

    renderer, _ = search.renderer_for(search.BASE)
    tokenizer, codes = renderer.tokenizer, renderer.codes

    def tokens(r):
        text = fmt.render(tokenizer, r["state"], r["question"], codes[:len(r["target"])], "d3")
        return len(tokenizer.encode(text, add_special_tokens=False))

    def stats(rows):
        n = sorted(tokens(r) for r in rows)
        pick = lambda q: n[min(len(n) - 1, int(q * len(n)))]
        return {"rows": len(n), "p50": pick(.5), "p90": pick(.9), "max": n[-1],
                "over_1536": sum(x > 1536 for x in n), "over_8192": sum(x > 8192 for x in n),
                "over_32768": sum(x > 32768 for x in n), "over_context": sum(x > MAX_CONTEXT for x in n)}

    result = {"tokens": {}, "overlap": {}}
    tests = {name: records.rows(TEST / f"{name}.jsonl") for name in SETS}
    for name, rows in tests.items():
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        result["tokens"][name] = {s: stats(v) for s, v in sorted(by.items())}
        print("TOKENS", name, json.dumps(result["tokens"][name]), flush=True)
    if POOL.exists():
        result["tokens"]["train-nocap-64k"] = stats(records.rows(POOL))
        print("TOKENS pool", json.dumps(result["tokens"]["train-nocap-64k"]), flush=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    overlap()


def overlap() -> None:
    """Exact normalized states, then sampled 13-word shingles of the non-synthetic sets, against every
    training mix row (train and dev files) and the uncapped pool.

    As in decontam, a shingle held by more than MAX_REPEATS training rows is a template (licence text,
    boilerplate) and does not count; a test row is flagged with MIN_SHARED or more other shared shingles.
    Flagged ids go to audit.json so that scores can be reported with and without them.
    """
    import numpy as np

    from qwen_decision.v3 import decontam, fresh_holdback, readout_experiment as records

    result = json.loads((OUT / "audit.json").read_text()) if (OUT / "audit.json").exists() else {}
    tests = {name: records.rows(TEST / f"{name}.jsonl") for name in SETS}
    corpora = sorted((ROOT / "data/mix").glob("v3-gold*.jsonl")) + ([POOL] if POOL.exists() else [])
    states = {name: {fresh_holdback.state_key(r["state"]): r["id"] for r in rows} for name, rows in tests.items()}
    texts = {name: rows for name, rows in tests.items() if name != "babilong"}
    owners = [(name, r) for name, rows in texts.items() for r in rows]
    own = [np.unique(np.concatenate([decontam.shingles(decontam.words(t)) for t in decontam.strings(r["state"])]
                                    or [np.empty(0, dtype=np.uint64)])) for _, r in owners]
    keys = np.unique(np.concatenate(own))
    exact = defaultdict(set)
    rows_with = Counter()            # shingle -> training rows holding it
    sources_with = defaultdict(set)  # shingle -> training sources (kept while it is not a template)
    for corpus in corpora:
        with corpus.open() as f:
            for line in f:
                r = json.loads(line)
                key = fresh_holdback.state_key(r["state"])
                for name, ids in states.items():
                    if key in ids:
                        exact[name].add(ids[key])
                found = set()
                for text in decontam.strings(r["state"]):
                    h = decontam.shingles(decontam.words(text))
                    if len(h):
                        at = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
                        found.update(h[keys[at] == h].tolist())
                for h in found:
                    rows_with[h] += 1
                    if rows_with[h] <= decontam.MAX_REPEATS:
                        sources_with[h].add(r["source"])
        print("SCANNED", corpus.name, flush=True)
    result["overlap"] = {}
    for name, rows in tests.items():
        flagged = {}
        for (owner, r), h in zip(owners, own):
            if owner != name:
                continue
            hits = [x for x in h.tolist() if 0 < rows_with.get(x, 0) <= decontam.MAX_REPEATS]
            if len(hits) >= decontam.MIN_SHARED:
                flagged[r["id"]] = {"shared": len(hits), "kept_shingles": len(h), "share": len(hits) / len(h),
                                    "training_sources": sorted({s for x in hits for s in sources_with[x]})}
        excluded = sorted(exact[name] | {i for i, v in flagged.items() if v["share"] >= EXCLUDE_SHARE})
        result["overlap"][name] = {"rows": len(rows), "exact_state_in_training_mixes": sorted(exact[name]),
                                   "shingle_check": name in texts, "excluded": excluded, "flagged": flagged,
                                   "flagged_sources": dict(Counter(s for v in flagged.values()
                                                                   for s in v["training_sources"]))}
        print("OVERLAP", name, len(rows), "exact", len(exact[name]), "flagged", len(flagged),
              json.dumps(result["overlap"][name]["flagged_sources"]), flush=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["build", "pool", "pool-long", "audit", "overlap"])
    stage = parser.parse_args().stage
    {"build": build, "pool": pool, "pool-long": pool_long, "audit": audit, "overlap": overlap}[stage]()


if __name__ == "__main__":
    main()
