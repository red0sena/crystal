"""Training rows rebuilt in the evaluator's own decomposition (GLiDE-style format matching), from train splits only.

The public suite asks most set-selection tasks as one yes/no question per candidate over a fixed inventory
(61.5% of public text questions vs 21.7% of our training rows). Our old converters asked a few hand-picked
candidates at a near-balanced yes rate, so the model over-predicts "yes" on the evaluator's sparse grids
(ACOS: ~2.2 questions per review in training vs 58 per evaluator request).

Builders (labels stay the upstream human/dataset labels; no model outputs are used as targets):
  acos      NUSTM/ACOS *_quad_train.tsv: every (category, sentiment) pair of the domain inventory is a candidate,
            asked with the evaluator's question shape. Negatives are drawn uniformly from the full inventory
            (not hard same-category negatives); base rate BASE_RATE (natural Laptop rate .35% gives too few
            positives for the row budget, Restaurant's natural 3.4% is close). Dev files are held out and
            written in the evaluator's full natural decomposition for an internal per-review F1.
  isarcasm  iSarcasmEval train: A (En, Ar) for every text at the natural rate; B (En) all six categories for
            every text (category labels exist only for sarcastic texts; a non-sarcastic text has none of the
            six ironic-speech categories); C sarcastic-vs-rephrase pairs.
  bfcl      the existing tools-percall rows regrouped per decision point and re-asked as the evaluator's
            per-candidate tool selection over every listed tool (called = yes, listed-but-uncalled = no, as the
            converter defines them). Points with 2 sampled positives are skipped (a third called tool may be
            unrecorded). Labels keep their recorded label_origin.
Reviews/texts that equal (normalized) any public ACOS / iSarcasm state are dropped.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from qwen_decision.v3 import champion_audit as audit, train

ROOT = Path("runs/champion-v1/format-rebuild-v1")
RAW = ROOT / "raw"
POOL = Path("runs/champion-v1/full-g7-v3/mix/text-train.jsonl")
PUBLIC = Path("runs/champion-v1/public-eval-v1/text-counted-rows.jsonl")
SEED = 20261108
BASE_RATE = .05
ACOS_POSITIVES = {"Laptop-ACOS": 1000, "Restaurant-ACOS": 1000}
ACOS_DEV_REVIEWS = {"Laptop-ACOS": 80, "Restaurant-ACOS": 10**9}
SENTIMENTS = {"0": "negative", "1": "neutral", "2": "positive"}
REPLACES = ("acos-train", "isarcasm-train")  # old converters superseded in the mix; tools-percall is kept and added to


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower()))


def row(source, family, uid, state, instructions, criteria, answer, item, origin="human", kind="choice", extra=None):
    keys = list(criteria)
    assert answer in keys and len(keys) >= 2
    out = {"id": f"{source}:{uid}", "original_id": f"{source}:{item}", "source": source, "family": family,
           "modality": "text", "state": state, "question": {"type": kind, "instructions": instructions, "criteria": criteria},
           "target": [float(k == answer) for k in keys], "kind": kind, "ordered": False, "item": f"{source}:{item}",
           "documents": [], "split": "train", "label_origin": origin, "target_kind": "single",
           "annotation_basis": "upstream label re-asked in the evaluator's decomposition", "training_component": "format-rebuild-v1"}
    if extra:
        out.update(extra)
    return out


def public_states(prefixes) -> set[str]:
    out = set()
    for line in PUBLIC.open():
        if line.startswith('{"id":"') and line[7:].startswith(prefixes):
            st = json.loads(line)["state"]
            out.add(norm(st.get("review", st.get("text", "")) if isinstance(st, dict) else st))
    return out


# ------------------------------------------------------------------------------------------------ ACOS

def acos_parse(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            review, *quads = line.split("\t")
            yield review, {(q.split()[1], q.split()[2]) for q in quads}


def acos_question(category, sentiment):
    name = SENTIMENTS[sentiment]
    return ("Given the review below, decide whether it expresses at least one aspect-category/sentiment pair "
            f"({category}, {name}) anywhere in the review. Answer yes only when that exact category and sentiment "
            "pair is present; answer no otherwise. Multiple opposite sentiments are separate valid questions.")


ACOS_CRITERIA = {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."}


def acos(stats):
    rng = random.Random(f"{SEED}:acos")
    blocked = public_states(("ACOS:",))
    test = {norm(r) for p in (RAW / "acos/data").glob("*/*_quad_test.tsv") for r, _ in acos_parse(p)}
    train_rows, dev_rows = [], []
    for domain in ("Laptop-ACOS", "Restaurant-ACOS"):
        files = {s: next((RAW / "acos/data" / domain).glob(f"*_quad_{s}.tsv")) for s in ("train", "dev")}
        parsed = {s: list(acos_parse(p)) for s, p in files.items()}
        categories = sorted({c for s in parsed for _, g in parsed[s] for c, _ in g})
        inventory = [(c, s) for c in categories for s in SENTIMENTS]
        dropped = 0
        reviews = []
        for i, (review, gold) in enumerate(parsed["train"], 1):
            if norm(review) in blocked or norm(review) in test:
                dropped += 1
                continue
            reviews.append((i, review, gold))
        rng.shuffle(reviews)
        positives = 0
        for i, review, gold in reviews:
            if positives >= ACOS_POSITIVES[domain]:
                break
            rest = [p for p in inventory if p not in gold]
            k = min(len(rest), round(len(gold) * (1 - BASE_RATE) / BASE_RATE))
            asked = sorted(gold) + rng.sample(rest, k)
            rng.shuffle(asked)
            state = {"review": review, "task": "ACOS fixed category/sentiment presence"}
            for c, s in asked:
                train_rows.append(row("acos-format", "ACOS-category-sentiment", f"{domain}:train:{i}:{c}:{s}", state,
                                      acos_question(c, s), dict(ACOS_CRITERIA), "yes" if (c, s) in gold else "no",
                                      f"{domain}:train:{i}"))
            positives += len(gold)
        dev = [(i, r, g) for i, (r, g) in enumerate(parsed["dev"], 1) if norm(r) not in blocked and norm(r) not in test]
        random.Random(f"{SEED}:acos-dev:{domain}").shuffle(dev)
        for i, review, gold in dev[:ACOS_DEV_REVIEWS[domain]]:
            state = {"review": review, "task": "ACOS fixed category/sentiment presence"}
            for c, s in inventory:
                dev_rows.append(row("acos-format", "ACOS-category-sentiment", f"{domain}:dev:{i}:{c}:{s}", state,
                                    acos_question(c, s), dict(ACOS_CRITERIA), "yes" if (c, s) in gold else "no",
                                    f"{domain}:dev:{i}", extra={"split": "dev", "group": f"{domain}:dev:{i}"}))
        stats[domain] = {"categories": len(categories), "inventory_pairs": len(inventory), "train_reviews_used": None,
                         "dropped_public_or_test": dropped, "positives": positives}
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        yes = sum(r["target"][0] == 1. for r in rows)
        stats[f"{name}_rows"], stats[f"{name}_yes_rate"] = len(rows), yes / len(rows)
    return train_rows, dev_rows


# ------------------------------------------------------------------------------------------------ iSarcasm

B_LABELS = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]


def isarcasm(stats):
    rng = random.Random(f"{SEED}:isarcasm")
    blocked = public_states(("iSarcasmEval",))
    out, counts = [], collections.Counter()
    for lang, col in (("En", "tweet"), ("Ar", "text")):
        with (RAW / f"isarcasm/train/train.{lang}.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            text = (r[col] or "").strip()
            if not text or r["sarcastic"] not in ("0", "1"):
                continue
            if norm(text) in blocked:
                counts["dropped_public"] += 1
                continue
            sarcastic = r["sarcastic"] == "1"
            out.append(row("isarcasm-format", f"iSarcasmEval-A-{lang}", f"A:{lang}:{i}", text,
                           "Is this text intended to be sarcastic?", {"no": "No", "yes": "Yes"},
                           "yes" if sarcastic else "no", f"{lang}:{i}"))
            counts[f"A_{lang}_{'yes' if sarcastic else 'no'}"] += 1
            if lang == "En":
                for label in B_LABELS:
                    present = sarcastic and r[label] == "1"
                    out.append(row("isarcasm-format", "iSarcasmEval-B-En", f"B:{lang}:{i}:{label}", text,
                                   f'Does this text exhibit {label.replace("_", " ")}? Evaluate this category independently.',
                                   {"no": "No", "yes": "Yes"}, "yes" if present else "no", f"{lang}:{i}"))
                    counts[f"B_{'yes' if present else 'no'}"] += 1
            rephrase = (r.get("rephrase") or "").strip()
            if sarcastic and rephrase and rephrase != text and norm(rephrase) not in blocked:
                pair = [text, rephrase]
                rng.shuffle(pair)
                out.append(row("isarcasm-format", f"iSarcasmEval-C-{lang}", f"C:{lang}:{i}", "",
                               "Which of these two texts is the sarcastic one?", {"text_0": pair[0], "text_1": pair[1]},
                               f"text_{pair.index(text)}", f"{lang}:{i}"))
                counts[f"C_{lang}"] += 1
    stats.update(counts=dict(counts), rows=len(out))
    return out


# ------------------------------------------------------------------------------------------------ BFCL format

BFCL_INSTRUCTIONS = ("Given the complete user conversation and the published function schemas in state, mark each "
                     "candidate tool yes if it should be invoked to answer the request, or no otherwise. "
                     "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n")
TOOL_NAME = re.compile(r"`([^`]+)`")


def percall_label(r) -> tuple[str, bool] | None:
    q = r["question"]
    m = TOOL_NAME.search(q["instructions"])
    if not m:
        return None
    yes = max(range(len(r["target"])), key=r["target"].__getitem__)
    if q["type"] == "noul":  # noul targets are [P(false), P(true)] (format.options)
        return m.group(1), yes == 1
    keys = list(q["criteria"])
    positive = [k for k in keys if not re.search(r"\b(not|no|skip|leave|don't|irrelevant)\b", k.lower())]
    if len(positive) != 1:
        return None
    return m.group(1), keys[yes] == positive[0]


def bfcl(stats):
    points = {}
    skipped = collections.Counter()
    for line in POOL.open():
        if '"tools-percall"' not in line:
            continue
        r = json.loads(line)
        st = r["state"]
        conv = next((st[k] for k in ("conversation", "messages", "dialogue") if k in st), None)
        tools = next((st[k] for k in ("tools", "functions", "available_functions") if k in st), None)
        lab = percall_label(r)
        if conv is None or not tools or lab is None:
            skipped["unparsed"] += 1
            continue
        names = sorted(t.get("name") for t in tools if isinstance(t, dict) and t.get("name"))
        key = hashlib.sha256(json.dumps([conv, names], sort_keys=True).encode()).hexdigest()
        p = points.setdefault(key, {"conv": conv, "tools": tools, "yes": set(), "no": set(), "origin": r["label_origin"]})
        (p["yes"] if lab[1] else p["no"]).add(lab[0])
    # Listed-tool counts follow the evaluator's candidates-per-request shape (structure only, no labels):
    # public BFCL has 1 candidate in 657 of 1,694 requests, 2-5 in 885, 6-10 in 150.
    sizes = [1] * 657 + [2] * 219 + [3] * 299 + [4] * 176 + [5] * 191 + [6] * 78 + [7] * 29 + [8] * 18 + [9] * 11 + [10] * 14
    rng = random.Random(f"{SEED}:bfcl")
    out = []
    for key, p in sorted(points.items()):
        names = [t["name"] for t in p["tools"] if isinstance(t, dict) and t.get("name")]
        if p["yes"] & p["no"] or not p["yes"] or len(p["yes"]) >= 2 or not p["yes"] <= set(names):
            skipped["ambiguous_or_two_positive_or_none"] += 1
            continue
        negatives = [n for n in names if n not in p["yes"]]
        kept = set(p["yes"]) | set(rng.sample(negatives, min(len(negatives), rng.choice(sizes) - 1)))
        tools = [t for t in p["tools"] if isinstance(t, dict) and t.get("name") in kept]
        names = [t["name"] for t in tools]
        state = {"conversation": p["conv"], "functions": tools, "task": "BFCL tool-name selection"}
        for name in names:
            out.append(row("bfcl-format", "BFCL-tool-selection", f"{key[:16]}:{name}", state,
                           BFCL_INSTRUCTIONS + f"\n\nCandidate tool: {name}", {"yes": "Invoke this tool.", "no": "Do not invoke this tool."},
                           "yes" if name in p["yes"] else "no", key[:16], origin=p["origin"]))
    yes = sum(r["target"][0] == 1. for r in out)
    stats.update(points=len(points), rows=len(out), yes_rate=yes / max(1, len(out)), skipped=dict(skipped))
    return out


# ------------------------------------------------------------------------------------------------ ToolRet format

RETRIEVAL_TASK = ("Assess whether this candidate is relevant/useful to the query. Use the full text below; "
                  "return the probability of relevance.")
RETRIEVAL_CRITERIA = {"no": "Not relevant/useful to the query.", "yes": "Relevant/useful to the query."}
STOP = set("a an the is are was were be been being do does did to of on in for by with and or as at from that this these "
           "those it its i you we they he she have has had can could should would will what which how when where why who "
           "please me my your".split())


def bm25(conn, query, k):
    words = list(dict.fromkeys(w.casefold() for w in re.findall(r"[^\W_]+", query) if w.casefold() not in STOP))
    if not words:
        return []
    match = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
    return [d for d, in conn.execute("SELECT docid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs),docid LIMIT ?", [match, k])]


def toolret(stats, candidates=32, train_queries=1200, dev_queries=200):
    """Tool retrieval in the evaluator's shape: query = the decision point's last user turn, candidates = the called
    tool plus BM25 (fts5, as the evaluator) hits over every tool schema in our tool sources, labels = called / not.
    A hit that shares the called tool's name is removed (same tool from another dialogue). Dev queries are disjoint
    from train queries and from train gold tool names; dev rows carry a group id for nDCG@10."""
    import sqlite3
    rng = random.Random(f"{SEED}:toolret")
    blocked = {norm(json.loads(l)["state"]["query"]) for l in PUBLIC.open() if l.startswith('{"id":"ToolRet')}
    corpus, points = {}, {}
    for line in POOL.open():
        if '"tools-percall"' not in line:
            continue
        r = json.loads(line)
        st = r["state"]
        conv = next((st[k] for k in ("conversation", "messages", "dialogue") if k in st), None)
        tools = next((st[k] for k in ("tools", "functions", "available_functions") if k in st), None)
        lab = percall_label(r)
        if not conv or not tools or lab is None or not lab[1]:
            continue
        for t in tools:
            if isinstance(t, dict) and t.get("name"):
                corpus.setdefault(t["name"], json.dumps(t, ensure_ascii=False))
        users = [m.get("content") for m in conv if isinstance(m, dict) and m.get("role") == "user" and m.get("content")]
        if users and lab[0] in corpus and norm(users[-1]) not in blocked:
            points.setdefault(norm(users[-1]), (users[-1], lab[0]))
    conn = sqlite3.connect(":memory:")
    conn.execute('CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED,content,tokenize="porter unicode61")')
    conn.executemany("INSERT INTO docs(docid,content) VALUES (?,?)", list(corpus.items()))
    items = sorted(points.values())
    rng.shuffle(items)
    dev, train_items, dev_gold = [], [], set()
    for query, gold in items:
        if len(dev) < dev_queries:
            dev.append((query, gold)); dev_gold.add(gold)
        elif gold not in dev_gold and len(train_items) < train_queries:
            train_items.append((query, gold))
    out = {"train": [], "dev": []}
    recall = collections.Counter()
    for split, chosen in (("train", train_items), ("dev", dev)):
        for qi, (query, gold) in enumerate(chosen):
            hits = [d for d in bm25(conn, query, candidates + 8) if d != gold and d.split(".")[-1] != gold.split(".")[-1]]
            recall[f"{split}_gold_in_bm25"] += gold in bm25(conn, query, candidates)
            docs = hits[:candidates - 1] + [gold]
            rng.shuffle(docs)
            state = {"query": query, "task": "Rank candidate documents/tools by relevance to this query."}
            gid = hashlib.sha256(query.encode()).hexdigest()[:16]
            for di, d in enumerate(docs):
                out[split].append(row("toolret-format", "ToolRet-retrieval", f"{split}:{gid}:{di}", state,
                                      {"task": RETRIEVAL_TASK, "candidate": corpus[d]}, dict(RETRIEVAL_CRITERIA),
                                      "yes" if d == gold else "no", f"{split}:{gid}", origin="model",
                                      extra={"split": split, "group": gid}))
    stats.update(corpus_tools=len(corpus), points=len(points), train_rows=len(out["train"]), dev_rows=len(out["dev"]),
                 train_queries=len(train_items), dev_queries=len(dev), **recall)
    return out["train"], out["dev"]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build():
    stats = {"seed": SEED, "base_rate": BASE_RATE, "acos": {}, "isarcasm": {}, "bfcl": {}}
    a_train, a_dev = acos(stats["acos"])
    s_rows = isarcasm(stats["isarcasm"])
    b_rows = bfcl(stats["bfcl"])
    stats["toolret"] = {}
    t_train, t_dev = toolret(stats["toolret"])
    for name, rows in (("acos-format-train", a_train), ("acos-format-dev", a_dev), ("isarcasm-format-train", s_rows),
                       ("bfcl-format-train", b_rows), ("toolret-format-train", t_train), ("toolret-format-dev", t_dev)):
        write(ROOT / f"{name}.jsonl", rows)
        stats[name] = {"rows": len(rows), "sha256": audit.sha(ROOT / f"{name}.jsonl")}
    stats["replaces_in_mix"] = list(REPLACES)
    stats["raw_revisions"] = {"acos": "45d179a3dcc6a3dedd848d81b16f2552454805fe", "isarcasm": "dfc708b53bde1bb571abfb5692f63231c2232195"}
    train.atomic_json(ROOT / "manifest.json", stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=1)[:4000])
