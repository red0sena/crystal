"""Synthetic long-context decision tasks (2k-15k tokens): lookup, paragraph retrieval, buried QA, in-context labels.

Variant ② trained on mixed data scored far below the untuned model on JEVal's long-context sources
(HELMET JSON lookup 0.08, LongBench retrieval 0.27/0.48), while d3-lite scores ~1.0: our training
data is almost all under 1k tokens. These tasks teach the competence with our own material; nothing
here is copied from JEVal or the index.

- kv-lookup: a JSON object of random identifiers; pick the value of one key among 10-255 candidates
  (all candidates are values present in the object).
- paragraph-find: 15-60 SQuAD 2.0 train contexts labelled "Paragraph N"; pick the paragraph that
  contains a given sentence (lightly edited: a word dropped, case changed).
- buried-qa: a SQuAD 2.0 train question whose context is hidden among 10-40 other contexts; pick the
  answer among the gold and answers of other questions on the same pages.
- icl-labels: 60-300 labelled CLINC/BANKING77 train utterances (from the filtered phase-2 sources) with
  intent names replaced by arbitrary codenames; label a new utterance, so the examples must be read.

Contamination: every SQuAD paragraph is checked alone with the strict filter before use and dropped
if flagged; the intent utterances come from sources that already passed it. Composed examples then
go through the filter in documents mode (their only other text is our own framing).
Licences: SQuAD 2.0 CC-BY-SA-4.0; CLINC150 CC-BY-3.0; BANKING77 CC-BY-4.0.
"""
from __future__ import annotations

import ast
import glob
import json
import random
import string

import pyarrow.parquet as pq

from qwen_decision.v3.decontam import Filter
from qwen_decision.v3.sources import common

NAME = "long-context"
SQUAD = "rajpurkar/squad_v2"
SQUAD_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"
PER_TASK = 4000
LOOKUP = ["Which value is stored under the key {key} in the JSON object?",
          "Look up the key {key} in the data and choose its value.",
          "In the object above, what is the value of {key}?"]
FIND = ["Which paragraph contains the following sentence?\n\n{sentence}",
        "The sentence below is taken from one of the numbered paragraphs. Which one?\n\n{sentence}",
        "Find the paragraph this passage comes from:\n\n{sentence}"]
QA = ["Answer the question using the documents above.\n\nQuestion: {question}",
      "One of the documents answers this question. What is the answer?\n\n{question}",
      "{question}"]
ICL = ["Using the labelled examples above, which label fits this message?\n\nMessage: {text}",
       "Label the new message the same way the examples are labelled.\n\nMessage: {text}",
       "Following the examples, assign a label to: {text}"]
WORDS = ["amber", "basil", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lumen", "maple", "nectar", "onyx", "pepper", "quartz", "raven", "sable", "tundra", "umber", "violet",
         "willow", "xenon", "yarrow", "zephyr", "acorn", "birch", "cobalt", "dune"]


def ident(rng: random.Random) -> str:
    style = rng.random()
    if style < .4:
        return "-".join("".join(rng.choices("0123456789abcdef", k=n)) for n in (8, 4, 4, 4, 12))
    if style < .7:
        return f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{rng.randint(10, 9999)}"
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(6, 12)))


def kv_lookup(rng: random.Random, n: int):
    for i in range(n):
        size = rng.choice([25, 50, 100, 200])  # up to ~12k tokens with UUID keys and values
        data = {}
        while len(data) < size:
            data[ident(rng)] = ident(rng)
        key = rng.choice(list(data))
        k = min(len(data), rng.choice([10, 30, 50, 100, 255]))
        others = rng.sample([v for kk, v in data.items() if kk != key], k - 1)
        options = others + [data[key]]
        rng.shuffle(options)
        state = json.dumps(data, indent=rng.choice([None, 2]))
        question, target = common.choice(state, common.instruction(rng, LOOKUP, key=key), options,
                                         gold=options.index(data[key]))
        yield common.example(NAME, f"kv:{i}", "long-lookup", state, question, target, "generated")


def squad_rows():
    path, revision = common.fetch(SQUAD, ["squad_v2/train-*.parquet"], revision=SQUAD_REVISION)
    rows = pq.read_table(glob.glob(str(path / "squad_v2/train-*.parquet"))[0]).to_pylist()
    for r in rows:
        r["answers"] = r["answers"] if isinstance(r["answers"], dict) else ast.literal_eval(r["answers"])
    return rows


def clean_contexts(rows, flt: Filter):
    """Distinct SQuAD contexts that pass the strict filter on their own, with their answerable questions."""
    contexts = {}
    for r in rows:
        contexts.setdefault(r["context"], {"title": r["title"], "qa": []})
        if r["answers"]["text"]:
            contexts[r["context"]]["qa"].append((r["question"], r["answers"]["text"][0]))
    clean = {c: v for c, v in contexts.items() if not flt.flagged([c])}
    return clean, len(contexts)


def paragraph_find(rng: random.Random, pool: list[str], n: int):
    for i in range(n):
        chosen = rng.sample(pool, rng.choice([15, 25, 40, 60]))
        gold = rng.randrange(len(chosen))
        sentences = [s for s in chosen[gold].split(". ") if len(s.split()) >= 8]
        if not sentences:
            continue
        words = rng.choice(sentences).split()
        if rng.random() < .5:
            del words[rng.randrange(len(words))]
        sentence = " ".join(words)
        state = "\n\n".join(f"Paragraph {j + 1}:\n{p}" for j, p in enumerate(chosen))
        options = [f"Paragraph {j + 1}" for j in range(len(chosen))]
        question, target = common.choice(state, common.instruction(rng, FIND, sentence=sentence), options, gold=gold)
        yield common.example(NAME, f"find:{i}", "long-retrieval", state, question, target, "train")


def buried_qa(rng: random.Random, clean: dict, n: int):
    with_qa = [c for c, v in clean.items() if v["qa"]]
    by_title = {}
    for c, v in clean.items():
        by_title.setdefault(v["title"], []).append(c)
    made = 0
    while made < n:
        context = rng.choice(with_qa)
        q, answer = rng.choice(clean[context]["qa"])
        distractors = rng.sample([c for c in clean if c != context], rng.choice([10, 20, 40]))
        docs = distractors + [context]
        rng.shuffle(docs)
        nearby = {a for c in by_title[clean[context]["title"]] for _, a in clean[c]["qa"] if a != answer}
        nearby |= {a for c in docs if c != context for _, a in clean[c]["qa"][:2]}
        k = min(len(nearby) + 1, rng.choice([4, 8, 16]))
        if k < 3:
            continue
        options = rng.sample(sorted(nearby), k - 1) + [answer]
        rng.shuffle(options)
        state = "\n\n".join(f"Document {j + 1} ({clean[d]['title'].replace('_', ' ')}):\n{d}" for j, d in enumerate(docs))
        question, target = common.choice(state, common.instruction(rng, QA, question=q), options, gold=options.index(answer))
        yield common.example(NAME, f"qa:{made}", "long-qa", state, question, target, "train")
        made += 1


def icl_labels(rng: random.Random, n: int):
    pools = []
    for source in ("clinc-train", "banking77-train"):
        by_label = {}
        for r in common.load([source]):
            state = r["state"]
            text = next(v for v in state.values() if isinstance(v, str)) if isinstance(state, dict) else state
            label = list(r["question"]["criteria"])[r["target"].index(1.)]
            if label != "out of scope":
                by_label.setdefault(label, []).append(text)
        pools.append(by_label)
    for i in range(n):
        by_label = rng.choice(pools)
        labels = rng.sample(sorted(by_label), rng.choice([8, 15, 30, 50]))
        names = rng.sample([f"{a}-{b}" for a in WORDS for b in WORDS if a != b], len(labels))
        rename = dict(zip(labels, names))
        shots = [(t, rename[l]) for l in labels for t in rng.sample(by_label[l], min(len(by_label[l]), rng.randint(3, 6)))]
        gold_label = rng.choice(labels)
        used = {t for t, _ in shots}
        query = rng.choice([t for t in by_label[gold_label] if t not in used] or by_label[gold_label])
        if query in used:
            continue
        rng.shuffle(shots)
        state = "Labelled examples:\n" + "\n".join(f"- {t} => {name}" for t, name in shots)
        options = [rename[l] for l in labels]
        question, target = common.choice(state, common.instruction(rng, ICL, text=query), options,
                                         gold=options.index(rename[gold_label]))
        yield common.example(NAME, f"icl:{i}", "in-context", state, question, target, "train")


def build(meta: dict):
    rng = random.Random(20261016)
    flt = Filter(common.DECONTAM)
    clean, total = clean_contexts(squad_rows(), flt)
    meta["squad_contexts"] = {"distinct": total, "clean": len(clean)}
    pool = sorted(clean)
    yield from kv_lookup(rng, PER_TASK)
    yield from paragraph_find(rng, pool, PER_TASK)
    yield from buried_qa(rng, clean, PER_TASK)
    yield from icl_labels(rng, PER_TASK)


if __name__ == "__main__":
    meta = {"upstream": f"{SQUAD} (train) + clinc-train + banking77-train + generated", "revision": SQUAD_REVISION,
            "licence": "SQuAD 2.0 CC-BY-SA-4.0; CLINC150 CC-BY-3.0; BANKING77 CC-BY-4.0; generated parts ours",
            "filter_note": "documents mode: every SQuAD context passed the strict filter alone before composition"}
    examples = list(build(meta))
    common.write(NAME, examples, meta, documents=True)
