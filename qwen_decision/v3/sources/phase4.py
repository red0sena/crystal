"""Phase-4 sources (2026-10-18): verified, openly licensed data for the JEVal new-domain gaps left after g4b.
The gaps (best merge vs d3-lite) were finance yes/no (FinQA −0.43, FinanceBench −0.24), entity linking
(AIDA −0.33), and many-label classification from in-context examples (LongBench TREC −0.15). The
fresh-v1 gaps were fine meaning (WiC) and entailment (CB, QASC).

  python -m qwen_decision.v3.sources.phase4 [name ...]

- tatqa-yesno: TAT-QA (CC-BY-4.0, train+dev, the files finance.py reads). Each row is a yes/no question
  about a report table and text: is the answer to the question X?
  - X is the human answer half of the time, else a distractor that finance.distractors picks.
  - Half of the "change" questions that year_direction can check against the table instead ask whether
    the figure went up from the earlier year to the later one (see year_direction).
  - A table is one document for this source and tatqa-train together (mix.DOCUMENTS, shared key), so
    neither source can put it on the other side of the dev split.
- icl-classify: classification from labelled examples given in the prompt, the LongBench-TREC format.
  - Data: DBpedia classes (DeveloperOats/DBPedia_Classes; the card says CC0, but DBpedia's text is taken
    from Wikipedia, CC-BY-SA; 70 level-2 and 219 level-3 classes) and finance tweet topics
    (zeroshot/twitter-financial-news-topic, MIT, 20 topics).
  - Each prompt shows 1-3 examples for each of 10-60 labels, then a new text; the options are those labels.
  - In 30% of prompts the labels are replaced by distinct arbitrary names, so only the examples tell them
    apart.
  - Every text is either an example text or a query text (text_role), for both DBpedia levels, so no
    query is shown as an example in any prompt. Finance tweets are split three ways: example texts,
    query texts, and the rest, which fintweet-train asks about directly. TREC itself (the source of
    LongBench-TREC) is not used.
- wned-el: entity linking (KILT `wned` validation, 3,396 mentions, KILT code MIT, text from Wikipedia).
  - The mention is marked [START_ENT] ... [END_ENT], as in JEVal's AIDA rows.
  - Options: the gold Wikipedia title plus 9 other gold titles of the set that look most like the mention
    (character trigram overlap). AIDA (JEVal) is not used.
  - Known weakness (audit 2026-10-18): picking the title most like the mention, without reading the text,
    is right about 75% of the time, since few of the set's titles share a name.
- vitaminc-train: VitaminC (Schuster et al. 2021, tals/vitaminc, CC-BY-SA-3.0), fact verification against a
  Wikipedia sentence.
  - Only "real" revision pairs are used. The synthetic part is built from FEVER claims, and FEVER is a
    JEVal source.
  - Whole cases (case_id: a claim and its revised contrast claim, each against the evidence before and
    after the edit, so the verdict flips) are sampled until about 30,000 rows. The labels follow the data
    (about half supports).
  - 70% are 3-way choices, 30% are "does the evidence support the claim" yes/no.
- paws-train: PAWS-Wiki labeled_final (Zhang et al. 2019; Google: "may be freely used for any purpose").
  Word-scrambled sentence pairs that do or do not mean the same thing, labels by human raters. 20,000
  sampled. This is whole-sentence meaning; it does not teach the word-sense task of WiC.
- scitail-train: SciTail (Khot et al. 2018, allenai/scitail dgem_format train, Apache-2.0 per the
  allenai/scitail repository). Does a science sentence support the hypothesis? One hypothesis is one item.
- fintweet-train: finance tweets (zeroshot/twitter-financial-news-topic and -sentiment, MIT). 20 topics, or
  bearish / bullish / neutral. These are direct questions, not the in-context form above, on tweets that
  icl-classify does not use.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re

import pandas as pd
import pyarrow.parquet as pq

from qwen_decision.v3.sources import common, finance

RAW = common.RAW / "phase4"


def digest(text: str) -> str:
    return hashlib.sha1(re.sub(r"[\W_]+", " ", str(text).lower()).strip().encode()).hexdigest()[:12]


# --- TAT-QA yes/no ------------------------------------------------------------------------------------------
TATQA_YESNO = "tatqa-yesno"
INCREASE = ["Based on the report, did this figure go up from {early} to {late}?",
            "Was the figure in question higher in {late} than in {early}?"]
YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def cell_number(text: str) -> float | None:
    s = text.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    negative = s.startswith("(") and s.endswith(")")
    try:
        value = float(s.strip("()").strip())
    except ValueError:
        return None
    return -value if negative else value


def year_direction(q: dict, doc: dict) -> tuple[int, int, bool] | None:
    """(earlier year, later year, whether the figure went up) for a "change" question, checked on the table.

    The first version (gold4) asked "is the change an increase?" for any question containing "change" and took
    the sign of the answer. That was wrong twice over (audit 2026-10-18): "exchange" matched (foreign exchange
    contracts are not changes), and annotators often wrote the change as earlier minus later year (65 in
    2019, 128 in 2018, answer 128 - 65 = 63). So a question is kept only when
    - it says "change" as a word, is arithmetic, and is not an absolute/average/percentage/ratio question;
    - its derivation is one number minus another and equals the answer;
    - both numbers are in one table row, under columns whose headers name two different years (the years
      the question names, if it names any).
    The direction then comes from the two table values, not from the annotator's sign.
    """
    text = q["question"].lower()
    if q["answer_type"] != "arithmetic" or not re.search(r"\bchange\b", text) \
            or re.search(r"\b(?:absolute|average|proportion|ratio|percentage|how many)\b|%", text):
        return None
    m = re.fullmatch(r"\(?([-\d,.$()]+?)-([-\d,.$()]+?)\)?", (q.get("derivation") or "").replace(" ", ""))
    value = finance.as_number(q["answer"])
    if not m or value is None:
        return None
    a, b = cell_number(m.group(1)), cell_number(m.group(2))
    if a is None or b is None or abs((a - b) - value) > max(.01, 1e-4 * abs(value)):
        return None
    table = doc["table"]["table"]
    years = {}
    for column in range(max(len(row) for row in table)):
        found = next((YEAR.search(row[column]) for row in table[:3] if column < len(row) and YEAR.search(row[column])), None)
        if found:
            years[column] = int(found.group(0))
    pairs = set()
    for row in table:
        at = collections.defaultdict(set)
        for column, cell in enumerate(row):
            number = cell_number(cell)
            if number is not None and column in years:
                at[number].add(years[column])
        pairs |= {(ya, yb) for ya in at.get(a, ()) for yb in at.get(b, ()) if ya != yb}
    named = {int(y) for y in YEAR.findall(q["question"])}
    if len(pairs) != 1 or (named and named != set(next(iter(pairs)))):
        return None
    ya, yb = next(iter(pairs))
    return min(ya, yb), max(ya, yb), (a > b) == (ya > yb)


def tatqa_yesno(stats: dict):
    rng = random.Random(2026101811)
    counts = collections.Counter()
    for split in ("train", "dev"):
        for doc in json.loads((finance.TATQA_DIR / f"tatqa_dataset_{split}.json").read_text()):
            for q in doc["questions"]:
                gold, value = finance.show_answer(q)
                if not gold:
                    continue
                state = finance.render_context(doc, rng)
                direction = year_direction(q, doc)
                if direction and rng.random() < .5:
                    early, late, up = direction
                    state = {**state, "question": q["question"]} if isinstance(state, dict) else f"{state}\n\nQuestion:\n{q['question']}"
                    question, target = common.noul(rng.choice(INCREASE).format(early=early, late=late), up)
                    counts["direction_up" if up else "direction_down"] += 1
                else:
                    truth = rng.random() < .5
                    shown = gold
                    if not truth:
                        wrong = finance.distractors(q, gold, value, doc, rng, k=1)
                        if not wrong:
                            continue
                        shown = wrong[0]
                    question, target = common.noul(f"Is the answer to \"{q['question']}\" {shown}?", truth)
                    counts["verify_true" if truth else "verify_false"] += 1
                yield common.example(TATQA_YESNO, f"{split}:{doc['table']['uid']}:{q['uid']}", "finance-yesno", state,
                                     question, target, split)
    stats["counts"] = dict(counts)


# --- In-context classification ------------------------------------------------------------------------------
ICL = "icl-classify"
ICL_ROWS = 12_000
ICL_ASK = ["Using the labelled examples above, which label fits the new text?", "Classify the new text the way the examples are labelled.",
           "Which category does the final text belong to, following the examples?"]
FIN_TOPICS = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend", "Earnings",
              "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO", "Legal | Regulation",
              "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]


def shown(text) -> str:
    return " ".join(str(text).split())[:300]


def text_role(corpus: str, text: str) -> str:
    """A text's one role, fixed by its hash: "example" (shown with its label in prompts), "query" (asked
    about), or, for finance tweets, "direct" (left to fintweet-train). gold4 split each DBpedia level's
    texts on its own, so a level-3 query could be a labelled level-2 example in another prompt, and
    fintweet-train asked directly about the ICL query tweets (98 of 311 dev queries seen in train)."""
    bucket = int(digest(shown(text)), 16) % 10
    if corpus == "fintopic":
        return "example" if bucket < 2 else "query" if bucket < 4 else "direct"
    return "example" if bucket < 5 else "query"


def icl_pools():
    pools = {}
    db = pd.read_csv(RAW / "DeveloperOats__DBPedia_Classes/DBPEDIA_train.csv")
    for level in ("l2", "l3"):
        by = collections.defaultdict(set)
        for text, label in zip(db["text"], db[level]):
            by[label].add(shown(text))
        pools[f"dbpedia-{level}"] = ("dbpedia", by)
    fin = pd.read_csv(RAW / "zeroshot__twitter-financial-news-topic/topic_train.csv")
    by = collections.defaultdict(set)
    for text, label in zip(fin["text"], fin["label"]):
        by[FIN_TOPICS[int(label)]].add(shown(text))
    pools["fintopic"] = ("fintopic", by)
    split = {}
    for name, (corpus, by) in pools.items():
        demos, queries = {}, {}
        for label, texts in by.items():
            examples = sorted(t for t in texts if text_role(corpus, t) == "example")
            if len(examples) >= 3:  # a label needs examples to be offered at all
                demos[label] = examples
                queries[label] = sorted(t for t in texts if text_role(corpus, t) == "query")
        split[name] = (demos, queries)
    return split


def icl(stats: dict):
    rng = random.Random(2026101812)
    split = icl_pools()
    used = set()
    counts = collections.Counter()
    for n in range(ICL_ROWS):
        name = rng.choices(["dbpedia-l3", "dbpedia-l2", "fintopic"], weights=[.5, .2, .3])[0]
        demos, queries = split[name]
        labels = sorted(demos)
        k = min(len(labels), rng.randint(10, 60))
        chosen = rng.sample(labels, k)
        golds = [lab for lab in chosen if queries[lab]]
        if not golds:
            continue
        gold = rng.choice(golds)
        query = next((t for t in rng.sample(queries[gold], min(20, len(queries[gold]))) if (name, t) not in used), None)
        if query is None:
            continue
        used.add((name, query))
        renamed = rng.random() < .3
        # distinct names: gold4 drew them independently and dropped prompts with a clash, which left 22% renamed
        names = dict(zip(chosen, [f"class {c}" for c in rng.sample(range(100, 1000), k)])) if renamed else {lab: lab for lab in chosen}
        shots = [(t, lab) for lab in chosen for t in rng.sample(demos[lab], min(len(demos[lab]), rng.randint(1, 3)))]
        rng.shuffle(shots)
        state = "\n".join(f"Text: {t}\nLabel: {names[lab]}" for t, lab in shots) + f"\n\nNew text: {query}"
        options = [names[lab] for lab in chosen]
        question, target = common.choice(state, rng.choice(ICL_ASK), options, chosen.index(gold))
        counts[name] += 1
        counts["renamed"] += renamed
        yield common.example(ICL, f"{name}:{digest(query)}", "in-context-classification", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- WNED entity linking ------------------------------------------------------------------------------------
WNED = "wned-el"
EL_ASK = ["The text between [START_ENT] and [END_ENT] mentions an entity. Which Wikipedia entity is it?",
          "Which entity does the marked mention refer to?"]


def trigrams(text: str) -> set[str]:
    t = f"  {text.lower()}  "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def wned(stats: dict):
    rng = random.Random(2026101813)
    rows = pq.read_table(RAW / "facebook__kilt_tasks/wned/validation-00000-of-00001.parquet").to_pylist()
    gold_of = []
    for r in rows:
        outputs = r["output"]
        title = next((p["title"] for o in outputs for p in (o.get("provenance") or []) if p.get("title")), None) or (outputs[0]["answer"] if outputs else None)
        gold_of.append(title)
    titles = sorted({t for t in gold_of if t})
    grams = {t: trigrams(t) for t in titles}
    for r, gold in zip(rows, gold_of):
        if not gold or "[START_ENT]" not in r["input"]:
            continue
        mention = r["input"].split("[START_ENT]", 1)[1].split("[END_ENT]", 1)[0].strip()
        m = trigrams(mention)
        ranked = sorted((t for t in titles if t != gold), key=lambda t: -len(m & grams[t]) / max(1, len(m | grams[t])))
        options = ranked[:9] + [gold]
        rng.shuffle(options)
        question, target = common.choice(r["input"], rng.choice(EL_ASK), options, options.index(gold))
        yield common.example(WNED, r["id"].split(":")[0], "entity-linking", r["input"], question, target, "validation")


# --- VitaminC -----------------------------------------------------------------------------------------------
VITAMINC = "vitaminc-train"
VITAMINC_ROWS = 30_000
VC_KEYS = {"SUPPORTS": "the evidence supports the claim", "REFUTES": "the evidence contradicts the claim",
           "NOT ENOUGH INFO": "the evidence is not enough to decide"}
VC_ASK = ["Given the evidence, is the claim supported, refuted, or undecided?", "How does the evidence bear on the claim?"]
VC_YESNO = ["Does the evidence support the claim?", "Is the claim backed by the evidence?"]


def vitaminc(stats: dict):
    rng = random.Random(2026101814)
    rows = [r for r in map(json.loads, (RAW / "tals__vitaminc/train.jsonl").open())
            if r["revision_type"] == "real" and not str(r.get("FEVER_id") or "")]
    # Whole cases: gold4 drew 10,000 rows per label, which kept only 4% of rows with their contrast partner.
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case_id"]].append(r)
    cases = sorted(by)
    rng.shuffle(cases)
    picked = []
    for case in cases:
        if len(picked) >= VITAMINC_ROWS:
            break
        picked += by[case]
    names = list(VC_KEYS)
    for r in sorted(picked, key=lambda r: r["unique_id"]):
        state = {"evidence": r["evidence"], "claim": r["claim"]}
        if rng.random() < .7:
            options = names if rng.random() < .5 else dict(VC_KEYS)
            question, target = common.choice(state, rng.choice(VC_ASK), options, names.index(r["label"]))
        else:
            question, target = common.noul(rng.choice(VC_YESNO), r["label"] == "SUPPORTS")
        yield common.example(VITAMINC, f"{r['case_id']}:{r['unique_id']}", "fact-verification", state, question, target, "train")
    stats["sampled"] = f"{len(picked)} rows in whole cases (seed 2026101814) of {len(rows)} real-revision rows"
    stats["labels"] = dict(collections.Counter(r["label"] for r in picked))


# --- PAWS ---------------------------------------------------------------------------------------------------
PAWS = "paws-train"
PAWS_ROWS = 20_000
SAME = ["Do the two sentences mean the same thing?", "Is sentence 2 a paraphrase of sentence 1?", "Do both sentences describe the same facts?"]


def paws(stats: dict):
    rng = random.Random(2026101815)
    rows = pq.read_table(RAW / "google-research-datasets__paws/labeled_final/train-00000-of-00001.parquet").to_pylist()
    for r in sorted(rng.sample(rows, PAWS_ROWS), key=lambda r: int(r["id"])):
        state = {"sentence 1": r["sentence1"], "sentence 2": r["sentence2"]}
        question, target = common.noul(rng.choice(SAME), int(r["label"]) == 1)
        yield common.example(PAWS, f"{digest(r['sentence1'])}:{r['id']}", "paraphrase", state, question, target, "train")


# --- SciTail ------------------------------------------------------------------------------------------------
SCITAIL = "scitail-train"
SCI_ASK = ["Does the premise support the hypothesis?", "Can the hypothesis be concluded from the premise?"]


def scitail(stats: dict):
    rng = random.Random(2026101816)
    for i, r in enumerate(pq.read_table(RAW / "allenai__scitail/dgem_format/train-00000-of-00001.parquet").to_pylist()):
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        question, target = common.noul(rng.choice(SCI_ASK), r["label"] == "entails")
        yield common.example(SCITAIL, f"{digest(r['hypothesis'])}:{i}", "science-entailment", state, question, target, "train")


# --- Finance tweets -----------------------------------------------------------------------------------------
FINTWEET = "fintweet-train"
TOPIC_ASK = ["What is this finance tweet about?", "Pick the topic of the tweet."]
SENT_ASK = ["Is this finance tweet bearish, bullish, or neutral?", "What market sentiment does the tweet express?"]


def fintweet(stats: dict):
    rng = random.Random(2026101817)
    topic = pd.read_csv(RAW / "zeroshot__twitter-financial-news-topic/topic_train.csv")
    skipped = collections.Counter()
    for text, label in zip(topic["text"], topic["label"]):
        if text_role("fintopic", text) != "direct":  # an icl-classify example or query tweet
            skipped["topic"] += 1
            continue
        question, target = common.choice(text, rng.choice(TOPIC_ASK), FIN_TOPICS, int(label))
        yield common.example(FINTWEET, f"{digest(text)}:topic", "finance-topic", text, question, target, "train")
    sent = pd.read_csv(RAW / "zeroshot__twitter-financial-news-sentiment/sent_train.csv")
    for text, label in zip(sent["text"], sent["label"]):
        if text_role("fintopic", text) != "direct":
            skipped["sentiment"] += 1
            continue
        question, target = common.choice(text, rng.choice(SENT_ASK), ["bearish", "bullish", "neutral"], int(label))
        yield common.example(FINTWEET, f"{digest(text)}:sentiment", "finance-sentiment", text, question, target, "train")
    stats["left_to_icl_classify"] = dict(skipped)


BUILDERS = [
    (TATQA_YESNO, tatqa_yesno, {"upstream": "TAT-QA train+dev (as finance.py)", "licence": "CC-BY-4.0", "label_origin": "verify rows: human answers, distractors programmatic; direction rows: computed from the two table values named by the human derivation (year_direction)"}),
    (ICL, icl, {"upstream": "DeveloperOats/DBPedia_Classes train; zeroshot/twitter-financial-news-topic train", "licence": "DBpedia: card CC0-1.0, text from Wikipedia (CC-BY-SA, per dbpedia.org); tweets MIT", "label_origin": "dataset labels (DBpedia: ontology classes of Wikipedia articles); prompts programmatic"}),
    (WNED, wned, {"upstream": "facebook/kilt_tasks wned validation", "licence": "KILT MIT; text from Wikipedia (CC-BY-SA)", "label_origin": "human (Wikipedia links)"}),
    (VITAMINC, vitaminc, {"upstream": "tals/vitaminc train (real revisions only)", "licence": "CC-BY-SA-3.0", "label_origin": "human"}),
    (PAWS, paws, {"upstream": "google-research-datasets/paws labeled_final train", "licence": "free for any purpose (google-research-datasets/paws LICENSE)", "label_origin": "human"}),
    (SCITAIL, scitail, {"upstream": "allenai/scitail dgem_format train", "licence": "Apache-2.0 (github allenai/scitail LICENSE)", "label_origin": "human"}),
    (FINTWEET, fintweet, {"upstream": "zeroshot/twitter-financial-news-topic + -sentiment train", "licence": "MIT", "label_origin": "human"}),
]


def tatqa_yesno_tables(e: dict) -> tuple[list[str], list[str]]:
    """finance.tatqa_tables for rows whose state may lack the question (it is then in the instructions)."""
    state = e["state"]
    if isinstance(state, dict):
        return [state["text"]] + ([state["question"]] if "question" in state else []), [state["table"]]
    head, _, rest = state.partition("\n\nText:\n")
    return [rest], [head.removeprefix("Table:\n")]


if __name__ == "__main__":
    import sys
    wanted = set(sys.argv[1:])
    for name, build, meta in BUILDERS:
        if not wanted or name in wanted:
            common.write(name, build(meta), meta, tables=tatqa_yesno_tables if name == TATQA_YESNO else None)
