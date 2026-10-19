"""Finance builders (2026-10-17): entity-level headline sentiment and table+text numerical reasoning.

  python -m qwen_decision.v3.sources.finance

SEntFiN 1.0 (Sinha et al. 2022, github.com/pyRis/SEntFiN @ eba43610, MIT): 10,753 Indian financial news
headlines with a human sentiment label per named entity (14,404 labels; 1,233 headlines give different
entities different sentiments). Only the human-annotated SEntFiN.csv is read; the paper's large set of
model predictions is not. One question per (headline, entity): positive / neutral / negative toward that
entity. Repeated headlines take the id of their first occurrence, so all their entities are one item.
Aimed at entity-level financial sentiment (index FinEntity, a different corpus; the decontam filter runs).

TAT-QA (Zhu et al. 2021, github.com/NExTplusplus/TAT-QA @ 870accc4, CC BY 4.0): 2,201 hybrid contexts
from financial reports (a table plus paragraphs) with 13,215 human-written questions and answers (train +
dev; the test split is never read). Each question becomes a 4-option choice: the human answer (with its
scale: thousand/million/billion/percent) and three distractors taken, in this order, from the other
answers on the same context, the context's own table values or labels, and only then from simple
perturbations of the gold (sign, decimal shift). A distractor equal to the gold (numerically or as
normalized text) is never used. The id carries the context uid, so all questions on one table are one item.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from qwen_decision.v3.sources import common

DOWNLOADS = common.RAW / "downloads"

# --- SEntFiN -------------------------------------------------------------------------------------------------
SENTFIN = "sentfin-train"
SENTFIN_CSV = DOWNLOADS / "sentfin-eba43610/SEntFiN.csv"
SENTFIN_SHA256 = "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"
DECISION = re.compile(r"'(.+?)': '(positive|negative|neutral)'")  # the CSV's dicts are not valid Python literals
SENTIMENT = [
    "What sentiment does this headline express toward {entity}?",
    "Is this news good, bad or neither for {entity}?",
    "How is {entity} portrayed in the headline?",
    "Judge the headline's tone specifically about {entity}, not the market as a whole.",
    "For {entity} alone, is the headline positive, neutral or negative?",
]
LABELS = {"positive": "favourable for the entity", "neutral": "neither favourable nor unfavourable",
          "negative": "unfavourable for the entity"}


def sentfin(stats: dict):
    with SENTFIN_CSV.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == SENTFIN_SHA256
    rng = random.Random(2026101701)
    first = {}
    counts = collections.Counter()
    for r in csv.DictReader(SENTFIN_CSV.open(encoding="utf-8")):
        title = " ".join(r["Title"].split())
        decisions = DECISION.findall(r["Decisions"])
        if not decisions or len(decisions) != r["Decisions"].count("': '"):
            counts["unparsed"] += 1
            continue
        item = first.setdefault(title.lower(), r["S No."])
        for k, (entity, label) in enumerate(decisions):
            options = list(LABELS) if rng.random() < .5 else {key: d for key, d in LABELS.items()}
            keys = list(LABELS)
            state = title if rng.random() < .5 else {"headline": title}
            question, target = common.choice(state, common.instruction(rng, SENTIMENT, entity=entity), options,
                                             keys.index(label))
            counts[label] += 1
            counts["multi_entity" if len(decisions) > 1 else "single_entity"] += 1
            yield common.example(SENTFIN, f"{item}:{r['S No.']}:{k}", "finance-sentiment", state, question, target, "all")
    stats["counts"] = dict(counts)


# --- TAT-QA --------------------------------------------------------------------------------------------------
TATQA = "tatqa-train"
TATQA_DIR = DOWNLOADS / "tatqa-870accc4"
TATQA_SHA256 = {"train": "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69",
                "dev": "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"}
ASK = [
    "Using the table and the text, pick the correct answer.",
    "Answer the question from this financial report excerpt.",
    "Which option answers the question correctly, given the report?",
    "Read the table and paragraphs, work out the answer, and select it.",
]
NUMBER = re.compile(r"^\$?\s*\(?-?\$?\s*[\d,]*\.?\d+\s*%?\)?$")
YEAR = range(1900, 2101)


def as_number(text) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not NUMBER.match(s):
        return None
    negative = ("(" in s and s.endswith(")")) or "-" in s
    digits = re.sub(r"[^\d.]", "", s)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def show_number(value: float, scale: str) -> str:
    text = f"{value:,.2f}".rstrip("0").rstrip(".") if value != int(value) else f"{int(value):,}"
    if scale == "percent":
        return text + "%"
    return f"{text} {scale}" if scale else text


def show_answer(q: dict) -> tuple[str, float | None]:
    answer, scale = q["answer"], q.get("scale") or ""
    if q["answer_type"] in ("arithmetic", "count") or (q["answer_type"] == "span" and as_number(answer[0]) is not None
                                                        and scale):
        value = as_number(answer if not isinstance(answer, list) else answer[0])
        if value is not None:
            return show_number(value, scale), value
    if isinstance(answer, list):
        return ", ".join(" ".join(str(a).split()) for a in answer), None
    return " ".join(str(answer).split()), None


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def render_context(doc: dict, rng: random.Random):
    table = "\n".join(" | ".join(" ".join(c.split()) for c in row) for row in doc["table"]["table"])
    text = "\n".join(p["text"].strip() for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
    if rng.random() < .5:
        return {"table": table, "text": text}
    return f"Table:\n{table}\n\nText:\n{text}"


def distractors(q: dict, gold: str, value: float | None, doc: dict, rng: random.Random, k: int = 3) -> list[str] | None:
    scale = q.get("scale") or ""
    seen = {norm(gold)}
    picked = []

    def offer(text: str, number: float | None = None):
        if len(picked) >= k or not text or norm(text) in seen:
            return
        if value is not None and number is not None and abs(number - value) <= 1e-9 * max(1., abs(value)):
            return
        seen.add(norm(text))
        picked.append(text)

    others = [(show_answer(o), o["answer_type"]) for o in doc["questions"] if o["uid"] != q["uid"]]
    cells = [c for row in doc["table"]["table"] for c in row if c.strip()]
    # Years in headers are numbers too; they are distractors only when the gold is itself a year.
    year_ok = value is not None and value == int(value) and int(value) in YEAR
    keep = lambda v: year_ok or not (v == int(v) and int(v) in YEAR)
    if value is not None:
        pool = [v for (t, v), _ in others if v is not None and keep(v)]
        rng.shuffle(pool)
        for v in pool:
            offer(show_number(v, scale), v)  # the gold's scale, so options differ only in value
        numbers = [v for v in map(as_number, cells) if v is not None and keep(v)]
        rng.shuffle(numbers)
        for v in numbers:
            offer(show_number(v, scale), v)
        for v in (-value, value * 10, value / 10, value + 1, value - 1):
            offer(show_number(round(v, 2), scale), round(v, 2))
    else:
        if q["answer_type"] == "multi-span" and len(q["answer"]) > 1:
            items = [" ".join(str(a).split()) for a in q["answer"]]
            offer(", ".join(items[:-1]))
            labels = [" ".join(row[0].split()) for row in doc["table"]["table"] if row and row[0].strip()]
            outside = [x for x in labels if norm(x) not in {norm(i) for i in items}]
            if outside:
                swapped = list(items)
                swapped[rng.randrange(len(items))] = rng.choice(outside)
                offer(", ".join(swapped))
        # Same answer type (a single span is not offered a list), then table labels and report sentences;
        # texts of a similar length come first so the gold does not stand out by length alone.
        pool = [t for (t, v), kind in others if v is None and kind == q["answer_type"]]
        sentences = [x.strip() for p in doc["paragraphs"] for x in re.split(r"(?<=[.;])\s+", p["text"]) if x.strip()]
        texts = [" ".join(c.split()) for c in cells if as_number(c) is None] + sentences
        rng.shuffle(pool)
        rng.shuffle(texts)
        similar = lambda t: .4 <= len(t) / max(1, len(gold)) <= 2.5
        for t in [t for t in pool + texts if similar(t)] + pool + texts:
            offer(t)
    return picked if len(picked) == k else None


def tatqa(stats: dict):
    rng = random.Random(2026101702)
    counts = collections.Counter()
    for split in ("train", "dev"):
        path = TATQA_DIR / f"tatqa_dataset_{split}.json"
        with path.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == TATQA_SHA256[split]
        for doc in json.loads(path.read_text()):
            for q in doc["questions"]:
                gold, value = show_answer(q)
                if not gold:
                    counts["empty_answer"] += 1
                    continue
                wrong = distractors(q, gold, value, doc, rng)
                if wrong is None:
                    counts["too_few_distractors"] += 1
                    continue
                options = [gold] + wrong
                rng.shuffle(options)
                state = render_context(doc, rng)
                if isinstance(state, dict):
                    state = {**state, "question": q["question"]}
                else:
                    state = f"{state}\n\nQuestion:\n{q['question']}"
                question, target = common.choice(state, rng.choice(ASK), options, options.index(gold))
                counts[f"{q['answer_type']}"] += 1
                counts[split] += 1
                yield common.example(TATQA, f"{split}:{doc['table']['uid']}:{q['uid']}", "finance-numeric", state,
                                     question, target, split)
    stats["counts"] = dict(counts)


def tatqa_tables(e: dict) -> tuple[list[str], list[str]]:
    """(paragraphs and question, table) of a TAT-QA example, for common.write's table-aware filter."""
    state = e["state"]
    if isinstance(state, dict):
        return [state["text"], state["question"]], [state["table"]]
    head, _, rest = state.partition("\n\nText:\n")
    return [rest], [head.removeprefix("Table:\n")]


if __name__ == "__main__":
    meta = {"upstream": "github.com/pyRis/SEntFiN SEntFiN.csv (human annotations only)",
            "revision": f"eba43610fe3cb57a3e5773cd6c037da1f8992ad5 sha256:{SENTFIN_SHA256}",
            "split": "all (no official split)", "licence": "MIT (repository LICENSE)", "label_origin": "human"}
    common.write(SENTFIN, sentfin(meta), meta)
    meta = {"upstream": "github.com/NExTplusplus/TAT-QA dataset_raw (train, dev; test never read)",
            "revision": "870accc41953dcde885aabeb963d94aabdc0fbc3", "split": "train+dev",
            "licence": "CC BY 4.0 (README)", "label_origin": "human answers; distractors programmatic"}
    common.write(TATQA, tatqa(meta), meta, tables=tatqa_tables)
