"""Contamination index of every Decision Index row and every JEVal row, and the training-data filter.

Training on an index test item makes the board count it wrong (and audits find it); JEVal is our
held-out proxy. Both are indexed as:
  - 13-word shingles of each text field (normalized: lowercase, alphanumeric words), keeping the
    shingles whose hash is 0 mod SAMPLE so the table stays small; a passage of ~100 shared words
    still shares several kept shingles;
  - exact normalized text of short fields and of each line of a field (at most SHORT_WORDS words),
    which shingles cannot catch (intent utterances inside a templated instruction, claims, short
    questions).
Both keep only what appears in at most MAX_REPEATS rows: text repeated across many rows is a
template ("Summarize the following news within 89 words:", a fixed task preamble, a corpus document
offered to many queries), and matching it would drop unrelated training data. Items made only of
such repeated parts (cfcolor palettes) cannot be caught by text; filter those by their own keys.

Rows are read straight from the row files, with no edition filter: rows an edition retires or
replaces (the 0.2 GSM8K rows, ForecastBench, the ToolRet queries cut in 0.2) still hold test items.

  python -m qwen_decision.v3.decontam build --rows <selected-rows (0.1)> <added-rows> <gsm8k-rows> --out data/decontam
  python -m qwen_decision.v3.decontam check   # positive controls; must print "passed": true
  # in code: Filter("data/decontam").flagged(texts) -> reason or None
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NGRAM, SAMPLE, SHORT_WORDS = 13, 4, 30
MAX_REPEATS = 8  # a shingle or short text found in more index/JEVal rows than this is a template, not an item
MIN_SHARED = 2  # kept shingles in common before anything is flagged
SHARED_ABSOLUTE, SHARED_FRACTION = 10, .2  # documents mode: then 10 in all, or a fifth of the example's own
LONG_FIELD = 120  # words; in documents mode, the lines of longer fields are not checked
_WORD = re.compile(r"[a-z0-9]+|[^\sa-z0-9]", re.UNICODE)
_P = np.uint64(1099511628211)


_DIGITS = re.compile(r"[0-9]+")


def words(text: str) -> list[str]:
    """Lowercased words with every digit run masked to 0; any non-ASCII character (e.g. CJK) is its own word.

    Masking numbers folds templates that differ only in a number ("within 89 words", "within 108
    words") into one repeated text, and makes a numeric rewrite of a test item match it.
    """
    return [_DIGITS.sub("0", w) for w in _WORD.findall(text.lower()) if w.isalnum()]


def strings(value) -> list[str]:
    """Every text leaf of a state/question value (JSON objects are walked, not dumped)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in strings(k) + strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return [str(value)]


def shingles(ws: list[str]) -> np.ndarray:
    """Kept 13-word shingle hashes (uint64) of one word list."""
    if len(ws) < NGRAM:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((zlib.crc32(w.encode()) for w in ws), dtype=np.uint64, count=len(ws))
    h = np.zeros(len(ws) - NGRAM + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(NGRAM):
            h = h * _P + ids[j:len(ws) - NGRAM + 1 + j]
    return h[h % np.uint64(SAMPLE) == 0]


def short_key(ws: list[str]) -> str | None:
    return " ".join(ws) if 3 <= len(ws) <= SHORT_WORDS else None


def short_keys(text: str) -> set[str]:
    """Short-text keys of a whole field and of each of its lines."""
    keys = {short_key(words(text))}
    if "\n" in text:
        keys |= {short_key(words(line)) for line in text.split("\n")}
    return keys - {None}


def row_texts(state, questions: dict) -> list[str]:
    out = strings(state)
    for q in questions.values():
        out += strings(q.get("instructions")) + strings(q.get("criteria"))
    return out


def build(args) -> None:
    import pyarrow.parquet as pq

    from collections import Counter

    hashes, short, counts = [], Counter(), {"index_rows": 0, "jeval_rows": 0}

    def add(texts):
        keys = set()
        for t in texts:
            keys |= short_keys(t)
        short.update(keys)  # counted once per row
        hashes.append(np.unique(np.concatenate([shingles(words(t)) for t in texts] or [np.empty(0, np.uint64)])))

    for path in args.rows:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                add(row_texts(r.get("state"), r["questions"]))
                counts["index_rows"] += 1
    for r in pq.read_table(ROOT / "data/jeval/data/test.parquet").to_pylist():
        add(row_texts(r["state"], json.loads(r["questions_json"])))
        counts["jeval_rows"] += 1
    table, rows_with = np.unique(np.concatenate(hashes), return_counts=True)
    template_shingles = int((rows_with > MAX_REPEATS).sum())
    table = table[rows_with <= MAX_REPEATS]
    templates = sum(1 for n in short.values() if n > MAX_REPEATS)
    short = {k for k, n in short.items() if n <= MAX_REPEATS}
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "shingles.npy", table)
    with gzip.open(args.out / "short.txt.gz", "wt") as f:
        f.write("\n".join(sorted(short)))
    meta = {**counts, "rows_files": [str(p) for p in args.rows], "kept_shingles": int(len(table)), "short_texts": len(short),
            "short_template_lines_skipped": templates, "template_shingles_skipped": template_shingles,
            "max_repeats": MAX_REPEATS,
            "ngram": NGRAM, "sample": SAMPLE, "short_words": SHORT_WORDS, "min_shared": MIN_SHARED}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


class Filter:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.table = np.load(directory / "shingles.npy")
        with gzip.open(directory / "short.txt.gz", "rt") as f:
            self.short = set(f.read().split("\n"))
        self.meta = json.loads((directory / "meta.json").read_text())

    def flagged(self, texts: list[str], documents: bool = False, options: list[str] = ()) -> str | None:
        """Why these texts (one training example) overlap the index or JEVal, or None.

        Default (strict): any short-text key of a field or of any of its lines, or MIN_SHARED kept
        shingles, flags the example.
        documents=True (only for sources whose document-level disjointness from the test was checked
        separately, e.g. ContractNLI contracts): lines inside fields longer than LONG_FIELD words are
        not checked (headings, signature blocks), and shingles flag only at SHARED_ABSOLUTE in all or
        SHARED_FRACTION of the example's own kept shingles (a standard clause shares a few, a copied
        test passage many).

        `options` (answer option texts) are checked by shingles only: sharing a short option such as
        "stays the same" with a test question is not training on that question.
        """
        shared = kept = 0
        for t in [*texts, *options]:
            ws = words(t)
            keys = set() if t in options else short_keys(t) if not documents or len(ws) <= LONG_FIELD else {short_key(ws)} - {None}
            for key in keys:
                if key in self.short:
                    return f"short text match: {key[:80]}"
            h = shingles(ws)
            if len(h):
                # The table is sorted and unique, so a binary search per shingle is enough.
                at = np.minimum(np.searchsorted(self.table, h), len(self.table) - 1)
                shared += int((self.table[at] == h).sum())
                kept += len(h)
                if not documents and shared >= MIN_SHARED:
                    return f"{shared} shared 13-word shingles"
        if documents and shared >= MIN_SHARED and (shared >= SHARED_ABSOLUTE or shared >= SHARED_FRACTION * kept):
            return f"{shared} of {kept} kept 13-word shingles shared"
        return None


def check(args) -> None:
    """Positive controls: index rows of every benchmark, JEVal rows and edited items must be flagged."""
    import random
    from collections import Counter, defaultdict
    import pyarrow.parquet as pq

    f, rng = Filter(args.out), random.Random(0)
    by = defaultdict(list)
    for path in json.loads((args.out / "meta.json").read_text())["rows_files"]:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                by[r["_evaluation"]["dataset"]].append(r)
    missed, n = Counter(), Counter()
    for name, rows in by.items():
        for r in rng.sample(rows, min(30, len(rows))):
            n[name] += 1
            missed[name] += not f.flagged(row_texts(r.get("state"), r["questions"]))
    jeval = pq.read_table(ROOT / "data/jeval/data/test.parquet").to_pylist()
    missed_jeval = [r for r in rng.sample(jeval, 500) if not f.flagged(row_texts(r["state"], json.loads(r["questions_json"])))]
    jeval_missed = len(missed_jeval)
    utterance = by["CLINC150+OOS"][0]["questions"]["q1"]["instructions"].split("\n")[-1]
    report = {"index_rows_checked": sum(n.values()), "benchmarks": len(n),
              "index_rows_missed": {k: f"{missed[k]}/{n[k]}" for k in n if missed[k]}, "jeval_missed_of_500": jeval_missed,
              "clinc_utterance_recased": f.flagged([utterance.upper() + "!!"]),
              "unrelated_text": f.flagged(["The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon."]),
              "jeval_missed_examples": [(r["source"], str(r["state"])[:80], r["questions_json"][:120]) for r in missed_jeval]}
    # cfcolor palettes and POP909 note lists repeat across rows, so text cannot single them out; neither is a
    # training source (they are excluded by name), and ForecastBench is retired from the index.
    # CLadder and BBH items are generated from templates that differ in numbers or symbols only, so a few
    # rows are indistinguishable from their siblings; both are excluded as training sources by name.
    textless, templated = {"cfcolor", "POP909-CL", "ForecastBench"}, {"CLadder", "BBH fixed-option tasks"}
    report["passed"] = all(missed[k] <= (6 if k in templated else 1) for k in n if k not in textless) and jeval_missed <= 2 and bool(report["clinc_utterance_recased"]) \
        and report["unrelated_text"] is None
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rows", type=Path, nargs="+", required=True, help="row files (.jsonl.gz), read with no edition filter")
    b.add_argument("--out", type=Path, default=ROOT / "data/decontam")
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, default=ROOT / "data/decontam")
    args = p.parse_args()
    {"build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
