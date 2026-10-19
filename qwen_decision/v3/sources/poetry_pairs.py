"""Poetic minimal pairs (2026-10-19): which version of a public-domain poem is the original?

  python -m qwen_decision.v3.sources.poetry_pairs --fetch   # download the pinned upstream files
  python -m qwen_decision.v3.sources.poetry_pairs           # build data/phase2/poem-pairs-gen.jsonl

The skill of the index's BPoMP (catalog 20: an original limerick against a perturbed copy) on poems that are
not in it. BPoMP itself and its OEDILF source are never read for training; the release is read only to drop any
poem of ours that shares lines with a BPoMP original (BPOMP below).

Poems (all public domain):
- Limericks from Project Gutenberg books: Edward Lear's A Book of Nonsense and More Nonsense (several PG
  editions, deduplicated) and Ethel Watts Mumford Grant's The Limerick Up to Date Book (1903). Lear's editions
  that print lines 3 and 4 as one line are split at the internal rhyme.
- Limerick-shaped and rhymed-quatrain (ABAB, AABB) windows of Allison Parrish's Gutenberg Poetry Corpus
  (CC0; 3.1M lines of 1,191 PG books, no stanza breaks), found by rhyme and syllable shape: a window starts
  after a line that ends a sentence and ends one itself. At most PER_BOOK windows of a kind per book.
Rhymes and syllables come from the CMU Pronouncing Dictionary (BSD-style licence); a word it lacks rhymes
by its last three letters. Synonyms come from WordNet 3.0 (the NLTK copy phase6_lang already pinned).

Perturbations, as in BPoMP (Abdibayev et al. 2021, 2022), all programmatic, so the gold (the unaltered
text) is known by construction (label_origin programmatic):
- corrupted rhyme: a rhyming line-end word replaced by a WordNet synonym that does not rhyme and has about
  the same syllable count;
- shuffled rhyming words / lines: two rhyming line-end words, or two rhyming lines, swapped;
- replaced last line: BPoMP's (delayed) beam search let a language model rewrite the last line; here the last
  line is replaced by the last line of a donor quatrain (about as many syllables), one that keeps the rhyme or
  any. Donor quatrains (DONORS of them) are never used as originals, so no line crosses the dev split;
- delete 1-3 words: any words, line-end rhyming words, or other words. Half the deletions keep BPoMP's
  spacing (the word removed, runs of spaces collapsed, a space left before punctuation), half are tidied;
- insert 1-3 words (not in BPoMP): the poem's own words put in mid-line. Without them the original is the
  longest version in ~90% of the questions, a shortcut; with them ~65% (stats.original_longest).
Most questions ask for the original among two versions; some show three (two different perturbations), and
REVERSED of the two-version questions ask which one was altered. Instructions are our own phrasings.
Item key (first id segment): the poem, so every version of a poem stays on one side of the dev split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase5 import fetch_spec, hub_dir

NAME = "poem-pairs-gen"
RAW = common.RAW / "poetry"
POETRY_REPO, POETRY_REVISION = "biglam/gutenberg-poetry-corpus", "fcd42e249fed48dbd1d3b9b969528ef9298d3464"
CMUDICT_COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
CMUDICT = RAW / f"cmusphinx__cmudict@{CMUDICT_COMMIT[:12]}/cmudict.dict"
# Project Gutenberg plain texts (sha256 of the files fetched on 2026-10-19; PG rewrites headers now and then)
BOOKS = {
    13646: ("A Book of Nonsense", "Edward Lear", "9aed4aad00ff09c9b5da1375097d3ccc00011b5df570d0cc3c34b01816d5af5e"),
    982: ("The Book of Nonsense", "Edward Lear", "0a5da7e27565f8a50940bffc3eb55778ac4d3e146ed07cfc5f075aff9e557cba"),
    13648: ("More Nonsense", "Edward Lear", "cb8447b321941969f90a66ec8ddf1520533fa4b1c9bfb551a12d1e1d532d8045"),
    13650: ("Nonsense Books", "Edward Lear", "284c4a29169b8995cf8ec4e01b0e3802fe57bc542f009b9a73ab940940b35c90"),
    78335: ("The Limerick Up to Date Book", "Ethel Watts Mumford Grant",
            "cfdd1d466b0266bcf3622f8cc95c2eac08e1b6dbfa3beaf4a6a7eef2454f7971"),
}
FETCH = {
    "cmudict": ("github", "cmusphinx/cmudict", CMUDICT_COMMIT, ["LICENSE", "README", "cmudict.dict"]),
    **{f"pg{n}": ("url", f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt", f"pg{n}.txt") for n in BOOKS},
}
BPOMP = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/downloads"
SEED = 2026101920
PER_BOOK = 40          # corpus windows of each kind kept per Gutenberg book
DONORS = .15           # share of quatrains kept aside as donors of replacement last lines
QUATRAINS = 3000       # quatrains used as originals (of ~30k windows), so limericks keep about half the rows
QUATRAIN_VARIANTS = 3  # versions made of each quatrain; limericks get every applicable variant once,
LIMERICK_INSERTIONS = 2  # and each insertion variant this many times (different words)
THREE_WAY, REVERSED, SLASHES = .12, .2, .1
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
VOWEL = re.compile(r"[aeiouy]+")
ENDS_SENTENCE = re.compile(r"[.!?;:]['\"”’)]*$")

VARIANTS = ["corrupted rhyme", "shuffled rhyming words", "shuffled rhyming lines", "replaced last line (rhyming)",
            "replaced last line", "delete 1 word", "delete 2 words", "delete 3 words", "delete 1 rhyming EOL word",
            "delete 2 rhyming EOL words", "delete 3 rhyming EOL words", "delete 1 nonrhyming word",
            "delete 2 nonrhyming words", "delete 3 nonrhyming words", "insert 1 word", "insert 2 words", "insert 3 words"]
ORIGINAL = [
    "Which version is the {form} as it was originally written? The other has been altered.",
    "One of these is the authentic {form} and the rest are corrupted copies. Which one is authentic?",
    "Pick the original text. An altered copy may have swapped, replaced or missing words or lines, so check the "
    "rhyme scheme, the rhythm and the grammar.",
    "These versions of a {form} differ by a small edit. Choose the one the poet actually wrote.",
    "Which candidate has not been tampered with?",
    "Identify the unaltered {form}: its rhymes, meter and sense should all hold together.",
    "Only one of these texts is the real {form}. Which is it?",
]
ALTERED = [
    "Which version has been altered from the original {form}?",
    "One of these two texts was corrupted by an edit. Which one?",
    "Spot the tampered copy of this {form}.",
    "Which of the two has been changed from what the poet wrote?",
]
KEYS = [["A", "B", "C"], ["version 1", "version 2", "version 3"], ["first", "second", "third"],
        ["poem_1", "poem_2", "poem_3"], ["text A", "text B", "text C"]]


# --- pronunciation ------------------------------------------------------------------------------------------
class Sounds:
    """CMU Pronouncing Dictionary: rhyme parts (from the last stressed vowel) and syllable counts."""

    def __init__(self, path: Path):
        self.prons = collections.defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                word, *phones = line.split()
                self.prons[re.sub(r"\(\d+\)$", "", word)].append(phones)

    def known(self, word: str) -> bool:
        return norm_word(word) in self.prons

    def rhyme_parts(self, word: str) -> set[str]:
        parts = set()
        for phones in self.prons.get(norm_word(word), []):
            stressed = [i for i, p in enumerate(phones) if p[-1] in "12"] or [i for i, p in enumerate(phones) if p[-1] == "0"]
            if stressed:
                parts.add(" ".join(re.sub(r"\d", "", p) for p in phones[stressed[-1]:]))
        return parts

    def rhymes(self, a: str, b: str, identical: bool = True) -> bool:
        a, b = norm_word(a), norm_word(b)
        if not a or not b:
            return False
        if a == b:
            return identical
        pa, pb = self.rhyme_parts(a), self.rhyme_parts(b)
        if pa and pb:
            return bool(pa & pb)
        return len(a) >= 3 and len(b) >= 3 and a[-3:] == b[-3:]

    def syllables(self, text: str) -> int:
        count = 0
        for w in WORD.findall(text):
            phones = self.prons.get(norm_word(w))
            count += sum(p[-1].isdigit() for p in phones[0]) if phones else max(1, len(VOWEL.findall(w.lower())))
        return count


def norm_word(word: str) -> str:
    return word.lower().replace("’", "'").strip("'")


def norm_line(line: str) -> str:
    return " ".join(w.lower() for w in WORD.findall(line))


def last_word(line: str) -> re.Match | None:
    found = list(WORD.finditer(line))
    return found[-1] if found else None


def end_word(line: str) -> str:
    m = last_word(line)
    return m.group(0) if m else ""


# --- poems --------------------------------------------------------------------------------------------------
def scheme_groups(lines: list[str], snd: Sounds, form: str) -> list[list[int]] | None:
    """Rhyme groups (line indices) of a limerick (AABBA) or quatrain (ABAB/AABB), or None if the shape fails."""
    e = [end_word(x) for x in lines]
    if form == "limerick":
        a = [i for i in (0, 1, 4) if any(snd.rhymes(e[i], e[j]) for j in (0, 1, 4) if j != i)]
        if len(a) < 2 or not snd.rhymes(e[2], e[3]):
            return None
        return [a, [2, 3]]
    if snd.rhymes(e[0], e[2], False) and snd.rhymes(e[1], e[3], False) and not snd.rhymes(e[0], e[1]):
        return [[0, 2], [1, 3]]
    if snd.rhymes(e[0], e[1], False) and snd.rhymes(e[2], e[3], False) and not snd.rhymes(e[1], e[2]):
        return [[0, 1], [2, 3]]
    return None


def limerick_shape(lines: list[str], snd: Sounds, strict: bool) -> bool:
    s = [snd.syllables(x) for x in lines]
    long_, short = (s[0], s[1], s[4]), (s[2], s[3])
    if sum(short) / 2 >= sum(long_) / 3:
        return False
    if not strict:
        return True
    e = [end_word(x) for x in lines]
    return (all(6 <= x <= 13 for x in long_) and all(3 <= x <= 8 for x in short) and sum(short) / 2 + 2 <= sum(long_) / 3
            and snd.rhymes(e[0], e[1], False) and snd.rhymes(e[2], e[3], False) and not snd.rhymes(e[1], e[2])
            and (snd.rhymes(e[1], e[4]) or snd.rhymes(e[0], e[4])))


def split_internal(line: str, snd: Sounds) -> list[str] | None:
    """Lear prints a limerick's lines 3 and 4 as one line; split it where the two halves' last words rhyme."""
    for m in re.finditer(r"[,;:!?]['\"”’]?\s+", line):
        left, right = line[:m.end()].rstrip(), line[m.end():].strip()
        if len(WORD.findall(left)) >= 2 and len(WORD.findall(right)) >= 2 and snd.rhymes(end_word(left), end_word(right), False):
            return [left, right[:1].upper() + right[1:]]
    return None


def book_text(n: int) -> str:
    path = RAW / f"pg{n}/pg{n}.txt"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == BOOKS[n][2], path
    text = data.decode("utf-8").replace("\r", "")
    start = re.search(r"\*\*\* ?START OF TH(E|IS) PROJECT GUTENBERG[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF TH(E|IS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def book_limericks(snd: Sounds, stats: dict) -> list[dict]:
    """Limericks of the PG books: runs of indented lines; 5-line runs, or 4-line runs with lines 3+4 joined."""
    found = []
    for n, (title, author, _) in BOOKS.items():
        runs, run = [], []
        for line in book_text(n).split("\n"):
            verse = re.match(r"^ {2,}\S", line) and not line.strip().startswith("[") and WORD.search(line)
            if verse:
                run.append(line.strip())
            elif run:
                runs.append(run)
                run = []
        runs += [run] if run else []
        kept = 0
        for run in runs:
            if len(run) == 4:
                halves = split_internal(run[2], snd)
                run = run[:2] + halves + run[3:] if halves else run
            if len(run) != 5:
                continue
            groups = scheme_groups(run, snd, "limerick")
            if groups and limerick_shape(run, snd, strict=False):
                found.append({"lines": run, "form": "limerick", "groups": groups, "origin": f"pg{n}",
                              "collection": title, "poet": author})
                kept += 1
        stats["book_limericks"][f"pg{n}"] = kept
    return found


def corpus_windows(snd: Sounds, stats: dict) -> list[dict]:
    """Limerick-shaped and rhymed-quatrain windows of the Gutenberg Poetry Corpus, aligned to sentence ends."""
    table = pq.read_table(next((hub_dir(POETRY_REPO, POETRY_REVISION) / "data").glob("*.parquet")))
    lines, books = table.column("line").to_pylist(), table.column("gutenberg_id").to_pylist()
    rng = random.Random(SEED)
    by_book = collections.defaultdict(lambda: {"limerick": [], "quatrain": []})
    ok = [bool(x) and 12 <= len(x) <= 90 and not re.search(r"\d|[A-Z]{4,}|_", x) and len(WORD.findall(x)) >= 2 for x in lines]
    i = 0
    while i < len(lines) - 4:
        starts = (i == 0 or books[i - 1] != books[i] or bool(ENDS_SENTENCE.search(lines[i - 1].strip())))
        if not (starts and ok[i] and lines[i].strip()[:1].isupper()):
            i += 1
            continue
        taken = 0
        for size, form in ((5, "limerick"), (4, "quatrain")):
            window = [x.strip() for x in lines[i:i + size]]
            if books[i + size - 1] != books[i] or not all(ok[i:i + size]) or not ENDS_SENTENCE.search(window[-1]):
                continue
            if form == "limerick" and not limerick_shape(window, snd, strict=True):
                continue
            groups = scheme_groups(window, snd, form)
            if groups:
                by_book[books[i]][form].append({"lines": window, "form": form, "groups": groups, "origin": f"gpc:{books[i]}"})
                taken = size
                break
        i += taken or 1
    found = []
    for book, kinds in sorted(by_book.items()):
        for form, poems in kinds.items():
            stats["corpus_windows_seen"][form] += len(poems)
            found += rng.sample(poems, min(PER_BOOK, len(poems)))
    stats["corpus_books"] = len(by_book)
    return found


def bpomp_lines() -> set[str]:
    seen = set()
    for path in sorted(BPOMP.glob("BPoMP_p*.json")):
        for pairs in json.loads(path.read_text()).values():
            for pair in pairs:
                if isinstance(pair, list) and pair:
                    seen.update(x for x in map(norm_line, pair[0].split("\n")) if len(x.split()) >= 4)
    if not seen:
        raise SystemExit(f"BPoMP release not found under {BPOMP}: run the suite rebuild download first")
    return seen


def deduplicate(poems: list[dict], banned: set[str], stats: dict) -> list[dict]:
    """Drop poems sharing two lines with a kept poem (editions of one Lear limerick) or with a BPoMP original."""
    kept, owner = [], {}
    for p in poems:
        normed = [x for x in map(norm_line, p["lines"]) if len(x.split()) >= 4]
        if sum(x in banned for x in normed) >= 2 or any(x in banned and len(x.split()) >= 8 for x in normed):
            stats["dropped_bpomp_overlap"] += 1
            continue
        if sum(x in owner for x in normed) >= 2:
            stats["dropped_duplicate"][p["origin"].split(":")[0]] += 1
            continue
        for x in normed:
            owner.setdefault(x, len(kept))
        p["key"] = hashlib.sha1("\n".join(map(norm_line, p["lines"])).encode()).hexdigest()[:12]
        kept.append(p)
    return kept


# --- perturbations ------------------------------------------------------------------------------------------
class Synonyms:
    """WordNet 3.0 single-word synonyms of a word's first senses, re-inflected for plural -s and -ed/-ing."""

    def __init__(self):
        from qwen_decision.v3.sources.phase6_lang import wordnet  # parses data/raw/phase6/nltk_wordnet/wordnet.zip
        _, self.senses, self.synsets = wordnet()

    def __call__(self, word: str, snd: Sounds) -> list[str]:
        w = norm_word(word)
        forms = [(w, "")]
        for suffix, cuts in (("s", ("s", "es")), ("ed", ("ed", "d")), ("ing", ("ing",))):
            for cut in cuts:
                if w.endswith(cut) and len(w) > len(cut) + 2:
                    forms.append((w[:-len(cut)], suffix))
            if w.endswith("ing") and suffix == "ing":
                forms.append((w[:-3] + "e", "ing"))
        out = []
        for lemma, suffix in forms:
            for pos in "nvar":
                for offset in self.senses.get((pos, lemma), [])[:3]:
                    for syn in self.synsets[(pos, offset)][1]:
                        syn = syn.lower()
                        if not syn.isalpha() or syn == lemma:
                            continue
                        if suffix == "s":
                            syn += "es" if re.search(r"(s|x|z|ch|sh)$", syn) else "s"
                        elif suffix == "ed":
                            syn += "d" if syn.endswith("e") else "ed"
                        elif suffix == "ing":
                            syn = (syn[:-1] if syn.endswith("e") else syn) + "ing"
                        if snd.known(syn) and syn != w and syn not in out:
                            out.append(syn)
        return out


def match_case(new: str, old: str) -> str:
    return new.capitalize() if old[:1].isupper() else new


def replace_span(line: str, span: tuple[int, int], new: str) -> str:
    return line[:span[0]] + new + line[span[1]:]


def corrupted_rhyme(p: dict, snd: Sounds, syn: Synonyms, rng: random.Random) -> list[str] | None:
    lines = p["lines"]
    for i in rng.sample(sorted(i for g in p["groups"] for i in g), sum(map(len, p["groups"]))):
        m = last_word(lines[i])
        group = next(g for g in p["groups"] if i in g)
        others = [end_word(lines[j]) for j in group if j != i]
        target = snd.syllables(m.group(0))
        options = [s for s in syn(m.group(0), snd) if abs(snd.syllables(s) - target) <= 1
                   and not snd.rhymes(s, m.group(0)) and not any(snd.rhymes(s, o) for o in others)]
        if options:
            new = list(lines)
            new[i] = replace_span(lines[i], m.span(), match_case(rng.choice(options), m.group(0)))
            return new
    return None


def shuffled_rhyming_words(p: dict, rng: random.Random) -> list[str] | None:
    lines = p["lines"]
    pairs = [(i, j) for g in p["groups"] for a, i in enumerate(g) for j in g[a + 1:]
             if norm_word(end_word(lines[i])) != norm_word(end_word(lines[j]))]
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    mi, mj = last_word(lines[i]), last_word(lines[j])
    new = list(lines)
    new[i] = replace_span(lines[i], mi.span(), mj.group(0))
    new[j] = replace_span(lines[j], mj.span(), mi.group(0))
    return new


def shuffled_rhyming_lines(p: dict, rng: random.Random) -> list[str] | None:
    lines = p["lines"]
    pairs = [(i, j) for g in p["groups"] for a, i in enumerate(g) for j in g[a + 1:] if norm_line(lines[i]) != norm_line(lines[j])]
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    new = list(lines)
    new[i], new[j] = lines[j], lines[i]
    return new


def replaced_last_line(p: dict, donors: list[str], snd: Sounds, rng: random.Random, rhyming: bool) -> list[str] | None:
    lines = p["lines"]
    last, length = end_word(lines[-1]), snd.syllables(lines[-1])
    pool = [x for x in donors if snd.rhymes(end_word(x), last, False) == rhyming and abs(snd.syllables(x) - length) <= 2
            and norm_word(end_word(x)) != norm_word(last)]
    if not pool:
        return None
    return lines[:-1] + [rng.choice(pool)]


def delete_words(p: dict, k: int, which: str, rng: random.Random, tidy: bool) -> list[str] | None:
    """Delete k words: any, the rhyming line-end words, or the others."""
    lines = p["lines"]
    rhyming = {i for g in p["groups"] for i in g}
    spans = []
    for i, line in enumerate(lines):
        found = list(WORD.finditer(line))
        for n, m in enumerate(found):
            eol = n == len(found) - 1 and i in rhyming
            if which == "any" or (which == "rhyming") == eol:
                spans.append((i, m.span()))
    if len(spans) < k:
        return None
    new = list(lines)
    for i, (a, b) in sorted(rng.sample(spans, k), key=lambda s: (s[0], -s[1][0])):
        new[i] = new[i][:a] + new[i][b:]
    new = [re.sub(r" {2,}", " ", x) for x in new]
    if tidy:
        new = [re.sub(r"\s+([,.;:!?])", r"\1", x).strip() for x in new]
    return new if new != lines else None


def insert_words(p: dict, k: int, rng: random.Random) -> list[str] | None:
    """Insert k of the poem's own words before k words that do not start a line (breaks meter and grammar)."""
    lines = p["lines"]
    found = [(i, m) for i, line in enumerate(lines) for m in WORD.finditer(line)]
    spots = [(i, m.start()) for n, (i, m) in enumerate(found) if n and found[n - 1][0] == i]
    if len(spots) < k:
        return None
    new = list(lines)
    for i, at in sorted(rng.sample(spots, k), key=lambda s: (s[0], -s[1])):
        _, m = rng.choice(found)
        word = m.group(0) if m.group(0) == "I" or m.start() > 0 else m.group(0).lower()
        new[i] = new[i][:at] + word + " " + new[i][at:]
    return new


def perturb(variant: str, p: dict, ctx: dict, rng: random.Random) -> list[str] | None:
    snd = ctx["sounds"]
    if variant == "corrupted rhyme":
        return corrupted_rhyme(p, snd, ctx["synonyms"], rng)
    if variant == "shuffled rhyming words":
        return shuffled_rhyming_words(p, rng)
    if variant == "shuffled rhyming lines":
        return shuffled_rhyming_lines(p, rng)
    if variant.startswith("replaced last line"):
        return replaced_last_line(p, ctx["donors"], snd, rng, rhyming="rhyming" in variant)
    k = int(variant.split()[1])
    if variant.startswith("insert"):
        return insert_words(p, k, rng)
    which = "rhyming" if "rhyming EOL" in variant else "other" if "nonrhyming" in variant else "any"
    return delete_words(p, k, which, rng, tidy=rng.random() < .5)


# --- rows ---------------------------------------------------------------------------------------------------
def render(lines: list[str], slashes: bool) -> str:
    return (" / " if slashes else "\n").join(lines)


def state_for(p: dict, rng: random.Random):
    r = rng.random()
    if r < .35:
        return ""
    if r < .6:
        return {"form": p["form"]}
    if r < .8 and p.get("poet"):
        return {"form": p["form"], "poet": p["poet"], "collection": p["collection"]}
    return {"form": p["form"], "note": rng.choice(["One version below is the text as published; the other was edited.",
                                                    "The versions differ by one small edit.",
                                                    "Public-domain verse; one copy was altered."])}


def rows(poems: list[dict], ctx: dict, stats: dict):
    rng = random.Random(SEED + 1)
    for p in poems:
        if p["form"] == "limerick":
            variants = [v for v in VARIANTS for _ in range(LIMERICK_INSERTIONS if v.startswith("insert") else 1)]
        else:
            variants = rng.sample(VARIANTS, len(VARIANTS))
        made, seen = 0, set()
        for variant in variants:
            if p["form"] == "quatrain" and made >= QUATRAIN_VARIANTS:
                break
            altered = perturb(variant, p, ctx, rng)
            if altered is None or [norm_line(x) for x in altered] == [norm_line(x) for x in p["lines"]] or tuple(altered) in seen:
                stats["not_applicable"][variant] += 1
                continue
            seen.add(tuple(altered))
            versions = [p["lines"], altered]
            if rng.random() < THREE_WAY:
                other = rng.choice([v for v in VARIANTS if v != variant])
                extra = perturb(other, p, ctx, rng)
                if extra is not None and extra not in versions:
                    versions.append(extra)
            reversed_ = len(versions) == 2 and rng.random() < REVERSED
            order = rng.sample(range(len(versions)), len(versions))
            keys = rng.choice(KEYS)[:len(versions)]
            slashes = rng.random() < SLASHES
            options = {keys[n]: render(versions[o], slashes) for n, o in enumerate(order)}
            gold = order.index(1 if reversed_ else 0)
            form = p["form"] if p["form"] == "limerick" else rng.choice(["poem", "stanza", "verse"])
            ask = rng.choice(ALTERED if reversed_ else ORIGINAL).format(form=form)
            state = state_for(p, rng)
            question, target = common.choice(state, ask, options, gold)
            stats["rows"][f"{p['form']}:{variant}"] += 1
            stats["three_way"] += len(versions) == 3
            stats["reversed"] += reversed_
            if not reversed_:  # length shortcut: is the original the longest version?
                stats["original_longest"] += len(render(versions[0], False)) >= max(len(render(v, False)) for v in versions)
            made += 1
            yield common.example(NAME, f"{p['key']}:{variant.replace(' ', '-')}:{made}", "poetry-minimal-pairs", state,
                                 question, target, "train")


def build(stats: dict):
    snd = Sounds(CMUDICT)
    poems = book_limericks(snd, stats) + corpus_windows(snd, stats)
    poems = deduplicate(poems, bpomp_lines(), stats)
    rng = random.Random(SEED + 2)
    donors, subjects = [], []
    for p in poems:  # limericks are scarce, so donors are quatrains only
        if p["form"] == "quatrain" and int(p["key"], 16) % 1000 < DONORS * 1000:
            donors.append(p["lines"][-1])
        else:
            subjects.append(p)
    quatrains = [p for p in subjects if p["form"] == "quatrain"]
    stats["quatrains_available"] = len(quatrains)
    keep = {id(p) for p in rng.sample(quatrains, min(QUATRAINS, len(quatrains)))}
    subjects = [p for p in subjects if p["form"] == "limerick" or id(p) in keep]
    stats["poems"] = dict(collections.Counter(f"{p['form']}:{p['origin'].split(':')[0]}" for p in subjects))
    stats["donor_lines"] = len(donors)
    ctx = {"sounds": snd, "synonyms": Synonyms(), "donors": donors}
    rng.shuffle(subjects)
    yield from rows(subjects, ctx, stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download the pinned upstream files")
    args = p.parse_args()
    if args.fetch:
        RAW.mkdir(parents=True, exist_ok=True)
        local, revision = common.fetch(POETRY_REPO, ["README.md", "data/*"], revision=POETRY_REVISION)
        records = [{"name": "gutenberg_poetry", "repo": POETRY_REPO, "revision": revision, "path": str(local)}]
        records += [fetch_spec(name, spec, RAW) for name, spec in FETCH.items()]
        for r in records:
            if "url" in r:
                r["sha256"] = hashlib.sha256(open(r["path"], "rb").read()).hexdigest()
            print(json.dumps(r), flush=True)
    else:
        stats = {"book_limericks": {}, "corpus_windows_seen": collections.Counter(), "dropped_bpomp_overlap": 0,
                 "dropped_duplicate": collections.Counter(), "not_applicable": collections.Counter(),
                 "rows": collections.Counter(), "three_way": 0, "reversed": 0, "original_longest": 0}
        meta = {"upstream": {"gutenberg_poetry_corpus": f"{POETRY_REPO}@{POETRY_REVISION[:12]} (CC0)",
                             "cmudict": f"github cmusphinx/cmudict@{CMUDICT_COMMIT[:12]} (BSD-style)",
                             "wordnet": "nltk_data corpora/wordnet.zip (WordNet 3.0, Princeton licence; phase6_lang pin)",
                             "gutenberg_books": {f"pg{n}": f"{t} ({a}), sha256 {h[:12]}" for n, (t, a, h) in BOOKS.items()}},
                "split": "train (generated)", "licence": "public-domain poems; CC0 corpus; generated perturbations (ours)",
                "label_origin": "programmatic: the unaltered poem is the gold by construction",
                "never_read_for_training": "BPoMP release and OEDILF; BPoMP originals only drop overlapping poems",
                "stats": stats}
        common.write(NAME, build(stats), meta)
