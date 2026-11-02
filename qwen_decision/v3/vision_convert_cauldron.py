"""The Cauldron converters (2026-11-01): HuggingFaceM4/the_cauldron train subsets in the board's multiple-choice
form. See vision_mix for the item fields. Converter names are cauldron/<subset>.

A Cauldron row is an image set with one or more QA turns; each turn can become an item. Rows come from row groups
drawn at random over all shards (at most `per_unit` items a row group, `per_row` a row), and images are decoded
only for the rows used. Cauldron's answer-format lines ("Answer briefly.", "Answer with the letter.", ...) are
stripped from questions, and its added trailing period from answers.

Options, by answer kind:
  mc      the source's own options (parsed from the user text), shuffled unless they name letters or refer to each
          other ("both A and B", "all of the above")
  yes/no  ["Yes", "No"], ordered, balanced per subset by dropping surplus turns of the larger class
  number  gold plus 3 from kit.numeric_distractors (same-image numbers first, then nearby values; for small
          integers the 5 nearest, like the board's count options), with the gold's rank among the 4 balanced by
          kit.pick_by_rank. Not ordered: the board shuffles numbers too, and sorted counts would put a gold 0 at A
  text    gold plus 3 distractors: same-image answers of the same class first, then same-subset answers to questions
          that start alike, then any same-subset answer of the class (class: has digits or not, plus the
          question's first words for VQA-style sets, or a subset's own, e.g. MapQA ranges/lists/states). Answers
          equal to the gold, or whose words contain or are contained in the gold's, are dropped. Document sets
          (docvqa, infographic_vqa, textvqa, st_vqa, ocrvqa) treat numbers as text, as the board's InfographicVQA
          does, since they are mostly ids, dates and amounts
Options of unordered items are shuffled here too (the trainer shuffles again).

  spatial      vsr (yes/no, human), clevr (yes/no, counts, and attribute questions whose options are the CLEVR
               values of that attribute, program), visual7w (4-way mc, human)
  counting     tallyqa (numbers, human)
  multi-image  nlvr2 (yes/no on two images, human); spot_the_diff (which description says what differs between
               two surveillance frames; distractors are other pairs' descriptions sharing no content word with the
               gold, since these scenes' descriptions are alike and a near one could also be true; "no
               differences" rows dropped; human, 11-02)
  general      vqav2 (yes/no, numbers, text; human), okvqa (text; human; images are vqav2's COCO train2014 files,
               since this subset carries only paths), aokvqa (4-way mc parsed from "Options: a, b, c, d."; the
               rationale is dropped; human), cocoqa (text, "?" added; answers from human captions, program)
  chart        plotqa (yes/no, numbers with at most 2 decimals, text; program), figureqa (yes/no, program), dvqa
               (yes/no, numbers, chart labels; program), mapqa (yes/no, ranges, state lists, states; program)
  document     docvqa, infographic_vqa (options lowercased as on the board), textvqa, st_vqa (human); ocrvqa (book
               metadata, program); visualmrc (human sentences; only items where the gold does not share the most
               words with the question, since its answers restate their questions); tabmwp (the single number in
               a solution's last line for How/What questions, human); tat_qa (the row's text context is given with
               every question, human); hitab (numbers only), finqa, multihiertt (human; table-parse turns dropped)
  web          screen2words (which summary describes the screen; distractors are other screens' summaries sharing
               no content word with the gold, human)
  meme         hateful_memes (the board's "Is this meme hateful?", meme text not shown, human)
  science      ai2d, tqa, scienceqa (mc; ScienceQA lectures dropped, hints kept), iconqa (mc and free numbers/text);
               human
  math         raven (A-H drawn in the image, or yes/no on puzzle + candidate; program), geomverse (the solution's
               final number, program), intergps (mc, human), clevr_math (numbers; images are clevr's CLEVR train
               files; program)
Skipped: robut_sqa (8,514 rows are about 1,850 SQA sequences
repeated with paraphrased questions and perturbed tables, the RobuT/LETA augmentation, so questions may be
model-written).
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from decision_index.vision import convert as kit

from qwen_decision.v3.vision_mix import ROOT, item, norm

CAULDRON = ROOT / "data/raw/HuggingFaceM4__the_cauldron@847a98a779b1"
MAX_TEXT = 2500
POOL_MAX = 5000      # answers kept per distractor pool
POOL_TURNS = 20000   # turns read at least, so small subsets get full pools
FORMAT_LINES = (
    "Answer briefly.", "Be succinct.", "Quick response, please.", "Give a very brief answer.", "Short answer required.",
    "Write a very short answer.", "Your response must be concise.", "Ensure brevity in your answer.",
    "Make the answer very short.", "Keep it short and to the point.", "Provide a short and direct response.",
    "Your answer should be compact.", "Offer a very short reply.", "Your answer should be very brief.",
    "Provide a succinct answer.", "Offer a terse response.", "Concise answer only.", "Keep it brief.",
    "Answer yes or no.", "Answer with the letter.", "Yes or no.", "Answer by yes or no.",
    "Answer the question using a single word or phrase.")
FORMAT = sorted({x.lower() for x in FORMAT_LINES}, key=len, reverse=True)
FORMAT_HEAD = ("answer by yes or no.", "answer yes or no.")  # RAVEN puts it first
ABBREV = re.compile(r"(?:^|[\s.])(?:[A-Za-z]|inc|co|ltd|jr|sr|st|corp|etc|no|vs|dr|mr|mrs|ms)\.$", re.IGNORECASE)
CHOICE = re.compile(r"^([A-J])\. ?(.*)$")
NUMBER = re.compile(r"^(\$)?(-?\d[\d,]*(?:\.\d+)?%?)$")
LETTER_REF = re.compile(r"\b(?:all|none|both|neither) of the (?:above|below)\b|^(?:both |either |neither )?[A-H]"
                        r"(?:\s*(?:,|and|&|or)\s*[A-H])+$", re.IGNORECASE)
NO_ANSWER = {"unanswerable", "answering does not require reading text in the image"}  # TextVQA's markers
YES_NO = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}
STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "with", "page", "screen", "app", "application",
        "showing", "shows", "show", "display", "displaying", "displays", "displayed", "option", "options", "which",
        "is", "this", "that", "by", "from", "at", "as", "its", "it", "or", "be", "are", "screenshot", "view", "list"}
CLEVR_VALUES = {"shape": ["cube", "sphere", "cylinder"], "material": ["rubber", "metal"], "size": ["small", "large"],
                "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]}
CLEVR_OF = {v: k for k, vs in CLEVR_VALUES.items() for v in vs}


# --- text helpers ------------------------------------------------------------------------------------------------

def question(text: str) -> str:
    """The user text without Cauldron's answer-format lines (trailing, or RAVEN's leading one)."""
    q, prev = text.strip(), None
    while q != prev:
        prev, low = q, q.lower()
        for x in FORMAT:
            if low.endswith(x):
                q = q[:-len(x)].strip()
                break
        for x in FORMAT_HEAD:
            if q.lower().startswith(x):
                q = q[len(x):].strip()
    return q


def answer(text: str) -> str:
    """The assistant text as a short answer: after the last "Answer:", first line, Cauldron's period removed."""
    a = text.strip()
    if "Answer:" in a:
        a = a.rsplit("Answer:", 1)[1]
    a = a.strip().split("\n")[0].strip()
    if a.endswith(".") and not a.endswith("..") and not ABBREV.search(a):
        a = a[:-1].strip()
    return a


def yes_no(a: str) -> str | None:
    return YES_NO.get(a.strip().rstrip(".").lower())


def number(a: str, max_dec: int = 5) -> str | None:
    """A numeric answer in kit.NUM form (optional "$" kept, thousands commas and a ".0" tail dropped), else None."""
    m = NUMBER.match(a.strip())
    if not m:
        return None
    s = m[2]
    if "," in s:
        if not re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", s):
            return None
        s = s.replace(",", "")
    s = re.sub(r"^(-?\d+)\.0(%?)$", r"\1\2", s)
    if not kit.NUM.match(s) or ("." in s and len(s.rstrip("%").split(".")[1]) > max_dec):
        return None
    return (m[1] or "") + s


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9%]+", s.lower()))


def qkey(q: str, n: int = 4) -> str:
    return " ".join(re.findall(r"[a-z]+", q.lower())[:n])


def turn(q: str, kind: str, a=None, sub: str = "", **kw) -> dict:
    """One parsed turn. kind: mc (options, gold), yesno (a = yes|no), num (a), text (a, cls, keys)."""
    return {"q": q, "kind": kind, "a": a, "sub": sub or kind, **kw}


def text_turn(q: str, a: str, sub: str = "text", cls: str | None = None, key: str | None = None, **kw) -> dict:
    cls = cls or ("digit" if re.search(r"\d", a) else "plain")
    return turn(q, "text", a, sub, cls=cls, keys=[f"{cls}|{key if key is not None else qkey(q)}", f"{cls}|*"], **kw)


def auto(q: str, a: str, max_dec: int = 5, text: bool = True, digits_as_text: bool = False, qtype: int = 0,
         **kw) -> dict | None:
    """yes/no, number or text, by the answer. digits_as_text: numbers are text answers of the "digit" class (the
    board's InfographicVQA way, for documents where numbers are often ids, dates and amounts). qtype: the class
    also holds the question's first qtype words, so distractors answer the same kind of question ("what color")."""
    if not q or not a or a.lower() in NO_ANSWER:
        return None
    if yes_no(a):
        return turn(q, "yesno", yes_no(a), "yes/no")
    n = None if digits_as_text else number(a, max_dec)
    if n is not None:
        return turn(q, "num", n, "number", key=qkey(q))
    if not digits_as_text and (NUMBER.match(a.strip()) or re.fullmatch(r"-?[\d.]+e[-+]?\d+", a.strip())):
        return None  # a number we cannot format (too many decimals, scientific notation)
    if qtype:
        kw["cls"] = ("digit" if re.search(r"\d", a) else "plain") + ":" + qkey(q, qtype)
    return text_turn(q, a, **kw) if text else None


def mc(user: str, assistant: str, sub: str = "mc") -> dict | None:
    """ "...Question: q\\n[Hint: h\\n]Choices:\\nA. x\\nB. y\\nAnswer with the letter." -> mc turn (yes/no if the
    options are yes and no)."""
    lines = question(user).split("\n")
    if "Choices:" not in lines:
        return None
    i = lines.index("Choices:")
    head = re.split(r"(?:^|\n)Question: ", "\n".join(lines[:i]))[-1].strip()
    head = "\n".join(x for x in head.split("\n")
                     if not re.fullmatch(r"Hint: (?:Please )?(?:select|choose) the (?:best|better|correct) answer\.?",
                                         x.strip(), re.IGNORECASE))
    letters, opts = [], []
    for x in lines[i + 1:]:
        m = CHOICE.match(x.strip())
        if not m:
            return None
        letters.append(m[1])
        opts.append(m[2].strip())
    m = re.search(r"\b([A-J])\b", assistant.replace("Answer:", ""))
    if not head or letters != list(kit.LETTERS[:len(letters)]) or not m or m[1] not in letters:
        return None
    if all(o.endswith(".") for o in opts):
        opts = [o[:-1].strip() for o in opts]
    gold = letters.index(m[1])
    if len(opts) == 2 and {o.lower() for o in opts} == {"yes", "no"}:
        return turn(head, "yesno", opts[gold].lower(), "yes/no")
    ordered = any(LETTER_REF.search(o) for o in opts) or all(re.fullmatch(r"[A-Z]", o) for o in opts)
    return turn(head, "mc", sub=sub, options=opts, gold=gold, ordered=ordered)


# --- options -----------------------------------------------------------------------------------------------------

def number_options(gold: str, near: list[str], pool: list[str], rng: random.Random, counts: Counter):
    """Gold plus 3 numbers, ascending; the gold's rank is balanced by pick_by_rank over the candidates."""
    pre = "$" if gold.startswith("$") else ""
    g = gold[len(pre):]

    def bare(xs):
        return [x[len(pre):] for x in xs if x.startswith(pre) and x[len(pre):].endswith("%") == g.endswith("%")]

    cands = kit.numeric_distractors(g, bare(near), rng)
    for _ in range(4):  # each call gives 3 of the kit's nearby values at random
        cands += kit.numeric_distractors(g, [], rng)
    if len(set(cands)) < 6:
        cands += kit.numeric_distractors(g, bare(rng.sample(pool, min(len(pool), 30))), rng)
    gv = float(g.rstrip("%"))
    seen_near = set(bare(near))
    vals = {}
    for c in cands:
        v = float(c.rstrip("%"))
        # the kit's gold +- 10^-dec fallbacks could be rounding variants of the gold: not for larger numbers
        close = c not in seen_near and abs(v - gv) < 0.02 * abs(gv)
        if abs(v - gv) > 1e-9 and v not in vals and c.endswith("%") == g.endswith("%") and not close:
            vals[v] = c
    if re.fullmatch(r"-?\d+%?", g) and (abs(gv) < 30 or re.fullmatch(r"(?:19|20)\d\d", g)):
        # small integers (counts) and years: the 5 nearest, as the board's count options are
        pct = "%" if g.endswith("%") else ""
        vals |= {gv + d: f"{int(gv + d)}{pct}" for d in (-4, -3, -2, -1, 1, 2, 3, 4) if gv < 0 or gv + d >= 0}
        vals = dict(sorted(vals.items(), key=lambda kv: (abs(kv[0] - gv), rng.random()))[:5])
    picked = kit.pick_by_rank(gv, sorted(vals.items()), 3, rng, counts)
    if not picked or len(picked) < 3:
        return None
    opts = [s for _, s in sorted([(gv, g)] + picked)]
    return [pre + s for s in opts], opts.index(g)


def text_distractors(gold: str, near: list[str], pools: list[list[str]], rng: random.Random, k: int = 3,
                     far: bool = False) -> list:
    """far: also drop answers sharing any content word with the gold (for summaries)."""
    g = words(gold)
    out, seen = [], {norm(gold)}
    for group in [near] + [rng.sample(p, min(len(p), 60)) for p in pools]:
        group = list(dict.fromkeys(group))
        rng.shuffle(group)
        for a in group:
            w = words(a)
            if norm(a) in seen or not w or w <= g or g <= w or (far and (w & g) - STOP):
                continue
            out.append(a)
            seen.add(norm(a))
            if len(out) == k:
                return out
    return out


# --- driver ------------------------------------------------------------------------------------------------------

class Lookup:
    """Image bytes by file name from another subset (okvqa and clevr_math carry only paths)."""

    def __init__(self, subset: str):
        self.where = {}
        for f in sorted((CAULDRON / subset).glob("*.parquet")):
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                rows = pf.read_row_group(g, columns=["images.list.element.path"]).column(0).to_pylist()
                for i, ims in enumerate(rows):
                    for j, im in enumerate(ims):
                        self.where.setdefault(Path(im["path"]).name, (f, g, i, j))

    def fetch(self, names: list[str]) -> dict[str, bytes]:
        by_group = defaultdict(list)
        for name in names:
            if name in self.where:
                f, g, i, j = self.where[name]
                by_group[(f, g)].append((name, i, j))
        out = {}
        for (f, g), want in sorted(by_group.items()):
            col = pq.ParquetFile(f).read_row_group(g, columns=["images"]).column(0)
            for name, i, j in want:
                out[name] = col[i].as_py()[j]["bytes"]
        return out


class Spec:
    """parse(texts, rng) -> a parsed turn or None per turn; per_row: items per row (default: enough to reach cap
    from one pass); spare: candidates gathered per cap; keep: an extra item filter."""

    def __init__(self, subset: str, parse, label: str, per_row: int | None = None, max_images: int = 5,
                 lookup: str | None = None, spare: float = 1.5, keep=None):
        self.subset, self.parse, self.label, self.per_row = subset, parse, label, per_row
        self.max_images, self.lookup, self.spare, self.keep = max_images, lookup, spare, keep


def convert(spec: Spec, saver, cap: int, rng: random.Random):
    files = {f: pq.ParquetFile(f) for f in sorted((CAULDRON / spec.subset).glob("*.parquet"))}
    units = [(f, g, pf.metadata.row_group(g).num_rows) for f, pf in files.items() for g in range(pf.num_row_groups)]
    rng.shuffle(units)
    total = sum(n for _, _, n in units)
    per_row = spec.per_row or max(1, math.ceil(1.3 * cap / total))
    per_unit = max(8, math.ceil(spec.spare * cap / len(units)))
    pools = defaultdict(list)
    cands, seen = [], 0  # cands: (unit, row, turn index, the row's parsed turns)
    for u, (f, g, _) in enumerate(units):
        rows = files[f].read_row_group(g, columns=["texts"]).column(0).to_pylist()
        seen += sum(len(r) for r in rows)
        order = list(range(len(rows)))
        rng.shuffle(order)
        taken = 0
        for r in order:
            parsed = spec.parse(rows[r], rng)
            for t in parsed:
                if t and t["kind"] in ("num", "text"):
                    for key in t.get("keys", [t.get("key")]):
                        if len(pools[(t["kind"], key)]) < POOL_MAX:
                            pools[(t["kind"], key)].append(t["a"])
            if taken < per_unit:
                live = [i for i, t in enumerate(parsed) if t]
                for i in rng.sample(live, min(per_row, len(live), per_unit - taken)):
                    cands.append((u, r, i, parsed))
                    taken += 1
        if len(cands) >= spec.spare * cap and seen >= POOL_TURNS:
            break
    yn, ranks, chosen = Counter(), Counter(), []
    slack = max(4, cap // 200)
    for u, r, i, parsed in cands:
        t = parsed[i]
        ordered = t.get("ordered", False)
        if t["kind"] == "mc":
            opts, gold = t["options"], t["gold"]
        elif t["kind"] == "yesno":
            gold = 0 if t["a"] == "yes" else 1
            if yn[gold] > yn[1 - gold] + slack:
                continue
            opts, ordered = ["Yes", "No"], True
        elif t["kind"] == "num":
            near = [o["a"] for o in parsed if o and o is not t and o["kind"] == "num"]
            got = number_options(t["a"], near, pools[("num", t["key"])], rng, ranks)
            if not got:
                continue
            opts, gold = got
        else:
            near = [o["a"] for o in parsed if o and o is not t and o["kind"] == "text" and o["cls"] == t["cls"]]
            ds = text_distractors(t["a"], near + t.get("extra", []), [pools[("text", k)] for k in t["keys"]], rng,
                                  far=t.get("far", False))
            if len(ds) < t.get("min_ds", 3):
                continue
            opts, gold = [t["a"]] + ds, 0
        if len(t["q"]) + sum(len(o) for o in opts) > MAX_TEXT:
            continue
        if not ordered:  # the trainer shuffles too; this keeps the pool's gold positions even for other readers
            order = rng.sample(range(len(opts)), len(opts))
            opts, gold = [opts[k] for k in order], order.index(gold)
        f, g, _ = units[u]
        it = item(f"cauldron/{spec.subset}", f"{f.name.split('-')[1]}-{g}-{r}-{i}", ["?"], t["q"], opts, gold,
                  t["sub"], spec.label, ordered=ordered)
        if it is None or (spec.keep and not spec.keep(it)):
            continue
        if t["kind"] == "yesno":
            yn[gold] += 1
        chosen.append((u, r, it))
        if len(chosen) >= cap:
            break
    by_unit = defaultdict(list)
    for u, r, it in chosen:
        by_unit[u].append((r, it))
    found = {}
    if spec.lookup:  # this subset's rows carry only image paths: fetch them all at once from the other subset
        names = []
        for u in by_unit:
            f, g, _ = units[u]
            col = files[f].read_row_group(g, columns=["images.list.element.path"]).column(0)
            names += [Path(im["path"]).name for r, _ in by_unit[u] for im in col[r].as_py()]
        found = Lookup(spec.lookup).fetch(names)
    for u in sorted(by_unit, key=lambda u: (units[u][0], units[u][1])):
        f, g, _ = units[u]
        if spec.lookup:
            col = files[f].read_row_group(g, columns=["images.list.element.path"]).column(0)
        else:
            col = files[f].read_row_group(g, columns=["images"]).column(0)
        rows = {r: col[r].as_py() for r in {r for r, _ in by_unit[u]}}
        if spec.lookup:
            rows = {r: [{"bytes": found.get(Path(im["path"]).name)} for im in ims] for r, ims in rows.items()}
        saved = {}
        for r, it in by_unit[u]:
            ims = rows[r]
            if not 1 <= len(ims) <= spec.max_images or any(not im["bytes"] for im in ims):
                yield None
                continue
            if r not in saved:
                saved[r] = [saver.bytes(im["bytes"]) for im in ims]
            yield it | {"images": saved[r]}


# --- per-subset parsing: fn(texts, rng) -> one parsed turn or None per turn ----------------------------------------

def each(fn):
    def parse(texts, rng):
        return [fn(question(t["user"]), t["assistant"], rng) for t in texts]
    return parse


def p_yesno(sub):
    def fn(q, a, rng):
        v = yes_no(answer(a))
        return turn(q, "yesno", v, sub) if q and v else None
    return each(fn)


def p_mc(sub="mc"):
    def parse(texts, rng):
        return [mc(t["user"], t["assistant"], sub) for t in texts]
    return parse


def p_auto(max_dec=5, text=True, digits_as_text=False, lower=False, qtype=0):
    def fn(q, a, rng):
        a = answer(a)
        return auto(q, a.lower() if lower and not yes_no(a) else a, max_dec, text, digits_as_text, qtype)
    return each(fn)


def p_clevr(texts, rng):
    out = []
    for t in texts:
        q, a = question(t["user"]), answer(t["assistant"])
        v = a.lower()
        if v in CLEVR_OF:
            values = CLEVR_VALUES[CLEVR_OF[v]]
            opts = [v] + rng.sample([x for x in values if x != v], min(3, len(values) - 1))
            out.append(turn(q, "mc", sub=CLEVR_OF[v], options=opts, gold=0))
        else:
            out.append(auto(q, a, text=False))
    return out


def p_coco(texts, rng):
    out = []
    for t in texts:
        q = question(t["user"])
        q = q if q[-1:] in "?." else q + "?"
        out.append(auto(q, answer(t["assistant"]), qtype=2))
    return out


def p_aokvqa(texts, rng):
    out = []
    for t in texts:
        lines = t["user"].strip().split("\n")
        m = re.fullmatch(r"Options: (.*)\.", lines[-1].strip())
        opts = m[1].split(", ") if m else []
        if len(opts) == 4 and opts[0][1:2].islower() and sum(o[:1].islower() for o in opts[1:]) >= 2:
            opts[0] = opts[0][0].lower() + opts[0][1:]
        a = answer(t["assistant"]).lower()
        hits = [i for i, o in enumerate(opts) if o.lower() == a]
        ok = len(opts) == 4 and len(hits) == 1 and len({o.lower() for o in opts}) == 4
        out.append(turn(lines[0].strip(), "mc", sub="mc", options=opts, gold=hits[0]) if ok else None)
    return out


def p_dvqa(texts, rng):
    out = []
    for t in texts:
        q, a = question(t["user"]), answer(t["assistant"])
        out.append(auto(q, a if number(a) or yes_no(a) else a.lower(), key=""))
    return out


def p_mapqa(texts, rng):
    ranges = []
    for t in texts:
        ranges += re.findall(r"in the range (\S+?-\S+?)\s*\?", t["user"])
    out, parsed = [], []
    for t in texts:
        q, a = question(t["user"]), answer(t["assistant"])
        if yes_no(a) or number(a):
            out.append(auto(q, a))
            continue
        if re.match(r"What is the (?:value|highest value|lowest value)\b", q) or re.fullmatch(r"\S*\d\S*-\S*\d\S*", a):
            out.append(turn(q, "text", a, "range", cls="range", keys=[], extra=ranges, min_ds=2))
        elif ", " in a:
            out.append(text_turn(q, a, "states", cls="list", key=""))
        else:
            out.append(text_turn(q, a, "state", cls="state", key=""))
        parsed.append(out[-1])
    states = list({s for p in parsed if p["cls"] == "list" for s in p["a"].split(", ")})
    for p in parsed:
        if p["cls"] == "state":
            p["extra"] = states
    return out


def p_tabmwp(texts, rng):
    out = []
    for t in texts:
        q, sol = t["user"].strip(), t["assistant"].strip()
        asked = re.findall(r"[^.?!]*\?", q)
        last = sol.split("\n")[-1]
        nums = re.findall(r"\$?\d[\d,]*(?:\.\d+)?", last)
        n = number(nums[0].rstrip(",")) if len(nums) == 1 and "/" not in last and ":" not in last else None
        ok = asked and re.match(r"\s*(How|What)\b", asked[-1]) and not re.match(r"\s*What time", asked[-1])
        out.append(turn(q, "num", n, "number", key=qkey(asked[-1])) if ok and n else None)
    return out


def p_tatqa(texts, rng):
    out, context = [], ""
    for k, t in enumerate(texts):
        lines = [x for x in t["user"].strip().split("\n") if x.strip()]
        scale = [x for x in lines if x.startswith("Answer scale should be:")]
        lines = [x for x in lines if not x.startswith("Answer scale should be:")]
        if k == 0:
            context = "\n".join(lines[:-1])
        q = "\n".join(([context] if context else []) + lines[-1:] + scale)
        out.append(auto(q, answer(t["assistant"]), key=qkey(lines[-1])) if lines else None)
    return out


def p_table(texts, rng):
    out = []
    for t in texts:
        a = t["assistant"].strip()
        out.append(None if a.startswith("{") else auto(question(t["user"]), answer(a)))
    return out


def p_table_numbers(texts, rng):
    """HiTab's text answers mostly have no same-table alternatives (one or two turns a table): numbers only."""
    return [t if t and t["kind"] != "text" else None for t in p_table(texts, rng)]


def p_screen(texts, rng):
    return [text_turn("Which summary best describes this screen?", answer(t["assistant"]), "summary", key="",
                      far=True) for t in texts[:1]]


SPOT_Q = ("The first image is the image on the left, the second image is the image on the right. Which description "
          "says what is different between the two images?")


def p_spot(texts, rng):
    return [None if "no difference" in answer(t["assistant"]).lower()
            else text_turn(SPOT_Q, answer(t["assistant"]), "difference", key="", far=True) for t in texts[:1]]


def p_memes(texts, rng):
    return [turn("Is this meme hateful?", "yesno", yes_no(answer(t["assistant"])), "binary")
            if yes_no(answer(t["assistant"])) else None for t in texts]


def p_iconqa(texts, rng):
    out = []
    for t in texts:
        if t["user"].startswith("Question:"):
            out.append(mc(t["user"], t["assistant"]))
        else:
            out.append(auto(question(t["user"]), answer(t["assistant"]), key=qkey(t["user"], 6)))
    return out


def p_raven(texts, rng):
    out = []
    for t in texts:
        a = t["assistant"].strip()
        if re.fullmatch(r"[A-H]", a):
            out.append(turn(question(t["user"]), "mc", sub="choose", options=list("ABCDEFGH"),
                            gold="ABCDEFGH".index(a), ordered=True))
        elif yes_no(a):
            out.append(turn("The first image is a puzzle with a missing panel, the second image is a candidate "
                            "panel. Does the candidate correctly complete the sequence?", "yesno", yes_no(a), "verify"))
        else:
            out.append(None)
    return out


def p_geomverse(texts, rng):
    out = []
    for t in texts:
        m = re.search(r"final answer is (-?\d+(?:\.\d+)?)\.?\s*$", t["assistant"].strip())
        n = number(m[1]) if m else None
        out.append(turn(t["user"].strip(), "num", n, "number", key="") if n else None)
    return out


def p_visualmrc(texts, rng):
    return [text_turn(question(t["user"]), answer(t["assistant"]) + ".", "sentence", key="") for t in texts]


def no_echo(it: dict) -> bool:
    """VisualMRC answers restate their question: keep items whose gold is not the option closest to it in words."""
    q = words(it["question"])
    share = [len(words(o) & q) for o in it["options"]]
    return share[it["gold"]] < max(s for i, s in enumerate(share) if i != it["gold"])


SPECS = {
    "vsr": ("spatial", Spec("vsr", p_yesno("relation"), "human")),
    "clevr": ("spatial", Spec("clevr", p_clevr, "program", per_row=2)),
    "visual7w": ("spatial", Spec("visual7w", p_mc(), "human", per_row=2)),
    "tallyqa": ("counting", Spec("tallyqa", p_auto(0, text=False), "human")),
    "nlvr2": ("multi-image", Spec("nlvr2", p_yesno("statement"), "human", per_row=2)),
    "spot_the_diff": ("multi-image", Spec("spot_the_diff", p_spot, "human")),
    "vqav2": ("general", Spec("vqav2", p_auto(qtype=2), "human", per_row=2)),
    "okvqa": ("general", Spec("okvqa", p_auto(qtype=2), "human", lookup="vqav2")),
    "aokvqa": ("general", Spec("aokvqa", p_aokvqa, "human")),
    "cocoqa": ("general", Spec("cocoqa", p_coco, "program")),
    "plotqa": ("chart", Spec("plotqa", p_auto(2), "program")),
    "figureqa": ("chart", Spec("figureqa", p_yesno("yes/no"), "program")),
    "dvqa": ("chart", Spec("dvqa", p_dvqa, "program")),
    "mapqa": ("chart", Spec("mapqa", p_mapqa, "program")),
    "docvqa": ("document", Spec("docvqa", p_auto(digits_as_text=True), "human", per_row=3)),
    "infographic_vqa": ("document", Spec("infographic_vqa", p_auto(digits_as_text=True, lower=True), "human")),
    "textvqa": ("document", Spec("textvqa", p_auto(digits_as_text=True, qtype=2), "human", per_row=2)),
    "st_vqa": ("document", Spec("st_vqa", p_auto(digits_as_text=True, qtype=2), "human")),
    "ocrvqa": ("document", Spec("ocrvqa", p_auto(digits_as_text=True, qtype=4), "program")),
    "visualmrc": ("document", Spec("visualmrc", p_visualmrc, "human", per_row=4, spare=10, keep=no_echo)),
    "tabmwp": ("document", Spec("tabmwp", p_tabmwp, "human", spare=3)),
    "tat_qa": ("document", Spec("tat_qa", p_tatqa, "human")),
    "hitab": ("document", Spec("hitab", p_table_numbers, "human")),
    "finqa": ("document", Spec("finqa", p_table, "human", spare=4)),
    "multihiertt": ("document", Spec("multihiertt", p_table, "human")),
    "screen2words": ("web", Spec("screen2words", p_screen, "human")),
    "hateful_memes": ("meme", Spec("hateful_memes", p_memes, "human", spare=2)),
    "ai2d": ("science", Spec("ai2d", p_mc(), "human")),
    "tqa": ("science", Spec("tqa", p_mc(), "human", per_row=2)),
    "scienceqa": ("science", Spec("scienceqa", p_mc(), "human")),
    "iconqa": ("science", Spec("iconqa", p_iconqa, "human")),
    "raven": ("math", Spec("raven", p_raven, "program")),
    "geomverse": ("math", Spec("geomverse", p_geomverse, "program")),
    "intergps": ("math", Spec("intergps", p_mc(), "human")),
    "clevr_math": ("math", Spec("clevr_math", p_auto(0, text=False), "program", per_row=2, lookup="clevr")),
}


def _converter(spec: Spec):
    def fn(saver, cap, rng):
        return convert(spec, saver, cap, rng)
    return fn


CONVERTERS = {f"cauldron/{name}": (skill, _converter(spec)) for name, (skill, spec) in SPECS.items()}
