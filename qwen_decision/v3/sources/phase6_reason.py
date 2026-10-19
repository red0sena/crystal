"""Phase-6 sources, reading / tables / reasoning (2026-10-19).

  python -m qwen_decision.v3.sources.phase6_reason --fetch          # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase6_reason [name ...]       # build data/phase2/<name>.jsonl

Eleven sources vetted in runs/analysis/phase6-candidates-vetted.json (licence, download, fields), re-checked on the
downloaded files: coreference reading (Quoref), dialogue answerability (QuAC), rule conversations (ShARC), conditional
policy answers (ConditionalQA), entity claims (CREAK), NLI (SNLI), infobox NLI (InfoTABS), table QA
(WikiTableQuestions), word problems (SVAMP) and spatial reasoning (StepGame, SpartQA). Train splits only. Every gold
is a human label or computed by the dataset's generator; distractor options are chosen by the rules in each
builder's docstring. Downloads are only read (zips are read in place, nothing is run). The first segment of every
item id is the grouping unit (passage / page / rule text / document / entity / premise / table / problem / story),
so mix.SOURCES uses _after(1) for all of them.
"""
from __future__ import annotations

import argparse
import ast
import collections
import html
import json
import random
import re
import zipfile

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import fetch_spec, hub_dir

RAW = common.RAW / "phase6"

# name -> ("github", repo, commit, [paths kept]) | ("hub", repo, [allow patterns]) | ("url", url, file)
FETCH = {
    "quoref": ("url", "https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip",
               "quoref-train-dev-v0.1.zip"),
    "quac": ("url", "https://s3.amazonaws.com/my89public/quac/train_v0.2.json", "train_v0.2.json"),
    "sharc": ("url", "https://sharc-data.github.io/data/sharc1-official.zip", "sharc1-official.zip"),
    "conditionalqa": ("github", "haitian-sun/ConditionalQA", "77bd295952daf415548b3244db10880d3d55cfe0",
                      ["LICENSE", "README.md", "v1_0/train.json", "v1_0/documents.json"]),
    "creak": ("github", "yasumasaonoe/creak", "5e892b28b6dad3e87769cfc01f00a6577ca44068",
              ["LICENSE.txt", "LICENSE", "README.md", "data/creak/train.json"]),
    "snli": ("hub", "stanfordnlp/snli", ["README.md", "plain_text/train-*"]),
    "infotabs": ("github", "infotabs/infotabs", "34e7dc0900a0d861ad62be3d7371d8eb0aace4fe",
                 ["LICENSE", "README.md", "data/maindata/infotabs_train.tsv", "data/tables/json/"]),
    "wtq": ("url", "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip",
            "WikiTableQuestions-1.0.2-compact.zip"),
    "wtq-licence": ("github", "ppasupat/WikiTableQuestions", "7d455a5a707b96341ef72aff9428749d443d8aa9", ["LICENSE", "README.md"]),
    "svamp": ("hub", "ChilleD/SVAMP", ["README.md", "data/train-*"]),
    "stepgame": ("hub", "ZhengyanShi/StepGame", ["README.md", "train.jsonl"]),
    # SpatialLM-StepGame (Li, Hogg and Cohn 2024, MIT, doi 10.5518/1468) lists the original StepGame's wrong sentence
    # templates; correct.py is only read (its pattern list is copied below as STEPGAME_WRONG), never run. Its data.zip
    # (bAbI-format files) is not needed.
    "stepgame-fixed-readme": ("url", "https://archive.researchdata.leeds.ac.uk/1321/1/README_Fangjun-etal_2024.txt",
                              "README_Fangjun-etal_2024.txt"),
    "stepgame-fixed-script": ("url", "https://archive.researchdata.leeds.ac.uk/1321/2/correct.py", "correct.py.txt"),
    "spartqa-yn": ("hub", "tasksource/spartqa-yn", ["README.md", "spartqa_YN_train.jsonl"]),
    "spartqa-mchoice": ("hub", "tasksource/spartqa-mchoice", ["README.md", "spartqa_CO_train.jsonl"]),
}


def _n(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def _contains(a: str, b: str) -> bool:
    """Whether the normalised word sequence of one text holds the other's (either way)."""
    a, b = f" {_n(a)} ", f" {_n(b)} "
    return a.strip() and b.strip() and (a in b or b in a)


def _number(text: str) -> float | None:
    m = re.fullmatch(r"\s*[$£€]?\s*([-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+)\s*%?\s*", str(text))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _same_answer(a: str, b: str) -> bool:
    """Equal or nested text, or the same number: b would also be a correct answer when a is."""
    x, y = _number(a), _number(b)
    if x is not None and y is not None:
        return abs(x - y) < 1e-9
    return bool(_contains(a, b))


def _fmt_number(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


def _shuffled(rng: random.Random, gold, others: list) -> tuple[list, int]:
    options = [gold] + list(others)
    rng.shuffle(options)
    return options, options.index(gold)


def _labelled(rng: random.Random, label_sets: list, gold: int) -> tuple[dict, int]:
    """A label set (list of (name, description) in a fixed meaning order) in shuffled order, and the gold's position."""
    labels = rng.choice(label_sets)
    order = list(range(len(labels)))
    rng.shuffle(order)
    return {labels[i][0]: labels[i][1] for i in order}, order.index(gold)


# --- Quoref: which person/entity a coreferring question asks for ---------------------------------------------
QUOREF = "quoref-train"
QUOREF_ZIP = RAW / "quoref/quoref-train-dev-v0.1.zip"
QUOREF_ASK = ["Answer the question about the passage.", "According to the passage, which of these answers the question?",
              "Read the passage and pick the answer to the question."]
_STOP = set("""The A An In On At If When After Before Both One Two Three His Her Their It Its He She They We I You This That These
Those Which What While Since As However Also Then There So But And For To With From During Each Some All Many Most Other Another
Although Though Despite By Of Under Over Upon Into Through Throughout Following According Meanwhile Later Eventually Finally
Unlike Like Not No Only Yet Once Now Here Who Whom Whose Where Why How Because Until Among Between Within Without Near Instead
Mr Mrs Ms Dr Sir Lady Lord Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July
August September October November December Christmas Easter""".split())
_TITLES = {"sir", "lady", "lord", "mr", "mrs", "ms", "dr", "king", "queen", "prince", "princess", "saint", "st", "general",
           "captain", "colonel", "major", "father", "mother", "sister", "brother", "uncle", "aunt", "president", "pope",
           "count", "countess", "duke", "duchess", "emperor", "empress", "sergeant", "lieutenant", "professor", "rev", "judge",
           "bishop", "archbishop", "cardinal", "earl", "baron", "baroness", "governor", "senator", "mayor", "chief", "sheriff",
           "doctor", "detective", "officer", "inspector", "agent", "coach", "mister", "miss", "madame", "dame", "reverend"}
_CONNECT = {"de", "da", "van", "von", "der", "den", "del", "della", "di", "du", "la", "le", "bin", "ibn", "al", "of"}
_TOKEN = re.compile(r"\S+")


def passage_names(text: str) -> list[str]:
    """Capitalised name phrases of a passage, in first-seen order. A one-word phrase must occur somewhere other than
    at the start of a sentence (so "Scientists" or "Later" are not names); leading function words, and a sentence's
    first word when it is used in lower case elsewhere or looks like a verb/adverb, are dropped; bare initials too."""
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", text)  # "Jerusalem.In September"
    tokens = [m.group(0) for m in _TOKEN.finditer(text)]
    lower = set(re.findall(r"\b[a-z]+\b", text))
    found, phrase, initial = {}, [], False

    def close():
        nonlocal phrase
        words = phrase
        while words and (words[-1].lower() in _CONNECT):
            words = words[:-1]
        lead = initial
        while words and words[0] in _STOP and words[0] not in ("Sir", "Lady", "Lord", "Mr", "Mrs", "Ms", "Dr"):
            words, lead = words[1:], False
        if lead and len(words) > 1 and (words[0].lower() in lower or re.search(r"(?:ed|ing|ly)$", words[0])):
            words, lead = words[1:], False  # "Created Viscount Tyrconnel" starts a sentence with a verb
        if words and not all(w in _STOP for w in words) and not all(re.fullmatch(r"[A-Z]\.?", w) for w in words):
            name = " ".join(words)
            seen = found.setdefault(name, [0, 0])
            seen[0 if lead else 1] += 1
        phrase = []

    for i, raw in enumerate(tokens):
        word = raw.strip("\"“”‘'([{")
        core = re.sub(r"(['’]s)?[.,;:!?)\]}\"”’]*$", "", word)
        is_cap = bool(core) and core[0].isupper() and core[0].isalpha()
        if is_cap or (phrase and core in _CONNECT and i + 1 < len(tokens)):
            if not phrase:
                prev = tokens[i - 1] if i else ""
                initial = i == 0 or bool(re.search(r"[.!?:]['\"”’)]*$", prev)) or raw[0] in "\"“‘("
            abbreviation = re.fullmatch(r"[A-Z]\.|(?:Mr|Mrs|Ms|Dr|St|Jr|Sr)\.", word)
            phrase.append(word if abbreviation else core)
            ends = word != core and not abbreviation
            if ends or raw != raw.rstrip("\"”’)"):
                close()
        elif phrase:
            close()
    if phrase:
        close()
    return [n for n, (lead, inner) in found.items() if inner or len(n.split()) > 1]


_NOT_PERSON = set("""university college school academy church cathedral abbey chapel temple street road avenue river lake
mountain mount island islands company corporation records band club party army navy force air united states kingdom republic
empire north south east west new old great royal national international museum hospital hotel theatre theater hall house park
square bridge castle palace station airport council committee society association festival award awards prize cup league
championship olympic olympics games chart charts billboard album music studio studios film films pictures productions broadway
world war battle act bay valley county city province district hills beach coast ocean sea canal fort day bible god church
records entertainment media group orchestra symphony opera ballet gallery library institute foundation railway line
bank court parliament senate congress office department ministry government american british english french german
italian spanish russian chinese japanese irish scottish welsh canadian australian european christian catholic show series
tour song single magazine times post news project side radio tv television network channel records press journal review
company inc ltd co""".split())


def _name_words(name: str) -> bool:
    """2-4 plain capitalised words, none of them a place/organisation word: could be a person's name."""
    words = name.split()
    return (2 <= len(words) <= 4 and not any(_n(w) in _NOT_PERSON or w.lower() in _CONNECT for w in words)
            and all(re.fullmatch(r"[^\W\d_][^\W\d]*(?:['’.-][^\W\d_]*)*", w) for w in words))


def quoref_distractors(rng: random.Random, question: str, gold: str, passage: str) -> list[str] | None:
    """3 names of the passage for a Quoref question (see quoref()), most person-like and gold-shaped first."""
    names = passage_names(passage)
    gw = _n(gold).split()
    family = set(gw)
    for name in names:
        if _contains(name, gold):
            family |= set(_n(name).split()) - _CONNECT

    lower = set(re.findall(r"\b[a-z]+\b", passage))

    def related(w: str) -> bool:  # "Carracloughs" for "Carraclough"
        return any(w == f or w.rstrip("s") == f or len(w) >= 5 and len(f) >= 5 and w[:5] == f[:5] and abs(len(w) - len(f)) <= 2
                   for f in family)

    def fine(c: str) -> bool:
        cw = _n(c).split()
        return (bool(cw) and not _contains(c, gold) and not any(map(related, cw)) and c.split()[0] not in _STOP
                and not (len(cw) == 1 and cw[0] in lower))

    singles = {n for n in names if len(n.split()) == 1}
    weak = [n for n in names if _name_words(n)]
    # a person is usually also called by first name or surname alone somewhere in the passage
    persons = [n for n in weak if n.split()[0] in singles or n.split()[-1] in singles or n.split()[0].lower().rstrip(".") in _TITLES]
    first = lambda ns: [n.split()[1] if n.split()[0].lower().rstrip(".") in _TITLES else n.split()[0] for n in ns]
    last = lambda ns: [n.split()[-1] for n in ns]
    similar = lambda ns: [n for n in ns if abs(len(n.split()) - len(gw)) <= 1]
    plain = [n for n in names if not any(_n(w) in _NOT_PERSON for w in n.split())]
    kind = re.search(r"\b(first|last|full|sur)\s*name", question.lower())
    kind = kind.group(1) if kind else None
    about_person = kind or re.search(r"\b(who|whom|whose|person|people|man|woman|men|women|character|boy|girl|wife|husband|"
                                     r"son|daughter|father|mother|brother|sister|friend|singer|actor|actress|artist|player)\b",
                                     question.lower())
    if kind == "first" and len(gw) == 1:
        pools = [first(persons), first(weak)]
    elif kind in ("last", "sur") and len(gw) == 1:
        pools = [last(persons), last(weak)]
    elif about_person and len(gw) > 1:
        pools = [similar(persons), persons, similar(weak), weak]
    elif about_person:
        pools = [[w for w in singles if w in set(first(persons) + last(persons))], first(persons) + last(persons),
                 [w for w in singles if w in set(first(weak) + last(weak))], [w for w in plain if len(w.split()) == 1]]
    else:
        pools = [[n for n in names if len(n.split()) == len(gw)], similar(names)]
    pools.append(plain if about_person else names)
    in_question = lambda c: _n(c) and f" {_n(c)} " in f" {_n(question)} "
    picked = []
    for pool in pools:
        pool = [c for c in dict.fromkeys(pool) if fine(c) and not in_question(c) and c not in picked]
        rng.shuffle(pool)
        for c in pool:
            if len(picked) < 3 and not any(_contains(c, p) for p in picked):
                picked.append(c)
    return picked if len(picked) == 3 else None


def quoref(stats: dict):
    """Dasigi et al. 2019 (CC-BY-4.0; official zip, train file). Crowd questions on Wikipedia passages that need
    coreference to answer, collected against a QA model. Single-span answers that start with a capital letter (names
    and titles; about 90% of single-span answers). Options: the gold and 3 other name phrases of the passage, shaped like
    the gold (first names for "first name" questions, surnames for "last name", multi-word names for full names). A
    name is never one that contains or is contained in the gold, nor shares a word with any passage name holding the
    gold ("Hickson" is never a distractor for "Catherine" when the passage says "Catherine Hickson"), nor a name in
    the question. Passages are kept whole (at most about 6.8k characters). One passage (context_id) is one item."""
    rng = random.Random(2026101931)
    data = json.loads(zipfile.ZipFile(QUOREF_ZIP).read("quoref-train-dev-v0.1/quoref-train-v0.1.json"))
    counts = collections.Counter()
    for article in data["data"]:
        for p in article["paragraphs"]:
            for q in p["qas"]:
                if len(q["answers"]) != 1:
                    counts["several_answers"] += 1
                    continue
                gold = q["answers"][0]["text"].strip()
                if not gold[:1].isupper():
                    counts["not_a_name"] += 1
                    continue
                distractors = quoref_distractors(rng, q["question"], gold, p["context"])
                if not distractors:
                    counts["few_names"] += 1
                    continue
                options, gi = _shuffled(rng, gold, distractors)
                state = {"passage": p["context"], "question": q["question"].strip()}
                question, target = common.choice(state, rng.choice(QUOREF_ASK), options, gi)
                counts["rows"] += 1
                yield common.example(QUOREF, f"{p['context_id']}:{q['id']}", "coreference-reading", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- QuAC: is the question answered by the section; yes/no follow-ups ----------------------------------------
QUAC = "quac-train"
QUAC_FILE = RAW / "quac/train_v0.2.json"
QUAC_ANSWERABLE = ["Does the section text answer this question?", "Can the question be answered from the section?",
                   "Is the answer to the question in the text?"]
QUAC_YESNO = ["Going by the section, is the answer to the question yes?", "According to the text, should the question be answered yes?",
              "Does the section say yes to the question?"]
QUAC_CAP = 30_000
QUAC_YESNO_RATIO = 1.5    # yes rows per no row among the yes/no questions (natural: 3.8)
QUAC_UNANSWERABLE = .4    # share of unanswerable rows among the answerability questions (natural: .17)


def quac(stats: dict):
    """Choi et al. 2018 (CC-BY-SA-4.0 per quac.ai; train_v0.2.json from the official S3 URL). A student asks about a
    Wikipedia section they cannot see; a teacher who sees it answers with a span, CANNOTANSWER, and marks yes/no
    answers. Two questions per turn, given the section (page, section title, text) and the student's earlier
    questions in the dialog:
    - answerability: does the section answer the question (CANNOTANSWER = no);
    - yes/no: for turns the teacher marked y or n, is the answer yes.
    Capped at 30k by a seeded sample: about 11.5k yes/no rows (all n turns, 1.5 y per n) and 18.5k answerability rows
    (40% unanswerable). Several dialogs (different students) share one section, so the item is the Wikipedia page."""
    rng = random.Random(2026101932)
    data = json.load(open(QUAC_FILE))["data"]
    pools = collections.defaultdict(list)
    for article in data:
        for p in article["paragraphs"]:
            context = re.sub(r"\s*CANNOTANSWER\s*$", "", p["context"]).strip()
            earlier = []
            for t, q in enumerate(p["qas"]):
                key = (article["title"], article["section_title"], context, p["id"], t, q["question"].strip(), tuple(earlier))
                if q["orig_answer"]["text"] == "CANNOTANSWER":
                    pools["unanswerable"].append(key)
                else:
                    pools["answerable"].append(key)
                    if q["yesno"] in ("y", "n"):
                        pools["yes" if q["yesno"] == "y" else "no"].append(key)
                earlier.append(q["question"].strip())
    n_no = len(pools["no"])
    n_yes = min(len(pools["yes"]), int(n_no * QUAC_YESNO_RATIO))
    rest = QUAC_CAP - n_no - n_yes
    n_unans = min(len(pools["unanswerable"]), int(rest * QUAC_UNANSWERABLE))
    take = {"no": n_no, "yes": n_yes, "unanswerable": n_unans, "answerable": rest - n_unans}
    counts = collections.Counter({f"pool_{k}": len(v) for k, v in pools.items()})
    for kind, n in take.items():
        rows = pools[kind]
        for title, section, context, dialog, t, q, earlier in sorted(rng.sample(rows, n), key=lambda r: (r[3], r[4])):
            state = {"page": title, "section": section, "text": context,
                     "earlier questions": "\n".join(earlier) if earlier else "(none)", "question": q}
            if kind in ("yes", "no"):
                question, target = common.noul(rng.choice(QUAC_YESNO), kind == "yes")
                family, tag = "yesno-qa", "yn"
            else:
                question, target = common.noul(rng.choice(QUAC_ANSWERABLE), kind == "answerable")
                family, tag = "answerability", "ans"
            counts[kind] += 1
            yield common.example(QUAC, f"{digest(title)}:{dialog}:q{t}:{tag}", family, state, question, target, "train")
    stats["counts"] = dict(counts)


# --- ShARC: answer from a rule text, or ask a follow-up question ---------------------------------------------
SHARC = "sharc-train"
SHARC_ZIP = RAW / "sharc/sharc1-official.zip"
SHARC_LABELS = [  # (yes, no, irrelevant, more information) order
    [("yes", None), ("no", None), ("irrelevant", "the rule text does not answer this question"),
     ("ask a follow-up question", "more information is needed from the user first")],
    [("answer yes", None), ("answer no", None), ("the rule text does not cover the question", None),
     ("need more information", "a follow-up question must be asked first")],
]
SHARC_ASK = ["Given the rule text, the user's situation and the follow-up answers so far, what should the reply be?",
             "Using only the rule text and what the user has said, how should the question be answered?",
             "Decide the reply: yes, no, the rule text does not apply, or ask the user something first."]


def sharc(stats: dict):
    """Saeidi et al. 2018 (CC-BY-SA-3.0; official sharc1-official.zip, json/sharc_train.json). Rule texts from
    government web pages; crowd workers wrote questions, scenarios and the follow-up question/answer trees, and the
    answers were verified. The reply is Yes, No, Irrelevant (the question was paired with another rule text), or a
    follow-up question (= more information needed): a 4-way choice. The hidden 'evidence' field is not shown.
    Identical inputs are kept once; inputs with conflicting replies are dropped. One rule text is one item (it can
    belong to more than one tree)."""
    rng = random.Random(2026101933)
    data = json.loads(zipfile.ZipFile(SHARC_ZIP).read("sharc1-official/json/sharc_train.json"))
    by_input = collections.defaultdict(list)
    for r in data:
        history = tuple((h["follow_up_question"].strip(), h["follow_up_answer"].strip()) for h in r["history"])
        by_input[(r["snippet"], r["question"].strip(), r["scenario"].strip(), history)].append(r)
    counts = collections.Counter()
    for (snippet, q, scenario, history), rows in by_input.items():
        replies = {r["answer"] if r["answer"] in ("Yes", "No", "Irrelevant") else "more" for r in rows}
        if len(replies) != 1:
            counts["conflicting_replies"] += 1
            continue
        reply = replies.pop()
        counts["duplicates_merged"] += len(rows) - 1
        state = {"rule text": snippet, "question": q, "scenario": scenario or "(none given)",
                 "follow-up so far": "\n".join(f"Q: {a}\nA: {b}" for a, b in history) or "(none)"}
        options, gold = _labelled(rng, SHARC_LABELS, ["Yes", "No", "Irrelevant", "more"].index(reply))
        question, target = common.choice(state, rng.choice(SHARC_ASK), options, gold)
        counts[reply] += 1
        yield common.example(SHARC, f"{digest(snippet)}:{rows[0]['utterance_id']}", "rule-conversation", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- ConditionalQA: yes/no answers under a policy document ---------------------------------------------------
CQA = "conditionalqa-train"
CQA_DIR = RAW / "haitian-sun__ConditionalQA@77bd295952da/v1_0"
CQA_CHARS = 6000
CQA_ASK = ["Going by the guidance, is the answer to the person's question yes?", "According to the document, should the answer be yes?",
           "Does the guidance answer this question with yes?"]


def _cqa_line(element: str) -> str:
    m = re.match(r"<(h[1-6]|p|li|tr|td|th)>(.*?)</\1>$", element.strip(), re.S)
    tag, text = (m.group(1), m.group(2)) if m else ("p", element)
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
    if tag.startswith("h"):
        return "#" * int(tag[1]) + " " + text
    return f"- {text}" if tag == "li" else text


def cqa_document(contents: list[str], needed: set[str]) -> tuple[str, str]:
    """The document's sections (split at headings) that hold an evidence or condition element, in order, with the
    page's first heading; the whole sections while they fit CQA_CHARS, else the evidence elements with one element of
    context on each side under their section headings. Falls back to the start of the document."""
    sections, current = [], []
    for el in contents:
        if re.match(r"\s*<h[1-6]>", el) and current:
            sections.append(current)
            current = []
        current.append(el)
    if current:
        sections.append(current)
    chosen = [s for s in sections if needed & set(s)]
    if not chosen:
        return "\n".join(map(_cqa_line, contents))[:CQA_CHARS], "start"
    head = [] if chosen[0] is sections[0] else [contents[0]]
    text = "\n".join(map(_cqa_line, head + [el for s in chosen for el in s]))
    if len(text) <= CQA_CHARS:
        return text, "sections"
    lines = [_cqa_line(contents[0])] if head else []
    for s in chosen:
        keep = {0} | {j + d for j, el in enumerate(s) if el in needed for d in (-1, 0, 1)}
        lines += [_cqa_line(s[j]) for j in sorted(keep) if 0 <= j < len(s)]
    return "\n".join(lines)[:CQA_CHARS], "windows"


def conditionalqa(stats: dict):
    """Sun et al. 2021 (CC-BY-SA-4.0 per the project page; github haitian-sun/ConditionalQA v1_0, train.json and
    documents.json). Annotators wrote scenarios and questions about UK government (gov.uk) policy pages and the
    answers, with the conditions under which an answer holds. Kept: yes/no questions with one answer value (yes or
    no, with or without conditions; the gold is that value). Dropped: span answers, questions answered both yes and
    no under different conditions, and unanswerable ones. The document is cut to the sections holding the
    annotated evidence and conditions (cqa_document), because whole pages run to 47k characters. One page (url) is
    one item."""
    rng = random.Random(2026101934)
    rows = json.loads((CQA_DIR / "train.json").read_text())
    docs = {d["url"]: d for d in json.loads((CQA_DIR / "documents.json").read_text())}
    counts = collections.Counter()
    for r in rows:
        values = {a[0].strip().lower() for a in r["answers"]}
        if r["not_answerable"] or not values or not values <= {"yes", "no"} or len(values) != 1:
            counts["not_single_yesno"] += 1
            continue
        doc = docs[r["url"]]
        needed = set(r["evidences"]) | {c for a in r["answers"] for c in a[1]}
        text, how = cqa_document(doc["contents"], needed)
        state = {"page": doc["title"], "document": text, "scenario": r["scenario"].strip(), "question": r["question"].strip()}
        question, target = common.noul(rng.choice(CQA_ASK), values == {"yes"})
        counts[f"{values.pop()}{'_with_conditions' if any(a[1] for a in r['answers']) else ''}"] += 1
        counts[f"document_{how}"] += 1
        yield common.example(CQA, f"{digest(r['url'])}:{r['id']}", "policy-yesno", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- CREAK: is a claim about an entity true ------------------------------------------------------------------
CREAK = "creak-train"
CREAK_FILE = RAW / "yasumasaonoe__creak@5e892b28b6da/data/creak/train.json"
CREAK_ASK = ["Is this claim true?", "Is the statement correct?", "Does this claim hold up?"]


def creak(stats: dict):
    """Onoe et al. 2021 (MIT; github yasumasaonoe/creak data/creak/train.json). Crowd-written claims about a Wikipedia
    entity needing commonsense and entity knowledge, labelled true/false by their writers and validated. The
    writer's explanation is not shown. One seed entity is one item."""
    rng = random.Random(2026101935)
    counts = collections.Counter()
    for r in map(json.loads, CREAK_FILE.open()):
        state = r["sentence"] if rng.random() < .5 else {"claim": r["sentence"]}
        question, target = common.noul(rng.choice(CREAK_ASK), r["label"] == "true")
        counts[r["label"]] += 1
        yield common.example(CREAK, f"{digest(r['entity'])}:{r['ex_id']}", "commonsense-claim", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- SNLI: caption entailment ---------------------------------------------------------------------------------
SNLI = "snli-train"
SNLI_DIR = hub_dir("stanfordnlp/snli", "cdb5c3d5eed6")
SNLI_ROWS = 30_000
SNLI_LABELS = [  # (entailment, neutral, contradiction) order
    [("entailment", "the hypothesis must be true when the premise is"), ("neutral", "the hypothesis may or may not be true"),
     ("contradiction", "the hypothesis cannot be true when the premise is")],
    [("definitely true", None), ("might be true", None), ("definitely false", None)],
    [("follows", None), ("can't tell", None), ("conflicts", None)],
]
SNLI_ASK = ["If the premise describes the scene correctly, where does the hypothesis stand?",
            "Is the hypothesis implied by the premise, ruled out by it, or neither?", "Judge the hypothesis against the premise."]


def snli(stats: dict):
    """Bowman et al. 2015 (CC-BY-SA-4.0; stanfordnlp/snli train parquet). Flickr30k captions as premises,
    hypotheses and labels written by crowd workers. Rows with a gold label (0/1/2; -1 = no consensus is skipped);
    a seeded sample of 30k. One premise is one item."""
    rng = random.Random(2026101936)
    table = pq.read_table(SNLI_DIR / "plain_text/train-00000-of-00001.parquet").to_pylist()
    labelled = [i for i, r in enumerate(table) if r["label"] in (0, 1, 2)]
    counts = collections.Counter(skipped_no_label=len(table) - len(labelled))
    for i in sorted(rng.sample(labelled, SNLI_ROWS)):
        r = table[i]
        options, gold = _labelled(rng, SNLI_LABELS, r["label"])
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        question, target = common.choice(state, rng.choice(SNLI_ASK), options, gold)
        counts[("entailment", "neutral", "contradiction")[r["label"]]] += 1
        yield common.example(SNLI, f"{digest(r['premise'])}:{i}", "nli", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- InfoTABS: infobox NLI ------------------------------------------------------------------------------------
INFOTABS = "infotabs-train"
INFOTABS_DIR = RAW / "infotabs__infotabs@34e7dc0900a0/data"
INFOTABS_LABELS = [  # (E, N, C) order
    [("entailed", "the table shows the statement is true"), ("neutral", "the table does not settle it"),
     ("contradicted", "the table shows the statement is false")],
    [("true", None), ("not enough information", None), ("false", None)],
]
INFOTABS_ASK = ["According to the table, is the statement true, false, or undetermined?",
                "Does the infobox support the statement, contradict it, or neither?", "Judge the statement against the table."]


def infotabs_table(table: dict) -> str:
    return "\n".join(f"{k}: {'; '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in table.items())


def infotabs(stats: dict):
    """Gupta et al. 2020 (Apache-2.0, the data repository's LICENSE; github infotabs/infotabs). Wikipedia infoboxes
    as premises; crowd-written hypotheses labelled entailed / neutral / contradicted by their writers and
    validated. Train split; the infobox is rendered as "key: value" lines (values of one key joined by "; "). One
    table is one item."""
    rng = random.Random(2026101937)
    counts = collections.Counter()
    lines = (INFOTABS_DIR / "maindata/infotabs_train.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    for n, line in enumerate(lines[1:], 1):
        r = dict(zip(header, line.split("\t")))
        table = json.loads((INFOTABS_DIR / f"tables/json/{r['table_id']}.json").read_text())
        state = {"table": infotabs_table(table), "statement": r["hypothesis"].strip()}
        options, gold = _labelled(rng, INFOTABS_LABELS, "ENC".index(r["label"]))
        question, target = common.choice(state, rng.choice(INFOTABS_ASK), options, gold)
        counts[r["label"]] += 1
        yield common.example(INFOTABS, f"{r['table_id']}:{r['annotater_id']}-{n}", "table-nli", state, question, target, "train")
    stats["counts"] = dict(counts)


def table_parts(e: dict) -> tuple[list[str], list[str]]:
    """For common.write(tables=...): the table text is checked by shingles only."""
    state = e["state"]
    return [v for k, v in state.items() if k != "table"], [state["table"]]


# --- WikiTableQuestions: answer a question about a table ------------------------------------------------------
WTQ = "wtq-train"
WTQ_ZIP = RAW / "wtq/WikiTableQuestions-1.0.2-compact.zip"
WTQ_CHARS = 6000
WTQ_ASK = ["Answer the question using the table.", "Which option answers the question about the table?",
           "Use the table to pick the right answer."]
_EMPTY = {"", "-", "—", "–", "n a", "none", "tba", "tbd", "?"}


def _wtq_unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": " ", "p": "|", "\\": "\\"}.get(m.group(1), m.group(0)), text).strip()


def wtq_distractors(rng: random.Random, gold: str, body: list[list[str]]) -> tuple[list[str], str] | None:
    """3 cells of the table: from the gold's column when the gold is a cell (or, for text, inside one), else (computed numbers)
    numeric cells within a factor of 3 of the gold and then gold +-1..3. Never a cell equal to the gold, nested in
    it or holding it, numerically equal, or equal once signs and punctuation are dropped ("-3" for "+3")."""
    width = max(map(len, body))
    columns = [[row[j] for row in body if j < len(row)] for j in range(width)]
    value = _number(gold)
    # "-3" is not shown for "+3"; a numeric gold gets numeric cells only
    ok = lambda c: (_n(c) not in _EMPTY and _n(c) != _n(gold) and not _same_answer(gold, c)
                    and (value is None or _number(c) is not None))
    exact = [j for j, col in enumerate(columns) if any(_n(c) == _n(gold) or (_number(c) is not None and value is not None
                                                                              and abs(_number(c) - value) < 1e-9) for c in col)]
    inside = [] if value is not None else [j for j, col in enumerate(columns) if j not in exact and any(_contains(c, gold) for c in col)]
    tiers, how = [], ""
    for js, label in ((exact, "column"), (inside, "column_partial")):
        if js:
            tiers.append([c for j in js for c in columns[j]])
            how = how or label
    if value is not None:
        if not how:
            how = "numbers"
        near = []
        for col in columns:
            for c in col:
                v = _number(c)
                if v is not None and (value == 0 and abs(v) <= 3 or value != 0 and v * value > 0 and 1 / 3 <= v / value <= 3):
                    near.append(c)
        tiers.append(near)
        shown = (lambda v: f"{v:,.0f}") if "," in gold and value.is_integer() else _fmt_number
        tiers.append([shown(value + d) for d in (-3, -2, -1, 1, 2, 3) if value + d >= 0 or value < 0])
    elif how:
        kinds = [j for j in range(width) if j not in exact + inside and sum(_number(c) is None for c in columns[j]) > len(columns[j]) / 2]
        tiers.append([c for j in kinds for c in columns[j]])
    if not how:
        return None
    picked = []
    for pool in tiers:
        pool = [c for c in dict.fromkeys(pool) if ok(c)]
        rng.shuffle(pool)
        for c in pool:
            if len(picked) < 3 and not any(_same_answer(c, p) or _n(c) == _n(p) for p in picked):
                picked.append(c)
    return (picked, how) if len(picked) == 3 else None


def wtq(stats: dict):
    """Pasupat and Liang 2015 (CC-BY-SA-4.0, the repository LICENSE; v1.0.2 compact zip from the GitHub release,
    data/training.tsv = the official training portion, tables from csv/*.tsv). Crowd questions and answers on
    Wikipedia tables. Single-answer questions on tables of at most WTQ_CHARS characters (rows as " | " lines; about 5%
    of questions are on longer tables and are left out). Options: the gold and 3 cells chosen by wtq_distractors.
    Yes/no answers and answers neither in the table nor numeric are left out. One table file is one item."""
    rng = random.Random(2026101938)
    z = zipfile.ZipFile(WTQ_ZIP)
    lines = z.read("WikiTableQuestions/data/training.tsv").decode().splitlines()
    tables = {}
    counts = collections.Counter()
    for line in lines[1:]:
        qid, utterance, context, target_value = line.split("\t")
        values = target_value.split("|")
        if len(values) != 1:
            counts["several_answers"] += 1
            continue
        gold = _wtq_unescape(values[0])
        if context not in tables:
            rows = [[_wtq_unescape(c) for c in row.split("\t")]
                    for row in z.read("WikiTableQuestions/" + context.replace(".csv", ".tsv")).decode().splitlines()]
            tables[context] = rows
        rows = tables[context]
        text = "\n".join(" | ".join(row) for row in rows)
        if len(text) > WTQ_CHARS:
            counts["table_too_long"] += 1
            continue
        if _n(gold) in ("yes", "no") or not _n(gold):
            counts["yes_no_or_empty"] += 1
            continue
        found = wtq_distractors(rng, gold, rows[1:])
        if not found:
            counts["no_distractors"] += 1
            continue
        distractors, how = found
        options, gi = _shuffled(rng, gold, distractors)
        state = {"table": text, "question": utterance.strip()}
        question, target = common.choice(state, rng.choice(WTQ_ASK), options, gi)
        counts[how] += 1
        table_id = context.removeprefix("csv/").removesuffix(".csv").replace("/", "_")
        yield common.example(WTQ, f"{table_id}:{qid}", "table-qa", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- SVAMP: arithmetic word problems --------------------------------------------------------------------------
SVAMP = "svamp-train"
SVAMP_DIR = hub_dir("ChilleD/SVAMP", "5e0bf1e5e7c0")
SVAMP_ASK = ["Solve the word problem and pick the answer.", "Which number answers the question?", "Work out the answer to the problem."]
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b if b else None}


def _arith(node):
    """Value of a parsed + - * / expression of numbers (never eval)."""
    if isinstance(node, ast.Expression):
        return _arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _arith(node.left), _arith(node.right)
        return None if a is None or b is None else _OPS[type(node.op)](a, b)
    raise ValueError("unsupported expression")


def svamp_variants(equation: str) -> list[float]:
    """Values of the equation with one operation changed (another operator, or the operands swapped)."""
    tree = ast.parse(equation, mode="eval")
    values = []
    evaluate = lambda: _arith(tree)
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]:
        op, left, right = node.op, node.left, node.right
        for new in _OPS:
            for swap in (False, True):
                if new is type(op) and not swap:
                    continue
                node.op = new()
                node.left, node.right = (right, left) if swap else (left, right)
                try:
                    values.append(evaluate())
                except ValueError:
                    pass
        node.op, node.left, node.right = op, left, right
    return [v for v in values if v is not None]


def svamp(stats: dict):
    """Patel et al. 2021 (MIT; ChilleD/SVAMP train parquet, 700 problems; the test file is not used). Human-written
    variations of ASDiv-A one-unknown arithmetic problems; the answer is computed from the annotated equation (checked
    against the Answer field). Options: the gold and 3 distractors - the equation with one operation changed (another
    operator or swapped operands), then sums, differences, products and quotients of two numbers of the problem,
    then numbers of the problem - keeping non-negative values (whole numbers when the gold is whole) that differ from
    the gold. One problem is one item."""
    rng = random.Random(2026101939)
    counts = collections.Counter()
    for r in pq.read_table(SVAMP_DIR / "data/train-00000-of-00001.parquet").to_pylist():
        gold = float(r["Answer"])
        try:
            computed = _arith(ast.parse(r["Equation"], mode="eval"))
        except (ValueError, SyntaxError):
            computed = None
        if computed is None or abs(computed - gold) > 1e-6:
            counts["equation_mismatch"] += 1
            continue
        body = f"{r['Body'].strip()} {r['Question'].strip()}"
        numbers = list(dict.fromkeys(float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", body)))
        pairs = [f(a, b) for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j for f in _OPS.values()]
        whole = gold.is_integer()
        good = lambda v: v is not None and v >= 0 and (not whole or abs(v - round(v)) < 1e-9) and abs(v - gold) > 1e-9
        picked = []
        for pool in (svamp_variants(r["Equation"]), pairs, numbers):
            pool = sorted({round(v, 6) for v in pool if good(v)})
            rng.shuffle(pool)
            for v in pool:
                if len(picked) < 3 and all(abs(v - p) > 1e-9 for p in picked):
                    picked.append(v)
        if len(picked) < 3:
            counts["few_distractors"] += 1
            continue
        options, gi = _shuffled(rng, _fmt_number(gold), [_fmt_number(v) for v in picked])
        state = {"problem": r["Body"].strip(), "question": r["Question"].strip()} if rng.random() < .5 else body
        question, target = common.choice(state, rng.choice(SVAMP_ASK), options, gi)
        counts[r["Type"].replace("Divison", "Division")] += 1
        yield common.example(SVAMP, f"{r['ID']}:0", "math", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- StepGame: multi-hop spatial relations ----------------------------------------------------------------------
STEPGAME = "stepgame-train"
STEPGAME_FILE = hub_dir("ZhengyanShi/StepGame", "6d859381dfd5") / "train.jsonl"
STEPGAME_PER_HOP = 3000
STEPGAME_LABELS = {"left": None, "right": None, "above": None, "below": None, "upper-left": None, "upper-right": None,
                   "lower-left": None, "lower-right": None, "overlap": "at the same position"}
STEPGAME_ASK = ["Where is the first agent relative to the second?", "Which spatial relation holds between the two agents?",
                "Answer the question about the agents' positions."]
# Sentence patterns whose wording contradicts the relation StepGame's generator meant (SpatialLM-StepGame, Li et al.
# 2024, correct.py: incorrect_to_correct_mapping). Stories with any such sentence are dropped, as the authors did.
STEPGAME_WRONG = [re.compile(p) for p in (
    r"(\b\w+\b) and (\b\w+\b) are parallel, and \1 on the right of \2",
    r"(\b\w+\b) and (\b\w+\b) are parallel, and \1 is to the right of \2",
    r"(\b\w+\b) and (\b\w+\b) are horizontal and \1 is to the right of \2",
    r"(\b\w+\b) and (\b\w+\b) are both there with the object \1 is to the right of object \2",
    r"(\b\w+\b) is at the bottom of (\b\w+\b) and is on the same vertical plane",
    r"(\b\w+\b) presents below (\b\w+\b)",
    r"(\b\w+\b) is placed at the bottom of (\b\w+\b)",
    r"(\b\w+\b) is there and (\b\w+\b) is at the 10 position of a clock face",
    r"(\b\w+\b) is positioned below (\b\w+\b) and to the left",
    r"Object (\b\w+\b) is above object (\b\w+\b) and to the right of it, too",
    r"(\b\w+\b) is diagonally to the upper right of (\b\w+\b)",
    r"(\b\w+\b) is to the right and above (\b\w+\b) at an angle of about 45 degrees",
    r"(\b\w+\b) is diagonally left and above (\b\w+\b)",
)]


def stepgame(stats: dict):
    """Shi et al. 2022 (MIT; ZhengyanShi/StepGame train.jsonl, k = 1-5 hops, no noise sentences). Stories generated
    from a grid of agents with templated sentences; the relation asked about is computed by the generator. The
    original release has sentence templates that state the opposite relation (e.g. "P and N are parallel, and P on
    the right of N" generated for "P left of N"); checked here on the 1-hop rows and listed by Li et al. 2024
    (SpatialLM-StepGame, MIT): every story with such a sentence is dropped. A seeded sample of 3k stories per hop
    count; options: the 9 relation labels, in shuffled order. One story is one item."""
    rng = random.Random(2026101940)
    by_hop = collections.defaultdict(list)
    counts = collections.Counter()
    for i, r in enumerate(map(json.loads, STEPGAME_FILE.open())):
        if any(p.search(s) for s in r["story"] for p in STEPGAME_WRONG):
            counts["wrong_template_dropped"] += 1
            continue
        by_hop[int(r["k_hop"])].append((i, r))
    for hop in sorted(by_hop):
        for i, r in sorted(rng.sample(by_hop[hop], min(STEPGAME_PER_HOP, len(by_hop[hop]))), key=lambda x: x[0]):
            story = " ".join(s.strip() for s in r["story"])
            state = {"story": story, "question": r["question"]}
            labels = list(STEPGAME_LABELS)
            rng.shuffle(labels)
            question, target = common.choice(state, rng.choice(STEPGAME_ASK), {k: STEPGAME_LABELS[k] for k in labels},
                                             labels.index(r["label"]))
            counts[f"{hop}hop"] += 1
            yield common.example(STEPGAME, f"s{i}:{hop}hop", "spatial-reasoning", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- SpartQA: questions over a described block scene -----------------------------------------------------------
SPARTQA = "spartqa-train"
SPARTQA_YN = hub_dir("tasksource/spartqa-yn", "150c819e88bb") / "spartqa_YN_train.jsonl"
SPARTQA_MC = hub_dir("tasksource/spartqa-mchoice", "d29ba4560446") / "spartqa_CO_train.jsonl"
SPARTQA_ROWS = 10_000
SPARTQA_YN_LABELS = [  # (Yes, No, DK) order
    [("yes", None), ("no", None), ("cannot be determined", "the description does not settle it")],
    [("yes", None), ("no", None), ("don't know", "the scene as described allows either answer")],
]
SPARTQA_YN_ASK = ["Based on the description of the blocks, answer the question.", "Answer from the scene description.",
                  "Use the description of the scene to answer the question."]
SPARTQA_MC_ASK = ["Based on the scene description, which option answers the question?", "Pick the answer that fits the described scene.",
                  "Which choice is right, given the description of the blocks?"]


def spartqa(stats: dict):
    """Mirzaee et al. 2021, SpartQA-Auto (MIT, github HLR/SpartQA_generation LICENSE.md; files from the tasksource
    mirrors spartqa-yn and spartqa-mchoice, train only). Scene descriptions of blocks and shapes generated from
    NLVR-style scenes; questions and answers computed from the scene graph with spatial rules. Yes/no questions have
    the answers Yes, No and DK (not determined by the description): a 3-way choice. Choice questions keep the
    dataset's 4 candidates (two objects, "both of them", "none of them"), shuffled. Repeated (story, question) pairs
    are kept once and dropped when their answers differ. A seeded sample of 10k of each. One story is one item."""
    rng = random.Random(2026101941)
    counts = collections.Counter()
    for kind, path in (("yn", SPARTQA_YN), ("mc", SPARTQA_MC)):
        rows = [json.loads(line) for line in path.open()]
        answers = collections.defaultdict(set)
        for r in rows:
            answers[(r["story"], r["question"])].add(json.dumps([r["answer"], r.get("candidate_answers")]))
        seen, unique = set(), []
        for i, r in enumerate(rows):
            key = (r["story"], r["question"])
            if len(answers[key]) > 1:
                counts[f"{kind}_conflicting_dropped"] += 1
            elif key not in seen:
                seen.add(key)
                unique.append((i, r))
        for i, r in sorted(rng.sample(unique, min(SPARTQA_ROWS, len(unique))), key=lambda x: x[0]):
            state = {"scene": r["story"].strip(), "question": re.sub(r"\s+", " ", r["question"]).strip().replace(" ,", ",").replace(" ?", "?")}
            if kind == "yn":
                options, gold = _labelled(rng, SPARTQA_YN_LABELS, ["Yes", "No", "DK"].index(r["answer"]))
                question, target = common.choice(state, rng.choice(SPARTQA_YN_ASK), options, gold)
                counts[f"yn_{r['answer']}"] += 1
            else:
                candidates = [re.sub(r"\s+", " ", c).strip() for c in r["candidate_answers"]]
                if len({_n(c) for c in candidates}) != len(candidates):
                    counts["mc_duplicate_candidates"] += 1
                    continue
                gold_text = candidates[r["answer"]]
                rng.shuffle(candidates)
                question, target = common.choice(state, rng.choice(SPARTQA_MC_ASK), candidates, candidates.index(gold_text))
                counts[f"mc_{['first', 'second', 'both', 'none'][r['answer']]}"] += 1
            yield common.example(SPARTQA, f"{digest(r['story'])}:{kind}{i}", "spatial-reasoning", state, question, target, "train")
    stats["counts"] = dict(counts)


BUILDERS = [
    (QUOREF, quoref, {"upstream": "quoref-train-dev-v0.1.zip (quoref-dataset S3) quoref-train-v0.1.json", "licence": "CC-BY-4.0",
                      "licence_url": "https://allenai.org/data/quoref",
                      "label_origin": "human (crowd answers); distractors programmatic (other names of the passage)"}),
    (QUAC, quac, {"upstream": "https://s3.amazonaws.com/my89public/quac/train_v0.2.json", "licence": "CC-BY-SA-4.0",
                  "licence_url": "https://quac.ai/", "label_origin": "human (teacher's CANNOTANSWER and yes/no marks)"}),
    (SHARC, sharc, {"upstream": "sharc1-official.zip json/sharc_train.json", "licence": "CC-BY-SA-3.0",
                    "licence_url": "https://sharc-data.github.io/data.html", "label_origin": "human"}),
    (CQA, conditionalqa, {"upstream": "github haitian-sun/ConditionalQA@77bd2959 v1_0 train.json + documents.json", "licence": "CC-BY-SA-4.0",
                          "licence_url": "https://haitian-sun.github.io/conditionalqa/",
                          "note": "README: 'should ONLY be used for NLP research purpose'; answers not verified by legal professionals",
                          "label_origin": "human"}),
    (CREAK, creak, {"upstream": "github yasumasaonoe/creak@5e892b28 data/creak/train.json", "licence": "MIT (LICENSE.txt)",
                    "label_origin": "human"}),
    (SNLI, snli, {"upstream": "stanfordnlp/snli@cdb5c3d5 plain_text train (30k sampled)", "licence": "CC-BY-SA-4.0", "label_origin": "human"}),
    (INFOTABS, infotabs, {"upstream": "github infotabs/infotabs@34e7dc09 data/maindata/infotabs_train.tsv + data/tables/json",
                          "licence": "Apache-2.0 (repository LICENSE)", "label_origin": "human"}),
    (WTQ, wtq, {"upstream": "WikiTableQuestions v1.0.2 compact zip (GitHub release) data/training.tsv", "licence": "CC-BY-SA-4.0 (repository LICENSE)",
                "label_origin": "human answers; distractors programmatic (table cells)"}),
    (SVAMP, svamp, {"upstream": "ChilleD/SVAMP@5e0bf1e5 data/train parquet", "licence": "MIT",
                    "label_origin": "human-written problems, answers computed from the annotated equations; distractors programmatic"}),
    (STEPGAME, stepgame, {"upstream": "ZhengyanShi/StepGame@6d859381 train.jsonl, wrong-template stories removed per SpatialLM-StepGame",
                          "licence": "MIT", "label_origin": "programmatic (generator)"}),
    (SPARTQA, spartqa, {"upstream": "tasksource/spartqa-yn@150c819e + tasksource/spartqa-mchoice@d29ba456 train", "licence": "MIT (HLR/SpartQA_generation)",
                        "label_origin": "programmatic (generator)"}),
]
TABLES = {INFOTABS: table_parts, WTQ: table_parts}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", nargs="*", default=None, help="download these (all when none given)")
    p.add_argument("names", nargs="*")
    args = p.parse_args()
    if args.fetch is not None:
        RAW.mkdir(parents=True, exist_ok=True)
        log = RAW / "fetch.jsonl"
        for name in args.fetch or list(FETCH):
            record = fetch_spec(name, FETCH[name], RAW)
            print(json.dumps(record), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
    else:
        wanted = set(args.names)
        for name, build, meta in BUILDERS:
            if not wanted or name in wanted:
                common.write(name, build(meta), meta, tables=TABLES.get(name))
