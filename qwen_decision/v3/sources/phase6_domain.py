"""Phase-6 sources, domain set (2026-10-19): expert domains, content judgments and task dialogues.

  python -m qwen_decision.v3.sources.phase6_domain --fetch          # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase6_domain [name ...]       # build data/phase2/<name>.jsonl

Expert answers (MAUD merger-agreement deal points, PubHealth fact-check verdicts, HEAD-QA healthcare exams),
human content ratings (Civil Comments toxicity, HateXplain), and task-dialogue / assistant intents (MultiWOZ 2.2,
MTOP, SNIPS, ReDial questionnaires). Licences and download locations were vetted in
runs/analysis/phase6-candidates-vetted.json and re-read on the downloaded files (see each docstring).
Only train splits are read (MAUD also dev, which shares the train contracts). Nothing downloaded is executed:
HF loader scripts (head_qa.py, health_fact) are not fetched; archives are read with zipfile.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import glob
import io
import json
import random
import re
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import fetch_spec, hub_dir, readable_camel

RAW = common.RAW / "phase6"

# name -> ("github", repo, commit, [path prefixes kept from the tarball]) | ("hub", repo, [allow patterns]) | ("url", url, file)
FETCH = {
    "maud": ("hub", "theatticusproject/maud", ["README.md", "MAUD v1 README (1).pdf", "MAUD_v1/MAUD_train.csv", "MAUD_v1/MAUD_dev.csv"]),
    # PubHealth (Kotonya & Toni 2020): the Drive file linked from github neemakot/Health-Fact-Checking README (MIT LICENSE).
    # It is a zip despite the name the HF loader gives it.
    "pubhealth": ("url", "https://drive.usercontent.google.com/download?id=1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj&export=download&confirm=t",
                  "PUBHEALTH.zip"),
    "headqa": ("hub", "dvilares/head_qa", ["README.md", "data/head-qa-es-en-pdfs.zip"]),
    "civilcomments": ("hub", "google/civil_comments", ["README.md", "data/train-*"]),
    "hatexplain": ("github", "punyajoy/HateXplain", "01d742279dac941981f53806154481c0e15ee686",
                   ["LICENSE", "README.md", "Data/dataset.json", "Data/post_id_divisions.json"]),
    "multiwoz": ("github", "budzianowski/multiwoz", "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b",
                 ["LICENSE", "README.md", "data/MultiWOZ_2.2/README.md", "data/MultiWOZ_2.2/schema.json", "data/MultiWOZ_2.2/train/"]),
    "mtop": ("url", "https://dl.fbaipublicfiles.com/mtop/mtop.zip", "mtop.zip"),
    "snips": ("github", "sonos/nlu-benchmark", "b86ac7f1577868c42158d0dec77db50956046696",
              ["LICENSE", "README.md", "2017-06-custom-intent-engines/"]),
    "redial": ("hub", "community-datasets/re_dial", ["README.md", "data/train-*"]),
}


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def head(text: str, limit: int) -> str:
    """The start of a long text, cut at a sentence end (or a space) before `limit` characters, marked [...]."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    cut = cut[:end + 1] if end > limit * .6 else cut[:cut.rfind(" ")]
    return cut.rstrip() + " [...]"


# --- MAUD: deal-point answers for merger-agreement clauses --------------------------------------------------
MAUD = "maud-train"
MAUD_DIR = hub_dir("theatticusproject/maud", "37d5c3b95d18") / "MAUD_v1"
MAUD_MAX_CHARS = 10_000
MAUD_ASK = ["Read the merger-agreement excerpt. Which answer to the deal point question does it support?",
            "As a lawyer reviewing this clause for a deal points study, pick the answer that fits the deal point.",
            "Based on the excerpt, what is the answer to the deal point question?"]
MAUD_ASK_YN = ["Based on the excerpt, is the answer to the deal point question yes?",
               "Reading the clause, should this deal point be marked yes?",
               "Does the merger-agreement excerpt answer the deal point question in the affirmative?"]
MAUD_ASK_ML = ["Does the candidate answer apply to this deal point, given the excerpt?",
               "For this deal point, should the candidate answer be checked?",
               "Reading the clause, is the candidate answer one of the answers to the deal point?"]


def maud_question(q: str) -> str:
    q = re.sub(r"\s*-?\s*\b[Aa]nswer\b.*$", "", q.strip())
    return re.sub(r"\s*\(Y/N\)\s*$", "", q).strip()


def maud_shingles(text: str, n: int = 8) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def maud(stats: dict):
    """Wang et al. 2023, MAUD v1 (CC-BY-4.0, HF theatticusproject/maud). Deal-point texts extracted from 152 public
    merger agreements; answers by lawyers for the 2021 ABA Public Target Deal Points Study. Train and dev CSVs
    (the dev split shares the train contracts; test is not read). Three row kinds, as the dataset casts them:
    - multiple-choice deal points: the options are every answer the dataset gives for that question (train+dev);
    - Yes/No deal points: a yes/no question;
    - multilabel deal points (cast by MAUD as one binary row per candidate answer): does the candidate apply?
    State: the excerpt, its topic (text_type) and the deal point question (the dataset's "-Answer (Y/N)" tails
    removed). Texts over 10,000 characters (about 1%) are left out rather than cut, since the evidence can be
    anywhere in the excerpt. Item = contract; the "rare answers" rows (lawyers' edits of a main-set text) are
    assigned to the contract whose texts share the most 8-word shingles with them."""
    rng = random.Random(2026101961)
    frames = []
    for split in ("train", "dev"):
        df = pd.read_csv(MAUD_DIR / f"MAUD_{split}.csv", dtype=str, keep_default_na=False)
        df["split"], df["row"] = split, range(len(df))
        frames.append(df)
    rows = pd.concat(frames).to_dict("records")
    options = collections.defaultdict(dict)
    for r in rows:
        if r["subquestion"] == "<NONE>":
            label = options[r["question"]].setdefault(r["answer"], int(r["label"]))
            assert label == int(r["label"]), (r["question"], r["answer"])
    # rare-answer rows -> the contract they were edited from
    owner = collections.defaultdict(set)
    for r in rows:
        if r["contract_name"] != "<RARE_ANSWERS>":
            for s in maud_shingles(r["text"]):
                owner[s].add(r["contract_name"])
    rare_group = {}
    counts = collections.Counter()
    for r in rows:
        if r["contract_name"] == "<RARE_ANSWERS>" and r["text"] not in rare_group:
            votes = collections.Counter(next(iter(owner[s])) for s in maud_shingles(r["text"]) if len(owner.get(s, ())) == 1)
            best = votes.most_common(1)
            rare_group[r["text"]] = best[0][0] if best and best[0][1] >= 5 else f"rare-{digest(r['text'])}"
            counts["rare_text_matched" if best and best[0][1] >= 5 else "rare_text_unmatched"] += 1
    for r in rows:
        if len(r["text"]) > MAUD_MAX_CHARS:
            counts["too_long"] += 1
            continue
        group = rare_group.get(r["text"], r["contract_name"]) if r["contract_name"] == "<RARE_ANSWERS>" else r["contract_name"]
        group = group.replace("contract_", "c")
        state = {"excerpt": r["text"].strip(), "topic": r["text_type"], "deal point question": maud_question(r["question"])}
        if r["subquestion"] != "<NONE>":
            state["candidate answer"] = r["subquestion"]
            question, target = common.noul(rng.choice(MAUD_ASK_ML), r["label"] == "1")
            kind = "multilabel"
        elif set(options[r["question"]]) == {"Yes", "No"}:
            question, target = common.noul(rng.choice(MAUD_ASK_YN), r["answer"] == "Yes")
            kind = "yes_no"
        else:
            answers = sorted(options[r["question"]], key=options[r["question"]].get)
            rng.shuffle(answers)
            question, target = common.choice(state, rng.choice(MAUD_ASK), answers, answers.index(r["answer"]))
            kind = f"choice{len(answers)}"
        counts[kind] += 1
        counts[f"{r['split']}_{r['data_type']}"] += 1
        yield common.example(MAUD, f"{group}:{r['split']}{r['row']}", "contract-deal-points", state, question, target,
                             r["split"])
    stats["counts"] = dict(counts)


# --- PubHealth: professional fact-checkers' verdicts on health claims ----------------------------------------
PUBHEALTH = "pubhealth-train"
PUBHEALTH_ZIP = RAW / "pubhealth/PUBHEALTH.zip"
PUBHEALTH_CHARS = 2_500
PUBHEALTH_OPTIONS = {"true": "the claim is accurate", "false": "the claim is inaccurate",
                     "mixture": "the claim is partly accurate and partly not",
                     "unproven": "there is not enough evidence to prove or disprove the claim"}
PUBHEALTH_ASK = ["A fact-checking article discusses this claim. What is the verdict on the claim?",
                 "Given the article, how should the claim be rated?",
                 "Judge the claim using the article: true, false, a mixture, or unproven?"]


def pubhealth_text(text: str) -> str:
    text = text.strip()
    if text.startswith('"') and '""' in text:
        text = text[1:-1] if text.endswith('"') else text[1:]
        text = text.replace('""', '"')
    return squash(text)


def pubhealth(stats: dict):
    """Kotonya & Toni 2020, PUBHEALTH (MIT, github neemakot/Health-Fact-Checking). Public-health claims with the
    verdicts of journalist fact-checkers (Snopes, PolitiFact, Reuters, Health News Review, ...), normalised to
    true / false / mixture / unproven by the authors. train.tsv of the official Drive file; rows whose label is
    missing (the 28 rows broken by stray newlines in the TSV, label -1 in the HF loader) are skipped, as are rows
    with an article under 200 characters. State: the claim and the first ~2,500 characters of the fact-check
    article (cut at a sentence end; the full article is often 4-40k characters and ends with the ruling).
    Item = claim_id (rows whose normalised claim text repeats under another id share the first id)."""
    rng = random.Random(2026101962)
    csv.field_size_limit(10 ** 9)
    with zipfile.ZipFile(PUBHEALTH_ZIP) as z:
        text = z.read("PUBHEALTH/train.tsv").decode("utf-8")
    header, *rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    assert header == ["claim_id", "claim", "date_published", "explanation", "fact_checkers", "main_text", "sources", "label", "subjects"]
    counts = collections.Counter()
    first_id = {}
    for r in rows:
        if len(r) == 9 and r[7] in PUBHEALTH_OPTIONS:
            first_id.setdefault(norm(pubhealth_text(r[1])), r[0])
    keys = list(PUBHEALTH_OPTIONS)
    for i, r in enumerate(rows):
        if len(r) != 9 or r[7] not in PUBHEALTH_OPTIONS:
            counts["no_label"] += 1
            continue
        claim, article = pubhealth_text(r[1]), pubhealth_text(r[5])
        if len(article) < 200 or not claim:
            counts["short_article"] += 1
            continue
        state = {"claim": claim, "article": head(article, PUBHEALTH_CHARS)}
        order = keys[:]
        rng.shuffle(order)
        question, target = common.choice(state, rng.choice(PUBHEALTH_ASK), {k: PUBHEALTH_OPTIONS[k] for k in order}, order.index(r[7]))
        counts[r[7]] += 1
        group = re.sub(r"\W+", "-", first_id[norm(claim)])
        yield common.example(PUBHEALTH, f"{group}:r{i}", "health-fact-check", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- HEAD-QA: Spanish healthcare specialisation exams --------------------------------------------------------
HEADQA = "headqa-train"
HEADQA_ZIP = hub_dir("dvilares/head_qa", "d444ead98dd5") / "data/head-qa-es-en-pdfs.zip"
HEADQA_FILES = {"es": "HEAD/train_HEAD.json", "en": "HEAD_EN/train_HEAD_EN.json"}
HEADQA_ASK = {"en": ["This question is from a Spanish healthcare specialisation exam. Choose the correct answer.",
                     "Answer the exam question.", "Which option is correct?"],
              "es": ["Esta pregunta es de un examen de especialización sanitaria. Elige la respuesta correcta.",
                     "Responde a la pregunta del examen.", "¿Qué opción es la correcta?"]}
HEADQA_SUBJECT = {"en": {"biology": "biology", "nursery": "nursing", "pharmacology": "pharmacology", "medicine": "medicine",
                         "psychology": "psychology", "chemistry": "chemistry"},
                  "es": {"biology": "biología", "nursery": "enfermería", "pharmacology": "farmacia", "medicine": "medicina",
                         "psychology": "psicología", "chemistry": "química"}}


def headqa(stats: dict):
    """Vilares & Gómez-Rodríguez 2019, HEAD-QA (MIT dataset licence; exams of the Spanish Ministerio de Sanidad,
    reused under its terms: no distortion of the content, cite the source, state the last update - the 2013 and
    2014 MIR/EIR/BIR/FIR/QIR/PIR exams as released in HEAD-QA v1.0, 2019). The official answer keys are the gold.
    train_HEAD.json (Spanish original) and train_HEAD_EN.json (the authors' English machine translation) read
    from the zip directly (no loader script); the exam's own 5 options in the exam's order (not shuffled: some
    options refer to the others, "todas las anteriores", and the terms forbid altering the content; the keys are
    spread over positions 1-5 already). Questions tied to an image, and the few with repeated option texts, are
    skipped. Item = exam (one booklet: category and year), so both languages of a question stay together."""
    rng = random.Random(2026101963)
    counts = collections.Counter()
    with zipfile.ZipFile(HEADQA_ZIP) as z:
        data = {lang: json.loads(z.read(path)) for lang, path in HEADQA_FILES.items()}
    for lang in ("es", "en"):
        for name, exam in sorted(data[lang]["exams"].items()):
            for q in exam["data"]:
                if q.get("image") or re.search(r"\bimagen? n", q["qtext"], re.I):
                    counts["image"] += 1
                    continue
                answers = sorted(q["answers"], key=lambda a: a["aid"])
                texts = [squash(a["atext"]) for a in answers]
                gold = [str(a["aid"]) for a in answers].index(str(q["ra"])) if str(q["ra"]) in [str(a["aid"]) for a in answers] else None
                if gold is None or len({norm(t) for t in texts}) != len(texts) or not all(texts):
                    counts["bad_options"] += 1
                    continue
                state = {"subject": HEADQA_SUBJECT[lang][exam["category"]], "question": squash(q["qtext"])}
                question, target = common.choice(state, rng.choice(HEADQA_ASK[lang]), texts, gold)
                counts[f"{lang}_{len(texts)}"] += 1
                yield common.example(HEADQA, f"{name}:{lang}{q['qid']}", "healthcare-exam", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- Civil Comments: crowd toxicity ratings -------------------------------------------------------------------
CIVIL = "civilcomments-train"
CIVIL_DIR = hub_dir("google/civil_comments", "f2970eb3a557") / "data"
CIVIL_PER_CLASS = 15_000
CIVIL_ATTR_PER_CLASS = 625  # x 2 classes x 4 attributes = 5,000
CIVIL_ASK = ["Is this comment toxic (rude, disrespectful, or likely to make someone leave the discussion)?",
             "Would most readers find this comment toxic?", "Is this online comment toxic?"]
CIVIL_ATTR_ASK = {
    "insult": ["Is this comment insulting toward a person or group?", "Does the comment insult someone?"],
    "threat": ["Does this comment threaten someone (an intention to hurt or harm)?", "Is there a threat in this comment?"],
    "identity_attack": ["Does this comment attack people for their identity (race, religion, gender, sexuality, ...)?",
                        "Is this comment a negative or hateful remark about an identity group?"],
    "obscene": ["Is this comment obscene (swearing, vulgar or sexual language)?", "Does the comment use obscene language?"],
}


def civilcomments(stats: dict):
    """Borkan et al. 2019, Civil Comments (CC0-1.0, HF google/civil_comments). Each value is the fraction of
    crowd raters who marked the attribute. Train split only. Kept: a fraction >= 0.5 is
    yes, <= 0.1 is no, anything between is skipped as unclear. 15,000 yes + 15,000 no toxicity questions (a
    seeded sample of the comments, duplicate texts removed), and up to 5,000 questions on insult / threat /
    identity attack / obscene (625 yes + 625 no each; half the no examples are toxic comments that lack the
    attribute). The HF version has no comment id, so item = a digest of the comment text."""
    rng = random.Random(2026101964)
    cols = ["text", "toxicity", *CIVIL_ATTR_ASK]
    counts = collections.Counter()
    values = collections.defaultdict(list)
    for f in sorted(glob.glob(str(CIVIL_DIR / "train-*.parquet"))):
        t = pq.read_table(f, columns=cols)
        for c in cols:
            values[c].extend(t[c].to_pylist())
    texts = values.pop("text")
    seen, unique = set(), []
    for i, text in enumerate(texts):
        key = digest(text)
        if text and text.strip() and key not in seen:
            seen.add(key)
            unique.append(i)
    counts["unique_comments"] = len(unique)
    tox = values["toxicity"]
    yes = [i for i in unique if tox[i] >= .5]
    no = [i for i in unique if tox[i] <= .1]
    picked = [(i, True) for i in rng.sample(yes, min(CIVIL_PER_CLASS, len(yes)))] + \
             [(i, False) for i in rng.sample(no, min(CIVIL_PER_CLASS, len(no)))]
    out = []
    for i, truth in picked:
        question, target = common.noul(rng.choice(CIVIL_ASK), truth)
        out.append((i, "toxic", question, target))
        counts[f"toxic_{'yes' if truth else 'no'}"] += 1
    for attr in CIVIL_ATTR_ASK:
        v = values[attr]
        pos = [i for i in unique if v[i] >= .5]
        neg_toxic = [i for i in unique if v[i] <= .1 and tox[i] >= .5]
        neg_any = [i for i in unique if v[i] <= .1]
        chosen_pos = rng.sample(pos, min(CIVIL_ATTR_PER_CLASS, len(pos)))
        chosen_neg = rng.sample(neg_toxic, min(len(chosen_pos) // 2, len(neg_toxic)))
        rest = set(chosen_neg)
        chosen_neg += [i for i in rng.sample(neg_any, len(chosen_pos)) if i not in rest][:len(chosen_pos) - len(chosen_neg)]
        for i, truth in [(i, True) for i in chosen_pos] + [(i, False) for i in chosen_neg]:
            question, target = common.noul(rng.choice(CIVIL_ATTR_ASK[attr]), truth)
            out.append((i, attr, question, target))
            counts[f"{attr}_{'yes' if truth else 'no'}"] += 1
    out.sort(key=lambda x: (x[0], x[1]))
    for i, kind, question, target in out:
        text = texts[i].strip()
        yield common.example(CIVIL, f"c{digest(text)}:{kind}", "toxicity-rating", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- HateXplain: hate speech / offensive / normal --------------------------------------------------------------
HATEX = "hatexplain-train"
HATEX_DIR = RAW / "punyajoy__HateXplain@01d742279dac/Data"
HATEX_OPTIONS = {"hate speech": "attacks or demeans a group of people because of who they are",
                 "offensive": "offensive or abusive, but not hate speech", "normal": "neither hateful nor offensive"}
HATEX_LABEL = {"hatespeech": 0, "offensive": 1, "normal": 2}
HATEX_ASK = ["How should this social-media post be classified?", "Is this post hate speech, offensive, or normal?",
             "A moderator is reviewing this post. Which category does it belong to?"]


def hatexplain(stats: dict):
    """Mathew et al. 2021, HateXplain (MIT, github punyajoy/HateXplain). Twitter and Gab posts, each labelled by
    3 crowd annotators as hate speech, offensive or normal; the gold is the majority (posts without a majority
    are skipped; the official train ids contain none). Train ids of post_id_divisions.json only. User mentions
    and URLs are the dataset's own <user> / <url> placeholders. Item = post id."""
    rng = random.Random(2026101965)
    data = json.loads((HATEX_DIR / "dataset.json").read_text())
    train = json.loads((HATEX_DIR / "post_id_divisions.json").read_text())["train"]
    counts = collections.Counter()
    for pid in sorted(train):
        post = data[pid]
        votes = collections.Counter(a["label"] for a in post["annotators"]).most_common()
        if votes[0][1] < 2 or votes[0][0] not in HATEX_LABEL:
            counts["no_majority"] += 1
            continue
        text = squash(" ".join(post["post_tokens"]))
        keys = list(HATEX_OPTIONS)
        rng.shuffle(keys)
        gold = keys.index(list(HATEX_OPTIONS)[HATEX_LABEL[votes[0][0]]])
        question, target = common.choice(text, rng.choice(HATEX_ASK), {k: HATEX_OPTIONS[k] for k in keys}, gold)
        counts[f"{votes[0][0]}_{votes[0][1]}of3"] += 1
        yield common.example(HATEX, f"{pid}:label", "hate-speech-rating", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- MultiWOZ 2.2: active intent and requested slot values in task dialogues ------------------------------------
MULTIWOZ = "multiwoz-train"
MWOZ_DIR = RAW / "budzianowski__multiwoz@fe0c8e65cfcd/data/MultiWOZ_2.2"
# the active_intent names used in the dialogue files (the schema calls the taxi and police ones book_taxi / police)
MWOZ_INTENTS = {"find_hotel": "find a hotel", "book_hotel": "book a hotel", "find_train": "find a train",
                "book_train": "book train tickets", "find_attraction": "find an attraction", "find_restaurant": "find a restaurant",
                "book_restaurant": "book a restaurant table", "find_hospital": "find a hospital department",
                "find_taxi": "book a taxi", "find_bus": "find a bus", "find_police": "find the police station"}
MWOZ_SERVICE_INTENTS = {"hotel": ["find_hotel", "book_hotel"], "train": ["find_train", "book_train"], "attraction": ["find_attraction"],
                        "restaurant": ["find_restaurant", "book_restaurant"], "hospital": ["find_hospital"], "taxi": ["find_taxi"],
                        "bus": ["find_bus"], "police": ["find_police"]}
MWOZ_INTENT_CAP, MWOZ_SLOT_CAP = 12_000, 18_000
MWOZ_INTENT_ASK = ["What is the user trying to do at this point of the conversation?",
                   "Which task is the user pursuing in their latest message?",
                   "At the user's last turn, which request is active?"]
MWOZ_SLOT_ASK = ["Given the conversation so far, what does the user want for: {slot}?",
                 "Which value has the user asked for ({slot})?", "What is the user's choice for the {slot} at this point?"]
MWOZ_ANY = "any (no preference)"
NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def mwoz_key(value: str) -> str:
    """Comparison key: times as h:mm, number words as digits, else the normalised text."""
    v = value.lower().strip()
    m = re.fullmatch(r"(?:after |before |around |by )?(\d{1,2})[:.](\d{2})", v)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return NUMBER_WORDS.get(v, norm(v))


def mwoz_same(a: str, b: str) -> bool:
    ka, kb = mwoz_key(a), mwoz_key(b)
    if not ka or not kb:
        return True
    if ka == kb or ka in kb or kb in ka:
        return True
    ta, tb = set(ka.split()) - {"the", "hotel", "guest", "house", "restaurant", "college"}, set(kb.split()) - {"the", "hotel", "guest", "house", "restaurant", "college"}
    if ta and ta == tb:
        return True
    return difflib.SequenceMatcher(None, ka, kb).ratio() >= .8


def multiwoz(stats: dict):
    """Zang et al. 2020, MultiWOZ 2.2 (MIT, github budzianowski/multiwoz, commit pinned). Wizard-of-Oz dialogues
    between crowd workers; dialogue states (active intent and slot values per service) annotated by crowd
    workers and corrected in 2.1/2.2. Train dialogues only. Two questions on user turns, state = the dialogue up
    to and including that turn:
    - the active intent: frames whose active_intent is not NONE; with several, the one whose state changed this
      turn (else skipped). Options: the intents of the dialogue's services and two others, never another active
      intent, named by hand (schema intent names are terse).
    - a slot value that the turn sets or changes: the gold is the new value (the schema's alternative spellings
      are all excluded from the options); options: other values of that slot seen in the dialogue (states and
      span annotations), the schema's possible values, then values of that slot elsewhere in train - none equal,
      containing or near-identical (times compared as h:mm, typo-level similarity >= 0.8) to any gold spelling.
      "dontcare" is shown as "any (no preference)".
    Seeded sample: 12,000 intent + 18,000 slot questions. Item = dialogue_id."""
    rng = random.Random(2026101966)
    schema = {s["service_name"]: s for s in json.loads((MWOZ_DIR / "schema.json").read_text())}
    slot_info = {sl["name"]: sl for s in schema.values() for sl in s["slots"]}
    dialogs = [d for f in sorted(MWOZ_DIR.glob("train/dialogues_*.json")) for d in json.loads(f.read_text())]
    pool = collections.defaultdict(set)
    for d in dialogs:
        for t in d["turns"]:
            for f in t["frames"]:
                for k, v in f.get("state", {}).get("slot_values", {}).items():
                    pool[k].update(x for x in v if x != "dontcare")
    pool = {k: sorted(v) for k, v in pool.items()}
    counts = collections.Counter()
    candidates = {"intent": [], "slot": []}
    for di, d in enumerate(dialogs):
        prev = {}
        for ti, t in enumerate(d["turns"]):
            if t["speaker"] != "USER":
                continue
            active, changed = {}, set()
            for f in t["frames"]:
                st = f["state"]
                if st["active_intent"] != "NONE":
                    active[f["service"]] = st["active_intent"]
                for k, v in st["slot_values"].items():
                    if not {mwoz_key(x) for x in v} & {mwoz_key(x) for x in prev.get(k, [])}:
                        changed.add(f["service"])
                        candidates["slot"].append((di, ti, k))
                if st["requested_slots"]:
                    changed.add(f["service"])
                prev.update(st["slot_values"])
            if len(active) == 1:
                candidates["intent"].append((di, ti, next(iter(active))))
            elif active:
                live = [s for s in active if s in changed]
                if len(live) == 1:
                    candidates["intent"].append((di, ti, live[0]))
                else:
                    counts["intent_ambiguous"] += 1
    counts["intent_candidates"], counts["slot_candidates"] = len(candidates["intent"]), len(candidates["slot"])
    for kind, cap in (("intent", MWOZ_INTENT_CAP), ("slot", MWOZ_SLOT_CAP)):
        rng.shuffle(candidates[kind])
        candidates[kind] = candidates[kind][:cap]
    wanted = sorted([(di, ti, "intent", s) for di, ti, s in candidates["intent"]] + [(di, ti, "slot", k) for di, ti, k in candidates["slot"]])
    for di, ti, kind, what in wanted:
        d = dialogs[di]
        turn = d["turns"][ti]
        frames = {f["service"]: f["state"] for f in turn["frames"]}
        text = "\n".join(f"{'User' if u['speaker'] == 'USER' else 'System'}: {squash(u['utterance'])}" for u in d["turns"][:ti + 1])
        did = d["dialogue_id"].removesuffix(".json")
        if kind == "intent":
            gold = frames[what]["active_intent"]
            others_active = {st["active_intent"] for s, st in frames.items() if s != what and st["active_intent"] != "NONE"}
            near = [i for s in d["services"] for i in MWOZ_SERVICE_INTENTS.get(s, [])]
            rest = [i for i in MWOZ_INTENTS if i not in near]
            options = list(dict.fromkeys([gold] + near + rng.sample(rest, min(2, len(rest)))))
            options = [o for o in options if o == gold or o not in others_active]
            if len(options) < 3:
                options += [o for o in rng.sample(rest, len(rest)) if o not in options and o not in others_active][:3 - len(options)]
            rng.shuffle(options)
            question, target = common.choice(text, rng.choice(MWOZ_INTENT_ASK), [MWOZ_INTENTS[o] for o in options], options.index(gold))
            counts["intent"] += 1
            yield common.example(MULTIWOZ, f"{did}:t{turn['turn_id']}:intent", "dialogue-intent", text, question, target, "train")
            continue
        service = what.split("-")[0]
        golds = frames[service]["slot_values"][what]
        gold = MWOZ_ANY if golds[0] == "dontcare" else golds[0]
        info = slot_info.get(what, {})
        seen = []
        for u in d["turns"]:
            for f in u["frames"]:
                seen += f.get("state", {}).get("slot_values", {}).get(what, [])
                seen += [s["value"] for s in f.get("slots", []) if s.get("slot") == what and isinstance(s.get("value"), str)]
        in_dialog = list(dict.fromkeys(v.lower().strip() for v in seen if v != "dontcare"))
        rng.shuffle(in_dialog)
        possible = [v for v in info.get("possible_values") or [] if not (v == "0" and "people" in what)]
        rng.shuffle(possible)
        corpus = rng.sample(pool.get(what, []), min(30, len(pool.get(what, []))))
        distractors = []
        for v in in_dialog + possible + corpus:
            if len(distractors) == 4:
                break
            if v == "dontcare" or any(mwoz_same(v, g) for g in golds if g != "dontcare") or any(mwoz_same(v, x) for x in distractors):
                continue
            distractors.append(v)
        if gold != MWOZ_ANY and len(distractors) >= 2 and rng.random() < .15:
            distractors[-1] = MWOZ_ANY
        if len(distractors) < 2:
            counts["slot_few_options"] += 1
            continue
        options = [gold] + distractors
        rng.shuffle(options)
        desc = info.get("description", what.replace("-", " "))
        question, target = common.choice(text, rng.choice(MWOZ_SLOT_ASK).format(slot=desc), options, options.index(gold))
        counts["slot_dontcare" if gold == MWOZ_ANY else "slot"] += 1
        yield common.example(MULTIWOZ, f"{did}:t{turn['turn_id']}:{what}", "dialogue-state", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- MTOP: assistant intents in six languages -------------------------------------------------------------------
MTOP = "mtop-train"
MTOP_ZIP = RAW / "mtop/mtop.zip"
MTOP_LANGS = ["en", "de", "es", "fr", "hi", "th"]
MTOP_OTHER_CAP = 3_000
MTOP_ASK = ["What does the user want the assistant to do?", "Which intent does this request to a virtual assistant express?",
            "Classify the user's request to the assistant."]
MTOP_WORDS = {"AIRQUALITY": "air quality", "RSVP": "RSVP"}


def mtop_name(intent: str) -> str:
    return " ".join(MTOP_WORDS.get(w, w.lower()) for w in intent.removeprefix("IN:").split("_"))


def mtop(stats: dict):
    """Li et al. 2021, MTOP (CC-BY-SA-4.0, mtop/LICENSE.txt in the official zip dl.fbaipublicfiles.com/mtop/mtop.zip).
    Utterances to a virtual assistant annotated with a top-level intent (117 intents, 11 domains); the five
    non-English sets are human translations of the English ones. Train files only: every English row and a
    seeded 3,000 per other language. Options: the gold intent, up to 9 other intents of the same domain, then
    intents of other domains (10 in all), written as words (IN:GET_WEATHER -> "get weather").
    Item = the utterance ID, which the translations share."""
    rng = random.Random(2026101967)
    counts = collections.Counter()
    rows = {}
    by_domain = collections.defaultdict(set)
    with zipfile.ZipFile(MTOP_ZIP) as z:
        for lang in MTOP_LANGS:
            rows[lang] = [line.split("\t") for line in z.read(f"mtop/{lang}/train.txt").decode("utf-8").splitlines() if line.strip()]
            for r in rows[lang]:
                by_domain[r[4]].add(r[1])
    intents = sorted({i for v in by_domain.values() for i in v})
    for lang in MTOP_LANGS:
        chosen = rows[lang] if lang == "en" else rng.sample(rows[lang], min(MTOP_OTHER_CAP, len(rows[lang])))
        for r in sorted(chosen, key=lambda r: r[0]):
            uid, gold, utterance, domain = r[0], r[1], squash(r[3]), r[4]
            same = sorted(by_domain[domain] - {gold})
            near = rng.sample(same, min(9, len(same)))
            rest = [i for i in intents if i != gold and i not in near]
            options = [gold] + near + rng.sample(rest, 9 - len(near))
            rng.shuffle(options)
            question, target = common.choice(utterance, rng.choice(MTOP_ASK), [mtop_name(o) for o in options], options.index(gold))
            counts[lang] += 1
            yield common.example(MTOP, f"{uid}:{lang}", "assistant-intent", utterance, question, target, "train")
    stats["counts"] = dict(counts)


# --- SNIPS: seven assistant intents -------------------------------------------------------------------------------
SNIPS = "snips-train"
SNIPS_DIR = RAW / "sonos__nlu-benchmark@b86ac7f15778/2017-06-custom-intent-engines"
SNIPS_INTENTS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
SNIPS_OPTIONS = {"add to playlist": "add a song or artist to a playlist", "book restaurant": "reserve a table at a restaurant",
                 "get weather": "ask about the weather", "play music": "play a song, album, artist or genre",
                 "rate book": "give a rating to a book", "search creative work": "look for a book, film, song, show or game",
                 "search screening event": "look for movie showtimes or cinemas"}
SNIPS_ASK = ["Which of these voice-assistant intents does the request express?", "What does the user ask the assistant to do?",
             "Pick the intent of this spoken command."]


def snips(stats: dict):
    """Coucke et al. 2018, SNIPS NLU benchmark 2017-06 custom intent engines (CC0-1.0, github sonos/nlu-benchmark,
    commit pinned). Crowd-written queries for 7 intents; train_<Intent>_full.json files only (the validate files,
    used as the test set elsewhere, are not read). Rows with undecodable bytes (a few in PlayMusic) are skipped.
    Item = intent + index in its file."""
    rng = random.Random(2026101968)
    counts = collections.Counter()
    keys = list(SNIPS_OPTIONS)
    for k, intent in enumerate(SNIPS_INTENTS):
        assert keys[k] == readable_camel(intent).lower()
        data = json.loads((SNIPS_DIR / intent / f"train_{intent}_full.json").read_bytes().decode("utf-8", errors="replace"))
        for i, row in enumerate(data[intent]):
            text = squash("".join(part["text"] for part in row["data"]))
            if "�" in text or not text:
                counts["bad_text"] += 1
                continue
            order = keys[:]
            rng.shuffle(order)
            question, target = common.choice(text, rng.choice(SNIPS_ASK), {o: SNIPS_OPTIONS[o] for o in order}, order.index(keys[k]))
            counts[intent] += 1
            yield common.example(SNIPS, f"{intent}-{i}", "assistant-intent", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- ReDial: what the seeker liked, and who brought a movie up ---------------------------------------------------
REDIAL = "redial-train"
REDIAL_DIR = hub_dir("community-datasets/re_dial", "eb588549012b") / "data"
REDIAL_LIKED_ASK = ["Based on the conversation, did the seeker like {movie}?", "Does the seeker say they liked {movie}?",
                    "From what the seeker says, is {movie} a movie they liked?"]
REDIAL_SUGGESTED_ASK = ["Was {movie} suggested by the recommender (rather than brought up by the seeker)?",
                        "Did the recommender suggest {movie}?", "Is {movie} one of the recommender's suggestions?"]
REDIAL_LIKED_RATIO = 2  # liked questions kept per disliked one
REDIAL_SUGGESTED_CAP = 22_000


def redial_answer(a: int | None, b: int | None) -> int | None:
    """One label from the two participants' forms: both (or the only one) must say the same 0/1."""
    given = [x for x in (a, b) if x is not None]
    if not given or any(x not in (0, 1) for x in given) or len(set(given)) > 1:
        return None
    return given[0]


def redial(stats: dict):
    """Li et al. 2018, ReDial (CC-BY-4.0, HF community-datasets/re_dial train parquet). Crowd workers paired as a
    movie seeker and a recommender; after the chat both filled a form per mentioned movie (suggested by the
    recommender or mentioned by the seeker; seen; liked / disliked / did not say). A label is used when the
    forms agree (or only one form exists); "did not say" and disagreements are skipped. Questions: did the seeker
    like the movie (all dislikes, and liked ones sampled 2 per dislike since 94% are likes), and was it suggested
    by the recommender (seeded sample of 22,000). The chat is shown with the @movieId tokens replaced by the
    movie names, as Seeker / Recommender turns. Item = conversationId."""
    rng = random.Random(2026101969)
    rows = pq.read_table(REDIAL_DIR / "train-00000-of-00001.parquet").to_pylist()
    counts = collections.Counter()
    liked, disliked, suggested = [], [], []
    texts = {}
    for r in sorted(rows, key=lambda r: r["conversationId"]):
        names = {m["movieId"]: squash(m["movieName"]) for m in r["movieMentions"] or []}
        lines, mentioned = [], set()
        for m in r["messages"]:
            def name(match):
                mentioned.add(match.group(1))
                return names.get(match.group(1), match.group(0))
            text = re.sub(r"@(\d+)", name, m["text"])
            speaker = "Seeker" if m["senderWorkerId"] == r["initiatorWorkerId"] else "Recommender"
            if lines and lines[-1].startswith(speaker + ":"):
                lines[-1] += " " + squash(text)
            else:
                lines.append(f"{speaker}: {squash(text)}")
        if any(f"@{x}" in "\n".join(lines) for x in mentioned):
            counts["unresolved_mention"] += 1
            continue
        cid = r["conversationId"]
        texts[cid] = "\n".join(lines)
        forms = collections.defaultdict(dict)
        for side in ("initiatorQuestions", "respondentQuestions"):
            for q in r[side] or []:
                forms[q["movieId"]][side] = q
        for movie_id, f in sorted(forms.items()):
            if movie_id not in names or movie_id not in mentioned:
                counts["movie_not_in_chat"] += 1
                continue
            a, b = f.get("initiatorQuestions"), f.get("respondentQuestions")
            like = redial_answer(a and a["liked"], b and b["liked"])
            sugg = redial_answer(a and a["suggested"], b and b["suggested"])
            if like is None:
                counts["liked_skipped"] += 1
            else:
                (liked if like else disliked).append((cid, movie_id, names[movie_id]))
            if sugg is None:
                counts["suggested_skipped"] += 1
            else:
                suggested.append((cid, movie_id, names[movie_id], sugg))
    picked = [(c, m, n, "liked", True) for c, m, n in rng.sample(liked, min(len(liked), REDIAL_LIKED_RATIO * len(disliked)))]
    picked += [(c, m, n, "liked", False) for c, m, n in disliked]
    picked += [(c, m, n, "suggested", bool(s)) for c, m, n, s in rng.sample(suggested, min(len(suggested), REDIAL_SUGGESTED_CAP))]
    for cid, movie_id, movie, kind, truth in sorted(picked):
        ask = REDIAL_LIKED_ASK if kind == "liked" else REDIAL_SUGGESTED_ASK
        question, target = common.noul(rng.choice(ask).format(movie=movie), truth)
        counts[f"{kind}_{'yes' if truth else 'no'}"] += 1
        yield common.example(REDIAL, f"{cid}:{movie_id}:{kind}", "dialogue-preference", texts[cid], question, target, "train")
    stats["counts"] = dict(counts)


BUILDERS = [
    (MAUD, maud, {"upstream": "theatticusproject/maud@37d5c3b9 MAUD_v1/MAUD_train.csv + MAUD_dev.csv", "licence": "CC-BY-4.0",
                  "licence_url": "https://huggingface.co/datasets/theatticusproject/maud",
                  "label_origin": "human (lawyers; ABA Public Target Deal Points Study)"}),
    (PUBHEALTH, pubhealth, {"upstream": "PUBHEALTH train.tsv (Google Drive 1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj, linked from github neemakot/Health-Fact-Checking@02136d3e)",
                            "licence": "MIT", "licence_url": "https://github.com/neemakot/Health-Fact-Checking/blob/master/LICENSE",
                            "label_origin": "human (journalist fact-checkers' verdicts, normalised to 4 labels by the authors)"}),
    (HEADQA, headqa, {"upstream": "dvilares/head_qa@d444ead9 data/head-qa-es-en-pdfs.zip HEAD/train_HEAD.json + HEAD_EN/train_HEAD_EN.json",
                      "licence": "MIT (dataset); exams reused under the Spanish Ministerio de Sanidad terms (no distortion, cite the "
                                 "source, state the last update: 2013-2014 exams as released in HEAD-QA v1.0, 2019)",
                      "licence_url": "https://github.com/aghie/head-qa; https://aghie.github.io/head-qa/#legal-requirements",
                      "label_origin": "human (official exam answer keys); English text is the authors' machine translation"}),
    (CIVIL, civilcomments, {"upstream": "google/civil_comments@f2970eb3 data/train-*.parquet", "licence": "CC0-1.0",
                            "licence_url": "https://huggingface.co/datasets/google/civil_comments",
                            "label_origin": "human (fraction of crowd raters; >= 0.5 yes, <= 0.1 no)"}),
    (HATEX, hatexplain, {"upstream": "github punyajoy/HateXplain@01d74227 Data/dataset.json, train ids of Data/post_id_divisions.json",
                         "licence": "MIT", "licence_url": "https://github.com/punyajoy/HateXplain/blob/master/LICENSE",
                         "label_origin": "human (majority of 3 crowd annotators)"}),
    (MULTIWOZ, multiwoz, {"upstream": "github budzianowski/multiwoz@fe0c8e65 data/MultiWOZ_2.2 train + schema.json", "licence": "MIT",
                          "licence_url": "https://github.com/budzianowski/multiwoz/blob/master/LICENSE",
                          "label_origin": "human (crowd dialogue-state annotation); distractor options programmatic"}),
    (MTOP, mtop, {"upstream": "dl.fbaipublicfiles.com/mtop/mtop.zip <lang>/train.txt (en all, 3k per other language)",
                  "licence": "CC-BY-SA-4.0", "licence_url": "https://dl.fbaipublicfiles.com/mtop/mtop.zip (mtop/LICENSE.txt)",
                  "label_origin": "human (annotated intents; human translations)"}),
    (SNIPS, snips, {"upstream": "github sonos/nlu-benchmark@b86ac7f1 2017-06-custom-intent-engines/<Intent>/train_<Intent>_full.json",
                    "licence": "CC0-1.0", "licence_url": "https://github.com/sonos/nlu-benchmark/blob/master/LICENSE",
                    "label_origin": "human (crowd-written per intent)"}),
    (REDIAL, redial, {"upstream": "community-datasets/re_dial@eb588549 data/train-00000-of-00001.parquet", "licence": "CC-BY-4.0",
                      "licence_url": "https://redialdata.github.io/website/",
                      "label_origin": "human (both participants' movie forms, agreeing)"}),
]


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
                common.write(name, build(meta), meta)
