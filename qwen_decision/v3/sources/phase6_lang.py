"""Phase-6 sources: entities, relations, word sense and other languages (2026-10-19).

  python -m qwen_decision.v3.sources.phase6_lang --fetch [key ...]   # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase6_lang [name ...]          # build data/phase2/<name>.jsonl

Licences, download locations and fields were vetted in runs/analysis/phase6-candidates-vetted.json and re-read
on the pinned upstream files (LICENSE/README/card of each). Builders follow runs/analysis/phase6-builder-conventions.md:
human or computed gold only, train (or dev where the dataset's own docs allow it) splits only, files read and never
run, our own instruction phrasings, the first id segment is the grouping unit for the dev split.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import detokenize, fetch_spec, hub_dir

RAW = common.RAW / "phase6"
NLTK_DATA = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"  # nltk/nltk_data, branch gh-pages

# key -> ("github", repo, commit, [paths kept]) | ("hub", repo, [allow patterns]) | ("url", url, file)
FETCH = {
    "fewrel": ("github", "thunlp/FewRel", "278a2315d2138810a379cd8d5718914dc56e2582",
               ["LICENSE", "README.md", "data/train_wiki.json", "data/pid2name.json"]),
    "gap": ("github", "google-research-datasets/gap-coreference", "83135f26025cdd705c04f77e940d0c9201116b80",
            ["LICENSE", "README.md", "gap-development.tsv", "gap-validation.tsv"]),
    "nltk_semcor": ("url", f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA}/packages/corpora/semcor.zip", "semcor.zip"),
    "nltk_wordnet": ("url", f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA}/packages/corpora/wordnet.zip", "wordnet.zip"),
    "klue": ("hub", "klue/klue", ["README.md", "ynat/train-*", "nli/train-*", "re/train-*"]),
    "chid": ("hub", "thu-coai/chid", ["README.md", "original/train_data.txt"]),
    "cmrc2018": ("hub", "hfl/cmrc2018", ["README.md", "data/train-*"]),
    "drcd": ("github", "DRCKnowledgeTeam/DRCD", "b944790de5af02c5fbb7cd9cb1473d27d169eebf", ["README.md", "DRCD_training.json"]),
    "jglue": ("github", "yahoojapan/JGLUE", "6f071c09316baae89c3d083a90985b4b1cb9968c",
              ["LICENSE", "README.md", "datasets/jnli-v1.3/train-v1.3.json", "datasets/jcommonsenseqa-v1.3/train-v1.3.json"]),
    # primary_task train is 12 parquet files, 2.7 GB in all (under the 3 GB budget). secondary_task (the gold-passage
    # subset of the same questions, 27 MB) was fetched as the fallback but is not used: primary_task covers it.
    "tydiqa": ("hub", "google-research-datasets/tydiqa", ["README.md", "primary_task/train-*", "secondary_task/train-*"]),
}


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def squash(text: str) -> str:
    """Normalised text without any spacing, for containment checks in scripts written without spaces."""
    return re.sub(r"[\W_]+", "", str(text).lower())


NLI_OPTIONS = {"entailment": "the premise makes the hypothesis true",
               "neutral": "the premise neither confirms nor rules out the hypothesis",
               "contradiction": "the premise makes the hypothesis false"}
NLI_ASK = ["What is the relation between the premise and the hypothesis?",
           "Given the premise, is the hypothesis entailed, contradicted, or neither?",
           "Does the premise support the hypothesis, rule it out, or leave it open?"]


# --- FewRel 1.0: which Wikidata relation a sentence states between two marked entities -----------------------
FEWREL = "fewrel-train"
FEWREL_DIR = RAW / "thunlp__FewRel@278a2315d213/data"
FEWREL_ASK = ["Which relation does the sentence state from the head entity << >> to the tail entity [[ ]]?",
              "Read the sentence. How is the head entity (<< >>) related to the tail entity ([[ ]])? Read it as head - relation - tail.",
              "Pick the relation that holds from the head (<< >>) to the tail ([[ ]]) according to this sentence."]
FEWREL_PER_RELATION = 500  # of 700 per relation: 32,000 rows, every relation equally often
# Relations that can hold together for one ordered (head, tail) pair, so neither is a distractor when the other is gold.
# Inverses (owned by / subsidiary, tributary / mouth of the watercourse) stay: the question fixes the direction.
FEWREL_CLASH = [
    {"P131", "P276", "P706", "P17", "P495", "P27", "P551", "P937", "P740", "P159", "P1408", "P931", "P4552", "P403", "P1001"},
    {"P150", "P527", "P355", "P674", "P974"},
    {"P127", "P137", "P176", "P178", "P123", "P750", "P264", "P449", "P466"},
    {"P57", "P58", "P86", "P175", "P1877"},
    {"P710", "P1923", "P1346", "P991"},
    {"P31", "P136", "P101", "P135", "P140", "P106", "P39"},
    {"P3450", "P118"},
    {"P306", "P400"},
]


def _fewrel_description(text: str) -> str | None:
    """The first clause of Wikidata's property description, without notes that cite other properties by id."""
    text = re.sub(r"\s*\([^()]*\bP\d+\b[^()]*\)", "", text)
    text = re.sub(r"\s*\[[^\[\]]*\]", "", text).replace("e.g.", "for example")
    first = re.split(r";|\.\s", text)[0]
    first = re.sub(r",?\s*\b(inverse of|see also|see|opposite of)\b[^,;]*\bP\d+\b.*$", "", first).strip().rstrip(". ")
    return None if not first or re.search(r"\bP\d+\b|Property:", first) else first


def fewrel(stats: dict):
    """Han et al. 2018, FewRel 1.0 (MIT, github thunlp/FewRel). Wikipedia sentences aligned to Wikidata triples by
    distant supervision and kept only when crowd annotators confirmed the sentence states the relation. train_wiki
    only (64 relations). Head marked << >>, tail [[ ]]; options: the gold relation and 9 others, each with its
    Wikidata name and description; relations that can hold for the same pair (FEWREL_CLASH) are never distractors.
    500 sentences per relation; one sentence is one item."""
    rng = random.Random(2026101951)
    data = json.loads((FEWREL_DIR / "train_wiki.json").read_text())
    names = json.loads((FEWREL_DIR / "pid2name.json").read_text())
    relations = sorted(data)
    shown = {p: names[p][0] for p in relations}
    assert len(set(shown.values())) == len(shown)
    options_of = {shown[p]: _fewrel_description(names[p][1]) for p in relations}
    clash = {p: set().union(*([g for g in FEWREL_CLASH if p in g] or [set()])) | {p} for p in relations}
    counts = collections.Counter()
    for pid in relations:
        rows = list(enumerate(data[pid]))
        rng.shuffle(rows)
        for i, r in sorted(rows[:FEWREL_PER_RELATION]):
            tokens = list(r["tokens"])
            head = [p for p in r["h"][2][0]]
            tail = [p for p in r["t"][2][0]]
            if set(head) & set(tail):
                counts["overlapping_mentions"] += 1
                continue
            marked = list(tokens)
            marked[head[0]], marked[tail[0]] = "<<" + marked[head[0]], "[[" + marked[tail[0]]
            marked[head[-1]], marked[tail[-1]] = marked[head[-1]] + ">>", marked[tail[-1]] + "]]"
            sentence = detokenize(marked)
            state = {"sentence": sentence, "head": detokenize([tokens[k] for k in head]), "tail": detokenize([tokens[k] for k in tail])}
            pool = [p for p in relations if p not in clash[pid]]
            chosen = [pid] + rng.sample(pool, 9)
            rng.shuffle(chosen)
            options = {shown[p]: options_of[shown[p]] for p in chosen}
            question, target = common.choice(state, rng.choice(FEWREL_ASK), options, chosen.index(pid))
            counts["rows"] += 1
            yield common.example(FEWREL, f"{digest(' '.join(tokens))}:{pid}:{i}", "relation-classification", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- GAP: which name an ambiguous pronoun refers to --------------------------------------------------------
GAP = "gap-train"
GAP_DIR = RAW / "google-research-datasets__gap-coreference@83135f26025c"
GAP_ASK = ["Who does the pronoun marked << >> refer to?", "In this passage, the marked pronoun << >> stands for which person?",
           "Resolve the pronoun in << >>: which of these does it point to?"]


def gap(stats: dict):
    """Webster et al. 2018, GAP (Apache-2.0, github google-research-datasets/gap-coreference). Wikipedia snippets
    with an ambiguous pronoun and two person names; raters marked whether each name corefers with the pronoun.
    development + validation files only (the README reserves test for evaluation). Options: name A, name B, or
    neither; rows where one name contains the other are left out. One Wikipedia page (URL) is one item, since
    several snippets come from the same page."""
    rng = random.Random(2026101952)
    counts = collections.Counter()
    for split, file in (("development", "gap-development.tsv"), ("validation", "gap-validation.tsv")):
        for r in csv.DictReader((GAP_DIR / file).open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            text, pronoun, offset = r["Text"], r["Pronoun"], int(r["Pronoun-offset"])
            a, b = r["A"].strip(), r["B"].strip()
            if text[offset:offset + len(pronoun)] != pronoun:
                counts["offset_mismatch"] += 1
                continue
            if not norm(a) or not norm(b) or norm(a) in norm(b) or norm(b) in norm(a):
                counts["names_overlap"] += 1
                continue
            a_ref, b_ref = r["A-coref"].upper() == "TRUE", r["B-coref"].upper() == "TRUE"
            if a_ref and b_ref:
                counts["both_true"] += 1
                continue
            neither = f"neither {a} nor {b}"
            gold = a if a_ref else b if b_ref else neither
            options = [a, b, neither]
            rng.shuffle(options)
            marked = f"{text[:offset]}<<{pronoun}>>{text[offset + len(pronoun):]}"
            question, target = common.choice(marked, rng.choice(GAP_ASK), options, options.index(gold))
            counts["A" if a_ref else "B" if b_ref else "neither"] += 1
            yield common.example(GAP, f"{digest(r['URL'])}:{r['ID']}", "coreference", marked, question, target, split)
    stats["counts"] = dict(counts)


# --- SemCor 3.0: which WordNet sense a word has in a sentence ----------------------------------------------
SEMCOR = "semcor-train"
SEMCOR_ZIP = RAW / "nltk_semcor/semcor.zip"
WORDNET_ZIP = RAW / "nltk_wordnet/wordnet.zip"
SEMCOR_ROWS = 30_000
SEMCOR_POS = {"1": ("n", "noun"), "2": ("v", "verb"), "3": ("a", "adjective"), "4": ("r", "adverb"), "5": ("a", "adjective")}
SEMCOR_ASK = ['Which meaning does "{word}" (marked << >>) have in this sentence?',
              'Choose the definition of the {pos} "{lemma}" that fits the marked word << >> here.',
              "In this sentence, what does the word marked << >> mean?"]
FRESH = common.ROOT / "data/fresh/fresh-v1.jsonl"


def wordnet() -> tuple[dict, dict, dict]:
    """WordNet 3.0 from the NLTK package, parsed from its files: sense key -> (pos, offset), (pos, lemma) -> synset
    offsets in sense order, (pos, offset) -> (definition without the example sentences, synonym lemmas)."""
    keys, senses, synsets = {}, {}, {}
    with zipfile.ZipFile(WORDNET_ZIP) as z:
        for line in z.read("wordnet/index.sense").decode().splitlines():
            key, offset, *_ = line.split()
            keys[key] = (SEMCOR_POS[key.split("%")[1][0]][0], offset)
        for pos, file in (("n", "noun"), ("v", "verb"), ("a", "adj"), ("r", "adv")):
            for line in z.read(f"wordnet/index.{file}").decode().splitlines():
                if line.startswith(" "):
                    continue
                f = line.split()
                n_ptr = int(f[3])
                n_senses = int(f[2])
                senses[(pos, f[0])] = f[6 + n_ptr:6 + n_ptr + n_senses]
            for line in z.read(f"wordnet/data.{file}").decode("latin-1").splitlines():
                if line.startswith(" "):
                    continue
                head, _, gloss = line.partition(" | ")
                f = head.split()
                words = [re.sub(r"\(.*\)$", "", f[4 + 2 * i]).replace("_", " ") for i in range(int(f[3], 16))]
                definition = re.split(r'[;:]\s*"', gloss)[0].strip().rstrip(";").strip()  # definition; "example"; ...
                synsets[(pos, f[0])] = (definition, words)
    return keys, senses, synsets


def _wic_sentences() -> list[str]:
    found = []
    for line in FRESH.open():
        r = json.loads(line)
        if r.get("source") == "wic":
            found += [norm(v) for k, v in r["state"].items() if k.startswith("sentence")]
    return found


def _semcor_tokens(s: ET.Element) -> list[tuple[str, dict]]:
    out = []
    for el in s:
        text = (el.text or "").strip()
        if not text:
            continue
        out.append(({"``": "\u201c", "''": "\u201d"}.get(text, text.replace("_", " ")), el.attrib if el.tag == "wf" else {}))
    return out


def _semcor_detok(words: list[str]) -> str:
    """Brown tokens to text; `` and '' (opening and closing quotes) became curly quotes in _semcor_tokens."""
    text = re.sub(r" ([?!]|n't)", r"\1", detokenize(words))
    return text.replace("\u201c ", "\u201c").replace(" \u201d", "\u201d")


def semcor(stats: dict):
    """SemCor 3.0 (Brown corpus sentences sense-tagged by Princeton lexicographers against WordNet 1.6, mapped to
    WordNet 3.0 by R. Mihalcea; NLTK package, Princeton WordNet licence) with WordNet 3.0 definitions (NLTK
    package), both read from the zip files by this code. One tagged word per sentence: a single sense tag, no
    proper-name tag, and a lemma with 2-10 senses for its part of speech. Options: the definitions of all senses of
    that lemma and part of speech (example sentences left out: WiC is built from WordNet examples), each with the
    synset's other lemmas. Sentences equal to, containing or contained in a fresh-v1 WiC sentence are dropped.
    30,000 rows; one Brown document is one item."""
    rng = random.Random(2026101953)
    keys, senses, synsets = wordnet()
    wic = [f" {w} " for w in _wic_sentences()]
    counts = collections.Counter()
    candidates = []
    with zipfile.ZipFile(SEMCOR_ZIP) as z:
        files = sorted(n for n in z.namelist() if re.fullmatch(r"semcor/brown(1|2|v)/tagfiles/br-\w+\.xml", n))
        for name in files:
            doc = name.rsplit("/", 1)[1].removesuffix(".xml")
            root = ET.fromstring(z.read(name))
            for s in root.iter("s"):
                tokens = _semcor_tokens(s)
                words = [t for t, _ in tokens]
                plain = _semcor_detok(words)
                if any(w in f" {norm(plain)} " or f" {norm(plain)} " in w for w in wic):
                    counts["dropped_wic_overlap"] += 1
                    continue
                eligible = []
                for i, (word, a) in enumerate(tokens):
                    if a.get("cmd") != "done" or "ot" in a or "pn" in a or not a.get("lexsn") or not a.get("lemma"):
                        continue
                    if ";" in a["lexsn"] or ";" in a.get("wnsn", "") or a.get("wnsn") == "0":
                        counts["multi_or_missing_sense"] += 1
                        continue
                    key = f"{a['lemma'].lower()}%{a['lexsn']}"
                    if key not in keys:
                        counts["key_not_in_wordnet"] += 1
                        continue
                    pos, offset = keys[key]
                    offsets = senses.get((pos, a["lemma"].lower()), [])
                    if not 2 <= len(offsets) <= 10:
                        counts["monosemous" if len(offsets) < 2 else "over_10_senses"] += 1
                        continue
                    if offset not in offsets:
                        counts["sense_not_listed"] += 1
                        continue
                    definitions = [synsets[(pos, o)][0] for o in offsets]
                    if len({norm(d) for d in definitions}) != len(definitions) or not all(norm(d) for d in definitions):
                        counts["duplicate_definitions"] += 1
                        continue
                    eligible.append((i, pos, a["lemma"].lower(), offsets, offset, key))
                if not eligible:
                    continue
                i, pos, lemma, offsets, offset, key = rng.choice(eligible)
                marked = _semcor_detok(words[:i] + [f"<<{words[i]}>>"] + words[i + 1:])
                candidates.append((doc, s.get("snum"), i, words[i], marked, pos, lemma, offsets, offset))
    stats["sentences_with_a_candidate"] = len(candidates)
    rng.shuffle(candidates)
    for doc, snum, i, word, marked, pos, lemma, offsets, offset in sorted(candidates[:SEMCOR_ROWS]):
        order = list(offsets)
        rng.shuffle(order)
        options = {}
        for o in order:
            definition, words = synsets[(pos, o)]
            others = [w for w in dict.fromkeys(words) if w.lower() != lemma.replace("_", " ")]
            options[definition] = f"also: {', '.join(others[:4])}" if others else None
        pos_name = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb"}[pos]
        ask = rng.choice(SEMCOR_ASK).format(word=word, lemma=lemma.replace("_", " "), pos=pos_name)
        question, target = common.choice({"sentence": marked}, ask, options, order.index(offset))
        counts[f"{pos_name}"] += 1
        yield common.example(SEMCOR, f"{doc}:{snum}:{i}", "word-sense", {"sentence": marked}, question, target, "train")
    stats["counts"] = dict(counts)


# --- KLUE: Korean news topics, NLI and relation extraction --------------------------------------------------
KLUE = "klue-train"
KLUE_DIR = hub_dir("klue/klue", "349481ec73ff")
KLUE_PER_TASK = 10_000
YNAT_TOPICS = {"IT과학": "IT and science", "경제": "economy", "사회": "society", "생활문화": "life and culture",
               "세계": "world", "스포츠": "sports", "정치": "politics"}
YNAT_ASK = ["Which topic does this Korean news headline belong to?", "Classify the topic of the headline.",
            "이 뉴스 제목은 어느 분야의 기사입니까?"]
KLUE_RE_ASK = ["Which relation does the sentence state between the subject << >> and the object [[ ]]?",
               "Read the Korean sentence. What does it say about how the subject (<< >>) relates to the object ([[ ]])?",
               "Pick the relation from the subject << >> to the object [[ ]] that this sentence expresses."]
KLUE_RE_NAMES = {
    "no_relation": "no relation between them is stated",
    "org:dissolved": "organization dissolved on (date)", "org:founded": "organization founded on (date)",
    "org:place_of_headquarters": "organization's headquarters location", "org:alternate_names": "other name of the organization",
    "org:member_of": "organization is a member of", "org:members": "organization has as a member",
    "org:political/religious_affiliation": "organization's political or religious affiliation", "org:product": "organization's product",
    "org:founded_by": "organization founded by", "org:top_members/employees": "organization's top member or employee",
    "org:number_of_employees/members": "organization's number of employees or members",
    "per:date_of_birth": "person's date of birth", "per:date_of_death": "person's date of death",
    "per:place_of_birth": "person's place of birth", "per:place_of_death": "person's place of death",
    "per:place_of_residence": "person's place of residence", "per:origin": "person's origin (nationality or ethnicity)",
    "per:employee_of": "person works for", "per:schools_attended": "person attended (school)",
    "per:alternate_names": "other name of the person", "per:parents": "person's parent", "per:children": "person's child",
    "per:siblings": "person's sibling", "per:spouse": "person's spouse", "per:other_family": "person's other family member",
    "per:colleagues": "person's colleague", "per:product": "person's work or product", "per:religion": "person's religion",
    "per:title": "person's title or position",
}
KLUE_RE_CLASH = [
    {"per:place_of_birth", "per:origin", "per:place_of_residence", "per:place_of_death"},
    {"per:employee_of", "per:schools_attended"}, {"per:alternate_names", "per:title"},
    {"per:parents", "per:children", "per:siblings", "per:spouse", "per:other_family", "per:colleagues"},
    {"org:member_of", "org:political/religious_affiliation", "org:members"},
    {"org:founded_by", "org:top_members/employees"}, {"org:product", "per:product"},
]


def _klue_read(task: str) -> list[dict]:
    return pq.read_table(KLUE_DIR / f"{task}/train-00000-of-00001.parquet").to_pylist()


def klue(stats: dict):
    """KLUE (Park et al. 2021; CC-BY-SA-4.0, HF klue/klue). Train splits of three tasks, up to 10,000 rows each:
    - YNAT: Yonhap news headlines, topic (7) chosen by at least 2 of 3 crowd annotators (KLUE paper section 3.1);
    - NLI: hypotheses written and labelled by crowd workers for premises from 6 Korean sources; whole premises
      are sampled so the 3 hypotheses of a premise stay together;
    - RE: Wikipedia/WikiTree sentences, subject << >> and object [[ ]], 30 annotated relations; options: the gold,
      "no relation" (when it is not the gold) and others of the subject's type first, 10 in all; relations that can
      hold together (KLUE_RE_CLASH) or that the same entity pair has elsewhere in the data are never distractors.
      Rows whose label does not fit the subject's type, or whose entity pair carries an overlapping relation in
      another sentence (birthplace here, origin there), are left out as doubtful.
    Items: the headline id (YNAT), the premise (NLI), the sentence (RE)."""
    rng = random.Random(2026101954)
    counts = collections.Counter()
    # YNAT
    rows = _klue_read("ynat")
    topics = list(YNAT_TOPICS)
    rng.shuffle(rows)
    for r in sorted(rows[:KLUE_PER_TASK], key=lambda r: r["guid"]):
        order = list(topics)
        rng.shuffle(order)
        gold = topics[r["label"]]
        question, target = common.choice(r["title"], rng.choice(YNAT_ASK), {t: YNAT_TOPICS[t] for t in order}, order.index(gold))
        counts["ynat"] += 1
        yield common.example(KLUE, f"{r['guid']}:topic", "topic-classification", r["title"], question, target, "train")
    # NLI
    assert list(NLI_OPTIONS) == ["entailment", "neutral", "contradiction"]  # KLUE NLI label ids 0, 1, 2
    by_premise = collections.defaultdict(list)
    for r in _klue_read("nli"):
        by_premise[r["premise"]].append(r)
    premises = sorted(by_premise)
    rng.shuffle(premises)
    taken = 0
    for premise in premises:
        if taken >= KLUE_PER_TASK:
            break
        for r in by_premise[premise]:
            state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
            question, target = common.choice(state, rng.choice(NLI_ASK), NLI_OPTIONS, r["label"])  # 0/1/2 as in NLI_OPTIONS
            taken += 1
            counts["nli"] += 1
            yield common.example(KLUE, f"nli{digest(premise)}:{r['guid']}", "nli", state, question, target, "train")
    # RE
    rows = _klue_read("re")
    names = list(KLUE_RE_NAMES)
    pair_labels = collections.defaultdict(set)
    for r in rows:
        pair_labels[(norm(r["subject_entity"]["word"]), norm(r["object_entity"]["word"]))].add(names[r["label"]])
    clash = {n: set().union(*([g for g in KLUE_RE_CLASH if n in g] or [set()])) | {n} for n in names}
    rng.shuffle(rows)
    taken = 0
    for r in rows:
        if taken >= KLUE_PER_TASK:
            break
        sentence, subj, obj = r["sentence"], r["subject_entity"], r["object_entity"]
        if sentence[subj["start_idx"]:subj["end_idx"] + 1] != subj["word"] or sentence[obj["start_idx"]:obj["end_idx"] + 1] != obj["word"]:
            counts["re_offset_mismatch"] += 1
            continue
        if not (subj["end_idx"] < obj["start_idx"] or obj["end_idx"] < subj["start_idx"]):
            counts["re_overlapping_entities"] += 1
            continue
        marks = sorted([(subj["start_idx"], subj["end_idx"], "<<", ">>"), (obj["start_idx"], obj["end_idx"], "[[", "]]")], reverse=True)
        marked = sentence
        for s, e, left, right in marks:
            marked = marked[:s] + left + marked[s:e + 1] + right + marked[e + 1:]
        gold = names[r["label"]]
        prefix = "org:" if subj["type"] == "ORG" else "per:"
        if gold != "no_relation" and (subj["type"] not in ("ORG", "PER") or not gold.startswith(prefix)):
            counts["re_subject_type_mismatch"] += 1  # e.g. a club typed PER: the annotation is doubtful
            continue
        seen = pair_labels[(norm(subj["word"]), norm(obj["word"]))]
        if (seen - {gold, "no_relation"}) & clash[gold]:
            counts["re_conflicting_pair"] += 1  # the same pair carries an overlapping relation elsewhere (birthplace vs origin)
            continue
        banned = clash[gold] | seen
        same = [n for n in names if n.startswith(prefix) and n not in banned]
        other = [n for n in names if n != "no_relation" and not n.startswith(prefix) and n not in banned]
        rng.shuffle(same)
        rng.shuffle(other)
        chosen = [gold] + (["no_relation"] if gold != "no_relation" else [])
        chosen += (same + other)[:10 - len(chosen)]
        rng.shuffle(chosen)
        state = {"sentence": marked, "subject": f"{subj['word']} ({subj['type']})", "object": f"{obj['word']} ({obj['type']})"}
        question, target = common.choice(state, rng.choice(KLUE_RE_ASK), [KLUE_RE_NAMES[n] for n in chosen], chosen.index(gold))
        taken += 1
        counts["re"] += 1
        counts["re_no_relation"] += gold == "no_relation"
        yield common.example(KLUE, f"re{digest(sentence)}:{r['guid']}", "relation-classification", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- ChID: which idiom fills a blank in a Chinese passage ---------------------------------------------------
CHID = "chid-train"
CHID_FILE = hub_dir("thu-coai/chid", "47ad8022d369") / "original/train_data.txt"
CHID_ROWS = 20_000
CHID_ASK = ["Which idiom best fills the blank ____ in this Chinese passage?", "选择最适合填入空白处（____）的成语。",
            "Pick the chengyu that fits the gap marked ____."]


def chid(stats: dict):
    """Zheng et al. 2019, ChID (Apache-2.0; github chujiezheng/ChID-Dataset, files from HF thu-coai/chid original/).
    News and novel passages where an idiom was cut out of human-written text (so the gold is the original idiom);
    the 7 candidates are the dataset's own (the gold, similar idioms and random ones, chosen by the authors' rules).
    Train split; 20,000 passages sampled, one blank each (the passage's other blanks are filled with their own
    idioms). One passage is one item."""
    rng = random.Random(2026101955)
    with CHID_FILE.open(encoding="utf-8") as f:
        total = sum(1 for _ in f)
    wanted = set(rng.sample(range(total), int(CHID_ROWS * 1.1)))
    counts = collections.Counter()
    picked = []
    with CHID_FILE.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n not in wanted:
                continue
            r = json.loads(line)
            pieces = r["content"].split("#idiom#")
            if len(pieces) - 1 != len(r["groundTruth"]) or len(r["candidates"]) != len(r["groundTruth"]):
                counts["malformed"] += 1
                continue
            j = rng.randrange(len(r["groundTruth"]))
            gold, cands = r["groundTruth"][j], list(r["candidates"][j])
            if gold not in cands or len(set(cands)) != len(cands) or not 2 <= len(cands) <= 10:
                counts["bad_candidates"] += 1
                continue
            text = pieces[0]
            for k, piece in enumerate(pieces[1:]):
                text += ("____" if k == j else r["groundTruth"][k]) + piece
            picked.append((n, j, text, gold, cands, r["content"]))
    rng.shuffle(picked)
    for n, j, text, gold, cands, content in sorted(picked[:CHID_ROWS]):
        rng.shuffle(cands)
        question, target = common.choice(text, rng.choice(CHID_ASK), cands, cands.index(gold))
        counts["rows"] += 1
        yield common.example(CHID, f"{digest(content)}:{n}:{j}", "idiom-cloze", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- CMRC2018 + DRCD: Chinese span questions as a choice among spans of the passage --------------------------
CMRC = "cmrc-train"
CMRC_FILE = hub_dir("hfl/cmrc2018", "137f2c45a242") / "data/train-00000-of-00001.parquet"
DRCD_FILE = RAW / "DRCKnowledgeTeam__DRCD@b944790de5af/DRCD_training.json"
CMRC_ROWS = 15_000
CMRC_ASK = ["Answer the question from the passage.", "根据文章，选出问题的正确答案。", "Which span of the passage answers the question?"]


def _shape(text: str) -> str | None:
    """'1963年' -> '0年': numbers, dates and counts get a shape so that other spans of the same shape can be found;
    a title in 《》 has the shape 'title' (other titles in the passage)."""
    if re.fullmatch(r"《[^《》]+》", text):
        return "title"
    if not re.search(r"\d", text) or len(text) > 20:
        return None
    return re.sub(r"\d+(?:[.,]\d+)*", "0", text)


def _digits(text: str) -> str:
    return "".join(re.findall(r"\d", text))


def _cmrc_items():
    """(source, group, item id, passage, question, gold text, gold start, [(other answer text, start)])"""
    rows = pq.read_table(CMRC_FILE).to_pylist()
    by_context = collections.defaultdict(list)
    for r in rows:
        by_context[r["context"]].append(r)
    for context, qs in by_context.items():
        answers = [(q["answers"]["text"][0], q["answers"]["answer_start"][0]) for q in qs if q["answers"]["text"]]
        for q in qs:
            if not q["answers"]["text"]:
                continue
            yield ("cmrc2018", f"cmrc{digest(context)}", q["id"], context, q["question"],
                   q["answers"]["text"][0], q["answers"]["answer_start"][0], answers)
    data = json.loads(DRCD_FILE.read_text(encoding="utf-8"))["data"]
    for article in data:
        for p in article["paragraphs"]:
            answers = [(q["answers"][0]["text"], q["answers"][0]["answer_start"]) for q in p["qas"] if q.get("answers")]
            for q in p["qas"]:
                if not q.get("answers"):
                    continue
                yield ("drcd", f"drcd{article['id']}", q["id"], p["context"], q["question"],
                       q["answers"][0]["text"], q["answers"][0]["answer_start"], answers)


def cmrc(stats: dict):
    """CMRC2018 (Cui et al. 2019; CC-BY-SA-4.0, HF hfl/cmrc2018 train; Simplified Chinese Wikipedia) and DRCD
    (Shao et al. 2018; CC-BY-SA-3.0, github DRCKnowledgeTeam/DRCD DRCD_training.json; Traditional Chinese
    Wikipedia). Crowd-written questions with a span answer; options: the gold span and 3 distractors - spans of
    the same shape when the gold is a number, date or 《title》 ("1963年" -> other years in the passage), then the answers to
    the passage's other questions. A distractor never equals, contains, is contained in or overlaps the gold, and
    never has the gold's digits. 15,000 rows sampled from both; one passage (CMRC) or article (DRCD) is one item."""
    rng = random.Random(2026101956)
    counts = collections.Counter()
    built = []
    for source, group, qid, context, question, gold, start, answers in _cmrc_items():
        gold = gold.strip()
        if not gold or context[start:start + len(gold)] != gold:
            counts[f"{source}_offset_mismatch"] += 1
            continue
        g, gd = squash(gold), _digits(gold)

        def ok(text: str, at: int) -> bool:
            t = squash(text)
            if not t or t in g or g in t or (gd and _digits(text) == gd):
                return False
            return at < 0 or at + len(text) <= start or at >= start + len(gold)

        shape = _shape(gold)
        same_shape = []
        if shape == "title":
            same_shape = [(m.group(0), m.start()) for m in re.finditer(r"《[^《》]{1,40}》", context)]
        elif shape:
            pattern = re.escape(shape).replace("0", r"\d+(?:[.,]\d+)*")
            same_shape = [(m.group(0), m.start()) for m in re.finditer(rf"(?<!\d){pattern}", context)]
        others = [(t.strip(), a) for t, a in answers if t.strip()]
        rng.shuffle(same_shape)
        rng.shuffle(others)
        if shape:
            others.sort(key=lambda x: _shape(x[0]) != shape)  # same-shape answers of other questions first
        distractors = {}
        for text, at in same_shape + others:
            if len(distractors) == 3:
                break
            if ok(text, at) and all(squash(text) not in squash(d) and squash(d) not in squash(text) for d in distractors):
                distractors[text] = None
        if len(distractors) < 3:
            counts[f"{source}_few_distractors"] += 1
            continue
        built.append((source, group, qid, context, question, gold, list(distractors)))
    stats["candidates"] = len(built)
    rng.shuffle(built)
    for source, group, qid, context, question, gold, distractors in sorted(built[:CMRC_ROWS], key=lambda x: (x[1], x[2])):
        options = [gold] + distractors
        rng.shuffle(options)
        state = {"passage": context, "question": question}
        ask = rng.choice(CMRC_ASK)
        q, target = common.choice(state, ask, options, options.index(gold))
        counts[source] += 1
        yield common.example(CMRC, f"{group}:{qid}", "span-reading", state, q, target, "train")
    stats["counts"] = dict(counts)


# --- JGLUE: JCommonsenseQA and JNLI (Japanese) --------------------------------------------------------------
JGLUE = "jglue-train"
JGLUE_DIR = RAW / "yahoojapan__JGLUE@6f071c09316b/datasets"
JCQA_ASK = ["Choose the best answer to the question.", "質問に最も合う答えを選んでください。", "Which choice answers the question?"]


def jglue(stats: dict):
    """Kurihara et al. 2022, JGLUE v1.3 (CC-BY-SA-4.0, github yahoojapan/JGLUE). Crowd-built (Yahoo! Crowdsourcing):
    - JCommonsenseQA train: questions on ConceptNet concepts with their own 5 options; one question is one item;
    - JNLI train: premise/hypothesis pairs from YJ Captions with crowd labels (3-way). Both directions of a pair and
      all captions of the same image(s) are one item (image ids joined by union-find)."""
    rng = random.Random(2026101957)
    counts = collections.Counter()
    for r in map(json.loads, (JGLUE_DIR / "jcommonsenseqa-v1.3/train-v1.3.json").open(encoding="utf-8")):
        options = [r[f"choice{i}"] for i in range(5)]
        if len(set(options)) != 5:
            counts["jcqa_duplicate_options"] += 1
            continue
        order = list(range(5))
        rng.shuffle(order)
        question, target = common.choice(r["question"], rng.choice(JCQA_ASK), [options[i] for i in order], order.index(r["label"]))
        counts["jcqa"] += 1
        yield common.example(JGLUE, f"jcqa{r['q_id']}:q", "commonsense-qa", r["question"], question, target, "train")
    rows = list(map(json.loads, (JGLUE_DIR / "jnli-v1.3/train-v1.3.json").open(encoding="utf-8")))
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        images = r["yjcaptions_id"].split("-")[0].split("_")
        for im in images[1:]:
            parent[find(im)] = find(images[0])
        find(images[0])
    labels = list(NLI_OPTIONS)
    for r in rows:
        group = find(r["yjcaptions_id"].split("-")[0].split("_")[0])
        state = {"premise": r["sentence1"], "hypothesis": r["sentence2"]}
        question, target = common.choice(state, rng.choice(NLI_ASK), NLI_OPTIONS, labels.index(r["label"]))
        counts[f"jnli_{r['label']}"] += 1
        yield common.example(JGLUE, f"jnli{group}:{r['sentence_pair_id']}", "nli", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- TyDi QA (primary task): yes/no answers and whether a passage answers the question -----------------------
TYDI = "tydiqa-train"
TYDI_DIR = hub_dir("google-research-datasets/tydiqa", "da78f23f9119")
TYDI_ROWS = 30_000
TYDI_MAX_PASSAGE = 2500  # characters; longer passages are skipped rather than cut, so the answer is never cut off
TYDI_YESNO_ASK = ["Based on the passage, is the answer to the question yes?", "According to the passage, should the question be answered yes?",
                  "Using only the passage, answer the yes/no question: is it yes?"]
TYDI_ANSWERS_ASK = ["Does this passage answer the question?", "Can the question be answered from this passage?",
                    "Is the answer to the question in this passage?"]


def _tydi_keys(answer: str, question: str) -> list[str]:
    """Pieces of the minimal answer that a passage must not contain to count as not answering: the answer itself,
    its digit runs, and its words of 4+ characters that are not in the question."""
    q = squash(question)
    keys = [squash(answer)] + re.findall(r"\d+", answer)
    keys += [squash(w) for w in re.split(r"[\s\W]+", answer) if len(squash(w)) >= 4 and squash(w) not in q]
    return [k for k in dict.fromkeys(keys) if k]


def _tydi_footer(text: str) -> bool:
    """Category lists and similar footers ("Luokka:Vuoden 1960 rikokset"), which make trivial negatives."""
    lines = [l for l in text.splitlines() if l.strip()]
    return sum(bool(re.match(r"^[^\s:]{2,20}:\S", l)) for l in lines) * 2 > len(lines)


def tydiqa(stats: dict):
    """Clark et al. 2020, TyDi QA primary task (Apache-2.0, HF google-research-datasets/tydiqa, train; 11
    languages; questions written by native speakers before seeing the article, answers annotated by native
    speakers on the Wikipedia article). Two questions:
    - yes/no: questions whose annotation is YES or NO, asked over the annotated passage (all of them);
    - answerability (noul): for questions with a minimal answer span, the annotated passage (yes) and another
      passage of the same article (no), at random among those that are not next to it and contain neither the
      answer, its numbers nor its longer words, and are not category footers. (Picking the most question-like passage instead gave passages that
      partly answer "what is X?" questions, so it is not used.) Pairs are balanced across languages.
    Passages over 2,500 characters are skipped. 30,000 rows; one article (URL) is one item."""
    rng = random.Random(2026101958)
    files = sorted(glob.glob(str(TYDI_DIR / "primary_task/train-*.parquet")))
    yesno, spans = [], collections.defaultdict(list)
    for fi, file in enumerate(files):
        t = pq.read_table(file, columns=["annotations", "language"])
        for ri, (a, lang) in enumerate(zip(t.column("annotations").to_pylist(), t.column("language").to_pylist())):
            p, yn = a["passage_answer_candidate_index"][0], a["yes_no_answer"][0]
            if p < 0:
                continue
            if yn in ("YES", "NO"):
                yesno.append((fi, ri))
            elif a["minimal_answers_start_byte"][0] >= 0:
                spans[lang].append((fi, ri))
    pairs_per_language = (TYDI_ROWS - len(yesno)) // 2 // len(spans) + 1
    chosen = {}
    for lang in sorted(spans):
        rng.shuffle(spans[lang])
        for k in spans[lang][:int(pairs_per_language * 1.3)]:  # spares for examples without a usable negative
            chosen[k] = ("span", lang)
    for k in yesno:
        chosen[k] = ("yesno", None)
    counts = collections.Counter()
    rows, seen = [], set()
    for fi, file in enumerate(files):
        wanted = sorted(ri for (f, ri), _ in chosen.items() if f == fi)
        if not wanted:
            continue
        t = pq.read_table(file).take(wanted).to_pylist()
        for ri, r in zip(wanted, t):
            kind, lang = chosen[(fi, ri)]
            doc = r["document_plaintext"].encode("utf-8")
            starts, ends = r["passage_answer_candidates"]["plaintext_start_byte"], r["passage_answer_candidates"]["plaintext_end_byte"]
            passages = [doc[s:e].decode("utf-8", "replace").strip() for s, e in zip(starts, ends)]
            a = r["annotations"]
            p = a["passage_answer_candidate_index"][0]
            gold_passage = passages[p]
            if not gold_passage or len(gold_passage) > TYDI_MAX_PASSAGE:
                counts["passage_too_long"] += 1
                continue
            group = digest(r["document_url"])
            qid = digest(r["language"] + r["question_text"] + r["document_url"])
            if qid in seen:
                counts["repeated_question"] += 1  # the same question on the same article appears more than once
                continue
            seen.add(qid)
            base = {"article": r["document_title"], "passage": gold_passage, "question": r["question_text"]}
            if kind == "yesno":
                question, target = common.noul(rng.choice(TYDI_YESNO_ASK), a["yes_no_answer"][0] == "YES")
                counts[f"yesno_{a['yes_no_answer'][0]}"] += 1
                rows.append((None, common.example(TYDI, f"{group}:{qid}:yn", "yes-no-reading", base, question, target, "train")))
                continue
            answer = doc[a["minimal_answers_start_byte"][0]:a["minimal_answers_end_byte"][0]].decode("utf-8", "replace").strip()
            if not answer or squash(answer) not in squash(gold_passage):
                counts["answer_not_in_passage"] += 1
                continue
            keys = _tydi_keys(answer, r["question_text"])
            negatives = [i for i, x in enumerate(passages)
                         if abs(i - p) > 1 and 80 <= len(x) <= TYDI_MAX_PASSAGE and not _tydi_footer(x)
                         and not any(k in squash(x) for k in keys)]
            if not negatives:
                counts["no_negative"] += 1
                continue
            neg = rng.choice(negatives)
            pair = []
            for label, text in ((True, gold_passage), (False, passages[neg])):
                state = {**base, "passage": text}
                question, target = common.noul(rng.choice(TYDI_ANSWERS_ASK), label)
                pair.append(common.example(TYDI, f"{group}:{qid}:{'pos' if label else 'neg'}", "answerability", state, question, target, "train"))
            rows.append((lang, pair))
    taken = collections.Counter()
    for lang, item in rows:
        if lang is None:
            yield item
            continue
        if taken[lang] >= pairs_per_language:
            continue
        taken[lang] += 1
        counts[f"pairs_{lang}"] += 1
        yield from item
    stats["counts"] = dict(counts)


BUILDERS = [
    (FEWREL, fewrel, {"upstream": "github thunlp/FewRel@278a2315 data/train_wiki.json + pid2name.json", "licence": "MIT",
                      "label_origin": "human (crowd-confirmed distant-supervision sentences); distractors programmatic"}),
    (GAP, gap, {"upstream": "github google-research-datasets/gap-coreference@83135f26 gap-development.tsv + gap-validation.tsv",
                "licence": "Apache-2.0", "label_origin": "human (raters)"}),
    (SEMCOR, semcor, {"upstream": f"nltk_data@{NLTK_DATA[:8]} corpora/semcor.zip (SemCor 3.0) + corpora/wordnet.zip (WordNet 3.0)",
                      "licence": "Princeton WordNet licence (SemCor 1.6/WordNet; SemCor 3.0 mapping by R. Mihalcea, same terms)",
                      "label_origin": "human (Princeton sense tags), mapped WN1.6 -> WN3.0 by a program (UNT); options from WordNet",
                      "wic_filter": "sentences equal to / containing / contained in a fresh-v1 WiC sentence dropped (normalised)"}),
    (KLUE, klue, {"upstream": "klue/klue@349481ec ynat, nli, re train", "licence": "CC-BY-SA-4.0",
                  "label_origin": "human (crowd annotators); RE distractors programmatic"}),
    (CHID, chid, {"upstream": "thu-coai/chid@47ad8022 original/train_data.txt", "licence": "Apache-2.0",
                  "label_origin": "programmatic cloze of human-written text (original idiom); candidates by the dataset's rules"}),
    (CMRC, cmrc, {"upstream": "hfl/cmrc2018@137f2c45 train + github DRCKnowledgeTeam/DRCD@b944790d DRCD_training.json",
                  "licence": "CMRC2018 CC-BY-SA-4.0; DRCD CC-BY-SA-3.0",
                  "label_origin": "human answers (crowd); distractor spans programmatic"}),
    (JGLUE, jglue, {"upstream": "github yahoojapan/JGLUE@6f071c09 jcommonsenseqa-v1.3 + jnli-v1.3 train", "licence": "CC-BY-SA-4.0",
                    "label_origin": "human (crowd)"}),
    (TYDI, tydiqa, {"upstream": "google-research-datasets/tydiqa@da78f23f primary_task train", "licence": "Apache-2.0",
                    "label_origin": "human (native-speaker annotators); negative passages chosen by a program"}),
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
