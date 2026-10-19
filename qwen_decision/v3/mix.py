"""Build a gold-only training mix and its grouped gold dev set, written as files.

  python -m qwen_decision.v3.mix --name v3-gold --rows 200000

Policy (2026-10-17): nothing is learned from a teacher model. Teacher-distilled sources (yuri_v3) are
not read at all, and sources whose label origin is not verified (openjev_v2) are left out until it is.

Groups and shares (MIXES) set how much of the mix each kind of data gets; within a group, sources are
weighted by sqrt(size) and never repeated (a source smaller than its quota is used whole and the
remainder goes to the group's other sources). Rows are shuffled once, so the first n rows of the file
are a proportional subset.

Every row gets two separate fields:
- label_origin (who decided the gold): human (annotators, exams, courts, the dataset's authors),
  programmatic (computed: executed code, solved puzzles, key-value lookup, causal closure), model
  (a public dataset whose gold was produced by a generating model pipeline, e.g. ToolACE, Glaive);
- target_kind (shape of the gold): single (one correct option), valid_set (several options are all
  correct: Habermas's tied winners; any of them is right), vote_distribution (annotators disagreed:
  VAST; the plurality option(s) are right, the rest of the votes are kept for reporting only).

Gold dev set (<name>.dev.jsonl): about --dev-per-source rows of every source, held out before sampling
as whole linked groups. Rows are linked, across sources, when they are the same document: they share at
least half of the smaller one's normalized paragraphs (an ECtHR case in ecthr_a and ecthr_b, a contract or
article under different keys or titles, a MuSiQue question and its unanswerable contrast version);
paragraphs in more than BOILERPLATE rows (licence text, task templates) do not count. The builders keep
no upstream document ids, so this content linking is the grouping; see data_groups().
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from qwen_decision.v3.decontam import strings
from qwen_decision.v3.sources import common

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/mix"
EXCLUDED = {
    "sciq-train": "CC-BY-NC licence",
    "long-context": "synthetic long families replaced by real train splits and kv-lookup",
    "mtbench-human-train": "its questions are a public benchmark (MT-Bench); 25% of rows hit the index filter",
    "conditionalqa-train": "README: the data 'should ONLY be used for NLP research purpose'",
}
CORPUS_EXCLUDED = {
    "yuri_v3": "labels distilled from Jev 1.13 (corpus README): no learning from teacher models",
    "openjev_v2": "Open-Jev reference labels; some configs mention a judge model and teacher/control labels; "
                  "per-config origin not verified",
}
def _after(n: int):
    return lambda i: ":".join(i.split(":")[1:1 + n])


def _musique(i: str) -> str:
    return re.search(r"\d+hop[^:]*", i).group(0)


def _lexglue(i: str) -> str:
    _, config, _, index = i.split(":")[:4]  # ...:<index>[:pos|neg] for the multi-label yes/no questions
    return f"{'ecthr' if config in ('ecthr_a', 'ecthr_b') else config}:{index}"  # a/b label the same cases


def _phishing_origin(row: dict) -> str:
    return "model" if row["id"].split(":")[1] == "llmgen" else "human"  # LLMGen: GPT-written emails


# Every source must be registered: (label_origin, upstream item key from the row id). label_origin is who
# decided the gold: human (annotators, exams, courts, dataset authors), programmatic (computed), model
# (produced by a generating model pipeline). The item key names the upstream item/document, so its
# variants (MuSiQue answerable/unanswerable, When2Call pair/4-way, ECtHR a/b) never straddle a split.
SOURCES = {
    "acos-train": ("human", lambda i: ":".join(i.split(":")[1:4])),  # one review, several aspects
    # API-Bank train golds are API calls written by ChatGPT agents (Li et al. 2023: "generated automatically
    # by the LLM"; only the evaluation set is human-annotated): model, so out under the default origins.
    "apibank-train": ("model", _after(2)),
    "aqua-rat-train": ("human", _after(1)), "arc-train": ("human", _after(1)),
    "banking77-train": ("human", _after(2)), "boolq-train": ("human", _after(1)), "clinc-train": ("human", _after(2)),
    "code-output-gen": ("programmatic", _after(1)), "commonsenseqa-train": ("human", _after(1)),
    "contractnli-train": ("human", _after(2)), "corr2cause-train": ("programmatic", _after(1)),
    "cuad-train": ("human", _after(2)), "esci-train": ("human", _after(2)), "gsm8k-train": ("human", _after(1)),
    "habermas-train": ("human", _after(1)), "hellaswag-train": ("human", _after(2)), "hover-train": ("human", _after(1)),
    "humicroedit-train": ("human", _after(2)), "isarcasm-train": ("human", lambda i: ":".join(i.split(":")[2:4])),
    "kv-lookup": ("programmatic", _after(2)), "lexglue-train": ("human", _lexglue),
    "lexglue-short-train": ("human", _after(3)), "logic-puzzles-gen": ("programmatic", _after(2)),
    "massive-intents": ("human", _after(3)), "mathqa-train": ("human", _after(1)), "medmcqa-train": ("human", _after(1)),
    "mmlu-aux-train": ("human", _after(1)), "musique-train": ("human", _musique),
    "newyorker-matching-train": ("human", _after(1)), "nli4ct-train": ("human", _after(2)),
    "sentfin-train": ("human", _after(1)),  # first occurrence of the headline: all its entities, repeats included
    "tatqa-train": ("human", _after(2)),  # split:table uid: every question on one table+text context
    # phase 3 (2026-10-18, qwen_decision/v3/sources/phase3.py): item = context / post / premise / dialogue / story
    "siqa-train": ("human", _after(1)), "cosmosqa-train": ("human", _after(1)), "mnli-train": ("human", _after(1)),
    "faithdial-train": ("human", _after(1)), "moralstories-train": ("human", _after(1)), "ecare-train": ("human", _after(1)),
    "goemotions-train": ("human", _after(1)),
    # phase 4 (2026-10-18, qwen_decision/v3/sources/phase4.py)
    # icl-classify: the query text (id icl-classify:<corpus>:<text hash>), so one DBpedia text asked at level 2
    # and level 3 is one item (gold4 keyed it per level)
    "tatqa-yesno": ("human", _after(2)), "icl-classify": ("human", lambda i: i.split(":")[2]), "wned-el": ("human", _after(1)),
    "vitaminc-train": ("human", _after(1)), "paws-train": ("human", _after(2)), "scitail-train": ("human", _after(1)),
    "fintweet-train": ("human", _after(1)),
    # phase 5 (2026-10-19, qwen_decision/v3/sources/phase5.py): item = table / passage / story / chat / prompt /
    # sentence / context document (world:document) / conversation
    "tabfact-train": ("human", _after(1)), "condaqa-train": ("human", _after(1)), "ropes-train": ("human", _after(1)),
    "folio-train": ("human", _after(1)), "abcd-train": ("human", _after(1)), "helpsteer2-train": ("human", _after(1)),
    "finer-train": ("human", _after(1)), "zeshel-train": ("human", _after(2)), "fewnerd-train": ("human", _after(1)),
    "drop-train": ("human", _after(1)), "taskmaster-train": ("human", _after(1)),
    # phase 6 (2026-10-19, qwen_decision/v3/sources/phase6_{pref,reason,domain,lang}.py): the first id segment is the
    # prompt / topic / passage / table / contract / dialogue / document; HEAD-QA by exam question (es and en share it)
    "helpsteer3-train": ("human", _after(1)), "oasst2-train": ("human", _after(1)), "hhrlhf-train": ("human", _after(1)),
    "tldr-feedback-train": ("human", _after(1)), "arena55k-train": ("human", _after(1)), "mtbench-human-train": ("human", _after(1)),
    "argq-train": ("human", _after(1)), "ukpconvarg-train": ("human", _after(1)), "claimstance-train": ("human", _after(1)),
    "quoref-train": ("human", _after(1)), "quac-train": ("human", _after(1)), "sharc-train": ("human", _after(1)),
    "conditionalqa-train": ("human", _after(1)), "creak-train": ("human", _after(1)), "snli-train": ("human", _after(1)),
    "infotabs-train": ("human", _after(1)), "wtq-train": ("human", _after(1)), "svamp-train": ("human", _after(1)),
    "stepgame-train": ("programmatic", _after(1)), "spartqa-train": ("programmatic", _after(1)),
    "maud-train": ("human", _after(1)), "pubhealth-train": ("human", _after(1)),
    "headqa-train": ("human", lambda i: f"{i.split(':')[1]}:{re.sub(r'^[a-z]+', '', i.split(':')[2])}"),
    "civilcomments-train": ("human", _after(1)), "hatexplain-train": ("human", _after(1)), "multiwoz-train": ("human", _after(1)),
    "mtop-train": ("human", _after(1)), "snips-train": ("human", _after(1)), "redial-train": ("human", _after(1)),
    "fewrel-train": ("human", _after(1)), "gap-train": ("human", _after(1)), "semcor-train": ("human", _after(1)),
    "klue-train": ("human", _after(1)), "chid-train": ("human", _after(1)), "cmrc-train": ("human", _after(1)),
    "jglue-train": ("human", _after(1)), "tydiqa-train": ("human", _after(1)),
    "openbookqa-train": ("human", _after(1)), "phishing": (_phishing_origin, _after(4)),  # file:label:body hash (file:label alone made 3 items)
    "pubmedqa-labeled": ("human", _after(1)), "quality-train": ("human", lambda i: i.split(":")[1].split("_")[0]),
    "ragtruth-train": ("human", _after(1)), "squad2-answerability": ("human", _after(1)),
    "strategyqa-train": ("human", _after(1)), "tools-percall": ("model", _after(2)), "vast-train": ("human", _after(1)),
    "wanli-train": ("human", _after(1)), "when2call-train": ("model", _after(2)), "winogrande-train": ("human", _after(1)),
}


TARGET_KIND = {"valid_set": ["habermas-train"], "vote_distribution": ["vast-train"]}
AREAS = {
    "knowledge": ["mmlu-aux-train", "medmcqa-train", "arc-train", "openbookqa-train", "aqua-rat-train", "mathqa-train",
                  "gsm8k-train", "code-output-gen", "logic-puzzles-gen", "strategyqa-train", "commonsenseqa-train",
                  "hellaswag-train", "winogrande-train", "corr2cause-train", "pubmedqa-labeled"],
    "language": ["wanli-train", "nli4ct-train", "vast-train", "isarcasm-train", "acos-train", "boolq-train",
                 "squad2-answerability", "ragtruth-train", "habermas-train", "phishing"],
    "retrieval": ["clinc-train", "banking77-train", "massive-intents", "esci-train", "hover-train"],
    "tools": ["tools-percall", "when2call-train", "apibank-train"],
    "arts": ["humicroedit-train", "newyorker-matching-train"],
    "finance": ["sentfin-train", "tatqa-train"],
    "commonsense-new": ["siqa-train", "cosmosqa-train", "ecare-train", "moralstories-train"],
    "language-new": ["mnli-train", "faithdial-train", "goemotions-train"],
    "phase4": ["tatqa-yesno", "icl-classify", "wned-el", "vitaminc-train", "paws-train", "scitail-train", "fintweet-train"],
    "phase5": ["tabfact-train", "condaqa-train", "ropes-train", "folio-train", "abcd-train", "helpsteer2-train", "finer-train",
               "zeshel-train", "fewnerd-train", "drop-train", "taskmaster-train"],
    "preference": ["helpsteer3-train", "oasst2-train", "hhrlhf-train", "tldr-feedback-train", "arena55k-train", "argq-train",
                   "ukpconvarg-train", "claimstance-train"],
    "reasoning-6": ["quoref-train", "quac-train", "sharc-train", "creak-train", "snli-train", "infotabs-train", "wtq-train",
                    "svamp-train", "stepgame-train", "spartqa-train"],
    "domain-6": ["maud-train", "pubhealth-train", "headqa-train", "civilcomments-train", "hatexplain-train", "multiwoz-train",
                 "mtop-train", "snips-train", "redial-train"],
    "language-6": ["fewrel-train", "gap-train", "semcor-train", "klue-train", "chid-train", "cmrc-train", "jglue-train", "tydiqa-train"],
    "legal": ["contractnli-train", "lexglue-train", "lexglue-short-train", "cuad-train"],
    "long-reading": ["quality-train", "musique-train", "kv-lookup"],
}
MIXES = {
    # group -> (share of rows, sources); the remaining sources form "public-train".
    "v3-gold": {"long": (.12, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                "public-train": (.88, None)},
    # v3-gold plus the finance sources (sentfin-train, tatqa-train, picked up as public-train) and BOOST.
    "v3-gold2": {"long": (.12, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                 "public-train": (.88, None)},
    # g4 (2026-10-18): about 400k distinct rows seen once instead of 200k rows three times, the phase-3
    # sources, and full HellaSwag/WinoGrande pools. Long rows keep about their v3-gold2 count (6% of 400k),
    # since they carry most of the tokens.
    "v3-gold3": {"long": (.06, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                 "public-train": (.94, None)},
    # v3-gold3 plus the phase-4 sources (finance yes/no, in-context classification, entity linking, fact
    # verification, paraphrase, science entailment, finance tweets).
    "v3-gold4": {"long": (.06, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                 "public-train": (.94, None)},
    # v3-gold4 after its data audit (2026-10-18): the same plan on the rebuilt phase-4 sources (TAT-QA direction
    # questions checked on the table, ICL texts with one role each, fintweet-train without the ICL tweets,
    # VitaminC whole cases) with document grouping for WNED, PAWS, SciTail and the tweets, and VitaminC drawn
    # as whole cases (WHOLE_ITEMS).
    "v3-gold5": {"long": (.06, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                 "public-train": (.94, None)},
    # v3-gold5 plus the phase-5 and phase-6 sources (2026-10-19): many more kinds of decisions (PHASE5, PHASE6).
    "v3-gold6": {"long": (.05, ["quality-train", "musique-train", "kv-lookup", "cuad-train", "lexglue-train", "contractnli-train"]),
                 "public-train": (.95, None)},
}
# Sources weighted up within their group (x the sqrt-size weight): g1 lost most to d3-lite on HellaSwag
# (0.65 vs 0.90), BANKING77 (0.69 vs 0.89) and JEVal WinoGrande (0.17 vs 0.43), and the mix used only
# 40-60% of their prepared rows.
BOOST = {"v3-gold2": {"hellaswag-train": 2., "winogrande-train": 2., "banking77-train": 2.},
         "v3-gold3": {"hellaswag-train": 2., "winogrande-train": 2., "banking77-train": 2., "goemotions-train": 2.},
         "v3-gold4": {"hellaswag-train": 2., "winogrande-train": 2., "banking77-train": 2., "goemotions-train": 2.,
                      "tatqa-yesno": 2., "icl-classify": 2., "wned-el": 2.}}
BOOST["v3-gold5"] = BOOST["v3-gold6"] = BOOST["v3-gold4"]
# Sources new to a mix are left out of the others, so rebuilding an older mix keeps its source list.
PHASE5 = ["tabfact-train", "condaqa-train", "ropes-train", "folio-train", "abcd-train", "helpsteer2-train", "finer-train",
          "zeshel-train", "fewnerd-train", "drop-train", "taskmaster-train"]
PHASE6 = ["helpsteer3-train", "oasst2-train", "hhrlhf-train", "tldr-feedback-train", "arena55k-train", "argq-train",
          "ukpconvarg-train", "claimstance-train", "quoref-train", "quac-train", "sharc-train", "creak-train", "snli-train",
          "infotabs-train", "wtq-train", "svamp-train", "stepgame-train", "spartqa-train", "maud-train", "pubhealth-train",
          "headqa-train", "civilcomments-train", "hatexplain-train", "multiwoz-train", "mtop-train", "snips-train", "redial-train",
          "fewrel-train", "gap-train", "semcor-train", "klue-train", "chid-train", "cmrc-train", "jglue-train", "tydiqa-train"]
ONLY_IN = {"sentfin-train": {"v3-gold2", "v3-gold3", "v3-gold4", "v3-gold5", "v3-gold6"},
           "tatqa-train": {"v3-gold2", "v3-gold3", "v3-gold4", "v3-gold5", "v3-gold6"},
           **{s: {"v3-gold3", "v3-gold4", "v3-gold5", "v3-gold6"} for s in ("siqa-train", "cosmosqa-train", "mnli-train", "faithdial-train",
                                                                           "moralstories-train", "ecare-train", "goemotions-train")},
           **{s: {"v3-gold4", "v3-gold5", "v3-gold6"} for s in ("tatqa-yesno", "icl-classify", "wned-el", "vitaminc-train", "paws-train",
                                                                "scitail-train", "fintweet-train")},
           **{s: {"v3-gold6"} for s in PHASE5 + PHASE6}}
# Sources sampled as whole items (all rows of an item key or none): a VitaminC case is a claim and its contrast
# claim against the evidence before and after an edit, and the pair is what teaches the verdict to follow the
# evidence. gold4 sampled rows one by one and kept 4% of rows with their partner.
WHOLE_ITEMS = {"v3-gold5": {"vitaminc-train"}, "v3-gold6": {"vitaminc-train", "condaqa-train"}}
BOILERPLATE = 50   # a short paragraph (< LONG_PIECE chars) in more rows than this is a template or tool schema line
LONG_PIECE = 200   # normalized characters from which a paragraph is passage text (a SQuAD/BoolQ passage, a clause)
LONG_BOILERPLATE = 1000  # a long paragraph in more rows than this is licence text (Gutenberg) or a fixed preamble
MIN_PIECE = 60     # normalized characters for a paragraph to count as document text
SAME_DOCUMENT = .5  # share of the smaller row's paragraphs two rows must share to be one document


# Model-made golds are allowed only for tool use (user decision 2026-10-17): no human-labelled tool-selection
# train data was found, and the tools area cannot be left empty. Everything else stays human/programmatic.
MODEL_ORIGIN_ALLOWED = {"tools-percall", "when2call-train", "apibank-train"}


def allowed_origin(row: dict, origins: set[str]) -> bool:
    return row["label_origin"] in origins or (row["label_origin"] == "model" and row["source"] in MODEL_ORIGIN_ALLOWED)


def registered(source: str) -> tuple:
    if source not in SOURCES:
        raise KeyError(f"source {source!r} is not registered in mix.SOURCES (label origin and item key unknown)")
    return SOURCES[source]


def label_origin(row: dict) -> str:
    origin = registered(row["source"])[0]
    return origin(row) if callable(origin) else origin


def item_key(row: dict) -> str:
    return f"{row['source'].replace('-short', '')}:{registered(row['source'])[1](row['id'])}"


def target_kind(row: dict) -> str:
    if max(row["target"]) >= .99:
        return "single"
    kind = next((k for k, names in TARGET_KIND.items() if row["source"] in names), None)
    if kind is None:
        raise ValueError(f"{row['id']}: a spread target from a source with no declared target_kind")
    return kind


def _norm(text: str, digits: bool = True) -> str:
    text = re.sub(r"[\W_]+", " ", text.lower()).strip()
    return text if digits else re.sub(r"\d+", "0", text)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


_CONTESTS: dict[str, int] = {}


def _newyorker(row: dict) -> list[str]:
    """The cartoon's contest: instances of one contest have their own ids and differently worded descriptions."""
    if not _CONTESTS:
        import pyarrow.parquet as pq
        from qwen_decision.v3.sources import arts
        path, _ = common.fetch(arts.NEWYORKER_REPO, ["matching/train-*", "matching/validation-*"],
                               revision=arts.NEWYORKER_REVISION)
        for f in path.glob("matching/*.parquet"):
            for r in pq.read_table(f, columns=["instance_id", "contest_number"]).to_pylist():
                _CONTESTS[r["instance_id"]] = r["contest_number"]
    return [f"contest:{_CONTESTS[row['id'].split(':', 1)[1]]}"]


def _premise(row: dict) -> list[str]:
    """Corr2Cause: one premise (variables and their correlations) is asked with many hypotheses."""
    parts = [part for text in strings(row["state"]) for part in text.split("\n\n") if "closed system" in part]
    return [_digest(_norm(part[part.index("closed system"):])) for part in parts[:1]]


def _values(state, keys: tuple[str, ...] = ()) -> str:
    """The state's text without field names (a row's state is a string or a dict of one or two fields)."""
    if isinstance(state, dict):
        chosen = [state[k] for k in keys if k in state] or list(state.values())
        return " ".join(v for v in chosen if isinstance(v, str))
    return state if isinstance(state, str) else ""


def _trials(row: dict) -> list[str]:
    """NLI4CT: statements about one clinical trial (primary or secondary) are one document."""
    return sorted(set(re.findall(r"NCT\d{8}", json.dumps(row["state"]))))


# Upstream documents that several items (different ids, different text) are built from. Rows sharing a
# document key are one group, like rows sharing an item key; found 2026-10-17 when 146 of 150 New Yorker
# dev rows turned out to share their cartoon (contest) with train rows.
def _tatqa(row: dict) -> list[str]:
    """TAT-QA: one table+text context, shared by tatqa-train and tatqa-yesno (a source-independent key)."""
    return [f"shared:tatqa:{row['id'].split(':')[2]}"]


def _tweet(row: dict) -> list[str]:
    """A finance tweet asked in-context (icl-classify fintopic) or directly (fintweet-train): one shared key."""
    corpus, key = row["id"].split(":")[1:3] if row["source"] == "icl-classify" else ("fintopic", row["id"].split(":")[1])
    return [f"shared:tweet:{key}"] if corpus == "fintopic" else []


def _sentences(*fields: str):
    """Rows sharing any of these state fields (a PAWS sentence in several scrambled pairs, a SciTail premise
    retrieved for several hypotheses) are one document."""
    return lambda row: [_digest(_norm(row["state"][f])) for f in fields]


# Found 2026-10-18 (v3-gold4 audit): 28 of 308 WNED dev rows came from articles with rows in train, and 131 of
# 300 PAWS dev rows had a sentence that train rows also had.
DOCUMENTS = {"newyorker-matching-train": _newyorker, "corr2cause-train": _premise, "nli4ct-train": _trials,
             "tatqa-train": _tatqa, "tatqa-yesno": _tatqa,
             "wned-el": lambda row: [row["id"].split(":", 1)[1].split("_", 1)[1]],  # <uuid>_<article title>
             "paws-train": _sentences("sentence 1", "sentence 2"), "scitail-train": _sentences("premise"),
             "icl-classify": _tweet, "fintweet-train": _tweet}

# Sources whose items recur reworded: When2Call paraphrases one request many times ("Retrieve 6 months of
# historical daily stock prices" / "... for Apple Inc."), MathQA and AQuA-RAT repeat problems with other
# numbers, WinoGrande pairs twin sentences that differ in a word or two. Rows of one source are joined when
# NEAR_SAME of the shorter text's word trigrams (digits masked) are in the other.
NEAR_TEXT = {"when2call-train": lambda r: _values(r["state"], ("request", "question", "user_message")),
             "mathqa-train": lambda r: _values(r["state"]), "aqua-rat-train": lambda r: _values(r["state"]),
             # an empty-state WinoGrande row shows the two filled-in sentences as its options
             "winogrande-train": lambda r: _values(r["state"]) or " ".join(sorted(r["question"]["criteria"])[:1])}
NEAR_SAME = .6


def near_duplicate_pairs(rows: list[dict]) -> list[tuple[int, int]]:
    """Index pairs of NEAR_TEXT rows (same source) whose texts are near duplicates."""
    grams = {}
    for i, r in enumerate(rows):
        if r["source"] in NEAR_TEXT:
            words = _norm(NEAR_TEXT[r["source"]](r), digits=False).split()
            g = {" ".join(words[k:k + 3]) for k in range(len(words) - 2)}
            if len(g) >= 3:
                grams[i] = g
    pairs = []
    by_source = defaultdict(list)
    for i in grams:
        by_source[rows[i]["source"]].append(i)
    for members in by_source.values():
        index = defaultdict(list)
        for i in members:
            for g in grams[i]:
                index[g].append(i)
        for i in members:  # candidates share a rarer trigram; the overlap is then counted on all trigrams
            candidates = {j for g in grams[i] if len(index[g]) <= 200 for j in index[g] if j > i}
            for j in candidates:
                if len(grams[i] & grams[j]) >= NEAR_SAME * min(len(grams[i]), len(grams[j])):
                    pairs.append((i, j))
    return pairs


def document_keys(row: dict) -> list[str]:
    function = DOCUMENTS.get(row["source"])
    # keys starting with "shared:" join rows across sources (one upstream document behind several sources)
    return [k if k.startswith("shared:") else f"{row['source']}:doc:{k}" for k in function(row)] if function else []


def annotate(row: dict) -> dict:
    return {**row, "label_origin": label_origin(row), "target_kind": target_kind(row), "item": item_key(row),
            "documents": document_keys(row)}


# Option texts that point at other options by letter or position ("both B and C", "all of the above") or carry
# their own letter ("(a): ..."): shuffling the display order or the codes would change what they mean.
POSITIONAL = re.compile(
    r"(?i:\b(?:both|either|neither)\s+)\(?[A-E1-5]\)?(?i:\s+(?:and|or|nor)\s+)\(?[A-E1-5]\)?(?!\w)"
    r"|(?i:\b(?:options?|choices?|answers?)\s+)\(?[A-E1-5]\)?(?!\w)"
    r"|(?i:\b(?:all|none|both|any|neither) of the (?:above|below|previous|following|options|choices)\b)"
    r"|^\s*\([a-eA-E]\)\s*:?")


def positional_options(question: dict) -> bool:
    """True when the order of a choice question's options must be kept (see POSITIONAL)."""
    if question["type"] != "choice":
        return False
    return any(POSITIONAL.search(k if v is None else f"{k} {v}") for k, v in question["criteria"].items())


def gold_set(row: dict) -> list[int]:
    """Options that count as correct: the single gold, every valid_set member, or the plurality of the votes."""
    t = row["target"]
    return [i for i, p in enumerate(t) if p >= max(t) - 1e-6] if row["target_kind"] != "valid_set" else \
           [i for i, p in enumerate(t) if p > 0]


def pieces(row: dict) -> list[str]:
    """Normalized paragraph hashes of the state (the question with its options when the state has no text)."""
    texts = strings(row["state"])
    if sum(len(t) for t in texts) < 40:  # e.g. WinoGrande rows whose sentence versions are the options
        texts = strings(row["question"])
    out = []
    for text in texts:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # hard-wrapped lines (QuALITY, contracts) are one paragraph
        for block in re.split(r"\n\s*\n+", text):
            # The whole paragraph as one long piece (a passage many questions share) and its sentences
            # as pieces of their own (a passage quoted in part, under another layout).
            whole = re.sub(r"[\W_]+", " ", block.lower()).strip()
            if len(whole) >= LONG_PIECE:
                out.append("L:" + hashlib.sha1(whole.encode()).hexdigest())
            for para in re.split(r"(?<=[.!?])\s+(?=[A-Z])", block):
                norm = re.sub(r"[\W_]+", " ", para.lower()).strip()
                if MIN_PIECE <= len(norm) and norm != whole:
                    out.append(("L:" if len(norm) >= LONG_PIECE else "S:") + hashlib.sha1(norm.encode()).hexdigest())
    if not out:  # a short item: its whole text is its identity
        norm = re.sub(r"[\W_]+", " ", " ".join(texts).lower()).strip()
        out.append(hashlib.sha1(norm.encode()).hexdigest())
    return out


# Sources grouped by their item keys (and near duplicates) only, not by shared paragraphs: every API-Bank row
# carries API catalog text that chained 66% of its rows into one >1,000-row group no dev set could sample.
# A dialogue is one item (mix.SOURCES), so its steps still stay together.
# ABCD rows carry their subflow's written policy, which would chain every chat of a subflow into one group.
CONTENT_LINK_EXEMPT = {"apibank-train", "abcd-train"}


def data_groups(rows: list[dict]) -> list[int]:
    """rows[i] -> group id: rows are linked when they are the same document.

    Same document = at least half of the smaller row's paragraphs are shared (SAME_DOCUMENT). Linking on
    any one shared paragraph chained MuSiQue questions through their reused Wikipedia distractors (and
    NDAs through standard clauses) into one 23k-row group, so no dev set could be held out for them.
    Repeated paragraphs are dropped first: short ones in more than BOILERPLATE rows (task templates, tool
    schema lines; left in, they merged every WinoGrande or Corr2Cause row into one group) and long ones
    in more than LONG_BOILERPLATE rows (licence text). A passage asked about by many questions stays: a
    single row-count cap split 51 questions on one passage into 51 groups. A row left with no paragraph
    is identified by its whole text. Finally rows with the same upstream item key (annotate) are joined.
    """
    row_pieces = [set() if r.get("source") in CONTENT_LINK_EXEMPT else set(pieces(r)) for r in rows]
    frequency = Counter(h for ps in row_pieces for h in ps)
    signature, node_of = {}, []
    for r, ps in zip(rows, row_pieces):
        own = [h for h in ps if frequency[h] <= (LONG_BOILERPLATE if h.startswith("L:") else BOILERPLATE)]
        if not own:  # only template/licence text: the row's whole text is its identity
            text = " ".join(strings(r["state"]) + strings(r["question"]))
            own = ["whole:" + hashlib.sha1(re.sub(r"[\W_]+", " ", text.lower()).encode()).hexdigest()]
        node_of.append(signature.setdefault(tuple(sorted(own)), len(signature)))
    nodes = list(signature)
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_piece = defaultdict(list)
    for n, key in enumerate(nodes):
        for h in key:
            by_piece[h].append(n)
    shared = Counter()
    for members in by_piece.values():
        if 1 < len(members):
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    shared[members[a], members[b]] += 1
    for (a, b), n in shared.items():
        if n >= SAME_DOCUMENT * min(len(nodes[a]), len(nodes[b])):
            parent[find(b)] = find(a)
    for i, j in near_duplicate_pairs(rows):
        parent[find(node_of[j])] = find(node_of[i])
    first = {}
    for i, r in enumerate(rows):  # variants of one upstream item or document are one group whatever their text
        for key in ([r["item"]] if r.get("item") else []) + list(r.get("documents") or []):
            if key in first:
                parent[find(node_of[i])] = find(first[key])
            else:
                first[key] = node_of[i]
    return [find(node_of[i]) for i in range(len(rows))]


def hold_out(rows: list[dict], per_source: int, rng: random.Random, max_share: float = .1) -> tuple[set[int], dict]:
    """Indices of a grouped dev set: per source, min(per_source, max_share of its rows) rows (2026-10-18 rules).

    - Group-aware, as a group shuffle split: a group (mix.data_groups) is entirely dev or entirely train.
    - Row-proportional: a random row is drawn and its whole group taken, so a source's dev reflects how its
      rows are spread over small and large documents (drawing groups uniformly over-samples small ones).
    - A group larger than half the source's target is never taken (one document must not be most of a dev
      set). The share of each source's rows in such groups is reported as uncovered.
    - Size: the target is set by power, not by a fraction of the mix. With paired comparisons of checkpoints
      on the same rows, 300 rows detect about a 5-point change in a source's accuracy (80% power, about 12%
      discordant rows), and the macro over ~50 sources about 0.7 points. max_share keeps small sources from
      losing more than 10% of their rows.
    """
    group = data_groups(rows)
    members = defaultdict(list)
    for i, g in enumerate(group):
        members[g].append(i)
    by_source = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r["source"]].append(i)
    dev, coverage, targets = set(), {}, {}
    for source in sorted(by_source):
        indices = by_source[source]
        target = max(1, min(per_source, round(max_share * len(indices))))
        cap = max(1, target // 2)
        targets[source] = target
        coverage[source] = round(sum(len(members[group[i]]) <= cap for i in indices) / len(indices), 3)
        order = list(indices)
        rng.shuffle(order)
        taken = 0
        for i in order:
            if taken >= target:
                break
            g = group[i]
            if i in dev or len(members[g]) > cap:
                continue
            for j in members[g]:
                dev.add(j)
                taken += rows[j]["source"] == source
    sizes = Counter(len(m) for m in members.values())
    stats = {"groups": len(members), "largest_group": max(len(m) for m in members.values()),
             "rows_in_groups_over_cap": sum(n * c for n, c in sizes.items() if n > per_source // 2),
             "cross_source_groups": sum(len({rows[i]["source"] for i in m}) > 1 for m in members.values()),
             "per_source_target": targets, "eligible_row_share": coverage,
             "low_coverage_sources": sorted(s for s, c in coverage.items() if c < .8),
             "rule": "row-proportional group sampling; target min(per_source, max_share*rows); groups over target/2 excluded"}
    return dev, stats


FRESH_SENTENCE = 60


def sentences(row: dict) -> set[str]:
    """Normalized sentences (and lines) of FRESH_SENTENCE+ characters in a row's state and option texts."""
    out = set()
    for text in strings(row["state"]) + strings(row["question"].get("criteria")):
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            norm = _norm(part)
            if len(norm) >= FRESH_SENTENCE:
                out.add(norm)
    return out


def whole_items(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """About n rows made of whole items (WHOLE_ITEMS): items in random order until n rows are reached."""
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item"]].append(r)
    items = sorted(by_item)
    rng.shuffle(items)
    out = []
    for item in items:
        if len(out) >= n:
            break
        out += by_item[item]
    return out


def area_of(source: str) -> str:
    return next((a for a, names in AREAS.items() if source in names), "other")


def quotas(sizes: dict[str, int], want: int, boost: dict[str, float] | None = None) -> dict[str, int]:
    """sqrt(size) weights (times boost[source]), capped at each source's size, leftovers redistributed."""
    quota, left, open_ = {}, want, dict(sizes)
    weight = lambda s, n: (boost or {}).get(s, 1.) * math.sqrt(n)
    while left > 0 and open_:
        total = sum(weight(s, n) for s, n in open_.items())
        share = {s: left * weight(s, n) / total for s, n in open_.items()}
        # sorted, not a set: set order follows PYTHONHASHSEED and made the row sampling differ between runs
        full = sorted(s for s in open_ if share[s] >= open_[s] - quota.get(s, 0))
        if not full:
            for s in open_:
                quota[s] = quota.get(s, 0) + int(share[s])
            break
        for s in full:
            left -= open_[s] - quota.get(s, 0)
            quota[s] = open_[s]
            del open_[s]
    return quota


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="v3-gold", choices=sorted(MIXES))
    p.add_argument("--rows", type=int, default=200000)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--dev-per-source", type=int, default=150,
                   help="dev rows per source (v3-gold3 uses 300: see hold_out for the power reasoning)")
    p.add_argument("--dev-max-share", type=float, default=.1, help="at most this share of a source's rows goes to dev")
    p.add_argument("--label-origins", default="human,programmatic",
                   help="keep rows whose label_origin is listed; model-made golds only for MODEL_ORIGIN_ALLOWED")
    args = p.parse_args()
    names = sorted(f.name[:-6] for f in (ROOT / "data/phase2").glob("*.jsonl")
                   if f.name[:-6] not in EXCLUDED and args.name in ONLY_IN.get(f.name[:-6], {args.name}))
    allowed = set(args.label_origins.split(","))
    every = [annotate(r) for r in common.load(names)]
    # Rows whose state is a fresh-v1 evaluation state (data/fresh, fresh_eval.py) never enter a mix: that set
    # must stay unseen. Exact normalized-state match, as fresh_eval.build checks the other way, and since
    # 2026-10-19 any shared sentence of FRESH_SENTENCE+ normalized characters (8 HellaSwag train rows had an
    # ActivityNet caption that fresh-v1 SWAG rows also have).
    fresh_file, fresh_dropped, fresh_sentence_dropped = ROOT / "data/fresh/fresh-v1.jsonl", Counter(), Counter()
    if fresh_file.exists():
        state_key = lambda st: hashlib.sha1(re.sub(r"[\W_]+", " ", json.dumps(st, ensure_ascii=False).lower()).strip().encode()).hexdigest()
        fresh_rows = [json.loads(line) for line in fresh_file.open()]
        fresh = {state_key(r["state"]) for r in fresh_rows}
        fresh_dropped = Counter(r["source"] for r in every if state_key(r["state"]) in fresh)
        every = [r for r in every if state_key(r["state"]) not in fresh]
        seen = set().union(*map(sentences, fresh_rows))
        hit = [bool(sentences(r) & seen) for r in every]
        fresh_sentence_dropped = Counter(r["source"] for r, h in zip(every, hit) if h)
        every = [r for r, h in zip(every, hit) if not h]
    origin_dropped = Counter(f"{r['source']}:{r['label_origin']}" for r in every if not allowed_origin(r, allowed))
    rows = [r for r in every if allowed_origin(r, allowed)]
    names = sorted({r["source"] for r in rows})
    dev_index, group_stats = hold_out(rows, args.dev_per_source, random.Random(args.seed + 1), args.dev_max_share)
    dev = [rows[i] for i in sorted(dev_index)]
    keys = lambda r: {r["item"], *r.get("documents", [])}
    leaked = set().union(*map(keys, dev)) & set().union(*(keys(r) for i, r in enumerate(rows) if i not in dev_index))
    if leaked:
        raise SystemExit(f"{len(leaked)} upstream items or documents on both sides of the dev split, e.g. {sorted(leaked)[:3]}")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if i not in dev_index:
            by[r["source"]].append(r)
    spec = MIXES[args.name]
    claimed = {s for _, (_, sources) in spec.items() if sources for s in sources}
    rng = random.Random(args.seed)
    picked, plan = [], {}
    for group, (share, sources) in spec.items():
        sources = sources or [s for s in names if s not in claimed]
        missing = [s for s in sources if s not in by]
        q = quotas({s: len(by[s]) for s in sources if s in by}, round(args.rows * share), BOOST.get(args.name))
        plan[group] = {"share": share, "quota": q, "missing_sources": missing}
        for s, n in q.items():
            picked += whole_items(by[s], n, rng) if s in WHOLE_ITEMS.get(args.name, ()) else rng.sample(by[s], n)
    rng.shuffle(picked)
    OUT.mkdir(parents=True, exist_ok=True)
    for path, items in ((OUT / f"{args.name}.jsonl", picked), (OUT / f"{args.name}.dev.jsonl", dev)):
        with path.open("w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in picked)
    manifest = {"name": args.name, "rows": len(picked), "seed": args.seed, "plan": plan,
                "excluded": EXCLUDED, "corpus_excluded": CORPUS_EXCLUDED,
                "label_origins_kept": sorted(allowed), "model_origin_allowed": sorted(MODEL_ORIGIN_ALLOWED),
                "label_origin_dropped": dict(origin_dropped.most_common()), "boost": BOOST.get(args.name, {}), "whole_items": sorted(WHOLE_ITEMS.get(args.name, ())), "fresh_v1_state_dropped": dict(fresh_dropped),
                "fresh_v1_sentence_dropped": dict(fresh_sentence_dropped),
                "dev_item_leak_check": "0 upstream items or documents on both sides (item_key, document_keys)",
                "label_origin": dict(Counter(r["label_origin"] for r in picked)),
                "target_kind": dict(Counter(r["target_kind"] for r in picked)),
                "dev": {"rows": len(dev), "per_source": dict(sorted(Counter(r["source"] for r in dev).items())),
                        "grouping": group_stats, "boilerplate_rows": BOILERPLATE, "min_piece_chars": MIN_PIECE},
                "available": {s: len(v) for s, v in sorted(by.items())}, "counts": dict(counts.most_common()),
                "areas": {s: area_of(s) for s in counts}, "note": "prefixes of the file are proportional subsets"}
    (OUT / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"rows": len(picked), "dev_rows": len(dev), "label_origin": manifest["label_origin"],
                      "target_kind": manifest["target_kind"], "grouping": group_stats,
                      "missing": {g: v["missing_sources"] for g, v in plan.items() if v["missing_sources"]}}))


if __name__ == "__main__":
    main()
