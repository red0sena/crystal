"""LexGLUE train splits as legal classification and holding questions (two outputs).

Upstream: coastalcph/lex_glue on the Hub, the parquet files on `main` at REVISION (no loading
script, no remote code). Only `<config>/train-*.parquet` is read. Licence: the card's YAML says
cc-by-4.0 for the whole benchmark (its "Licensing Information" section is empty). The subsets are
re-releases: ECtHR (Chalkidis et al. 2019/2021) and SCOTUS (SCDB labels; opinions are US public
domain), EUR-LEX (EU legal texts, (c) European Union, reuse allowed), LEDGAR (SEC EDGAR filings),
UNFAIR-ToS and CaseHOLD (Harvard CAP; CaseHOLD itself is released under Apache-2.0 by its authors).
LexGLUE redistributes them all under CC-BY-4.0; the manifest records the per-subset notes.

Framings (several own phrasings each; the index's instruction lines are never copied):
- ecthr_a / ecthr_b (facts paragraphs -> ECHR articles violated / alleged): a case with 0 or 1
  label is a choice over the 10 articles plus a "none" option; a multi-label case gives two noul
  questions, one gold article and one article not among its labels (for ecthr_a the negative is,
  when possible, an article that was alleged (ecthr_b) but not found violated).
- scotus: the 13 SCDB issue areas LexGLUE uses (label i = SCDB issue area i+1; area 14 "Private
  Action" is not in LexGLUE), single-label choice.
- ledgar: 100 provision types, single-label choice, CAP_LEDGAR rows.
- unfair_tos: 8 unfair-clause types + a "none" option for sentences with 0 or 1 labels; rows
  with several labels are skipped.
- case_hold: the 5 candidate holdings for the <HOLDING> slot, CAP_CASEHOLD rows.
- eurlex is skipped: the labels are bare EuroVoc ids (100163, ...) and neither the card, the
  parquet repo nor the old loading script (branch `script`) names them; coastalcph/multi_eurlex
  only lists ids too (names live inside its 2.8 GB tarball, under CC-BY-SA-4.0).

Overlap: the proxy benchmark (data/jeval/data/test.parquet) holds LexGLUE TEST items (sources
"lex-glue-*"). Every training document is compared with every JEVal lex-glue state: identical
word sequences, and 13-gram containment above MAX_TEST_CONTAINMENT in either direction (share of
the training text's 13-grams found in one JEVal state, or of the JEVal state's found in the
training text) are dropped here, before the filter. Because this document-level check is done
here, the long-document output (ecthr_a, ecthr_b, scotus -> NAME) is written in documents mode;
the short-text output (ledgar, unfair_tos, case_hold -> SHORT_NAME) keeps the strict filter.

Lengths: a document is kept only if every question rendered on it (echo layout, the longer of the
two, Qwen3.5 tokenizer) is <= MAX_PROMPT_TOKENS; nothing is truncated. The manifest records drops,
the share above 32k tokens and p50/p90/max per config.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
import pandas as pd
import yaml

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.decontam import words
from qwen_decision.v3.sources import common

NAME = "lexglue-train"
SHORT_NAME = "lexglue-short-train"
REPO = "coastalcph/lex_glue"
REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
JEVAL = common.ROOT / "data/jeval/data/test.parquet"
MAX_PROMPT_TOKENS = 64_000
REPORT_TOKENS = 32_000
MAX_TEST_CONTAINMENT = .2
NGRAM = 13
CAP_LEDGAR = 10_000
CAP_CASEHOLD = 10_000
LICENCE = {
    "benchmark": "CC-BY-4.0 (card YAML `license: cc-by-4.0`; the card's Licensing Information section is empty)",
    "ecthr_a/ecthr_b": "ECtHR judgments (HUDOC, reuse permitted); dataset by Chalkidis et al., redistributed under CC-BY-4.0",
    "scotus": "US Supreme Court opinions are public domain; SCDB issue-area labels (CC-BY-NC-3.0 upstream at scdb.wustl.edu)",
    "ledgar": "SEC EDGAR filings (public); LEDGAR labels by Tuggener et al., redistributed under CC-BY-4.0",
    "unfair_tos": "Lippi et al. 2019 annotations of 50 public ToS, redistributed under CC-BY-4.0",
    "case_hold": "CaseHOLD (Zheng et al. 2021, Harvard CAP text) released by its authors under Apache-2.0",
}

# ---------------------------------------------------------------- ECtHR
ARTICLES = ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]  # LexGLUE label order
ARTICLE_RIGHTS = {
    "2": "Right to life", "3": "Prohibition of torture and inhuman or degrading treatment",
    "5": "Right to liberty and security", "6": "Right to a fair trial",
    "8": "Right to respect for private and family life", "9": "Freedom of thought, conscience and religion",
    "10": "Freedom of expression", "11": "Freedom of assembly and association",
    "14": "Prohibition of discrimination", "P1-1": "Protection of property",
}
ARTICLE_KEYS = [
    lambda a: "Article 1 of Protocol 1" if a == "P1-1" else f"Article {a}",
    lambda a: "Art. 1 Prot. 1" if a == "P1-1" else f"Art. {a}",
    lambda a: "Protocol No. 1, Article 1" if a == "P1-1" else f"Article {a} ECHR",
]
NONE_OPTION = {
    "ecthr_a": [("No violation", "the Court found none of these articles violated"),
                ("none", "no listed article was breached"), ("no violation found", None)],
    "ecthr_b": [("None of these", "the Court examined none of the listed articles"),
                ("no listed article", "none of these articles was alleged to be violated"), ("none", None)],
}
ECTHR_CHOICE = {
    "ecthr_a": [
        "These are the facts of a case before the European Court of Human Rights. Which Convention article, if any, did the Court hold was breached?",
        "Based on the facts below, which article of the European Convention on Human Rights was the respondent State found to have violated? Pick the none option if no violation was found.",
        "Read the case facts. Choose the Convention provision the Strasbourg Court found violated, or the option for no violation.",
        "From these ECtHR case facts, predict the outcome: which listed article did the judgment declare violated?",
        "Which right under the Convention did the Court rule was infringed in this application, if any?",
    ],
    "ecthr_b": [
        "These are the facts of a case before the European Court of Human Rights. Which Convention article did the applicant claim was violated and the Court examine?",
        "Based on the facts below, which article of the European Convention on Human Rights is the alleged violation about? Pick the none option if no listed article was at issue.",
        "Read the case facts. Which Convention provision did the Court consider as allegedly breached?",
        "From these ECtHR case facts, identify the article whose violation was alleged.",
        "Which right under the Convention was the complaint in this application about, if any of the listed ones?",
    ],
}
ECTHR_NOUL = {
    "ecthr_a": [
        "Did the European Court of Human Rights find a violation of {key} ({right}) in this case?",
        "Given the case facts, was {key} of the Convention ({right}) held to be violated?",
        "Is {key} ({right}) among the articles the Court found breached here?",
        "Based on these facts, did the judgment conclude that {right} ({key}) was infringed?",
    ],
    "ecthr_b": [
        "Was a violation of {key} ({right}) alleged and examined by the Court in this case?",
        "Given the case facts, is {key} of the Convention ({right}) one of the articles the complaint relied on?",
        "Did the Court consider an alleged breach of {right} ({key}) in this application?",
        "Is {key} ({right}) among the provisions the applicant claimed were violated?",
    ],
}
ECTHR_NOUL_CRITERIA = {
    "ecthr_a": {"true": "the Court found this article violated", "false": "the Court did not find this article violated"},
    "ecthr_b": {"true": "this article was alleged to be violated", "false": "this article was not at issue"},
}
FACTS_KEYS = ["facts", "case facts", "statement of facts", "the facts"]

# ---------------------------------------------------------------- SCOTUS
ISSUE_AREAS = [
    ("Criminal Procedure", "rights of persons accused or convicted of crimes"),
    ("Civil Rights", "discrimination, voting and other non-First-Amendment civil liberties"),
    ("First Amendment", "speech, press, religion and association"),
    ("Due Process", "procedural and substantive due process outside criminal cases"),
    ("Privacy", "abortion, contraception, privacy and freedom-of-information claims"),
    ("Attorneys", "lawyers' fees, bar admission and attorney discipline"),
    ("Unions", "labor unions and union-employer relations"),
    ("Economic Activity", "business, commerce, regulation, torts and contracts"),
    ("Judicial Power", "jurisdiction, standing and the reach of the courts"),
    ("Federalism", "the division of power between the federal and state governments"),
    ("Interstate Relations", "disputes between states, e.g. boundaries"),
    ("Federal Taxation", "federal tax law and the Internal Revenue Code"),
    ("Miscellaneous", "anything else"),
]
SCOTUS_CHOICE = [
    "Read this US Supreme Court opinion and decide which broad legal issue area it belongs to.",
    "Which Supreme Court Database issue area best describes the subject of this decision?",
    "Classify the case below by its main area of law.",
    "What general category of legal issue does this opinion of the Supreme Court address?",
    "Assign this Supreme Court case to one issue area.",
]
OPINION_KEYS = ["opinion", "decision", "supreme court opinion", "case"]

# ---------------------------------------------------------------- LEDGAR
LEDGAR_CHOICE = [
    "This paragraph comes from a commercial contract filed with the SEC. What is its provision type?",
    "Label the contract clause with the heading it would most likely carry.",
    "Which category of contractual provision does this text belong to?",
    "Classify the agreement provision below by topic.",
    "Pick the label that names the main subject of this contract paragraph.",
]
CLAUSE_KEYS = ["provision", "clause", "contract paragraph", "text"]

# ---------------------------------------------------------------- UNFAIR-ToS
UNFAIR = [  # LexGLUE label order
    ("Limitation of liability", "the provider limits or excludes its liability for losses"),
    ("Unilateral termination", "the provider may suspend or end the service or account at its discretion"),
    ("Unilateral change", "the provider may change the terms or the service without consent"),
    ("Content removal", "the provider may delete or remove user content at will"),
    ("Contract by using", "merely using the service binds the user to the terms"),
    ("Choice of law", "the terms select which law governs disputes"),
    ("Jurisdiction", "disputes must be brought before specific courts"),
    ("Arbitration", "disputes go to arbitration instead of a court"),
]
UNFAIR_NONE = [("None", "the sentence is not one of these unfair terms"), ("not unfair", None),
               ("no unfair term", "none of the eight types applies")]
UNFAIR_CHOICE = [
    "This sentence is from an online platform's terms of service. Is it a potentially unfair term under EU consumer law, and of which type?",
    "Classify the terms-of-service sentence: which kind of unfair clause is it, if any?",
    "Does this consumer contract sentence contain a potentially unfair term? Choose its type or the option for none.",
    "Pick the category of unfair contractual term that this ToS sentence represents; most sentences are fair.",
]
SENTENCE_KEYS = ["sentence", "terms of service sentence", "clause", "tos excerpt"]

# ---------------------------------------------------------------- CaseHOLD
HOLDING_KEYS = [list("ABCDE"), ["(a)", "(b)", "(c)", "(d)", "(e)"], [f"holding {i}" for i in range(1, 6)]]
HOLDING_CHOICE = [
    "In this excerpt from a US court opinion, a citation is followed by <HOLDING>, where the cited case's holding was stated. Which holding belongs there?",
    "Choose the parenthetical holding that the court most likely gave for the citation marked <HOLDING>.",
    "Pick the statement of law that fits the <HOLDING> slot in the passage.",
    "The passage cites a precedent and summarizes its holding at <HOLDING>. Which candidate summary is the right one?",
    "Which of these holdings is the one the opinion attributes to the case cited at the <HOLDING> mark?",
]
PASSAGE_KEYS = ["passage", "opinion excerpt", "excerpt", "context"]


class Overlap:
    """Document-level overlap with a set of test texts: exact word sequence or 13-gram containment."""

    def __init__(self, texts: list[str]):
        self.exact = {tuple(words(t)) for t in texts}
        self.size, self.index = [], defaultdict(list)
        for i, t in enumerate(texts):
            g = self.grams(t)
            self.size.append(len(g))
            for h in g:
                self.index[h].append(i)

    @staticmethod
    def grams(text: str) -> set[int]:
        w = words(text)
        return {hash(tuple(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}

    def near(self, text: str, limit: float = MAX_TEST_CONTAINMENT) -> bool:
        if tuple(words(text)) in self.exact:
            return True
        g = self.grams(text)
        hits = Counter(i for h in g for i in self.index.get(h, ()))
        return any(n / max(1, len(g)) > limit or n / max(1, self.size[i]) > limit for i, n in hits.items())


def jeval_states(prefix: str = None, equals: str = None) -> list[str]:
    d = pd.read_parquet(JEVAL, columns=["source", "state"])
    mask = d.source.str.startswith(prefix) if prefix else d.source == equals
    return [s if isinstance(s, str) else str(s) for s in d[mask].state]


def read(config: str) -> pd.DataFrame:
    local, _ = common.fetch(REPO, ["README.md", f"{config}/train-*.parquet"], revision=REVISION)
    return pd.concat([pd.read_parquet(p) for p in sorted((local / config).glob("train-*.parquet"))], ignore_index=True)


def ledgar_names() -> list[str]:
    local, _ = common.fetch(REPO, ["README.md"], revision=REVISION)
    card = yaml.safe_load((local / "README.md").read_text().split("---")[1])
    (info,) = [c for c in card["dataset_info"] if c["config_name"] == "ledgar"]
    (feature,) = [f for f in info["features"] if f["name"] == "label"]
    names = feature["dtype"]["class_label"]["names"]
    return [names[i] if i in names else names[str(i)] for i in range(len(names))]


def shuffled(rng: random.Random, items: list, p: float = .5) -> list[int]:
    order = list(range(len(items)))
    if rng.random() < p:
        rng.shuffle(order)
    return order


class Builder:
    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(common.ROOT / "models/qwen3.5-0.8b")
        self.codes, _ = fmt.answer_codes(self.tokenizer)
        self.overlap = Overlap(jeval_states(prefix="lex-glue"))

    def tokens(self, e: dict) -> int:
        k = len(fmt.options(e["question"])[0])
        return len(self.tokenizer.encode(fmt.render(self.tokenizer, e["state"], e["question"], self.codes[:k], "echo")))

    def documents(self, config: str, docs, stats: dict):
        """docs yields (doc id, text for the overlap check, [examples]); yields the kept examples."""
        counts = Counter()
        lengths = []
        for doc_id, text, examples in docs:
            counts["documents"] += 1
            if not examples:
                counts["documents_without_question"] += 1
                continue
            if self.overlap.near(text):
                counts["documents_near_jeval"] += 1
                continue
            n = max(self.tokens(e) for e in examples)
            if n > REPORT_TOKENS:
                counts["documents_over_32k_tokens"] += 1
            if n > MAX_PROMPT_TOKENS:
                counts["documents_over_token_cap"] += 1
                continue
            lengths.append(n)
            counts["documents_kept"] += 1
            counts["questions"] += len(examples)
            yield from examples
        q = np.percentile(lengths, [50, 90]).round().tolist() if lengths else [0, 0]
        counts = dict(counts)
        counts["share_over_32k_tokens"] = round(counts.get("documents_over_32k_tokens", 0) / max(1, counts["documents"]), 4)
        counts["prompt_tokens"] = {"p50": q[0], "p90": q[1], "max": int(max(lengths, default=0))}
        stats.setdefault("configs", {})[config] = counts
        print(config, counts, flush=True)

    # ------------------------------------------------------------ long documents
    def ecthr(self, config: str, rng: random.Random, alleged: dict | None):
        df = read(config)
        for i, row in df.iterrows():
            text = "\n".join(str(p) for p in row.text)
            labels = sorted(int(x) for x in row["labels"])
            yield f"{config}:train:{i}", text, list(self.ecthr_questions(config, i, text, labels, rng, alleged))

    def ecthr_questions(self, config, i, text, labels, rng, alleged):
        name_key = rng.choice(ARTICLE_KEYS)
        # Paragraphs stay joined: as a JSON list each "5. The applicant was born in 19xx and lives in X."
        # paragraph is a short field of its own and trips the filter's short-text check (a template line).
        state = text if rng.random() < .5 else {rng.choice(FACTS_KEYS): text}
        if len(labels) <= 1:
            with_rights = rng.random() < .75
            criteria_items = [(name_key(a), ARTICLE_RIGHTS[a] if with_rights else None) for a in ARTICLES]
            gold = labels[0] if labels else len(ARTICLES)
            order = shuffled(rng, criteria_items)
            none_key, none_desc = rng.choice(NONE_OPTION[config])
            criteria = {criteria_items[j][0]: criteria_items[j][1] for j in order}
            criteria[none_key] = none_desc
            gold = order.index(gold) if gold < len(ARTICLES) else len(ARTICLES)
            question, target = common.choice(state, rng.choice(ECTHR_CHOICE[config]), criteria, gold)
            yield common.example(NAME, f"{config}:train:{i}", "legal-classification", state, question, target, "train")
            return
        positive = rng.choice(labels)
        hard = [a for a in (alleged or {}).get(text, ()) if a not in labels]
        negative = rng.choice(hard) if hard and rng.random() < .5 else rng.choice([a for a in range(len(ARTICLES)) if a not in labels])
        for tag, article, truth in (("pos", positive, True), ("neg", negative, False)):
            a = ARTICLES[article]
            instructions = rng.choice(ECTHR_NOUL[config]).format(key=name_key(a), right=ARTICLE_RIGHTS[a])
            question, target = common.noul(instructions, truth, ECTHR_NOUL_CRITERIA[config] if rng.random() < .4 else None)
            yield common.example(NAME, f"{config}:train:{i}:{tag}", "legal-classification", state, question, target, "train")

    def scotus(self, rng: random.Random):
        df = read("scotus")
        for i, row in df.iterrows():
            with_desc = rng.random() < .5
            order = shuffled(rng, ISSUE_AREAS)
            criteria = {ISSUE_AREAS[j][0]: ISSUE_AREAS[j][1] if with_desc else None for j in order}
            state = row.text if rng.random() < .5 else {rng.choice(OPINION_KEYS): row.text}
            question, target = common.choice(state, rng.choice(SCOTUS_CHOICE), criteria, order.index(int(row.label)))
            yield f"scotus:train:{i}", row.text, [
                common.example(NAME, f"scotus:train:{i}", "legal-classification", state, question, target, "train")]

    def long(self, stats: dict):
        rng = random.Random(2026101621)
        b = read("ecthr_b")
        alleged = {"\n".join(str(p) for p in t): [int(x) for x in ls] for t, ls in zip(b.text, b["labels"])}
        yield from self.documents("ecthr_a", self.ecthr("ecthr_a", rng, alleged), stats)
        yield from self.documents("ecthr_b", self.ecthr("ecthr_b", rng, None), stats)
        yield from self.documents("scotus", self.scotus(rng), stats)

    # ------------------------------------------------------------ short texts
    def ledgar(self, rng: random.Random):
        names = ledgar_names()
        assert len(names) == 100, len(names)
        df = read("ledgar")
        for i in sorted(rng.sample(range(len(df)), CAP_LEDGAR)):
            row = df.iloc[i]
            order = shuffled(rng, names, .6)
            criteria = {names[j]: None for j in order}
            state = row.text if rng.random() < .5 else {rng.choice(CLAUSE_KEYS): row.text}
            question, target = common.choice(state, rng.choice(LEDGAR_CHOICE), criteria, order.index(int(row.label)))
            yield f"ledgar:train:{i}", row.text, [
                common.example(SHORT_NAME, f"ledgar:train:{i}", "legal-classification", state, question, target, "train")]

    def unfair_tos(self, rng: random.Random):
        df = read("unfair_tos")
        for i, row in df.iterrows():
            labels = [int(x) for x in row["labels"]]
            if len(labels) > 1:
                continue
            with_desc = rng.random() < .6
            order = shuffled(rng, UNFAIR)
            criteria = {UNFAIR[j][0]: UNFAIR[j][1] if with_desc else None for j in order}
            none_key, none_desc = rng.choice(UNFAIR_NONE)
            criteria[none_key] = none_desc
            gold = order.index(labels[0]) if labels else len(UNFAIR)
            text = row.text.strip()
            state = text if rng.random() < .5 else {rng.choice(SENTENCE_KEYS): text}
            question, target = common.choice(state, rng.choice(UNFAIR_CHOICE), criteria, gold)
            yield f"unfair_tos:train:{i}", text, [
                common.example(SHORT_NAME, f"unfair_tos:train:{i}", "legal-classification", state, question, target, "train")]

    def case_hold(self, rng: random.Random):
        df = read("case_hold")
        for i in sorted(rng.sample(range(len(df)), CAP_CASEHOLD)):
            row = df.iloc[i]
            endings = [str(x) for x in row.endings]
            if len(set(endings)) != 5:
                continue
            keys = rng.choice(HOLDING_KEYS)
            order = shuffled(rng, endings, .3)
            criteria = {keys[k]: endings[j] for k, j in enumerate(order)}
            state = row.context if rng.random() < .5 else {rng.choice(PASSAGE_KEYS): row.context}
            question, target = common.choice(state, rng.choice(HOLDING_CHOICE), criteria, order.index(int(row.label)))
            yield f"case_hold:train:{i}", row.context + "\n" + "\n".join(endings), [
                common.example(SHORT_NAME, f"case_hold:train:{i}", "legal-holding", state, question, target, "train")]

    def short(self, stats: dict):
        rng = random.Random(2026101622)
        yield from self.documents("ledgar", self.ledgar(rng), stats)
        yield from self.documents("unfair_tos", self.unfair_tos(rng), stats)
        yield from self.documents("case_hold", self.case_hold(rng), stats)


def meta(configs: str, note: str) -> dict:
    return {"upstream": f"{REPO} (parquet on main, {configs})", "revision": REVISION, "split": "train",
            "licence": "CC-BY-4.0", "licence_notes": LICENCE, "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "prompt_tokens_measured": "fmt.render echo layout (longer than d3), Qwen3.5 tokenizer; no truncation",
            "max_test_containment": MAX_TEST_CONTAINMENT, "overlap_reference": "JEVal rows with source lex-glue-* (state)",
            "skipped": {"eurlex": "label ids have no names in the lex_glue card/repo/loading script"}, "filter_note": note}


if __name__ == "__main__":
    builder = Builder()
    # build generators add their statistics to meta; write() reads meta only after consuming the rows.
    long_meta = meta("ecthr_a, ecthr_b, scotus", "documents mode: every document was checked in build() against every "
                     "JEVal lex-glue state (exact words, or >20% 13-gram containment either way) and dropped on a match")
    common.write(NAME, builder.long(long_meta), long_meta, documents=True)
    short_meta = meta("ledgar, unfair_tos, case_hold", "strict mode, after the same JEVal lex-glue document check in build()")
    short_meta.update(caps={"ledgar": CAP_LEDGAR, "case_hold": CAP_CASEHOLD},
                      unfair_tos_note="rows with more than one label are skipped; 0 labels -> the none option")
    common.write(SHORT_NAME, builder.short(short_meta), short_meta, documents=False)
