"""Scale the text mix toward ~2M rows with established, human-labelled public datasets (train splits only).

User request (2026-11-08): keep the g7+enrichment mix and its option-count ratios, add many reputable datasets
from diverse domains, light augmentation only. Rules kept from the earlier builders:
- train splits only; labels are the datasets' own (human, or source-defined for Sarcasm headlines -> "programmatic");
- every row passes the contamination filter (public suite rows + JEVal, rebuilt in data-scale-v1/decontam);
- rows whose normalized state already appears in the existing pool are skipped (no duplicate items);
- datasets reserved for our internal held-out checks are not used (PIQA, SWAG, COPA, QASC, QuaRTz, WiC, ETHICS,
  TimeDial, epistemic reasoning, presupposition NLI, CB), nor ANLI train (deliberately excluded before), nor TREC (JEVal).
- light augmentation: one question per source item, 2-3 of our own phrasings, binary classification asked as a
  yes/no statement 40% of the time; option order is shuffled at training time anyway.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from qwen_decision.v3 import decontam, format as fmt, train

ROOT = Path("runs/champion-v1/data-scale-v1")
RAW = ROOT / "raw"
OUT = ROOT / "sources"
POOL = [Path("runs/champion-v1/full-g7-v3/mix/text-train.jsonl"), Path("runs/champion-v1/full-g7-v3/mix/phishing-url-train.jsonl")]
SEED = 20261108


def norm(text) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower()))[:400]


def get(repo, filename, revision=None):
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=RAW))


def parquet(repo, filename, revision=None):
    return pq.read_table(get(repo, filename, revision)).to_pylist()


def jsonl(repo, filename):
    return [json.loads(l) for l in get(repo, filename).open(encoding="utf-8") if l.strip()]


def converted(repo, prefix):
    """Rows of a script-based dataset from the Hub's auto-converted parquet branch."""
    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev) if f.startswith(prefix) and f.endswith(".parquet")]
    return [r for f in sorted(files) for r in parquet(repo, f, rev)]


# ------------------------------------------------------------------ question helpers

def choice(phr, options, gold, ordered=False):
    criteria = options if isinstance(options, dict) else {o: None for o in options}
    keys = list(criteria)
    assert 2 <= len(keys) <= fmt.MAX_OPTIONS and len(set(keys)) == len(keys), keys[:5]
    t = [0.] * len(keys)
    t[gold] = 1.
    return {"type": "choice", "instructions": phr, "criteria": criteria}, t, ordered


def noul(phr, truth):
    return {"type": "noul", "instructions": phr}, [1. - float(truth), float(truth)], False


def binary(rng, state_kind, labels, gold, choice_phr, yes_phr):
    """2-way classification: as a choice, or 40% of the time as 'is it <label>?' yes/no."""
    if rng.random() < .4:
        asked = rng.randrange(2)
        return noul(rng.choice(yes_phr).format(label=labels[asked]), asked == gold)
    return choice(rng.choice(choice_phr), list(labels), gold)


# ------------------------------------------------------------------ sources
# Each yields (uid, state, (question, target, ordered), family). Caps in SPECS.

def ag_news(rng):
    labels = ["World", "Sports", "Business", "Science/Technology"]
    for i, r in enumerate(parquet("fancyzhx/ag_news", "data/train-00000-of-00001.parquet")):
        yield i, r["text"], choice(rng.choice(["Which news section does this article belong to?",
                                               "Pick the topic of this news story."]), labels, r["label"]), "topic"


def dbpedia14(rng):
    labels = ["Company", "Educational institution", "Artist", "Athlete", "Office holder", "Mean of transportation",
              "Building", "Natural place", "Village", "Animal", "Plant", "Album", "Film", "Written work"]
    for i, r in enumerate(parquet("fancyzhx/dbpedia_14", "dbpedia_14/train-00000-of-00001.parquet")):
        yield i, {"title": r["title"], "abstract": r["content"]}, choice(rng.choice(
            ["What kind of entity does this encyclopedia entry describe?", "Classify the subject of this Wikipedia abstract."]),
            labels, r["label"]), "entity-type"


def dbpedia_levels(level, rng):
    rows = list(csv.DictReader(get("DeveloperOats/DBPedia_Classes", "DBPEDIA_train.csv").open(encoding="utf-8")))
    labels = sorted({r[level] for r in rows})
    names = [re.sub(r"(?<!^)(?=[A-Z])", " ", x) for x in labels]
    for i, r in enumerate(rows):
        yield i, r["text"], choice(rng.choice([f"Which fine-grained category describes the subject of this text?",
                                               "Assign the most specific ontology class to the entity described here."]),
                                   names, labels.index(r[level])), "entity-type"


def yahoo(rng):
    labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
              "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
    for f in ("yahoo_answers_topics/train-00000-of-00002.parquet",):
        for i, r in enumerate(parquet("community-datasets/yahoo_answers_topics", f)):
            state = {"question": r["question_title"], "details": r["question_content"], "best_answer": r["best_answer"]}
            yield i, state, choice(rng.choice(["Which forum category fits this question?",
                                               "Choose the topic of this community Q&A thread."]), labels, r["topic"]), "topic"


def yelp(rng):
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, r in enumerate(parquet("Yelp/yelp_review_full", "yelp_review_full/train-00000-of-00001.parquet")):
        yield i, r["text"], choice(rng.choice(["How many stars did the reviewer most likely give?",
                                               "Predict the star rating of this review."]), labels, r["label"], True), "sentiment"


def amazon_polarity(rng):
    for i, r in enumerate(parquet("fancyzhx/amazon_polarity", "amazon_polarity/train-00000-of-00004.parquet")):
        yield i, {"title": r["title"], "review": r["content"]}, binary(rng, "review", ["negative", "positive"], r["label"],
            ["Is this product review negative or positive?", "What is the overall sentiment of the review?"],
            ["The review is {label} overall.", "Is the customer's opinion {label}?"]), "sentiment"


def imdb(rng):
    for i, r in enumerate(parquet("stanfordnlp/imdb", "plain_text/train-00000-of-00001.parquet")):
        yield i, r["text"], binary(rng, "review", ["negative", "positive"], r["label"],
            ["Did the critic like the movie?", "Classify the sentiment of this movie review."],
            ["This movie review is {label}.", "Is the reviewer's verdict {label}?"]), "sentiment"


def sst2(rng):
    for i, r in enumerate(parquet("stanfordnlp/sst2", "data/train-00000-of-00001.parquet")):
        yield i, r["sentence"], binary(rng, "phrase", ["negative", "positive"], r["label"],
            ["What sentiment does this phrase express?", "Is this snippet from a review negative or positive?"],
            ["The sentiment of this phrase is {label}.", "Does the snippet sound {label}?"]), "sentiment"


def sst5(rng):
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    for i, r in enumerate(jsonl("SetFit/sst5", "train.jsonl")):
        yield i, r["text"], choice(rng.choice(["Rate the sentiment of this sentence on a five-point scale.",
                                               "How positive or negative is this sentence?"]), labels, r["label"], True), "sentiment"


def emotion(rng):
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for i, r in enumerate(parquet("dair-ai/emotion", "split/train-00000-of-00001.parquet")):
        yield i, r["text"], choice(rng.choice(["Which emotion does the writer express?", "Name the main feeling in this message."]),
                                   labels, r["label"]), "emotion"


TWEET = {
    "sentiment": (["negative", "neutral", "positive"], ["What is the sentiment of this tweet?", "Is the tweet negative, neutral or positive?"]),
    "emoji": (["❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘", "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"],
              ["Which emoji did the author most likely attach to this tweet?", "Predict the emoji that goes with this tweet."]),
    "offensive": (["not offensive", "offensive"], ["Is this tweet offensive?", "Does the tweet contain offensive language?"]),
    "hate": (["not hateful", "hateful"], ["Is this tweet hate speech against immigrants or women?", "Does the tweet express hate toward a group?"]),
    "irony": (["not ironic", "ironic"], ["Is this tweet ironic?", "Does the author use irony here?"]),
}


def tweet_eval(task, rng):
    labels, phr = TWEET[task]
    for i, r in enumerate(parquet("cardiffnlp/tweet_eval", f"{task}/train-00000-of-00001.parquet")):
        yield i, r["text"], choice(rng.choice(phr), labels, r["label"]), "social-media"


def tweet_stance(rng):
    for target, cfg in (("abortion", "stance_abortion"), ("atheism", "stance_atheism"), ("climate change", "stance_climate"),
                        ("feminism", "stance_feminist"), ("Hillary Clinton", "stance_hillary")):
        for i, r in enumerate(parquet("cardiffnlp/tweet_eval", f"{cfg}/train-00000-of-00001.parquet")):
            yield f"{cfg}:{i}", r["text"], choice(rng.choice([f"What stance does this tweet take on {target}?",
                                                               f"Is the author against, in favor of, or neutral about {target}?"]),
                                                   ["none", "against", "favor"], r["label"]), "stance"


def hwu64(rng):
    names = {r["id"]: r["name"] for r in parquet("DeepPavlov/hwu64", "intents/intents-00000-of-00001.parquet")}
    labels = [names[k] for k in sorted(names)]
    for i, r in enumerate(parquet("DeepPavlov/hwu64", "data/train-00000-of-00001.parquet")):
        yield i, r["utterance"], choice(rng.choice(["Which intent does this request to a home assistant express?",
                                                     "Route this user command to its intent."]),
                                        labels, sorted(names).index(r["label"])), "intent"


def newsgroups(rng):
    rows = jsonl("SetFit/20_newsgroups", "train.jsonl")
    labels = sorted({r["label_text"] for r in rows})
    for i, r in enumerate(rows):
        if len(r["text"].split()) >= 5:
            yield i, r["text"][:4000], choice(rng.choice(["Which Usenet newsgroup was this post written for?",
                                                           "Pick the discussion group this message belongs to."]),
                                               labels, labels.index(r["label_text"])), "topic"


def pubmed_rct(rng):
    labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    for i, r in enumerate(jsonl("armanc/pubmed-rct20k", "train.jsonl")):
        if str(r.get("label", "")).upper() in labels:
            yield i, r["text"], choice(rng.choice(["Which section of a clinical-trial abstract does this sentence come from?",
                                                   "Label the rhetorical role of this abstract sentence."]),
                                       [l.lower() for l in labels], labels.index(r["label"].upper())), "scientific"


def glue(task, rng):
    path = f"{task}/train-00000-of-00001.parquet"
    for i, r in enumerate(parquet("nyu-mll/glue", path)):
        if r["label"] < 0:
            continue
        if task == "mnli":
            yield i, {"premise": r["premise"], "hypothesis": r["hypothesis"]}, choice(rng.choice(
                ["Does the premise entail, contradict, or say nothing definite about the hypothesis?",
                 "What is the relation between the two sentences?"]), ["entailment", "neutral", "contradiction"], r["label"]), "nli"
        elif task == "qqp":
            yield i, {"question_1": r["question1"], "question_2": r["question2"]}, binary(rng, "pair", ["different", "duplicate"], r["label"],
                ["Do these two questions ask the same thing?", "Are the questions duplicates or different?"],
                ["The two questions are {label}.", "Would you mark this pair as {label}?"]), "paraphrase"
        elif task == "qnli":
            yield i, {"question": r["question"], "sentence": r["sentence"]}, binary(rng, "pair", ["answers it", "does not answer it"], r["label"],
                ["Does the sentence contain the answer to the question?", "Is the question answered by this sentence?"],
                ["The sentence {label}.", "Relative to the question, the sentence {label}: true?"]), "qa-verification"
        elif task == "mrpc":
            yield i, {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, binary(rng, "pair", ["not equivalent", "equivalent"], r["label"],
                ["Do the two sentences mean the same thing?", "Are these sentences paraphrases?"],
                ["The sentences are {label}.", "Is the pair {label} in meaning?"]), "paraphrase"
        elif task == "rte":
            yield i, {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, binary(rng, "pair", ["entailed", "not entailed"], r["label"],
                ["Is the hypothesis entailed by the premise?", "Does the first text imply the second?"],
                ["The hypothesis is {label}.", "Is the second sentence {label} by the first?"]), "nli"
        elif task == "cola":
            yield i, r["sentence"], binary(rng, "sentence", ["unacceptable", "acceptable"], r["label"],
                ["Is this sentence grammatically acceptable English?", "Would a linguist accept this sentence as well-formed?"],
                ["This sentence is grammatically {label}.", "Is the sentence {label} English?"]), "linguistics"


def snli(rng):
    for i, r in enumerate(parquet("stanfordnlp/snli", "plain_text/train-00000-of-00001.parquet")):
        if r["label"] in (0, 1, 2):
            yield i, {"premise": r["premise"], "hypothesis": r["hypothesis"]}, choice(rng.choice(
                ["Given the caption, is the hypothesis true, false, or undetermined?", "Classify the relation of the hypothesis to the premise."]),
                ["entailment", "neutral", "contradiction"], r["label"]), "nli"


def art(rng):
    for i, r in enumerate(parquet("allenai/art", "anli/train-00000-of-00001.parquet")):
        if r["hypothesis_1"].strip() == r["hypothesis_2"].strip():
            continue  # a few aNLI items repeat the same hypothesis twice
        yield i, {"beginning": r["observation_1"], "ending": r["observation_2"]}, choice(rng.choice(
            ["Which explanation best fills the gap between the beginning and the ending?", "What most plausibly happened in between?"]),
            [r["hypothesis_1"], r["hypothesis_2"]], r["label"] - 1), "abductive"


def race(rng):
    for i, r in enumerate(parquet("ehovy/race", "all/train-00000-of-00001.parquet")):
        if len(set(r["options"])) == 4:
            yield i, r["article"], choice(r["question"] if r["question"].strip() else "Choose the best answer.",
                                          r["options"], "ABCD".index(r["answer"])), "reading"


def sciq(rng):
    for i, r in enumerate(parquet("allenai/sciq", "data/train-00000-of-00001.parquet")):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        if len(set(opts)) == 4:
            yield i, r["support"] or "", choice(r["question"], opts, 0), "science"


def medmcqa(rng):
    for i, r in enumerate(parquet("openlifescienceai/medmcqa", "data/train-00000-of-00001.parquet")):
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        if len(set(opts)) == 4 and r["cop"] in (0, 1, 2, 3) and r.get("choice_type") == "single":
            yield i, {"subject": r["subject_name"]}, choice(r["question"], opts, r["cop"]), "medical"


def aqua(rng):
    for i, r in enumerate(parquet("deepmind/aqua_rat", "raw/train-00000-of-00001.parquet")):
        opts = [re.sub(r"^[A-E]\)", "", o).strip() for o in r["options"]]
        if len(opts) == 5 and len(set(opts)) == 5 and r["correct"] in "ABCDE":
            yield i, "", choice(r["question"], opts, "ABCDE".index(r["correct"])), "math"


def mmlu_aux(rng):
    for i, r in enumerate(parquet("cais/mmlu", "auxiliary_train/train-00000-of-00001.parquet")):
        r = r.get("train", r)
        if len(set(r["choices"])) == len(r["choices"]) >= 2:
            yield i, "", choice(r["question"], r["choices"], r["answer"]), "knowledge"


def squad2(rng):
    for i, r in enumerate(parquet("rajpurkar/squad_v2", "squad_v2/train-00000-of-00001.parquet")):
        answerable = bool(r["answers"]["text"])
        yield i, {"passage": r["context"], "question": r["question"]}, noul(rng.choice(
            ["The passage contains the answer to the question.", "Can the question be answered from the passage alone?"]), answerable), "qa-verification"


def boolq(rng):
    for i, r in enumerate(parquet("google/boolq", "data/train-00000-of-00001.parquet")):
        yield i, {"passage": r["passage"]}, noul(r["question"].rstrip("?") + "?", r["answer"]), "reading"


def csqa(rng):
    for i, r in enumerate(parquet("tau/commonsense_qa", "data/train-00000-of-00001.parquet")):
        opts = list(r["choices"]["text"])
        if len(set(opts)) == len(opts) and r["answerKey"] in r["choices"]["label"]:
            yield i, "", choice(r["question"], opts, list(r["choices"]["label"]).index(r["answerKey"])), "commonsense"


def medqa(rng):
    for i, r in enumerate(jsonl("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl")):
        keys = sorted(r["options"])
        opts = [r["options"][k] for k in keys]
        if len(set(opts)) == len(opts):
            yield i, "", choice(r["question"], opts, keys.index(r["answer_idx"])), "medical"


def sarcasm_headlines(rng):
    rows = jsonl("raquiba/Sarcasm_News_Headline", "train.json")  # JSON lines despite the .json name
    for i, r in enumerate(rows):
        yield i, r["headline"], binary(rng, "headline", ["serious news", "satire"], int(r["is_sarcastic"]),
            ["Is this headline from a satirical outlet or a serious news site?", "Real news or satire?"],
            ["This headline is {label}.", "Would you classify the headline as {label}?"]), "sarcasm"


def figqa(rng):
    rows = list(csv.DictReader(get("nightingal3/fig-qa", "train.csv").open(encoding="utf-8")))
    for i, r in enumerate(rows):
        if r.get("labels") in ("0", "1"):
            yield i, r["startphrase"], choice(rng.choice(["What does this figurative phrase mean?",
                                                          "Pick the correct interpretation of the metaphor."]),
                                              [r["ending1"], r["ending2"]], int(r["labels"])), "figurative"


def scienceqa(rng):
    for i, r in enumerate(parquet("derek-thomas/ScienceQA", "data/train-00000-of-00001-1028f23e353fbe3e.parquet")):
        if r.get("image") is None and len(set(r["choices"])) == len(r["choices"]) >= 2:
            yield i, r.get("hint") or "", choice(r["question"], list(r["choices"]), r["answer"]), "science"


def mutual(rng):
    for i, r in enumerate(jsonl("tasksource/mutual", "train.jsonl")):
        opts = r["options"]
        if len(set(opts)) == len(opts) and r["answers"] in "ABCD":
            yield i, r["article"], choice(rng.choice(["Which reply best continues this conversation?",
                                                      "Pick the most logical next utterance."]), opts, "ABCD".index(r["answers"])), "dialogue"


def siqa(rng):
    for i, r in enumerate(converted("allenai/social_i_qa", "default/train")):
        opts = [r["answerA"], r["answerB"], r["answerC"]]
        if len(set(opts)) == 3:
            yield i, r["context"], choice(r["question"], opts, int(r["label"]) - 1), "commonsense"


def cosmos(rng):
    for i, r in enumerate(converted("allenai/cosmos_qa", "default/train")):
        opts = [r[f"answer{k}"] for k in range(4)]
        if len(set(opts)) == 4:
            yield i, r["context"], choice(r["question"], opts, int(r["label"])), "reading"


# ---- second wave (2026-11-08): fill the option-count buckets the first wave left short (yes/no, 2-way, 3-way, 5-10)

def civil_toxicity(rng):
    """Jigsaw Civil Comments: crowd-rated attribute fractions; only clear items (>=0.5 yes, <0.1 no), negatives thinned."""
    attrs = {"toxicity": ["Is this comment toxic?", "Would moderators flag this comment as toxic?"],
             "insult": ["Does this comment insult someone?", "Is this comment insulting?"],
             "identity_attack": ["Does this comment attack people for their identity?", "Is there an identity-based attack here?"]}
    for f in ("data/train-00000-of-00002.parquet",):
        for i, r in enumerate(parquet("google/civil_comments", f)):
            a = rng.choice(list(attrs))
            v = r[a]
            if 0.1 <= v < 0.5 or len(r["text"].split()) < 4 or (v < 0.1 and rng.random() > .12):
                continue
            yield i, r["text"][:3000], noul(rng.choice(attrs[a]), v >= .5), "toxicity"


def multirc(rng):
    for i, r in enumerate(parquet("aps/super_glue", "multirc/train-00000-of-00001.parquet")):
        phr = rng.choice(['Question: {q}\nIs "{a}" a correct answer?', 'For the question "{q}", is the answer "{a}" right?'])
        yield i, r["paragraph"], noul(phr.format(q=r["question"], a=r["answer"]), r["label"] == 1), "reading"


def paws(rng):
    for i, r in enumerate(parquet("google-research-datasets/paws", "labeled_final/train-00000-of-00001.parquet")):
        yield i, {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, binary(rng, "pair", ["different meaning", "paraphrase"], r["label"],
            ["Do the two sentences mean the same thing?", "Is the second sentence a paraphrase of the first?"],
            ["The two sentences are a {label} pair.", "Would you label this pair as {label}?"]), "paraphrase"


def yelp_polarity(rng):
    for i, r in enumerate(parquet("fancyzhx/yelp_polarity", "plain_text/train-00000-of-00001.parquet")):
        yield i, r["text"][:4000], binary(rng, "review", ["negative", "positive"], r["label"],
            ["Was the customer's experience negative or positive?", "Classify the sentiment of this business review."],
            ["The review is {label}.", "Is the reviewer's experience {label}?"]), "sentiment"


def scitail(rng):
    for i, r in enumerate(parquet("allenai/scitail", "snli_format/train-00000-of-00001.parquet")):
        if r["gold_label"] in ("entailment", "neutral"):
            yield i, {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, binary(rng, "pair", ["not supported", "supported"],
                int(r["gold_label"] == "entailment"),
                ["Does the science passage support the hypothesis?", "Is the hypothesis backed by the premise?"],
                ["The hypothesis is {label} by the premise.", "Would you say the statement is {label}?"]), "science-nli"


def topic_yesno(which, rng):
    """Fresh items of a topic dataset asked as 'is it about <label>?' (true label half the time)."""
    if which == "yahoo":
        labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
                  "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
        rows = parquet("community-datasets/yahoo_answers_topics", "yahoo_answers_topics/train-00001-of-00002.parquet")
        get_ = lambda r: ({"question": r["question_title"], "details": r["question_content"], "best_answer": r["best_answer"]}, r["topic"])
        phr = ["Does this question belong in the {label} category?", "Is this Q&A thread about {label}?"]
    else:
        labels = ["company", "educational institution", "artist", "athlete", "office holder", "means of transportation",
                  "building", "natural place", "village", "animal", "plant", "album", "film", "written work"]
        rows = parquet("fancyzhx/dbpedia_14", "dbpedia_14/train-00000-of-00001.parquet")
        get_ = lambda r: ({"title": r["title"], "abstract": r["content"]}, r["label"])
        phr = ["Is the subject of this entry a {label}?", "Does this abstract describe a {label}?"]
    for i, r in enumerate(rows):
        state, gold = get_(r)
        asked = gold if rng.random() < .5 else rng.choice([k for k in range(len(labels)) if k != gold])
        yield i, state, noul(rng.choice(phr).format(label=labels[asked]), asked == gold), "topic"


def vitaminc(rng):
    crit = {"SUPPORTS": "the evidence supports the claim", "REFUTES": "the evidence contradicts the claim",
            "NOT ENOUGH INFO": "the evidence is not enough to decide"}
    for i, r in enumerate(jsonl("tals/vitaminc", "train.jsonl")):
        if r["label"] in crit:
            yield r["unique_id"], {"evidence": r["evidence"], "claim": r["claim"]}, choice(rng.choice(
                ["Does the evidence support or refute the claim, or is it not enough?", "Check the claim against the evidence."]),
                crit, list(crit).index(r["label"])), "fact-verification"


def ocnli(rng):
    names = ["neutral", "entailment", "contradiction"]  # HF ClassLabel order for clue/ocnli
    order = ["entailment", "neutral", "contradiction"]
    for i, r in enumerate(parquet("clue/clue", "ocnli/train-00000-of-00001.parquet")):
        if 0 <= r["label"] < 3:
            yield i, {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, choice(rng.choice(
                ["Does the premise entail, contradict, or leave open the hypothesis?", "What is the relation between these two Chinese sentences?"]),
                order, order.index(names[r["label"]])), "nli"


def dream(rng):
    for i, r in enumerate(converted("dataset-org/dream", "plain_text/train")):
        if len(set(r["choice"])) == len(r["choice"]) and r["answer"] in r["choice"]:
            yield i, "\n".join(r["dialogue"]), choice(r["question"], list(r["choice"]), r["choice"].index(r["answer"])), "dialogue-reading"


def emotion_more(rng):
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for i, r in enumerate(parquet("dair-ai/emotion", "unsplit/train-00000-of-00001.parquet")):
        yield i, r["text"], choice(rng.choice(["What emotion is the writer expressing?", "Pick the feeling behind this message."]),
                                   labels, r["label"]), "emotion"


def amazon_stars(rng):
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, r in enumerate(jsonl("SetFit/amazon_reviews_multi_en", "train.jsonl")):
        yield i, r["text"], choice(rng.choice(["What star rating goes with this product review?", "How many stars did the buyer give?"]),
                                   labels, int(r["label"]), True), "sentiment"


SPECS = {  # name: (builder, cap, label_origin)
    "ag-news": (ag_news, 30000, "human"), "dbpedia14": (dbpedia14, 25000, "human"),
    "dbpedia-l2": (lambda rng: dbpedia_levels("l2", rng), 15000, "human"), "dbpedia-l3": (lambda rng: dbpedia_levels("l3", rng), 25000, "human"),
    "yahoo-topics": (yahoo, 30000, "human"), "yelp-full": (yelp, 25000, "human"), "amazon-polarity": (amazon_polarity, 15000, "human"),
    "imdb": (imdb, 10000, "human"), "sst2": (sst2, 15000, "human"), "sst5": (sst5, 8544, "human"), "emotion": (emotion, 16000, "human"),
    "tweet-sentiment": (lambda rng: tweet_eval("sentiment", rng), 35000, "human"),
    "tweet-emoji": (lambda rng: tweet_eval("emoji", rng), 15000, "human"),
    "tweet-offensive": (lambda rng: tweet_eval("offensive", rng), 10000, "human"),
    "tweet-hate": (lambda rng: tweet_eval("hate", rng), 9000, "human"),
    "tweet-irony": (lambda rng: tweet_eval("irony", rng), 2862, "human"), "tweet-stance": (tweet_stance, 3000, "human"),
    "hwu64": (hwu64, 12000, "human"), "20newsgroups": (newsgroups, 11314, "human"), "pubmed-rct": (pubmed_rct, 25000, "human"),
    "mnli-more": (lambda rng: glue("mnli", rng), 40000, "human"), "qqp": (lambda rng: glue("qqp", rng), 30000, "human"),
    "qnli": (lambda rng: glue("qnli", rng), 25000, "human"), "mrpc": (lambda rng: glue("mrpc", rng), 3668, "human"),
    "rte": (lambda rng: glue("rte", rng), 2490, "human"), "cola": (lambda rng: glue("cola", rng), 8551, "human"),
    "snli-more": (snli, 30000, "human"), "art-anli": (art, 30000, "human"), "race": (race, 40000, "human"),
    "sciq": (sciq, 11679, "human"), "medmcqa-more": (medmcqa, 30000, "human"), "aqua-more": (aqua, 20000, "human"),
    "mmlu-aux-more": (mmlu_aux, 30000, "human"), "squad2-more": (squad2, 30000, "human"), "boolq-more": (boolq, 9427, "human"),
    "csqa-more": (csqa, 9741, "human"), "medqa-usmle": (medqa, 10178, "human"),
    "sarcasm-headlines": (sarcasm_headlines, 10000, "programmatic"), "figqa": (figqa, 10000, "human"),
    "scienceqa-text": (scienceqa, 8000, "human"), "mutual": (mutual, 7088, "human"),
    "siqa-more": (siqa, 20000, "human"), "cosmosqa-more": (cosmos, 15000, "human"),
    # second wave
    "civil-toxicity": (civil_toxicity, 30000, "human"), "multirc": (multirc, 27243, "human"), "paws-more": (paws, 35000, "human"),
    "yelp-polarity": (yelp_polarity, 25000, "human"), "scitail-more": (scitail, 23596, "human"),
    "yahoo-yesno": (lambda rng: topic_yesno("yahoo", rng), 20000, "human"), "dbpedia-yesno": (lambda rng: topic_yesno("dbpedia", rng), 15000, "human"),
    "vitaminc-more": (vitaminc, 30000, "human"), "ocnli": (ocnli, 25000, "human"), "dream": (dream, 6116, "human"),
    "emotion-more": (emotion_more, 10000, "human"), "amazon-stars": (amazon_stars, 10000, "human"),
}


_POOL_KEYS = None


def pool_keys():
    global _POOL_KEYS
    if _POOL_KEYS is None:
        keys = set()
        for p in POOL:
            for line in p.open():
                r = json.loads(line)
                keys.add(hashlib.sha1(norm(json.dumps(r["state"], ensure_ascii=False) + " " + json.dumps(r["question"].get("instructions"), ensure_ascii=False)).encode()).hexdigest()[:16])
                keys.add(hashlib.sha1(norm(json.dumps(r["state"], ensure_ascii=False)).encode()).hexdigest()[:16])
        _POOL_KEYS = keys
    return _POOL_KEYS


def other_states(name):
    """State keys of every other data-scale source already built (so e.g. yahoo-yesno never reuses a yahoo-topics item)."""
    keys = set()
    for p in OUT.glob("*.jsonl"):
        if p.stem != name:
            for line in p.open():
                st = json.loads(line)["state"]
                if st:
                    keys.add(hashlib.sha1(norm(json.dumps(st, ensure_ascii=False)).encode()).hexdigest()[:16])
    return keys


def build_one(name):
    out = OUT / f"{name}.jsonl"
    if (OUT / f"{name}.manifest.json").exists():
        return json.loads((OUT / f"{name}.manifest.json").read_text())
    builder, cap, origin = SPECS[name]
    rng = random.Random(f"{SEED}:{name}")
    items = list(builder(rng))
    random.Random(f"{SEED}:{name}:order").shuffle(items)
    flt = decontam.Filter(ROOT / "decontam")
    keys = pool_keys()
    others = other_states(name)
    kept, counts, seen = [], collections.Counter(), set()
    for uid, state, (question, target, ordered), family in items:
        if len(kept) >= cap:
            break
        sk = hashlib.sha1(norm(json.dumps(state, ensure_ascii=False)).encode()).hexdigest()[:16] if state else None
        qk = hashlib.sha1(norm(json.dumps(state, ensure_ascii=False) + " " + json.dumps(question["instructions"], ensure_ascii=False)).encode()).hexdigest()[:16]
        if qk in seen or qk in keys or (sk and len(norm(json.dumps(state))) > 80 and sk in keys and question["type"] != "choice"):
            counts["duplicate"] += 1
            continue
        if sk and sk in others:
            counts["used_by_other_source"] += 1
            continue
        why = flt.flagged(decontam.strings(state) + decontam.strings(question["instructions"]), options=decontam.strings(question.get("criteria")))
        if why:
            counts["contaminated"] += 1
            continue
        seen.add(qk)
        kept.append({"id": f"{name}:{uid}", "original_id": f"{name}:{uid}", "source": name, "family": family, "modality": "text",
                     "kind": question["type"], "state": state, "question": question, "target": target, "ordered": ordered,
                     "item": f"{name}:{uid}", "documents": [], "split": "train", "label_origin": origin, "target_kind": "single",
                     "training_component": "data-scale-v1"})
    OUT.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"name": name, "cap": cap, "available": len(items), "kept": len(kept), **counts, "label_origin": origin}
    train.atomic_json(OUT / f"{name}.manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    return manifest


# ------------------------------------------------------------------ mix assembly

FORMAT = Path("runs/champion-v1/format-rebuild-v1")
FORMAT_SOURCES = ("acos-format-train", "isarcasm-format-train")  # evaluator-format rebuilds (user-approved 11-08) ...
FORMAT_REPLACES = ("acos-train", "isarcasm-train")                 # ... replace the old converters of the same data
BUCKETS = ("yes/no", "2-way", "3", "4", "5-10", "11-50", "51+")
# first-wave sources over-filled the 4-way and 51+ buckets; keep only the first N (already shuffled) rows
TRIMS = {"medmcqa-more": 20000, "mmlu-aux-more": 20000, "aqua-more": 15000, "dbpedia-l3": 10000, "dbpedia-l2": 10000}


def bucket(q) -> str:
    if q["type"] == "noul":
        return "yes/no"
    keys = list(q["criteria"])
    n = len(keys)
    if n == 2:
        return "yes/no" if {k.lower() for k in keys} <= {"yes", "no", "true", "false"} else "2-way"
    return "3" if n == 3 else "4" if n == 4 else "5-10" if n <= 10 else "11-50" if n <= 50 else "51+"


DEV = Path("runs/champion-v1/full-g7-v3/mix/text-dev.jsonl")  # checkpoint-selection dev set of the full-FT run


def state_key(state):
    text = norm(json.dumps(state, ensure_ascii=False))
    return hashlib.sha1(text.encode()).hexdigest()[:16] if len(text) >= 20 else None  # empty/trivial states don't count


def mix():
    rows, per_source, dev_overlap = [], collections.Counter(), collections.Counter()
    dev_keys = {k for k in (state_key(json.loads(l)["state"]) for l in DEV.open()) if k}
    def take(path, limit=None, skip=(), check_dev=True):
        n = 0
        for line in path.open():
            r = json.loads(line)
            if r["source"] in skip:
                continue
            if check_dev and state_key(r["state"]) in dev_keys:  # new rows never share a state with the selection dev set
                dev_overlap[r["source"]] += 1
                continue
            if limit is not None and n >= limit:
                break
            rows.append(line if line.endswith("\n") else line + "\n"); per_source[r["source"]] += 1; n += 1
    for p in POOL:
        take(p, skip=FORMAT_REPLACES, check_dev=False)  # the g7 pool/dev split is kept as it was
    for name in FORMAT_SOURCES:
        take(FORMAT / f"{name}.jsonl")
    for name in SPECS:
        p = OUT / f"{name}.jsonl"
        if p.exists():
            take(p, TRIMS.get(name))
    random.Random(f"{SEED}:mix").shuffle(rows)
    dst = ROOT / "mix"
    dst.mkdir(parents=True, exist_ok=True)
    buckets = collections.Counter()
    with (dst / "text-train.jsonl").open("w") as f:
        for line in rows:
            f.write(line)
            buckets[bucket(json.loads(line)["question"])] += 1
    n = len(rows)
    manifest = {"rows": n, "buckets_pct": {b: round(100 * buckets[b] / n, 2) for b in BUCKETS}, "buckets": dict(buckets),
                "pool": [str(p) for p in POOL], "removed_from_pool": list(FORMAT_REPLACES), "format_rebuild": list(FORMAT_SOURCES),
                "trims": TRIMS, "removed_dev_state_overlap": dict(dev_overlap), "per_source": dict(sorted(per_source.items())),
                "dev": "runs/champion-v1/full-g7-v3/mix/text-dev.jsonl (unchanged, for checkpoint selection)"}
    train.atomic_json(dst / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("rows", "buckets_pct")}), flush=True)
    return manifest


if __name__ == "__main__":
    if sys.argv[1:] == ["mix"]:
        mix(); sys.exit()
    names = sys.argv[1:] or list(SPECS)
    for n in names:
        try:
            build_one(n)
        except Exception as e:
            print(json.dumps({"name": n, "error": repr(e)[:400]}), flush=True)
