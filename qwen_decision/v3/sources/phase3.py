"""Phase-3 sources (2026-10-18): human-labelled data for the skills g3 lost most on, from datasets that are in
none of our evaluation sets (JEVal, the public index, fresh-v1). Builds:

  python -m qwen_decision.v3.sources.phase3

- siqa-train: Social IQa (Sap et al. 2019, allenai/social_i_qa train, parquet conversion; licence CC-BY-4.0 per
  the AllenAI release, the Hub card has no tag). Social commonsense, 3 options. A context is one item.
- cosmosqa-train: Cosmos QA (Huang et al. 2019, allenai/cosmos_qa train, CC-BY-4.0). Reading comprehension
  that needs commonsense, 4 options ("None of the above" keeps its place: mix.positional_options). One blog
  post is one item.
- mnli-train: MultiNLI (Williams et al. 2018, nyu-mll/multi_nli train, CC-BY-3.0 card; per-genre source
  licences in the paper). 36,000 pairs sampled, balanced over the 3 labels. 75% are 3-way choices, 25% are
  "does the premise entail it" yes/no. Pairs of one premise (promptID) are one item. WANLI is a different
  corpus; ANLI (index, CC-BY-NC) is not used.
- faithdial-train: FaithDial (Dziri et al., TACL 2022, McGill-NLP/FaithDial train+valid, MIT). Information-
  seeking dialogue. The original Wizard-of-Wikipedia response that annotators marked as a hallucination
  (BEGIN) is paired with its human-edited faithful rewrite. 70% ask which response is fully supported by the
  knowledge, 30% ask yes/no about one of the two. The original is longer in 60% of the raw pairs, so pairs
  are sampled to make it longer in exactly half: response length carries no signal (HaluEval dialogue
  rewards the opposite of a length habit, measured 2026-10-18). One dialogue is one item.
- moralstories-train: Moral Stories (Emelin et al. 2021, demelin/moral_stories full, MIT). Situation and
  intention, and in half of the rows the norm. Which of the two actions is the moral one? 12,000 stories.
- ecare-train: e-CARE (Du et al. 2022, github Waste-Wood/e-CARE @ c3b52648, train+dev, MIT). Causal
  reasoning: the more plausible cause or effect of a premise, 2 options. A premise is one item.
- goemotions-train: GoEmotions (Demszky et al. 2020, google-research-datasets/go_emotions simplified train,
  Apache-2.0). Single-label comments only. All 28 emotions are options, as in JEVal. JEVal's go-emotions
  (its test split) and fresh-v1's goemotions (validation) are then trained-on datasets and are marked so
  in their reports.
Labels are human (annotators, writers). Distractors in FaithDial are the human-flagged originals, nothing
is generated.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

RAW = common.RAW / "phase3"


def digest(text: str) -> str:
    return hashlib.sha1(re.sub(r"[\W_]+", " ", text.lower()).strip().encode()).hexdigest()[:12]


def parquet(repo: str, file: str):
    return pq.read_table(RAW / repo.replace("/", "__") / file).to_pylist()


# --- Social IQa ---------------------------------------------------------------------------------------------
SIQA = "siqa-train"
SOCIAL = ["How would you answer the question about this situation?", "Pick the most sensible answer about the people involved.",
          "Which option best fits what people would think, feel or do here?", "Using everyday social sense, choose the answer.",
          "Which response to the question is most plausible?"]


def siqa(stats: dict):
    rng = random.Random(2026101801)
    for i, r in enumerate(parquet("allenai/social_i_qa", "default/train/0000.parquet")):
        options = [r["answerA"], r["answerB"], r["answerC"]]
        if len({o.strip().lower() for o in options}) < 3:
            continue
        state = {"situation": r["context"], "question": r["question"]} if rng.random() < .6 else f"{r['context']}\nQuestion: {r['question']}"
        question, target = common.choice(state, rng.choice(SOCIAL), options, int(r["label"]) - 1)
        yield common.example(SIQA, f"{digest(r['context'])}:{i}", "social-commonsense", state, question, target, "train")


# --- Cosmos QA ----------------------------------------------------------------------------------------------
COSMOS = "cosmosqa-train"
READ = ["Read the passage and answer the question.", "Which answer is best supported by the passage and common sense?",
        "Answer the question about the passage.", "Choose the most reasonable answer given the text."]


def cosmos(stats: dict):
    rng = random.Random(2026101802)
    for r in parquet("allenai/cosmos_qa", "default/train/0000.parquet"):
        options = [r[f"answer{k}"] for k in range(4)]
        if len({o.strip().lower() for o in options}) < 4:
            continue
        doc = (r["id"].split("##") + ["", "", "", ""])[3] or digest(r["context"])
        state = {"passage": r["context"], "question": r["question"]} if rng.random() < .6 else f"{r['context']}\n\nQuestion: {r['question']}"
        question, target = common.choice(state, rng.choice(READ), options, int(r["label"]))
        yield common.example(COSMOS, f"{doc}:{digest(r['id'])}", "reading-commonsense", state, question, target, "train")


# --- MultiNLI -----------------------------------------------------------------------------------------------
MNLI = "mnli-train"
MNLI_ROWS = 36_000
NLI_CHOICE = ["How does the hypothesis relate to the premise?", "Given the premise, is the hypothesis entailed, contradicted, or neither?",
              "Classify the relation between the two statements."]
NLI_YESNO = ["Does the premise entail the hypothesis?", "If the premise is true, must the hypothesis be true?"]
NLI_KEYS = {"entailment": "the premise makes the hypothesis true", "neutral": "the hypothesis may or may not be true",
            "contradiction": "the premise makes the hypothesis false"}


def mnli(stats: dict):
    rng = random.Random(2026101803)
    rows = [r for r in parquet("nyu-mll/multi_nli", "data/train-00000-of-00001.parquet") if r["label"] in (0, 1, 2)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    picked = [r for label in (0, 1, 2) for r in rng.sample(by[label], MNLI_ROWS // 3)]
    names = list(NLI_KEYS)
    for r in sorted(picked, key=lambda r: r["pairID"]):
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        if rng.random() < .75:
            options = names if rng.random() < .5 else dict(NLI_KEYS)
            question, target = common.choice(state, rng.choice(NLI_CHOICE), options, r["label"])
        else:
            question, target = common.noul(rng.choice(NLI_YESNO), r["label"] == 0)
        yield common.example(MNLI, f"{r['promptID']}:{r['pairID']}", "nli", state, question, target, "train")
    stats["sampled"] = f"{MNLI_ROWS} balanced over labels (seed 2026101803) from {len(rows)}"


# --- FaithDial ----------------------------------------------------------------------------------------------
FAITHDIAL = "faithdial-train"
SUPPORTED = ["Which response is fully supported by the knowledge, with nothing made up?",
             "Only one reply sticks to the given knowledge. Which one?",
             "Which response contains no hallucinated information?"]
SUPPORTED_YESNO = ["Is every claim in the response supported by the knowledge?", "Does the response stay faithful to the knowledge?"]


def faithdial(stats: dict):
    rng = random.Random(2026101804)
    pairs = []
    for split in ("train", "valid"):
        for d, dialogue in enumerate(json.loads((RAW / "McGill-NLP__FaithDial/data" / f"{split}.json").read_text())):
            for u, utt in enumerate(dialogue["utterances"]):
                if "Hallucination" in utt["BEGIN"] and utt["original_response"].strip() != utt["response"].strip():
                    pairs.append((f"{split}{d}", u, utt))
    longer = [p for p in pairs if len(p[2]["original_response"]) > len(p[2]["response"])]
    shorter = [p for p in pairs if len(p[2]["original_response"]) < len(p[2]["response"])]
    n = min(len(longer), len(shorter))
    balanced = rng.sample(longer, n) + rng.sample(shorter, n)
    for dialogue, u, utt in sorted(balanced, key=lambda p: (p[0], p[1])):
        history = "\n".join(f"{'User' if (len(utt['history']) - k) % 2 else 'Assistant'}: {h}" for k, h in enumerate(utt["history"]))
        state = {"knowledge": utt["knowledge"], "dialogue history": history}
        faithful, hallucinated = utt["response"].strip(), utt["original_response"].strip()
        if rng.random() < .7:
            options = [faithful, hallucinated]
            rng.shuffle(options)
            question, target = common.choice(state, rng.choice(SUPPORTED), options, options.index(faithful))
        else:
            truth = rng.random() < .5
            state = {**state, "response": faithful if truth else hallucinated}
            question, target = common.noul(rng.choice(SUPPORTED_YESNO), truth)
        yield common.example(FAITHDIAL, f"{dialogue}:{u}", "dialogue-faithfulness", state, question, target, "train")
    stats.update(hallucination_pairs=len(pairs), length_balanced=2 * n)


# --- Moral Stories ------------------------------------------------------------------------------------------
MORAL = "moralstories-train"
MORAL_ASK = ["Which action is the morally right one?", "Which of these actions would most people consider ethical?",
             "Pick the action that does not violate common moral norms."]


def moral(stats: dict):
    rng = random.Random(2026101805)
    for r in map(json.loads, (RAW / "demelin__moral_stories/data/moral_stories_full.jsonl").open()):
        state = {"situation": r["situation"], "intention": r["intention"]}
        if rng.random() < .5:
            state = {"norm": r["norm"], **state}
        options = [r["moral_action"], r["immoral_action"]]
        rng.shuffle(options)
        question, target = common.choice(state, rng.choice(MORAL_ASK), options, options.index(r["moral_action"]))
        yield common.example(MORAL, r["ID"], "ethics", state, question, target, "train")


# --- e-CARE -------------------------------------------------------------------------------------------------
ECARE = "ecare-train"
CAUSE = ["What is the more plausible cause?", "Which option most likely caused this?"]
EFFECT = ["What is the more plausible effect?", "Which option most likely happens as a result?"]


def ecare(stats: dict):
    rng = random.Random(2026101806)
    for split in ("train", "dev"):
        for r in map(json.loads, (RAW / "ecare-github" / f"{split}_full.jsonl").open()):
            options = [r["hypothesis1"], r["hypothesis2"]]
            state = r["premise"] if rng.random() < .5 else {"event": r["premise"]}
            ask = CAUSE if r["ask-for"] == "cause" else EFFECT
            question, target = common.choice(state, rng.choice(ask), options, int(r["label"]))
            yield common.example(ECARE, f"{digest(r['premise'])}:{r['index']}", "causal-commonsense", state, question, target, split)


# --- GoEmotions ---------------------------------------------------------------------------------------------
GOEMOTIONS = "goemotions-train"
EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
            "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
EMOTION_ASK = ["Which emotion does this comment express most?", "What is the main emotion of the writer?",
               "Label the comment with its emotion; choose neutral if none applies."]


def goemotions(stats: dict):
    rng = random.Random(2026101807)
    for r in parquet("google-research-datasets/go_emotions", "simplified/train-00000-of-00001.parquet"):
        if len(r["labels"]) != 1:
            continue
        state = r["text"] if rng.random() < .6 else {"comment": r["text"]}
        question, target = common.choice(state, rng.choice(EMOTION_ASK), EMOTIONS, int(r["labels"][0]))
        yield common.example(GOEMOTIONS, r["id"], "emotion", state, question, target, "train")


BUILDERS = [
    (SIQA, siqa, {"upstream": "allenai/social_i_qa (refs/convert/parquet, train)", "licence": "CC-BY-4.0 (AllenAI release; Hub card untagged)"}),
    (COSMOS, cosmos, {"upstream": "allenai/cosmos_qa (refs/convert/parquet, train)", "licence": "CC-BY-4.0"}),
    (MNLI, mnli, {"upstream": "nyu-mll/multi_nli train", "licence": "CC-BY-3.0 (card); genre sources per the MultiNLI paper"}),
    (FAITHDIAL, faithdial, {"upstream": "McGill-NLP/FaithDial train+valid", "licence": "MIT"}),
    (MORAL, moral, {"upstream": "demelin/moral_stories moral_stories_full.jsonl", "licence": "MIT"}),
    (ECARE, ecare, {"upstream": "github Waste-Wood/e-CARE dataset/{train,dev}_full.jsonl", "revision": "c3b52648adcc919303a297a377811e24d703e197", "licence": "MIT"}),
    (GOEMOTIONS, goemotions, {"upstream": "google-research-datasets/go_emotions simplified train", "licence": "Apache-2.0"}),
]


if __name__ == "__main__":
    import sys
    wanted = set(sys.argv[1:])
    for name, build, meta in BUILDERS:
        if not wanted or name in wanted:
            meta = {**meta, "split": "train (see module docstring)", "label_origin": "human"}
            common.write(name, build(meta), meta)
