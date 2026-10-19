"""Phase-6 preference sources (2026-10-19): human judgments of responses and arguments.

  python -m qwen_decision.v3.sources.phase6_pref --fetch          # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase6_pref [name ...]       # build data/phase2/<name>.jsonl

Every gold here is a human judgment: annotators, volunteers or users chose the better of two responses
(model-written or human-written), summaries or arguments, or labelled an argument's stance. No label comes
from a model (MT-Bench's gpt4_pair split and HH-RLHF's red-team/online/rejection-sampled files are not used).
Licences and download locations were vetted in runs/analysis/phase6-candidates-vetted.json and re-read on
the downloaded cards. Only train splits are read. FETCH lists the pinned upstream files.

Shared conventions of the builders:
- The two candidates are shown in random order (seeded), so the gold position is uniform.
- A state is at most MAX_CHARS characters. Responses are never cut (the judgment is about the whole text);
  for HelpSteer3, OASST2 and HH-RLHF, whose judgment is about the final replies only, the oldest context
  turns are left out (marked "[n earlier turns omitted]") until the row fits, and a row whose final user turn
  and replies alone are too long is dropped. Arena battles are judged over the whole conversation, so a
  battle that does not fit is dropped rather than trimmed.
- The item key (first id segment) keeps related rows together: HelpSteer3, HH-RLHF and Arena use the digest
  of the first user turn, since they contain the same conversation at several lengths and a continuation must
  not land on the other side of the dev split from its shorter version; OASST2 the message tree, TL;DR the
  Reddit post, MT-Bench the question, ArgQ and Claim Stance the topic, UKPConvArg the debate-stance file.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import random
import re

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import fetch_spec, hub_dir

RAW = common.RAW / "phase6"
SFF_PARQUET = "a4d82a1068e43c8edee69c452a7e763722bd5496"  # refs/convert/parquet of openai/summarize_from_feedback
UKP_COMMIT = "ca9d24e41b8805ff2e1d3585d552b49bb64a65f7"

# name -> ("github", repo, commit, [path prefixes kept from the tarball]) | ("hub", repo, [allow patterns]) | ("url", url, file)
FETCH = {
    "helpsteer3": ("hub", "nvidia/HelpSteer3", ["README.md", "preference/train.jsonl.gz"]),
    "oasst2": ("hub", "OpenAssistant/oasst2", ["README.md", "data/train-00000-of-00001-88ba0162028a73fc.parquet"]),
    "hhrlhf": ("hub", "Anthropic/hh-rlhf", ["README.md", "helpful-base/train.jsonl.gz", "harmless-base/train.jsonl.gz"]),
    # the Hub repo is script-based; this is the Hub's own parquet conversion of the comparisons train split, pinned
    "tldr-feedback": ("url", "https://huggingface.co/datasets/openai/summarize_from_feedback/resolve/"
                             f"{SFF_PARQUET}/comparisons/train/0000.parquet", "comparisons-train-0000.parquet"),
    "arena55k": ("hub", "lmarena-ai/arena-human-preference-55k", ["README.md", "train.csv"]),
    "mtbench-human": ("hub", "lmsys/mt_bench_human_judgments", ["README.md", "data/human-00000-of-00001-25f4910818759289.parquet"]),
    "argq": ("hub", "ibm-research/argument_quality_ranking_30k", ["README.md", "train.csv"]),
    "ukpconvarg": ("github", "UKPLab/acl2016-convincing-arguments", UKP_COMMIT,
                   ["README.md", "data/readme.txt", "data/UKPConvArg1Strict-CSV/"]),
    "claimstance": ("hub", "ibm-research/claim_stance", ["README.md", "train.csv"]),
}


def fetch_one(name: str) -> dict:
    return fetch_spec(name, FETCH[name], RAW)


MAX_CHARS = 12_000  # characters of state per row (about 3-4k tokens of English, more for CJK text)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def render(turns: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{role}: {text.strip()}" for role, text in turns)


def fit_context(turns: list[tuple[str, str]], budget: int) -> str | None:
    """The conversation before the replies, oldest turns left out until it fits `budget` characters; None
    when even the last turn does not fit."""
    kept, used = [], 0
    for role, text in reversed(turns):
        size = len(role) + len(text.strip()) + 4
        if used + size > budget:
            break
        kept.append((role, text))
        used += size
    if not kept:
        return None
    kept.reverse()
    omitted = len(turns) - len(kept)
    return (f"[{omitted} earlier turn{'s' if omitted > 1 else ''} omitted]\n\n" if omitted else "") + render(kept)


def pair_state(turns: list[tuple[str, str]], first: str, second: str, names=("response A", "response B")) -> dict | None:
    """State for "which reply to this conversation is better": the prompt (one user turn) or the conversation,
    then the two replies; None when the row cannot fit MAX_CHARS."""
    budget = MAX_CHARS - len(first) - len(second)
    if len(turns) == 1:
        prompt = turns[0][1].strip()
        if len(prompt) > budget:
            return None
        return {"prompt": prompt, names[0]: first, names[1]: second}
    context = fit_context(turns, budget - 40)  # room for the "[n earlier turns omitted]" line
    if context is None:
        return None
    return {"conversation": context, names[0]: first, names[1]: second}


def ab(rng: random.Random, better: str, worse: str) -> tuple[str, str, int]:
    """The two texts in random order and the position of the better one."""
    return (better, worse, 0) if rng.random() < .5 else (worse, better, 1)


def unique_pairs(items: list[tuple], counts: collections.Counter) -> list:
    """items: (key, better, payload), key naming the context and the unordered pair of candidates. A pair seen
    more than once (HelpSteer3 repeats rows; UKPConvArg has one argument under two ids; Arena and MT-Bench have
    identical answers from different models) is kept once when its labels agree and dropped when they conflict."""
    by_key = {}
    for key, better, payload in items:
        by_key.setdefault(key, []).append((better, payload))
    kept = []
    for labelled in by_key.values():
        if len({better for better, _ in labelled}) > 1:
            counts["conflicting_labels"] += 1
            continue
        counts["repeated_pair"] += len(labelled) - 1
        kept.append(labelled[0][1])
    return kept


def take_groups(rng: random.Random, rows: list, key, quota: int) -> list:
    """A deterministic sample of whole groups (rows sharing key(row)) of at most `quota` rows, in input order."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    order = sorted(groups)
    rng.shuffle(order)
    chosen, total = set(), 0
    for g in order:
        if total + len(groups[g]) <= quota:
            chosen.add(g)
            total += len(groups[g])
    return [r for r in rows if key(r) in chosen]


# --- HelpSteer3: which response is better (human preference strength) ------------------------------------
HELPSTEER3 = "helpsteer3-train"
HELPSTEER3_DIR = hub_dir("nvidia/HelpSteer3", "f6d145777bcb")
HS3_ASK = ["Which response answers the user's last message better?", "Which of the two responses is more helpful?",
           "Pick the better reply to the user."]
HS3_ROLE = {"user": "User", "assistant": "Assistant"}


def helpsteer3(stats: dict):
    """NVIDIA HelpSteer3 preference config (CC-BY-4.0). Prompts from ShareGPT/WildChat (general, STEM, code,
    multilingual), two responses from ~20 permissively licensed open models, and an overall preference -3..3
    (negative: response 1 better) aggregated from up to 3 human annotators. Train file only; kept
    |preference| >= 2 (one response clearly better). The train file repeats about a third of its rows
    verbatim; each pair is kept once. The context is shown as a prompt or a conversation; the two responses in
    random order. Item: digest of the first user turn."""
    rng = random.Random(2026101921)
    counts = collections.Counter()
    items = []
    for i, r in enumerate(map(json.loads, gzip.open(HELPSTEER3_DIR / "preference/train.jsonl.gz"))):
        strength = int(r["overall_preference"])
        if abs(strength) < 2:
            counts["weak_preference"] += 1
            continue
        better, worse = (r["response1"], r["response2"]) if strength < 0 else (r["response2"], r["response1"])
        better, worse = better.strip(), worse.strip()
        if not better or not worse or _norm(better) == _norm(worse):
            counts["empty_or_same"] += 1
            continue
        key = (json.dumps(r["context"], ensure_ascii=False), *sorted([_norm(better), _norm(worse)]))
        items.append((key, _norm(better), (i, r, better, worse)))
    for i, r, better, worse in unique_pairs(items, counts):
        turns = [(HS3_ROLE[t["role"]], t["content"]) for t in r["context"]]
        first, second, gold = ab(rng, better, worse)
        state = pair_state(turns, first, second)
        if state is None:
            counts["too_long"] += 1
            continue
        question, target = common.choice(state, rng.choice(HS3_ASK), ["response A", "response B"], gold)
        counts[r["domain"]] += 1
        counts["trimmed_context"] += state.get("conversation", "").startswith("[")
        yield common.example(HELPSTEER3, f"{digest(r['context'][0]['content'])}:{i}", "response-preference",
                             state, question, target, "train")
    stats["counts"] = dict(counts)


# --- OASST2: which human-written reply the volunteers ranked higher ---------------------------------------
OASST2 = "oasst2-train"
OASST2_DIR = hub_dir("OpenAssistant/oasst2", "179dd21fc551")
OASST_ASK = ["Which reply is better?", "Which of the two replies would reviewers rank higher?",
             "Pick the better reply to the last message."]
OASST_ROLE = {"prompter": "User", "assistant": "Assistant"}


def oasst2(stats: dict):
    """OpenAssistant OASST2 (Apache-2.0). Volunteer-written conversations in 35+ languages; volunteers ranked
    sibling assistant replies (rank 0 = best; checked: mean human quality label falls with rank). For each
    parent with two or more ranked replies, the best and the worst kept reply are paired; the conversation so
    far (the parent chain) is the context. Every message used is reviewed (review_result true), not deleted
    and not synthetic. Train parquet only. Item: message tree id."""
    rng = random.Random(2026101922)
    rows = pq.read_table(OASST2_DIR / "data/train-00000-of-00001-88ba0162028a73fc.parquet",
                         columns=["message_id", "parent_id", "message_tree_id", "text", "role", "lang", "rank",
                                  "review_result", "deleted", "synthetic"]).to_pylist()
    by_id = {r["message_id"]: r for r in rows}
    ok = lambda m: m is not None and m["review_result"] is True and not m["deleted"] and not m["synthetic"]
    replies = collections.defaultdict(list)
    for r in rows:
        if r["role"] == "assistant" and r["rank"] is not None and ok(r) and r["text"].strip():
            replies[r["parent_id"]].append(r)
    counts = collections.Counter()
    for parent_id in sorted(replies):
        ranked = sorted(replies[parent_id], key=lambda m: m["rank"])
        if len(ranked) < 2:
            counts["one_ranked_reply"] += 1
            continue
        best, worst = ranked[0], ranked[-1]
        if best["rank"] == ranked[1]["rank"] or worst["rank"] == ranked[-2]["rank"] or _norm(best["text"]) == _norm(worst["text"]):
            counts["tied_or_same"] += 1
            continue
        chain, m = [], by_id.get(parent_id)
        while m is not None:
            chain.append(m)
            m = by_id.get(m["parent_id"]) if m["parent_id"] else None
        if not chain or chain[-1]["parent_id"] is not None or not all(ok(c) for c in chain):
            counts["context_not_reviewed"] += 1
            continue
        turns = [(OASST_ROLE[c["role"]], c["text"]) for c in reversed(chain)]
        first, second, gold = ab(rng, best["text"].strip(), worst["text"].strip())
        state = pair_state(turns, first, second, ("reply A", "reply B"))
        if state is None:
            counts["too_long"] += 1
            continue
        question, target = common.choice(state, rng.choice(OASST_ASK), ["reply A", "reply B"], gold)
        counts[f"lang_{best['lang']}"] += 1
        counts["pairs"] += 1
        yield common.example(OASST2, f"{best['message_tree_id']}:{parent_id}", "response-preference", state, question, target, "train")
    stats["counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --- HH-RLHF: the more helpful / the less harmful reply --------------------------------------------------
HHRLHF = "hhrlhf-train"
HHRLHF_DIR = hub_dir("Anthropic/hh-rlhf", "09be8c5bbc57")
HH_ASK = {"helpful-base": ["Which reply is more helpful?", "Which assistant reply serves the user better?",
                           "Pick the more helpful of the two replies."],
          "harmless-base": ["Which reply is more harmless?", "Which of the two replies is less harmful?",
                            "Pick the safer, less harmful reply."]}
HH_FAMILY = {"helpful-base": "response-preference", "harmless-base": "response-safety"}
HH_ROWS = 30_000  # half from each file
HH_ROLE = {"Human": "User", "Assistant": "Assistant"}
_HH_MARK = "\n\nAssistant:"


def hh_turns(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n\n(Human|Assistant): ?", text)
    return [(HH_ROLE[parts[k]], parts[k + 1]) for k in range(1, len(parts) - 1, 2)]


def hhrlhf(stats: dict):
    """Anthropic HH-RLHF (MIT), helpful-base and harmless-base train files only: crowdworkers talked with a
    52B context-distilled model and chose the more helpful (helpful-base) or the more harmless (harmless-base)
    of its two replies. The card says the data is for preference modelling, which is what this asks. The shared
    dialogue is the context, the two differing final replies the options, in random order. Rows whose
    transcripts differ before the last reply are left out. 15k rows per file, sampled as whole items.
    Item: digest of the first human turn (one dialogue appears at several lengths)."""
    rng = random.Random(2026101923)
    counts = collections.Counter()
    for part in ("helpful-base", "harmless-base"):
        rows = []
        for i, r in enumerate(map(json.loads, gzip.open(HHRLHF_DIR / part / "train.jsonl.gz"))):
            chosen, rejected = r["chosen"], r["rejected"]
            ic, ir = chosen.rfind(_HH_MARK), rejected.rfind(_HH_MARK)
            if ic < 0 or chosen[:ic] != rejected[:ir]:
                counts[f"{part}_differs_earlier"] += 1
                continue
            better, worse = chosen[ic + len(_HH_MARK):].strip(), rejected[ir + len(_HH_MARK):].strip()
            if not better or not worse or _norm(better) == _norm(worse):
                counts[f"{part}_empty_or_same"] += 1
                continue
            turns = hh_turns(chosen[:ic])
            if not turns or turns[0][0] != "User" or turns[-1][0] != "User":
                counts[f"{part}_unparsed"] += 1
                continue
            first, second, gold = ab(rng, better, worse)
            state = pair_state(turns, first, second, ("reply A", "reply B"))
            if state is None:
                counts[f"{part}_too_long"] += 1
                continue
            rows.append((digest(turns[0][1]), i, state, gold))
        rows = take_groups(rng, rows, lambda x: x[0], HH_ROWS // 2)
        for group, i, state, gold in rows:
            question, target = common.choice(state, rng.choice(HH_ASK[part]), ["reply A", "reply B"], gold)
            counts[part] += 1
            yield common.example(HHRLHF, f"{group}:{part.split('-')[0]}{i}", HH_FAMILY[part], state, question, target, "train")
    stats["counts"] = dict(counts)


# --- TL;DR feedback: which summary of a Reddit post is better ---------------------------------------------
TLDR = "tldr-feedback-train"
TLDR_FILE = RAW / "tldr-feedback/comparisons-train-0000.parquet"
TLDR_ASK = ["Which summary of the post is better?", "Which TL;DR captures the post better?",
            "Pick the better summary of this Reddit post."]
TLDR_ROWS = 30_000
TLDR_WEAK = {4, 6}  # the 9-point confidence scale's "slightly better" marks (1-4: summary 0, 6-9: summary 1)


def tldr_feedback(stats: dict):
    """OpenAI summarize-from-feedback comparisons (Modified MIT for the feedback data; Reddit TL;DR posts
    CC-BY-4.0, Webis-TLDR-17), read from the Hub's parquet conversion of the train split (the repo's loading
    script is not run). Contracted labelers chose the better of two summaries (model samples or the post's own
    TL;DR) of a Reddit post. Kept: TL;DR rows (every train row is a Reddit post; CNN/DM would have an article and
    none does); rows with a "slightly better" confidence mark (4 or 6) dropped; a summary pair labelled more than
    once is kept once when the labels agree and dropped when they conflict. 30k rows sampled as whole posts.
    Item: Reddit post id."""
    rng = random.Random(2026101924)
    counts = collections.Counter()
    pairs = collections.defaultdict(list)
    for r in pq.read_table(TLDR_FILE).to_pylist():
        info = r["info"]
        if r["split"] != "train" or info.get("article") or not info.get("post"):
            counts["not_tldr_train"] += 1
            continue
        if (r["extra"] or {}).get("confidence") in TLDR_WEAK:
            counts["weak_confidence"] += 1
            continue
        texts = [s["text"].strip() for s in r["summaries"]]
        if _norm(texts[0]) == _norm(texts[1]) or not all(texts):
            counts["empty_or_same"] += 1
            continue
        better, worse = texts[r["choice"]], texts[1 - r["choice"]]
        pairs[(info["id"], *sorted(map(_norm, texts)))].append((info, better, worse))
    rows = []
    for key, labelled in pairs.items():
        if len({_norm(b) for _, b, _ in labelled}) > 1:
            counts["conflicting_labels"] += 1
            continue
        counts["repeat_labels"] += len(labelled) - 1
        rows.append(labelled[0])
    rows = take_groups(rng, rows, lambda x: x[0]["id"], TLDR_ROWS)
    per_post = collections.Counter()
    for info, better, worse in rows:
        post = f"Subreddit: r/{info['subreddit']}\nTitle: {info['title'].strip()}\n\n{info['post'].strip()}"
        first, second, gold = ab(rng, better, worse)
        state = {"post": post, "summary A": first, "summary B": second}
        question, target = common.choice(state, rng.choice(TLDR_ASK), ["summary A", "summary B"], gold)
        per_post[info["id"]] += 1
        counts["rows"] += 1
        yield common.example(TLDR, f"{info['id']}:{per_post[info['id']]}", "summary-preference", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- Chatbot Arena 55k: which model's answer the user voted for -------------------------------------------
ARENA = "arena55k-train"
ARENA_DIR = hub_dir("lmarena-ai/arena-human-preference-55k", "18c298340948")
ARENA_ASK = ["Which response would the user prefer?", "The user compared these two responses. Which did they vote for?",
             "Pick the response the user is more likely to choose."]
ARENA_MULTI_ASK = ["Over the whole conversation, which assistant would the user prefer?",
                   "The user chatted with both assistants. Which one did they vote for?",
                   "Pick the assistant the user is more likely to choose."]


def arena55k(stats: dict):
    """LMSYS Chatbot Arena human preference 55k (Apache-2.0 on the card; Kaggle release). Real users' prompts,
    two anonymous models' answers, and the user's vote. Ties dropped; battles with a missing answer or identical
    answers dropped; a battle repeated with the same prompt and answers kept once (dropped if the votes
    conflict). Single-turn: prompt and two responses; multi-turn: the user turns with both assistants' replies
    interleaved (assistant A / B), judged over the whole conversation, so too-long battles are dropped (not
    trimmed). A/B order randomised. Item: digest of the first user turn."""
    rng = random.Random(2026101925)
    counts = collections.Counter()
    items = []
    with (ARENA_DIR / "train.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["winner_tie"] == "1" or r["winner_model_a"] == r["winner_model_b"]:
                counts["tie"] += 1
                continue
            prompts, a, b = json.loads(r["prompt"]), json.loads(r["response_a"]), json.loads(r["response_b"])
            if not prompts or len(a) != len(prompts) or len(b) != len(prompts) or any(x is None for x in prompts + a + b):
                counts["missing_turn"] += 1
                continue
            if all(_norm(x) == _norm(y) for x, y in zip(a, b)):
                counts["same_answers"] += 1
                continue
            better, worse = (a, b) if r["winner_model_a"] == "1" else (b, a)
            side = lambda answers: "\x00".join(map(_norm, answers))
            key = (side(prompts), *sorted([side(better), side(worse)]))
            items.append((key, side(better), (r["id"], prompts, better, worse)))
    for battle, prompts, better, worse in unique_pairs(items, counts):
        left, right, gold = ab(rng, better, worse)
        if len(prompts) == 1:
            state = {"prompt": prompts[0].strip(), "response A": left[0].strip(), "response B": right[0].strip()}
            options, ask = ["response A", "response B"], ARENA_ASK
        else:
            turns = []
            for p, x, y in zip(prompts, left, right):
                turns += [("User", p), ("Assistant A", x), ("Assistant B", y)]
            state = {"conversation": render(turns)}
            options, ask = ["assistant A", "assistant B"], ARENA_MULTI_ASK
        if sum(len(v) for v in state.values()) > MAX_CHARS:
            counts["too_long"] += 1
            continue
        question, target = common.choice(state, rng.choice(ask), options, gold)
        counts["single_turn" if len(prompts) == 1 else "multi_turn"] += 1
        yield common.example(ARENA, f"{digest(prompts[0])}:{battle}", "response-preference", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- MT-Bench human judgments: expert votes between two models' answers -----------------------------------
MTBENCH = "mtbench-human-train"
MTBENCH_DIR = hub_dir("lmsys/mt_bench_human_judgments", "f7d2896d2cc5")
MT_ASK = {1: ["Which response answers the question better?", "Which of the two answers is better?",
              "Pick the better answer to the question."],
          2: ["Judge only the answers to the user's second (follow-up) message: which assistant answered it better?",
              "Both assistants got the same follow-up question. Whose answer to the follow-up is better?",
              "Compare the second answers of the two conversations. Which is better?"]}


def mtbench_human(stats: dict):
    """MT-Bench human judgments (CC-BY-4.0), the "human" split only (gpt4_pair holds GPT-4 judge labels and is
    never read). Expert annotators (mostly graduate students) compared two of six models' answers to 80
    two-turn MT-Bench questions, judging turn 1 or turn 2. Votes on one (question, turn, pair of answer texts;
    two models sometimes gave the same answer) are combined: tie votes left out, the majority winner is the
    gold, equal votes dropped. Turn 1: the question and both answers; turn 2: both full conversations, asking
    about the second answers; rows over MAX_CHARS dropped. Item: question id."""
    rng = random.Random(2026101926)
    rows = pq.read_table(MTBENCH_DIR / "data/human-00000-of-00001-25f4910818759289.parquet").to_pylist()
    votes = collections.defaultdict(collections.Counter)
    shown, models = {}, {}
    counts = collections.Counter()
    for r in rows:
        if r["winner"] not in ("model_a", "model_b"):
            counts["tie_votes"] += 1
            continue
        side = {}
        for x in ("a", "b"):
            conv = r[f"conversation_{x}"]
            text = conv[1]["content"] if r["turn"] == 1 else json.dumps([t["content"] for t in conv], ensure_ascii=False)
            side[x] = _norm(text)
            shown.setdefault((r["question_id"], r["turn"], side[x]), conv)
        if side["a"] == side["b"]:
            counts["same_answers"] += 1
            continue
        key = (r["question_id"], r["turn"], *sorted(side.values()))
        votes[key][side[r["winner"][-1]]] += 1
        models.setdefault(key, sorted([r["model_a"], r["model_b"]]))
    for key in sorted(votes):
        qid, turn, t1, t2 = key
        v = votes[key]
        if v[t1] == v[t2]:
            counts["split_votes"] += 1
            continue
        winner, loser = (t1, t2) if v[t1] > v[t2] else (t2, t1)
        counts["unanimous" if not v[loser] else "majority"] += 1
        first, second, gold = ab(rng, winner, loser)
        ca, cb = shown[(qid, turn, first)], shown[(qid, turn, second)]
        if turn == 1:
            state = {"question": ca[0]["content"].strip(), "response A": ca[1]["content"].strip(),
                     "response B": cb[1]["content"].strip()}
            options = ["response A", "response B"]
        else:
            show = lambda c: render([("User" if t["role"] == "user" else "Assistant", t["content"]) for t in c])
            state = {"conversation A": show(ca), "conversation B": show(cb)}
            options = ["conversation A", "conversation B"]
        if sum(len(v) for v in state.values()) > MAX_CHARS:
            counts["too_long"] += 1
            continue
        question, target = common.choice(state, rng.choice(MT_ASK[turn]), options, gold)
        counts[f"turn{turn}"] += 1
        m1, m2 = models[key]
        yield common.example(MTBENCH, f"{qid}:t{turn}:{m1}~{m2}", "response-preference", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- IBM ArgQ-Rank-30k: the stronger argument, and the argument's stance ----------------------------------
ARGQ = "argq-train"
ARGQ_DIR = hub_dir("ibm-research/argument_quality_ranking_30k", "590726b3765b")
ARGQ_PAIR_ASK = ["Which argument is stronger?", "Which of the two arguments would be better to use, as is, in a speech for this side?",
                 "Pick the higher-quality argument for this stance."]
ARGQ_STANCE_ASK = {True: ["Does this argument support the topic?", "Is this an argument in favour of the topic?"],
                   False: ["Does this argument oppose the topic?", "Is this an argument against the topic?"]}
ARGQ_STANCE_TEXT = {1: "supporting the topic", -1: "contesting the topic"}
ARGQ_ROWS, ARGQ_PAIRS = 20_000, 14_000  # pairs up to 14k; stance questions fill the rest
ARGQ_GAP = .3          # minimum WA quality difference of a pair
ARGQ_STANCE_CONF = .8  # minimum weighted agreement of the stance label
ARGQ_USES = 3          # an argument appears in at most this many pairs (weak arguments are the scarce side)


def _argument(text: str) -> str:
    return text.strip().lstrip(".`").strip()


def argq(stats: dict):
    """IBM Debater ArgQ-Rank-30k (CC-BY-SA-3.0 per the card's licensing text; the YAML says cc-by-3.0, the
    stricter one is recorded). Crowd contributors wrote arguments for and against 71 topics; 10 annotators per
    argument judged whether they would use it as is in a speech (WA quality score) and its stance. Train file
    (49 topics) only. Two questions:
    - pairs of arguments on the same topic and stance whose WA scores differ by at least 0.3: which is stronger
      (up to 14k pairs, spread evenly over topic-stance groups, each argument in at most 3 pairs; the
      arguments with low scores are few, so the groups run out before 14k);
    - an argument's stance (stance agreement >= 0.8), asked as "supports?" or "opposes?" at random, filling
      the source to 20k rows.
    Item: topic."""
    rng = random.Random(2026101927)
    with (ARGQ_DIR / "train.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == "train"]
    counts = collections.Counter()
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["topic"], int(r["stance_WA"]))].append(r)
    per_group = math.ceil(ARGQ_PAIRS / len(groups))
    pairs = []
    for (topic, stance) in sorted(groups):
        members = sorted(groups[(topic, stance)], key=lambda r: r["argument"])
        candidates = [(x, y) for i, x in enumerate(members) for y in members[i + 1:]
                      if abs(float(x["WA"]) - float(y["WA"])) >= ARGQ_GAP and _norm(x["argument"]) != _norm(y["argument"])]
        rng.shuffle(candidates)
        uses, taken = collections.Counter(), 0
        for x, y in candidates:
            if taken == per_group:
                break
            if uses[x["argument"]] >= ARGQ_USES or uses[y["argument"]] >= ARGQ_USES:
                continue
            uses[x["argument"]] += 1
            uses[y["argument"]] += 1
            taken += 1
            pairs.append((topic, stance, x, y))
    rng.shuffle(pairs)
    per_topic = collections.Counter()
    for topic, stance, x, y in pairs[:ARGQ_PAIRS]:
        better, worse = (x, y) if float(x["WA"]) > float(y["WA"]) else (y, x)
        first, second, gold = ab(rng, _argument(better["argument"]), _argument(worse["argument"]))
        state = {"topic": topic, "stance": ARGQ_STANCE_TEXT[stance], "argument A": first, "argument B": second}
        question, target = common.choice(state, rng.choice(ARGQ_PAIR_ASK), ["argument A", "argument B"], gold)
        per_topic[topic] += 1
        counts["pairs"] += 1
        yield common.example(ARGQ, f"{digest(topic)}:pair{per_topic[topic]}", "argument-quality", state, question, target, "train")
    confident = sorted((r for r in rows if float(r["stance_WA_conf"]) >= ARGQ_STANCE_CONF), key=lambda r: (r["topic"], r["argument"]))
    counts["stance_low_agreement"] = len(rows) - len(confident)
    rng.shuffle(confident)
    for r in confident[:ARGQ_ROWS - counts["pairs"]]:
        ask_support = rng.random() < .5
        state = {"topic": r["topic"], "argument": _argument(r["argument"])}
        question, target = common.noul(rng.choice(ARGQ_STANCE_ASK[ask_support]), (int(r["stance_WA"]) == 1) == ask_support)
        per_topic[r["topic"]] += 1
        counts["stance_yes" if target[1] else "stance_no"] += 1
        yield common.example(ARGQ, f"{digest(r['topic'])}:stance{per_topic[r['topic']]}", "argument-stance", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- UKPConvArg1Strict: the more convincing of two web-debate arguments ------------------------------------
UKPCONVARG = "ukpconvarg-train"
UKP_DIR = RAW / f"UKPLab__acl2016-convincing-arguments@{UKP_COMMIT[:12]}/data/UKPConvArg1Strict-CSV"
UKP_ASK = ["Which argument is more convincing?", "Which of the two arguments makes the stronger case for this side?",
           "Pick the more persuasive argument."]


def _slug(text: str) -> str:
    text = re.sub(r"-+$", "?", text.strip("_")).replace("-", " ").strip()
    return text[:1].upper() + text[1:]


def _ukp_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*<br\s*/?>\s*", "\n", text)).strip()


def ukpconvarg(stats: dict):
    """UKPConvArg1Strict (Habernal & Gurevych 2016; CC-BY-4.0, arguments from createdebate.com CC-BY-3.0 and
    convinceme.net public domain; github UKPLab/acl2016-convincing-arguments pinned). MTurk workers (5 per
    pair) chose the more convincing of two arguments for the same side of a debate; MACE-cleaned, ties and
    transitivity violations removed (the Strict version). 32 debate-stance files (16 debates, the corpus has
    no split). Some arguments appear twice under different ids, so a text pair can repeat: kept once, dropped
    when its labels conflict. The topic and side come from the file name. A/B order randomised. Item: the
    debate-stance file."""
    rng = random.Random(2026101928)
    counts = collections.Counter()
    items = []
    for path in sorted(UKP_DIR.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            assert header[:4] == ["#id", "label", "a1", "a2"], header
            for pair_id, label, a1, a2 in reader:
                if label not in ("a1", "a2"):
                    counts["bad_label"] += 1
                    continue
                a1, a2 = _ukp_text(a1), _ukp_text(a2)
                if not a1 or not a2 or _norm(a1) == _norm(a2):
                    counts["empty_or_same"] += 1
                    continue
                better, worse = (a1, a2) if label == "a1" else (a2, a1)
                items.append(((path.stem, *sorted([_norm(a1), _norm(a2)])), _norm(better), (path.stem, pair_id, better, worse)))
    for stem, pair_id, better, worse in unique_pairs(items, counts):
        debate, side = stem.split("_", 1)
        first, second, gold = ab(rng, better, worse)
        state = {"debate": _slug(debate), "side": _slug(side), "argument A": first, "argument B": second}
        question, target = common.choice(state, rng.choice(UKP_ASK), ["argument A", "argument B"], gold)
        counts["pairs"] += 1
        yield common.example(UKPCONVARG, f"{stem}:{pair_id}", "argument-quality", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- IBM Claim Stance: does a claim support or oppose the topic -------------------------------------------
CLAIMSTANCE = "claimstance-train"
CLAIMSTANCE_DIR = hub_dir("ibm-research/claim_stance", "ec4e2c2ec3e0")
CS_CHOICE_ASK = ["Does the claim support or oppose the topic?", "What is the claim's stance toward the topic?"]
CS_NOUL_ASK = {True: ["Does this claim support the topic?", "Is the claim an argument for the topic?"],
               False: ["Does this claim oppose the topic?", "Is the claim an argument against the topic?"]}
CS_OPTIONS = ["supports the topic", "opposes the topic"]


def claimstance(stats: dict):
    """IBM Debater Claim Stance (Bar-Haim et al. 2017; CC-BY-SA-3.0 per the card's licensing text). Claims
    found in Wikipedia by experts for 55 debate topics, each labelled PRO or CON by experts. Train file only
    (25 topics). Asked as a two-option choice or a yes/no "supports?"/"opposes?" question at random. Item: topic."""
    rng = random.Random(2026101929)
    counts = collections.Counter()
    with (CLAIMSTANCE_DIR / "train.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    for r in rows:
        stance = r["claims.stance"]
        if stance not in ("PRO", "CON"):
            counts["bad_stance"] += 1
            continue
        state = {"topic": r["topicText"].strip(), "claim": r["claims.claimCorrectedText"].strip()}
        if rng.random() < .5:
            options = CS_OPTIONS[:] if rng.random() < .5 else CS_OPTIONS[::-1]
            question, target = common.choice(state, rng.choice(CS_CHOICE_ASK), options, options.index(CS_OPTIONS[stance == "CON"]))
            counts["choice"] += 1
        else:
            ask_support = rng.random() < .5
            question, target = common.noul(rng.choice(CS_NOUL_ASK[ask_support]), (stance == "PRO") == ask_support)
            counts["noul"] += 1
        counts[stance] += 1
        yield common.example(CLAIMSTANCE, f"{r['topicId']}:{r['claims.claimId']}", "claim-stance", state, question, target, "train")
    stats["counts"] = dict(counts)


BUILDERS = [
    (HELPSTEER3, helpsteer3, {"upstream": "nvidia/HelpSteer3@f6d14577 preference/train.jsonl.gz", "licence": "CC-BY-4.0",
                              "url": "https://huggingface.co/datasets/nvidia/HelpSteer3",
                              "label_origin": "human preference (up to 3 annotators) between model-written responses"}),
    (OASST2, oasst2, {"upstream": "OpenAssistant/oasst2@179dd21f data/train parquet", "licence": "Apache-2.0",
                      "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
                      "label_origin": "human (volunteer ranking of human-written sibling replies)"}),
    (HHRLHF, hhrlhf, {"upstream": "Anthropic/hh-rlhf@09be8c5b helpful-base + harmless-base train.jsonl.gz", "licence": "MIT",
                      "url": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
                      "label_origin": "human (crowdworker choice between two model replies)"}),
    (TLDR, tldr_feedback, {"upstream": f"openai/summarize_from_feedback refs/convert/parquet@{SFF_PARQUET[:8]} comparisons/train",
                           "licence": "Modified MIT (feedback data); Reddit TL;DR posts CC-BY-4.0 (Webis-TLDR-17)",
                           "url": "https://github.com/openai/summarize-from-feedback (LICENSE); https://zenodo.org/record/1168855",
                           "label_origin": "human (contracted labelers' choice between two summaries)"}),
    (ARENA, arena55k, {"upstream": "lmarena-ai/arena-human-preference-55k@18c29834 train.csv", "licence": "Apache-2.0",
                       "url": "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k",
                       "label_origin": "human (Arena users' votes between two model answers)"}),
    (MTBENCH, mtbench_human, {"upstream": "lmsys/mt_bench_human_judgments@f7d2896d data/human parquet (gpt4_pair not read)",
                              "licence": "CC-BY-4.0", "url": "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments",
                              "label_origin": "human (expert votes, majority per question/turn/model pair)"}),
    (ARGQ, argq, {"upstream": "ibm-research/argument_quality_ranking_30k@590726b3 train.csv",
                  "licence": "CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0)",
                  "url": "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k",
                  "label_origin": "human (10 crowd annotators per argument: WA quality score, stance)"}),
    (UKPCONVARG, ukpconvarg, {"upstream": f"github UKPLab/acl2016-convincing-arguments@{UKP_COMMIT[:8]} data/UKPConvArg1Strict-CSV",
                              "licence": "CC-BY-4.0 (arguments: createdebate CC-BY-3.0, convinceme public domain)",
                              "url": "https://github.com/UKPLab/acl2016-convincing-arguments",
                              "label_origin": "human (MTurk pairwise convincingness, MACE + transitivity cleaned)"}),
    (CLAIMSTANCE, claimstance, {"upstream": "ibm-research/claim_stance@ec4e2c2e train.csv",
                                "licence": "CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0)",
                                "url": "https://huggingface.co/datasets/ibm-research/claim_stance",
                                "label_origin": "human (expert PRO/CON labels)"}),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", nargs="*", default=None, help="download these (all when none given)")
    p.add_argument("names", nargs="*")
    args = p.parse_args()
    if args.fetch is not None:
        RAW.mkdir(parents=True, exist_ok=True)
        log = RAW / "fetch-pref.jsonl"
        for name in args.fetch or list(FETCH):
            record = fetch_one(name)
            print(json.dumps(record), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
    else:
        wanted = set(args.names)
        for name, build, meta in BUILDERS:
            if not wanted or name in wanted:
                common.write(name, build(meta), meta)
