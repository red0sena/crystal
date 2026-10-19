"""When2Call train_sft + train_pref as the 4-way "what should the assistant do" choice, plus 2-way response choices.

Upstream: nvidia/When2Call (CC-BY-4.0) at the revision the index pins. The index scores the 3,652
test MCQ rows (questions from BFCL v2 Live), so only `train/when2call_train_sft.jsonl` (15,000)
and `train/when2call_train_pref.jsonl` (9,000) are used.

The train files carry no category field, so the category of an assistant reply is read from it:
`<TOOLCALL>` = tool call; a reply that asks for a detail ("could you", "please provide", "?") =
request for information; a reply that declines ("unable", "can't", "apologies") = cannot answer.
train_sft holds only the last two kinds; train_pref's chosen replies add the 3,000 tool calls. The
"direct answer" option (answering without the tool, which in When2Call is a hallucination) is never
the gold, as in the index. Each train_pref row also gives a 2-way choice between its chosen and
rejected reply texts (shuffled), the closest cast to the index's answer-text MCQ.

Overlap: train questions are generated from other API collections than BFCL Live (unverified by
the plan); the contamination filter indexes all When2Call test and BFCL rows. Most rows it drops
list a tool whose schema is also in ToolRet's tool corpus (APIGen/xLAM tools), not a test question;
the manifest's diagnostic_flags counts both kinds.
"""
from __future__ import annotations

from collections import Counter
import json
import random
import re

from qwen_decision.v3.decontam import Filter, strings
from qwen_decision.v3.sources import common

NAME = "when2call-train"
REPO = "nvidia/When2Call"
REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
ASK = re.compile(r"\?|please (provide|specify|share|tell|let me know|confirm|clarify)|i('ll| will)? need|could you|can you (provide|specify|tell)|need (to know|more|the|some|you)", re.I)
CANNOT = re.compile(r"unable|can't|cannot|can not|not able|don't have (the )?(ability|capabilit|access)|do not have (the )?(ability|capabilit|access)|beyond my|not possible|i'm sorry|apolog", re.I)
CATEGORIES = [
    ("direct", ["answer directly without using a tool", "reply with an answer straight away, no tool"],
     "Give the answer from the assistant's own knowledge without calling any tool."),
    ("tool_call", ["call one of the available tools", "make a tool call"],
     "One of the listed tools fits and every required argument is known, so call it."),
    ("request_for_info", ["ask the user for the missing information", "ask a clarifying question"],
     "A listed tool fits but a required argument is missing, so ask the user for it first."),
    ("cannot_answer", ["say that it cannot help with this", "decline: no suitable tool"],
     "No listed tool can do this and it cannot be answered reliably, so say so."),
]
FOUR_WAY = [
    "What should the assistant do next with this request, given the tools it has?",
    "The assistant can answer, call a tool, ask a follow-up question, or say it cannot help. Which is right here?",
    "Given the user's message and the available functions, choose the assistant's correct course of action.",
    "Decide how the assistant should respond to this user turn, considering which tools are available and what the user supplied.",
    "Pick the right behaviour for a tool-using assistant on this request.",
]
TWO_WAY = [
    "Two candidate replies are shown as options. Which one should the assistant send, given the tools it has?",
    "Which of these replies is the appropriate one for this request and tool set?",
    "Choose the better assistant response: the one that uses, or declines to use, the tools correctly.",
]


def category(reply: str) -> str | None:
    if "<TOOLCALL>" in reply:
        return "tool_call"
    ask, cannot = bool(ASK.search(reply)), bool(CANNOT.search(reply))
    if ask:  # replies that apologise and then ask are requests for the missing detail
        return "request_for_info"
    return "cannot_answer" if cannot else None


def state_of(r: dict, rng: random.Random):
    tools = [json.loads(t) for t in r["tools"]]
    question = r["messages"][-1]["content"] if r["messages"][-1]["role"] == "user" else r["messages"][0]["content"]
    return {rng.choice(["tools", "available_tools", "functions"]): tools, rng.choice(["question", "user_message", "request"]): question}


def four_way(state, gold: str, rng: random.Random):
    style = rng.random()
    order = list(CATEGORIES) if rng.random() < .5 else rng.sample(CATEGORIES, 4)
    if style < .5:
        options = {rng.choice(names): None for _, names, _ in order}
    else:
        options = {names[0]: desc for _, names, desc in order}
    keys = list(options)
    gold_key = keys[[c for c, _, _ in order].index(gold)]
    return common.choice(state, common.instruction(rng, FOUR_WAY), options, keys.index(gold_key))


def build(stats: dict):
    local, _ = common.fetch(REPO, ["train/*", "README.md"], revision=REVISION)
    rng = random.Random(62)
    for i, line in enumerate((local / "train/when2call_train_sft.jsonl").open()):
        r = json.loads(line)
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
        gold = category(r["messages"][1]["content"])
        stats["sft"][gold] += 1
        if gold is None:
            continue
        state = state_of({"tools": r["tools"], "messages": r["messages"][:1]}, rng)
        question, target = four_way(state, gold, rng)
        yield common.example(NAME, f"sft:{i}", "when2call", state, question, target, "train_sft")
    for i, line in enumerate((local / "train/when2call_train_pref.jsonl").open()):
        r = json.loads(line)
        chosen, rejected = r["chosen_response"]["content"], r["rejected_response"]["content"]
        gold = category(chosen)
        stats["pref"][gold] += 1
        state = state_of(r, rng)
        if gold is not None:
            question, target = four_way(state, gold, rng)
            yield common.example(NAME, f"pref:{i}:4way", "when2call", state, question, target, "train_pref")
        if chosen.strip() != rejected.strip():
            replies = [chosen.strip(), rejected.strip()]
            rng.shuffle(replies)
            question, target = common.choice(state, common.instruction(rng, TWO_WAY), replies, replies.index(chosen.strip()))
            yield common.example(NAME, f"pref:{i}:pair", "when2call", state, question, target, "train_pref")


def diagnose(rows: list[dict]) -> dict:
    """Which part of a state the filter objects to (manifest only; write() does the dropping)."""
    flt, out = Filter(common.DECONTAM), Counter()
    for e in rows:
        tools = next(v for v in e["state"].values() if isinstance(v, list))
        text = next(v for v in e["state"].values() if isinstance(v, str))
        out["question_text_flagged"] += flt.flagged([text]) is not None
        out["tool_schemas_flagged"] += flt.flagged(strings(tools)) is not None
    return dict(out)


if __name__ == "__main__":
    stats = {"sft": Counter(), "pref": Counter()}
    rows = list(build(stats))
    stats = {f"{k}_categories": {str(c): n for c, n in v.items()} for k, v in stats.items()}
    stats["diagnostic_flags"] = diagnose(rows)
    common.write(NAME, rows, {"upstream": REPO, "revision": REVISION, "split": "train_sft+train_pref", "licence": "CC-BY-4.0", **stats})
