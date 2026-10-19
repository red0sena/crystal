"""Per-function "should this function be called for this request?" questions from ToolACE and Glaive FC v2.

Upstreams (train only, no official test split in either):
  - Team-ACE/ToolACE `data.json` (Apache-2.0): 11,300 dialogues over 1-9 listed tools; assistant
    calls are `[Name(arg=..), Other(..)]`.
  - glaiveai/glaive-function-calling-v2 (Apache-2.0): 112,960 dialogues over 0-2 listed functions;
    calls are `<functioncall> {"name": ..}`.
Salesforce/xlam-function-calling-60k (CC-BY-4.0) is the preferred source, but it is gated and this
account has no access (403), so it is not used.

A decision point is a user turn and the assistant turns that follow it up to the next user turn.
Called functions are positives; listed-but-uncalled functions are negatives. With probability .6
the tool list also gets 1-12 functions from other dialogues (BFCL live_multiple-sized lists),
skipping any whose name or description shares a content word with the request or with a called
function, so an injected "no" is very unlikely to be a function the request actually needs.
Decision points where the assistant asks the user for missing arguments are skipped (the right
answer there is "relevant but not callable yet", which a yes/no cannot express).
State: the conversation so far (prior turns clipped) and the tool schemas, shuffled.
Up to 2 positives and 2 negatives per decision point; cap 25,000 questions, half from ToolACE (all it
has after the skips) and the rest from Glaive.

Overlap: ToolRet-Queries holds 1,000 ToolACE and 1,000 xLAM queries (and 33 other sources); every
decision point whose user text equals a ToolRet query turn (normalized) is dropped here, and the
contamination filter (which indexes all ToolRet and BFCL rows) catches the rest.
Tool schemas: ToolRet's tool corpus contains ToolACE's tool documents and BFCL/When2Call rows carry
function schemas, so a state that lists such a tool shares text with an index row although the
request is new (a trial run lost 15k of 25k questions this way, 840 to the request text itself).
Decision points listing a tool whose schema shares any kept shingle or short text with the index
are skipped before sampling (counted in the manifest, so they do not use up the cap), and injected
tools come only from schemas that share none; write() still filters every example, so what it drops
is overlap in the conversation itself.
"""
from __future__ import annotations

from collections import Counter
import json
import random
import re

import numpy as np
import pyarrow.parquet as pq

from qwen_decision.v3.decontam import Filter, shingles, short_keys, strings, words
from qwen_decision.v3.sources import common

NAME = "tools-percall"
TOOLACE = ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734")
GLAIVE = ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac")
TOOLRET = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/toolret_queries"
CAP = 25_000
TURN_CHARS = 1_200
MAX_TURNS = 8
ASK = re.compile(r"\?|please (provide|specify|share|tell|let me know|confirm|clarify)|i('ll| will)? need|could you|can you (provide|specify|tell)", re.I)
STOP = set("about after again also another any are been before being both call called calls can could data does each from "
           "have here into just like make more most need only other over please same should some such than that their them then "
           "there these they this those through used using very want what when where which will with would your info information "
           "get details list find search check given based".split())
NOUL = [
    "Should the assistant call `{name}` to handle the user's latest message?",
    "Is `{name}` one of the functions that has to be invoked to serve this request?",
    "Answering the user's most recent turn requires a call to the function `{name}`.",
    "Given the tools on offer, would a correct reply to the latest request include a call to `{name}`?",
    "Decide whether `{name}` should be used for the user's current request.",
]
CHOICE = [
    "Function under consideration: `{name}`. Should it be called for the user's latest request?",
    "Look at the available tools and the conversation. What should happen with `{name}` for the current request?",
    "Tool-routing decision for `{name}`: invoke it for this turn or leave it out?",
]
CHOICE_OPTIONS = [{"call it": "The request needs this function.", "do not call it": "This function is not needed for the request."},
                  {"yes": None, "no": None}, {"invoke": None, "skip": None}]


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower().replace("_", " ")) if len(w) > 3 and w not in STOP}


def clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= TURN_CHARS else text[:TURN_CHARS] + " …"


def toolret_turns() -> set[str]:
    out = set()
    for path in sorted(TOOLRET.glob("*/*.parquet")):
        for q in pq.read_table(path, columns=["query"]).column("query").to_pylist():
            parts = re.split(r"(?:^|\n)(?:user|assistant|system|tool|function)\s*:", q)
            for text in [q] + parts:
                key = " ".join(words(text))
                if len(key.split()) >= 3:
                    out.add(key)
    return out


# ---- ToolACE -------------------------------------------------------------------------------------
TOOLACE_MARK = "Here is a list of functions in JSON format that you can invoke:\n"
TOOLACE_ALT = "Here are the tools you can use:\n"


def toolace_tools(system: str) -> list[dict] | None:
    if TOOLACE_MARK in system:
        tools, _ = json.JSONDecoder().raw_decode(system.split(TOOLACE_MARK, 1)[1].lstrip())
        return tools
    if TOOLACE_ALT in system:
        tools = []
        for line in system.split(TOOLACE_ALT, 1)[1].split("\n"):
            if not line.startswith("{"):
                break
            t = json.loads(line)
            tools.append({"name": t["tool_name"], "description": t.get("definition", ""),
                          "parameters": {"type": "dict", "properties": {a["parameter_name"]: {k: v for k, v in a.items() if k != "parameter_name"}
                                                                        for a in t.get("arguments", [])}, "required": t.get("required", [])}})
        return tools
    return None


def toolace_points(rng: random.Random):
    local, _ = common.fetch(TOOLACE[0], ["data.json", "README.md"], revision=TOOLACE[1])
    for row, x in enumerate(json.loads((local / "data.json").read_text())):
        try:
            tools = toolace_tools(x["system"])
        except (json.JSONDecodeError, KeyError):
            continue
        if not tools:
            continue
        names = [t["name"] for t in tools]
        conv = [{"role": t["from"], "content": t["value"]} for t in x["conversations"]]
        for i, turn in enumerate(conv):
            if turn["role"] != "user":
                continue
            replies = []
            for t in conv[i + 1:]:
                if t["role"] == "user":
                    break
                if t["role"] == "assistant":
                    replies.append(t["content"])
            if not replies:
                continue
            called = set()
            for reply in replies:
                if reply.lstrip().startswith("["):
                    called |= {n for n in names if re.search(r"(?:^\[|,\s*)" + re.escape(n) + r"\(", reply.strip())}
            first = replies[0].strip()
            if first.startswith("[") and not called:
                continue  # a call we cannot attribute to a listed tool
            if not called and ASK.search(first):
                continue
            yield {"source": "toolace", "id": f"toolace:{row}:{i}", "tools": tools, "history": conv[:i + 1], "called": called}


# ---- Glaive --------------------------------------------------------------------------------------
GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
GLAIVE_CALL = re.compile(r"<functioncall>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"")


def glaive_tools(system: str) -> list[dict]:
    text = system.split("Use them if required -", 1)[1] if "Use them if required -" in system else ""
    tools, dec, i = [], json.JSONDecoder(), 0
    while True:
        j = text.find("{", i)
        if j < 0:
            return tools
        try:
            obj, i = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            return tools
        if isinstance(obj, dict) and "name" in obj:
            tools.append(obj)


def glaive_points(rng: random.Random):
    local, _ = common.fetch(GLAIVE[0], ["glaive-function-calling-v2.json", "README.md"], revision=GLAIVE[1])
    roles = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
    for row, x in enumerate(json.loads((local / "glaive-function-calling-v2.json").read_text())):
        tools = glaive_tools(x["system"])
        if not tools:
            continue
        names = [t["name"] for t in tools]
        parts = GLAIVE_TURN.split(x["chat"])
        conv = [{"role": roles[parts[k]], "content": parts[k + 1].replace("<|endoftext|>", "").strip()} for k in range(1, len(parts) - 1, 2)]
        for i, turn in enumerate(conv):
            if turn["role"] != "user":
                continue
            replies = []
            for t in conv[i + 1:]:
                if t["role"] == "user":
                    break
                if t["role"] == "assistant":
                    replies.append(t["content"])
            if not replies:
                continue
            called = {m for r in replies for m in GLAIVE_CALL.findall(r)}
            if called - set(names):
                continue
            if not called and ASK.search(replies[0]):
                continue
            yield {"source": "glaive", "id": f"glaive:{row}:{i}", "tools": tools, "history": conv[:i + 1], "called": called}


# ---- questions -----------------------------------------------------------------------------------
def tool_words(tool: dict) -> set[str]:
    return content_words(tool["name"]) | content_words(str(tool.get("description", "")))


def questions(point: dict, pool: list[dict], rng: random.Random):
    tools = list(point["tools"])
    names = {t["name"] for t in tools}
    request = " ".join(t["content"] for t in point["history"] if t["role"] == "user")
    avoid = content_words(request) | {w for t in tools if t["name"] in point["called"] for w in tool_words(t)}
    injected = []
    if rng.random() < .6:
        want = rng.randint(1, 12)
        for _ in range(want * 4):
            t = rng.choice(pool)
            if t["name"] in names or tool_words(t) & avoid:
                continue
            names.add(t["name"])
            injected.append(t)
            if len(injected) == want:
                break
    tools += injected
    rng.shuffle(tools)
    history = [{"role": t["role"], "content": clip(t["content"])} for t in point["history"][-MAX_TURNS:]]
    request_key = rng.choice(["conversation", "messages", "dialogue"])
    state = {request_key: history, rng.choice(["tools", "available_functions", "functions"]): tools}
    positives = sorted(point["called"])
    listed_neg = [t["name"] for t in point["tools"] if t["name"] not in point["called"]]
    other_neg = [t["name"] for t in injected]
    picks = [(n, True) for n in rng.sample(positives, min(2, len(positives)))]
    negatives = []
    for _ in range(2 if positives else rng.choice([1, 2])):
        source = listed_neg if listed_neg and (not other_neg or rng.random() < .6) else other_neg
        if source:
            negatives.append(source.pop(rng.randrange(len(source))))
    picks += [(n, False) for n in negatives]
    for name, call in picks:
        if rng.random() < .75:
            question, target = common.noul(common.instruction(rng, NOUL, name=name), call)
        else:
            options = rng.choice(CHOICE_OPTIONS)
            keys = list(options) if rng.random() < .5 else list(options)[::-1]
            gold = list(options)[0 if call else 1]
            question, target = common.choice(state, common.instruction(rng, CHOICE, name=name), {k: options[k] for k in keys}, keys.index(gold))
        yield name, call, state, question, target


def build(stats: dict):
    rng = random.Random(1)
    blocked = toolret_turns()
    flt, verdicts = Filter(common.DECONTAM), {}

    def clean(tool: dict) -> bool:
        """No text of the schema touches the index at all (stricter than flagged(), so any mix of clean tools stays clean)."""
        key = json.dumps(tool, sort_keys=True, ensure_ascii=False)
        if key not in verdicts:
            verdicts[key] = not any(short_keys(t) & flt.short or touches(shingles(words(t))) for t in strings(tool))
        return verdicts[key]

    def touches(h: np.ndarray) -> bool:
        if not len(h):
            return False
        at = np.minimum(np.searchsorted(flt.table, h), len(flt.table) - 1)
        return bool((flt.table[at] == h).any())

    total = 0
    for source, points_fn in (("toolace", toolace_points), ("glaive", glaive_points)):
        limit = CAP // 2 if source == "toolace" else CAP - total  # Glaive fills what ToolACE cannot
        points = list(points_fn(rng))
        pool = [t for t in {t["name"]: t for p in points for t in p["tools"]}.values() if clean(t)]
        rng.shuffle(points)
        made, yes = 0, 0
        for p in points:
            if made >= limit:
                break
            if any(" ".join(words(t["content"])) in blocked for t in p["history"] if t["role"] == "user"):
                stats["dropped_toolret_exact"][source] += 1
                continue
            if not all(clean(t) for t in p["tools"]):
                stats["skipped_tool_schema_in_index"][source] += 1
                continue
            for name, call, state, question, target in questions(p, pool, rng):
                yield common.example(NAME, f"{p['id']}:{name}", "tool-call", state, question, target, "train")
                made += 1
                yes += call
        total += made
        stats["questions"][source] = made
        stats["yes_share"][source] = round(yes / max(made, 1), 3)
        stats["decision_points_available"][source] = len(points)


if __name__ == "__main__":
    stats = {"dropped_toolret_exact": Counter(), "skipped_tool_schema_in_index": Counter(), "questions": {}, "yes_share": {},
             "decision_points_available": {}}
    rows = list(build(stats))
    for key in ("dropped_toolret_exact", "skipped_tool_schema_in_index"):
        stats[key] = dict(stats[key])
    common.write(NAME, rows, {"upstream": [TOOLACE[0], GLAIVE[0]], "revision": dict([TOOLACE, GLAIVE]), "split": "train",
                              "licence": {TOOLACE[0]: "Apache-2.0", GLAIVE[0]: "Apache-2.0"}, "cap": CAP,
                              "not_used": "Salesforce/xlam-function-calling-60k (gated, 403 for this account)", **stats})
