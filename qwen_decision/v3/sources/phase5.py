"""Phase-5 sources (2026-10-19): more kinds of decisions, each with human or computed gold and an open licence.

  python -m qwen_decision.v3.sources.phase5 --fetch          # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase5 [name ...]       # build data/phase2/<name>.jsonl

Chosen against the gaps left after g4b (entity linking, finance tables, tool/action choice, preference,
fresh-v1 CB/QASC) and for breadth. Each source was checked against the JEVal sources, the public index
families and fresh-v1 (none of them is one of those datasets), and its licence was read on the upstream page.
The builders follow below; FETCH lists the pinned upstream files.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import json
from pathlib import Path
import random
import re
import subprocess
import tarfile
import urllib.request
import zipfile

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest

RAW = common.RAW / "phase5"

# name -> ("github", repo, commit, [path prefixes kept from the tarball]) | ("hub", repo, [allow patterns]) | ("url", url, file)
FETCH = {
    "tabfact": ("github", "wenhuchen/Table-Fact-Checking", "2ab782ba42b5808076ac91fec846473aa5315a79",
                ["LICENSE", "README.md", "collected_data/", "data/all_csv/", "data/train_id.json", "data/val_id.json",
                 "data/test_id.json", "tokenized_data/train_examples.json", "tokenized_data/val_examples.json"]),
    "folio": ("github", "Yale-LILY/FOLIO", "5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d", ["LICENSE", "README.md", "data/v0.0/"]),
    "abcd": ("github", "asappresearch/abcd", "6b8700ce67c6b37b062dd7a60abc76d7ef832a97",
             ["LICENSE", "README.md", "data/abcd_v1.1.json.gz", "data/guidelines.json", "data/kb.json", "data/ontology.json"]),
    "taskmaster": ("github", "google-research-datasets/Taskmaster", "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
                   ["README.md", "TM-1-2019/", "TM-2-2020/", "TM-3-2020/"]),
    "condaqa": ("hub", "lasha-nlp/CONDAQA", ["README.md", "train.json", "dev.json"]),
    "ropes": ("hub", "allenai/ropes", ["README.md", "plain_text/train-*", "plain_text/validation-*"]),
    "helpsteer2": ("hub", "nvidia/HelpSteer2", ["README.md", "train.jsonl.gz", "preference/*"]),
    "finer139": ("hub", "nlpaueb/finer-139", ["README.md", "dataset_infos.json", "finer139.zip"]),
    "fewnerd": ("hub", "DFKI-SLT/few-nerd", ["README.md", "supervised/train-*", "supervised/validation-*"]),
    "drop": ("hub", "ucinlp/drop", ["README.md", "data/train-*"]),
    # Zeshel (Logeswaran et al. 2019): CC-BY-SA per https://github.com/lajanugen/zeshel README, which links this file.
    "zeshel": ("url", "https://drive.usercontent.google.com/download?id=1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO&export=download&confirm=t",
               "zeshel.tar.bz2"),
}


def fetch_one(name: str) -> dict:
    return fetch_spec(name, FETCH[name], RAW)


def fetch_spec(name: str, spec: tuple, raw: Path) -> dict:
    """Download one upstream ("hub" | "github" | "url" spec, as in FETCH) under `raw`; returns the record to log.
    Hub snapshots go to data/raw/<repo>@<revision> (common.fetch); downloaded files are only read, never run."""
    kind, *spec = spec
    if kind == "hub":
        repo, patterns = spec
        local, revision = common.fetch(repo, patterns)
        return {"name": name, "repo": repo, "revision": revision, "path": str(local)}
    if kind == "github":
        repo, commit, keep = spec
        local = raw / f"{repo.replace('/', '__')}@{commit[:12]}"
        local.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        with urllib.request.urlopen(url, timeout=600) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                path = member.name.split("/", 1)[1] if "/" in member.name else ""
                if not member.isfile() or not any(path == k or (k.endswith("/") and path.startswith(k)) for k in keep):
                    continue
                if ".." in Path(path).parts:
                    continue
                target = local / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
        return {"name": name, "repo": repo, "revision": commit, "path": str(local)}
    url, file = spec
    local = raw / name
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(local / file), url], check=True)
    return {"name": name, "url": url, "path": str(local / file)}


def hub_dir(repo: str, revision: str) -> Path:
    return common.RAW / f"{repo.replace('/', '__')}@{revision[:12]}"


def readable_camel(tag: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", tag)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r" ([,.;:%)\]’'])", r"\1", text)
    return re.sub(r"([($\[‘]) ", r"\1", text)


# --- TabFact: is a statement about a Wikipedia table true? ---------------------------------------------------
TABFACT = "tabfact-train"
TABFACT_DIR = RAW / "wenhuchen__Table-Fact-Checking@2ab782ba42b5"
TABFACT_ASK = ["Based on the table, is the statement true?", "Does the table support this statement?",
               "According to the table, is this claim correct?"]


def tabfact(stats: dict):
    """Chen et al. 2020 (MIT repository; tables from WikiTables). Statements written and labelled by annotators,
    collected_data r1 (simple) and r2 (complex), tables of the official train split only."""
    rng = random.Random(2026101901)
    train = set(json.loads((TABFACT_DIR / "data/train_id.json").read_text()))
    counts = collections.Counter()
    for part in ("r1_training_all.json", "r2_training_all.json"):
        for table_id, (statements, labels, caption) in sorted(json.loads((TABFACT_DIR / "collected_data" / part).read_text()).items()):
            if table_id not in train:
                counts["outside_train_split"] += len(statements)
                continue
            rows = [line.split("#") for line in (TABFACT_DIR / "data/all_csv" / table_id).read_text().splitlines() if line.strip()]
            table = (f"{caption}\n" if caption else "") + "\n".join(" | ".join(c.strip() for c in row) for row in rows)
            for i, (statement, label) in enumerate(zip(statements, labels)):
                state = {"table": table, "statement": statement} if rng.random() < .5 else f"Table:\n{table}\n\nStatement: {statement}"
                question, target = common.noul(rng.choice(TABFACT_ASK), int(label) == 1)
                counts["true" if int(label) == 1 else "false"] += 1
                yield common.example(TABFACT, f"{table_id.removesuffix('.html.csv')}:{part[:2]}-{i}", "table-fact-verification",
                                     state, question, target, "train")
    stats["counts"] = dict(counts)


# --- CondaQA: questions whose answer turns on a negation ----------------------------------------------------
CONDAQA = "condaqa-train"
CONDAQA_DIR = hub_dir("lasha-nlp/CONDAQA", "3c9caa2f2f69")
CONDAQA_LABEL = {"YES": 0, "NO": 1, "DON'T KNOW": 2}
CONDAQA_OPTIONS = ["yes", "no", "the passage does not say"]
CONDAQA_ASK = ["Answer the question about the passage.", "What does the passage imply for this question?",
               "Read the passage and answer the question."]


def condaqa(stats: dict):
    """Ravichander et al. 2022 (Apache-2.0). Wikipedia passages with a negation cue, crowd questions, and the
    crowd's edited passages (negation removed, scope changed, paraphrased) asked the same question. Train and
    dev; span answers (about 2%) are left out. One original passage and its edits are one item."""
    rng = random.Random(2026101902)
    counts = collections.Counter()
    for split, file in (("train", "train.json"), ("dev", "dev.json")):
        for r in map(json.loads, (CONDAQA_DIR / file).open()):
            if r["label"] not in CONDAQA_LABEL:
                counts["span_answer"] += 1
                continue
            state = {"passage": r["sentence1"], "question": r["sentence2"]}
            question, target = common.choice(state, rng.choice(CONDAQA_ASK), CONDAQA_OPTIONS, CONDAQA_LABEL[r["label"]])
            counts[r["label"]] += 1
            yield common.example(CONDAQA, f"{split}{r['PassageID']}:{r['PassageEditID']}:{r['QuestionID']}:{r['SampleID']}",
                                 "negation-reading", state, question, target, split)
    stats["counts"] = dict(counts)


# --- ROPES: apply a background rule to a new situation ------------------------------------------------------
ROPES = "ropes-train"
ROPES_DIR = hub_dir("allenai/ropes", "d59f1e2ee2b4")
_PAIRS = [("more", "less"), ("higher", "lower"), ("increase", "decrease"), ("increases", "decreases"),
          ("increased", "decreased"), ("larger", "smaller"), ("bigger", "smaller"), ("faster", "slower"),
          ("stronger", "weaker"), ("warmer", "cooler"), ("hotter", "colder"), ("greater", "smaller"), ("high", "low"),
          ("more likely", "less likely"), ("yes", "no"), ("better", "worse"), ("longer", "shorter"), ("heavier", "lighter"),
          ("older", "younger"), ("earlier", "later"), ("thicker", "thinner"), ("wider", "narrower"), ("closer", "farther"),
          ("increasing", "decreasing"), ("raise", "lower"), ("rise", "fall"), ("positive", "negative"), ("more", "fewer")]
OPPOSITE = {a: b for a, b in _PAIRS} | {b: a for a, b in _PAIRS}
_NOT_NAMES = set("""The A An In On At If When After Before Both One Two Three His Her Their It Its He She They We I You This That These
Those Which What While Since As However Also Then There So But And For To With From During Each Some All Many Most Other Another
Earth Sun Moon Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September
October November December Mr Mrs Ms Dr Day Year Team Group Town City Country Farm Lake River Island Planet Village State""".split())


def _n(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def ropes_options(question: str, answer: str, situation: str) -> list[str] | None:
    """The two (or more) answers the question chooses between, gold included, or None when they are not clear."""
    gold = _n(answer)
    clause = re.split(r"[,:;]", question.rstrip(" ?"))[-1]
    if " or " in clause:
        left, right = clause.rsplit(" or ", 1)
        left, right = left.strip(), right.strip()
        short = " ".join(left.split()[-len(right.split()):])
        for take in [short] + ([left] if len(left.split()) <= len(right.split()) + 2 else []):
            options = [take, right]
            if _n(take) != _n(right) and sum(_n(o) == gold for o in options) == 1:
                return [answer if _n(o) == gold else o for o in options]
    if gold in OPPOSITE:
        return [answer, OPPOSITE[gold]]
    if re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", answer.strip()) and answer.strip() in situation:
        # a named person/place/farm: the other names in the situation's sentences that mention the answer
        names = {}
        for sentence in re.split(r"(?<=[.!?])\s+", situation):
            if answer.strip() in sentence:
                for name in re.findall(r"\b[A-Z][a-z]+\b", sentence):
                    if name not in _NOT_NAMES and _n(name) != gold and _n(name) not in _n(question):
                        names.setdefault(name, None)
        if names:
            return [answer] + list(names)[:3]
    m = re.fullmatch(r"(.*?\b)([A-Z]|\d+)", answer.strip())
    if m and m.group(1).strip():
        head = m.group(1)
        found = {}
        for x in re.finditer(re.escape(head) + r"(?:[A-Z]|\d+)\b", situation, re.I):
            found.setdefault(_n(x.group(0)), x.group(0))
        found = list(found.values())
        if len(found) >= 2 and any(_n(f) == gold for f in found):
            return [answer if _n(f) == gold else f for f in found][:8]
    return None


def ropes(stats: dict):
    """Lin et al. 2019 (CC-BY-4.0). Crowd-written situations and questions on a science background passage.
    The answer is a span; the options are the alternatives the question names ("cup A or cup B"), an opposite
    ("more"/"less"), or the situation's other labelled items ("cup A" when the answer is "cup B"). Questions
    without clear alternatives are left out. One background passage is one item."""
    rng = random.Random(2026101903)
    counts = collections.Counter()
    for split in ("train", "validation"):
        for r in pq.read_table(ROPES_DIR / f"plain_text/{split}-00000-of-00001.parquet").to_pylist():
            answers = r["answers"]["text"]
            if len(answers) != 1:
                counts["several_answers"] += 1
                continue
            options = ropes_options(r["question"], answers[0], r["situation"])
            if not options:
                counts["no_options"] += 1
                continue
            options = list({_n(o): o for o in options}.values())
            rng.shuffle(options)
            gold = [_n(o) for o in options].index(_n(answers[0]))
            state = {"background": r["background"], "situation": r["situation"]}
            question, target = common.choice(state, r["question"], options, gold)
            counts[f"{len(options)}_options"] += 1
            yield common.example(ROPES, f"{digest(r['background'])}:{r['id']}", "rule-application", state, question, target, split)
    stats["counts"] = dict(counts)


# --- FOLIO: first-order logic in natural language -----------------------------------------------------------
FOLIO = "folio-train"
FOLIO_DIR = RAW / "Yale-LILY__FOLIO@5d7bb84c7eda/data/v0.0"
FOLIO_OPTIONS = {"true": "the premises prove the conclusion", "false": "the premises prove the conclusion wrong",
                 "uncertain": "the premises do not settle it"}
FOLIO_LABEL = {"True": 0, "False": 1, "Unknown": 2, "Uncertain": 2}
FOLIO_ASK = ["Given only the premises, is the conclusion true, false, or uncertain?",
             "Do the premises prove the conclusion, refute it, or leave it open?"]


def folio(stats: dict):
    """Han et al. 2022 (CC-BY-SA-4.0, github Yale-LILY/FOLIO). Expert-written premises and conclusions with
    labels checked by a first-order logic prover. Train and validation; one premise set (story) is one item."""
    rng = random.Random(2026101904)
    for split in ("train", "validation"):
        for i, r in enumerate(map(json.loads, (FOLIO_DIR / f"folio-{split}.jsonl").open())):
            premises = r["premises"]
            if isinstance(premises, str) and premises.startswith("["):
                premises = ast.literal_eval(premises)
            premises = "\n".join(premises) if isinstance(premises, list) else premises
            state = {"premises": premises, "conclusion": r["conclusion"]}
            question, target = common.choice(state, rng.choice(FOLIO_ASK), FOLIO_OPTIONS, FOLIO_LABEL[r["label"]])
            story = digest(premises)  # the validation file has no story ids; one premise set is one story
            yield common.example(FOLIO, f"{story}:{split[0]}{i}", "logic", state, question, target, split)


# --- ABCD: customer-service actions under a written policy --------------------------------------------------
ABCD = "abcd-train"
ABCD_DIR = RAW / "asappresearch__abcd@6b8700ce67c6/data"
ABCD_ACT_ASK = ["Following the policy, what should the agent do next?", "Which action should the agent take now?",
                "Given the guidelines and the chat so far, pick the agent's next action."]
ABCD_INTENT_ASK = ["What does the customer need help with?", "Which request is this customer making?"]
# guideline titles whose words differ from the subflow key
ABCD_TITLES = {"Reset Two-Factor Auth": "reset_2fa", "Return Due to Stain": "return_stain", "Return Due to Color": "return_color",
               "Return Due to Size": "return_size", "Shipping Status": "status", "Manage Shipping": "manage", "Missing Item": "missing",
               "Shipping Cost": "cost", "Boots FAQ": "boots", "Shirt FAQ": "shirt", "Jeans FAQ": "jeans", "Jacket FAQ": "jacket",
               "Pricing FAQ": "pricing", "Membership FAQ": "membership", "Timing FAQ": "timing", "Policy FAQ": "policy",
               "Invalid Credit Card": "credit_card", "Cart Not Updating": "shopping_cart", "Search Not Working": "search_results",
               "Website Too Slow": "slow_speed"}


def abcd(stats: dict):
    """Chen et al. 2021 (MIT). Agents (trained crowd workers) follow a written policy in customer-service chats
    and click actions ("Pull up Account", "Validate Purchase"). Two questions per train chat:
    - the next action, given the policy of the chat's subflow and the chat up to the action (30 actions);
    - the customer's request (55 subflows), from the opening turns.
    One chat is one item."""
    rng = random.Random(2026101905)
    data = json.load(gzip.open(ABCD_DIR / "abcd_v1.1.json.gz"))
    ontology = json.loads((ABCD_DIR / "ontology.json").read_text())
    guidelines = json.loads((ABCD_DIR / "guidelines.json").read_text())
    actions = [a for group in ontology["actions"].values() for a in group]
    shown = {a: a.replace("-", " ").title() for a in actions}
    policy, names = {}, {}
    keys = [s for subs in ontology["intents"]["subflows"].values() for s in subs]
    by_tokens = {tuple(sorted(k.split("_"))): k for k in keys}
    for flow in guidelines.values():
        for title, sub in flow["subflows"].items():
            steps = []
            for i, a in enumerate(sub.get("actions", []), 1):
                steps.append(f"{i}. {a['text']}" + "".join(f"\n   - {t}" for t in a.get("subtext", [])))
                key = a.get("button", "").lower().replace(" ", "-")
                if key in shown:
                    shown[key] = a["button"]
            key = by_tokens.get(tuple(sorted(_n(title).split()))) or ABCD_TITLES.get(title)
            if key:
                notes = sub.get("instructions") or []
                notes = "\n".join(notes) if isinstance(notes, list) else notes
                policy[key], names[key] = f"{title}\n" + (f"{notes}\n" if notes else "") + "\n".join(steps), title
    stats["subflows_with_policy"] = f"{len(policy)} of {len(keys)}"
    counts = collections.Counter()
    intents = [names.get(k, k.replace("_", " ")) for k in keys]
    for chat in data["train"]:
        turns, delexed = chat["original"], chat["delexed"]
        subflow = delexed[0]["targets"][0]
        lines = []
        customer_turns = 0
        for t, ((speaker, text), d) in enumerate(zip(turns, delexed)):
            if speaker == "action":
                button = d["targets"][2]
                if button in shown and subflow in policy:
                    state = {"policy": policy[subflow], "chat": "\n".join(lines) or "(no messages yet)"}
                    question, target = common.choice(state, rng.choice(ABCD_ACT_ASK), [shown[a] for a in actions], actions.index(button))
                    counts["action"] += 1
                    yield common.example(ABCD, f"{chat['convo_id']}:a{t}", "policy-action", state, question, target, "train")
                lines.append(f"[agent action] {shown.get(button, button)}: {text}")
                continue
            lines.append(f"{speaker.capitalize()}: {text}")
            if speaker == "customer":
                customer_turns += 1
                if customer_turns == 2 and subflow in keys:
                    state = "\n".join(lines)
                    question, target = common.choice(state, rng.choice(ABCD_INTENT_ASK), intents, keys.index(subflow))
                    counts["intent"] += 1
                    yield common.example(ABCD, f"{chat['convo_id']}:intent", "support-intent", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- HelpSteer2: which response is better (human ratings) ---------------------------------------------------
HELPSTEER = "helpsteer2-train"
HELPSTEER_DIR = hub_dir("nvidia/HelpSteer2", "990b2711a361")
HS_PREF_ASK = ["Which response answers the prompt better?", "Which of the two responses is more helpful?",
               "Pick the better response to the prompt."]
HS_ATTRIBUTE = {"correctness": ["Which response is more factually correct?", "Which response has fewer mistakes?"],
                "coherence": ["Which response is clearer and more coherent?"],
                "verbosity": ["Which response is more verbose?", "Which response says more than the prompt needs?"],
                "complexity": ["Which response takes more expertise to write?"]}


def _hs_text(text: str) -> str:
    return re.sub(r"\s*<extra_id_1>(User|Assistant)\s*", lambda m: f"\n\n{m.group(1)}: ", text).strip()


def helpsteer2(stats: dict):
    """NVIDIA HelpSteer2 (CC-BY-4.0). Responses written by models; preferences and 0-4 attribute ratings by
    human annotators (3-5 per response). Kept: train-split preferences of strength 2-3 (one response clearly
    better), and response pairs on one prompt whose mean rating of an attribute differs by 2 or more.
    The two responses are shown in random order. One prompt is one item."""
    rng = random.Random(2026101906)
    counts = collections.Counter()
    for r in map(json.loads, gzip.open(HELPSTEER_DIR / "preference/preference.jsonl.gz")):
        strength = int(r["preference_strength"])
        if r["split"] != "train" or abs(strength) < 2:
            counts["preference_skipped"] += 1
            continue
        pair, better = [r["response_1"], r["response_2"]], int(strength > 0)
        if rng.random() < .5:
            pair, better = pair[::-1], 1 - better
        prompt = _hs_text(r["prompt"])
        state = {"prompt": prompt, "response A": pair[0], "response B": pair[1]}
        question, target = common.choice(state, rng.choice(HS_PREF_ASK), ["response A", "response B"], better)
        counts["preference"] += 1
        yield common.example(HELPSTEER, f"{digest(prompt)}:pref{counts['preference']}", "response-preference", state, question, target, "train")
    by_prompt = collections.defaultdict(list)
    for r in map(json.loads, gzip.open(HELPSTEER_DIR / "train.jsonl.gz")):
        by_prompt[r["prompt"]].append(r)
    for raw_prompt, rows in sorted(by_prompt.items()):
        if len(rows) != 2:
            continue
        prompt = _hs_text(raw_prompt)
        for attribute, asks in HS_ATTRIBUTE.items():
            a, b = rows
            if abs(a[attribute] - b[attribute]) < 2:
                continue
            pair = [a, b] if rng.random() < .5 else [b, a]
            state = {"prompt": prompt, "response A": pair[0]["response"], "response B": pair[1]["response"]}
            question, target = common.choice(state, rng.choice(asks), ["response A", "response B"],
                                             int(pair[1][attribute] > pair[0][attribute]))
            counts[attribute] += 1
            yield common.example(HELPSTEER, f"{digest(prompt)}:{attribute}{counts[attribute]}", "response-attribute", state, question, target, "train")
    stats["counts"] = dict(counts)


# --- FiNER-139: what a number in a filing reports -----------------------------------------------------------
FINER = "finer-train"
FINER_DIR = hub_dir("nlpaueb/finer-139", "080f677a026e")
FINER_ASK = ["In this filing sentence, what does the number marked << >> report?",
             "Which XBRL item is the figure marked << >>?", "What financial quantity is the number in << >>?"]
FINER_EVERY = 4  # about 45k of the train sentences that carry a tag


def finer(stats: dict):
    """Loukas et al. 2022 (CC-BY-SA-4.0). Sentences of 10-K/10-Q filings whose numbers carry the filer's own
    XBRL tags (139 frequent tags), aligned to the text by the authors. Every FINER_EVERY-th train sentence by id;
    one tagged number per sentence; options: the gold tag and 7 others, preferring tags that share a word."""
    rng = random.Random(2026101907)
    info = json.loads((FINER_DIR / "dataset_infos.json").read_text())
    names = next(iter(info.values()))["features"]["ner_tags"]["feature"]["names"]
    tags = sorted({n[2:] for n in names if n != "O"})
    words = {t: set(readable_camel(t).lower().split()) for t in tags}
    counts = collections.Counter()
    with zipfile.ZipFile(FINER_DIR / "finer139.zip") as z, z.open("train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] % FINER_EVERY or len(r["tokens"]) > 250:
                continue
            spans, i = [], 0
            labels = [x if isinstance(x, str) else names[x] for x in r["ner_tags"]]
            while i < len(labels):
                if labels[i].startswith("B-"):
                    j = i + 1
                    while j < len(labels) and labels[j] == "I-" + labels[i][2:]:
                        j += 1
                    spans.append((i, j, labels[i][2:]))
                    i = j
                else:
                    i += 1
            if not spans:
                continue
            start, end, gold = rng.choice(spans)
            near = sorted(t for t in tags if t != gold and words[t] & words[gold])
            rng.shuffle(near)
            rest = [t for t in tags if t != gold and t not in near[:4]]
            options = [gold] + near[:4] + rng.sample(rest, 7 - min(4, len(near)))
            rng.shuffle(options)
            tokens = r["tokens"][:start] + ["<<"] + r["tokens"][start:end] + [">>"] + r["tokens"][end:]
            question, target = common.choice(detokenize(tokens), rng.choice(FINER_ASK), [readable_camel(t).rstrip("0123456789") for t in options],
                                             options.index(gold))
            counts["rows"] += 1
            yield common.example(FINER, str(r["id"]), "financial-number-tagging", detokenize(tokens), question, target, "train")
    stats["counts"] = dict(counts)


# --- Zeshel: entity linking among same-world namesakes ------------------------------------------------------
ZESHEL = "zeshel-train"
ZESHEL_DIR = RAW / "zeshel/zeshel"
ZESHEL_ASK = ["The text between [START_ENT] and [END_ENT] mentions an entity. Which page is it?",
              "Which entity does the marked mention refer to?", "Link the marked mention to its page."]
ZESHEL_WINDOW = 100


def zeshel(stats: dict):
    """Logeswaran et al. 2019 (CC-BY-SA, Fandom wikis; README of github lajanugen/zeshel). Mentions are links
    written by the wikis' editors. Train mentions (8 worlds); context: 100 words on each side, the mention
    marked as in KILT/AIDA. Options: the gold page and 7 pages of the same world whose titles share the most
    words (then characters) with the mention, each with its first 30 words, so the name alone rarely decides.
    One context document is one item."""
    rng = random.Random(2026101908)
    mentions = [json.loads(line) for line in (ZESHEL_DIR / "mentions/train.json").open()]
    counts = collections.Counter()
    for world in sorted({m["corpus"] for m in mentions}):
        docs = {d["document_id"]: d for d in map(json.loads, (ZESHEL_DIR / f"documents/{world}.json").open())}
        by_title = {}
        for d in docs.values():
            by_title.setdefault(d["title"], d)
        titles = sorted(by_title)
        title_words = {t: set(_n(t).split()) for t in titles}
        title_grams = {t: trigrams(t) for t in titles}
        index = collections.defaultdict(list)
        for t in titles:
            for w in title_words[t]:
                index[w].append(t)
        for m in (m for m in mentions if m["corpus"] == world):
            context, gold_doc = docs.get(m["context_document_id"]), docs.get(m["label_document_id"])
            if not context or not gold_doc or by_title.get(gold_doc["title"]) is not gold_doc:
                counts["missing_or_duplicate_title"] += 1
                continue
            gold = gold_doc["title"]
            words = set(_n(m["text"]).split())
            pool = {t for w in words if len(index[w]) <= 5000 for t in index[w]} - {gold}
            grams = trigrams(m["text"])
            ranked = sorted(pool, key=lambda t: (-len(title_words[t] & words), -len(grams & title_grams[t]) / max(1, len(grams | title_grams[t])), t))
            negatives = ranked[:7]
            while len(negatives) < 7:
                t = rng.choice(titles)
                if t != gold and t not in negatives:
                    negatives.append(t)
            options = [gold] + negatives
            rng.shuffle(options)
            described = {t: " ".join(by_title[t]["text"].split()[len(t.split()):][:30]) or None for t in options}
            tokens = context["text"].split(" ")
            s, e = m["start_index"], m["end_index"]
            text = " ".join(tokens[max(0, s - ZESHEL_WINDOW):s] + ["[START_ENT]"] + tokens[s:e + 1] + ["[END_ENT]"] + tokens[e + 1:e + 1 + ZESHEL_WINDOW])
            question, target = common.choice(text, rng.choice(ZESHEL_ASK), described, options.index(gold))
            counts[m["category"]] += 1
            yield common.example(ZESHEL, f"{world}:{m['context_document_id']}:{m['mention_id']}", "entity-linking", text, question, target, "train")
    stats["counts"] = dict(counts)


def trigrams(text: str) -> set[str]:
    t = f"  {text.lower()}  "
    return {t[i:i + 3] for i in range(len(t) - 2)}


# --- Few-NERD: what kind of entity a mention is, in context -------------------------------------------------
FEWNERD = "fewnerd-train"
FEWNERD_DIR = hub_dir("DFKI-SLT/few-nerd", "205f3e9c9f35")
FEWNERD_ASK = ["What kind of entity is the mention marked << >> in this sentence?", "In this sentence, the marked name is a ...",
               "Which type fits the marked mention here?"]
FEWNERD_WORDS = {"broadcastprogram": "broadcast program", "writtenart": "written work", "sportsfacility": "sports facility",
                 "attack/battle/war/militaryconflict": "attack, battle, war or military conflict", "sportsevent": "sports event",
                 "GPE": "geopolitical entity", "bodiesofwater": "body of water", "road/railway/highway/transit": "road, railway or transit line",
                 "government/governmentagency": "government or government agency", "media/newspaper": "media or newspaper",
                 "politicalparty": "political party", "showorganization": "show organization", "sportsleague": "sports league",
                 "sportsteam": "sports team", "astronomything": "astronomical object", "biologything": "biological thing",
                 "chemicalthing": "chemical", "educationaldegree": "educational degree", "livingthing": "living thing",
                 "artist/author": "artist or author"}
FEWNERD_ROWS = 40_000


def fewnerd(stats: dict):
    """Ding et al. 2021 (CC-BY-SA-4.0). Wikipedia sentences, 66 fine entity types annotated by trained
    annotators (supervised train split). One marked mention per sentence; options: the gold type, up to 3
    types of the same coarse class, and others, 8 in all. One sentence is one item."""
    rng = random.Random(2026101909)
    readme = (FEWNERD_DIR / "README.md").read_text()
    section = readme[readme.index("fine_ner_tags"):]
    names = []
    for m in re.finditer(r"'(\d+)': (\S+)", section):
        if int(m.group(1)) != len(names):
            break
        names.append(m.group(2))
    types = names[1:]
    shown = {t: f"{FEWNERD_WORDS.get(t.split('-', 1)[1], t.split('-', 1)[1])} ({t.split('-', 1)[0]})" for t in types}
    rows = pq.read_table(FEWNERD_DIR / "supervised/train-00000-of-00001.parquet").to_pylist()
    rng.shuffle(rows)
    counts = collections.Counter()
    for r in rows:
        if counts["rows"] >= FEWNERD_ROWS:
            break
        tags = list(r["fine_ner_tags"])
        spans, i = [], 0
        while i < len(tags):
            if tags[i]:
                j = i
                while j < len(tags) and tags[j] == tags[i]:
                    j += 1
                spans.append((i, j, names[tags[i]]))
                i = j
            else:
                i += 1
        if not spans:
            continue
        start, end, gold = rng.choice(spans)
        coarse = gold.split("-")[0]
        siblings = [t for t in types if t != gold and t.startswith(coarse + "-")]
        others = [t for t in types if not t.startswith(coarse + "-")]
        options = [gold] + rng.sample(siblings, min(3, len(siblings)))
        options += rng.sample(others, 8 - len(options))
        rng.shuffle(options)
        tokens = list(r["tokens"])
        text = detokenize(tokens[:start] + ["<<"] + tokens[start:end] + [">>"] + tokens[end:])
        question, target = common.choice(text, rng.choice(FEWNERD_ASK), [shown[t] for t in options], options.index(gold))
        counts["rows"] += 1
        yield common.example(FEWNERD, str(r["id"]), "entity-typing", text, question, target, "train")
    stats["counts"] = dict(counts)


# --- DROP: counting, comparing and adding over a passage ----------------------------------------------------
DROP = "drop-train"
DROP_DIR = hub_dir("ucinlp/drop", "95cda593fae7")


def drop(stats: dict):
    """Dua et al. 2019 (CC-BY-SA-4.0). Crowd questions needing discrete reasoning over Wikipedia passages
    (sports, history). Single-answer number and span questions; options: the gold and 3 distractors - other
    numbers in the passage or near the gold for numbers, other span answers on the same passage or capitalised
    phrases for spans. One passage is one item."""
    rng = random.Random(2026101910)
    rows = pq.read_table(DROP_DIR / "data/train-00000-of-00001.parquet").to_pylist()
    spans_of = collections.defaultdict(set)
    for r in rows:
        a = r["answers_spans"]
        if len(a["spans"]) == 1 and a["types"][0] == "span":
            spans_of[r["section_id"]].add(a["spans"][0])
    counts = collections.Counter()
    for r in rows:
        a = r["answers_spans"]
        if len(a["spans"]) != 1 or a["types"][0] not in ("number", "span"):
            counts["skipped_type"] += 1
            continue
        gold = a["spans"][0].strip()
        if a["types"][0] == "number":
            try:
                value = float(gold.replace(",", ""))
            except ValueError:
                counts["bad_number"] += 1
                continue
            numbers = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", r["passage"])}
            near = {value + d for d in (-2, -1, 1, 2, 10)} | ({value * 2, value / 2} if value else set())
            pool = sorted(v for v in numbers | near if v >= 0 and abs(v - value) > 1e-9)
            from_text = [v for v in pool if v in numbers]
            rng.shuffle(from_text)
            picked = from_text[:2]
            rest = [v for v in pool if v not in picked]
            picked += rng.sample(rest, min(3 - len(picked), len(rest)))
            fmt_number = lambda v: str(int(v)) if float(v).is_integer() else f"{v:g}"
            distractors = [fmt_number(v) for v in picked]
        else:
            g = _n(gold)
            pool = [s for s in sorted(spans_of[r["section_id"]]) if _n(s) and g not in _n(s) and _n(s) not in g]
            caps = sorted({m.group(0) for m in re.finditer(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*", r["passage"])
                           if _n(m.group(0)) and g not in _n(m.group(0)) and _n(m.group(0)) not in g})
            rng.shuffle(pool)
            rng.shuffle(caps)
            distractors = list(dict.fromkeys(pool + caps))[:3]
        if len(distractors) < 2 or _n(gold) in {_n(d) for d in distractors}:
            counts["few_distractors"] += 1
            continue
        options = [gold] + distractors
        rng.shuffle(options)
        question, target = common.choice({"passage": r["passage"]}, r["question"], options, options.index(gold))
        counts[a["types"][0]] += 1
        yield common.example(DROP, f"{r['section_id']}:{r['query_id']}", "discrete-reading", {"passage": r["passage"]},
                             question, target, "train")
    stats["counts"] = dict(counts)


# --- Taskmaster-1: what the customer finally agreed to ------------------------------------------------------
TASKMASTER = "taskmaster-train"
TASKMASTER_DIR = RAW / "google-research-datasets__Taskmaster@d92cb6af3005/TM-1-2019"
TM_ASK = ["In this conversation, which {slot} did the customer settle on?", "What {slot} did the customer finally agree to?"]


def tm_slot(name: str) -> str:
    domain, *rest = name.split(".")
    if rest and rest[0] == "num":
        phrase = "number of " + " ".join(rest[1:])
    else:
        phrase = " ".join(reversed(rest))
    return f"{phrase} ({domain.replace('_', ' ')})"


def taskmaster(stats: dict):
    """Byrne et al. 2019, Taskmaster-1 (CC-BY-4.0, github google-research-datasets/Taskmaster). Spoken (WOZ)
    and written self-dialogs in six domains; annotators marked values and which the customer accepted
    (".accept") or turned down (".reject"). For a slot with one accepted value and 1-7 rejected ones, the question
    asks which one the customer settled on. Official train split; one conversation is one item."""
    rng = random.Random(2026101911)
    train = {line.split(",")[0].strip() for line in (TASKMASTER_DIR / "train-dev-test/train.csv").read_text().splitlines() if line.strip()}
    counts = collections.Counter()
    for file in ("self-dialogs.json", "woz-dialogs.json"):
        for conv in json.loads((TASKMASTER_DIR / file).read_text()):
            if conv["conversation_id"] not in train:
                counts["not_train"] += 1
                continue
            accepted, rejected = collections.defaultdict(dict), collections.defaultdict(dict)
            for u in conv["utterances"]:
                for seg in u.get("segments") or []:
                    for ann in seg["annotations"]:
                        base = re.sub(r"\.(accept|reject)$", "", ann["name"])
                        if ann["name"].endswith(".accept"):
                            accepted[base].setdefault(_n(seg["text"]), seg["text"].strip())
                        elif ann["name"].endswith(".reject"):
                            rejected[base].setdefault(_n(seg["text"]), seg["text"].strip())
            text = "\n".join(f"{u['speaker'].capitalize()}: {u['text']}" for u in conv["utterances"])
            for slot, chosen in sorted(accepted.items()):
                # Only values the customer explicitly turned down are wrong options: other mentions of the slot
                # are often the same thing in other words ("my home" and the home address).
                others = {k: v for k, v in rejected[slot].items() if k and k not in chosen}
                if len(chosen) != 1 or not 1 <= len(others) <= 7:
                    continue
                options = list(chosen.values()) + list(others.values())
                rng.shuffle(options)
                gold = [_n(o) for o in options].index(next(iter(chosen)))
                question, target = common.choice(text, rng.choice(TM_ASK).format(slot=tm_slot(slot)), options, gold)
                counts[slot.split(".")[0]] += 1
                yield common.example(TASKMASTER, f"{conv['conversation_id']}:{slot}", "dialogue-state", text, question, target, "train")
    stats["counts"] = dict(counts)


BUILDERS = [
    (TABFACT, tabfact, {"upstream": "github wenhuchen/Table-Fact-Checking@2ab782ba collected_data r1+r2, train_id tables",
                        "licence": "MIT (repository LICENSE); tables from Wikipedia (CC-BY-SA)", "label_origin": "human"}),
    (CONDAQA, condaqa, {"upstream": "lasha-nlp/CONDAQA train+dev", "licence": "Apache-2.0", "label_origin": "human"}),
    (ROPES, ropes, {"upstream": "allenai/ropes train+validation", "licence": "CC-BY-4.0",
                    "label_origin": "human answers; options programmatic from the question/situation"}),
    (FOLIO, folio, {"upstream": "github Yale-LILY/FOLIO@5d7bb84c data/v0.0 train+validation", "licence": "CC-BY-SA-4.0",
                    "label_origin": "human (experts), checked by an FOL prover"}),
    (ABCD, abcd, {"upstream": "github asappresearch/abcd@6b8700ce abcd_v1.1 train", "licence": "MIT", "label_origin": "human (agents' clicks)"}),
    (HELPSTEER, helpsteer2, {"upstream": "nvidia/HelpSteer2 preference (train) + train attribute ratings", "licence": "CC-BY-4.0",
                             "label_origin": "human ratings of model-written responses"}),
    (FINER, finer, {"upstream": "nlpaueb/finer-139 finer139.zip train (every 4th sentence)", "licence": "CC-BY-SA-4.0",
                    "label_origin": "human (filers' XBRL tags), aligned to text by the authors"}),
    (ZESHEL, zeshel, {"upstream": "github lajanugen/zeshel (Google Drive archive) mentions/train", "licence": "CC-BY-SA (Fandom wikis)",
                      "label_origin": "human (wiki editors' links); options programmatic"}),
    (FEWNERD, fewnerd, {"upstream": "DFKI-SLT/few-nerd supervised train", "licence": "CC-BY-SA-4.0", "label_origin": "human"}),
    (DROP, drop, {"upstream": "ucinlp/drop train", "licence": "CC-BY-SA-4.0", "label_origin": "human answers; distractors programmatic"}),
    (TASKMASTER, taskmaster, {"upstream": "github google-research-datasets/Taskmaster@d92cb6af TM-1-2019 train", "licence": "CC-BY-4.0",
                              "label_origin": "human (annotated accepted values)"}),
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
            record = fetch_one(name)
            print(json.dumps(record), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
    else:
        wanted = set(args.names)
        for name, build, meta in BUILDERS:
            if not wanted or name in wanted:
                common.write(name, build(meta), meta)
