"""Phishing / scam email detection from two permissive corpora (not PhishNChips), as yes/no and 2-way choices.

Upstreams (cap 10,000, 50/50 phishing vs legitimate):
  - Phishing Email Curated Datasets (Champa, Rabbi & Zibran 2024, Zenodo 10.5281/zenodo.8339691,
    CC-BY-4.0) via its byte-identical Hub mirror kudzaiprichard/aura-phishing-email-corpus: CEAS-08,
    SpamAssassin (both classes from one stream), Nazario_5 (Nazario phishing + Enron ham) and
    Nigerian_5 (419 fraud + Enron ham). Structured sender / subject / body, like the index state.
  - Dizzzy0x00/LLMGen-Phishing-Email-Dataset (Apache-2.0), English GPT-written phishing and
    legitimate business emails, closer in style to the index's LLM-written workplace emails.
Not used: PhishNChips (its core_emails.csv is all 2,000 index items; its other files come from the
same lab and cross_domain_legitimate_v5 likely repeats core rows), zefang-liu (LGPL-3.0) and
ealvaradob/phishing-dataset (Apache-2.0, but unstructured lower-cased Ling-Spam-style spam vs ham).

Overlap: neither corpus is in the index or JEVal. Every email carrying a URL that equals a core
`url_raw`, or whose host is the host of a core phishing URL, is dropped (counted in the manifest),
and the contamination filter checks the text. Bodies are truncated to 2,000 characters; the Enron
ham shared by Nazario_5 and Nigerian_5 is deduplicated by body.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from urllib.parse import urlsplit

import pandas as pd

from qwen_decision.v3.sources import common

NAME = "phishing"
ZENODO = ("kudzaiprichard/aura-phishing-email-corpus", "539567cee21d03c83bd2fb1b142c18a80904eb72")
LLMGEN = ("Dizzzy0x00/LLMGen-Phishing-Email-Dataset", "d717e5457b71cb44dd038dafe6a81cea5e0badab")
CORE = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/phishnchips/core_emails.csv"
CAP = 10_000
BODY_CHARS = 2_000
# (file, label) -> how many to take; LLMGen takes all 1,269 legitimate and as many phishing.
QUOTA = {("Nazario_5.csv", 1): 1300, ("Nigerian_5.csv", 1): 700, ("CEAS_08.csv", 1): 1000, ("SpamAssasin.csv", 1): 731,
         ("Nazario_5.csv", 0): 600, ("Nigerian_5.csv", 0): 400, ("CEAS_08.csv", 0): 1531, ("SpamAssasin.csv", 0): 1200}
URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
SENDER = re.compile(r"^\s*\"?(?P<name>[^\"<]*?)\"?\s*<(?P<addr>[^>]+)>\s*$")
LLM_HEAD = re.compile(r"Subject:\s*(?P<subject>.*?)\s+(?:From|Sender):\s*(?P<sender>(?:[^@\n]{0,60}?<)?[\w.+-]+@[\w.-]+>?)\s*"
                      r"(?:To:\s*\S+@\S+\s*)?(?P<body>.*)", re.S)
CHOICE = [
    "Here is an email the user received. Is it a scam or phishing attempt, or genuine mail?",
    "Should the user trust this email? Decide whether it is malicious or legitimate.",
    "Classify the email: an attempt to deceive the recipient, or ordinary correspondence?",
    "An email assistant must decide how to treat this message. Is it fraudulent or benign?",
    "Look at the sender, subject and content. Is this email a scam or legitimate?",
]
OPTIONS = [
    {"phishing / scam": "Tries to trick the recipient (fake login, fraud, malicious link or attachment, unsolicited spam). Do not engage.",
     "legitimate": "Ordinary, genuine correspondence. Safe to read and act on."},
    {"malicious": None, "benign": None},
    {"do not trust": "The message is fraudulent, a phishing lure or spam.", "trust": "The message is a normal, genuine email."},
]
IS_BAD = [
    "This email is a phishing, scam or spam message.",
    "Is this message trying to deceive the recipient (phishing, fraud or spam)?",
    "The recipient should treat this email as malicious and not click or reply.",
]
IS_GOOD = [
    "This is a genuine email that the recipient can safely act on.",
    "Is this message ordinary, legitimate correspondence?",
]


def core_blocklist() -> tuple[set[str], set[str]]:
    urls, hosts = set(), set()
    with CORE.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            urls.add(r["url_raw"].strip().rstrip("/").lower())
            if int(r["phish_label"]):
                hosts.add((urlsplit(r["url_raw"].strip()).hostname or "").lower())
    return urls, hosts - {""}


def links(body: str) -> list[str]:
    return list(dict.fromkeys(u.rstrip(".,;:") for u in URL.findall(body)))


def clip(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text if len(text) <= BODY_CHARS else text[:BODY_CHARS].rsplit(" ", 1)[0] + " …"


def zenodo_rows(rng: random.Random) -> list[dict]:
    local, _ = common.fetch(ZENODO[0], ["README.md"] + sorted({f for f, _ in QUOTA}), revision=ZENODO[1])
    out = []
    for file in sorted({f for f, _ in QUOTA}):
        df = pd.read_csv(local / file).fillna("")
        for label in (0, 1):
            rows = df[df["label"] == label].to_dict("records")
            rng.shuffle(rows)
            for r in rows[:QUOTA[(file, label)] * 2]:  # extra room for empties, duplicates and blocklist drops
                if len(str(r["body"]).strip()) < 40:
                    continue
                m = SENDER.match(str(r["sender"]))
                email = {"sender": m["name"].strip() if m else "", "from": m["addr"].strip() if m else str(r["sender"]).strip(),
                         "subject": str(r["subject"]).strip(), "body": clip(str(r["body"]))}
                out.append({"source": f"zenodo:{file}", "quota": (file, label), "id": f"{label}:{hashlib.sha1(str(r['body']).encode()).hexdigest()[:12]}",
                            "label": label, "email": email, "urls": links(str(r["body"]))})
    return out


def llmgen_rows() -> list[dict]:
    local, _ = common.fetch(LLMGEN[0], ["README.md", "GPT_Phishing_Email_dataset.csv"], revision=LLMGEN[1])
    out = []
    for i, r in enumerate(pd.read_csv(local / "GPT_Phishing_Email_dataset.csv").to_dict("records")):
        text = str(r["content"]).replace("**", "")
        m = LLM_HEAD.match(text.strip())
        if m:
            s = SENDER.match(m["sender"])
            email = {"sender": s["name"].strip() if s else "", "from": s["addr"] if s else m["sender"].strip("<>"),
                     "subject": m["subject"].strip(), "body": clip(m["body"])}
        else:
            email = {"body": clip(text)}
        out.append({"source": "llmgen:gpt", "quota": ("llmgen", int(r["label"])), "id": f"gpt:{i}", "label": int(r["label"]),
                    "email": email, "urls": links(text)})
    return out


def select(rng: random.Random, stats: dict) -> list[dict]:
    core_urls, core_hosts = core_blocklist()
    llm = llmgen_rows()
    legit = sum(1 for r in llm if r["label"] == 0)
    quota = {**QUOTA, ("llmgen", 0): legit, ("llmgen", 1): legit}
    taken, seen, out = {k: 0 for k in quota}, set(), []
    for r in zenodo_rows(rng) + rng.sample(llm, len(llm)):
        body_key = hashlib.sha1(" ".join(r["email"]["body"].lower().split()).encode()).hexdigest()
        if taken[r["quota"]] >= quota[r["quota"]] or body_key in seen:
            continue
        if any(u.rstrip("/").lower() in core_urls or (urlsplit(u).hostname or "").lower() in core_hosts for u in r["urls"]):
            stats["dropped_core_url_or_host"] += 1
            continue
        seen.add(body_key)
        taken[r["quota"]] += 1
        out.append(r)
    stats["taken"] = {f"{f}:{'phishing' if lab else 'legitimate'}": n for (f, lab), n in taken.items()}
    return out[:CAP]


def build(stats: dict):
    rng = random.Random(56)
    for r in select(rng, stats):
        email = dict(r["email"])
        if r["urls"] and rng.random() < .6:
            email["link_url"] = r["urls"][0]
        state = email if rng.random() < .8 else "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in email.items() if v)
        bad = r["label"] == 1
        if rng.random() < .5:
            options = rng.choice(OPTIONS)
            keys = list(options) if rng.random() < .5 else list(options)[::-1]
            gold = list(options)[0 if bad else 1]
            question, target = common.choice(state, common.instruction(rng, CHOICE), {k: options[k] for k in keys}, keys.index(gold))
        elif rng.random() < .65:
            question, target = common.noul(rng.choice(IS_BAD), bad)
        else:
            question, target = common.noul(rng.choice(IS_GOOD), not bad)
        yield common.example(NAME, f"{r['source']}:{r['id']}", "phishing", state, question, target, "train")


if __name__ == "__main__":
    stats = {"dropped_core_url_or_host": 0}
    rows = list(build(stats))
    common.write(NAME, rows, {
        "upstream": {"zenodo-8339691 (via mirror)": ZENODO[0], "llmgen": LLMGEN[0]},
        "revision": {ZENODO[0]: ZENODO[1], LLMGEN[0]: LLMGEN[1]},
        "split": "all (no official splits); class-balanced sample", "cap": CAP,
        "licence": {ZENODO[0]: "CC-BY-4.0 (Zenodo record 8339691)", LLMGEN[0]: "Apache-2.0"}, **stats})
