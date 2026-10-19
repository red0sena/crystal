"""Phase-7 new domains (2026-10-19): safety of a request, community ethical judgment, vulnerability triage.

  python -m qwen_decision.v3.sources.phase7_domains --fetch      # download the pinned upstream files
  python -m qwen_decision.v3.sources.phase7_domains [name ...]   # build data/phase2/<name>.jsonl

Kinds of decisions none of our sources had, for the private new-domain part of the index. Vetted on 2026-10-19:
- Aegis 2.0 / Nemotron Content Safety V2 (nvidia/Aegis-AI-Content-Safety-Dataset-2.0, CC-BY-4.0): user prompts
  (from Anthropic HH-RLHF red-teaming, DAN jailbreaks and AI-assisted red-teaming) with a safe/unsafe label and
  violated categories. Only prompt labels are read: prompt_label_source is human for every row, while response
  labels partly come from an LLM jury (left out, with the responses). Train + validation; test never read.
  Hazard categories are used only from prompt-only rows, where they describe the prompt, and only when a
  prompt has exactly one category (not "Needs Caution" or "Other").
- Scruples v1.0 (Lourie et al. 2021; allenai/scruples, Apache-2.0): Reddit "Am I the asshole?" anecdotes with the
  community's verdict votes (author / other / everybody / nobody / more info), and dilemmas, two actions of
  which 5 crowd annotators picked the less ethical one. Train + dev; test never read. Rows store the vote
  shares (target_kind vote_distribution) but training targets their plurality (mix.gold_set), so only clear
  pluralities are kept: anecdotes need 3+ votes, at most 600 words and a unique plurality of 60%+, dilemmas
  4 of 5 votes.
- NVD CVE feeds (NIST; no licensing restrictions; CVE text under MITRE's CVE terms) with the CWE list
  (MITRE): the description of a vulnerability, and the NVD analyst's CVSS 3.1 base metrics and CWE (vendor
  and CNA scores are not used). One question per CVE: base severity (ordered), privileges required
  (ordered), attack vector, user interaction (yes/no) or the CWE among frequent ones. A CVE whose text carries a
  CVSS vector or score is left out (it gives the answers away), and a severity question is never asked of a
  text that states a severity (VulDB's "classified as critical"). Near-duplicate descriptions (vendor templates,
  versions) are joined for the dev split by mix.NEAR_TEXT. The feeds change daily:
  each year's file is checked against the sha256 in the .meta file fetched with it, and the manifest
  records them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import random
import re
import subprocess
import zipfile

from qwen_decision.v3.sources import common
from qwen_decision.v3.sources.phase4 import digest
from qwen_decision.v3.sources.phase5 import hub_dir

RAW = common.RAW / "phase7"


def majority(labels: list) -> object | None:
    top = collections.Counter(labels).most_common()
    return None if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


# --- Aegis 2.0: is a user request unsafe? ---------------------------------------------------------------
AEGIS = "aegis-prompts"
AEGIS_REPO, AEGIS_REVISION = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
AEGIS_UNSAFE = [
    "Is this user request harmful or unsafe for an assistant to help with?",
    "Should a careful assistant treat this message as unsafe?",
    "Does this prompt ask for, or contain, content that breaks a safety policy?",
    "Is this a harmful request?",
]
AEGIS_SAFE = ["Is this request safe for an assistant to answer?", "Can this message be handled without any safety concern?"]
AEGIS_CATEGORY = ["Which kind of safety risk does this request fall under?",
                  "Classify the unsafe request by its main hazard category.",
                  "What is the main policy area this message violates?"]
AEGIS_VAGUE = {"Needs Caution", "Other"}
AEGIS_STATE_KEYS = ["user_message", "prompt", "request"]


def prompt_only(r: dict) -> bool:
    return r.get("response") in (None, "", "None")


def aegis(stats: dict):
    rng = random.Random(2026101941)
    # Prompts are pooled by the same normalization the row id uses (case and punctuation ignored): the review of
    # 2026-10-19 found 294 duplicate ids and 34 ids with both labels when labels were pooled by raw text.
    labels, categories, splits = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(set)
    texts = collections.defaultdict(collections.Counter)
    for split in ("train", "validation"):
        for r in json.loads((hub_dir(AEGIS_REPO, AEGIS_REVISION) / f"{split}.json").read_text()):
            prompt = " ".join(str(r["prompt"]).split())
            if not prompt or prompt.upper() == "REDACTED":  # Suicide Detection rows, not distributed upstream
                stats["redacted"] += 1
                continue
            assert r["prompt_label_source"] == "human", r["id"]
            key = digest(prompt)
            texts[key][prompt] += 1
            labels[key].append(r["prompt_label"])
            splits[key].add(split)
            if prompt_only(r) and r["prompt_label"] == "unsafe":
                categories[key].append(tuple(sorted(c.strip() for c in (r["violated_categories"] or "").split(",") if c.strip())))
    taxonomy = sorted({c for cats in categories.values() for cs in cats for c in cs} - AEGIS_VAGUE)
    for key, found in labels.items():
        prompt = min(texts[key].items(), key=lambda kv: (-kv[1], kv[0]))[0]  # the most common spelling
        stats["variant_spellings_merged"] += len(texts[key]) - 1
        label = majority(found)
        if label is None:
            stats["label_ties_dropped"] += 1
            continue
        unsafe = label == "unsafe"
        r = rng.random()
        state = prompt if r < .6 else {rng.choice(AEGIS_STATE_KEYS): prompt}
        split = "+".join(sorted(splits[key]))
        flipped = rng.random() < .25
        question, target = common.noul(rng.choice(AEGIS_SAFE if flipped else AEGIS_UNSAFE), unsafe != flipped)
        stats["noul"][f"{label}{':flipped' if flipped else ''}"] += 1
        yield common.example(AEGIS, f"{key}:noul", "safety", state, question, target, split)
        cats = set(categories.get(key, []))
        if unsafe and len(cats) == 1 and len(next(iter(cats))) == 1 and next(iter(cats))[0] in taxonomy:
            gold = next(iter(cats))[0]
            options = [gold] + rng.sample([c for c in taxonomy if c != gold], rng.randint(4, 7))
            rng.shuffle(options)
            question, target = common.choice(state, rng.choice(AEGIS_CATEGORY), options, options.index(gold))
            stats["category"][gold] += 1
            yield common.example(AEGIS, f"{key}:category", "safety-category", state, question, target, split)


# --- Scruples: who is in the wrong, and which action is less ethical ------------------------------------
SCRUPLES = "scruples-train"
SCRUPLES_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/{}.tar.gz"
SCRUPLES_MIN_VOTES, SCRUPLES_MAX_WORDS = 3, 600
# Training targets the plurality of the votes (train.gold_targets via mix.gold_set: vote_distribution), not the vote
# shares, so a split vote would teach a coin flip as a hard label. Kept: a unique plurality with at least
# SCRUPLES_MIN_SHARE of the anecdote votes; dilemmas (5 annotators) need 4 of 5 (DILEMMA_MIN_SHARE).
SCRUPLES_MIN_SHARE, DILEMMA_MIN_SHARE = .6, .8


def clear_plurality(votes: list[float], min_share: float) -> bool:
    top = sorted(votes, reverse=True)
    return top[0] > top[1] and top[0] >= min_share * sum(votes)
# AITA verdicts in upstream order; descriptions are what a reader sees
VERDICTS = {"AUTHOR": "the author (the person who wrote the post) is in the wrong",
            "OTHER": "the other party is in the wrong", "EVERYBODY": "everyone involved is in the wrong",
            "NOBODY": "nobody is in the wrong", "INFO": "more information is needed to judge"}
VERDICT_KEYS = [{"AUTHOR": "author", "OTHER": "other party", "EVERYBODY": "everyone", "NOBODY": "no one", "INFO": "need more info"},
                {"AUTHOR": "the writer", "OTHER": "the others", "EVERYBODY": "all of them", "NOBODY": "nobody", "INFO": "unclear"}]
WHO_WRONG = ["Who is in the wrong in this story?", "Judging the situation the way most readers would, who behaved badly here?",
             "The author asks whether they were in the wrong. What would the community's verdict be?",
             "Read the post and decide who is at fault."]
AUTHOR_WRONG = ["Was the author in the wrong here?", "Would most readers say the person who wrote this acted wrongly?",
                "Is the writer at fault in this situation?"]
LESS_ETHICAL = ["Which of these two actions is less ethical?", "Most people would find one of these actions worse. Which one?",
                "Which action would people judge more harshly?"]
MORE_ETHICAL = ["Which of these two actions is more acceptable?", "Which action would most people find less objectionable?"]
SCRUPLES_STATE_KEYS = [("title", "post"), ("headline", "story"), ("question", "situation")]


def scruples_rows(name: str):
    import tarfile
    path = RAW / "scruples" / f"{name}.tar.gz"
    with tarfile.open(path) as t:
        for split in ("train", "dev"):
            member = f"{name}/{split}.scruples-{name}.jsonl"
            for line in t.extractfile(member):
                yield split, json.loads(line)


def scruples(stats: dict):
    rng = random.Random(2026101942)
    for split, r in scruples_rows("anecdotes"):
        votes = [r["label_scores"][k] for k in VERDICTS]
        if sum(votes) < SCRUPLES_MIN_VOTES or len(r["text"].split()) > SCRUPLES_MAX_WORDS:
            stats["anecdotes_dropped"] += 1
            continue
        binary = r["binarized_label_scores"]
        verdict = rng.random() < .6
        if not clear_plurality(votes if verdict else [binary["WRONG"], binary["RIGHT"]], SCRUPLES_MIN_SHARE):
            stats["anecdotes_split_vote_dropped"] += 1
            continue
        a, b = rng.choice(SCRUPLES_STATE_KEYS)
        state = {a: " ".join(r["title"].split()), b: r["text"].strip()}
        if verdict:
            keys = rng.choice(VERDICT_KEYS)
            order = rng.sample(list(VERDICTS), len(VERDICTS))
            criteria = {keys[v]: VERDICTS[v] for v in order}
            question, target = common.choice(state, rng.choice(WHO_WRONG), criteria,
                                              target=[r["label_scores"][v] / sum(votes) for v in order])
            stats["anecdotes"]["verdict"] += 1
            kind = "verdict"
        else:
            question, target = common.noul(rng.choice(AUTHOR_WRONG), binary["WRONG"] / (binary["WRONG"] + binary["RIGHT"]))
            stats["anecdotes"]["author_wrong"] += 1
            kind = "author-wrong"
        yield common.example(SCRUPLES, f"{r['post_id']}:{kind}", "community-judgment", state, question, target, split)
    for split, r in scruples_rows("dilemmas"):
        actions = [" ".join(a["description"].split()) for a in r["actions"]]
        votes = r["gold_annotations"]
        if len(actions) != 2 or actions[0].lower() == actions[1].lower() or sum(votes) == 0:
            stats["dilemmas_dropped"] += 1
            continue
        if not clear_plurality(votes, DILEMMA_MIN_SHARE):
            stats["dilemmas_split_vote_dropped"] += 1
            continue
        more = rng.random() < .3  # ask for the more acceptable action: the target flips
        dist = [v / sum(votes) for v in votes]
        dist = [1 - x for x in dist] if more else dist
        order = rng.sample(range(2), 2)
        question, target = common.choice("", rng.choice(MORE_ETHICAL if more else LESS_ETHICAL),
                                         [actions[i] for i in order], target=[dist[i] for i in order])
        stats["dilemmas"]["more_acceptable" if more else "less_ethical"] += 1
        yield common.example(SCRUPLES, f"d-{r['id']}:dilemma", "ethical-dilemma", "", question, target, split)


# --- NVD: vulnerability triage from a CVE description ---------------------------------------------------
NVD = "nvd-cve"
NVD_YEARS = range(2002, 2027)
NVD_FEED = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{}.{}"
CWE_URL = "https://cwe.mitre.org/data/csv/1000.csv.zip"
NVD_CVES = 40000  # CVEs sampled (one question each) from those with an NVD-analyst CVSS 3.1 score
NVD_STATUS = {"Analyzed", "Modified"}
SEVERITY = [("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("critical", "CRITICAL")]
PRIVILEGES = [("none", "NONE", "no account or privileges needed"), ("low", "LOW", "an ordinary user account"),
              ("high", "HIGH", "administrative or other high privileges")]
VECTORS = {"NETWORK": "remotely over a network", "ADJACENT_NETWORK": "from an adjacent network (same LAN, Bluetooth, Wi-Fi)",
           "LOCAL": "locally, with access to the system or by tricking a local user", "PHYSICAL": "only with physical access to the device"}
CVSS_TEXT = re.compile(r"\bCVSS|\bAV:[NALP]\b|\bPR:[NLH]\b|\bUI:[NR]\b|base score", re.I)
SEVERITY_HINT = re.compile(r"classified as (critical|problematic|important|low|medium|high)|\bcvss\b|"
                           r"\b(critical|high|medium|moderate|low)[- ]severity\b|severity (rating|score|of)", re.I)
NVD_ASK = {
    "severity": ["How severe is this vulnerability (CVSS 3.1 base severity)?", "Rate the severity of the vulnerability described.",
                 "What severity would an analyst give this vulnerability?"],
    "vector": ["From where can an attacker exploit this vulnerability?", "What is the attack vector of this vulnerability?",
               "How does an attacker need to reach the target to exploit this?"],
    "interaction": ["Does exploiting this vulnerability require a user to take some action (such as opening a file or clicking a link)?",
                    "Is user interaction needed for this attack to succeed?"],
    "privileges": ["What level of privileges does an attacker need before exploiting this?",
                   "Which privileges must the attacker already have?"],
    "cwe": ["Which weakness type (CWE) best describes this vulnerability?", "Classify the root cause of this vulnerability.",
            "What kind of software weakness is described here?"],
}
NVD_STATE_KEYS = ["description", "advisory", "vulnerability"]


def nvd_feed(year: int) -> tuple[list, str]:
    """One year of the NVD 2.0 feed, checked against the sha256 in the .meta file fetched with it."""
    raw = gzip.open(RAW / f"nvd/nvdcve-2.0-{year}.json.gz").read()
    meta = (RAW / f"nvd/nvdcve-2.0-{year}.meta").read_text()
    sha = re.search(r"sha256:(\w+)", meta).group(1).lower()
    assert hashlib.sha256(raw).hexdigest() == sha, year
    return json.loads(raw)["vulnerabilities"], sha


def cwe_names() -> dict[str, str]:
    with zipfile.ZipFile(RAW / "cwe/1000.csv.zip") as z:
        text = z.read(z.namelist()[0]).decode("utf-8", "replace")
    return {f"CWE-{r['CWE-ID']}": r["Name"] for r in csv.DictReader(io.StringIO(text))}


def nvd_primary(cve: dict) -> tuple[dict | None, str | None]:
    """The NVD analyst's CVSS 3.1 data and single CWE, when present (vendor/CNA scores are not used)."""
    metric = next((m["cvssData"] for m in cve.get("metrics", {}).get("cvssMetricV31", [])
                   if m["source"] == "nvd@nist.gov" and m["type"] == "Primary"), None)
    cwes = [d["value"] for w in cve.get("weaknesses", []) if w["source"] == "nvd@nist.gov" and w["type"] == "Primary"
            for d in w["description"] if d["value"].startswith("CWE-")]
    return metric, cwes[0] if len(cwes) == 1 else None


def nvd(stats: dict):
    rng = random.Random(2026101943)
    names = cwe_names()
    pool = []
    for year in NVD_YEARS:
        rows, sha = nvd_feed(year)
        stats["feeds"][str(year)] = sha
        for v in rows:
            cve = v["cve"]
            text = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
            text = " ".join(text.split())
            metric, cwe = nvd_primary(cve)
            if cve.get("vulnStatus") not in NVD_STATUS or metric is None or len(text.split()) < 8:
                continue
            if CVSS_TEXT.search(text):  # a CVSS vector in the text gives away vector, privileges and interaction
                stats["cvss_in_text_dropped"] += 1
                continue
            pool.append((cve["id"], text, metric, cwe if cwe in names else None))
    stats["eligible_cves"] = len(pool)
    common_cwes = [c for c, _ in collections.Counter(c for *_, c in pool if c).most_common(80)]
    rng.shuffle(pool)
    for cve_id, text, m, cwe in pool[:NVD_CVES]:
        kinds = ["severity", "vector", "interaction", "privileges"] + (["cwe"] * 2 if cwe else [])
        kind = rng.choice(kinds)
        if kind == "severity" and SEVERITY_HINT.search(text):
            kind = "vector"  # the text states a severity (VulDB's "classified as critical"): ask something else
        state = text if rng.random() < .5 else {rng.choice(NVD_STATE_KEYS): text}
        ask = rng.choice(NVD_ASK[kind])
        ordered = False
        if kind == "severity":
            question, target = common.choice(state, ask, [k for k, _ in SEVERITY], [v for _, v in SEVERITY].index(m["baseSeverity"]))
            ordered = True
        elif kind == "privileges":
            criteria = {k: d for k, _, d in PRIVILEGES}
            question, target = common.choice(state, ask, criteria, [v for _, v, _ in PRIVILEGES].index(m["privilegesRequired"]))
            ordered = True
        elif kind == "vector":
            order = rng.sample(list(VECTORS), len(VECTORS))
            question, target = common.choice(state, ask, {VECTORS[v]: None for v in order}, order.index(m["attackVector"]))
        elif kind == "interaction":
            question, target = common.noul(ask, m["userInteraction"] == "REQUIRED")
        else:
            options = [cwe] + rng.sample([c for c in common_cwes if c != cwe and c in names], rng.randint(3, 5))
            rng.shuffle(options)
            question, target = common.choice(state, ask, {c: names[c] for c in options}, options.index(cwe))
        stats["questions"][kind] += 1
        yield common.example(NVD, f"{cve_id}:{kind}", "vulnerability-triage", state, question, target, "corpus", ordered=ordered)


FETCH = {
    "aegis": ("hub", AEGIS_REPO, ["README.md", "train.json", "validation.json"]),  # pinned to AEGIS_REVISION in main()
    **{f"scruples_{n}": ("url", SCRUPLES_URL.format(n), f"{n}.tar.gz") for n in ("anecdotes", "dilemmas")},
    "cwe": ("url", CWE_URL, "1000.csv.zip"),
    **{f"nvd_{y}_{ext}": ("url", NVD_FEED.format(y, ext), f"nvdcve-2.0-{y}.{ext}") for y in NVD_YEARS for ext in ("json.gz", "meta")},
}
BUILDERS = [
    (AEGIS, aegis, {"upstream": f"{AEGIS_REPO}@{AEGIS_REVISION[:12]} train.json + validation.json (prompt labels only)",
                    "licence": "CC-BY-4.0", "label_origin": "human (prompt_label_source = human on every row)",
                    "redacted": 0, "label_ties_dropped": 0, "variant_spellings_merged": 0, "noul": collections.Counter(),
                    "category": collections.Counter()}),
    (SCRUPLES, scruples, {"upstream": "allenai/scruples v1.0 anecdotes + dilemmas (train, dev) from the README's storage links",
                          "licence": "Apache-2.0 (repository); anecdotes are Reddit posts",
                          "label_origin": "human (community votes on anecdotes; 5 crowd annotators per dilemma); targets are the vote shares",
                          "anecdotes_dropped": 0, "anecdotes_split_vote_dropped": 0, "dilemmas_dropped": 0, "dilemmas_split_vote_dropped": 0,
                          "anecdotes": collections.Counter(), "dilemmas": collections.Counter()}),
    (NVD, nvd, {"upstream": "NVD CVE JSON 2.0 yearly feeds (sha256 per year below; NVD updates them daily) + CWE list 1000.csv",
                "licence": "NVD data: no licensing restrictions (NIST); CVE descriptions: MITRE CVE terms (free use with notice); CWE: MITRE terms",
                "label_origin": "human (NVD analysts' CVSS 3.1 base metrics and CWE; vendor/CNA scores not used)",
                "feeds": {}, "eligible_cves": 0, "cvss_in_text_dropped": 0, "questions": collections.Counter()}),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", nargs="*", default=None, help="download these (all when none given)")
    p.add_argument("names", nargs="*")
    args = p.parse_args()
    if args.fetch is not None:
        for name in args.fetch or list(FETCH):
            if name == "aegis":
                local, revision = common.fetch(AEGIS_REPO, FETCH[name][2], revision=AEGIS_REVISION)
                record = {"name": name, "repo": AEGIS_REPO, "revision": revision, "path": str(local)}
            else:
                sub = RAW / ("nvd" if name.startswith("nvd") else "scruples" if name.startswith("scruples") else "cwe")
                _, url, file = FETCH[name]
                sub.mkdir(parents=True, exist_ok=True)
                subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", str(sub / file), url], check=True)
                record = {"name": name, "url": url, "path": str(sub / file)}
            print(json.dumps(record), flush=True)
    else:
        wanted = set(args.names)
        for name, build, meta in BUILDERS:
            if not wanted or name in wanted:
                common.write(name, build(meta), meta)
