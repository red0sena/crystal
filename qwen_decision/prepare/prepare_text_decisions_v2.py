"""Build text-decisions-v2: the unchanged v1 mixture plus 2-64 option, noul,
score, soft-target and long-document tasks.

No training or generation occurs here. Train rows that share text with any test
row (v1 or v2) are dropped; test rows are never moved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from qwen_decision.sources.fetch_text_sources import ROOT, sha, write_json
import qwen_decision.core.decision_format as fmt
from qwen_decision.paths import code_path

V1 = ROOT / "data/text-decisions-v1"
RAW = ROOT / "data/text-decisions-v2/raw"
OUT = ROOT / "data/text-decisions-v2"
SEED = 20261011
TRAIN_CAPS = {"banking_intent_wide": 6000, "clinc_intent": 6000, "massive_intent": 6000, "newsgroup_topic": 4000,
              "fine_emotion": 4000, "paraphrase_retrieval": 4000, "boolq_noul": 6000, "sts_score": 6000,
              "review_stars": 4000, "long_reading": 4000}
TEST_CAPS = {"newsgroup_topic": 2000, "fine_emotion": 2000, "review_stars": 2000}
WIDE_TEST_K = {"banking_intent_wide": (16, 32, 64), "clinc_intent": (16, 32, 64), "massive_intent": (16, 32, 60),
               "paraphrase_retrieval": (16, 64)}
WIDE_TEST_CAP = 1000
STS_LEVELS = [
    "0: The two sentences are completely dissimilar.",
    "1: Not equivalent, but on the same topic.",
    "2: Not equivalent, but they share some details.",
    "3: Roughly equivalent, but some important information differs or is missing.",
    "4: Mostly equivalent; only unimportant details differ.",
    "5: Completely equivalent; they mean the same thing.",
]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def content_hash(text):
    return digest(norm(text))


def seeded(key):
    return random.Random(int(digest(f"{SEED}:{key}")[:16], 16))


def part(key, fractions):
    p = int(digest(f"{SEED}:split:{key}")[:16], 16) / 2**64
    return "train" if p < fractions[0] else "dev" if p < sum(fractions) else "test"


def v1_part(key, fractions):
    # The exact v1 assignment, so BANKING77 texts stay on the side v1 used.
    p = int(digest(f"20261010:split:{key}")[:16], 16) / 2**64
    return "train" if p < fractions[0] else "dev" if p < sum(fractions) else "test"


LABEL_PREFIX = re.compile(r"^(sentence \d|premise|hypothesis|title|relevant passage|background passage):\s*", re.I)


def atoms(text):
    """Normalized paragraphs long enough to identify shared source text."""
    out = set()
    for piece in re.split(r"\n\s*\n", text):
        piece = LABEL_PREFIX.sub("", piece.strip())
        n = norm(piece)
        if len(n) >= 25:
            out.add(n)
    return out


def parquet(path):
    return pq.read_table(RAW / path).to_pylist()


def record(task, qtype, state, question, options, answer, src, split, *, group, target=None, sampling=None,
           fixed_k=None, provenance="source_ground_truth", annotation=None):
    names = list(options)
    assert len(set(names)) == len(names) and answer in options
    rid = digest(json.dumps([task, src, fixed_k], sort_keys=True))[:24]
    r = {"id": rid, "group_id": digest(group)[:32], "task": task, "qtype": qtype, "split": split,
         "request": {"state": state, "question": question, "options": options},
         "answer_id": answer, "sample_weight": 1.0, "label_provenance": provenance, "source": src}
    if target is not None:
        assert abs(sum(target.values()) - 1) < 1e-6 and set(target) <= set(options)
        r["target"] = target
    if sampling is not None:
        r["option_sampling"] = sampling
    if fixed_k is not None:
        r["fixed_k"] = fixed_k
    if annotation:
        r["annotation"] = annotation
    if sampling is None:
        assert 2 <= len(options) <= fmt.MAX_OPTIONS, (task, len(options))
    return r


def sample_k(rng, max_k):
    """One third each from [2,16], [17,32], [33,max] (clamped to the label count)."""
    buckets = [(2, 16), (17, 32), (33, fmt.MAX_OPTIONS)]
    lo, hi = rng.choice([b for b in buckets if b[0] <= max_k])
    return rng.randint(lo, min(hi, max_k))


def fixed_candidates(rng, labels, gold, k):
    others = [x for x in labels if x != gold]
    chosen = [gold, *rng.sample(others, k - 1)]
    rng.shuffle(chosen)
    return chosen


def wide_records(task, rows, labels, describe, question, split_of, src_of, rng_key):
    """rows: (text, gold_label, source_split, row_id). Train keeps the full label
    set and samples K per draw; dev samples one K; test gets fixed-K variants."""
    out = []
    for text, gold, source_split, row_id in rows:
        split = split_of(text, source_split)
        group = "text:" + content_hash(text)
        src = src_of(source_split, row_id)
        if split == "train":
            out.append(record(task, "choice", text, question, {x: describe(x) for x in labels}, gold, src, split,
                              group=group, sampling={"max_k": min(fmt.MAX_OPTIONS, len(labels))}))
        elif split == "dev":
            rng = seeded(f"{rng_key}:dev:{row_id}")
            k = sample_k(rng, min(fmt.MAX_OPTIONS, len(labels)))
            chosen = fixed_candidates(rng, labels, gold, k)
            out.append(record(task, "choice", text, question, {x: describe(x) for x in chosen}, gold, src, split,
                              group=group))
        else:
            for k in WIDE_TEST_K[task]:
                rng = seeded(f"{rng_key}:test:{k}:{row_id}")
                chosen = fixed_candidates(rng, labels, gold, k)
                out.append(record(f"{task}@{k}", "choice", text, question, {x: describe(x) for x in chosen}, gold,
                                  src, split, group=group, fixed_k=k))
    return out


def load_new():
    data = []
    # BANKING77 across all 77 intents, same split function as v1.
    intents = json.loads((V1 / "raw/PolyAI__banking77/dataset_infos.json").read_text())["default"]["features"]["label"]["names"]
    rows = []
    for split in ("train", "test"):
        with (V1 / "raw/banking77-original" / f"{split}.csv").open() as f:
            rows += [(r["text"], r["category"], split, f"{split}:{i}") for i, r in enumerate(csv.DictReader(f))]
    data += wide_records(
        "banking_intent_wide", rows, intents, lambda x: x.replace("_", " ").replace("?", ""),
        "Which of the supplied customer-support intents best matches this request?",
        lambda text, s: v1_part(content_hash(text), (.95, .05)) if s == "train" else "test",
        lambda s, i: {"dataset": "PolyAI/banking77", "split": s, "row_id": i}, "banking")

    readme = (RAW / "clinc__clinc_oos/README.md").read_text()
    seg = readme[readme.find("config_name: plus"):]
    seg = seg[seg.find("names:"):]
    clinc = {}
    for m in re.finditer(r"'(\d+)': (\S+)", seg):
        if int(m.group(1)) in clinc:
            break
        clinc[int(m.group(1))] = m.group(2)
    assert len(clinc) == 151 and clinc[42] == "oos"
    labels = sorted(v for k, v in clinc.items() if v != "oos")
    rows = [(r["text"], clinc[r["intent"]], split, f"{split}:{i}")
            for split in ("train", "validation", "test")
            for i, r in enumerate(parquet(f"clinc__clinc_oos/plus/{split}-00000-of-00001.parquet"))
            if clinc[r["intent"]] != "oos"]
    data += wide_records(
        "clinc_intent", rows, labels, lambda x: x.replace("_", " "), "Which intent best matches this user request?",
        lambda text, s: {"train": "train", "validation": "dev", "test": "test"}[s],
        lambda s, i: {"dataset": "clinc/clinc_oos", "config": "plus", "split": s, "row_id": i}, "clinc")

    massive = {s: parquet(f"mteb__amazon_massive_intent/en/{s}.parquet") for s in ("train", "validation", "test")}
    labels = sorted({r["label_text"] for rr in massive.values() for r in rr})
    assert len(labels) == 60, len(labels)
    rows = [(r["text"], r["label_text"], s, f"{s}:{r['id']}") for s, rr in massive.items() for r in rr]
    data += wide_records(
        "massive_intent", rows, labels, lambda x: x.replace("_", " "),
        "Which intent best matches this voice-assistant request?",
        lambda text, s: {"train": "train", "validation": "dev", "test": "test"}[s],
        lambda s, i: {"dataset": "mteb/amazon_massive_intent", "config": "en", "split": s, "row_id": i}, "massive")

    # MRPC equivalent pairs: find the paraphrase among random other sentences.
    mrpc = {s: pq.read_table(V1 / f"raw/nyu-mll__glue/mrpc/{s}-00000-of-00001.parquet").to_pylist()
            for s in ("train", "validation")}
    for source_split, rr in mrpc.items():
        pool = sorted({r["sentence2"] for r in rr} | {r["sentence1"] for r in rr})
        for r in rr:
            if r["label"] != 1:
                continue
            split = part(content_hash(r["sentence1"]), (.95, .05)) if source_split == "train" else "test"
            others = [x for x in pool if x not in (r["sentence1"], r["sentence2"])]
            src = {"dataset": "nyu-mll/glue", "config": "mrpc", "split": source_split, "row_id": r["idx"]}
            question = "Which option expresses the same meaning as the state sentence?"
            group = "text:" + content_hash(r["sentence1"])
            if split == "test":
                for k in WIDE_TEST_K["paraphrase_retrieval"]:
                    rng = seeded(f"mrpc:test:{k}:{r['idx']}")
                    chosen = [r["sentence2"], *rng.sample(others, k - 1)]
                    rng.shuffle(chosen)
                    opts = {f"s{i}": x for i, x in enumerate(chosen)}
                    gold = f"s{chosen.index(r['sentence2'])}"
                    data.append(record(f"paraphrase_retrieval@{k}", "choice", r["sentence1"], question, opts, gold,
                                       src, split, group=group, fixed_k=k))
            else:
                rng = seeded(f"mrpc:{split}:{r['idx']}")
                k = fmt.MAX_OPTIONS if split == "train" else sample_k(rng, fmt.MAX_OPTIONS)
                chosen = [r["sentence2"], *rng.sample(others, k - 1)]
                rng.shuffle(chosen)
                opts = {f"s{i}": x for i, x in enumerate(chosen)}
                gold = f"s{chosen.index(r['sentence2'])}"
                data.append(record("paraphrase_retrieval", "choice", r["sentence1"], question, opts, gold, src, split,
                                   group=group, sampling={"max_k": fmt.MAX_OPTIONS} if split == "train" else None))

    # 20 Newsgroups and GoEmotions use their full label sets (20 and 28).
    ng = {s: [json.loads(x) for x in (RAW / f"SetFit__20_newsgroups/{s}.jsonl").open() if x.strip()]
          for s in ("train", "test")}
    ng_labels = sorted({r["label_text"] for rr in ng.values() for r in rr})
    assert len(ng_labels) == 20
    for s, rr in ng.items():
        for i, r in enumerate(rr):
            text = r["text"].strip()
            if not 20 <= len(text) <= 20000:
                continue
            split = part(content_hash(text), (.95, .05)) if s == "train" else "test"
            rng = seeded(f"ng:{s}:{i}")
            labels = ng_labels[:]
            rng.shuffle(labels)
            data.append(record("newsgroup_topic", "choice", text, "Which newsgroup was this message posted to?",
                               {x: x for x in labels}, r["label_text"],
                               {"dataset": "SetFit/20_newsgroups", "split": s, "row_id": i}, split,
                               group="text:" + content_hash(text)))
    readme = (RAW / "google-research-datasets__go_emotions/README.md").read_text()
    seg = readme[readme.find("config_name: simplified"):]
    seg = seg[seg.find("names:"):]
    emo = {}
    for m in re.finditer(r"'(\d+)': (\S+)", seg):
        if int(m.group(1)) in emo:
            break
        emo[int(m.group(1))] = m.group(2)
    assert len(emo) == 28
    for s in ("train", "validation", "test"):
        for r in parquet(f"google-research-datasets__go_emotions/simplified/{s}-00000-of-00001.parquet"):
            if len(r["labels"]) != 1 or len(r["text"].strip()) < 10:
                continue
            labels = [emo[i] for i in range(28)]
            seeded(f"emo:{r['id']}").shuffle(labels)
            data.append(record("fine_emotion", "choice", r["text"], "Which emotion is most clearly expressed in this comment?",
                               {x: x for x in labels}, emo[r["labels"][0]],
                               {"dataset": "google-research-datasets/go_emotions", "config": "simplified", "split": s, "row_id": r["id"]},
                               {"train": "train", "validation": "dev", "test": "test"}[s], group="text:" + content_hash(r["text"])))

    # BoolQ as noul (yes/no).
    for s in ("train", "validation"):
        for i, r in enumerate(parquet(f"google__boolq/data/{s}-00000-of-00001.parquet")):
            split = part(content_hash(r["passage"]), (.95, .05)) if s == "train" else "test"
            q = r["question"].strip()
            q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
            opts = dict(fmt.NOUL_OPTIONS)
            if seeded(f"boolq:{s}:{i}").random() < .5:
                opts = dict(reversed(list(opts.items())))
            data.append(record("boolq_noul", "noul", r["passage"], q, opts, "yes" if r["answer"] else "no",
                               {"dataset": "google/boolq", "split": s, "row_id": i}, split,
                               group="text:" + content_hash(r["passage"])))

    # STS-B as score with soft targets between the two neighbouring levels.
    for s in ("train", "validation"):
        for r in parquet(f"nyu-mll__glue/stsb/{s}-00000-of-00001.parquet"):
            text = "Sentence 1:\n" + r["sentence1"] + "\n\nSentence 2:\n" + r["sentence2"]
            split = part(content_hash(text), (.95, .05)) if s == "train" else "test"
            value = min(5., max(0., float(r["label"])))
            lo = math.floor(value)
            target = {str(lo): 1. - (value - lo)}
            if value > lo:
                target[str(lo + 1)] = value - lo
            target = {k: v for k, v in target.items() if v > 0}
            data.append(record("sts_score", "score", text, "How similar in meaning are the two sentences?",
                               {str(i): STS_LEVELS[i] for i in range(6)}, str(round(value)),
                               {"dataset": "nyu-mll/glue", "config": "stsb", "split": s, "row_id": r["idx"]}, split,
                               group="text:" + content_hash(text), target=target, provenance="soft_from_continuous_rating",
                               annotation={"rating": value}))

    # Yelp stars as score. Deterministic hash sample; long reviews are kept.
    for s, cap in (("train", 9000), ("test", 4000)):
        rows = parquet(f"Yelp__yelp_review_full/yelp_review_full/{s}-00000-of-00001.parquet")
        order = sorted(range(len(rows)), key=lambda i: digest(f"{SEED}:yelp:{s}:{i}"))[:cap]
        for i in order:
            r = rows[i]
            text = r["text"].replace("\\n", "\n").strip()
            split = part(content_hash(text), (.95, .05)) if s == "train" else "test"
            data.append(record("review_stars", "score", text, "How many stars did the reviewer give?",
                               {str(k + 1): STARS[k] for k in range(5)}, str(r["label"] + 1),
                               {"dataset": "Yelp/yelp_review_full", "split": s, "row_id": i}, split,
                               group="text:" + content_hash(text)))

    # QuALITY long-document multiple choice, split by article.
    for s, path in (("train", "emozilla__quality/data/train-00000-of-00001-0bffe0bf790d89eb.parquet"),
                    ("validation", "emozilla__quality/data/validation-00000-of-00001-77baeb9538209706.parquet")):
        for i, r in enumerate(parquet(path)):
            split = part(content_hash(r["article"]), (.9, .1)) if s == "train" else "test"
            options = {f"option_{j}": x for j, x in enumerate(r["options"])}
            names = list(options)
            seeded(f"quality:{s}:{i}").shuffle(names)
            data.append(record("long_reading", "choice", r["article"], r["question"], {k: options[k] for k in names},
                               f"option_{r['answer']}", {"dataset": "emozilla/quality", "split": s, "row_id": i}, split,
                               group="article:" + content_hash(r["article"]), annotation={"hard": r["hard"]}))
    return data


class IdentitySet:
    def __init__(self, items):
        self.ids = {id(x) for x in items}

    def __contains__(self, item):
        return id(item) in self.ids


# v1 JSONL을 다시 사용해 기존 과제를 보존하고 qtype만 추가한다. 학습은 원본 Qwen에서 시작한다.
# 공식 MMLU-Pro test 일부가 우리 학습에 들어갔으므로 공식 전체 test를 미학습 평가로 쓰면 안 된다.
def read_v1(split):
    with (V1 / f"{split}.jsonl").open() as f:
        rows = [json.loads(x) for x in f]
    for r in rows:
        r["qtype"] = "choice"
    return rows


def main():
    if (OUT / "train.jsonl").exists():
        raise SystemExit("Output exists. Preserve this dataset or use a new version.")
    v1 = {s: read_v1(s) for s in ("train", "dev", "test")}
    v1_test_ids = IdentitySet(v1["test"])
    new = load_new()
    removed = Counter()
    # Never let distinctive training or development text appear in a test record.
    # Distinctive = a normalized paragraph of >=40 chars that occurs in at most two
    # test groups; widely repeated boilerplate (licence notes, signatures) is not.
    owners = defaultdict(set)
    new_test = [r for r in new if r["split"] == "test"]

    def texts(r):
        yield r["request"]["state"]
        if r["task"].startswith("paraphrase_retrieval"):
            yield from r["request"]["options"].values()

    for r in v1["test"] + new_test:
        for t in texts(r):
            for a in atoms(t):
                if len(a) >= 40:
                    owners[a].add(("v1" if r in v1_test_ids else "v2", r["group_id"]))
    # 반복 안내문/라이선스 문구까지 모두 중복으로 취급하지 않도록 고유한 문단만 검사한다.
    # 이 검사는 직접 중복 방지용이며 번역/패러프레이즈 또는 기반 모델 사전학습 중복을 증명하지 않는다.
    distinctive = {a: {o for o, _ in g} for a, g in owners.items() if len(g) <= 2}

    def hits(text, scope):
        return any(a in distinctive and distinctive[a] & scope for a in atoms(text))

    splits = {"train": [], "dev": [], "test": []}
    for r in v1["train"] + v1["dev"]:
        # v1 rows already satisfy v1's own split rule; check them only against new tests.
        if any(hits(t, {"v2"}) for t in texts(r)):
            removed[f"v1_{r['split']}_shares_text_with_v2_test:{r['task']}"] += 1
            continue
        splits[r["split"]].append(r)
    splits["test"].extend(v1["test"])
    by = defaultdict(list)
    for r in new:
        if r["split"] != "test":
            if hits(r["request"]["state"], {"v1", "v2"}) or hits(r["request"]["options"][r["answer_id"]], {"v1", "v2"}):
                removed[f"{r['split']}_shares_text_with_test:{r['task']}"] += 1
                continue
            if r["task"] == "paraphrase_retrieval":
                # Drop only the distractor sentences that occur in a test record.
                kept = {k: v for k, v in r["request"]["options"].items()
                        if k == r["answer_id"] or not hits(v, {"v1", "v2"})}
                removed["paraphrase_retrieval_distractors_dropped"] += len(r["request"]["options"]) - len(kept)
                if len(kept) < min(17, len(r["request"]["options"])):
                    removed[f"{r['split']}_too_few_clean_distractors:{r['task']}"] += 1
                    continue
                r["request"]["options"] = kept
        by[(r["split"], r["task"])].append(r)
    seen = set()
    for (split, task), rr in sorted(by.items()):
        rr.sort(key=lambda r: digest(r["id"]))
        unique = []
        for r in rr:
            key = (task, content_hash(r["request"]["state"] + "\n" + r["request"]["question"]), r["answer_id"])
            if key in seen:
                removed[f"duplicate:{split}:{task}"] += 1
                continue
            seen.add(key)
            unique.append(r)
        base = task.split("@")[0]
        cap = TRAIN_CAPS[base] if split == "train" else 1500 if split == "dev" else (
            WIDE_TEST_CAP if "@" in task else TEST_CAPS.get(base, 4000))
        removed[f"cap:{split}:{task}"] += max(0, len(unique) - cap)
        splits[split].extend(unique[:cap])
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rr in splits.items():
        rr.sort(key=lambda r: digest(r["id"]))
        assert len({r["id"] for r in rr}) == len(rr)
        path = OUT / f"{split}.jsonl"
        with path.open("w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"total": len(rr), "by_task": dict(sorted(Counter(r["task"] for r in rr).items())),
                        "by_qtype": dict(Counter(r["qtype"] for r in rr)), "file_sha256": sha(path)}
    examples = {}
    for r in sorted(splits["train"], key=lambda r: r["id"]):
        examples.setdefault(r["task"], r)
    write_json(OUT / "examples.json", examples)
    write_json(OUT / "manifest.json", {
        "version": "text-decisions-v2", "seed": SEED, "language": "English", "stats": stats,
        "removed": dict(sorted(removed.items())), "v1_files_sha256": {s: sha(V1 / f"{s}.jsonl") for s in ("train", "dev", "test")},
        "sources": json.loads((RAW / "sources.json").read_text()),
        "code_sha256": {n: sha(code_path(n)) for n in ("fetch_text_sources_v2.py", "prepare_text_decisions_v2.py", "decision_format.py")},
        "answer_codes": fmt.CODES, "max_options": fmt.MAX_OPTIONS, "system_prompt": fmt.SYSTEM,
        "policies": {
            "v1_tasks": "v1 train/dev/test records unchanged except training rows sharing a paragraph with any v2 test row",
            "wide_tasks": "train rows keep the full label set and sample K per draw (1/3 each from 2-16, 17-32, 33-max); dev uses one sampled K; test uses fixed K variants named task@K",
            "distractors": "uniformly random, never lexically selected",
            "leak_rule": "a new train/dev row is removed when its state or gold text shares a distinctive paragraph (>=40 normalized chars, present in at most two test groups) with any test row; v1 rows are checked only against new v2 tests; MRPC retrieval drops only contaminated distractors",
            "soft_targets": "STS-B continuous rating split between neighbouring levels",
            "long_documents": "QuALITY split by article; validation is test",
        },
    })
    print(json.dumps(stats, indent=1))
    print(json.dumps(dict(sorted(removed.items())), indent=1))


if __name__ == "__main__":
    main()
