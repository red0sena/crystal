"""Build a local, provenance-tracked mixture of text decision tasks.

No training and no answer generation occur here. Metadata and targets are kept
outside the request consumed by ChoiceScorer.
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
import zipfile

import numpy as np
import pyarrow.parquet as pq

from qwen_decision.sources.fetch_text_sources import OUT as RAW, ROOT, sha, write_json
from qwen_decision.core.scorer import validate_request
from qwen_decision.paths import code_path

OUT = ROOT / "data/text-decisions-v1"
SEED = 20261010
BLOOM = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
BLOOM_DESCRIPTIONS = [
    "Remember: retrieve facts, terms, or definitions from memory.",
    "Understand: explain, interpret, summarize, or classify meaning.",
    "Apply: use a procedure or principle in a specific situation.",
    "Analyze: examine components, relationships, assumptions, or evidence.",
    "Evaluate: make a justified judgment using criteria or standards.",
    "Create: construct an original plan, design, or coherent new product.",
]
ROUTING = ["Summarization", "Question Answering", "Question Generation", "Information Extraction", "Sentiment Analysis", "Textual Entailment", "Paraphrasing", "Text Categorization"]
TRAIN_CAPS = {"multiple_choice_qa": 6000, "physical_commonsense": 6000, "commonsense_qa": 6000, "cognitive_level": 5000}

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def content_hash(text):
    return digest(re.sub(r"\s+", " ", text).strip().casefold())

def seeded(key):
    return random.Random(int(digest(f"{SEED}:{key}")[:16], 16))

# Python hash() 대신 seed+SHA256을 사용해 프로세스나 실행 순서에 관계없이 분할한다.
# 문제 파생 과제(답 선택/분야 분류), 같은 지문, 같은 출처 그룹은 같은 split로 이동한다.
def part(key, fractions=(.90, .05)):
    p = int(digest(f"{SEED}:split:{key}")[:16], 16) / 2**64
    return "train" if p < fractions[0] else "dev" if p < sum(fractions) else "test"

def rows(repo, filename):
    return pq.read_table(RAW / repo.replace("/", "__") / filename).to_pylist()

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def source(repo, config, split, row_id, **extra):
    return {"dataset": repo, "config": config, "split": split, "row_id": str(row_id), **extra}

def record(task, state, question, names, descriptions, gold, src, split, *, base=None, group=None, provenance="source_ground_truth", weight=1., extra=None):
    assert len(names) == len(descriptions) and len(set(names)) == len(names)
    assert 0 <= gold < len(names) <= 26
    assert len(names) >= 2
    base_hash = content_hash(base if base is not None else state + "\n" + question)
    record_id = digest(json.dumps([task, src], sort_keys=True))[:24]
    result = {
        "id": record_id, "group_id": group or base_hash, "base_content_sha256": base_hash,
        "task": task, "split": split,
        "request": {"state": state, "question": question, "options": dict(zip(names, descriptions))},
        "answer_id": names[gold], "gold_index": gold,
        "label_provenance": provenance, "sample_weight": weight, "source": src,
    }
    if extra:
        result["annotation"] = extra
    validate_request(result["request"], 1.)
    return result

def mcq(task, row_id, question, options, gold, src, split, state="", group=None):
    return record(task, state, question, [f"option_{i}" for i in range(len(options))], options, gold, src, split,
                  base=state + "\n" + question, group=group)

def load_all():
    data = []
    for config in ("ARC-Challenge", "ARC-Easy"):
        for split in ("train", "validation", "test"):
            for r in rows("allenai/ai2_arc", f"{config}/{split}-00000-of-00001.parquet"):
                data.append(mcq("science_qa", r["id"], r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]), source("allenai/ai2_arc", config, split, r["id"]), {"train": "train", "validation": "dev", "test": "test"}[split]))
    for split in ("train", "validation"):
        for r in rows("tau/commonsense_qa", f"data/{split}-00000-of-00001.parquet"):
            assignment = part(r["id"], (.95, .05)) if split == "train" else "test"
            data.append(mcq("commonsense_qa", r["id"], r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]), source("tau/commonsense_qa", "default", split, r["id"]), assignment))

    archive = ROOT / "data/choice-benchmarks/ybisk__piqa/physicaliqa-train-dev.zip"
    with zipfile.ZipFile(archive) as z:
        for split, stem in (("train", "train"), ("validation", "dev")):
            member = next(n for n in z.namelist() if n.endswith(f"/{stem}.jsonl") or n == f"{stem}.jsonl")
            label_member = next(n for n in z.namelist() if n.endswith(f"/{stem}-labels.lst") or n == f"{stem}-labels.lst")
            rr = [json.loads(x) for x in z.read(member).decode().splitlines()]
            labels = [int(x) for x in z.read(label_member).decode().splitlines()]
            assert len(rr) == len(labels)
            for i, (r, label) in enumerate(zip(rr, labels)):
                assignment = part(content_hash(r["goal"]), (.95, .05)) if split == "train" else "test"
                data.append(mcq("physical_commonsense", i, r["goal"], [r["sol1"], r["sol2"]], label, source("ybisk/piqa", "plain_text", split, i), assignment))

    mmlu_file = ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet"
    mmlu = pq.read_table(mmlu_file, columns=["question_id", "question", "options", "answer_index", "category"]).to_pylist()
    domains = sorted({r["category"] for r in mmlu})
    for r in mmlu:
        # The same source question's answer and domain records ALWAYS travel together.
        group = content_hash(r["question"])
        assignment = part(group, (.80, .10))
        kept = [(i, text) for i, text in enumerate(r["options"]) if text != "N/A"]
        gold = [i for i, _ in kept].index(r["answer_index"])
        src = source("TIGER-Lab/MMLU-Pro", "default", "test", r["question_id"])
        data.append(mcq("multiple_choice_qa", r["question_id"], r["question"], [v for _, v in kept], gold, src, assignment, group=group))
        state = r["question"] + "\n\nAnswer candidates:\n" + "\n".join(v for _, v in kept)
        data.append(record("academic_domain", state, "Which academic domain best matches this problem?", domains, [x.replace("_", " ") for x in domains], domains.index(r["category"]), src, assignment, base=r["question"], group=group, provenance="source_category_metadata"))

    for split in ("train", "validation", "test"):
        for i, r in enumerate(rows("ehovy/race", f"all/{split}-00000-of-00001.parquet")):
            # All questions on the same article are one group.
            data.append(mcq("reading_comprehension", i, r["question"], r["options"], "ABCD".index(r["answer"]), source("ehovy/race", "all", split, i, example_id=r["example_id"]), {"train": "train", "validation": "dev", "test": "test"}[split], state=r["article"], group="race:" + content_hash(r["article"])))

    classifiers = [
        ("news_topic", "fancyzhx/ag_news", "data", "default", ("train", "test"), ["World", "Sports", "Business", "Sci/Tech"], "Which topic best matches this news report?", lambda r: r["text"]),
        ("sentiment", "stanfordnlp/sst2", "data", "default", ("train", "validation"), ["negative", "positive"], "What is the sentiment expressed in this text?", lambda r: r["sentence"]),
        ("emotion", "dair-ai/emotion", "split", "split", ("train", "validation", "test"), ["sadness", "joy", "love", "anger", "fear", "surprise"], "Which emotion is most clearly expressed in this text?", lambda r: r["text"]),
        ("textual_entailment", "nyu-mll/glue", "rte", "rte", ("train", "validation"), ["entailment", "not_entailment"], "Does the premise entail the hypothesis?", lambda r: "Premise:\n" + r["sentence1"] + "\n\nHypothesis:\n" + r["sentence2"]),
        ("paraphrase", "nyu-mll/glue", "mrpc", "mrpc", ("train", "validation"), ["not_equivalent", "equivalent"], "Do these two sentences convey the same meaning?", lambda r: "Sentence 1:\n" + r["sentence1"] + "\n\nSentence 2:\n" + r["sentence2"]),
        ("grammatical_acceptability", "nyu-mll/glue", "cola", "cola", ("train", "validation"), ["unacceptable", "acceptable"], "Is this sentence grammatically acceptable in English?", lambda r: r["sentence"]),
    ]
    for task, repo, prefix, config, splits, names, instruction, text_fn in classifiers:
        for split in splits:
            rr = rows(repo, f"{prefix}/{split}-00000-of-00001.parquet")
            for i, r in enumerate(rr):
                if r["label"] < 0:
                    continue
                text = text_fn(r)
                if split == "train":
                    assignment = "train" if "test" in splits else part(content_hash(text), (.95, .05))
                    if repo == "fancyzhx/ag_news":
                        assignment = part(content_hash(text), (.95, .05))
                else:
                    assignment = "dev" if split == "validation" and "test" in splits else "test"
                data.append(record(task, text, instruction, names, [n.replace("_", " ") for n in names], r["label"], source(repo, config, split, r.get("idx", i)), assignment, base=text))

    info = json.loads((RAW / "PolyAI__banking77/dataset_infos.json").read_text())["default"]
    intents = info["features"]["label"]["names"]
    assert len(intents) == 77
    for split in ("train", "test"):
        with (RAW / "banking77-original" / f"{split}.csv").open() as f:
            rr = list(csv.DictReader(f))
        for i, r in enumerate(rr):
            text, target = r["text"], r["category"]
            # All 77 intents are covered over the dataset; each request has 16
            # supplied candidates, compatible with the existing 26-code scorer.
            # This is NOT the standard full-77-class BANKING77 evaluation.
            rng = seeded(f"banking77:{split}:{i}")
            negatives = [x for x in intents if x != target]
            # Include lexically related confusions before random distractors.
            target_words = set(target.lower().split("_")) - {"my", "by", "not", "or", "to"}
            rng.shuffle(negatives)
            negatives.sort(key=lambda x: len(target_words & set(x.lower().split("_"))), reverse=True)
            hard = negatives[:7]
            rest = rng.sample(negatives[7:], 8)
            names = [target, *hard, *rest]
            rng.shuffle(names)
            assignment = part(content_hash(text), (.95, .05)) if split == "train" else "test"
            data.append(record("intent_selection", text, "Which of the supplied customer-support intents best matches this request?", names, [x.replace("_", " ").replace("?", "") for x in names], names.index(target), source("PolyAI/banking77", "default", split, i), assignment, base=text, extra={"candidate_protocol": "gold plus 7 lexical and 8 random distractors; 16 of 77 intents", "benchmark_equivalence": False}))

    # Preserve Bloom labels rather than conflating them with difficulty.
    cogdir = RAW / "mouryat9__CogBench/data"
    for split in ("train", "validation", "test"):
        for r in lines(cogdir / f"{split}.jsonl"):
            human = r["label_source"] == "original_author"
            if not human and (r.get("confidence") or 0.) < .8:
                continue
            level = int(r["bloom_level"])
            assert BLOOM[level - 1].casefold() == r["bloom_name"].casefold()
            assignment = part(content_hash(r["question_text"]), (.80, .10)) if human else {"train": "train", "validation": "dev", "test": "test"}[split]
            data.append(record("cognitive_level", r["question_text"], "Which Bloom cognitive process does this question primarily require?", BLOOM, BLOOM_DESCRIPTIONS, level - 1, source("mouryat9/CogBench", "default", split, r["question_id"]), assignment, base=r["question_text"], provenance="human_original_author" if human else "ccs_classifier_silver", weight=1. if human else .5, extra={"label_source": r["label_source"], "confidence": r["confidence"], "subject": r["subject"], "original_source": r["source"]}))

    # Compute relative difficulty boundaries from the TRAIN partition only.
    train_amc = rows("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC/train-00000-of-00001.parquet")
    amc_assignments = [part(content_hash(r["problem"]), (.90, .10)) for r in train_amc]
    ratings = [r["rating"] for r, split in zip(train_amc, amc_assignments) if split == "train"]
    thresholds = np.quantile(ratings, [1 / 3, 2 / 3]).tolist()
    write_json(OUT / "difficulty_rubric.json", {"source": "furonghuang-lab/Easy2Hard-Bench", "config": "E2H-AMC", "field": "rating", "cutoffs": thresholds, "method": "tertiles of training-only normalized IRT ratings", "labels": ["low", "medium", "high"], "scope": "relative empirical difficulty within AMC; not Bloom levels or universal cognitive difficulty"})
    for split in ("train", "eval"):
        rr = train_amc if split == "train" else rows("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC/eval-00000-of-00001.parquet")
        for i, r in enumerate(rr):
            if not math.isfinite(r["rating"]):
                continue
            label = int(r["rating"] > thresholds[0]) + int(r["rating"] > thresholds[1])
            assignment = amc_assignments[i] if split == "train" else "test"
            data.append(record("relative_difficulty", r["problem"], "Estimate this math problem's relative difficulty within AMC competitions using the training-calibrated low/medium/high bands.", ["low", "medium", "high"], ["Low difficulty within AMC", "Medium difficulty within AMC", "High difficulty within AMC"], label, source("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC", split, i), assignment, base=r["problem"], provenance="derived_from_published_irt_rating", extra={"original_rating": r["rating"], "cutoffs": thresholds}))

    ni = json.loads((RAW / "natural-instructions/manifest.json").read_text())
    by_category = defaultdict(list)
    for entry in ni["files"]:
        by_category[entry["category"]].append(entry)
    for category, entries in by_category.items():
        entries.sort(key=lambda x: digest(x["path"]))
        for j, entry in enumerate(entries):
            split = "test" if j == len(entries) - 1 else "dev" if len(entries) >= 3 and j == len(entries) - 2 else "train"
            task = json.loads((ROOT / entry["path"]).read_text())
            assert task["Categories"] == [category]
            definition = "\n".join(task["Definition"])
            instances = sorted(task["Instances"], key=lambda x: digest(x["id"]))[:150]
            family = "ni-family:" + digest(json.dumps(sorted(task["Source"])))
            for r in instances:
                # Actual task outputs and demonstrations are NEVER included.
                state = "Requested operation:\n" + definition + "\n\nInput to that operation:\n" + r["input"]
                data.append(record("task_type_routing", state, "Which task type best describes the requested operation? Classify the operation; do not carry it out.", ROUTING, ROUTING, ROUTING.index(category), source("allenai/natural-instructions", Path(entry["path"]).stem, "custom_source_family", r["id"]), split, base=r["input"], group=family, provenance="source_task_category_metadata", extra={"domains": task["Domains"], "source_family": task["Source"]}))
    return data

def rebalance_and_split(data):
    # Transitive grouping: identical base text and explicit family/article groups
    # both have to remain on one side of train/dev/test.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for r in data:
        union("group:" + r["group_id"], "content:" + r["base_content_sha256"])
    priority = {"train": 0, "dev": 1, "test": 2}
    resolved = {}
    for r in data:
        key = find("group:" + r["group_id"])
        if priority[r["split"]] > priority.get(resolved.get(key), -1):
            resolved[key] = r["split"]
    removed = Counter()
    clean = []
    seen = set()
    for r in data:
        key = find("group:" + r["group_id"])
        # Keep original held-out data fixed; conflicting training occurrences
        # are dropped instead of quietly moving them into test.
        if resolved[key] != r["split"]:
            removed["cross_split_group_conflict"] += 1
            continue
        dedup = (r["task"], r["base_content_sha256"], r["answer_id"])
        if dedup in seen:
            removed["same_task_text_target_duplicate"] += 1
            continue
        seen.add(dedup)
        r["group_id"] = digest(key)
        clean.append(r)
    groups = defaultdict(list)
    for r in clean:
        groups[(r["split"], r["task"])].append(r)
    selected = []
    for (split, task), rr in sorted(groups.items()):
        rr.sort(key=lambda r: digest(r["id"]))
        cap = TRAIN_CAPS.get(task, 4000) if split == "train" else 1500 if split == "dev" else 4000
        # Prioritize human cognition labels; silver labels are a supplement.
        if task == "cognitive_level" and split == "train":
            rr.sort(key=lambda r: (r["label_provenance"] != "human_original_author", digest(r["id"])))
        chosen = rr[:cap]
        removed[f"cap:{split}:{task}"] += max(0, len(rr) - cap)
        for r in chosen:
            names = list(r["request"]["options"])
            rng = seeded("display:" + r["id"])
            rng.shuffle(names)
            r["request"]["options"] = {k: r["request"]["options"][k] for k in names}
            r["gold_index"] = names.index(r["answer_id"])
        selected.extend(chosen)
    return selected, removed

def main():
    if (OUT / "train.jsonl").exists():
        raise SystemExit("Output already exists. Preserve this immutable dataset or use a new version.")
    data = load_all()
    prepared, removed = rebalance_and_split(data)
    # Refit after duplicate/group filtering: even a native train row can be
    # excluded when the same problem also occurs in a held-out source.
    # AMC 난이도는 절대 난이도가 아니라 해당 학습 집합 안의 상대적 IRT 난이도다.
    # 중복 제거 후 train만으로 경계를 맞춘다. dev/test를 써서 분위수 경계를 정하지 않는다.
    retained_ratings = [r["annotation"]["original_rating"] for r in prepared
                        if r["task"] == "relative_difficulty" and r["split"] == "train"]
    thresholds = np.quantile(retained_ratings, [1 / 3, 2 / 3]).tolist()
    rubric = json.loads((OUT / "difficulty_rubric.json").read_text())
    rubric.update(cutoffs=thresholds, method="tertiles of retained training-only normalized IRT ratings after duplicate/group filtering")
    write_json(OUT / "difficulty_rubric.json", rubric)
    for r in prepared:
        if r["task"] == "relative_difficulty":
            rating = r["annotation"]["original_rating"]
            index = int(rating > thresholds[0]) + int(rating > thresholds[1])
            r["answer_id"] = ["low", "medium", "high"][index]
            r["gold_index"] = list(r["request"]["options"]).index(r["answer_id"])
            r["annotation"]["cutoffs"] = thresholds
    stats = {}
    for split in ("train", "dev", "test"):
        rr = sorted((r for r in prepared if r["split"] == split), key=lambda r: digest(r["id"]))
        with (OUT / f"{split}.jsonl").open("w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"total": len(rr), "by_task": dict(sorted(Counter(r["task"] for r in rr).items())), "by_label_provenance": dict(Counter(r["label_provenance"] for r in rr)), "file_sha256": sha(OUT / f"{split}.jsonl")}
    example_records = {}
    for r in sorted(prepared, key=lambda r: r["id"]):
        if r["split"] == "train":
            example_records.setdefault(r["task"], r)
    write_json(OUT / "examples.json", example_records)
    sources = []
    for path in sorted(RAW.rglob("manifest.json")):
        sources.append({"manifest": str(path.relative_to(ROOT)), "sha256": sha(path)})
    for path in (ROOT / "data/mmlu-pro/data/test-00000-of-00001.parquet", ROOT / "data/choice-benchmarks/ybisk__piqa/physicaliqa-train-dev.zip"):
        sources.append({"local_file": str(path.relative_to(ROOT)), "sha256": sha(path)})
    manifest = {
        "version": "text-decisions-v1", "seed": SEED, "language": "English", "training_has_run": False,
        "source_records_before_filtering": len(data), "stats": stats, "removed": dict(removed), "sources": sources,
        "code_sha256": {n: sha(code_path(n)) for n in ("fetch_text_sources.py", "fetch_routing_sources.py", "prepare_text_decisions.py")},
        "system_prompt": "Use the provided state to answer the question. Select exactly one supplied option. Reply with only its option code.",
        "inference_protocol": "single forward; option-code logits only; no generated reasoning; existing LM head",
        "mmlu_pro_policy": "official test is intentionally repartitioned by question text into custom 80/10/10 train/dev/test before answer and domain transformations; old full-12032 score becomes training-contaminated once trained",
        "splits_policy": "native held-out splits preserved where available; CSQA/PIQA/SST2/GLUE validation treated as test with development rows drawn from native train; RACE article groups and NI Source families are disjoint; identical normalized base text across sources is also grouped; higher split priority drops conflicting training rows",
        "candidate_policy": "all original options for QA; all labels for small classifiers; BANKING77 gold plus 15 supplied distractors covers 77 intents across the mixture, not a standard full-77 classifier",
        "difficulty_policy": "AMC normalized IRT ratings binned using training-only tertiles; empirical relative difficulty is separate from the six Bloom cognitive processes",
        "sampling_policy": "deterministic hash samples with per-task caps; natural class frequencies, no label balancing; human cognition labels prioritized; silver cognition confidence >=0.8 and sample_weight=0.5",
        "license_policy": "upstream licenses and notices retained with each source; mixture not assigned a new universal license and not published",
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps(stats, indent=2), flush=True)

if __name__ == "__main__":
    main()
