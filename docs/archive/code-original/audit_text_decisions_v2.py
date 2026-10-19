"""Shortcut audit and text-light baselines for every v2 test task, plus a
re-check that no distinctive test paragraph occurs in train/dev.

Baselines (none reads the state with a model):
  random        expected accuracy of a uniform guess, mean 1/K
  train_major   the task's most frequent training label when it is offered
  longest       the option with the longest description
  first_word    training lookup: first word of the state -> most common label
  overlap_hub   option sharing the most words with the other options (text-blind)
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from prepare_text_decisions_v2 import atoms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/text-decisions-v2"
OUT = ROOT / "results/v2-eval/baselines.json"
STOP = {"the", "a", "an", "of", "to", "and", "or", "my", "by", "not", "in", "for", "is", "on"}


def read(name):
    with (DATA / f"{name}.jsonl").open() as f:
        return [json.loads(x) for x in f]


def first_word(text):
    words = re.findall(r"[A-Za-z]+", text.lower())
    return words[0] if words else ""


def words(text):
    return set(re.findall(r"[a-z]+", text.lower())) - STOP


def pick_expected(scores, gold):
    """Accuracy of argmax with uniform tie breaking."""
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return (gold in winners) / len(winners)


def main():
    train, dev, test = read("train"), read("dev"), read("test")
    major = defaultdict(Counter)
    lookup = defaultdict(lambda: defaultdict(Counter))
    for r in train:
        base = r["task"]
        major[base][r["answer_id"]] += 1
        lookup[base][first_word(r["request"]["state"])][r["answer_id"]] += 1
    by_task = defaultdict(list)
    for r in test:
        by_task[r["task"]].append(r)
    results = {}
    for task, rr in sorted(by_task.items()):
        base = task.split("@")[0]
        acc = Counter()
        for r in rr:
            names = list(r["request"]["options"])
            texts = list(r["request"]["options"].values())
            gold = names.index(r["answer_id"])
            acc["random"] += 1 / len(names)
            ranked = [n for n, _ in major[base].most_common() if n in names]
            acc["train_major"] += (ranked[0] == r["answer_id"]) if ranked else 1 / len(names)
            acc["longest"] += pick_expected([len(t) for t in texts], gold)
            table = lookup[base].get(first_word(r["request"]["state"]))
            guess = [n for n, _ in table.most_common() if n in names] if table else []
            acc["first_word"] += (guess[0] == r["answer_id"]) if guess else (
                (ranked[0] == r["answer_id"]) if ranked else 1 / len(names))
            ws = [words(t) for t in texts]
            acc["overlap_hub"] += pick_expected([sum(len(w & o) for j, o in enumerate(ws) if j != i) for i, w in enumerate(ws)], gold)
        n = len(rr)
        scores = {k: 100 * v / n for k, v in acc.items()}
        best = max((k for k in scores if k != "random"), key=scores.get)
        results[task] = {"n": n, **{k: round(v, 2) for k, v in scores.items()}, "best_baseline": best,
                         "best_baseline_pct": round(scores[best], 2)}
    # Leak re-check for the new v2 tasks: distinctive test paragraphs (>=40 chars,
    # in at most two test groups) found in train or dev states.
    owners = defaultdict(set)
    atom_tests = defaultdict(set)
    for r in test:
        for a in atoms(r["request"]["state"]):
            if len(a) >= 40:
                owners[a].add(r["group_id"])
                atom_tests[a].add(r["id"])
    distinctive = {a for a, g in owners.items() if len(g) <= 2}
    v1_tasks = set(json.loads((ROOT / "data/text-decisions-v1/manifest.json").read_text())["stats"]["train"]["by_task"])
    leaks = Counter()
    affected = set()
    for r in train + dev:
        if r["task"] in v1_tasks:
            continue
        shared = atoms(r["request"]["state"]) & distinctive
        if shared:
            leaks[f"{r['split']}:{r['task']}"] += 1
            for a in shared:
                affected |= atom_tests[a]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"baselines": results, "new_task_train_dev_rows_sharing_distinctive_test_paragraphs": dict(leaks),
                               "affected_test_ids": sorted(affected),
                               "definitions": __doc__}, indent=2) + "\n")
    for task, r in results.items():
        print(f"{task:28s} n={r['n']:5d} random={r['random']:6.2f} major={r['train_major']:6.2f} longest={r['longest']:6.2f} "
              f"first_word={r['first_word']:6.2f} hub={r['overlap_hub']:6.2f}")
    print("leaks", dict(leaks), "affected test items", len(affected))


if __name__ == "__main__":
    main()
