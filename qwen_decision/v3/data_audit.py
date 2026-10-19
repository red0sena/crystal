"""Training data against the test distribution: prompt length, option count, question type, volume.

  python -m qwen_decision.v3.data_audit --out runs/data-audit

Every prompt is rendered by format.py exactly as the engine renders it (d3 layout) and tokenized, so the
lengths are the model's real input lengths. Groups:
- suite:<benchmark>: the public 0.3 suite (exclusions applied), up to --per-benchmark requests each;
- jeval:<source>: the private-like proxy (JEVal), up to --per-source rows each;
- corpus:<source>: the Jev distillation corpus train split (sampled);
- phase2:<source>: our phase-2 sources (sampled);
- train-mix-augmented: the m2 recipe's draws with its training augmentations (option padding, noul as choice).
Writes summary.json (per group statistics) and prints the comparison tables.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
from transformers import AutoTokenizer

from decision_index.suite.io import Suite

from qwen_decision.v3 import format as fmt
from qwen_decision.v3 import jev_data, proxy_eval
from qwen_decision.v3.sources import common

ROOT = Path(__file__).resolve().parents[2]
LENGTH_EDGES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
OPTION_EDGES = (2, 3, 5, 10, 17, 33, 65, 161, 256)


def stats(lengths: list[int], options: list[int], types: Counter, extra: dict | None = None) -> dict:
    L, K = np.array(lengths), np.array(options)
    out = {"prompts": len(L),
           "tokens": {q: int(np.percentile(L, q)) for q in (10, 50, 90, 99)} | {"max": int(L.max()), "mean": float(L.mean())},
           "tokens_over": {str(e): float((L > e).mean()) for e in LENGTH_EDGES},
           "options": {q: int(np.percentile(K, q)) for q in (50, 90, 99)} | {"max": int(K.max())},
           "options_at_least": {str(e): float((K >= e).mean()) for e in OPTION_EDGES[1:]},
           "types": {t: n / len(L) for t, n in types.items()}}
    return out | (extra or {})


class Measure:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.codes, _ = fmt.answer_codes(tokenizer)

    def lengths(self, items: list[tuple]) -> list[int]:
        """items: (state, kit question); returns rendered prompt token counts."""
        texts = [fmt.render(self.tokenizer, state, q, self.codes[:len(fmt.options(q)[0])]) for state, q in items]
        out = []
        for i in range(0, len(texts), 256):
            out += [len(x) for x in self.tokenizer(texts[i:i + 256], add_special_tokens=False)["input_ids"]]
        return out


def kit_type(q: dict, ordered: bool = False) -> str:
    return "noul" if q["type"] == "noul" else "ordered" if ordered else "choice"


def suite_groups(m: Measure, per_benchmark: int, seed: int) -> dict:
    suite = Suite(ROOT / "external/decision-index/suite-0.3")
    names = {b["catalog_id"]: b["dataset"] for b in suite.manifest()["benchmarks"]}
    by = defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        by[names.get(r["_evaluation"]["catalog_id"], str(r["_evaluation"]["catalog_id"]))].append(r)
    out = {}
    for name, rows in sorted(by.items()):
        rng = random.Random(f"{seed}:{name}")
        chosen = rng.sample(rows, min(per_benchmark, len(rows)))
        items = [(r["state"], q) for r in chosen for q in r["questions"].values()]
        types = Counter(kit_type(q) for _, q in items)
        out[f"suite:{name}"] = stats(m.lengths(items), [len(fmt.options(q)[0]) for _, q in items], types,
                                     {"requests_total": len(rows), "questions_per_request_mean": len(items) / len(chosen)})
    return out


def jeval_groups(m: Measure, per_source: int, seed: int) -> dict:
    by = defaultdict(list)
    for r in proxy_eval.load_rows(per_source, seed=seed):
        by[r["source"]].append(r)
    out = {}
    for source, rows in sorted(by.items()):
        items, types = [], Counter()
        for r in rows:
            (q,) = json.loads(r["questions_json"]).values()
            kit, levels = proxy_eval.as_kit(q)
            items.append((r["state"], kit))  # passed to the engine exactly like this by proxy_eval
            types[kit_type(kit, levels is not None)] += 1
        out[f"jeval:{source}"] = stats(m.lengths(items), [len(fmt.options(q)[0]) for _, q in items], types,
                                       {"group": proxy_eval.group_of(source)})
    return out


def example_groups(m: Measure, prefix: str, rows: list[dict], per_source: int, seed: int, totals: Counter) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = {}
    for source, items in sorted(by.items()):
        chosen = random.Random(f"{seed}:{source}").sample(items, min(per_source, len(items)))
        pairs = [(e["state"], e["question"]) for e in chosen]
        types = Counter(kit_type(e["question"], e.get("ordered", False)) for e in chosen)
        out[f"{prefix}:{source}"] = stats(m.lengths(pairs), [len(fmt.options(q)[0]) for _, q in pairs], types,
                                          {"rows_total": totals[source]})
    return out


def augmented(m: Measure, sample: int, seed: int) -> dict:
    """The m2 recipe's training draws, rendered with its augmentations by train.Renderer."""
    from qwen_decision.v3 import train as T

    manifest = json.loads((ROOT / "runs/v3/m2-noncausal-d3/manifest.json").read_text())
    args = argparse.Namespace(**{k: manifest["args"][k] for k in ("phase2", "phase2_share", "seed", "pad_probability",
                                                                   "noul_choice_probability", "layout")})
    rows = T.mix_phase2(jev_data.load("train", manifest["args"]["train_rows"], seed=args.seed), args)
    pool = sorted({o for e in rows if e["question"]["type"] == "choice" and not e["ordered"] for o in e["question"]["criteria"]})
    renderer = T.Renderer(m.tokenizer, args, pool)
    rng = random.Random(seed)
    draws = [renderer(e, rng) for e in rng.sample(rows, min(sample, len(rows)))]
    types = Counter("noul-as-choice" if d["noul_as_choice"] else "padded" if d["padded"] else "plain" for d in draws)
    return {"train-mix-augmented": stats([len(d["input_ids"]) for d in draws], [len(d["code_ids"]) for d in draws], types,
                                         {"rows_total": len(rows)})}


def weighted(groups: dict, prefix: str, key: str, edge: str) -> float:
    """Mean over groups with this prefix (each benchmark/source weighs the same, as in the index)."""
    values = [g[key][edge] for name, g in groups.items() if name.startswith(prefix + ":")]
    return float(np.mean(values)) if values else math.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "runs/data-audit")
    p.add_argument("--per-benchmark", type=int, default=300)
    p.add_argument("--per-source", type=int, default=300)
    p.add_argument("--corpus-sample", type=int, default=60000)
    p.add_argument("--augmented-sample", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    m = Measure(AutoTokenizer.from_pretrained(ROOT / "models/qwen3.5-0.8b"))
    groups = suite_groups(m, args.per_benchmark, args.seed)
    groups |= jeval_groups(m, args.per_source, args.seed)
    corpus = jev_data.load("train")
    corpus_totals = Counter(r["source"] for r in corpus)
    corpus = random.Random(args.seed).sample(corpus, min(args.corpus_sample, len(corpus)))
    groups |= example_groups(m, "corpus", corpus, args.per_source, args.seed, corpus_totals)
    names = sorted(f.name[:-6] for f in (ROOT / "data/phase2").glob("*.jsonl"))
    phase2 = common.load(names)
    groups |= example_groups(m, "phase2", phase2, args.per_source, args.seed, Counter(r["source"] for r in phase2))
    groups |= augmented(m, args.augmented_sample, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(groups, indent=1) + "\n")
    print("share of prompts over N tokens (mean over benchmarks/sources)")
    print("group     " + "".join(f"{'>' + e:>9}" for e in map(str, LENGTH_EDGES)))
    for prefix in ("suite", "jeval", "corpus", "phase2"):
        print(f"{prefix:10}" + "".join(f"{weighted(groups, prefix, 'tokens_over', str(e)):9.3f}" for e in LENGTH_EDGES))
    g = groups["train-mix-augmented"]
    print(f"{'train-aug':10}" + "".join(f"{g['tokens_over'][str(e)]:9.3f}" for e in LENGTH_EDGES))
    print("share of prompts with at least N options")
    print("group     " + "".join(f"{'>=' + str(e):>9}" for e in OPTION_EDGES[1:]))
    for prefix in ("suite", "jeval", "corpus", "phase2"):
        print(f"{prefix:10}" + "".join(f"{weighted(groups, prefix, 'options_at_least', str(e)):9.3f}" for e in OPTION_EDGES[1:]))
    print(f"{'train-aug':10}" + "".join(f"{g['options_at_least'][str(e)]:9.3f}" for e in OPTION_EDGES[1:]))


if __name__ == "__main__":
    main()
