"""Option-order bias on JEVal: how often the answer flips when only the option order changes, and position skew.

  python -m qwen_decision.v3.bias_eval --model runs/v3/<run>/final --out runs/bias/<name>.json
  python -m qwen_decision.v3.bias_eval --model models/d3-lite --attention noncausal --out runs/bias/d3-lite.json

Each choice question of the JEVal split (score and noul questions are left out: their order is fixed or
binary) is scored in up to four display orders: as given, reversed, rotated by half and that rotation
reversed (engine.option_orders, the same orders `orders=4` averages). Two failures are kept apart, as in
"One Bias Score, Two Factors" (Ok et al. 2026):
- flip_rate: share of questions whose chosen option (by content) differs between orders. Averaging orders
  hides flips but can settle on the same wrong option, so accuracy is reported per order and averaged.
- position_rstd: standard deviation over display positions 1-4 of the recall of questions whose gold sits
  at that position, pooled over orders (RStd, Zheng et al. 2024, "Large Language Models Are Not Robust
  Multiple Choice Selectors"); 0 = no position preference. first_position_rate is the share of answers on
  the first shown option against the share of golds there.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from qwen_decision.v3 import format as fmt
from qwen_decision.v3.engine import Scorer, option_orders, reorder
from qwen_decision.v3.proxy_eval import as_kit, group_of, load_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--attention")
    p.add_argument("--split", choices=("dev", "restricted", "all"), default="dev")
    p.add_argument("--per-source", type=int, default=60)
    p.add_argument("--orders", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    scorer = Scorer(args.model, attention=args.attention)
    rows = load_rows(args.per_source, split=args.split)
    items, meta = [], []
    for r in rows:
        key, question = next(iter(json.loads(r["questions_json"]).items()))
        if question["type"] != "choice":
            continue
        labels = json.loads(r["label_json"])
        label = str(labels[key] if isinstance(labels, dict) else labels)
        question, _ = as_kit(question)
        keys = fmt.options(question)[0]
        if label not in keys:
            continue
        k = len(keys)
        for index, order in enumerate(option_orders(k, args.orders)):
            shown = reorder(question, order) if index else question
            text = fmt.render(scorer.tokenizer, r["state"], shown, scorer.codes[:k], scorer.layout)
            items.append((len(meta), scorer.tokenizer.encode(text, add_special_tokens=False), scorer.code_ids[:k], index))
        meta.append({"source": r["source"], "k": k, "gold": keys.index(label),
                     "orders": option_orders(k, args.orders)})
    results = [None] * len(items)
    for group in scorer.batches(items):
        for i, probabilities in zip(group, scorer.forward([items[i] for i in group])):
            results[i] = probabilities
    per_question = defaultdict(dict)
    for (q, _, _, index), probabilities in zip(items, results):
        order = meta[q]["orders"][index]
        per_question[q][index] = (order[max(range(len(order)), key=probabilities.__getitem__)], probabilities, order)
    by_source = defaultdict(lambda: {"n": 0, "flips": 0, "single_correct": 0, "single_n": 0, "avg_correct": 0,
                                     "first": 0, "gold_first": 0, "shown": 0})
    recall = defaultdict(lambda: [0, 0])  # display position of the gold -> [right, seen]
    for q, views in per_question.items():
        m, s = meta[q], by_source[meta[q]["source"]]
        s["n"] += 1
        chosen = {v[0] for v in views.values()}
        s["flips"] += len(chosen) > 1
        averaged = [0.] * m["k"]
        for choice, probabilities, order in views.values():
            s["single_n"] += 1
            s["single_correct"] += choice == m["gold"]
            s["shown"] += 1
            s["first"] += choice == order[0]
            s["gold_first"] += m["gold"] == order[0]
            gold_position = order.index(m["gold"])
            if gold_position < 4:
                recall[gold_position][0] += choice == m["gold"]
                recall[gold_position][1] += 1
            for position, option in enumerate(order):
                averaged[option] += probabilities[position]
        s["avg_correct"] += max(range(m["k"]), key=averaged.__getitem__) == m["gold"]
    sources = {name: {"group": group_of(name), "questions": s["n"], "flip_rate": s["flips"] / s["n"],
                      "accuracy_single_order": s["single_correct"] / s["single_n"],
                      "accuracy_order_averaged": s["avg_correct"] / s["n"],
                      "first_position_rate": s["first"] / s["shown"], "gold_first_rate": s["gold_first"] / s["shown"]}
               for name, s in sorted(by_source.items())}
    recalls = {str(pos + 1): right / seen for pos, (right, seen) in sorted(recall.items()) if seen}
    macro = lambda key: statistics.mean(v[key] for v in sources.values())
    out = {"model": args.model, "split": args.split, "per_source": args.per_source, "orders": args.orders,
           "questions": len(per_question), "sources": len(sources),
           "macro": {k: macro(k) for k in ("flip_rate", "accuracy_single_order", "accuracy_order_averaged",
                                            "first_position_rate", "gold_first_rate")},
           "position_recall": recalls, "position_rstd": statistics.pstdev(recalls.values()) if len(recalls) > 1 else 0.,
           "by_source": sources}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_source"}, indent=1))


if __name__ == "__main__":
    main()
