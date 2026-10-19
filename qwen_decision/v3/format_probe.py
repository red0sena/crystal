# python -m qwen_decision.v3.format_probe <checkpoint dir>  (2026-10-18 diagnostic, JEVal v2 dev)
"""Diagnostic only: JEVal same-source choice questions scored as given (placeholder keys + descriptions) and
with the descriptions moved into the keys (our training layout). Same rows, same model."""
import json, sys
from collections import defaultdict
from qwen_decision.v3.engine import Scorer
from qwen_decision.v3.proxy_eval import load_rows, as_kit
model = sys.argv[1]
sc = Scorer(model, attention="noncausal")
rows = load_rows(60, split="dev")
res = defaultdict(lambda: [0, 0, 0])
placeholder_share = defaultdict(lambda: [0, 0])
for r in rows:
    key, q = next(iter(json.loads(r["questions_json"]).items()))
    if q["type"] != "choice": continue
    lab = json.loads(r["label_json"]); lab = str(lab[key] if isinstance(lab, dict) else lab)
    q, _ = as_kit(q); crit = q["criteria"]
    if lab not in crit: continue
    descs = [v for v in crit.values()]
    placeholder_share[r["source"]][1] += 1
    if not all(isinstance(v, str) and v for v in descs) or len(set(descs)) != len(descs): continue
    placeholder_share[r["source"]][0] += 1
    keys = list(crit)
    flipped = {**q, "criteria": {v: None for v in descs}}
    p1 = sc.score(r["state"], {"q": q})["q"]; p2 = sc.score(r["state"], {"q": flipped})["q"]
    g = keys.index(lab)
    s = res[r["source"]]; s[0] += 1; s[1] += max(range(len(p1)), key=p1.__getitem__) == g; s[2] += max(range(len(p2)), key=p2.__getitem__) == g
tot = [0, 0, 0]
for src, (n, a, b) in sorted(res.items()):
    tot = [x + y for x, y in zip(tot, (n, a, b))]
    print(f"{src:26s} n {n:3d}  as given {a/n:.3f}  content-as-key {b/n:.3f}  diff {(b-a)/n:+.3f}")
print("ALL", tot[0], round(tot[1]/tot[0], 3), round(tot[2]/tot[0], 3))
