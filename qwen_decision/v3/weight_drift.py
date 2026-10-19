import json, re, torch
from collections import defaultdict
from safetensors import safe_open
def open_(p): return safe_open(p, "pt", device="cpu")
base = open_("models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors")
d3 = open_("models/d3-lite/model-00001-of-00001.safetensors")
g1 = open_("runs/v3/g1-gold-noncausal-d3/best/model.safetensors")
def norm(k):
    k = re.sub(r"^model\.", "", k); return k
bk = {norm(k): k for k in base.keys() if "language_model" in k}
dk = {norm(k): k for k in d3.keys()}; gk = {norm(k): k for k in g1.keys()}
grp = defaultdict(lambda: {"d3": [0., 0.], "g1": [0., 0.]})
for n, k in bk.items():
    w = base.get_tensor(k).float()
    m = re.search(r"layers\.(\d+)\.", n)
    g = "embed" if "embed_tokens" in n else (f"L{int(m.group(1)):02d}" if m else "other")
    for name, f, keys in (("d3", d3, dk), ("g1", g1, gk)):
        if n not in keys: continue
        v = f.get_tensor(keys[n]).float()
        grp[g][name][0] += (v - w).pow(2).sum().item(); grp[g][name][1] += w.pow(2).sum().item()
out = {g: {m: (a / b) ** .5 for m, (a, b) in v.items() if b} for g, v in sorted(grp.items())}
for g, v in out.items(): print(g, {m: round(x * 100, 2) for m, x in v.items()})
# readout vs base code rows
cfg = json.load(open("runs/v3/g1-gold-noncausal-d3/best/v3_config.json"))
ids = torch.tensor(cfg["token_ids"])
r = safe_open("models/d3-lite/readout.safetensors", "pt", device="cpu")
print("readout keys", list(r.keys()), [r.get_tensor(k).shape for k in r.keys()])
R = r.get_tensor(list(r.keys())[0]).float()
E0 = base.get_tensor(bk["language_model.embed_tokens.weight"]).float()[ids]
Ed = d3.get_tensor(dk["language_model.embed_tokens.weight"]).float()[ids]
Eg = g1.get_tensor(gk["language_model.embed_tokens.weight"]).float()[ids]
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print("code rows: d3 readout vs base", round(rel(R, E0) * 100, 2), "| d3 input emb vs base", round(rel(Ed, E0) * 100, 2), "| g1 tied vs base", round(rel(Eg, E0) * 100, 2))
