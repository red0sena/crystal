"""Figures for the crystal-0.8b model card: assets/logo.png and assets/board_0.8b.png.

    python make_figures.py OUT [--board BOARD_DIR]

--board rebuilds OUT/board_0.8b.json from the leaderboard's data files (BOARD_DIR/index.json and v03.json, from the
multimodalart/jev-decision-index Space) and our own public-kit run; without it the saved snapshot is reused.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

OUT = Path(sys.argv[1])
ROOT = Path("/home/khkim/qwen-decision")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"


def snapshot(board):
    """Every board model built on Qwen3.5-0.8B (or -Base), plus crystal-0.8b from our run of the public kit."""
    index = json.load(open(board / "index.json"))
    full = {m["engine"]: m for m in json.load(open(board / "v03.json"))["models"]}

    def entry(name, url, public, areas, benchmarks, x):
        return dict(name=name, url=url, public=public, full=x.get("v"), same_skills=x.get("same_skills"),
                    new_domains=x.get("new_domains"), areas={a["id"]: round(100 * a["skill"], 2) for a in areas},
                    raw={k: b["raw"] for k, b in benchmarks.items()})

    models = [entry(m["name"], m["meta"].get("model_url"), full[m["engine"]]["pub"], m["categories"], m["benchmarks"],
                    full[m["engine"]])
              for m in index["models"] if (m["meta"].get("base_model") or "").startswith("Qwen/Qwen3.5-0.8B")]
    run = json.load(open(ROOT / "runs/board/full-wise80-g6-g7e2/index.json"))
    models.append(entry("crystal-0.8b", "https://huggingface.co/Kyunghwan-kim/crystal-0.8b", run["index"], run["areas"],
                        run["benchmarks"], {}))
    models.sort(key=lambda m: -m["public"])
    return dict(board=index["suite"]["label"], generated_utc=index["generated_utc"], areas=index["suite"]["areas"],
                benchmarks={k: dict(name=b["short"], metric=b["metric"]) for k, b in index["benchmarks"].items()},
                models=models)


if "--board" in sys.argv:
    json.dump(snapshot(Path(sys.argv[sys.argv.index("--board") + 1])), open(OUT / "board_0.8b.json", "w"), indent=1)
SNAP = json.load(open(OUT / "board_0.8b.json"))

# ------------------------------------------------------------------ logo
fig = plt.figure(figsize=(8, 3), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 8)
ax.set_ylim(0, 3)
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.08, 0.08), 7.84, 2.84, boxstyle="round,pad=0,rounding_size=0.35",
                            facecolor="#0b1020", edgecolor="#1e293b", linewidth=1.5))
# brilliant-cut gem: table, crown facets, pavilion facets
cx, top, girdle, tip = 1.75, 2.35, 1.75, 0.55
tw, gw = 0.55, 1.05
crown = [
    ([(cx - tw, top), (cx + tw, top), (cx, girdle + 0.12)], "#e0f2fe"),
    ([(cx - tw, top), (cx - gw, girdle), (cx - 0.5, girdle)], "#7dd3fc"),
    ([(cx - tw, top), (cx - 0.5, girdle), (cx, girdle + 0.12)], "#bae6fd"),
    ([(cx + tw, top), (cx + gw, girdle), (cx + 0.5, girdle)], "#a5b4fc"),
    ([(cx + tw, top), (cx + 0.5, girdle), (cx, girdle + 0.12)], "#c7d2fe"),
    ([(cx - 0.5, girdle), (cx, girdle + 0.12), (cx + 0.5, girdle)], "#93c5fd"),
]
pavilion = [
    ([(cx - gw, girdle), (cx - 0.5, girdle), (cx, tip)], "#38bdf8"),
    ([(cx - 0.5, girdle), (cx, girdle), (cx, tip)], "#0ea5e9"),
    ([(cx, girdle), (cx + 0.5, girdle), (cx, tip)], "#6366f1"),
    ([(cx + 0.5, girdle), (cx + gw, girdle), (cx, tip)], "#8b5cf6"),
]
for pts, color in crown + pavilion:
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="#0b1020", linewidth=1.2, joinstyle="round"))
for x, y, s in [(cx + 0.95, 2.55, 90), (cx - 1.05, 2.45, 40), (cx + 1.15, 0.95, 30)]:
    ax.scatter([x], [y], marker="*", s=s, color="#e0f2fe", zorder=5)
ax.text(3.35, 1.72, "crystal", color="white", fontsize=50, fontweight="bold", va="center")
ax.text(6.42, 1.62, "0.8b", color="#7dd3fc", fontsize=26, fontweight="bold", va="center")
ax.text(3.38, 0.88, "typed decisions  ·  one forward pass per question", color="#94a3b8", fontsize=13, va="center")
fig.savefig(OUT / "logo.png", transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ public index, top Qwen3.5-0.8B models
ACCENT, OTHER = "#6366f1", "#cbd5e1"
TOP = 10
models = SNAP["models"][:TOP]
fig, ax = plt.subplots(figsize=(9, 0.34 * len(models) + 1.3), dpi=200)
fig.patch.set_facecolor("white")
ys = range(len(models))
ours = [m["name"] == "crystal-0.8b" for m in models]
ax.barh(list(ys), [m["public"] for m in models], height=0.68, color=[ACCENT if o else OTHER for o in ours], zorder=3)
for y, m, o in zip(ys, models, ours):
    ax.text(m["public"] + 0.6, y, f"{m['public']:.2f}", va="center", fontsize=9, color="#312e81" if o else "#475569",
            fontweight="bold" if o else "normal")
ax.set_yticks(list(ys), [m["name"] for m in models], fontsize=9.5)
for label, o in zip(ax.get_yticklabels(), ours):
    label.set_color("#312e81" if o else "#334155")
    label.set_fontweight("bold" if o else "normal")
ax.invert_yaxis()
ax.set_xlim(0, 60)
ax.grid(axis="x", color="#f1f5f9", zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#cbd5e1")
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelcolor="#64748b", labelsize=9)
ax.set_xlabel("public index (0 = random, 100 = perfect)", fontsize=9.5, color="#334155")
ax.set_title(f"Decision Index public suite: top {TOP} models built on Qwen3.5-0.8B", fontsize=11.5,
             color="#0f172a", loc="left", pad=10)
fig.tight_layout()
fig.savefig(OUT / "board_0.8b.png")
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
