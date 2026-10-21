"""Figures for the crystal-0.8b model card: assets/logo.png and assets/public_areas.png."""
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

# ------------------------------------------------------------------ public-suite areas
a = json.load(open(ROOT / "runs/board/full-wise80-g6-g7e2/index.json"))
b = json.load(open(ROOT / "runs/board-d3-lite/index.json"))
labels = ["Knowledge &\nReasoning", "Language\nUnderstanding", "Retrieval &\nClassification", "Tools &\nAutomation",
          "Arts &\nHuman Taste", "Public index"]
ours = [100 * x["skill"] for x in a["areas"]] + [a["index"]]
theirs = [100 * x["skill"] for x in b["areas"]] + [b["index"]]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
fig.patch.set_facecolor("white")
xs = range(len(labels))
w = 0.38
bars1 = ax.bar([x - w / 2 for x in xs], ours, w, color="#6366f1", label="crystal-0.8b", zorder=3)
bars2 = ax.bar([x + w / 2 for x in xs], theirs, w, color="#cbd5e1", label="d3-lite (official run)", zorder=3)
for bars, color in ((bars1, "#312e81"), (bars2, "#475569")):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}", ha="center", va="bottom",
                fontsize=9.5, color=color, fontweight="bold" if color == "#312e81" else "normal")
ax.axvline(4.5, color="#e2e8f0", linewidth=1.2, zorder=1)
ax.set_xticks(list(xs), labels, fontsize=10)
ax.set_ylabel("chance-corrected skill (0-100)", fontsize=10, color="#334155")
ax.set_ylim(0, 75)
ax.grid(axis="y", color="#f1f5f9", zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0, labelcolor="#64748b")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.set_title("Decision Index 0.3 public suite, all 140k requests", fontsize=12, color="#0f172a", loc="left", pad=12)
fig.tight_layout()
fig.savefig(OUT / "public_areas.png")
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
