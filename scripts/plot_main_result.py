"""Plot COMPACT main result on JAMEL test10 (avg reward per app).

Usage: python3 scripts/plot_main_result.py
Output: docs/compact_v2/fig_main_result.png
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# CJK font (Noto Sans CJK JP covers simplified Chinese glyphs)
for f in font_manager.fontManager.ttflist:
    if "Noto Sans CJK" in f.name:
        plt.rcParams["font.family"] = f.name
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------- data (from docs/compact_v2/deck_outline.md 页9) ----------------
models = ["Qwen3-VL-2B", "Qwen3-VL-4B"]
params_b = ["2.13B", "4.02B"]
baseline = [15.9, 15.8]
compact = [18.0, 19.3]
overhead = ["+6.24M (+0.29%)", "+9.65M (+0.24%)"]  # 4B value derived from +0.24%

C_BASE = "#9AA5B1"  # grey
C_COMP = "#2A9D8F"  # teal
C_DELTA = "#E76F51"  # orange

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)

x = np.arange(len(models))
w = 0.32

b1 = ax.bar(x - w / 2, baseline, w, label="Baseline (JAMEL)", color=C_BASE,
            edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + w / 2, compact, w, label="COMPACT (ours)", color=C_COMP,
            edgecolor="white", linewidth=0.5)

# value labels
for rect, v in zip(b1, baseline):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.25, f"{v:.1f}",
            ha="center", va="bottom", fontsize=11, color="#5b6470")
for rect, v, oh in zip(b2, compact, overhead):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.25, f"{v:.1f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1d7268")
    ax.text(rect.get_x() + rect.get_width() / 2, v - 1.6, f"参数 {oh}",
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

# delta arrows
for i in range(len(models)):
    y0, y1 = baseline[i], compact[i]
    ax.annotate("", xy=(x[i] + w / 2 - 0.02, y1 + 0.05), xytext=(x[i] - w / 2 + 0.02, y0 + 0.05),
                arrowprops=dict(arrowstyle="->", color=C_DELTA, lw=1.6,
                                connectionstyle="arc3,rad=-0.25"))
    d = y1 - y0
    ax.text(x[i], max(y0, y1) + 1.15, f"+{d:.1f} ({d / y0 * 100:+.0f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_DELTA)

ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n({p} 参数)" for m, p in zip(models, params_b)], fontsize=11)
ax.set_ylabel("test10 平均每应用奖励 ↑", fontsize=12)
ax.set_ylim(0, 23)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=10)

fig.tight_layout()
out = "docs/compact_v2/fig_main_result.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
