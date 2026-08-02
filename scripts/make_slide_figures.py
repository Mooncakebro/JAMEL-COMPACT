#!/usr/bin/env python
"""
Generate CVPR-style figures for COMPACT slide deck.

Outputs (docs/compact_v2/):
  1. fig_main_results.{pdf,png}  — bar chart: COMPACT vs baselines + JAMEL refs
  2. fig_ablation.{pdf,png}     — ablation study as styled table
  3. fig_per_app.{pdf,png}      — per-app grouped bar chart (2B models)

Data source: docs/compact_v2/deck_outline.md §实验数据（硬数据）
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, Patch

# ── CVPR style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,    # editable text in PDF
    "ps.fonttype":        42,
    "axes.linewidth":     0.8,
})

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "compact_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_BASE   = "#7B9EA8"    # muted teal-grey  (baseline)
C_COMPACT = "#E8743B"   # warm orange       (ours)
C_REF    = "#B0B0B0"    # light grey        (reference baselines from JAMEL)
C_ACCENT = "#C0392B"    # red               (delta annotations)

# ═════════════════════════════════════════════════════════════════════════════
#  Hard data (from deck_outline.md §实验数据（硬数据）)
# ═════════════════════════════════════════════════════════════════════════════

# Main results — our experiments
OUR_MODELS = ["Baseline-2B", "Baseline-4B", "COMPACT-2B", "COMPACT-4B"]
OUR_SCORES = [15.9, 15.8, 18.0, 19.3]
OUR_PARAMS = ["2.13B", "4.02B", "2.13B\n+6.24M\n(+0.29%)", "4.02B\n+9.79M\n(+0.24%)"]
OUR_IS_OURS = [False, False, True, True]

# Reference baselines from JAMEL paper (horizontal lines / annotations)
REF_ENTRIES = [
    ("JAMEL-9B",            20.7, "9B (separate model)"),
    ("ReAct-vision\n(Gemini 3.1 Flash-Lite)", 20.9, "closed-source"),
    ("MAI-UI-8B",            8.4, "8B"),
]

# Ablation (2B)
ABLATION_ROWS = [
    ("Baseline-2B\n(Qwen3-VL-2B SFT)",       15.9, False),
    ("COMPACT-2B\nw/o memory writing",        16.8, False),
    ("COMPACT-2B\nw/o auxiliary losses",       14.5, False),
    ("COMPACT-2B (full)",                     18.0, True),
]

# Per-app (synthetic — replace with real data when available)
TEST10_APPS = [
    "vipshop", "alibaba", "expedia", "taobao", "pinduoduo",
    "dongchedi", "youku", "keep", "meituan", "temu",
]
PER_APP_BASELINE_2B = np.array([14.8, 17.2, 13.5, 16.1, 15.3,
                                 18.0, 14.2, 16.9, 15.5, 17.5])
PER_APP_COMPACT_2B  = np.array([16.9, 19.1, 15.3, 18.2, 17.0,
                                 20.1, 16.5, 19.0, 17.8, 20.1])


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Main Results
# ═════════════════════════════════════════════════════════════════════════════

def make_main_results():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    n = len(OUR_MODELS)
    x = np.arange(n)
    bar_w = 0.52

    colors = [C_COMPACT if o else C_BASE for o in OUR_IS_OURS]
    bars = ax.bar(x, OUR_SCORES, width=bar_w, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Hatch baselines
    for i, is_ours in enumerate(OUR_IS_OURS):
        if not is_ours:
            bars[i].set_hatch("///")
            bars[i].set_edgecolor("#555555")

    # Value labels
    for i, (bar, sc) in enumerate(zip(bars, OUR_SCORES)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{sc:.1f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222")

    # Parameter labels below
    for i, (bar, pl) in enumerate(zip(bars, OUR_PARAMS)):
        ax.text(bar.get_x() + bar.get_width()/2, -0.7,
                pl, ha="center", va="top", fontsize=7.5,
                color="#555", linespacing=1.15)

    # Delta arrows: COMPACT vs same-size baseline
    for compact_idx, base_idx in [(2, 0), (3, 1)]:
        delta = OUR_SCORES[compact_idx] - OUR_SCORES[base_idx]
        y_lo, y_hi = OUR_SCORES[base_idx] + 0.4, OUR_SCORES[compact_idx] - 0.4
        ax.annotate("", xy=(compact_idx, y_hi), xytext=(compact_idx, y_lo),
                    arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.6,
                                    connectionstyle="bar,fraction=0.12"))
        ax.text(compact_idx + 0.42, (y_lo + y_hi)/2,
                f"+{delta:.1f}", ha="left", va="center",
                fontsize=9, fontweight="bold", color=C_ACCENT)

    # Reference baselines (dashed horizontal lines)
    for name, score, param_str in REF_ENTRIES:
        ax.axhline(y=score, color=C_REF, linestyle="--", lw=1.0,
                   alpha=0.7, zorder=2)
        # Label at right edge
        ax.text(n - 0.32, score + 0.15, f"{name}  ({score:.1f})",
                ha="right", va="bottom", fontsize=7.5,
                color="#666", style="italic")

    # Axes
    ax.set_xticks(x)
    short_labels = [m.replace("-", "\n") for m in OUR_MODELS]
    ax.set_xticklabels(short_labels, fontsize=9.5)
    ax.set_ylabel("Avg. Cumulative Reward (test10)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 24)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor=C_BASE, hatch="///", edgecolor="#555555",
              label="Baseline (Qwen3-VL SFT)"),
        Patch(facecolor=C_COMPACT, edgecolor="white", label="COMPACT (ours)"),
        plt.Line2D([0], [0], color=C_REF, linestyle="--", lw=1.0,
                   label="Reference (JAMEL paper)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              framealpha=0.92, edgecolor="#ccc", fontsize=8.5)

    ax.set_title("Main Results — ScaleWoB test10 (Coverage Reward)",
                 fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_main_results.{ext}")
    plt.close(fig)
    print(f"[fig] Saved fig_main_results -> {OUT_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Ablation Table
# ═════════════════════════════════════════════════════════════════════════════

def make_ablation_table():
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.axis("off")

    col_labels = ["Configuration", "Avg. Reward\n(test10)"]
    cell_text = [(row[0], f"{row[1]:.1f}") for row in ABLATION_ROWS]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=[0.60, 0.40],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    # Header
    for j in range(2):
        c = table[0, j]
        c.set_facecolor("#2C3E50")
        c.set_text_props(color="white", fontweight="bold", fontsize=10)
        c.set_edgecolor("#1a1a2e")

    # Body
    for i, (_, score, is_full) in enumerate(ABLATION_ROWS, start=1):
        for j in range(2):
            c = table[i, j]
            c.set_edgecolor("#d0d0d0")
            if is_full:
                c.set_facecolor("#FFF3E0")
                c.set_text_props(fontweight="bold", fontsize=10.5,
                                 color="#E65100")
            else:
                c.set_facecolor("#F8F9FA" if i % 2 == 0 else "white")

    # Delta annotations (relative to Baseline-2B)
    base_score = ABLATION_ROWS[0][1]
    for i, (_, score, _) in enumerate(ABLATION_ROWS, start=1):
        delta = score - base_score
        if abs(delta) < 0.01:
            continue
        sign = "+" if delta > 0 else ""
        c = table[i, 1]
        old_text = c.get_text().get_text()
        c.get_text().set_text(f"{old_text}  ({sign}{delta:.1f})")

    ax.set_title("Ablation Study — COMPACT-2B", fontsize=12,
                 fontweight="bold", pad=16)

    fig.text(0.5, 0.02, "Backbone: Qwen3-VL-2B-Instruct  |  "
             "Deltas relative to Baseline-2B",
             ha="center", fontsize=7.5, color="#777", style="italic")

    fig.tight_layout(pad=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_ablation.{ext}")
    plt.close(fig)
    print(f"[fig] Saved fig_ablation -> {OUT_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Per-App Breakdown
# ═════════════════════════════════════════════════════════════════════════════

def make_per_app_breakdown():
    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    x = np.arange(len(TEST10_APPS))
    w = 0.35

    bars_b = ax.bar(x - w/2, PER_APP_BASELINE_2B, w,
                    color=C_BASE, hatch="///", edgecolor="#555",
                    linewidth=0.5, label="Baseline-2B", zorder=3)
    bars_c = ax.bar(x + w/2, PER_APP_COMPACT_2B, w,
                    color=C_COMPACT, edgecolor="white",
                    linewidth=0.5, label="COMPACT-2B", zorder=3)

    # Mean lines
    ax.axhline(y=PER_APP_BASELINE_2B.mean(), color=C_BASE,
               linestyle=":", lw=1.0, alpha=0.6, zorder=2)
    ax.axhline(y=PER_APP_COMPACT_2B.mean(), color=C_COMPACT,
               linestyle=":", lw=1.0, alpha=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(TEST10_APPS, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Cumulative Reward", fontsize=10)
    ax.set_ylim(0, 24)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-App Breakdown — test10 (2B models, synthetic)",
                 fontsize=11, fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_per_app.{ext}")
    plt.close(fig)
    print(f"[fig] Saved fig_per_app -> {OUT_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_main_results()
    make_ablation_table()
    make_per_app_breakdown()
    print("\nDone. All figures saved to", OUT_DIR)
