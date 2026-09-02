#!/usr/bin/env python3
"""Two figures for the workshop paper: the grid-batch method diagram and base->tuned dots.

    ./.venv-hest/bin/python scripts/paper_figures.py
      -> waiv-asci/figures/{grid_batch,base_to_tuned}.pdf (+ .png previews, same stems)
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path("/admin/home/ryan.kim/waiv-asci/figures")
INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK, "pdf.fonttype": 42, "hatch.linewidth": 0.6})


def grid_batch():
    # rows: c_a (query), c_b (candidates), one greyed extra condition; columns: i_1..i_4, ..., i_T
    cols = ["$i_1$", "$i_2$", "$i_3$", "$i_4$", r"$\cdots$", "$i_T$"]
    T = len(cols); gap = 4                      # column index of the ellipsis
    fig, ax = plt.subplots(figsize=(5.5, 2.35))
    ax.set_xlim(-2.9, T + 3.4); ax.set_ylim(-0.75, 3.35); ax.axis("off"); ax.set_aspect("equal")
    ya, yb, yc = 0, 1, 2                      # y of query row, candidate row, other row
    qi = 2
    for t in range(T):
        if t == gap:
            for y in (ya, yb, yc):
                ax.text(t + 0.46, y + 0.42, r"$\cdots$", ha="center", va="center", fontsize=10, color=MUTED)
            continue
        ax.add_patch(Rectangle((t, yc), 0.92, 0.92, fc="white", ec=MUTED, lw=0.8, alpha=0.4, zorder=1))
        ax.add_patch(Rectangle((t, ya), 0.92, 0.92, fc="white", ec=MUTED, lw=0.8, zorder=1))
        if t == qi:
            ax.add_patch(Rectangle((t, yb), 0.92, 0.92, fc="#e0e7ff", ec=ACCENT, lw=2.2, zorder=2))
        else:
            ax.add_patch(Rectangle((t, yb), 0.92, 0.92, fc="white", ec=MUTED, lw=0.8, hatch="///", zorder=1))
        ax.text(t + 0.46, yc + 1.0, cols[t], ha="center", va="bottom", fontsize=8.5)
    ax.add_patch(Rectangle((-0.08, yb - 0.08), T + 0.08, 1.08, fc="none", ec=INK, lw=1.1, ls="--", zorder=3))
    ax.add_patch(Rectangle((qi, ya), 0.92, 0.92, fc="#e0e7ff", ec=ACCENT, lw=2.2, zorder=3))
    ax.text(qi + 0.46, ya + 0.44, "query", ha="center", va="center", color=ACCENT, fontsize=6.8, weight="bold")
    ax.text(qi + 0.46, yb + 0.44, "pos.", ha="center", va="center", color=ACCENT, fontsize=7.2, weight="bold")
    ax.text(-0.25, yc + 0.46, "other sampled\nconditions", ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.text(-0.25, yb + 0.46, "$c_b$  candidate row", ha="right", va="center", fontsize=8.5)
    ax.text(-0.25, ya + 0.46, "$c_a$  query row", ha="right", va="center", fontsize=8.5)
    ax.text(T + 0.25, yb + 0.46, "one softmax:\n1 positive, $T-1$ negatives (hatched),\nall under condition $c_b$",
            ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(T + 0.25, ya + 0.46, "query and positive:\nsame registered location (shaded)", ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(T / 2, yc + 1.55, "columns: registered tissue locations", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(-0.25, ya - 0.2, "rows: acquisition conditions (scanner, stain)", ha="right", va="top", fontsize=7.5, color=INK)
    fig.savefig(OUT / "grid_batch.pdf", bbox_inches="tight")
    fig.savefig(OUT / "grid_batch.png", dpi=150, bbox_inches="tight")


def base_to_tuned():
    ss = Path("/admin/home/ryan.kim/waiv/docs/seed_stats.md").read_text()
    name = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
            "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h"}
    vals, bb = {}, None
    for l in ss.splitlines():
        m = re.match(r"## (\w+)", l)
        if m: bb = m.group(1); continue
        m = re.match(r"\| (PathoROB RI|HEST) \| ([\d.]+) \| ([\d.]+) \+/- ([\d.]+) \(n=(\d+)\)", l)
        if m and bb in name:
            vals[(bb, m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)))
    assert len(vals) == 10, sorted(vals)
    order = sorted(name, key=lambda b: vals[(b, "PathoROB RI")][0])        # ascending base RI
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    spans = {"PathoROB RI": (0.40, 1.0), "HEST": (0.36, 0.45)}
    for ax, metric, xlabel in zip(axes, ["PathoROB RI", "HEST"],
                                  ["PathoROB robustness index", "HEST mean Pearson $r$"]):
        lo, hi = spans[metric]
        for k, bb in enumerate(order):
            b, t, sd, n = vals[(bb, metric)]
            y = len(order) - 1 - k
            ax.plot([b, t], [y, y], color=MUTED, lw=1.1, zorder=1)
            ax.scatter([b], [y], s=16, fc="white", ec=INK, lw=0.9, zorder=3)
            ax.errorbar([t], [y], xerr=[sd], fmt="o", ms=3.6, color=ACCENT, ecolor=ACCENT,
                        elinewidth=0.9, capsize=1.5, zorder=4)
            ax.text(max(b, t) + 0.018 * (hi - lo), y, f"{t - b:+.3f} ($n$={n})", ha="left", va="center",
                    fontsize=7.2, color=INK)
        ax.set_xlim(lo, hi + (0.14 if metric == "PathoROB RI" else 0.02))
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if metric == "PathoROB RI" else [0.36, 0.38, 0.40, 0.42, 0.44])
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.grid(axis="x", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[1].set_title("x-axis expanded $\\approx$4$\\times$ relative to the left panel", fontsize=7, color=MUTED, loc="left")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([name[bb] for bb in reversed(order)])
    h1 = axes[0].scatter([], [], s=16, fc="white", ec=INK, label="base (single evaluation)")
    h2 = axes[0].scatter([], [], s=16, color=ACCENT, label="fine-tuned, mean $\\pm$ sample SD over $n$ adapter seeds")
    fig.legend([h1, h2], [h.get_label() for h in (h1, h2)], loc="upper center", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.84, wspace=0.12)
    fig.savefig(OUT / "base_to_tuned.pdf", bbox_inches="tight")
    fig.savefig(OUT / "base_to_tuned.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    grid_batch(); base_to_tuned(); print("wrote", sorted(p.name for p in OUT.iterdir()))
