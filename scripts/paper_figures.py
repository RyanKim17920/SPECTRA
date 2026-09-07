#!/usr/bin/env python3
"""Two figures for the workshop paper: the grid-batch method diagram and base->tuned dots.

    ./.venv-hest/bin/python scripts/paper_figures.py
      -> waiv-asci/figures/{grid_batch,base_to_tuned}.pdf (+ .png previews, same stems)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import PAPER_FIGURES, PLISM_PACKED, REPO  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = PAPER_FIGURES
INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK, "pdf.fonttype": 42, "hatch.linewidth": 0.6})


PLISM = PLISM_PACKED
# Three acquisition conditions of one PLISM tissue-microarray design. Rows differ in
# scanner and/or stain; columns are registered tissue locations, so column i is the SAME
# physical location in every row. Locations chosen for tissue content (>0.9 non-white).
CONDITIONS = [("GMH_S210_to_GMH_S60",  "$c_a$  query row\n(scanner S210)"),
              ("GMH_S360_to_GMH_S60",  "$c_b$  candidate row\n(scanner S360)"),
              ("KRH_GT450_to_GMH_S60", "other condition\n(different stain)")]
LOCATIONS = [3123, 6940, 8675, 9022, 5205, 6246]
QUERY_COL = 2                                   # which column is the query / positive


def grid_batch():
    """Figure 1 drawn from REAL PLISM tiles rather than coloured boxes.

    The point the schematic could not make: the positive and its negatives are visually
    alike because they share one acquisition condition, while the query differs from its
    own positive in stain and scanner. That is the whole reason acquisition cannot be used
    to find the match. A previous box-and-hatch version is kept at
    figures/grid_batch.schematic.bak-*.{pdf,png}.
    """
    import numpy as np
    arrs = [np.load(PLISM / f"{c}.npy", mmap_mode="r") for c, _ in CONDITIONS]
    nrow, ncol = len(CONDITIONS), len(LOCATIONS)

    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 3.15),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for r, (arr, (_code, label)) in enumerate(zip(arrs, CONDITIONS)):
        for c, loc in enumerate(LOCATIONS):
            ax = axes[r, c]
            ax.imshow(np.asarray(arr[loc]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_edgecolor(MUTED)
            if r == 2:                                  # third condition: lighter spine only.
                # NO alpha/dimming -- this row really is lighter (mean brightness 205 vs
                # 163 and 150), and faking it would misrepresent the stain difference.
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d1d5db")
            if r == 0 and c == QUERY_COL:               # query
                for sp in ax.spines.values():
                    sp.set_linewidth(2.4); sp.set_edgecolor(ACCENT)
            if r == 1:                                  # candidate row
                if c == QUERY_COL:                      # the positive
                    for sp in ax.spines.values():
                        sp.set_linewidth(2.4); sp.set_edgecolor(ACCENT)
                else:                                   # negatives
                    for sp in ax.spines.values():
                        sp.set_linewidth(1.2); sp.set_edgecolor(INK)
            if r == 0:
                ax.set_title(f"$i_{{{c + 1}}}$" if c != ncol - 1 else "$i_T$",
                             fontsize=7.5, pad=2)
        axes[r, 0].set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center",
                              labelpad=6, color=INK if r < 2 else MUTED)

    # annotations
    axes[0, QUERY_COL].text(0.5, 0.5, "query", transform=axes[0, QUERY_COL].transAxes,
                            ha="center", va="center", fontsize=7, weight="bold",
                            color="white",
                            bbox=dict(fc=ACCENT, ec="none", pad=1.4, alpha=0.9))
    axes[1, QUERY_COL].text(0.5, 0.5, "positive", transform=axes[1, QUERY_COL].transAxes,
                            ha="center", va="center", fontsize=7, weight="bold",
                            color="white",
                            bbox=dict(fc=ACCENT, ec="none", pad=1.4, alpha=0.9))
    for c in range(ncol):
        if c != QUERY_COL:
            axes[1, c].text(0.5, 0.06, "neg.", transform=axes[1, c].transAxes,
                            ha="center", va="bottom", fontsize=6.2, color="white",
                            bbox=dict(fc=INK, ec="none", pad=0.9, alpha=0.75))

    fig.text(0.5, 1.03, "columns: corresponding registered locations",
             ha="center", va="bottom", fontsize=7.5, color=INK)
    fig.savefig(OUT / "grid_batch.pdf", bbox_inches="tight")
    # 1050 dpi (was 700, originally 200): this PNG is not just a preview --
    # annotate_grid_batch.py composites it as the left panel of the final figure, so its
    # pixel count sets the embedded raster resolution in grid_batch.pdf. 200 dpi here
    # capped the print figure at 100 ppi; 700 dpi supported a 400 ppi embed. The author
    # asked for 600 ppi, so this scales proportionally (700 * 600/400 = 1050) and keeps
    # the source comfortably above the final embed resolution.
    # The PLISM tiles themselves are 224x224 native and are drawn undownsampled.
    fig.savefig(OUT / "grid_batch.png", dpi=1050, bbox_inches="tight")


def base_to_tuned():
    ss = (REPO / "docs/seed_stats.md").read_text()
    name = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
            "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
            "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
    vals, bb = {}, None
    for l in ss.splitlines():
        m = re.match(r"## (\w+)", l)
        if m: bb = m.group(1); continue
        m = re.match(r"\| (PathoROB RI|HEST) \| ([\d.]+) \| ([\d.]+) \+/- ([\d.]+) \(n=(\d+)\)", l)
        if m and bb in name:
            vals[(bb, m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)))
    assert len(vals) == 2 * len(name), sorted(vals)
    order = sorted(name, key=lambda b: vals[(b, "PathoROB RI")][0])        # ascending base RI
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
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
                        elinewidth=1.1, capsize=2.2, zorder=4)
            # The SD bars are smaller than the marker at this scale (RI SD 0.002-0.007,
            # HEST SD 0.0003-0.0029), so the SD is also printed: a bar the reader cannot
            # see is not an error bar.
            dp = 4 if metric == "HEST" else 3
            ax.text(max(b, t) + 0.03 * (hi - lo), y,
                    f"{t - b:+.{dp}f}$\\,\\pm\\,${sd:.{dp}f} ($n$={n})",
                    ha="left", va="center", fontsize=6.6, color=INK)
        ax.set_xlim(lo, hi + (0.30 if metric == "PathoROB RI" else 0.036))
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
    # base_to_tuned() is NO LONGER RUN from here. figures/base_to_tuned.pdf is owned by
    # waiv-asci/scripts/make_base_to_tuned.py, which draws all FOUR benchmarks (PathoROB,
    # HEST, CPTAC, and the six THUNDER tasks) by parsing the generated tables. The two-panel
    # version below is superseded; running it overwrites the four-panel figure with a
    # strictly worse one, which is exactly what happened once. Kept for reference only.
    grid_batch(); print("wrote grid_batch (left panel);"
                        " now run waiv-asci/scripts/annotate_grid_batch.py to composite")
