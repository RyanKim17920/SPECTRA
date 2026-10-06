#!/usr/bin/env python3
"""Appendix figure ri_vs_step: PathoROB RI against optimisation step, per backbone.

Renders from paper/data/ri_vs_step.json (frozen by paper/scripts/extract_ri_vs_step.py;
needs no /data). The plotting code is scripts/paper_curves.py's figure() verbatim -- that
script now builds the same rows from the raw run directories and calls ``render`` here.

    python3 paper/scripts/ri_vs_step.py [--data PATH]
      -> $SPECTRA_PAPER_FIGURES/ri_vs_step.pdf (+ .png preview)
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_FIGURES  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = REPO / "paper/data/ri_vs_step.json"
OUT = PAPER_FIGURES

INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK, "pdf.fonttype": 42})

# display order matches the paper's tables
ORDER = ["Phikon-v2", "Midnight-12k", "Virchow2", "H-Optimus-0", "UNI2-h",
         "Virchow", "OpenMidnight"]
DISPLAY = {"H-Optimus-0": "H-optimus-0"}


def load(path=DATA):
    """-> (rows, base_ri) in the shapes paper_curves.parse_curves/parse_base_ri return."""
    d = json.loads(Path(path).read_text())
    rows = [(r["backbone"], r["seed"], r["selected_step"],
             [(int(s), float(y)) for s, y in r["points"]]) for r in d["rows"]]
    return rows, d["base_ri"]


def render(rows, base_ri, OUT=OUT):
    present = [b for b in ORDER if any(r[0] == b for r in rows)]
    n = len(present)
    # 2 x 4 rather than 1 x 7: with seven backbones a single row squeezes each panel so far
    # that the tick labels stop being legible.
    ncol = 4
    nrow = -(-n // ncol)
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.25 * nrow), sharex=True)
    axgrid = axgrid.ravel()
    for extra in axgrid[n:]:
        extra.axis("off")
    axes = list(axgrid[:n])

    for ax, bb in zip(axes, present):
        runs = [r for r in rows if r[0] == bb]
        for _bb, seed, sel, pts in runs:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-", color=ACCENT, lw=1.0, alpha=0.85, zorder=2)
            ax.plot(xs, ys, ".", color=ACCENT, ms=2.6, zorder=2)
            # vertical cap: scoring stops here, RI is not measured past this step
            ax.plot([xs[-1]], [ys[-1]], "|", color=ACCENT, ms=5.0, mew=1.2, zorder=3)
            if sel is not None:
                sy = dict(pts).get(sel)
                if sy is not None:
                    ax.plot([sel], [sy], "o", mfc="white", mec=ACCENT, mew=1.4, ms=6.0, zorder=4)
        ax.axvspan(100, 200, color="#e0e7ff", alpha=0.55, lw=0, zorder=0)
        if bb in base_ri:
            ax.axhline(base_ri[bb], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.set_title(DISPLAY.get(bb, bb), fontsize=10)
        ax.grid(color="#e5e7eb", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        ax.set_xticks([0, 250, 500])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # sharex hides tick labels on all but the bottom panel of each column; the last column
    # has no second-row panel, so re-enable its labels or it ends up with a bare axis.
    for i, ax in enumerate(axes):
        if i + ncol >= n:
            ax.tick_params(labelbottom=True, labelsize=8.5)

    fig.supxlabel("optimisation step", fontsize=10)
    fig.supylabel("PathoROB robustness index", fontsize=10)
    h1, = axes[0].plot([], [], "-", color=ACCENT, lw=1.0, label="one adapter seed")
    h2, = axes[0].plot([], [], "o", mfc="white", mec=ACCENT, mew=1.4, ms=6.0, ls="none",
                       label="1-SE selected checkpoint")
    h3 = axes[0].axvspan(0, 0, color="#e0e7ff", alpha=0.55, lw=0, label="steps 100--200")
    h5, = axes[0].plot([], [], "|", color=ACCENT, ms=5.0, mew=1.2, ls="none", label="last scored step")
    h4, = axes[0].plot([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="base (untuned)")
    fig.legend(handles=[h1, h2, h5, h4, h3], loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.99])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ri_vs_step.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ri_vs_step.png", dpi=150, bbox_inches="tight")
    for bb in present:
        runs = [r for r in rows if r[0] == bb]
        sels = [r[2] for r in runs if r[2] is not None]
        print(f"{bb:14s} curves={len(runs)} selected={sels}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    render(*load(ap.parse_args().data))
