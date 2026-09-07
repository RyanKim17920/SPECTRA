#!/usr/bin/env python3
"""RI-vs-step figure for the paper: shows the checkpoint-selection rule in action.

Reads runs/<RUN>/ri_curve.json for exactly the cells the paper's tables use (resolved via
pathorob_submetrics.cells_for) and plots RI against optimisation step, marking each run's
selected checkpoint. All 3 adapter seeds per backbone are plotted, matching the table n.

    ./.venv/bin/python scripts/paper_curves.py
      -> waiv-asci/figures/ri_vs_step.pdf (+ .png preview)

The point of the figure: every selected checkpoint lands in 100--200 steps, well before
the 500-step schedule ends, and RI is flat-to-decaying afterwards. That claim is made in
prose in the paper (Sec. checkpoint selection) with no supporting exhibit.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CELLS, PAPER_FIGURES, RUNS  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = RUNS
SEED_STATS = REPO / "docs/seed_stats.md"
OUT = PAPER_FIGURES

# seed_stats.md section name -> scoreboard backbone label
SLUG = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-Optimus-0", "uni2h": "UNI2-h", "virchow1": "Virchow",
        "openmidnightsq": "OpenMidnight"}

INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK, "pdf.fonttype": 42})

# display order matches the paper's tables
ORDER = ["Phikon-v2", "Midnight-12k", "Virchow2", "H-Optimus-0", "UNI2-h",
         "Virchow", "OpenMidnight"]
DISPLAY = {"H-Optimus-0": "H-optimus-0"}


def parse_curves():
    """-> list of (backbone, seed, selected_step, [(step, avg RI), ...])

    Reads runs/<RUN>/ri_curve.json for EXACTLY the cells the paper's tables use, resolved
    through pathorob_submetrics.cells_for.  An earlier version parsed the curve strings out
    of docs/final_scoreboard.md, which is a stale snapshot: it records "no curve" for runs
    whose ri_curve.json has since landed, so it under-reported the seed count per panel
    (Midnight-12k showed 1 curve when all 3 seeds have curves on disk).
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pathorob_submetrics as pm

    rows = []
    for slug, label in SLUG.items():
        _base, seed_cells = pm.cells_for(slug)
        for cell in seed_cells:
            text = (CELLS / cell / "model.py").read_text()
            run = re.search(r'^RUN = "(.*)"$', text, re.M)
            step = re.search(r'^STEP = "(.*)"$', text, re.M)
            if not (run and step):
                continue
            sel = int(step.group(1).replace("step_", ""))
            cur = RUNS_ROOT / run.group(1) / "ri_curve.json"
            if not cur.exists():
                print(f"  no ri_curve.json for {cell} ({run.group(1)})")
                continue
            pts = sorted((q["step"], q["avg_robustness_index"])
                         for q in json.loads(cur.read_text())["points"]
                         if q.get("avg_robustness_index") is not None)
            if len(pts) < 2:
                print(f"  {cell}: only {len(pts)} scored checkpoint(s), skipped")
                continue
            seed = re.search(r"-s(\d+)-", cell)
            rows.append((label, int(seed.group(1)) if seed else -1, sel, pts))
    return rows


def parse_base_ri():
    """-> {backbone label: base PathoROB RI}, read from the same doc the tables use."""
    base, cur = {}, None
    for line in SEED_STATS.read_text().splitlines():
        if line.startswith("## "):
            cur = SLUG.get(line[3:].strip())
        elif cur and line.startswith("| PathoROB RI |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            try:
                base.setdefault(cur, float(cells[1]))
            except ValueError:
                pass
    return base


def figure():
    rows = parse_curves()
    base_ri = parse_base_ri()
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
    figure()
