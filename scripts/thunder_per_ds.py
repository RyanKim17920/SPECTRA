#!/usr/bin/env python3
"""Per-dataset THUNDER deltas for the paper, with the recipe's own across-seed error bar.

The paper reports THUNDER only as a task mean over 16 classification datasets, which hides
that the mean is a sum of dataset-level moves in both directions. This builds the
per-dataset panel for the SHIPPED c50 arm at each seed's own 1-SE selected checkpoint --
the same cells scripts/seed_stats.py resolves, so the row means reproduce the paper tables.

Five panels: three macro-F1 panels (knn / linear_probing / simple_shot, higher is better)
plus a calibration ECE panel (linear_probing task, ECE metric, lower is better) and an
adversarial-drop panel (adversarial_attack task, f1 metric, "drop" setting, lower is
better). The ECE and adversarial panels use a colour scale REVERSED relative to the F1
panels so that "good" (F1 up, ECE/drop down) is always the same colour.

The adversarial panel uses DIFFERENT cells than the other four: adversarial numbers are
only valid under the fp32-attack rerun (fp16 autocast under-attacks, see
docs/thunder_ranks.md), so it reads scripts/thunder_ranks.FP32_CELLS -- the `*f-` cell
family for phikon2/midnight/virchow2/hoptimus0/uni2h (virchow2's base cell is the oddly
named "virchow2-basectrl-fp32adv"), and the same in-place cells as the other panels for
virchow1/openmidnightsq, which were patched with the fp32 attack directly.

Every panel also carries three summary columns on the right, separated from the 16 dataset
columns by a visible blank gap column: "mean" (all 16 datasets), "mean, no tcga_unif" and
"mean, no bach" -- row means of the per-dataset deltas with one dataset dropped. tcga_uniform
dominates the classification (F1) panels and bach dominates the ECE panel; these columns make
that dominance, and what the picture looks like without it, visible in the figure itself.
Summary cells are plain (non-bold, full-opacity) text -- the bold/faint "resolved at 2x the
across-seed SD" convention only applies to the per-dataset cells, since a summary column is
an aggregate, not a single seed-backed measurement.

Error bar policy follows seed_stats.py: a delta is judged against the SD of THIS recipe's
own seeds on THAT dataset, not against a floor imported from another arm or panel. (The
12-dataset floor in docs/thunder_seed_floor_12ds.json is measured on the older final5 runs
at a fixed step 500 and does not transfer to this 16-dataset c50 panel.)

    ./.venv-hest/bin/python scripts/thunder_per_ds.py
      -> waiv-asci/figures/thunder_per_ds.pdf (+ .png)
      -> docs/thunder_per_ds.md
"""
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import PAPER_FIGURES  # noqa: E402
import seed_stats as S
import thunder_ranks as TR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTFIG = PAPER_FIGURES
INK, MUTED = "#1f2937", "#9ca3af"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK, "pdf.fonttype": 42})

PAPER_BB = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1",
            "openmidnightsq"]
LABEL = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
         "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
         "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}

# (task, metric, setting, panel label, higher_is_better)
PANELS = [("knn", "f1", "", "$k$-NN", True),
          ("linear_probing", "f1", "", "linear probing", True),
          # simple_shot rows carry the shot count in the `setting` field ("16"), NOT "".
          # Hardcoding "" here silently emptied the whole few-shot panel.
          ("simple_shot", "f1", "16", "few-shot", True),
          ("linear_probing", "ECE", "", "calibration ECE (lower is better)", False),
          ("adversarial_attack", "f1", "drop", "adversarial drop (lower is better)", False)]

# Per-panel colour-scale half-range. The F1/ECE panels cluster within a few points; the
# adversarial drop swings much wider (median |delta| ~7.4, max ~31 across the 7 backbones),
# so it gets its own, wider scale rather than saturating almost every cell.
VMAX = {4: 15.0}
DEFAULT_VMAX = 6.0

DS_LABEL = {"patch_camelyon": "pcam", "tcga_crc_msi": "crc_msi", "tcga_tils": "tils",
            "tcga_uniform": "tcga_unif", "spider_breast": "spider_br",
            "spider_colorectal": "spider_col", "spider_skin": "spider_sk",
            "spider_thorax": "spider_th", "break_his": "breakhis"}

# Right-hand summary columns: (label, dataset id to exclude, or None for "all").
# tcga_uniform is the consistent drag on the F1 panels; bracs is the consistent drag on
# ECE (worse on 6 of 7 backbones, column mean +4.4); bach has the LARGEST single ECE swings
# (UNI2-h +15.0, Virchow -12.8) but they cancel, so its column mean is only +0.4. Both are
# shown because they are outliers for different reasons: bracs in direction, bach in spread.
SUMMARY_COLS = [("mean", None), ("mean, no tcga_unif", "tcga_uniform"),
                ("mean, no bracs", "bracs")]


def collect():
    rows = S.load_thunder()
    cells = sorted(d.name for d in S.CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())
    data, datasets = {}, set()
    for bb in PAPER_BB:
        base = S.BASE_CELL.get(bb, f"{bb}-base-control")
        seeds = [c for c in cells
                 if c.startswith(f"{bb}-c50-s") and c != base and S.rule_selected(c)]
        # Adversarial cells are the fp32-attack family (see module docstring); virchow1 and
        # openmidnightsq have no separate `*f-` family, so they fall back to the same cells
        # as every other panel.
        adv_base, adv_seeds = TR.FP32_CELLS.get(bb, (base, seeds))
        for pidx, (task, metric, setting, _, _) in enumerate(PANELS):
            is_adv = task == "adversarial_attack"
            b_cell, s_cells = (adv_base, adv_seeds) if is_adv else (base, seeds)
            for key, v in rows.get(b_cell, {}).items():
                d, t, m, st = key
                if t != task or m != metric or st != setting or d.startswith("benchmark_"):
                    continue
                datasets.add(d)
                tuned = [rows.get(c, {}).get(key) for c in s_cells]
                tuned = [x for x in tuned if x is not None]
                if not tuned:
                    continue
                data[(bb, pidx, d)] = (v, mean(tuned),
                                       stdev(tuned) if len(tuned) > 1 else float("nan"),
                                       len(tuned))
    return data, sorted(datasets)


def _row_deltas(data, bb, pidx, datasets):
    """(dataset, delta) pairs available for this backbone/panel."""
    out = []
    for d in datasets:
        cell = data.get((bb, pidx, d))
        if cell:
            b, t, sd, n = cell
            out.append((d, t - b))
    return out


def _summary_means(pairs):
    """One mean per SUMMARY_COLS entry, dropping the named dataset if present."""
    out = []
    for _, excl in SUMMARY_COLS:
        vals = [v for d, v in pairs if excl is None or d != excl]
        out.append(mean(vals) if vals else float("nan"))
    return out


def render(data, datasets):
    ncol_ds = len(datasets)
    ncol = ncol_ds + 1 + len(SUMMARY_COLS)  # +1 blank gap column
    gap_col = ncol_ds
    summary_start = ncol_ds + 1
    fig, axes = plt.subplots(len(PANELS), 1,
                             figsize=(0.40 * ncol + 2.4, 1.55 * len(PANELS) + 0.5),
                             sharex=True)

    for pidx, (ax, (task, metric, setting, tlabel, higher)) in enumerate(zip(axes, PANELS)):
        vmax = VMAX.get(pidx, DEFAULT_VMAX)
        cmap = plt.get_cmap("RdBu") if higher else plt.get_cmap("RdBu_r")
        nrow_g = len(PAPER_BB) + 1          # +1 = column-mean row across backbones
        grid = np.full((nrow_g, ncol), np.nan)
        row_pairs = {}
        for i, bb in enumerate(PAPER_BB):
            pairs = _row_deltas(data, bb, pidx, datasets)
            row_pairs[bb] = dict(pairs)
            for j, d in enumerate(datasets):
                if d in row_pairs[bb]:
                    grid[i, j] = row_pairs[bb][d]
            for k, sval in enumerate(_summary_means(pairs)):
                grid[i, summary_start + k] = sval
        # column means across backbones: shows which datasets move consistently
        mrow = len(PAPER_BB)
        for j in range(ncol):
            col = grid[:mrow, j]
            col = col[~np.isnan(col)]
            if len(col):
                grid[mrow, j] = col.mean()
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for j in range(ncol):
            v = grid[mrow, j]
            if not np.isnan(v):
                ax.text(j, mrow, f"{v:+.1f}", ha="center", va="center", fontsize=5.6,
                        color=INK if abs(v) < 4 else "white", weight="bold")
        ax.axhline(mrow - 0.5, color=MUTED, lw=0.6)

        for i, bb in enumerate(PAPER_BB):
            for j, d in enumerate(datasets):
                cell = data.get((bb, pidx, d))
                if not cell:
                    continue
                b, t, sd, n = cell
                delta = t - b
                resolved = (not np.isnan(sd)) and abs(delta) >= 2 * sd
                ax.text(j, i, f"{delta:+.1f}", ha="center", va="center",
                        fontsize=5.6, color=INK if abs(delta) < 4 else "white",
                        weight="bold" if resolved else "normal",
                        alpha=1.0 if resolved else 0.45)
            for k in range(len(SUMMARY_COLS)):
                val = grid[i, summary_start + k]
                if np.isnan(val):
                    continue
                ax.text(summary_start + k, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=5.6, color=INK if abs(val) < 4 else "white",
                        weight="normal", alpha=1.0)

        ax.axvline(gap_col - 0.5, color=MUTED, lw=0.5)
        ax.axvline(gap_col + 0.5, color=MUTED, lw=0.5)
        ax.set_yticks(range(len(PAPER_BB) + 1))
        ax.set_yticklabels([LABEL[b] for b in PAPER_BB] + ["mean"], fontsize=6.8)
        ax.set_title(tlabel, fontsize=8, loc="left", pad=3)
        ax.set_xticks(range(ncol))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012)
        lower_name = "ECE" if task == "linear_probing" else "adversarial drop"
        cb.set_label("tuned $-$ base" if higher else f"tuned $-$ base ({lower_name}, lower better)",
                     fontsize=6)
        cb.ax.tick_params(labelsize=5.5)

    xticklabels = ([DS_LABEL.get(d, d) for d in datasets] + [""] +
                   [lbl for lbl, _ in SUMMARY_COLS])
    axes[-1].set_xticklabels(xticklabels, rotation=60, ha="right", fontsize=6.2)

    # Short title only. The bold/faint and colour-scale conventions live in the LaTeX
    # caption, not baked into the image at 6pt where they are unreadable.
    fig.suptitle("Per-dataset THUNDER change, fine-tuned $-$ base", fontsize=11, y=0.998)
    fig.subplots_adjust(top=0.965, bottom=0.10, hspace=0.30)
    OUTFIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFIG / "thunder_per_ds.pdf", bbox_inches="tight")
    fig.savefig(OUTFIG / "thunder_per_ds.png", dpi=150, bbox_inches="tight")


def markdown(data, datasets):
    out = ["# THUNDER per-dataset deltas, shipped c50 recipe", "",
           "**Generated file -- do not hand-edit.**  Regenerate with:", "",
           "```", "./.venv-hest/bin/python scripts/thunder_per_ds.py", "```", "",
           "Each seed contributes its own 1-SE selected checkpoint (same cells as",
           "`scripts/seed_stats.py`).  `resolved` means |delta| >= 2x the across-seed SD of",
           "this recipe's own seeds on that dataset.  F1 panels: deltas are macro F1 points,",
           "higher is better.  ECE panel (linear_probing task, ECE metric): deltas are ECE",
           "points, LOWER is better -- a negative delta is an improvement.  Adversarial panel",
           "(adversarial_attack task, f1 metric, drop setting): deltas are accuracy-drop-under-",
           "attack points, LOWER is better; it uses the fp32-attack cell family, NOT the same",
           "cells as the other four panels (see the module docstring).  The three summary",
           "columns are row means of the per-dataset deltas with the named dataset dropped;",
           "they are plain aggregates and are never bolded.", ""]
    for pidx, (task, metric, setting, tlabel, higher) in enumerate(PANELS):
        heading = tlabel if metric == "f1" else f"{task} / {metric}"
        out += [f"## {heading}", "",
                "| backbone | " + " | ".join(DS_LABEL.get(d, d) for d in datasets) +
                " | " + " | ".join(lbl for lbl, _ in SUMMARY_COLS) + " | n resolved |",
                "|" + "---|" * (len(datasets) + len(SUMMARY_COLS) + 2)]
        for bb in PAPER_BB:
            cells, res = [], 0
            pairs = _row_deltas(data, bb, pidx, datasets)
            have = dict(pairs)
            for d in datasets:
                c = data.get((bb, pidx, d))
                if not c:
                    cells.append("--")
                    continue
                b, t, sd, n = c
                delta = t - b
                ok = (sd == sd) and abs(delta) >= 2 * sd
                res += ok
                cells.append(f"**{delta:+.1f}**" if ok else f"{delta:+.1f}")
            summary_cells = [f"{v:+.2f}" if v == v else "--" for v in _summary_means(pairs)]
            out.append(f"| {LABEL[bb]} | " + " | ".join(cells) + " | " +
                       " | ".join(summary_cells) + f" | {res}/{len(pairs)} |")
        out.append("")
    dest = S.REPO / "docs/thunder_per_ds.md"
    dest.write_text("\n".join(out))
    print(f"wrote {dest}")


def _assert_panels_populated(data):
    """Fail loudly if any panel is empty.

    The `setting` field differs per task ("" for knn/linear_probing/ECE, "16" for
    simple_shot, "drop" for adversarial). Getting it wrong yields a silently BLANK panel
    that still renders and still writes markdown -- exactly how the few-shot panel went
    missing. Never let that be silent again.
    """
    for pidx, panel in enumerate(PANELS):
        n = sum(1 for k in data if k[1] == pidx)
        if n == 0:
            raise SystemExit(f"PANEL {pidx} ({panel[3]!r}) HAS NO DATA -- check its "
                             f"(task, metric, setting) key: {panel[0]!r}/{panel[1]!r}/{panel[2]!r}")


if __name__ == "__main__":
    data, datasets = collect()
    _assert_panels_populated(data)
    print(f"{len(data)} (backbone, panel, dataset) cells over {len(datasets)} datasets")
    render(data, datasets)
    markdown(data, datasets)
