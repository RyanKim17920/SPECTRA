#!/usr/bin/env python3
"""figures/plism_traj.pdf -- held-out PLISM top-1 retrieval from base to end of training.

Renders from paper/data/plism_traj.json (frozen by paper/scripts/extract_plism_traj.py;
needs no /data). Base (step 0) is the untuned backbone's probe_before.json; the trajectory
is probe_step_*.json of the 500-step recipe runs (genMASK-b00-ms500-<backbone>-s0). Both
use groups.heldout and the `.embedding` view -- concatenated CLS and mean-patch tokens, no
projection head -- matching the paper protocol. Only three backbones have a base probe, so
only those get a step-0 point. Plotting code is the paper's make_plism_traj.py verbatim.

    python3 paper/scripts/plism_traj.py [--data PATH]  -> $SPECTRA_PAPER_FIGURES/plism_traj.pdf
"""
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_FIGURES  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", type=Path, default=REPO / "paper/data/plism_traj.json")
DATA = json.loads(ap.parse_args().data.read_text())["series"]

NAME = {"phikon": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus": "H-optimus-0", "uni2": "UNI2-h"}
SEL = {"phikon": 150, "midnight": 100, "virchow2": 150, "hoptimus": 100, "uni2": 150}


def series(bb):
    if bb not in DATA:
        return {}
    return {int(st): (sc, sn) for st, sc, sn in DATA[bb]["points"]}


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
colors = plt.cm.tab10.colors
for k, (ax, idx, title) in enumerate(
        [(axes[0], 0, "Scanner shift (stain fixed)"), (axes[1], 1, "Stain shift (scanner fixed)")]):
    for i, bb in enumerate(NAME):
        s = series(bb)
        if not s: continue
        xs = list(s); ys = [s[x][idx] for x in xs]
        has_base = 0 in s
        ax.plot(xs, ys, "-o" if has_base else "--o", color=colors[i], ms=3, lw=1.5,
                label=NAME[bb] + ("" if has_base else " (no base probe)"))
        if has_base:
            ax.plot([0], [s[0][idx]], "o", color=colors[i], ms=8, mfc="white", mew=1.8, zorder=5)
        sel = SEL[bb]
        if sel in s:
            ax.plot([sel], [s[sel][idx]], "*", color=colors[i], ms=13, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("top-1 retrieval among 256")
axes[0].set_ylim(0.5, 1.02)
axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
axes[1].text(0.98, 0.05, "hollow = base (step 0)\n$\\star$ = selected checkpoint",
             transform=axes[1].transAxes, ha="right", fontsize=7)
fig.tight_layout()
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER_FIGURES / "plism_traj.pdf", bbox_inches="tight")
print(f"wrote {PAPER_FIGURES / 'plism_traj.pdf'}")
for bb in NAME:
    s = series(bb)
    if 0 in s and SEL[bb] in s:
        b, t = s[0], s[SEL[bb]]
        print(f"  {NAME[bb]:14s} scanner {b[0]:.3f}->{t[0]:.3f} ({t[0]-b[0]:+.3f})   "
              f"stain {b[1]:.3f}->{t[1]:.3f} ({t[1]-b[1]:+.3f})")
    elif s:
        t = s[SEL[bb]]; print(f"  {NAME[bb]:14s} no base probe; at step {SEL[bb]}: "
                              f"scanner {t[0]:.3f}, stain {t[1]:.3f}")
