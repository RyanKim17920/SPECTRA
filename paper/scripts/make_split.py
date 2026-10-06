#!/usr/bin/env python3
"""split figure (no data inputs; drawn programmatically).

    python3 paper/scripts/make_split.py -> $SPECTRA_PAPER/figures/split.{pdf,png}
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts"))
from _config import PAPER_FIGURES as OUT  # noqa: E402
OUT.mkdir(parents=True,exist_ok=True)
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8})
S,T=7,13; hs,ht=2,3
fig,ax=plt.subplots(figsize=(5.2,2.9))
cols={"train":"#DBEAFE","scanner":"#FDE68A","stain":"#FCA5A5","both":"#C4B5FD"}
for r in range(S):
    for c in range(T):
        us,ut=r>=S-hs,c>=T-ht
        k="both" if us and ut else "scanner" if us else "stain" if ut else "train"
        ax.add_patch(Rectangle((c,S-1-r),1,1,fc=cols[k],ec="white",lw=1))
ax.set_xlim(0,T); ax.set_ylim(0,S); ax.set_aspect("equal")
ax.set_xticks([i+0.5 for i in range(T)]); ax.set_xticklabels([f"{i+1}" for i in range(T)],fontsize=7)
ax.set_yticks([i+0.5 for i in range(S)]); ax.set_yticklabels([f"{S-i}" for i in range(S)],fontsize=7)
ax.set_xlabel("staining protocol (13; 3 held out)"); ax.set_ylabel("scanner (7; 2 held out)")
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=cols["train"],label="training: 50 conditions"),
                   Patch(fc=cols["scanner"],label="held out, unseen scanner: 20"),
                   Patch(fc=cols["stain"],label="held out, unseen stain: 15"),
                   Patch(fc=cols["both"],label="held out, both unseen: 6")],
          loc="center left",bbox_to_anchor=(1.02,0.5),frameon=False,fontsize=7.5)
ax.set_title("PLISM acquisition conditions: one cell = one scanner–stain pair = one whole-slide image",fontsize=8)
fig.savefig(OUT/"split.pdf",bbox_inches="tight"); fig.savefig(OUT/"split.png",dpi=130,bbox_inches="tight"); print("split ok")
