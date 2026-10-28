#!/usr/bin/env python3
"""Figure 2: base -> fine-tuned dumbbell over all four benchmarks.

Reads paper/data/base_to_tuned.json (frozen from the generated tables by
paper/scripts/extract_base_to_tuned.py). Every SD there is already TWO sample SDs, so no
further scaling is applied here. Plotting code is the paper's make_base_to_tuned.py verbatim.

    python3 paper/scripts/make_base_to_tuned.py  -> $SPECTRA_PAPER/figures/base_to_tuned.{pdf,png}
"""
import json, sys, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts"))
from _config import PAPER_FIGURES  # noqa: E402
OUT=PAPER_FIGURES
DATA=json.loads((REPO/"paper/data/base_to_tuned.json").read_text())

plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8,
                     "axes.spines.top":False,"axes.spines.right":False,
                     # NeurIPS forbids Type 3 fonts; 42 embeds TrueType instead (text AND mathtext).
                     "pdf.fonttype":42,"ps.fonttype":42})
ORDER=["Phikon-v2","OpenMidnight","UNI2-h","Midnight-12k","H-optimus-0","Virchow","Virchow2"]
def getter(key):
    return lambda b:tuple(np.nan if v is None else v for v in DATA["panels"][key][b])
panels_top=[("robustness index",getter("pathorob_ri"),True,3,"PathoROB"),
            ("mean Pearson $r$",getter("hest_r"),True,4,"HEST"),
            ("macro-OvR AUC (38 tasks)",getter("cptac_auc"),True,4,"CPTAC")]
th=[("$k$-NN\nmacro-F1 (%)","thunder_knn",True),("Linear probe\nmacro-F1 (%)","thunder_lp",True),("Few-shot\nmacro-F1 (%)","thunder_fewshot",True),
    ("Segmentation\nF1 (%)","thunder_seg",True),("Calibration\nECE (%)","thunder_ece",False),("Adversarial\naccuracy drop (%)","thunder_adv",False)]
panels_bot=[(n,getter(k),up,1) for n,k,up in th]

BLUE,RED,GREY="#2563EB","#B91C1C","#9CA3AF"
fig=plt.figure(figsize=(8.2,6.0))
gs=fig.add_gridspec(2,6,height_ratios=[1,1],hspace=0.75,wspace=0.25)
axes_top=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6])]
axes_bot=[fig.add_subplot(gs[1,i]) for i in range(6)]
y=np.arange(len(ORDER))[::-1]

def draw(ax,name,get,up,fmt,annotate,title=None):
    vals=[get(b) for b in ORDER]
    for yi,(b,t,sd) in zip(y,vals):
        better=(t>b) if up else (t<b)
        col=BLUE if better else RED
        if not np.isnan(sd) and abs(t-b)<=sd: col=GREY
        ax.plot([b,t],[yi,yi],color=col,lw=1.6,zorder=1)
        ax.scatter([b],[yi],s=28,facecolor="white",edgecolor="black",lw=1.0,zorder=3)
        if not np.isnan(sd): ax.errorbar([t],[yi],xerr=[sd],fmt="none",ecolor=col,elinewidth=1.2,capsize=2,zorder=2)
        ax.scatter([t],[yi],s=28,color=col,zorder=4)
        d=t-b
        if annotate:
            s=f"{d:+.{fmt}f}"+("" if np.isnan(sd) else f" ± {sd:.{fmt}f}")
            ax.text(0.99,yi,s,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=6.5,color=col)
        else:
            txt=f"{d:+.{fmt}f}"+("" if np.isnan(sd) else f"$\\pm${sd:.{fmt}f}")
            ax.text(0.99,yi+0.02,txt,transform=ax.get_yaxis_transform(),ha="right",va="bottom",fontsize=5.6,color=col)
    lo=min(min(v[0],v[1]) for v in vals); hi=max(max(v[0],v[1]) for v in vals); span=hi-lo
    ax.set_xlim(lo-0.08*span, hi+(1.05 if annotate else 0.82)*span)
    tag=("higher is better $\\rightarrow$" if up else "$\\leftarrow$ lower is better")
    # The direction tag always goes on its OWN line, in the top row as well as the bottom.
    # On one line the top-row labels ("macro-OvR AUC (38 tasks)   higher is better ->")
    # ran wider than their panel; the bottom row already wrapped because its metric names
    # carry a newline. Direction is taken from `up`, so ECE and adversarial drop keep
    # "<- lower is better".
    ax.set_xlabel(name+"\n"+tag,fontsize=8)
    if title: ax.set_title(title,fontsize=10,pad=16)
    ax.set_yticks(y); ax.grid(axis="x",color="#E5E7EB",lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x",labelsize=7); ax.locator_params(axis="x",nbins=3)

for ax,(n,g,up,f,ti) in zip(axes_top,panels_top):
    draw(ax,n,g,up,f,annotate=True,title=ti); ax.set_yticklabels(ORDER if ax is axes_top[0] else [])
    ax.text(0.99,1.04,"$\\Delta$ vs base",transform=ax.transAxes,ha="right",va="bottom",fontsize=7.5,color="#374151")
for ax,(n,g,up,f) in zip(axes_bot,panels_bot):
    draw(ax,n,g,up,f,annotate=False); ax.set_yticklabels(ORDER if ax is axes_bot[0] else [])
    ax.text(0.99,1.02,"$\\Delta$ vs base",transform=ax.transAxes,ha="right",va="bottom",fontsize=6.5,color="#374151")
# Row titles sit ABOVE their own row and are left-aligned, so the top row is labelled and
# neither title floats between the two rows where it could be read as belonging to either.
# Left-aligned to avoid colliding with the right-aligned "Delta vs base" tags.
ROW_TITLES=[(axes_bot,"THUNDER: mean over 16 classification datasets (segmentation: 4 datasets)")]
axes_top[0].set_ylabel("Foundation model",fontsize=8); axes_bot[0].set_ylabel("Foundation model",fontsize=8)

handles=[Line2D([],[],marker="o",ls="",mfc="white",mec="black",label="base model (public checkpoint, single evaluation)"),
         Line2D([],[],marker="o",ls="-",color=BLUE,label="fine-tuned: mean of 3 adapter seeds, bar and number = ±2 SD; blue = improved over base"),
         Line2D([],[],marker="o",ls="-",color=RED,label="red = worse than base"),
         Line2D([],[],marker="o",ls="-",color=GREY,label="grey = |$\\Delta$| ≤ 2 SD (within seed noise)")]
fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.09))
fig.subplots_adjust(top=0.92,bottom=0.18,left=0.12,right=0.99)
for row,title in ROW_TITLES:
    ytop=max(a.get_position().y1 for a in row)
    xleft=min(a.get_position().x0 for a in row)
    fig.text(xleft,ytop+0.035,title,ha="left",va="bottom",fontsize=9)
OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/"base_to_tuned.pdf",bbox_inches="tight"); fig.savefig(OUT/"base_to_tuned.png",dpi=130,bbox_inches="tight")
print("ok")
