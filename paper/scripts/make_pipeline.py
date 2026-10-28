#!/usr/bin/env python3
"""pipeline figure (no data inputs; drawn programmatically).

    python3 paper/scripts/make_pipeline.py -> $SPECTRA_PAPER/figures/pipeline.{pdf,png}
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts"))
from _config import PAPER_FIGURES as OUT  # noqa: E402
OUT.mkdir(parents=True,exist_ok=True)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8.5})
fig,ax=plt.subplots(figsize=(8.6,2.9)); ax.set_xlim(0,123); ax.set_ylim(0,36); ax.axis("off")
BLUE,ORANGE,GREY,LIGHT="#2563EB","#D97706","#374151","#F3F4F6"
def box(x,y,w,h,title,body="",ec=GREY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3",fc=LIGHT,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h-1.0,title,ha="center",va="top",fontsize=8,fontweight="bold")
    if body: ax.text(x+w/2,y+(h-3.2)/2,body,ha="center",va="center",fontsize=7,linespacing=1.35)
def arrow(x0,y0,x1,y1,ls="-"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,color=GREY,lw=1.1,ls=ls))
# 1 data
box(1,10,19,19,"PLISM grid batch","$C=2$ conditions\n$\\times\\;T=900$ locations\n1,800 registered tiles")
# 2 backbone
box(24,10,20,19,"ViT backbone","")
ax.text(34,24.5,"weights frozen",ha="center",va="center",fontsize=8)
ax.add_patch(FancyBboxPatch((26,11.5),16,9,boxstyle="round,pad=0.2",fc="#FFF7ED",ec=ORANGE,lw=1.2))
ax.text(34,16,"LoRA, rank 32\nattention + MLP",ha="center",va="center",fontsize=8,color=ORANGE,linespacing=1.3)
# 3 views
box(48,20.5,20,8.5,"CLS view","CLS token $\\rightarrow v^{\\mathrm{cls}}$")
box(48,10,20,8.5,"GeM view","GeM of patch tokens\n$\\rightarrow v^{\\mathrm{gem}}$")
# 4 heads
box(72,20.5,16,8.5,"head $g_{\\mathrm{cls}}$","MLP, $\\ell_2$-norm $\\rightarrow z^{\\mathrm{cls}}$",ec=ORANGE)
box(72,10,16,8.5,"head $g_{\\mathrm{gem}}$","MLP, $\\ell_2$-norm $\\rightarrow z^{\\mathrm{gem}}$",ec=ORANGE)
# 5 loss
box(92,10,30,19,"Grid InfoNCE")
sx,sy,ss=94,12,10; n=6
ax.add_patch(Rectangle((sx,sy),ss,ss,fc="white",ec=GREY,lw=1))
for k in range(n): ax.add_patch(Rectangle((sx+k*ss/n,sy+ss-(k+1)*ss/n),ss/n,ss/n,fc=BLUE,ec="none"))
ax.text(sx+ss/2,sy-0.5,"row of $c_b$",ha="center",va="top",fontsize=7)
ax.text(sx-0.5,sy+ss/2,"row of $c_a$",ha="right",va="center",fontsize=7,rotation=90)
ax.text(105.5,19.5,"one softmax per query\npositive: registered tile\nnegatives: $T-1$ tiles,\nsame scanner and stain\n$\\tau=0.07$",
        ha="left",va="center",fontsize=7.5,linespacing=1.35)
ax.text(105.5,12,"$\\mathcal{L}=0.5\\,\\mathcal{L}_{\\mathrm{cls}}+0.5\\,\\mathcal{L}_{\\mathrm{gem}}$",ha="left",fontsize=8)
# arrows
arrow(20.4,19.5,23.6,19.5); arrow(44.4,21,47.6,24.5); arrow(44.4,17,47.6,14.2)
arrow(68.4,24.7,71.6,24.7); arrow(68.4,14.2,71.6,14.2); arrow(88.4,24.7,91.6,22); arrow(88.4,14.2,91.6,16.5)

# after training
ax.annotate("",xy=(34,4.5),xytext=(34,9.6),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=1.1,ls="--"))
ax.text(36,3.0,"after training: LoRA merged into the backbone; heads and GeM exponent discarded.\n"
        "All benchmarks use the backbone's original embedding and pooling; inference cost is unchanged.",
        ha="left",va="center",fontsize=7,color=GREY,linespacing=1.35)
ax.text(1,33.5,"orange = trained during adaptation",fontsize=8,color=ORANGE)
fig.savefig(OUT/"pipeline.pdf",bbox_inches="tight"); fig.savefig(OUT/"pipeline.png",dpi=130,bbox_inches="tight"); print("pipe ok")
