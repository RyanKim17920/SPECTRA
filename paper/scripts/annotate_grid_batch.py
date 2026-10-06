#!/usr/bin/env python3
"""Figure 1 (grid_batch). Left: the 3-row PLISM grid rendered by scripts/paper_figures.py
(grid_batch_left.png). Right: embedding-space push/pull sketch using the real tiles.

    python3 scripts/paper_figures.py && python3 paper/scripts/annotate_grid_batch.py
      -> $SPECTRA_PAPER/figures/grid_batch.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts"))
from _config import PAPER_FIGURES as OUT  # noqa: E402
im=np.asarray(Image.open(OUT/"grid_batch_left.png").convert("RGB"))
# Every pixel constant below was tuned against the old 200-dpi source (1254 px wide).
# paper_figures.py now renders that source at 1050 dpi so the print figure is not capped
# at 100 ppi, so the constants are scaled by the width ratio and the layout is unchanged.
SC=im.shape[1]/1254.0
def sc(v): return max(1,int(round(v*SC)))
im=np.concatenate([im[:sc(52)],im[sc(108):]],axis=0)          # drop blank band under the title
H,W,_=im.shape
def runs(mask,minlen):
    out=[];on=False
    for k,v in enumerate(mask):
        if v and not on: st=k;on=True
        if not v and on: out.append((st,k));on=False
    return [r for r in out if r[1]-r[0]>minlen]
rows=runs((im.mean(2)<235).mean(1)>0.4,sc(60))
cols=runs((im[rows[0][0]:rows[0][1]].mean(2)<235).mean(0)>0.4,sc(60))
def tile(r,c):   # chip-free square from the upper-left of the tile
    y0,y1=rows[r]; x0,x1=cols[c]; h=int((y1-y0)*0.38); o=sc(7)
    return im[y0+o:y0+o+h, x0+o:x0+o+h]
q=2
BLUE,RED,DARK,GREY="#2563EB","#B91C1C","#111827","#9CA3AF"
# NeurIPS forbids Type 3 fonts; 42 embeds TrueType instead (text AND mathtext).
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm",
                     "pdf.fonttype":42,"ps.fonttype":42})

fig=plt.figure(figsize=(14,5.2))
axL=fig.add_axes([0.0,0.0,0.66,1.0]); axL.imshow(im[:,sc(20):]); axL.axis("off")
axR=fig.add_axes([0.67,0.02,0.32,0.94]); axR.set_xlim(-0.12,1.12); axR.set_ylim(-0.12,1.12); axR.set_aspect("equal"); axR.axis("off")
axR.set_title("embedding space",fontsize=17,pad=4)
# faint background points
rng=np.random.default_rng(0); pts=rng.uniform(0.04,0.96,(40,2))
axR.scatter(pts[:,0],pts[:,1],s=14,color=GREY,alpha=0.35,zorder=0)

def thumb(img,x,y,s,ec,lw):
    axR.imshow(img,extent=[x-s,x+s,y-s,y+s],zorder=3)
    axR.add_patch(Rectangle((x-s,y-s),2*s,2*s,fc="none",ec=ec,lw=lw,zorder=4))
def arrow(p0,p1,color):
    axR.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=22,color=color,lw=3,zorder=5))

cx,cy=0.5,0.5; S=0.10
# query in the centre
thumb(tile(0,q),cx,cy,S,BLUE,4)
axR.text(cx,cy-S-0.02,"query",ha="center",va="top",fontsize=15,color=BLUE,fontweight="bold")
# positives: registered tiles from the two other conditions, pulled in
for (r,ang) in ((1,55),(2,205)):
    a=np.deg2rad(ang); px,py=cx+0.29*np.cos(a),cy+0.29*np.sin(a)
    thumb(tile(r,q),px,py,S*0.85,BLUE,3)
    ux,uy=cx-px,cy-py; n=np.hypot(ux,uy); ux,uy=ux/n,uy/n
    arrow((px+ux*(S*0.85+0.01),py+uy*(S*0.85+0.01)),(cx-ux*(S+0.02),cy-uy*(S+0.02)),BLUE)
axR.text(cx+0.29*np.cos(np.deg2rad(55))+0.10,cy+0.29*np.sin(np.deg2rad(55))+0.0,"pull\nregistered tile,\nother condition",ha="left",va="center",fontsize=14,color=BLUE)
# negatives: other locations of condition c_b, pushed out
for k,(c,ang) in enumerate(((0,125),(1,165),(3,275),(4,320),(5,10))):
    a=np.deg2rad(ang); px,py=cx+0.37*np.cos(a),cy+0.37*np.sin(a)
    thumb(tile(1,c),px,py,S*0.75,DARK,2.5)
    ux,uy=px-cx,py-cy; n=np.hypot(ux,uy); ux,uy=ux/n,uy/n
    arrow((px+ux*(S*0.75+0.01),py+uy*(S*0.75+0.01)),(px+ux*(S*0.75+0.11),py+uy*(S*0.75+0.11)),RED)
axR.text(0.0,0.12,"push\nother locations,\nsame condition",ha="left",va="center",fontsize=14,color=RED)
fig.savefig(OUT/"grid_batch.pdf",dpi=600,bbox_inches="tight"); fig.savefig(OUT/"grid_batch.png",dpi=100,bbox_inches="tight"); print("grid ok")
