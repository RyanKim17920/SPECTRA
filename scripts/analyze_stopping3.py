#!/usr/bin/env python3
import json, statistics as st
from collections import defaultdict
rows=json.load(open('docs/stopping_criterion_rows.json'))
BB=['phikon','midnight','virchow2']; SD={'phikon':5.8,'midnight':8.3,'virchow2':14.2}

def binned(sig, edges, sub=None):
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        v=[r for r in (sub or rows) if r.get(sig) is not None and lo<=r[sig]<hi]
        out.append((lo,hi,len(v), st.mean(x['hest_pct'] for x in v) if v else None,
                    max((x['hest_pct'] for x in v),default=None)))
    return out

print("### A. L2 HYPOTHESIS: binned HEST% vs adapter_rel_l2_delta (mean, per backbone)")
edges=[0.2,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.25]
hdr=f"{'L2 bin':<14}"+ "".join(f"{b:>22}" for b in BB); print(hdr)
for lo,hi in zip(edges[:-1],edges[1:]):
    line=f"[{lo:.2f},{hi:.2f})".ljust(14)
    for b in BB:
        v=[r for r in rows if r['backbone']==b and lo<=r['l2']<hi]
        line+= f"  n={len(v):<2} mean={st.mean(x['hest_pct'] for x in v):6.1f} " if v else f"  {'-':>20} "
    print(line)
print("\nPEAK L2 per backbone (argmax over bins with n>=3), and L2 at the single best row:")
for b in BB:
    bins=[(lo,hi,v) for lo,hi in zip(edges[:-1],edges[1:]) for v in [[r for r in rows if r['backbone']==b and lo<=r['l2']<hi]] if len(v)>=3]
    best=max(bins,key=lambda t:st.mean(x['hest_pct'] for x in t[2]))
    top=max((r for r in rows if r['backbone']==b),key=lambda r:r['hest_pct'])
    top5=sorted([r for r in rows if r['backbone']==b],key=lambda r:-r['hest_pct'])[:5]
    print(f"  {b:<9} best bin [{best[0]:.2f},{best[1]:.2f}) mean={st.mean(x['hest_pct'] for x in best[2]):5.1f} n={len(best[2])} | best row l2={top['l2']:.3f} hest={top['hest_pct']:.1f} | L2 of top-5 rows: {[round(r['l2'],3) for r in top5]}")

print("\n### B. CI BAND: binned HEST% vs mean confounder_insensitivity")
edges2=[0.2,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.93,0.96,1.0,1.15]
print(f"{'CI bin':<14}"+"".join(f"{b:>22}" for b in BB)+f"{'POOLED':>24}")
for lo,hi in zip(edges2[:-1],edges2[1:]):
    line=f"[{lo:.2f},{hi:.2f})".ljust(14)
    for b in BB+['ALL']:
        v=[r for r in rows if (b=='ALL' or r['backbone']==b) and lo<=r['ci']<hi]
        line+= f"  n={len(v):<2} mean={st.mean(x['hest_pct'] for x in v):6.1f} " if v else f"  {'-':>20} "
    print(line)

print("\n### C. Fraction of rows with HEST%>=70, by CI band (pooled and per bb)")
for lo,hi in [(0.0,0.70),(0.70,0.78),(0.78,0.90),(0.90,0.94),(0.94,2.0)]:
    v=[r for r in rows if lo<=r['ci']<hi]
    s=f"CI[{lo:.2f},{hi:.2f})  n={len(v):<3} >=70: {sum(1 for r in v if r['hest_pct']>=70)}/{len(v)}  mean={st.mean(x['hest_pct'] for x in v):6.1f}   "
    for b in BB:
        w=[r for r in v if r['backbone']==b]
        s+=f"{b[:4]} {sum(1 for r in w if r['hest_pct']>=70)}/{len(w)}  "
    print(s)

print("\n### D. LR-schedule confound: annealed (step==max_steps) vs unannealed rows")
for b in BB:
    for ann in (True,False):
        v=[r for r in rows if r['backbone']==b and r['annealed']==ann]
        if v: print(f"  {b:<9} annealed={str(ann):<5} n={len(v):<3} meanHEST={st.mean(x['hest_pct'] for x in v):6.1f}  meanCI={st.mean(x['ci'] for x in v):.3f}  meanL2={st.mean(x['l2'] for x in v):.3f}")

print("\n### E. WITHIN recipe-family L2->HEST direction (families with >=4 rows)")
fam=defaultdict(list)
for r in rows: fam[(r['backbone'],r['family'])].append(r)
def pear(x,y):
    n=len(x); mx,my=st.mean(x),st.mean(y)
    sx=sum((a-mx)**2 for a in x)**.5; sy=sum((b-my)**2 for b in y)**.5
    return None if sx==0 or sy==0 else sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)
for k,v in sorted(fam.items()):
    if len(v)<4: continue
    rl=pear([r['l2'] for r in v],[r['hest_pct'] for r in v])
    rc=pear([r['ci'] for r in v],[r['hest_pct'] for r in v])
    rs=pear([r['step'] for r in v],[r['hest_pct'] for r in v])
    f=lambda z: f"{z:+.2f}" if z is not None else " n/a"
    print(f"  {k[0]:<9} {k[1]:<28} n={len(v):<3} r(L2)={f(rl)} r(CI)={f(rc)} r(step)={f(rs)}")
