#!/usr/bin/env python3
import json, statistics as st
from collections import defaultdict
rows = json.load(open('docs/stopping_criterion_rows.json'))
BB = ['phikon','midnight','virchow2']
SD = {'phikon':5.8,'midnight':8.3,'virchow2':14.2}

def pearson(x,y):
    n=len(x)
    if n<3: return None,n
    mx,my=st.mean(x),st.mean(y)
    sx=sum((a-mx)**2 for a in x)**.5; sy=sum((b-my)**2 for b in y)**.5
    if sx==0 or sy==0: return None,n
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy), n
def spearman(x,y):
    def rk(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        i=0
        while i<len(s):
            j=i
            while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
            avg=(i+j)/2+1
            for k in range(i,j+1): r[s[k]]=avg
            i=j+1
        return r
    return pearson(rk(x),rk(y))

SIGS=['l2','ri','ci','pp','gi','idp','oodp','bacc','loss','top1','heldout_loss','heldout_top1','step']
print("="*90); print("CORRELATION of each internal signal with HEST pct_of_waiv, per backbone")
print(f"{'signal':<14}"+"".join(f"{b:>26}" for b in BB))
for s in SIGS:
    line=f"{s:<14}"
    for b in BB:
        d=[(r[s],r['hest_pct']) for r in rows if r['backbone']==b and r.get(s) is not None]
        p,_=pearson([a for a,_ in d],[c for _,c in d]) if len(d)>2 else (None,0)
        sp,_=spearman([a for a,_ in d],[c for _,c in d]) if len(d)>2 else (None,0)
        line+=f"   r={p:+.3f} rho={sp:+.3f} n={len(d):<3}" if p is not None else f"   {'--':>22}"
    print(line)

print(); print("="*90); print("PER-BACKBONE: rows sorted by HEST pct (top 8 and bottom 5)")
for b in BB:
    rs=sorted([r for r in rows if r['backbone']==b], key=lambda r:-r['hest_pct'])
    print(f"\n--- {b}  n={len(rs)}  (HEST 1SD = {SD[b]} pct pts) ---")
    print(f"{'hest%':>7} {'RI%':>7} {'l2':>6} {'RI':>7} {'CI':>7} {'PP':>7} {'top1':>6} {'step':>5} {'ann':>4}  run/family")
    for r in rs[:8]+[None]+rs[-5:]:
        if r is None: print('   ...'); continue
        f=lambda v,w=7,p=4: (f"{v:{w}.{p}f}" if isinstance(v,(int,float)) else f"{'--':>{w}}")
        print(f"{r['hest_pct']:7.1f} {f(r['ri_pct'],7,1)} {f(r['l2'],6,3)} {f(r['ri'])} {f(r['ci'])} {f(r['pp'])} {f(r['top1'],6,3)} {r['step']:5d} {str(r['annealed'])[0]:>4}  {r['run'][:46]}")
