#!/usr/bin/env python3
"""Apply rules on the FULL ri_curve grid (every logged checkpoint), then look up HEST."""
import json, os, statistics as st
from collections import defaultdict
rows=json.load(open('docs/stopping_criterion_rows.json'))
hest={(r['run'],r['step']):r['hest_pct'] for r in rows}
meta={r['run']:r for r in rows}
RUNS='/admin/home/ryan.kim/waiv/runs'
BB=['phikon','midnight','virchow2']
RI_BASE={"phikon":0.4686,"midnight":0.7589,"virchow2":0.8582}
RI_WAIV={"phikon":0.806,"midnight":0.924,"virchow2":0.918}

curves={}
for run in {r['run'] for r in rows}:
    d=json.load(open(os.path.join(RUNS,run,'ri_curve.json')))
    pts=[]
    for p in sorted(d['points'],key=lambda p:p['step']):
        ds=p.get('datasets',{})
        f=lambda k:[v[k] for v in ds.values() if isinstance(v,dict) and isinstance(v.get(k),(int,float))]
        l2=p.get('adapter_rel_l2_delta')
        l2v=st.mean([v for v in l2.values() if isinstance(v,(int,float))]) if isinstance(l2,dict) else l2
        bb=meta[run]['backbone']
        pts.append(dict(step=p['step'],l2=l2v,ri=p.get('avg_robustness_index'),
            ci=st.mean(f('confounder_insensitivity')) if f('confounder_insensitivity') else None,
            pp=st.mean(f('prediction_performance')) if f('prediction_performance') else None,
            hest=hest.get((run,p['step'])),bb=bb,
            ri_pct=100*(p['avg_robustness_index']-RI_BASE[bb])/(RI_WAIV[bb]-RI_BASE[bb])))
    curves[run]=pts

print("FULL-GRID checkpoint availability per run (steps; * = has HEST)")
for b in BB:
    print(f"\n### {b}")
    for run,pts in sorted(curves.items()):
        if pts[0]['bb']!=b: continue
        print(f"  {run[:58]:<58} " + " ".join(f"{p['step']}{'*' if p['hest'] is not None else ''}" for p in pts))
print()
print("CI TRAJECTORY on full grid (where is the CI>=0.75 crossing?)")
for b in BB:
    print(f"\n### {b}")
    for run,pts in sorted(curves.items()):
        if pts[0]['bb']!=b: continue
        cr=next((p for p in pts if p['ci'] is not None and p['ci']>=0.75), None)
        s=" ".join(f"{p['step']}:{p['ci']:.2f}{'*' if p['hest'] is not None else ''}" for p in pts if p['ci'] is not None)
        print(f"  {run[:52]:<52} cross@{cr['step'] if cr else 'never':>5} hest={('%.1f'%cr['hest']) if cr and cr['hest'] is not None else '--':>6}  {s}")
json.dump({k:v for k,v in curves.items()},open('docs/stopping_full_curves.json','w'))
