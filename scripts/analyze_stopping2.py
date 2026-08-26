#!/usr/bin/env python3
import json, statistics as st
from collections import defaultdict
rows=json.load(open('docs/stopping_criterion_rows.json'))
byrun=defaultdict(list)
for r in rows: byrun[r['run']].append(r)
multi={k:sorted(v,key=lambda r:r['step']) for k,v in byrun.items() if len(v)>=2}
print("runs with >=2 HEST'd checkpoints:",len(multi))
for b in ('phikon','midnight','virchow2'):
    ms=[k for k,v in multi.items() if v[0]['backbone']==b]
    print(f"  {b}: {len(ms)} runs, steps sets:", sorted({tuple(r['step'] for r in multi[k]) for k in ms}))
print()
print("WITHIN-RUN trajectories (HEST% at each step, and internal signals)")
for b in ('phikon','midnight','virchow2'):
    print(f"\n########## {b} ##########")
    for k in sorted(multi, key=lambda k:(multi[k][0]['backbone'],k)):
        v=multi[k]
        if v[0]['backbone']!=b: continue
        print(f"{k[:60]:<60} ms={v[0]['max_steps']}")
        for r in v:
            print(f"   step {r['step']:>5}  HEST%={r['hest_pct']:6.1f}  l2={r['l2']:.3f}  RI={r['ri']:.4f}  CI={r['ci']:.4f}  PP={r['pp']:.4f}  top1={r['top1'] if r['top1'] is None else round(r['top1'],3)}  loss={r['loss'] if r['loss'] is None else round(r['loss'],3)}")
