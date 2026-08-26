#!/usr/bin/env python3
"""Evaluate candidate model-agnostic stopping rules.
A rule sees only internal signals along a run's checkpoint sequence and picks a step.
Scored by the HEST pct_of_waiv actually measured at that step."""
import json, statistics as st
from collections import defaultdict
rows=json.load(open('docs/stopping_criterion_rows.json'))
BB=['phikon','midnight','virchow2']; SD={'phikon':5.8,'midnight':8.3,'virchow2':14.2}
byrun=defaultdict(list)
for r in rows: byrun[r['run']].append(r)
runs={k:sorted(v,key=lambda r:r['step']) for k,v in byrun.items() if len(v)>=2}

# ---------- how much does each signal move WITHIN a run vs ACROSS runs? ----------
print("SIGNAL DYNAMIC RANGE: within-run span (median) vs across-run span, per backbone")
print(f"{'bb':<9}{'signal':<8}{'median within-run span':>24}{'across-run span':>18}{'ratio':>8}")
for b in BB:
    rs=[r for r in rows if r['backbone']==b]
    for s in ['l2','ri','ci','pp','top1','hest_pct']:
        w=[max(x[s] for x in v)-min(x[s] for x in v) for v in runs.values()
           if v[0]['backbone']==b and all(x.get(s) is not None for x in v)]
        a=[x[s] for x in rs if x.get(s) is not None]
        if not w or not a: continue
        A=max(a)-min(a); W=st.median(w)
        print(f"{b:<9}{s:<8}{W:>24.4f}{A:>18.4f}{(W/A if A else 0):>8.2f}")
print()

# ---------- rules ----------
def first_where(v, pred):
    for r in v:
        if pred(r): return r
    return v[-1]

def mk_thresh(sig, c):
    return lambda v: first_where(v, lambda r: r.get(sig) is not None and r[sig] >= c)
def mk_thresh_le(sig, c):
    return lambda v: first_where(v, lambda r: r.get(sig) is not None and r[sig] <= c)

def rule_earliest(v): return v[0]
def rule_argmax_ri(v): return max(v, key=lambda r: r['ri'])
def rule_argmax_ci(v): return max(v, key=lambda r: r['ci'])
def rule_argmax_pp(v): return max(v, key=lambda r: r['pp'])
def rule_ci_sat(v, frac=0.98):
    """first checkpoint whose CI is within frac of the run's max CI (CI saturation)"""
    m=max(r['ci'] for r in v); return first_where(v, lambda r: r['ci']>=frac*m)
def rule_pp_drop(v, d=0.005):
    """last checkpoint before prediction_performance has fallen d below its running max"""
    best=v[0]['pp']; sel=v[0]
    for r in v:
        if r['pp'] < best-d: return sel
        best=max(best,r['pp']); sel=r
    return sel
def rule_ri_flat(v, eps=0.002):
    """first checkpoint after which RI improves by < eps"""
    for i in range(len(v)-1):
        if v[i+1]['ri']-v[i]['ri'] < eps: return v[i]
    return v[-1]
def rule_l2_flat(v, eps=0.02):
    for i in range(len(v)-1):
        if v[i+1]['l2']-v[i]['l2'] < eps: return v[i]
    return v[-1]
def rule_ci_over_pp(v):
    """argmax of CI x PP (joint) -- balances confounder removal against accuracy erosion"""
    return max(v, key=lambda r: r['ci']*r['pp'])

def report(name, fn, note=""):
    per={}
    for b in BB:
        sel=[]; orc=[]; early=[]
        for k,v in runs.items():
            if v[0]['backbone']!=b: continue
            s=fn(v); sel.append(s); orc.append(max(x['hest_pct'] for x in v))
        if not sel: per[b]=None; continue
        per[b]=dict(n=len(sel), hest=st.mean(x['hest_pct'] for x in sel),
                    minhest=min(x['hest_pct'] for x in sel),
                    ri=st.mean(x['ri_pct'] for x in sel),
                    regret=st.mean(o-x['hest_pct'] for o,x in zip(orc,sel)),
                    steps=sorted({x['step'] for x in sel}))
    worst=min(p['hest'] for p in per.values() if p)
    line=f"{name:<40} worstBB_meanHEST%={worst:6.1f} | "
    for b in BB:
        p=per[b]
        line+=f"{b[:4]}: H={p['hest']:5.1f} RI={p['ri']:5.1f} reg={p['regret']:4.1f} n={p['n']} | " if p else f"{b[:4]}: -- | "
    print(line)
    return worst, per

print("RULE EVALUATION on the 27 runs with >=2 HEST'd checkpoints")
print("(H = mean HEST pct_of_waiv at selected step; RI = mean RI pct_of_waiv; reg = mean regret vs per-run oracle)")
print("-"*150)
results=[]
results.append(("oracle (cheats)", *report("ORACLE  argmax HEST (upper bound)", lambda v: max(v,key=lambda r:r['hest_pct']))))
results.append(("earliest", *report("R0  stop at EARLIEST checkpoint", rule_earliest)))
results.append(("latest", *report("R0b stop at LAST checkpoint", lambda v: v[-1])))
report("R1  argmax avg_robustness_index", rule_argmax_ri)
report("R2  argmax mean confounder_insens", rule_argmax_ci)
report("R3  argmax mean prediction_perf", rule_argmax_pp)
report("R4  CI-saturation (first CI>=0.98*max)", rule_ci_sat)
report("R5  RI-flat (first ARI gain<0.002)", rule_ri_flat)
report("R6  L2-flat (first dL2<0.02)", rule_l2_flat)
report("R7  argmax CI*PP", rule_ci_over_pp)
for d in (0.0,0.002,0.005,0.01):
    report(f"R8  PP-drop guard d={d}", lambda v,d=d: rule_pp_drop(v,d))
print("-"*150)
print("threshold sweeps (first checkpoint crossing the threshold):")
for c in (0.60,0.70,0.75,0.80,0.85,0.90,0.95):
    report(f"R9  first CI >= {c}", mk_thresh('ci',c))
for c in (0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05):
    report(f"R10 first L2 >= {c}", mk_thresh('l2',c))
for c in (0.90,0.94,0.96,0.98,1.00,1.02):
    report(f"R11 first RI >= {c}*Waiv target (ri_pct)", lambda v,c=c: first_where(v, lambda r: r['ri_pct'] is not None and r['ri_pct']>=100*c))

print("-"*150); print("COMPOSITE rules:")
def comp_min(v):
    a=mk_thresh('ci',0.75)(v); b=rule_pp_drop(v,0.0)
    return a if a['step']<=b['step'] else b
def comp_ci_guard(v):
    """first CI>=0.75; but if CI ever >=0.93 stop no later than the last ckpt below 0.93"""
    a=mk_thresh('ci',0.75)(v)
    below=[r for r in v if r['ci']<0.93]
    if below and a['step']>below[-1]['step']: return below[-1]
    return a
def comp_ci_ppguard(v):
    a=mk_thresh('ci',0.75)(v); b=rule_pp_drop(v,0.005)
    return a if a['step']<=b['step'] else b
report("C1  min(CI>=0.75, PP-drop d=0)", comp_min)
report("C2  CI>=0.75 with CI<0.93 ceiling", comp_ci_guard)
report("C3  min(CI>=0.75, PP-drop d=0.005)", comp_ci_ppguard)
report("C4  first CI>=0.75 (repeat, best single)", mk_thresh('ci',0.75))
print()
print("Per-run detail for C4 (first CI>=0.75) vs oracle:")
for b in BB:
    print(f"  --- {b} ---")
    for k,v in sorted(runs.items()):
        if v[0]['backbone']!=b: continue
        s=mk_thresh('ci',0.75)(v); o=max(v,key=lambda r:r['hest_pct'])
        print(f"    {k[:52]:<52} pick step {s['step']:>5} HEST={s['hest_pct']:6.1f} | oracle step {o['step']:>5} HEST={o['hest_pct']:6.1f} | regret {o['hest_pct']-s['hest_pct']:5.1f}")
