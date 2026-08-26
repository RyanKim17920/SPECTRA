#!/usr/bin/env python3
"""Empirical seed-noise error bars on the AGGREGATE pct_of_waiv criterion.

Reproduces docs/aggregate_criterion_resolvability.md.  Reads only data already on
disk: docs/thunder_seed_floor_12ds.json (THUNDER per-seed/per-dataset F1),
/data/ryan.kim/thunder/outputs/res (THUNDER bases), /data/ryan.kim/hest_work/results
(HEST final5 summaries), runs/final5-*/ri_curve.json (RI @ step 500).

Usage:  python scripts/aggregate_criterion_resolvability.py
"""
import json, math, statistics as st, sys
from pathlib import Path
from itertools import combinations
from itertools import combinations

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import collect_final5 as _c5   # noqa: E402
import eval_common as _ec      # noqa: E402
BB = ['phikon','midnight','virchow2']
TASKS = ['knn','linear_probing','simple_shot']
SEEDS = [0,1,2,3,4]

# ---------------- THUNDER ----------------
J = json.load(open(REPO/'docs/thunder_seed_floor_12ds.json'))
DS12 = J['datasets_12']
ps = J['per_seed_scores']          # key "bb/task" -> seed -> ds -> f1
WAIV_T = {
 "phikon":{"base":{"knn":74.0,"linear_probing":79.3,"simple_shot":71.8},
           "ft":{"knn":77.7,"linear_probing":80.7,"simple_shot":73.3}},
 "midnight":{"base":{"knn":80.0,"linear_probing":84.4,"simple_shot":71.5},
             "ft":{"knn":81.7,"linear_probing":84.6,"simple_shot":75.2}},
 "virchow2":{"base":{"knn":82.9,"linear_probing":84.8,"simple_shot":73.9},
             "ft":{"knn":82.6,"linear_probing":85.1,"simple_shot":76.6}},
}
OUR_BASE_T = {  # measured, 12ds mean, frozen, f1 -- recomputed this session
 ('phikon','knn'):0.70281, ('phikon','linear_probing'):0.76541, ('phikon','simple_shot'):0.69330,
 ('midnight','knn'):0.78254, ('midnight','linear_probing'):0.82880, ('midnight','simple_shot'):0.70639,
 ('virchow2','knn'):0.80874, ('virchow2','linear_probing'):0.83253, ('virchow2','simple_shot'):0.72749,
}
# ours per seed = 12ds task mean
T_ours = {}
for b in BB:
    for t in TASKS:
        d = ps[f'{b}/{t}']
        for s in SEEDS:
            T_ours[(b,t,s)] = st.mean(d[str(s)][ds] for ds in DS12)

def pct_t(b,t,s,cap=True):
    den = (WAIV_T[b]['ft'][t]-WAIV_T[b]['base'][t])/100.0
    v = (T_ours[(b,t,s)]-OUR_BASE_T[(b,t)])/den*100.0
    return min(v,100.0) if cap else v

# ---------------- HEST ----------------
# F-E fix (2026-08-26): one loader, one field.  virchow2's base was the ROUNDED
# `results.avg` 0.40324 while the numerator below read `results.avg` too -- self-
# consistent but inconsistent with collect_final5 / scoreboard / final_recipe_report,
# which all read `hest_perf_per_encoder.custom_encoder` (0.4032685).  Both ends now come
# off the same field through the same loader.
HEST_BASE=_c5.HEST_BASE
WAIV_HEST=_ec.HEST_WAIV
HRES=_c5.HEST_WORK_DIR/'results'
POOL={a:_c5.hest_pooling(a) for a in BB}
H_ours={}
for b in BB:
    for s in SEEDS:
        g=sorted(HRES.glob(f'f5_final5-{b}-s{s}-t900-*_s0000500_{POOL[b]}_summary.json'))
        assert len(g)==1,(b,s,g)
        H_ours[(b,s)]=_c5._hest_read_metric(g[0])          # F-E: custom_encoder, not results.avg
def pct_h(b,s,cap=True):
    v=(H_ours[(b,s)]-HEST_BASE[b])/(WAIV_HEST[b]-HEST_BASE[b])*100.0
    return min(v,100.0) if cap else v

# ---------------- RI ----------------
# F-F fix: read from PathoROB results on disk; the literals' cited provenance
# (probe_before.json) had no such field.
RI_BASE=_c5.RI_BASE
WAIV_RI=_c5.RI_WAIV
R_ours={}
for b in BB:
    for s in SEEDS:
        g=sorted((REPO/'runs').glob(f'final5-{b}-s{s}-t900-*/ri_curve.json'))
        assert len(g)==1,(b,s,g)
        pts=json.load(open(g[0]))['points']
        v=[p for p in pts if p['step']==500]
        assert len(v)==1
        R_ours[(b,s)]=v[0]['avg_robustness_index']
def pct_r(b,s,cap=True):
    v=(R_ours[(b,s)]-RI_BASE[b])/(WAIV_RI[b]-RI_BASE[b])*100.0
    return min(v,100.0) if cap else v

# ================= reporting helpers =================
def sd(xs): return st.stdev(xs) if len(xs)>1 else float('nan')

OUT={}
def show(title,rows):
    print('\n### '+title)
    for r in rows: print(r)

print('='*70); print('RAW PER-SEED pct_of_waiv (uncapped)')
for b in BB:
    for t in TASKS:
        vs=[pct_t(b,t,s,cap=False) for s in SEEDS]
        print(f'THUNDER {b:9s} {t:15s} ' + ' '.join(f'{v:8.1f}' for v in vs) + f'  mean {st.mean(vs):8.1f} sd {sd(vs):6.2f}')
for b in BB:
    vs=[pct_h(b,s,cap=False) for s in SEEDS]
    print(f'HEST    {b:9s} {"avg":15s} ' + ' '.join(f'{v:8.1f}' for v in vs) + f'  mean {st.mean(vs):8.1f} sd {sd(vs):6.2f}')
for b in BB:
    vs=[pct_r(b,s,cap=False) for s in SEEDS]
    print(f'RI      {b:9s} {"avg_ri":15s} ' + ' '.join(f'{v:8.1f}' for v in vs) + f'  mean {st.mean(vs):8.1f} sd {sd(vs):6.2f}')

# raw underlying quantities
print('\nRAW underlying (12ds f1 / hest avg / ri) per seed')
for b in BB:
    for t in TASKS:
        vs=[T_ours[(b,t,s)] for s in SEEDS]
        print(f'  T {b:9s} {t:15s} base={OUR_BASE_T[(b,t)]:.5f} ours={[round(v,5) for v in vs]} sd={sd(vs):.5f} waivgain={(WAIV_T[b]["ft"][t]-WAIV_T[b]["base"][t])/100:.4f}')
for b in BB:
    vs=[H_ours[(b,s)] for s in SEEDS]
    print(f'  H {b:9s} base={HEST_BASE[b]:.5f} ours={[round(v,5) for v in vs]} sd={sd(vs):.5f} waivgain={WAIV_HEST[b]-HEST_BASE[b]:.4f}')
for b in BB:
    vs=[R_ours[(b,s)] for s in SEEDS]
    print(f'  R {b:9s} base={RI_BASE[b]:.5f} ours={[round(v,5) for v in vs]} sd={sd(vs):.5f} waivgain={WAIV_RI[b]-RI_BASE[b]:.4f}')

# ============ AGGREGATES ============
DEGEN = [('midnight','linear_probing'),('virchow2','knn'),('virchow2','linear_probing')]
def agg_thunder(s, cap=True, exclude_degen=False, scheme='flat9'):
    cells=[(b,t) for b in BB for t in TASKS]
    if exclude_degen: cells=[c for c in cells if c not in DEGEN]
    if scheme=='flat9':
        return st.mean(pct_t(b,t,s,cap) for b,t in cells)
    else: # per-backbone mean then mean over backbones
        per=[]
        for b in BB:
            ts=[t for t in TASKS if (b,t) in cells]
            if ts: per.append(st.mean(pct_t(b,t,s,cap) for t in ts))
        return st.mean(per)

def agg_hest(s,cap=True): return st.mean(pct_h(b,s,cap) for b in BB)
def agg_ri(s,cap=True):   return st.mean(pct_r(b,s,cap) for b in BB)

print('\n'+'='*70); print('AGGREGATES PER SEED-INDEX (index-matched across backbones)')
def rep(name, fn):
    vs=[fn(s) for s in SEEDS]
    s_=sd(vs); 
    print(f'{name:52s} vals={[round(v,2) for v in vs]} mean={st.mean(vs):7.2f} SD={s_:6.3f} 2SD={2*s_:6.3f}')
    return vs
res={}
for cap in (True,False):
    tag='capped' if cap else 'uncapped'
    for ex in (False,True):
        etag='incl-degen' if not ex else 'excl-degen'
        for sch in ('flat9','nested'):
            k=f'THUNDER {tag} {etag} {sch}'
            res[k]=rep(k, lambda s,c=cap,e=ex,x=sch: agg_thunder(s,c,e,x))
for cap in (True,False):
    tag='capped' if cap else 'uncapped'
    res[f'HEST {tag}']=rep(f'HEST {tag}', lambda s,c=cap: agg_hest(s,c))
    res[f'RI {tag}']=rep(f'RI {tag}', lambda s,c=cap: agg_ri(s,c))
# overall mean of three
for cap in (True,False):
    tag='capped' if cap else 'uncapped'
    res[f'MEAN3 {tag} incl-degen flat9']=rep(f'MEAN3 {tag} incl-degen flat9',
        lambda s,c=cap: st.mean([agg_thunder(s,c,False,'flat9'),agg_hest(s,c),agg_ri(s,c)]))
    res[f'MEAN3 {tag} excl-degen flat9']=rep(f'MEAN3 {tag} excl-degen flat9',
        lambda s,c=cap: st.mean([agg_thunder(s,c,True,'flat9'),agg_hest(s,c),agg_ri(s,c)]))

# ---- variance decomposition estimator (uses independence ACROSS BACKBONES only) ----
print('\n'+'='*70); print('DECOMPOSED SD (independence across backbones ONLY; within-backbone correlation measured)')
def decomposed_sd_thunder(cap=True, exclude_degen=False):
    # aggregate = (1/9) sum_{b,t} pct  = (1/3) sum_b [ (1/k_b) sum_t pct ]  weight w_b
    cells=[(b,t) for b in BB for t in TASKS]
    if exclude_degen: cells=[c for c in cells if c not in DEGEN]
    n=len(cells)
    var=0.0
    for b in BB:
        ts=[t for t in TASKS if (b,t) in cells]
        if not ts: continue
        per_seed=[sum(pct_t(b,t,s,cap) for t in ts)/n for s in SEEDS]  # contribution of backbone b
        var += st.variance(per_seed)
    return math.sqrt(var)
def decomposed_sd_simple(fn_pct, cap=True):
    var=0.0
    for b in BB:
        per_seed=[fn_pct(b,s,cap)/3.0 for s in SEEDS]
        var+=st.variance(per_seed)
    return math.sqrt(var)
for cap in (True,False):
    tag='capped' if cap else 'uncapped'
    for ex in (False,True):
        etag='incl' if not ex else 'excl'
        v=decomposed_sd_thunder(cap,ex); print(f'THUNDER {tag} {etag}-degen flat9  decomposed SD={v:6.3f}  2SD={2*v:6.3f}')
    v=decomposed_sd_simple(pct_h,cap); print(f'HEST    {tag}                  decomposed SD={v:6.3f}  2SD={2*v:6.3f}')
    v=decomposed_sd_simple(pct_r,cap); print(f'RI      {tag}                  decomposed SD={v:6.3f}  2SD={2*v:6.3f}')
# MEAN3 decomposed
for cap in (True,False):
    tag='capped' if cap else 'uncapped'
    for ex in (False,True):
        cells=[(b,t) for b in BB for t in TASKS]
        if ex: cells=[c for c in cells if c not in DEGEN]
        n=len(cells); var=0.0
        for b in BB:
            ts=[t for t in TASKS if (b,t) in cells]
            contrib=[(sum(pct_t(b,t,s,cap) for t in ts)/n + pct_h(b,s,cap)/3 + pct_r(b,s,cap)/3)/3 for s in SEEDS]
            var+=st.variance(contrib)
        v=math.sqrt(var); print(f'MEAN3   {tag} {"incl" if not ex else "excl"}-degen  decomposed SD={v:6.3f}  2SD={2*v:6.3f}')

# ---- empirical correlations within backbone across tasks ----
print('\n'+'='*70); print('EMPIRICAL WITHIN-BACKBONE CROSS-TASK CORRELATION of raw 12ds task mean (n=5 seeds)')
def pearson(x,y):
    mx,my=st.mean(x),st.mean(y)
    num=sum((a-mx)*(b-my) for a,b in zip(x,y))
    den=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/den if den else float('nan')
for b in BB:
    for t1,t2 in combinations(TASKS,2):
        x=[T_ours[(b,t1,s)] for s in SEEDS]; y=[T_ours[(b,t2,s)] for s in SEEDS]
        print(f'  {b:9s} {t1:15s} vs {t2:15s} r={pearson(x,y):+.3f}')
print(' cross-benchmark within backbone (THUNDER task-mean vs HEST vs RI):')
for b in BB:
    tm=[st.mean(T_ours[(b,t,s)] for t in TASKS) for s in SEEDS]
    h=[H_ours[(b,s)] for s in SEEDS]; r=[R_ours[(b,s)] for s in SEEDS]
    print(f'  {b:9s} T~H r={pearson(tm,h):+.3f}  T~RI r={pearson(tm,r):+.3f}  H~RI r={pearson(h,r):+.3f}')

# ---- seeds needed ----
print('\n'+'='*70); print('SEEDS NEEDED for aggregate 2SE < target')
def seeds_needed(sd1,target): 
    # 2*sd1/sqrt(k) < target
    return math.ceil((2*sd1/target)**2)
for label,sd1 in [('THUNDER incl flat9', decomposed_sd_thunder(True,False)),
                  ('THUNDER excl flat9', decomposed_sd_thunder(True,True)),
                  ('HEST', decomposed_sd_simple(pct_h,True)),
                  ('RI', decomposed_sd_simple(pct_r,True))]:
    print(f'{label:22s} SD1={sd1:6.3f} 2SE@n=1={2*sd1:6.2f}  MDD(n=1)={2*math.sqrt(2)*sd1:6.2f}  '
          f'k for 2SE<10: {seeds_needed(sd1,10)}  k for 2SE<5: {seeds_needed(sd1,5)}')


# ---- Satterthwaite effective df and 95% CI on the 2SE itself ----------------
print('\n'+'='*70); print('VARIANCE COMPONENTS, EFFECTIVE df, 95% CI ON 2SE')
try:
    from scipy.stats import chi2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _components_thunder(cap=True, ex=False):
    cells = [(b, t) for b in BB for t in TASKS]
    if ex:
        cells = [c for c in cells if c not in DEGEN]
    n = len(cells); comps = {}
    for b in BB:
        ts = [t for t in TASKS if (b, t) in cells]
        if not ts:
            continue
        comps[b] = st.variance([sum(pct_t(b, t, s, cap) for t in ts) / n for s in SEEDS])
    return comps

def _components_simple(fnp, cap=True):
    return {b: st.variance([fnp(b, s, cap) / 3.0 for s in SEEDS]) for b in BB}

def _components_mean3(cap=True, ex=False):
    cells = [(b, t) for b in BB for t in TASKS]
    if ex:
        cells = [c for c in cells if c not in DEGEN]
    n = len(cells); comps = {}
    for b in BB:
        ts = [t for t in TASKS if (b, t) in cells]
        comps[b] = st.variance([
            (sum(pct_t(b, t, s, cap) for t in ts) / n + pct_h(b, s, cap) / 3 + pct_r(b, s, cap) / 3) / 3
            for s in SEEDS])
    return comps

def _satter(comps, df_each=4):
    V = sum(comps.values())
    den = sum(v * v / df_each for v in comps.values())
    df = V * V / den if den > 0 else float('nan')
    s_ = math.sqrt(V)
    if _HAVE_SCIPY:
        lo = s_ * math.sqrt(df / chi2.ppf(0.975, df)); hi = s_ * math.sqrt(df / chi2.ppf(0.025, df))
    else:
        lo = hi = float('nan')
    return s_, df, lo, hi

for _name, _c in [
    ('THUNDER capped incl-degen',   _components_thunder(True, False)),
    ('THUNDER capped excl-degen',   _components_thunder(True, True)),
    ('THUNDER uncapped excl-degen', _components_thunder(False, True)),
    ('HEST capped',                 _components_simple(pct_h, True)),
    ('RI capped',                   _components_simple(pct_r, True)),
    ('MEAN3 capped incl-degen',     _components_mean3(True, False)),
    ('MEAN3 capped excl-degen',     _components_mean3(True, True)),
]:
    _s, _df, _lo, _hi = _satter(_c)
    _tot = sum(_c.values())
    print(f'{_name:30s} SD1={_s:6.3f} 2SE(n=1)={2*_s:6.2f} '
          f'[95% CI {2*_lo:5.2f}, {2*_hi:6.2f}] eff_df={_df:5.1f}  '
          f'var share {[f"{b}:{v/_tot*100:.0f}%" for b, v in _c.items()]}')
