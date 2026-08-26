#!/usr/bin/env python3
"""Join per-checkpoint RI-curve internals with HEST summaries.
Emits docs/stopping_criterion_rows.json  (one row per (run, step) with both signals)."""
import json, os, re, glob, statistics as st, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import collect_final5 as _c5   # noqa: E402
import eval_common as _ec      # noqa: E402

RUNS = os.path.join(REPO, "runs")
HEST = str(_c5.HEST_WORK_DIR / "results")

# F-E/F-F fix (2026-08-26): every one of these used to be a literal copied from
# collect_final5, and two of them were the WRONG number -- virchow2's HEST base was the
# rounded `results.avg` 0.40324 rather than the `custom_encoder` 0.4032685 that the
# numerator below is now read from, and the RI base carried collect_final5's false
# probe_before.json provenance.  There is one loader for each of these now.
HEST_BASE = _c5.HEST_BASE
HEST_WAIV = _ec.HEST_WAIV
RI_BASE   = _c5.RI_BASE
RI_WAIV   = _c5.RI_WAIV
POOL      = {a: _c5.hest_pooling(a) for a in ("phikon", "midnight", "virchow2")}

def backbone_of(run, cfg):
    bb = (cfg or {}).get("encoder", {}).get("backbone", "")
    if "phikon" in bb: return "phikon"
    if "midnight" in bb.lower() or "Midnight" in bb: return "midnight"
    if "virchow" in bb.lower(): return "virchow2"
    for k in ("virchow2", "midnight", "phikon"):
        if k in run: return k
    return None

def family(run, cfg):
    """coarse recipe family from config knobs"""
    c = cfg or {}
    bits = []
    ms = c.get("max_steps")
    lr = c.get("lr")
    bits.append(f"lr{lr:g}" if lr else "lr?")
    bits.append(f"ms{ms}")
    if c.get("mask_same_core") or c.get("mask_core") or c.get("core_map"): bits.append("mask")
    cw = c.get("cls_weight"); 
    if cw not in (None, 0.5): bits.append(f"cls{cw:g}")
    if c.get("retention_kl_weight"): bits.append(f"kl{c['retention_kl_weight']:g}")
    t = c.get("temperature")
    if t not in (None, 0.07): bits.append(f"t{t:g}")
    return "-".join(bits)

rows = []
skipped = {"no_curve":0,"no_hest":0,"no_bb":0}
for run in sorted(os.listdir(RUNS)):
    rd = os.path.join(RUNS, run)
    cp = os.path.join(rd, "ri_curve.json")
    if not os.path.isfile(cp): continue
    try: curve = json.load(open(cp))
    except Exception: skipped["no_curve"] += 1; continue
    cfgp = os.path.join(rd, "config.json")
    cfg = json.load(open(cfgp)) if os.path.isfile(cfgp) else {}
    bb = backbone_of(run, cfg)
    if bb is None: skipped["no_bb"] += 1; continue
    pool = POOL[bb]
    found_any = False
    for pt in curve.get("points", []):
        step = pt.get("step")
        if step is None: continue
        hp = f"{HEST}/f5_{run}_s{step:07d}_{pool}_summary.json"
        if not os.path.isfile(hp):
            continue
        try: hs = json.load(open(hp))
        except Exception: continue
        # F-E: the SAME field the base is read from -- custom_encoder, not results.avg.
        havg = (hs.get("hest_perf_per_encoder") or {}).get("custom_encoder")
        if havg is None: continue
        found_any = True
        l2 = pt.get("adapter_rel_l2_delta")
        if isinstance(l2, dict):
            l2vals = {k: v for k, v in l2.items() if isinstance(v, (int, float))}
            l2s = st.mean(l2vals.values()) if l2vals else None
        else:
            l2vals = {}; l2s = l2 if isinstance(l2, (int, float)) else None
        ds = pt.get("datasets", {})
        def dmean(field):
            vs = [d[field] for d in ds.values() if isinstance(d, dict) and isinstance(d.get(field), (int, float))]
            return st.mean(vs) if vs else None
        ri = pt.get("avg_robustness_index")
        tm = pt.get("train_metrics") or {}
        rows.append(dict(
            run=run, backbone=bb, family=family(run, cfg),
            max_steps=cfg.get("max_steps"), lr=cfg.get("lr"),
            annealed=(cfg.get("max_steps") == step),
            step=step, pooling=pool,
            l2=l2s, l2_by_ds=l2vals,
            ri=ri,
            ri_pct=None if ri is None else 100*(ri-RI_BASE[bb])/(RI_WAIV[bb]-RI_BASE[bb]),
            ci=dmean("confounder_insensitivity"), pp=dmean("prediction_performance"),
            gi=dmean("generalization_index"), idp=dmean("ID_performance"),
            oodp=dmean("OOD_performance"), bacc=pt.get("avg_balanced_accuracy"),
            loss=tm.get("loss"), top1=tm.get("top1"),
            heldout_loss=tm.get("heldout_loss"), heldout_top1=tm.get("heldout_top1"),
            loss_cls=tm.get("loss_cls"), loss_mean=tm.get("loss_mean"),
            hest=havg,
            hest_pct=100*(havg-HEST_BASE[bb])/(HEST_WAIV[bb]-HEST_BASE[bb]),
        ))
    if not found_any: skipped["no_hest"] += 1

out = "/admin/home/ryan.kim/waiv/docs/stopping_criterion_rows.json"
json.dump(rows, open(out, "w"), indent=1)
from collections import Counter
print("rows", len(rows), "runs", len({r['run'] for r in rows}))
print("per backbone rows:", Counter(r['backbone'] for r in rows))
print("per backbone runs:", {b: len({r['run'] for r in rows if r['backbone']==b}) for b in ('phikon','midnight','virchow2')})
print("skipped", skipped)
print("missing l2:", sum(1 for r in rows if r['l2'] is None))
print("steps seen:", sorted(Counter(r['step'] for r in rows).items()))
print("->", out)
