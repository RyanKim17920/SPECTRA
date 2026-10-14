"""SPECTRA-CONCH v2 report: per-checkpoint PathoROB RI per arm x readout, the paper's online
1-SE pick (SE 0.007, same rule as ~/SPECTRA scripts/final_recipe_report.py::select_step_1se),
and conch_eval metrics at the pick.

    python scripts/conch_v2_report.py A_export-ninf_s0:A B_split-paper_s0:B C_split-paper_s1:C
"""
import json
import sys
from pathlib import Path

RUNS = Path("/data/ryan.kim/ashby/runs/spectra_conch_v2")
PR = Path("/data/ryan.kim/ashby/scratch/spectra_conch_pathorob/results")
DS = ("tcga", "camelyon", "tolkach_esca")
SE = 0.007


def ri(name):
    out = {}
    for ds in DS:
        p = PR / name / ds / "-1_0" / "results_summary.json"
        if p.exists():
            out[ds] = json.loads(p.read_text())["robustness_index"]
    return out


def select_1se(points, se=SE):
    """points: [(step, ri_avg)] step-ordered. Online 1-SE: B=best so far; STOP at first t>=2nd
    with R_t - B <= SE; RETURN the earliest seen checkpoint with RI >= B - SE."""
    best = None
    for i, (st, r) in enumerate(points):
        if best is None:
            best = r; continue
        if r - best <= se:
            best = max(best, r)
            return next(s for s, q in points[:i + 1] if q >= best - se), "STOP"
        best = max(best, r)
    return None, "UNTERMINATED"


def main():
    rep = {}
    for spec in sys.argv[1:]:
        run, tag = spec.split(":")
        for ro, sfx in (("export", ""), ("clsmean", "_clsmean")):
            base = ri("conch_base" + sfx)
            rows = []
            for st in range(50, 501, 50):
                R = ri(f"{tag}_s{st}{sfx}")
                if len(R) == 3:
                    rows.append((st, R, sum(R.values()) / 3))
            if not rows:
                continue
            pick, why = select_1se([(s, a) for s, _, a in rows])
            ev = RUNS / run / f"conch_eval{sfx}.json"
            ev = json.loads(ev.read_text()) if ev.exists() else {}
            print(f"\n== {tag} ({run}) readout={ro}  base RI tcga/cam/esca/avg = "
                  + "/".join(f"{base.get(d, float('nan')):.3f}" for d in DS) + f"/{sum(base.values())/3:.3f}"
                  + f"   1-SE pick = {pick} ({why})")
            print(f"{'step':>4} {'tcga':>6} {'cam':>6} {'esca':>6} {'avg':>6} {'d_avg':>6} | {'xscan':>6} {'xstain':>6} {'zs':>6} {'probe':>6} {'drift':>6}")
            b0 = ev.get("0", {})
            allrows = [(0, base, sum(base.values()) / 3)] + rows
            for st, R, a in allrows:
                e = ev.get(str(st), b0 if st == 0 else {})
                g = lambda k: f"{e[k]:.3f}" if k in e else "  -  "
                mark = " <== 1-SE" if st == pick else ""
                print(f"{st:4d} {R['tcga']:.3f} {R['camelyon']:.3f} {R['tolkach_esca']:.3f} {a:.3f} {a - allrows[0][2]:+.3f} | "
                      f"{g('plism_cross_scanner_top1'):>6} {g('plism_cross_stain_top1'):>6} {g('zs_acc'):>6} {g('probe_acc'):>6} {g('drift_cos'):>6}{mark}")
            rep[f"{tag}_{ro}"] = dict(run=run, readout=ro, base=base, pick=pick, why=why,
                                      rows=[dict(step=s, ri=R, avg=a, eval=ev.get(str(s))) for s, R, a in rows],
                                      base_eval=b0)
    out = Path("/data/ryan.kim/ashby/results/spectra_conch/v2_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
