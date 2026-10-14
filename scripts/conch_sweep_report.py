"""Sweep table for SPECTRA-CONCH arms: conch_eval.json per step + PathoROB RI where computed.

    python scripts/conch_sweep_report.py anc1 anc5 anc10 pc5 pc10
"""
import json
import sys
from pathlib import Path

RUNS = Path("/data/ryan.kim/ashby/runs/spectra_conch")
PR = Path("/data/ryan.kim/ashby/scratch/spectra_conch_pathorob/results")
DS = ("camelyon", "tcga", "tolkach_esca")


def ri(name):
    out = {}
    for ds in DS:
        p = PR / name / ds / "-1_0" / "results_summary.json"
        if p.exists():
            out[ds] = json.loads(p.read_text())["robustness_index"]
    return out


def main():
    arms = sys.argv[1:] or ["anc1", "anc5", "anc10", "pc5", "pc10"]
    b = json.loads((RUNS / "anc1" / "conch_eval.json").read_text())["0"]; rb = ri("conch_base")
    rows = [("base", 0, b, rb)]
    for a in arms:
        p = RUNS / a / "conch_eval.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for st in sorted((int(s) for s in d if s != "0")):
            rows.append((a, st, d[str(st)], ri(f"{a}_s{st}")))
    print(f"{'arm':6s} {'step':>4s} {'xstain':>6s} {'d':>6s} {'xscan':>6s} {'zs':>6s} {'drift':>6s} {'probe':>6s} {'RI cam':>6s} {'RI tcga':>7s} {'RI esca':>7s} {'RI avg':>6s}  gate")
    out = []
    for a, st, r, R in rows:
        avg = sum(R.values()) / 3 if len(R) == 3 else None
        gate = (r["plism_cross_stain_top1"] - b["plism_cross_stain_top1"] >= 0.15) and r["zs_acc"] >= 0.73
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{a:6s} {st:4d} {r['plism_cross_stain_top1']:.3f} {r['plism_cross_stain_top1']-b['plism_cross_stain_top1']:+.3f} "
              f"{r['plism_cross_scanner_top1']:.3f} {r['zs_acc']:.3f} {r.get('drift_cos', 1):.3f} {r.get('probe_acc', float('nan')):.3f} "
              f"{f(R.get('camelyon')):>6s} {f(R.get('tcga')):>7s} {f(R.get('tolkach_esca')):>7s} {f(avg):>6s}  {'PASS' if gate else ''}")
        out.append(dict(arm=a, step=st, **{k: v for k, v in r.items() if not isinstance(v, dict)}, ri=R, ri_avg=avg, gate=gate))
    Path("/data/ryan.kim/ashby/results/spectra_conch").mkdir(parents=True, exist_ok=True)
    Path("/data/ryan.kim/ashby/results/spectra_conch/sweep_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
