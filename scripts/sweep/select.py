#!/usr/bin/env python
"""Pick the stop step of each Q8 SELECTION run with the slope rule (spectra.sweep.stopping).

    python scripts/sweep/select.py <run_dir> [<run_dir> ...] [--theta 0.016] [--json picks.json]
    python scripts/sweep/select.py --sweep q8_selection          # every run dir of a sweep

Input = each run's OWN in-run PLISM held-out primary (metrics.jsonl / posthoc.jsonl), step 0
included; nothing outside PLISM is read. The rule needs the step-0 eval and every 25-step eval
up to the last candidate (250). Per run it prints the pick, the slope at every candidate and the
GTEx utility guard at the pick next to the step-0 guard (``guard_ok``: guard(pick) >=
guard(0) - max(0.010, guard_se(0))). The guard is reported, it does not move the pick.

Then: python scripts/sweep/make_final.py --backbone <bb> --picks <pick[,pick...]>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from spectra.sweep import stopping as R  # noqa: E402
from spectra.sweep.config import runs_root  # noqa: E402

GUARD_DELTA = 0.010


def eval_records(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for fn in ("metrics.jsonl", "posthoc.jsonl"):
        p = run_dir / fn
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "eval":
                    out[int(r["step"])] = r
    return out


def guard_check(recs: dict[int, dict], step: int) -> dict:
    g0, gp = recs.get(0, {}), recs.get(step, {})
    if "guard_lin_bacc" not in g0 or "guard_lin_bacc" not in gp:
        return {"guard_ok": None}
    tol = max(GUARD_DELTA, float(g0.get("guard_lin_bacc_se", 0.0)))
    return {"guard_0": g0["guard_lin_bacc"], "guard_pick": gp["guard_lin_bacc"],
            "guard_ok": gp["guard_lin_bacc"] >= g0["guard_lin_bacc"] - tol}


def pick_run(run_dir: Path, theta: float = R.THETA) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    P = R.read_primary(run_dir)
    row = {"run": run_dir.name, "backbone": cfg["backbone"]["name"], "seed": cfg["run"]["seed"],
           "complete": R.complete(P), "pick": None}
    if not row["complete"]:
        have = sorted(P)
        row["missing"] = [s for s in [0, *range(R.EVAL_EVERY, R.CANDIDATES[-1] + 1, R.EVAL_EVERY)] if s not in P]
        row["have"] = have
        return row
    pk, slopes = R.slope_pick(P, R.CANDIDATES, theta)
    row.update(pick=pk, primary_pick=P[pk], primary_0=P[0],
               slopes={s: (None if v is None else round(v, 5)) for s, v in slopes.items()},
               **guard_check(eval_records(run_dir), pk))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--sweep", default=None, help="every run dir under <runs root>/<sweep>")
    ap.add_argument("--theta", type=float, default=R.THETA)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    dirs = list(a.run_dirs)
    if a.sweep:
        dirs += sorted(d for d in (runs_root() / a.sweep).iterdir() if (d / "config.json").exists())
    if not dirs:
        ap.error("no run dirs")
    rows = [pick_run(d, a.theta) for d in dirs]
    for r in rows:
        if r["pick"] is None:
            print(f"{r['run']:40s} INCOMPLETE: missing in-run evals at steps {r.get('missing')}")
            continue
        g = "" if r.get("guard_ok") is None else \
            f" guard {r['guard_pick']:.4f} (step 0 {r['guard_0']:.4f}) {'ok' if r['guard_ok'] else 'FAILS'}"
        print(f"{r['run']:40s} {r['backbone']:9s} seed {r['seed']}: pick {r['pick']:4d}  "
              f"P {r['primary_0']:.4f} -> {r['primary_pick']:.4f}{g}")
        print("    slope/25 steps: " + " ".join(f"{s}:{v}" for s, v in r["slopes"].items()))
    by_bb: dict[str, list[int]] = {}
    for r in rows:
        if r["pick"] is not None:
            by_bb.setdefault(r["backbone"], []).append(r["pick"])
    for bb, pk in sorted(by_bb.items()):
        print(f"\n{bb}: python scripts/sweep/make_final.py --backbone {bb} --picks {','.join(map(str, pk))}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
