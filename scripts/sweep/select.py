#!/usr/bin/env python
"""R1 stop step of each v2 SELECTION run (spectra.sweep.stopping).

    python scripts/sweep/select.py <run_dir> [<run_dir> ...] [--json picks.json]
    python scripts/sweep/select.py --sweep selection          # every run dir of a sweep

Input = each run's OWN in-run held-out PLISM primary P (metrics.jsonl / posthoc.jsonl), one
value per saved checkpoint 25, 50, ..., 250. R1 = argmax_{s >= 25} P(s), ties -> the earlier
step. Nothing outside PLISM is read (no labels, no benchmark). The rule needs every candidate
step; a run with a missing eval is reported INCOMPLETE (fill it with posthoc_eval.py).

Then: python scripts/sweep/make_final.py --backbone <bb> --step <pick>
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


def pick_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    P = R.read_primary(run_dir)
    row = {"run": run_dir.name, "backbone": cfg["backbone"]["name"], "seed": cfg["run"]["seed"],
           "complete": R.complete(P), "pick": None, "P": {int(s): P[s] for s in P}}
    if not row["complete"]:
        row["missing"] = [s for s in R.CANDIDATES if s not in P]
        return row
    row["pick"] = R.r1_pick(P)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--sweep", default=None, help="every run dir under <runs root>/<sweep>")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    dirs = list(a.run_dirs)
    if a.sweep:
        dirs += sorted(d for d in (runs_root() / a.sweep).iterdir() if (d / "config.json").exists())
    if not dirs:
        ap.error("no run dirs")
    rows = [pick_run(d) for d in dirs]
    for r in rows:
        if r["pick"] is None:
            print(f"{r['run']:40s} INCOMPLETE: missing in-run evals at steps {r['missing']}")
            continue
        P = r["P"]
        base = f"P(0) {P[0]:.4f} -> " if 0 in P else ""
        print(f"{r['run']:40s} {r['backbone']:9s} seed {r['seed']}: R1 pick {r['pick']:4d}  "
              f"{base}P(pick) {P[r['pick']]:.4f}")
        print("    P: " + " ".join(f"{s}:{P[s]:.4f}" for s in R.CANDIDATES))
        print(f"    python scripts/sweep/make_final.py --backbone {r['backbone']} --step {r['pick']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
