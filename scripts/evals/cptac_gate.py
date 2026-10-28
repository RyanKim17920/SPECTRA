"""CPTAC regression gate: compare a model's cptac.json against a reference model's cptac.json.

    python scripts/evals/cptac_gate.py --key NEW --ref REF [--tol 0.0006]

Both results live under $SPECTRA_EVAL_OUT/<key>/cptac.json. Prints the headline mean AUC and
the per-task deltas; per-task deltas above ``--tol`` are flagged and the exit status is 1.
Use it to check that a re-run of the same model reproduces its earlier result, or to list the
tasks that moved between a base backbone and a tuned checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptac_common as C  # noqa: E402


def load(key):
    p = C.SHARED_DIR.parent / key / "cptac.json"
    if not p.exists():
        raise SystemExit(f"{key}: no {p}")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=0.0006)
    a = ap.parse_args()
    new, ref = load(a.key), load(a.ref)
    mk = "mean_classification_macro_ovr_auc"
    print(f"headline mean AUC: {a.key} {new['metrics'][mk]:.4f}  {a.ref} {ref['metrics'][mk]:.4f}  "
          f"delta {new['metrics'][mk] - ref['metrics'][mk]:+.4f}")
    pn = new["per_task"]["classification_macro_ovr_auc"]
    pr = ref["per_task"]["classification_macro_ovr_auc"]
    se = new["per_task"]["classification_macro_ovr_auc_bootstrap_se"]
    deltas, flagged = [], 0
    for k in sorted(pn):
        d = pn[k] - pr[k]
        deltas.append(abs(d))
        bad = abs(d) > a.tol
        flagged += bad
        print(f"  {k:36s} ref {pr[k]:.4f} new {pn[k]:.4f} d {d:+.4f} bootSE {se[k]:.3f}{' <' if bad else ''}")
    print(f"per-task |delta|: max {max(deltas):.4f} mean {statistics.mean(deltas):.4f}; "
          f"{flagged}/{len(deltas)} above tol {a.tol}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
