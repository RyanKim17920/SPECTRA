#!/usr/bin/env python
"""The PLAN.md §4 Phase-2 gate: does *our* pipeline reproduce PathoROB's own number?

Runs PathoROB's ``robustness_index`` module over features our extractor wrote, then puts
three independent numbers side by side for each dataset:

1. **ours**      -- ``results/robustness_index/{model}/{dataset}/-1_0/results_summary.json``
                    produced right now from our npz files.
2. **reference** -- the ``phikonv2_clsmean`` row PathoROB *committed to their repo*.
3. **Waiv**      -- the base-phikon-v2 row quoted in Waiv's Table 1 (PLAN.md §1).

(2) and (3) are independent sources and are worth cross-checking against each other
regardless of what we compute.

The gate is (1) vs (2): if our RI does not land on their committed RI, our harness is
wrong and every downstream fine-tuning result is meaningless.

    ./.venv-pathorob/bin/python scripts/pathorob_gate.py --model phikonv2_clsmean_ours

The metric itself runs in ``.venv-pathorob`` (their exact pins). This script only needs
numpy + the adapter, so either venv can drive it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from waivphaet.eval.pathorob_adapter import (  # noqa: E402
    TARGETS,
    PathoRobPaths,
    read_results,
    run_robustness_index,
)

REFERENCE_MODEL = "phikonv2_clsmean"
#: Absolute RI delta we are willing to call "reproduced". The metric is a deterministic
#: kNN over fixed embeddings, so the only sources of drift are float ordering and JPEG
#: decode; anything above this is a real pipeline difference, not noise.
TOLERANCE = 0.005


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="phikonv2_clsmean_ours")
    ap.add_argument("--datasets", nargs="+", default=["camelyon"])
    ap.add_argument("--root", default=str(REPO / "third_party" / "PathoROB"))
    ap.add_argument("--skip-run", action="store_true", help="read existing results only")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    paths = PathoRobPaths(root=Path(args.root))

    if not args.skip_run:
        run_robustness_index(args.model, args.datasets, paths=paths)

    rows, verdicts = [], []
    for ds in args.datasets:
        res = read_results(args.model, ds, paths=paths)
        ours = float(res["robustness_index"])
        ours_bal = res.get("balanced_accuracy", None)
        ref = float(read_results(REFERENCE_MODEL, ds, paths=paths)["robustness_index"])
        waiv = TARGETS["phikon_v2_base"][ds]
        d_ref, d_waiv = ours - ref, ours - waiv
        ok = abs(d_ref) <= args.tolerance
        verdicts.append(ok)
        rows.append(
            {"dataset": ds, "ours": ours, "pathorob_reference": ref, "waiv_table1": waiv,
             "delta_vs_reference": d_ref, "delta_vs_waiv": d_waiv,
             "reference_vs_waiv": ref - waiv, "bal_acc": ours_bal, "pass": ok}
        )

    w = max(len(r["dataset"]) for r in rows) + 2
    print()
    print(f"{'dataset':<{w}}{'ours':>10}{'PathoROB ref':>14}{'Waiv T1':>10}"
          f"{'d(ref)':>10}{'d(Waiv)':>10}{'bal_acc':>10}   gate")
    print("-" * (w + 68))
    for r in rows:
        bal = f"{r['bal_acc']:.4f}" if r['bal_acc'] is not None else "-"
        print(f"{r['dataset']:<{w}}{r['ours']:>10.6f}{r['pathorob_reference']:>14.6f}"
              f"{r['waiv_table1']:>10.3f}{r['delta_vs_reference']:>+10.6f}"
              f"{r['delta_vs_waiv']:>+10.6f}{bal:>10}   {'PASS' if r['pass'] else 'FAIL'}")
    if len(rows) == 3:
        avg = sum(r["ours"] for r in rows) / 3
        avg_ref = sum(r["pathorob_reference"] for r in rows) / 3
        print(f"{'AVG':<{w}}{avg:>10.6f}{avg_ref:>14.6f}{0.469:>10.3f}"
              f"{avg - avg_ref:>+10.6f}{avg - 0.469:>+10.6f}")
    ok = all(verdicts)
    print(f"\nGATE (|ours - PathoROB reference| <= {args.tolerance}): "
          f"{'PASS' if ok else 'FAIL'}\n")
    print(json.dumps(rows, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
