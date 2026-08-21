#!/usr/bin/env python
"""Sweep k for the nuisance projection and report RI as a delta against the k=0 control.

For each ``k`` this materialises ``{src_model}_svd{k:03d}`` via
:mod:`scripts.apply_svd_nuisance`, runs PathoROB's own ``robustness_index`` over it, and
collects the result. Nothing is reimplemented -- the metric is
``waivphaet.eval.pathorob_adapter.run_robustness_index``, i.e. their module in their venv.

**Everything is reported as a delta against k=0, never as an absolute.** k=0 is a
byte-identical copy of the source features, so its RI must equal the source row exactly;
that makes it the only honest reference for the sweep. Absolutes across arms invite the
mistake this repo has already made twice -- reading a difference off an instrument whose
seed floor is larger than the difference (see the HEST and THUNDER notes). Ranking on the
delta at least keeps the comparison paired.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATASETS = ("camelyon", "tcga", "tolkach_esca")


def dst_model_name(src_model: str, k: int) -> str:
    """``phikonv2_clsmean_ours`` + k=8 -> ``phikonv2_clsmean_ours_svd008``.

    Zero-padded so the feature dirs and the results dirs sort in rank order rather than
    lexicographically (``_svd8`` after ``_svd128`` is how a sweep table gets misread).
    """
    return f"{src_model}_svd{k:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, required=True, help="npz from fit_svd_nuisance.py")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64],
                    help="ranks to sweep; 0 (the control) is added if absent")
    ap.add_argument("--src-model", default="phikonv2_clsmean_ours")
    ap.add_argument("--pathorob-root", type=Path, default=REPO / "third_party" / "PathoROB")
    ap.add_argument("--features-dir", type=Path, default=None,
                    help="default: {pathorob-root}/data/features")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--python-exe", default=None,
                    help="interpreter for PathoROB's metric (default .venv-pathorob)")
    ap.add_argument("--out", type=Path, required=True, help="summary JSON to write")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the apply+score plan and exit; touches nothing")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from apply_svd_nuisance import apply_fit

    from waivphaet.eval.pathorob_adapter import (
        PathoRobPaths,
        read_results,
        run_robustness_index,
    )

    features_dir = args.features_dir or (args.pathorob_root / "data" / "features")
    datasets = list(args.datasets)
    # The control always runs, and it runs FIRST: if k=0 does not reproduce the source row
    # there is no point spending GPU-free-but-not-free minutes on the rest of the sweep.
    ks = sorted({0, *args.ks})

    if args.dry_run:
        for k in ks:
            dst = dst_model_name(args.src_model, k)
            print(f"python scripts/apply_svd_nuisance.py --fit {args.fit} --k {k} "
                  f"--src-model {args.src_model} --dst-model {dst} "
                  f"--features-dir {features_dir} --datasets {' '.join(datasets)}")
            print(f"  -> run_robustness_index({dst!r}, {datasets})")
        print(f"  -> {args.out}")
        return 0

    paths = PathoRobPaths(root=args.pathorob_root)
    rows: dict[str, dict] = {}
    for k in ks:
        dst = dst_model_name(args.src_model, k)
        t0 = time.time()
        info = apply_fit(args.fit, k, args.src_model, dst, features_dir, tuple(datasets))
        print(f"[score] k={k} -> {dst}: {info['vectors']} vectors, D={info['D']} "
              f"({time.time() - t0:.1f}s)", flush=True)
        run_robustness_index(dst, datasets, paths=paths, python_exe=args.python_exe)
        ri = {d: float(read_results(dst, d, paths=paths)["robustness_index"]) for d in datasets}
        ri["avg"] = sum(ri.values()) / len(datasets)
        rows[str(k)] = {"model": dst, "ri": ri}
        print(f"[score] k={k} RI {json.dumps({d: round(v, 4) for d, v in ri.items()})}",
              flush=True)

    base = rows["0"]["ri"]
    for k, row in rows.items():
        row["delta_vs_k0"] = {d: row["ri"][d] - base[d] for d in base}

    summary = {
        "fit": str(args.fit),
        "src_model": args.src_model,
        "datasets": datasets,
        "ks": ks,
        "rows": rows,
        # Ranked on the paired delta, not on the absolute -- see the module docstring.
        "best_k": max(rows, key=lambda k: rows[k]["delta_vs_k0"]["avg"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "out": str(args.out),
        "best_k": summary["best_k"],
        "avg_delta": {k: round(r["delta_vs_k0"]["avg"], 5) for k, r in rows.items()},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
