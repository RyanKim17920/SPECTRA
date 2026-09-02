#!/usr/bin/env python3
"""THUNDER seed floor for the two GATED backbones (H-Optimus-0, UNI2-h).

WHY THIS FILE EXISTS
--------------------
`scripts/thunder_seed_floor_12ds.py` measures the floor from the **final5**
study: 3 backbones (phikon-v2, midnight, Virchow2) x 5 training seeds at
step 500, evaluated in the OLD THUNDER corpus (`/data/ryan.kim/thunder`,
Resize(224, bilinear)).  H-Optimus-0 and UNI2-h have no final5 runs, so they
had no floor, and `docs/final_scoreboard.md` marked every graded cell for
those two backbones NOT REPORTABLE.

They DO have a seed pair in the new `pathfm-full-evals` corpus:

    <bb>-c3s-s0-step125_optimized   vs   <bb>-c3s-s1-step125_optimized

Same recipe, same checkpoint step, different training seed -- the exact
construction the final5 floor uses, only with n_seeds = 2 instead of 5.

This script applies the IDENTICAL statistical definition (it imports the
estimators from thunder_seed_floor_12ds so there is one implementation) to
those pairs, and additionally recomputes the same pairs for phikon-v2,
midnight and Virchow2 in BOTH corpora, so the transform change
(Resize(224,bilinear) -> Resize(256,bicubic)+CenterCrop) can be shown not to
move the floor.

Usage:
    python3 scripts/thunder_seed_floor_gated.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from thunder_seed_floor_12ds import (  # noqa: E402
    DS5,
    DS12,
    TASKS,
    pair_stats,
    read,
    sd,
)

NEW_ROOT = "/data/ryan.kim/pathfm-full-evals/thunder/outputs/res"
OLD_ROOT = "/data/ryan.kim/thunder/outputs/res"

#: label -> (corpus, run_seed0, run_seed1, classification pooling)
#: Pooling per scripts/collect_final5.py::THUNDER_BASE_DIRS.  hoptimus and
#: uni2 are cls-pooled for classification, like phikon-v2.
PAIRS = {
    "hoptimus0/c3s-step125": ("new", "hoptimus0-c3s-s0-step125_optimized",
                              "hoptimus0-c3s-s1-step125_optimized", "cls"),
    "uni2h/c3s-step125": ("new", "uni2h-c3s-s0-step125_optimized",
                          "uni2h-c3s-s1-step125_optimized", "cls"),
    # transform-control pairs: same checkpoints, both corpora
    "phikon/c3s-step250": ("both", "phikon2-c3s-s0-step250_optimized",
                           "phikon2-c3s-s1-step250_optimized", "cls"),
    "midnight/c3s-step125": ("both", "midnight-c3s-s0-step125_optimized",
                             "midnight-c3s-s1-step125_optimized", "clsmean"),
    "virchow2/c3s-step125": ("both", "virchow2-c3s-s0-step125_optimized",
                             "virchow2-c3s-s1-step125_optimized", "clsmean"),
}

#: OLD-corpus run-dir names for the same checkpoints (different naming scheme).
OLD_ALIASES = {
    "phikon2-c3s-s0-step250_optimized": "f5_ci-phikon-s0-392669_s0000250",
    "phikon2-c3s-s1-step250_optimized": "f5_ci-phikon-s1-392672_s0000250",
    "midnight-c3s-s0-step125_optimized": "f5_ci-midnight-s0-392670_s0000125",
    "midnight-c3s-s1-step125_optimized": "f5_ci-midnight-s1-392673_s0000125",
    "virchow2-c3s-s0-step125_optimized": "f5_ci-virchow2-s0-392671_s0000125",
    "virchow2-c3s-s1-step125_optimized": "f5_ci-virchow2-s1-392674_s0000125",
}

#: base-control run per gated backbone, for the FT-delta-vs-floor comparison.
BASE_CONTROL = {
    "hoptimus0/c3s-step125": "hoptimus0-base-control_optimized",
    "uni2h/c3s-step125": "uni2h-base-control_optimized",
}

#: Waiv's published THUNDER gains for the gated backbones, F1 fractions.
#: docs/waiv_published.json, Table 2 (their base row vs their fine-tuned row).
WAIV_GAIN = {
    ("hoptimus0/c3s-step125", "knn"): 0.004,
    ("hoptimus0/c3s-step125", "linear_probing"): 0.003,
    ("hoptimus0/c3s-step125", "simple_shot"): 0.012,
    ("uni2h/c3s-step125", "knn"): 0.001,
    ("uni2h/c3s-step125", "linear_probing"): -0.008,
    ("uni2h/c3s-step125", "simple_shot"): -0.003,
}


def cell(root: Path, ra: str, rb: str, task: str, adaptation: str) -> dict:
    A, B = {}, {}
    for ds in DS12:
        va = read(root, ds, ra, task, adaptation)
        vb = read(root, ds, rb, task, adaptation)
        if va is not None and vb is not None:
            A[ds], B[ds] = va, vb
    out = {"n_datasets_found": len(A),
           "missing": sorted(set(DS12) - set(A))}
    for tag, dslist in (("12ds", DS12), ("5ds", DS5)):
        if not set(dslist) <= set(A):
            out[tag] = {"error": f"missing {sorted(set(dslist) - set(A))}"}
            continue
        st = pair_stats([A[d] for d in dslist], [B[d] for d in dslist])
        st["task_mean_s0"] = statistics.fmean([A[d] for d in dslist])
        st["task_mean_s1"] = statistics.fmean([B[d] for d in dslist])
        st["seed_sd_of_task_mean"] = sd([st["task_mean_s0"], st["task_mean_s1"]])
        st["pairdiff_2sd"] = 2 * math.sqrt(2) * st["seed_sd_of_task_mean"]
        out[tag] = st
    out["per_dataset"] = {d: {"s0": A[d], "s1": B[d]} for d in sorted(A)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--old-root", default=OLD_ROOT)
    ap.add_argument("--adaptation", default="frozen")
    ap.add_argument("--out-json", default="docs/thunder_seed_floor_gated.json")
    args = ap.parse_args()

    roots = {"new": Path(args.new_root), "old": Path(args.old_root)}
    cells: dict[str, dict] = {}

    for label, (corpora, ra, rb, pool) in PAIRS.items():
        which = ("new", "old") if corpora == "both" else (corpora,)
        for corpus in which:
            root = roots[corpus]
            a = OLD_ALIASES.get(ra, ra) if corpus == "old" else ra
            b = OLD_ALIASES.get(rb, rb) if corpus == "old" else rb
            for task in TASKS:
                key = f"{label}|{task}|{corpus}"
                c = cell(root, a, b, task, args.adaptation)
                c.update(pooling=pool, corpus=corpus, runs=[a, b],
                         n_seeds=2, backbone=label.split("/")[0])
                cells[key] = c

    # FT delta vs our own base-control, for the gated backbones only.
    deltas = {}
    for label, base_run in BASE_CONTROL.items():
        _, ra, rb, _ = PAIRS[label]
        for task in TASKS:
            vals = {}
            for name, run in (("base", base_run), ("s0", ra), ("s1", rb)):
                xs = [read(roots["new"], d, run, task, args.adaptation) for d in DS12]
                vals[name] = statistics.fmean(xs) if all(v is not None for v in xs) else None
            if None in vals.values():
                deltas[f"{label}|{task}"] = {"error": "missing datasets"}
                continue
            d0 = vals["s0"] - vals["base"]
            d1 = vals["s1"] - vals["base"]
            deltas[f"{label}|{task}"] = {
                **vals, "delta_s0": d0, "delta_s1": d1,
                "delta_seed_mean": (d0 + d1) / 2,
            }

    doc = {
        "generated_by": "scripts/thunder_seed_floor_gated.py",
        "adaptation": args.adaptation,
        "roots": {k: str(v) for k, v in roots.items()},
        "datasets_12": DS12,
        "datasets_5": DS5,
        "n_seeds": 2,
        "caveat": ("n_seeds = 2 (a single seed pair).  The 3-backbone floor in "
                   "thunder_seed_floor_12ds.md uses n_seeds = 5 / 10 pairs.  "
                   "offset_2se still has df = n_datasets - 1 = 11 and is usable; "
                   "pairdiff_2sd has df = 1 across seeds and is indicative only."),
        "waiv_gain_f1": {f"{k[0]}|{k[1]}": v for k, v in WAIV_GAIN.items()},
        "cells": cells,
        "gated_ft_deltas_vs_base_control": deltas,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(doc, indent=2) + "\n")

    hdr = (f"{'cell':<44}{'pool':<9}{'12ds off2SE':>13}{'5ds off2SE':>12}"
           f"{'pd2SD':>9}{'|Waiv|':>9}{'ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key, c in cells.items():
        b = c.get("12ds", {})
        if "error" in b:
            print(f"{key:<44}{c['pooling']:<9} {b['error']}")
            continue
        label, task, _ = key.split("|")
        w = WAIV_GAIN.get((label, task))
        wtxt = f"{abs(w):.3f}" if w is not None else "-"
        rtxt = f"{b['offset_2se'] / abs(w):.2f}" if w else "-"
        print(f"{key:<44}{c['pooling']:<9}{b['offset_2se']:>13.4f}"
              f"{c['5ds']['offset_2se']:>12.4f}{b['pairdiff_2sd']:>9.4f}"
              f"{wtxt:>9}{rtxt:>8}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
