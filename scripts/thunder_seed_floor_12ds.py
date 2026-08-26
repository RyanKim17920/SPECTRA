#!/usr/bin/env python3
"""Measure THUNDER seed floors on the FULL 12-dataset PAPER_CLS task mean.

WHY THIS FILE EXISTS
--------------------
docs/thunder_seed_floor.md measured the seed floor as a task mean over only
5 datasets (bach, mhist, break_his, bracs, ccrcc) from ONE seed pair
(fast5_ctrl vs fast5_ctrlseed).  But every number we actually REPORT is a task
mean over the full 12 PAPER_CLS datasets (scripts/collect_final5.py::PAPER_CLS).
A 12-dataset mean averages more per-dataset noise than a 5-dataset mean, so the
5-dataset floor is the wrong bar for the quantity we report -- and, being too
large, it forces cells to read UNRESOLVABLE that may in fact be resolvable.

It also had NO floor for simple_shot at all.

This script re-measures the floor directly, on the 12-dataset mean, from seed
replicates that already exist on disk (the final5 3-backbone x 5-seed study),
and re-measures the SAME seeds restricted to the SAME 5 datasets so the ratio
(12ds floor / 5ds floor) is apples-to-apples.

STATISTICAL DEFINITION -- identical to scripts/thunder_seed_floor.py
-------------------------------------------------------------------
For an ordered pair of seed replicates (a, b) and a dataset list D of size n:

    d_ds        = f1(b, ds) - f1(a, ds)          for ds in D
    centred_2se = 2 * SD(d) / sqrt(n)            (LEGACY -- prices only the
                                                  part of seed noise that
                                                  averages away)
    offset_2se  = |mean(d)| + 2 * SD(d) / sqrt(n)   (RECOMMENDED bar)
    rms_2se     = 2 * RMS(d) / sqrt(n)

Note n is the number of DATASETS, not the number of seeds -- this is exactly
what the original script did, and mean(d) is by construction the difference of
the two runs' task means.  offset_2se therefore contains the rigid seed-to-seed
offset, which does NOT average away across datasets.

With 5 seeds we get 10 unordered pairs instead of 1.  We report the MEAN and
the MAX offset_2se over those 10 pairs.  The MEAN is the direct analogue of the
single published number; the MAX is the conservative reading.

We additionally report the direct across-seed dispersion of the task mean:

    seed_sd     = SD over the 5 seeds of the 12-dataset (or 5-dataset) task mean
    pairdiff_2sd = 2 * sqrt(2) * seed_sd

which is the 2-sigma bar on the difference between two independent single runs
(what "arm vs ctrl, one run each" actually is).  It uses all 5 seeds jointly
and so is the most stable estimator here.

Usage:
    python scripts/thunder_seed_floor_12ds.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
from pathlib import Path

# --- dataset lists ---------------------------------------------------------
# The full list we actually report on (scripts/collect_final5.py::PAPER_CLS).
DS12 = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist",
    "patch_camelyon", "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]
# The 5 datasets the published floor used (docs/thunder_seed_floor.md).
DS5 = ["bach", "mhist", "break_his", "bracs", "ccrcc"]

TASKS = ["knn", "linear_probing", "simple_shot"]
BACKBONES = ["phikon", "midnight", "virchow2"]
SEEDS = [0, 1, 2, 3, 4]

# Pooling protocol used by the final5 FT runs, per backbone, for CLASSIFICATION.
# Source: scripts/collect_final5.py -- THUNDER_BASE_DIRS maps the BASE dirs
# phikon->base_cls, midnight->mbase_clsmean, virchow2->vbase_clsmean.  The FT
# run dirs do not encode pooling in their name; the pooling is a property of
# the eval protocol used for that backbone.
POOLING = {"phikon": "cls", "midnight": "clsmean", "virchow2": "clsmean"}

DEFAULT_ROOT = os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder")

# Waiv's published THUNDER gains, F1 fractions (scripts/scoreboard.py WAIV_THUNDER).
WAIV_GAIN = {
    ("phikon", "knn"): 0.037,
    ("phikon", "linear_probing"): 0.014,
    ("phikon", "simple_shot"): 0.015,
    ("midnight", "knn"): 0.017,
    ("midnight", "linear_probing"): 0.002,
    ("midnight", "simple_shot"): 0.037,
    ("virchow2", "knn"): -0.003,
    ("virchow2", "linear_probing"): 0.003,
    ("virchow2", "simple_shot"): 0.027,
}


# --- score extraction (mirrors scripts/collect_thunder.py::_score) ---------

def _flat(d: dict) -> float | None:
    f1 = d.get("f1")
    return f1.get("metric_score") if isinstance(f1, dict) else f1


def score(blob: dict, task: str) -> float | None:
    if task == "linear_probing":
        return _flat(blob)
    keys = [k for k in blob if k.isdigit()]
    if not keys:
        return _flat(blob)
    return _flat(blob[keys[0] if len(keys) == 1 else max(keys, key=int)])


def read(root: Path, dataset: str, run: str, task: str, adaptation: str = "frozen"):
    p = root / dataset / run / task / adaptation / "outputs.json"
    if not p.is_file():
        return None
    with p.open() as fh:
        return score(json.load(fh), task)


def find_run(root: Path, backbone: str, seed: int) -> str | None:
    """Locate the f5_final5-<bb>-s<seed>-t900-<jobid>_s0000500 run dir name."""
    for cand in sorted((root / "bach").glob(f"f5_final5-{backbone}-s{seed}-t900-*_s0000500")):
        return cand.name
    return None


# --- statistics ------------------------------------------------------------

def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def pair_stats(a_scores, b_scores):
    d = [b - a for a, b in zip(a_scores, b_scores)]
    n = len(d)
    m, s, r = statistics.fmean(d), sd(d), rms(d)
    se = s / math.sqrt(n)
    return {
        "n": n, "mean_delta": m, "sd_delta": s, "rms_delta": r,
        "centred_2se": 2 * se,
        "offset_2se": abs(m) + 2 * se,
        "rms_2se": 2 * r / math.sqrt(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--adaptation", default="frozen")
    ap.add_argument("--out-json", default="docs/thunder_seed_floor_12ds.json")
    ap.add_argument("--out-md", default="docs/thunder_seed_floor_12ds.md")
    args = ap.parse_args()

    root = Path(args.root)
    if (root / "outputs" / "res").is_dir():
        root = root / "outputs" / "res"

    runs = {}
    for bb in BACKBONES:
        for s in SEEDS:
            r = find_run(root, bb, s)
            if r:
                runs[(bb, s)] = r

    cells = {}
    per_seed_dump = {}
    for bb in BACKBONES:
        for task in TASKS:
            # per-seed per-dataset scores
            tbl = {}
            for s in SEEDS:
                run = runs.get((bb, s))
                if not run:
                    continue
                row = {}
                for ds in DS12:
                    v = read(root, ds, run, task, args.adaptation)
                    if v is not None:
                        row[ds] = v
                tbl[s] = row
            usable12 = [s for s in SEEDS if s in tbl and all(ds in tbl[s] for ds in DS12)]
            usable5 = [s for s in SEEDS if s in tbl and all(ds in tbl[s] for ds in DS5)]
            per_seed_dump[f"{bb}/{task}"] = {str(s): tbl.get(s, {}) for s in SEEDS}

            cell = {"backbone": bb, "task": task, "pooling": POOLING[bb],
                    "runs": {str(s): runs.get((bb, s)) for s in SEEDS},
                    "n_seeds_12ds": len(usable12), "n_seeds_5ds": len(usable5)}

            for tag, dslist, usable in (("12ds", DS12, usable12), ("5ds", DS5, usable5)):
                if len(usable) < 2:
                    cell[tag] = {"error": "fewer than 2 usable seeds"}
                    continue
                means = {s: statistics.fmean([tbl[s][ds] for ds in dslist]) for s in usable}
                pairs = {}
                for a, b in itertools.combinations(usable, 2):
                    pairs[f"s{a}-s{b}"] = pair_stats(
                        [tbl[a][ds] for ds in dslist], [tbl[b][ds] for ds in dslist])
                offs = [p["offset_2se"] for p in pairs.values()]
                cents = [p["centred_2se"] for p in pairs.values()]
                rmss = [p["rms_2se"] for p in pairs.values()]
                seed_sd = sd(list(means.values()))
                cell[tag] = {
                    "n_datasets": len(dslist),
                    "n_seeds": len(usable),
                    "seeds": usable,
                    "task_mean_per_seed": {str(s): means[s] for s in usable},
                    "task_mean_grand": statistics.fmean(list(means.values())),
                    "seed_sd_of_task_mean": seed_sd,
                    "pairdiff_2sd": 2 * math.sqrt(2) * seed_sd,
                    "n_pairs": len(pairs),
                    "offset_2se_mean": statistics.fmean(offs),
                    "offset_2se_max": max(offs),
                    "offset_2se_min": min(offs),
                    "centred_2se_mean": statistics.fmean(cents),
                    "rms_2se_mean": statistics.fmean(rmss),
                    "pairs": pairs,
                }
            cells[f"{bb}/{task}"] = cell

    doc = {
        "generator": "scripts/thunder_seed_floor_12ds.py",
        "root": str(root),
        "adaptation": args.adaptation,
        "metric": "f1",
        "datasets_12": DS12,
        "datasets_5": DS5,
        "seeds": SEEDS,
        "definition": {
            "centred_2se": "2*SD(d)/sqrt(n_datasets), d = per-dataset F1 delta between two seed replicates. LEGACY.",
            "offset_2se": "|mean(d)| + 2*SD(d)/sqrt(n_datasets). RECOMMENDED -- contains the rigid seed offset, which does not average away.",
            "rms_2se": "2*RMS(d)/sqrt(n_datasets), RMS non-centred.",
            "offset_2se_mean/max": "mean / max of offset_2se over all unordered seed pairs.",
            "seed_sd_of_task_mean": "SD across seeds of the task mean itself.",
            "pairdiff_2sd": "2*sqrt(2)*seed_sd -- 2-sigma bar on the difference of two independent single runs.",
        },
        "waiv_gain_f1": {f"{k[0]}/{k[1]}": v for k, v in WAIV_GAIN.items()},
        "cells": cells,
        "per_seed_scores": per_seed_dump,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(doc, indent=2) + "\n")

    # --- console table ---
    hdr = f"{'backbone/task':<28}{'pool':<9}{'n':<4}{'5ds off2SE':>12}{'12ds off2SE':>13}{'ratio':>8}{'12ds sdSD':>11}{'pd2SD':>9}{'|Waiv|':>9}{'r12/W':>8}"
    print(hdr)
    print("-" * len(hdr))
    for key, c in cells.items():
        a, b = c.get("5ds", {}), c.get("12ds", {})
        if "error" in a or "error" in b:
            print(f"{key:<28}{c['pooling']:<9}{c['n_seeds_12ds']:<4} insufficient")
            continue
        w = abs(WAIV_GAIN.get((c["backbone"], c["task"]), float("nan")))
        ratio = b["offset_2se_mean"] / a["offset_2se_mean"]
        print(f"{key:<28}{c['pooling']:<9}{b['n_seeds']:<4}"
              f"{a['offset_2se_mean']:>12.4f}{b['offset_2se_mean']:>13.4f}{ratio:>8.3f}"
              f"{b['seed_sd_of_task_mean']:>11.4f}{b['pairdiff_2sd']:>9.4f}"
              f"{w:>9.3f}{b['offset_2se_mean']/w:>8.2f}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
