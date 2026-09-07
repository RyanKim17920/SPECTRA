#!/usr/bin/env python3
"""Re-derive and PERSIST the THUNDER seed floors from the ctrl-vs-ctrlseed pair.

WHY THIS FILE EXISTS
--------------------
Four 2SE seed floors (cls/knn, cls/linear_probing, clsmean/knn, clsmean/linear_probing)
were quoted throughout the THUNDER analysis but had only ever been computed in-session and
never written to disk, so every THUNDER verdict was formally ungraded.  This script
recomputes them from the raw per-dataset outputs.json files and writes the result to
docs/thunder_seed_floor.json (+ a markdown twin) so the numbers are reproducible.

THE PAIR
--------
fast5_ctrl vs fast5_ctrlseed differ ONLY in the training seed, so their per-dataset F1
difference is pure seed noise.  Two pooling protocols were run:

    cls      : fast5_ctrl          vs fast5_ctrlseed          (note: NO _cls suffix)
    clsmean  : fast5_ctrl_clsmean  vs fast5_ctrlseed_clsmean

over 5 datasets: bach, mhist, break_his, bracs, ccrcc.

THE METHODOLOGICAL DEFECT BEING FIXED
-------------------------------------
The construction in use was

    floor = 2 * SD(delta) / sqrt(n)                        # "paired-t" / centred bar

SD(delta) is dispersion *about the mean delta*, so a CONSISTENT offset between the two
seeds is invisible to it -- indeed it SHRINKS the bar, because a rigid shift adds nothing
to the spread.  That is exactly backwards: a reproducible seed-to-seed offset is the most
dangerous kind of seed noise, since it survives averaging over datasets.

The failure is not hypothetical.  For clsmean/linear_probing the mean seed delta is about
-0.0132 -- roughly 1.7x the 0.0076 bar that this construction generates.  The bar declares
"anything under 0.0076 is noise" while the two seeds themselves disagree by 0.0132.

So we report FOUR quantities per (pooling, task) cell:

  centred_2se   2 * SD(d)/sqrt(n)                 the legacy bar; dispersion about mean d
  offset_2se    |mean(d)| + 2*SD(d)/sqrt(n)       legacy bar shifted up by the offset
  rms_2se       2 * RMS(d)/sqrt(n),               SD taken about ZERO, not about mean(d);
                RMS(d) = sqrt(mean(d^2))          the natural non-centred analogue
  paired_t / p  t = mean(d)/(SD(d)/sqrt(n)), df=n-1, two-sided p

RECOMMENDATION (see docs output): use `offset_2se` as the reporting bar.  Justification is
printed with the table.

Usage:
    python scripts/thunder_seed_floor.py
    python scripts/thunder_seed_floor.py --root results_backup/thunder_res --no-outputs-suffix
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

DATASETS = ["bach", "mhist", "break_his", "bracs", "ccrcc"]
TASKS = ["knn", "linear_probing"]

# pooling -> (seed-A run name, seed-B run name)
PAIRS = {
    "cls": ("fast5_ctrl", "fast5_ctrlseed"),
    "clsmean": ("fast5_ctrl_clsmean", "fast5_ctrlseed_clsmean"),
}

DEFAULT_ROOT = os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder")


def _flat(d: dict) -> float | None:
    f1 = d.get("f1")
    return f1.get("metric_score") if isinstance(f1, dict) else f1


def score(blob: dict, task: str) -> float | None:
    """F1 out of one outputs.json.

    linear_probing is flat; knn nests under the single selected k (THUNDER persists only
    the k it picked on val, so there is exactly one digit key -- mhist came back as "3").
    Mirrors scripts/collect_thunder.py::_score so the floors are on the same metric as the
    numbers they grade.
    """
    if task == "linear_probing":
        return _flat(blob)
    keys = [k for k in blob if k.isdigit()]
    if not keys:
        return _flat(blob)
    return _flat(blob[keys[0] if len(keys) == 1 else max(keys, key=int)])


def read(root: Path, dataset: str, run: str, task: str, adaptation: str = "frozen") -> float | None:
    p = root / dataset / run / task / adaptation / "outputs.json"
    if not p.is_file():
        return None
    with p.open() as fh:
        return score(json.load(fh), task)


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """Sample SD about the mean (ddof=1)."""
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def rms(xs):
    """Root-mean-square about ZERO -- i.e. SD with the mean NOT subtracted."""
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def t_sf(t: float, df: int) -> float:
    """Two-sided p for Student-t, via the regularised incomplete beta.  No scipy here."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((a + m2 - 1) * (a + m2)),
                    -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))):
            d = 1.0 + num * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < 3e-16:
            break
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="THUNDER base folder (outputs/res is appended) or a results tree "
                         "that already IS the <dataset>/<run>/... level, e.g. "
                         "results_backup/thunder_res")
    ap.add_argument("--adaptation", default="frozen")
    ap.add_argument("--out-json", default="docs/thunder_seed_floor.json")
    ap.add_argument("--out-md", default="docs/thunder_seed_floor.md")
    args = ap.parse_args()

    root = Path(args.root)
    if (root / "outputs" / "res").is_dir():
        root = root / "outputs" / "res"

    cells = {}
    for pooling, (run_a, run_b) in PAIRS.items():
        for task in TASKS:
            per_ds, missing = {}, []
            for ds in DATASETS:
                a = read(root, ds, run_a, task, args.adaptation)
                b = read(root, ds, run_b, task, args.adaptation)
                if a is None or b is None:
                    missing.append(ds)
                    continue
                per_ds[ds] = {"seed_a": a, "seed_b": b, "delta": b - a}
            if len(per_ds) < 2:
                cells[f"{pooling}/{task}"] = {"error": "insufficient data", "missing": missing}
                continue
            d = [v["delta"] for v in per_ds.values()]
            n = len(d)
            m, s, r = mean(d), sd(d), rms(d)
            se = s / math.sqrt(n)
            t = m / se if se else float("inf")
            cells[f"{pooling}/{task}"] = {
                "pooling": pooling, "task": task, "n": n,
                "runs": {"seed_a": run_a, "seed_b": run_b},
                "missing_datasets": missing,
                "per_dataset": per_ds,
                "mean_delta": m,
                "sd_delta": s,
                "rms_delta": r,
                "se_delta": se,
                "centred_2se": 2 * se,
                "offset_2se": abs(m) + 2 * se,
                "rms_2se": 2 * r / math.sqrt(n),
                "paired_t": t,
                "paired_p": t_sf(t, n - 1),
                "offset_over_centred_bar": abs(m) / (2 * se) if se else float("inf"),
            }

    doc = {
        "generated_by": "scripts/thunder_seed_floor.py",
        "metric": "f1 (metric_score), adaptation=%s" % args.adaptation,
        "source_root": str(root),
        "datasets": DATASETS,
        "pairs": {k: list(v) for k, v in PAIRS.items()},
        "constructions": {
            "centred_2se": "2*SD(delta)/sqrt(n) -- LEGACY. SD is about mean(delta), so a "
                           "consistent seed offset is invisible to it and in fact shrinks "
                           "the bar. Do not use alone.",
            "offset_2se": "|mean(delta)| + 2*SD(delta)/sqrt(n) -- RECOMMENDED. Charges the "
                          "systematic offset AND the dispersion.",
            "rms_2se": "2*RMS(delta)/sqrt(n), RMS about zero. Non-centred analogue; grows "
                       "with the offset but attenuates it by 1/sqrt(n).",
            "paired_p": "two-sided paired t-test that mean(delta)==0. Small p == the two "
                        "seeds genuinely disagree == centred_2se is unusable.",
        },
        "recommendation": (
            "Use offset_2se as the reporting bar. An arm's mean delta vs ctrl must exceed "
            "it to be called real. Rationale: the quantity we compare against the bar is a "
            "MEAN over the same 5 tasks, and a seed-to-seed offset propagates into that "
            "mean one-for-one -- it does not average away, so the bar must contain it. "
            "centred_2se prices only the part of seed noise that does average away, which "
            "is the part we already suppress by averaging; it therefore understates the "
            "floor by exactly the term that matters. rms_2se moves in the right direction "
            "but divides the offset by sqrt(n), so it still under-charges a rigid shift. "
            "Report paired_p alongside: where p is small the centred bar is not merely "
            "conservative-in-the-wrong-direction, it is invalid, and any verdict that "
            "leaned on it must be re-graded."
        ),
        "cells": cells,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(doc, indent=2) + "\n")

    lines = [
        "# THUNDER seed floors (ctrl vs ctrlseed)",
        "",
        f"Generated by `scripts/thunder_seed_floor.py` from `{root}`.",
        f"Metric: f1, adaptation={args.adaptation}. Datasets: {', '.join(DATASETS)}.",
        "",
        "| pooling | task | n | mean d | SD d | RMS d | centred 2SE (legacy) | offset 2SE (rec.) | RMS 2SE | paired t | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in cells.items():
        if "error" in c:
            lines.append(f"| {key} | | | _{c['error']}_ | | | | | | | |")
            continue
        lines.append(
            "| {pooling} | {task} | {n} | {mean_delta:+.4f} | {sd_delta:.4f} | {rms_delta:.4f} "
            "| {centred_2se:.4f} | **{offset_2se:.4f}** | {rms_2se:.4f} | {paired_t:+.2f} "
            "| {paired_p:.3f} |".format(**c)
        )
    lines += ["", "## Per-dataset deltas (seed_b - seed_a)", ""]
    lines.append("| pooling/task | " + " | ".join(DATASETS) + " |")
    lines.append("|---" * (len(DATASETS) + 1) + "|")
    for key, c in cells.items():
        if "error" in c:
            continue
        row = [f"{c['per_dataset'][ds]['delta']:+.4f}" if ds in c["per_dataset"] else "--"
               for ds in DATASETS]
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    lines += ["", "## Recommendation", "", doc["recommendation"], ""]
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
