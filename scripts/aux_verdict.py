#!/usr/bin/env python3
"""Post-hoc AUTHORITATIVE verdict on whether the factored-invariance aux heads trained.

WHY THIS FILE EXISTS, AND WHY IT OVERRIDES THE IN-JOB BLOCK
-----------------------------------------------------------
The aux stain/scanner heads are trained at the production geometry --grid-conditions 2.
Only 2 of the 50 (10 stain x 5 scanner) condition classes appear in any single step, so a
downstream null on RI/HEST is ambiguous between "factored invariance does not help" and
"the head never learned anything to factor". Distinguishing those is the ONLY job here.

The weak statistic that must not be used: comparing a first-N-step mean to a last-N-step
mean. Aux CE at C=2 is extremely noisy step to step (each step draws a different pair of
conditions, so the CE is over a different 2-class subproblem each time). A first-N/last-N
difference is a difference of two noisy point estimates and will show an apparent fall
essentially at random. The builder's first pass reported exactly that artefact: a CE that
"fell" under first-3/last-3 was flat under honest statistics, and the only real evidence
of learnability came from a SEPARATE C=8 150-step run, not from C=2 at all.

What is used instead, over the FULL logged history:

  1. OLS slope of CE on step, with the residual standard error of the slope, reported as
     a t ratio. A trend claim needs |t| clearly above the noise, not a sign.
  2. Half-over-half means of CE and accuracy -- every logged row participates, so no
     endpoint noise dominates.
  3. Final accuracy against the TRUE baseline. This is the load-bearing test, and the
     baseline is NOT 1/n_classes. There are two distinct reference points and both matter:

       * UNIFORM chance over the training vocabulary: 1/10 = 0.100 (CE ln 10 = 2.3026)
         for stain, 1/5 = 0.200 (CE ln 5 = 1.6094) for scanner. That vocabulary is the
         manifest's 13 stains / 7 scanners MINUS held-out stains HRH KR MY and scanners
         GT450 S210. Do NOT use 13x7 here. A head at uniform chance has learned nothing.

       * TRIVIAL in-batch baseline = 1/n_cond. THIS is the bar that matters at C=2, and
         the reason a naive "acc 0.27 > 0.10, therefore it learned" reading is wrong. With
         --grid-conditions C, at most C distinct condition labels are present in any step,
         so a head that ignores its input entirely and emits the single most frequent
         in-batch label already scores at least 1/C -- 0.500 at C=2, against a uniform
         "chance" of 0.100. An accuracy of 0.27 at C=2 is therefore not weak evidence of
         decoding; it is BELOW the constant predictor, i.e. the head is still near uniform
         and has learned nothing. At C=8 the trivial bar is 0.125 and an accuracy of 0.90
         clears it by a mile, which is why the C=8 run is real evidence and the C=2 runs,
         so far, are not.

     A head counts as decoding only if it beats BOTH baselines.

A head is TRAINED only if BOTH hold: its final CE sits significantly below uniform chance
CE (mean of the last half, minus two standard errors of that mean), AND its final accuracy
clears the trivial 1/n_cond in-batch baseline. A significant negative CE slope alone is
corroboration, not proof -- a head can shave CE by learning the class PRIOR without
decoding anything. Conversely a head that converged before the first logged row shows a
flat slope but high accuracy, and the accuracy test catches it where a slope test alone
would not. Anything else prints DID NOT TRAIN, which makes any downstream null for that
arm uninterpretable.

Usage:
    python scripts/aux_verdict.py runs/aux0.1-0.1MASK-...  [more run dirs ...]
    python scripts/aux_verdict.py --glob 'runs/aux*'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# TRAINING-split vocabulary. Verified: manifest 13 stains / 7 scanners, minus held-out
# stains HRH KR MY and scanners GT450 S210 -> 10 / 5.
CHANCE = {"stain": 10, "scanner": 5}

# Per-step the aux head sees the ANCHOR half of the batch: C*T = 2*900 = 1800 images.
# Binomial sd of an at-chance accuracy over 1800 draws is ~0.0071 (stain) / ~0.0094
# (scanner). The margin below is deliberately wider than that: consecutive logged steps
# share a model, so they are NOT independent samples and a nominal binomial z would
# overstate significance.
ACC_ABS_MARGIN = 0.02
T_CRIT = 3.0  # |slope| must exceed 3 standard errors to count as a real trend
CE_SE_MULT = 2.0  # final CE must sit this many SEMs below uniform chance CE


def best_constant_baseline(n_cond: int, axis: str, n_trials: int = 20000) -> float:
    """Expected accuracy of the best INPUT-IGNORING constant predictor, by simulation.

    1/n_cond is only a LOWER bound on the trivial bar. The sampler draws n_cond DISTINCT
    conditions from the 10 stain x 5 scanner training grid (history confirms
    batch_distinct_conditions == n_cond), and distinct CONDITIONS can still share a
    stain or a scanner -- e.g. at C=2 there is a 4/49 chance both conditions carry the
    same stain, in which case a constant predictor scores 1.000, not 0.500. Averaging the
    per-draw majority share over the sampler's distribution gives the honest bar:

        C=2  stain   ~0.54   (vs the 0.500 lower bound)
        C=2  scanner ~0.59   (vs the 0.500 lower bound)

    Using this instead of 1/n_cond makes the test STRICTER, which is the correct direction:
    a head must beat the best predictor that ignores its input entirely.
    """
    import random
    if n_cond <= 0:
        return float("nan")
    n_stain, n_scanner = CHANCE["stain"], CHANCE["scanner"]
    grid = [(st, sc) for st in range(n_stain) for sc in range(n_scanner)]
    idx = 0 if axis == "stain" else 1
    rng = random.Random(0)  # fixed seed: the baseline must be reproducible
    if n_cond > len(grid):
        return float("nan")
    total = 0.0
    for _ in range(n_trials):
        draw = rng.sample(grid, n_cond)
        counts: dict[int, int] = {}
        for cond in draw:
            counts[cond[idx]] = counts.get(cond[idx], 0) + 1
        # Tiles are split evenly across conditions, so label shares are count/n_cond.
        total += max(counts.values()) / n_cond
    return total / n_trials


def ols_slope(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, standard error of slope, t ratio) for y ~ a + b*x."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm = sum(x) / n
    ym = sum(y) / n
    sxx = sum((xi - xm) ** 2 for xi in x)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / sxx
    inter = ym - slope * xm
    resid = [yi - (inter + slope * xi) for xi, yi in zip(x, y)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    t = slope / se if se > 0 else float("inf") * (1 if slope > 0 else -1)
    return slope, se, t


def verdict_for_axis(steps, ce, acc, axis, n_cond) -> tuple[str, list[str]]:
    n_cls = CHANCE[axis]
    chance_acc = 1.0 / n_cls
    chance_ce = math.log(n_cls)
    lines = []

    finite = all(math.isfinite(v) for v in ce)
    slope, se, t = ols_slope(steps, ce)
    h = len(ce) // 2
    ce_h1, ce_h2 = sum(ce[:h]) / h, sum(ce[h:]) / len(ce[h:])
    ac_h1, ac_h2 = sum(acc[:h]) / h, sum(acc[h:]) / len(acc[h:])
    # Final accuracy = mean of the last half, not the last point.
    acc_final = ac_h2
    ce_final = ce_h2

    lines.append(
        f"    rows={len(ce)}  finite={finite}  chance: acc={chance_acc:.3f} CE={chance_ce:.4f}"
    )
    lines.append(
        f"    OLS  CE slope/step = {slope:+.6f}  SE = {se:.6f}  t = {t:+.2f}"
        f"   ({'significant fall' if t <= -T_CRIT else 'NOT distinguishable from flat'})"
    )
    lines.append(
        f"    half/half  CE {ce_h1:.4f} -> {ce_h2:.4f} ({ce_h2 - ce_h1:+.4f})"
        f"   ACC {ac_h1:.3f} -> {ac_h2:.3f} ({ac_h2 - ac_h1:+.3f})"
    )
    # SEM of the last-half CE mean, used to test CE against uniform chance CE.
    n2 = len(ce[h:])
    var2 = sum((v - ce_h2) ** 2 for v in ce[h:]) / max(n2 - 1, 1)
    sem2 = math.sqrt(var2 / n2) if n2 > 1 else float("inf")
    ce_bound = ce_final + CE_SE_MULT * sem2

    # THE bar that matters at C=2: a head that emits the most frequent in-batch label,
    # ignoring its input entirely. 1/n_cond is the lower bound; the simulated expected
    # majority share over the sampler's own distribution is the honest, stricter bar.
    lower_bound = 1.0 / n_cond if n_cond and n_cond > 0 else chance_acc
    trivial_acc = best_constant_baseline(int(n_cond), axis) if n_cond else chance_acc
    if not math.isfinite(trivial_acc):
        trivial_acc = lower_bound

    lines.append(
        f"    vs UNIFORM chance  ACC {acc_final:.3f} vs {chance_acc:.3f}"
        f" ({acc_final - chance_acc:+.3f})"
        f"   CE {ce_final:.4f} +2sem {ce_bound:.4f} vs {chance_ce:.4f}"
        f" ({'below' if ce_bound < chance_ce else 'NOT below'})"
    )
    lines.append(
        f"    vs BEST CONSTANT predictor (n_cond={n_cond:g}, sim {trivial_acc:.3f}, "
        f"1/n_cond bound {lower_bound:.3f})  ACC {acc_final:.3f} vs {trivial_acc:.3f} "
        f"(margin {acc_final - trivial_acc:+.3f}, need > +{ACC_ABS_MARGIN:.3f})"
        f"   <- decisive test"
    )

    beats_uniform = acc_final > chance_acc + ACC_ABS_MARGIN
    beats_trivial = acc_final > trivial_acc + ACC_ABS_MARGIN
    ce_below = ce_bound < chance_ce
    trending = math.isfinite(t) and t <= -T_CRIT

    if not finite:
        v = "BROKEN (non-finite CE) -- run is unusable"
    elif beats_trivial and beats_uniform and ce_below:
        v = "TRAINED" + ("" if trending else " (flat slope -- converged before first logged row)")
    elif beats_uniform and not beats_trivial:
        v = (f"DID NOT TRAIN -- ACC {acc_final:.3f} is above uniform chance but BELOW the "
             f"best-constant predictor {trivial_acc:.3f} (input-ignoring), i.e. still near "
             f"uniform. At C={n_cond:g} this is the expected reading for a head that has "
             f"learned nothing")
    elif trending and not beats_trivial:
        v = ("DID NOT TRAIN -- CE falls significantly but accuracy does not clear the "
             "constant predictor: the head is fitting the class prior, not decoding")
    else:
        v = "DID NOT TRAIN -- at chance"
    return v, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--glob", default=None)
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += sorted(Path().glob(args.glob))
    if not dirs:
        print("no run dirs given", file=sys.stderr)
        return 2

    overall = {}
    for d in dirs:
        hp = d / "history.json"
        print(f"\n=== {d.name} ===")
        if not hp.exists():
            print("  NO history.json -- run has not written one (still training, or died "
                  "before the first write). VERDICT: UNKNOWN")
            overall[d.name] = "UNKNOWN"
            continue
        try:
            h = json.loads(hp.read_text())
        except json.JSONDecodeError:
            print("  history.json unreadable. VERDICT: UNKNOWN")
            overall[d.name] = "UNKNOWN"
            continue

        axis_verdicts = []
        for axis in ("stain", "scanner"):
            rows = [r for r in h if f"loss_aux_{axis}" in r and f"acc_aux_{axis}" in r]
            if len(rows) < 4:
                print(f"  {axis}: {len(rows)} rows -- too few to test. VERDICT: UNKNOWN")
                axis_verdicts.append("UNKNOWN")
                continue
            steps = [float(r["step"]) for r in rows]
            ce = [float(r[f"loss_aux_{axis}"]) for r in rows]
            acc = [float(r[f"acc_aux_{axis}"]) for r in rows]
            w = rows[0].get(f"weight_aux_{axis}")
            det = rows[0].get("aux_detached")
            n_cond = float(rows[0].get("n_cond") or rows[0].get("batch_n_cond") or 0)
            v, lines = verdict_for_axis(steps, ce, acc, axis, n_cond)
            print(f"  {axis.upper()}  (weight={w}, detached={det})")
            for ln in lines:
                print(ln)
            print(f"    -> {v}")
            axis_verdicts.append(v.split(" --")[0].split(" (")[0])

        if not axis_verdicts:
            run_v = "AUX METRICS ABSENT"
        elif all(v == "TRAINED" for v in axis_verdicts):
            run_v = "TRAINED"
        elif any(v == "TRAINED" for v in axis_verdicts):
            run_v = "PARTIAL"
        elif any(v == "UNKNOWN" for v in axis_verdicts):
            run_v = "UNKNOWN"
        else:
            run_v = "DID NOT TRAIN"
        overall[d.name] = run_v
        interp = {
            "TRAINED": "a downstream null for this arm IS interpretable as 'factored "
                       "invariance did not help'",
            "PARTIAL": "only the TRAINED axis carries an interpretable null",
            "DID NOT TRAIN": "a downstream null for this arm is AMBIGUOUS and proves "
                             "nothing about the hypothesis",
            "UNKNOWN": "insufficient data -- do not read downstream metrics yet",
            "AUX METRICS ABSENT": "run is uninterpretable",
        }[run_v]
        print(f"  RUN VERDICT: {run_v} -- {interp}")

    print("\n=== SUMMARY ===")
    for k, v in overall.items():
        print(f"  {v:<16} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
