#!/usr/bin/env python3
"""FINAL VERDICT for the candidate one-fit-all recipe.

Recipe (identical hyperparameters on all three backbones):
    mask_same_core, same_core_logit_bias_cls=3.0, same_core_logit_bias_mean=-inf,
    lr=1e-4, wd=0.05, T=0.07, max_steps=500, ckpt_every=125,
    LoRA r=32 / alpha=64 / proj 512.

Checkpoint selection is a RULE, not a number: the FIRST checkpoint whose mean
`confounder_insensitivity` (averaged over the datasets present in that point) reaches
0.75.  The rule is applied here, per run, from runs/<run>/ri_curve.json -- nothing about
the step is hardcoded, so re-running this after more seeds land needs no edit.

The criterion this script grades against (deliberately NOT softened):
    pct_of_waiv = (ours - base) / (waiv - base) * 100, UNCAPPED.
    F-C fix (2026-08-26): the >=70 per-cell test and the >80 average are now functions of
    the SAME quantity -- the uncapped pct.  The 100 cap ("exceeding Waiv counts as 100")
    survives as PRESENTATION ONLY: it is printed beside the graded value and enters no
    arithmetic.  Previously the test ran on the uncapped interval while the averages
    summed the capped value, so one cell had three different published numbers
    (phikon/THUNDER printed 79.7, graded PASS on 99.6, capped lower bound 56.9) and a
    reader could not reproduce the verdict from any of them.
    PASS requires >= 70 on EVERY (backbone, benchmark) cell -- scored by the WORST
    cell, never by a cross-backbone mean -- AND the average of the three benchmark
    means must exceed 80.

Honesty rules that this script enforces and that make it different from a plain
collector:

  * A THUNDER task mean over fewer than all 12 PAPER_CLS datasets is PARTIAL and is
    graded by NOTHING.  The 12-dataset seed floors are floors for a 12-dataset mean; a
    mean over fewer datasets averages away less per-dataset noise and is therefore
    NOISIER, so applying a 12ds floor to it would understate the noise and manufacture
    resolvability.
  * A cell whose Waiv denominator is smaller than the benchmark's own seed noise is
    INDETERMINATE, not a score.  Detected generally as |waiv_gain| < floor, never by
    naming the offending cells.  (Today that catches midnight/linear_probing,
    virchow2/knn -- where Waiv REGRESSED, making the denominator negative -- and
    virchow2/linear_probing.)
  * Every cell carries a 95% CI.  A cell whose CI straddles the 70 bar is NOT RESOLVED:
    the data cannot tell PASS from FAIL, and saying either would be a claim the error
    bars do not support.
  * PASS is never printed for a cell that is PARTIAL, INDETERMINATE, or NOT RESOLVED,
    and the overall verdict is INDETERMINATE if any required cell is.
  * A cell built from fewer than MIN_N_FOR_VERDICT seeds is UNDERPOWERED and is graded
    by NOTHING.  Every CI path here will happily manufacture an error bar at n=1 -- RI
    falls back to the measured floor, HEST computes 2*SD/sqrt(1), THUNDER computes
    floor/gain/sqrt(1) -- and none of those is an across-seed measurement.
  * The overall average is the mean of the three benchmark means, and is UNDEFINED
    unless all three cover all three backbones.  A benchmark mean over a subset of
    backbones is a different quantity that happens to share the name; letting it carry
    a full 1/3 weight lets one backbone's single cell stand in for a benchmark.
  * Error bars for correlated readouts are combined by their MEAN, not in quadrature.
    The three THUNDER tasks read the same checkpoints over the same 12 datasets.
  * THUNDER half-widths come from the across-SEED SD of the 12-dataset task mean, not
    from the offset-2SE resolvability floor (whose SD is over DATASETS).  The floor
    keeps its own, separate job: gating cells whose Waiv denominator is itself noise.

See docs/EVAL_FIXES_2026-08-26.md for the audit these last five rules came from.

Usage:
    python3 scripts/final_recipe_report.py
    python3 scripts/final_recipe_report.py --json     # also writes docs/final_recipe_verdict.json
    python3 scripts/final_recipe_report.py --hest-assume-step 125   # see F5 in that doc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse, do not re-transcribe.  collect_thunder._score handles the two nesting shapes
# (knn persists only the selected k; simple_shot keeps every shot count and we want 16),
# and collect_final5 owns the base-dir mapping + the 12-dataset roster.
import collect_final5 as _c5          # noqa: E402
import collect_thunder as _ct         # noqa: E402
import eval_common as _ec             # noqa: E402
import scoreboard as _sb              # noqa: E402

PAPER_CLS = _c5.PAPER_CLS                       # the 12 classification datasets
THUNDER_ROOT = _c5.THUNDER_ROOT
WAIV_THUNDER = _sb.WAIV_THUNDER                 # published Table 2, 0-100 scale
WAIV_THUNDER_SOURCE = _sb.WAIV_THUNDER_SOURCE

# Arms this report can GRADE, derived -- not typed.  Grading a cell needs three
# denominators that only exist once the corresponding measurement has been made:
# our RI base, our HEST base, and a measured THUNDER seed floor.  `_c5.ARMS` is the
# full roster (it now includes `hoptimus` and `uni2`); the arms without those inputs
# are not silently dropped, they are listed in report["arms_not_reportable"] with the
# reason, and any discovered run belonging to one of them is reported there too.
# Today this evaluates to exactly ("phikon", "midnight", "virchow2").
ALL_ARMS = _c5.ARMS
CLS_TASKS = ("knn", "linear_probing", "simple_shot")   # segmentation deliberately excluded

RUN_GLOB = "genMASK-c3s-*"
CI_TARGET = 0.75
# F13 fix: the stopping rule's mean is only comparable across checkpoints when it is
# taken over the SAME dataset panel every time.
N_CI_DATASETS = 3   # camelyon, tolkach_esca, tcga          # stopping rule: first ckpt with mean confounder_insensitivity >= this
PASS_BAR = 70.0           # every cell must clear this
OVERALL_BAR = 80.0        # ... and the mean of the three benchmark means must exceed this

# ---------------------------------------------------------------------------
# Constants supplied with the brief (authoritative).
# ---------------------------------------------------------------------------
# F6 fix (2026-08-26): HEST base is no longer hardcoded here.  It is read from disk by
# collect_final5._load_hest_base(), from the SAME field
# (hest_perf_per_encoder.custom_encoder) that the fine-tuned reader uses.  This file
# previously hardcoded the rounded `results.avg` value while reading `results.avg` for
# FT -- self-consistent, but inconsistent with collect_final5/scoreboard, which read
# `custom_encoder`.  One field, repo-wide, now.
HEST_BASE = _c5.HEST_BASE
HEST_BASE_SOURCE = _c5.HEST_BASE_SOURCE

# F-F fix (2026-08-26): RI_BASE and RI_WAIV are no longer literals here.
#
#   RI_BASE  used to be {phikon 0.4686, midnight 0.7589, virchow2 0.8582} in FIVE files,
#            attributed by collect_final5's comment to probe_before.json.  THAT
#            PROVENANCE IS FALSE -- probe_before.json is the PLISM cross-scanner /
#            cross-stain probe and has no robustness_index field.  The real measurement
#            is PathoROB's own results_summary.json for the untuned feature dirs, and
#            that is what eval_common.load_ri_base() reads.
#   RI_WAIV  used to be re-typed here; it is a transcription of Waiv Table 1 that already
#            has exactly one owner, src/waivphaet/eval/pathorob_adapter.TARGETS.
#   HEST_WAIV likewise now has one owner (eval_common), for the same reason.
#
# The literals are kept ONLY as an assertion target: a disagreement between the value on
# disk and the value that was published is itself a bug, so it is measured and reported
# rather than quietly absorbed.
RI_BASE, RI_BASE_SOURCE = _ec.load_ri_base()
RI_WAIV, RI_WAIV_SOURCE = _ec.load_ri_waiv()
HEST_WAIV = _ec.HEST_WAIV
HEST_WAIV_SOURCE = _ec.HEST_WAIV_SOURCE

_RETIRED_LITERALS = {
    "RI_BASE": {"phikon": 0.4686, "midnight": 0.7589, "virchow2": 0.8582},
    "RI_WAIV": {"phikon": 0.806, "midnight": 0.924, "virchow2": 0.918},
}


def _literal_agreement():
    """Compare every retired literal against the value now read from disk."""
    out = {}
    for name, lit in _RETIRED_LITERALS.items():
        live = {"RI_BASE": RI_BASE, "RI_WAIV": RI_WAIV}[name]
        for a, v in lit.items():
            d = live[a] - v
            out["%s/%s" % (name, a)] = {
                "retired_literal": v, "from_disk": live[a], "delta": d,
                "agrees_to_4dp": abs(d) < 5e-5,
            }
    return out


# F-A fix (2026-08-26): the HEST seed SD is DERIVED FROM DISK, not hand-set.
#
# It used to be `HEST_SD_PCT = {phikon 5.8, midnight 8.3, virchow2 14.2}` -- a literal
# with no on-disk source and no producing script, which nevertheless determined every
# HEST confidence interval in this verdict.  It disagreed with both other estimates of
# the same quantity in the repo (phikon's literal was 32% NARROWER than either).
#
# The replacement is produced by scripts/hest_seed_sd.py using EXACTLY the estimator
# scoreboard.NOISE_SD documents at scripts/scoreboard.py:161-174 (pooled within-recipe
# across-seed SD, sqrt(sum df_f sd_f^2 / sum df_f)), read from docs/hest_seed_sd.json --
# mirroring how docs/thunder_seed_floor_12ds.json is produced and consumed.
#
# It is stored in RAW metric units, per (backbone, step), and converted to pct-of-waiv
# points at the point of use, because that conversion depends on the per-backbone gain.
HEST_SEED_SD, HEST_SEED_SD_PATH, _HEST_SD_BLOB = _ec.load_hest_seed_sd()
HEST_SEED_SD_SOURCE = "%s (estimator: %s)" % (HEST_SEED_SD_PATH, _HEST_SD_BLOB["estimator"])

# THUNDER 12-dataset seed floors, offset-2SE form, per (backbone, task), in RAW F1
# fractions.  Source: docs/thunder_seed_floor_12ds.md.  Valid ONLY for a full 12/12 mean.
THUNDER_FLOOR = {
    "phikon":   {"knn": 0.0233, "linear_probing": 0.0097, "simple_shot": 0.0087},
    "midnight": {"knn": 0.0100, "linear_probing": 0.0087, "simple_shot": 0.0104},
    "virchow2": {"knn": 0.0083, "linear_probing": 0.0088, "simple_shot": 0.0066},
}
THUNDER_FLOOR_SOURCE = "docs/thunder_seed_floor_12ds.md (n=5 training seeds, offset-2SE, 12/12 coverage)"

# --- F4 fix (2026-08-26): THUNDER error bars -------------------------------------
# THUNDER_FLOOR above is `offset_2se` = |mean(d)| + 2*SD(d)/sqrt(12), where d is the
# PER-DATASET F1 delta between two seed replicates and the SD is taken OVER THE 12
# DATASETS.  That is a resolvability floor -- "is Waiv's own gain even bigger than
# seed noise" -- and it is what the INDETERMINATE gate below still uses.  It is NOT a
# 95% half-width on our task mean: SD-over-datasets is the wrong variance component
# for that, and dividing it by sqrt(n_runs) (as this script used to) compounds the
# error.  The right quantity is `seed_sd_of_task_mean` -- the SD of the 12-dataset
# task mean itself across the 5 training seeds.  2*that is a one-run 95% half-width;
# averaging n independent runs shrinks it by sqrt(n).
# Source: docs/thunder_seed_floor_12ds.json -> cells[<bb>/<task>]["12ds"]
#                                              ["seed_sd_of_task_mean"]
THUNDER_SEED_SD_SOURCE = ("docs/thunder_seed_floor_12ds.json "
                          "cells[bb/task].12ds.seed_sd_of_task_mean (n=5 seeds, 12/12)")

_SEED_SD_FALLBACK = {
    "phikon":   {"knn": 0.006390, "linear_probing": 0.001888, "simple_shot": 0.002637},
    "midnight": {"knn": 0.002441, "linear_probing": 0.002126, "simple_shot": 0.002541},
    "virchow2": {"knn": 0.002108, "linear_probing": 0.002381, "simple_shot": 0.001978},
}


def _load_thunder_seed_sd():
    """Read seed_sd_of_task_mean from the floor JSON; fall back to the literals."""
    p = REPO / "docs" / "thunder_seed_floor_12ds.json"
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return {k: dict(v) for k, v in _SEED_SD_FALLBACK.items()}, "FALLBACK LITERALS"
    out = {a: {} for a in _SEED_SD_FALLBACK}
    for key, cell in (blob.get("cells") or {}).items():
        bb, _, task = key.partition("/")
        v = ((cell.get("12ds") or {}).get("seed_sd_of_task_mean"))
        if bb in out and v is not None:
            out[bb][task] = v
    for a, d in _SEED_SD_FALLBACK.items():
        for t, v in d.items():
            out.setdefault(a, {}).setdefault(t, v)
    return out, str(p)


THUNDER_SEED_SD, THUNDER_SEED_SD_PATH = _load_thunder_seed_sd()

# --- F2 fix (2026-08-26): minimum-n gate -----------------------------------------
# Every CI path in this script will happily manufacture an error bar at n=1:
# RI falls back to floor_ci, HEST computes 2*sd/sqrt(1), THUNDER computes
# floor/gain*100/sqrt(1).  None of those is an across-seed measurement, so an n=1
# cell was being graded PASS/FAIL on a number no data supports.  Cells below this
# many seeds now return UNDERPOWERED and are never scored.
MIN_N_FOR_VERDICT = 2

# F9 fix (2026-08-26): single source of truth for the HEST pooling rule.
HEST_POOLING = {a: _c5.hest_pooling(a) for a in ALL_ARMS}


def _missing_denominators(arm: str) -> list[str]:
    """Which grading inputs this arm still lacks.  Empty list = fully gradeable."""
    missing = []
    if arm not in RI_BASE:
        missing.append("RI base (eval_common.RI_BASE_MODEL_DIRS)")
    if arm not in RI_WAIV:
        missing.append("Waiv RI target (pathorob_adapter.TARGETS)")
    if arm not in HEST_BASE:
        missing.append("HEST base (collect_final5.HEST_BASE_FILES)")
    if arm not in HEST_WAIV:
        missing.append("Waiv HEST target (eval_common.HEST_WAIV)")
    if arm not in THUNDER_FLOOR:
        missing.append("THUNDER seed floor (docs/thunder_seed_floor_12ds.md)")
    return missing


ARMS = tuple(a for a in ALL_ARMS if not _missing_denominators(a))
ARMS_NOT_REPORTABLE = {a: _missing_denominators(a) for a in ALL_ARMS if a not in ARMS}

# Statuses that must never be reported as a score.
UNGRADED = {"PARTIAL", "INDETERMINATE", "NO_DATA", "NOT RESOLVED", "UNDERPOWERED"}


# ---------------------------------------------------------------------------
# Run discovery + the stopping rule
# ---------------------------------------------------------------------------
# Built from the arm roster, NOT typed out.  A hand-written alternation is a filter that
# looks like a parser: a run for a backbone missing from it does not error, it simply
# never appears in the report, and the report's own coverage checks (which count runs,
# not directories) cannot see the omission.
RUN_RE = re.compile(r"-(%s)-s(\d+)-t\d+-(\d+)$" % "|".join(ALL_ARMS))


def discover_runs() -> list[dict]:
    """Glob the recipe's runs and parse (backbone, seed, train job id) out of each name."""
    out = []
    for d in sorted((REPO / "runs").glob(RUN_GLOB)):
        if not d.is_dir():
            continue
        m = RUN_RE.search(d.name)
        if not m:
            continue
        out.append({
            "run": d.name,
            "backbone": m.group(1),
            "seed": int(m.group(2)),
            "jobid": m.group(3),
            "run_dir": d,
        })
    return out


def select_step(run_dir: Path) -> tuple[int | None, float | None, list[dict]]:
    """Apply the stopping rule to one run's ri_curve.json.

    Returns (selected_step, mean_confounder_insensitivity_at_that_step, trace) where
    trace is the per-checkpoint mean CI so the reader can audit the choice.  Returns
    (None, None, trace) when no checkpoint on the curve ever reaches the target -- that
    is a real outcome (the run has not gone far enough), not an error to paper over.
    """
    p = run_dir / "ri_curve.json"
    trace: list[dict] = []
    if not p.exists():
        return None, None, trace
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return None, None, trace
    for pt in blob.get("points", []):
        ds = pt.get("datasets") or {}
        per_ds = {k: v.get("confounder_insensitivity") for k, v in ds.items()
                  if isinstance(v, dict) and v.get("confounder_insensitivity") is not None}
        vals = list(per_ds.values())
        mean_ci = (sum(vals) / len(vals)) if vals else None
        # F13 fix (2026-08-26): the mean used to be taken over "datasets present in this
        # point", so a checkpoint probed WITHOUT camelyon (which sits at 0.08-0.40 while
        # the others reach 1.87) would show a mean ~+0.25 too high and trip the 0.75 gate
        # early.  A point is now eligible only with all N_CI_DATASETS present; short
        # points stay in the trace, flagged, so the skip is auditable.
        eligible = len(vals) == N_CI_DATASETS
        trace.append({"step": pt.get("step"), "mean_ci": mean_ci, "n_datasets": len(vals),
                      "per_dataset_ci": per_ds, "eligible": eligible,
                      "skip_reason": (None if eligible else
                                      "only %d/%d CI datasets present"
                                      % (len(vals), N_CI_DATASETS))})
    for row in trace:
        if row["eligible"] and row["mean_ci"] is not None and row["mean_ci"] >= CI_TARGET:
            return row["step"], row["mean_ci"], trace
    return None, None, trace


def ri_at_step(run_dir: Path, step: int) -> float | None:
    p = run_dir / "ri_curve.json"
    if not p.exists():
        return None
    blob = json.loads(p.read_text())
    for pt in blob.get("points", []):
        if pt.get("step") == step:
            return pt.get("avg_robustness_index")
    return None


# ---------------------------------------------------------------------------
# Benchmark readers
# ---------------------------------------------------------------------------
def hest_score(run: str, step: int, backbone: str) -> tuple[float | None, str]:
    """One HEST score, through collect_final5's ONE loader (F6 fix, 2026-08-26).

    This used to read `results.avg` -- a rounded field -- while collect_final5 and
    scoreboard read `hest_perf_per_encoder.custom_encoder`, and the hardcoded base
    came from `results.avg` too.  Base and FT must come from the same field.
    """
    pool = HEST_POOLING[backbone]
    p = _c5.HEST_WORK_DIR / "results" / f"f5_{run}_s{step:07d}_{pool}_summary.json"
    return _c5._hest_score(run, step, backbone), str(p)


def thunder_per_ds(model: str, task: str) -> dict[str, float]:
    """Per-dataset F1 for one model dir + task, over the 12 PAPER_CLS sets.

    Missing files are ABSENT from the dict rather than None-filled, so the caller can
    tell 'not evaluated yet' from 'evaluated and null' -- that distinction is what the
    PARTIAL guard runs on.
    """
    res_root = THUNDER_ROOT / "outputs" / "res"
    out: dict[str, float] = {}
    for ds in PAPER_CLS:
        p = res_root / ds / model / task / "frozen" / "outputs.json"
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        f1, _ece = _ct._score(blob, task)      # reuse: handles knn-k / simple_shot-shot nesting
        if f1 is not None:
            out[ds] = f1
    return out


def thunder_model_name(backbone: str, seed: int, jobid: str, step: int) -> str:
    return f"f5_ci-{backbone}-s{seed}-{jobid}_s{step:07d}"


def thunder_base_12ds(backbone: str) -> dict[str, tuple[float | None, int]]:
    """Our own BASE (unfinetuned) 12-dataset task means, with coverage.

    Read through collect_final5's base-dir mapping so base and FT go through the same
    path and the same pooling convention per backbone.
    """
    per_ds = _c5._thunder_base_per_ds(backbone)
    out = {}
    for task in CLS_TASKS:
        vals = [v for k, v in (per_ds.get(task) or {}).items() if k in PAPER_CLS and v is not None]
        out[task] = ((sum(vals) / len(vals)) if len(vals) == len(PAPER_CLS) else None, len(vals))
    return out


# ---------------------------------------------------------------------------
# pct_of_waiv + resolution
# ---------------------------------------------------------------------------
def pct_of_waiv_uncapped(ours: float, base: float, waiv: float) -> float:
    """(ours - base) / (waiv - base) * 100, NOT capped.

    This is the MEASUREMENT.  The 100 cap is a reporting convention, applied later by
    cap100() -- see the note there.
    """
    return (ours - base) / (waiv - base) * 100.0


def cap100(pct: float) -> float:
    """Apply the reporting cap: 'exceeded Waiv' counts as 100.

    Capped because 'exceeded Waiv' is still just 'matched the target' for a criterion
    that asks whether we reached them; letting a cell run to 140 would let one backbone
    buy off another's shortfall in the average.  This is a REPORTING/AGGREGATION rule,
    NOT a measurement: it must never be applied before the >=70 resolution test, because
    censoring a 137.7 to 100 and then subtracting a 63-point CI manufactures a spurious
    "NOT RESOLVED" out of a cell whose real lower bound is 74.7.
    """
    return min(100.0, pct)


# Backwards-compatible alias kept intentionally NOT defined: any remaining caller of the
# old capped-then-tested `pct_of_waiv` should fail loudly rather than silently regress.


def resolve(pct: float | None, ci: float | None, n: int | None = None) -> str:
    """Grade one cell against the 70 bar, honouring its error bar.

    `pct` MUST be the UNCAPPED point estimate.  The bar is a statement about the true
    fraction of Waiv's gain we captured, and the cap does not change that fraction.
    """
    if pct is None:
        return "NO_DATA"
    if n is not None and n < MIN_N_FOR_VERDICT:
        return "UNDERPOWERED"          # F2: below min n, any CI here is manufactured
    if ci is None:
        return "NOT RESOLVED"          # no error bar => no defensible verdict
    if pct - ci >= PASS_BAR:
        return "PASS"
    if pct + ci < PASS_BAR:
        return "FAIL"
    return "NOT RESOLVED"


def grade(pct_uncapped: float | None, ci: float | None, n: int | None = None) -> dict:
    """Build the {pct, pct_capped, pct_uncapped, ci, lower/upper_uncapped, status} block.

    F-C fix (2026-08-26).  BOTH HALVES OF THE CRITERION NOW RUN ON THE SAME QUANTITY.
    `pct` -- the field that feeds the benchmark averages, the overall average and the
    worst-cell search -- is the UNCAPPED estimate, which is also what the >=70 test
    runs on.  Previously `resolve()` tested the uncapped interval while the averages
    summed the CAPPED value, so phikon/THUNDER printed 79.7, graded PASS on 99.6, and
    carried a capped lower bound of 56.9: three different numbers for one cell, and a
    reader could not reproduce the verdict from any of them.

    WHY UNCAPPED IS THE RIGHT SIDE TO UNIFY ON.  The cap is a censoring operator.  It
    changes the point estimate but not the error bar, so a capped point with an uncapped
    interval is not an interval for anything; and averaging censored values makes the
    benchmark mean a function of how far ABOVE the target the best cells landed being
    thrown away, which is a different quantity from "the fraction of Waiv's gain we
    captured".  The cap survives ONLY as presentation: `pct_capped` is printed next to
    the uncapped value and marked with a star, so the old convention stays legible
    without ever entering an arithmetic path.
    """
    if pct_uncapped is None:
        return {"pct": None, "pct_capped": None, "pct_uncapped": None, "ci": ci,
                "lower_uncapped": None, "upper_uncapped": None, "status": "NO_DATA"}
    status = resolve(pct_uncapped, ci, n)
    capped = cap100(pct_uncapped)
    if status == "UNDERPOWERED":
        # F2: an UNDERPOWERED cell must not feed averages or the worst-cell search.
        return {"pct": None, "pct_capped": None, "pct_uncapped": pct_uncapped,
                "ci": ci, "lower_uncapped": None, "upper_uncapped": None,
                "was_capped": False, "n": n, "status": "UNDERPOWERED",
                "reason": "n=%s < MIN_N_FOR_VERDICT=%d" % (n, MIN_N_FOR_VERDICT)}
    return {
        "pct": pct_uncapped,                 # graded AND averaged: one quantity (F-C)
        "pct_capped": capped,                # presentation only
        "pct_uncapped": pct_uncapped,
        "ci": ci,
        "lower_uncapped": (pct_uncapped - ci) if ci is not None else None,
        "upper_uncapped": (pct_uncapped + ci) if ci is not None else None,
        "was_capped": capped < pct_uncapped - 1e-9,
        "status": status,
    }


def withheld(reason: str, extra: dict | None = None) -> dict:
    """A cell whose denominator failed the shared resolvability gate (F-B).

    Identical shape and identical status for RI, HEST and THUNDER: the benchmark a cell
    belongs to must not change what happens when its denominator is noise.
    """
    d = {"pct": None, "pct_capped": None, "pct_uncapped": None, "ci": None,
         "lower_uncapped": None, "upper_uncapped": None, "was_capped": False,
         "n": 0, "status": "INDETERMINATE", "reason": reason}
    if extra:
        d.update(extra)
    return d


def gate_denominator(bench: str, arm: str, gain: float | None, seed_sd: float | None,
                     sd_note: str = "") -> dict | None:
    """Run THE shared denominator gate.  Returns a withheld cell, or None to continue.

    F-B fix (2026-08-26).  This report used to gate THUNDER (|waiv_gain| < offset-2SE
    floor) and NOT gate RI or HEST at all, while scoreboard.py gated RI and HEST on
    `one seed-SD > 10 pct_of_waiv points` and never printed the offending cell.  The two
    files therefore disagreed on the SAME cell: scoreboard WITHHELD virchow2/HEST while
    this report printed 72.9 for it and averaged it into the HEST mean.  There is now one
    implementation (eval_common.denominator_unresolvable) and one threshold, applied to
    all three benchmarks with each benchmark supplying the same KIND of noise estimate --
    the across-seed SD, in raw units, of the exact statistic being divided.
    """
    unres, sd_pct, why = _ec.denominator_unresolvable(gain, seed_sd)
    if not unres:
        return None
    return withheld(
        "%s/%s denominator gate: %s%s" % (arm, bench, why, (" [%s]" % sd_note) if sd_note else ""),
        {"waiv_gain": gain, "seed_sd": seed_sd, "seed_sd_pct_points": sd_pct,
         "gate": "one seed-SD > %.0f pct_of_waiv points (eval_common)"
                 % _ec.UNRESOLVABLE_SD_PCT_LIMIT})


def fmt_cell(c: dict) -> str:
    """`graded (capped) +/-CI n= status` -- the graded value FIRST (F-C).

    The graded/averaged number is the uncapped one; the capped value follows in
    parentheses and is marked * when the cap actually bit, so it is visible that the two
    differ without either being mistaken for the other.
    """
    if c.get("pct") is None:
        return f"{c['status']:<16}"
    ci = c.get("ci")
    ci_s = f"+/-{ci:5.1f}" if ci is not None else "  +/-  ? "
    star = "*" if c.get("was_capped") else " "
    cap = c.get("pct_capped")
    cap_s = f"({cap:5.1f}{star})" if cap is not None else ""
    return f"{c['pct']:6.1f}{cap_s} {ci_s} n={c.get('n', 0)}  {c['status']}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_report(hest_assume_step: int | None = None) -> dict:
    """Build the verdict report.

    `hest_assume_step` (F5 fix, 2026-08-26) is an explicit, opt-in, reversible escape
    hatch.  Runs that have finished training and have HEST summaries at every step but
    have NO ri_curve.json cannot be assigned a step by the stopping rule, and used to be
    dropped from ALL THREE benchmarks.  With this flag they additionally enter a clearly
    marked SUPPLEMENTARY HEST cell at the given assumed step.  They never enter the RI or
    THUNDER cells (both of which need the curve), and they never enter the primary
    rule-selected HEST cell that the verdict is scored on.
    """
    runs = discover_runs()
    report: dict = {
        "criterion": {
            "pass_bar_per_cell": PASS_BAR,
            "overall_average_bar": OVERALL_BAR,
            "pct_capped_at": 100,
            "cap_is_presentation_only": ("F-C (2026-08-26): the 100 cap no longer enters "
                                         "ANY arithmetic.  The >=70 per-cell test, the "
                                         "benchmark means, the overall average and the "
                                         "worst-cell search all run on the UNCAPPED "
                                         "pct_of_waiv.  pct_capped is printed beside it "
                                         "so the old convention stays legible."),
            "graded_quantity": "pct_of_waiv, UNCAPPED, identical for the >=70 test and "
                               "for every average",
            "scored_by": "worst (backbone, benchmark) cell",
        },
        "stopping_rule": f"first checkpoint with mean confounder_insensitivity >= {CI_TARGET}",
        "sources": {
            "waiv_thunder": WAIV_THUNDER_SOURCE,
            "thunder_floors": THUNDER_FLOOR_SOURCE,
            "hest_seed_sd": HEST_SEED_SD_SOURCE,
            "ri_base": RI_BASE_SOURCE,
            "ri_waiv": RI_WAIV_SOURCE,
            "hest_waiv": HEST_WAIV_SOURCE,
            "denominator_gate": ("eval_common.denominator_unresolvable -- one seed-SD > "
                                 "%.0f pct_of_waiv points, applied identically to RI, "
                                 "HEST and THUNDER" % _ec.UNRESOLVABLE_SD_PCT_LIMIT),
            "ci_construction": ("eval_common.ci95 -- max(empirical 2*SD/sqrt(n), measured "
                                "seed floor 2*SD/sqrt(n)), applied identically to RI, "
                                "HEST and THUNDER"),
            "thunder_seed_sd": THUNDER_SEED_SD_SOURCE,
            "thunder_seed_sd_path": THUNDER_SEED_SD_PATH,
            "hest_base": HEST_BASE_SOURCE,
        },
        "min_n_for_verdict": MIN_N_FOR_VERDICT,
        # Arms that exist in the roster but cannot be GRADED yet, with the reason.
        # Recorded rather than dropped: "absent from the table" and "not measurable yet"
        # look identical in the output otherwise, and only one of them is a to-do.
        "arms_not_reportable": {
            a: {
                "missing_inputs": why,
                "runs_found": [r["run"] for r in runs if r["backbone"] == a],
            }
            for a, why in ARMS_NOT_REPORTABLE.items()
        },
        "retired_literal_agreement": _literal_agreement(),
        "hest_assume_step": hest_assume_step,
        "runs": [],
        "cells": {},
    }

    # ---- per-run: apply the rule, then read the three benchmarks -------------
    for r in runs:
        step, mean_ci, trace = select_step(r["run_dir"])
        rec = {
            "run": r["run"], "backbone": r["backbone"], "seed": r["seed"],
            "train_jobid": r["jobid"],
            "selected_step": step, "mean_confounder_insensitivity": mean_ci,
            "ci_trace": trace,
        }
        if step is None:
            has_curve = (r["run_dir"] / "ri_curve.json").exists()
            rec["note"] = (f"no checkpoint on the curve reaches mean CI >= {CI_TARGET}"
                           if has_curve else "no ri_curve.json -- stopping rule cannot "
                                             "be applied (RI and THUNDER cells excluded)")
            rec["has_ri_curve"] = has_curve
            # F5: HEST-only admission at an EXPLICIT assumed step.  Deliberately does
            # NOT set selected_step, so this run stays out of by_bb and therefore out of
            # the RI and THUNDER cells.
            if hest_assume_step is not None and not has_curve:
                h, hp = hest_score(r["run"], hest_assume_step, r["backbone"])
                if h is not None:
                    rec["assumed_step"] = hest_assume_step
                    rec["hest"] = h
                    rec["hest_path"] = hp
                    rec["hest_step_is_assumed"] = True
            report["runs"].append(rec)
            continue
        rec["ri"] = ri_at_step(r["run_dir"], step)
        h, hp = hest_score(r["run"], step, r["backbone"])
        rec["hest"] = h
        rec["hest_path"] = hp
        model = thunder_model_name(r["backbone"], r["seed"], r["jobid"], step)
        rec["thunder_model"] = model
        rec["thunder"] = {}
        for task in CLS_TASKS:
            per_ds = thunder_per_ds(model, task)
            rec["thunder"][task] = {
                "coverage": len(per_ds),
                "mean": (sum(per_ds.values()) / len(per_ds)) if len(per_ds) == len(PAPER_CLS) else None,
                "partial_mean": (sum(per_ds.values()) / len(per_ds)) if per_ds else None,
                "datasets_present": sorted(per_ds),
                "datasets_missing": [d for d in PAPER_CLS if d not in per_ds],
            }
        report["runs"].append(rec)

    by_bb = {a: [x for x in report["runs"] if x["backbone"] == a and x.get("selected_step")] for a in ARMS}

    # ---- RI -----------------------------------------------------------------
    for a in ARMS:
        vals = [x["ri"] for x in by_bb[a] if x.get("ri") is not None]
        n = len(vals)
        if n == 0:
            report["cells"].setdefault(a, {})["RI"] = {"pct": None, "ci": None, "n": 0, "status": "NO_DATA"}
            continue
        step_a = by_bb[a][0]["selected_step"]
        # The measured RI seed SD, per (backbone, step), through the ONE step-selection
        # rule shared with HEST (eval_common.seed_sd_at_step): exact step if measured,
        # otherwise the largest SD measured for that backbone at any step -- an
        # over-estimate by construction, the safe direction for an error bar.
        ri_per_step = {st: v.get("ri") for st, v in (_sb.NOISE_SD.get(a, {}) or {}).items()}
        raw_sd, sd_note = _ec.seed_sd_at_step(ri_per_step, step_a)
        gain = RI_WAIV[a] - RI_BASE[a]

        # F-B: the shared denominator gate, run BEFORE any number is formed.
        g = gate_denominator("RI", a, gain, raw_sd, sd_note)
        if g is not None:
            g.update({"base": RI_BASE[a], "waiv": RI_WAIV[a], "selected_step": step_a})
            report["cells"].setdefault(a, {})["RI"] = g
            continue

        pcts_unc = [pct_of_waiv_uncapped(v, RI_BASE[a], RI_WAIV[a]) for v in vals]
        mean_pct_unc = sum(pcts_unc) / n
        # F-D: ONE CI construction for all three benchmarks -- max(empirical, floor).
        # The empirical across-seed SD alone is not trustworthy (one degree of freedom at
        # n=2, and censoring can collapse it to exactly 0, claiming infinite precision);
        # the measured floor alone ignores the spread actually observed.  Never report an
        # error bar narrower than the instrument's known noise.
        floor_sd_pct = (raw_sd / gain * 100.0) if raw_sd is not None else None
        ci, emp_ci, floor_ci, ci_src = _ec.ci95(pcts_unc, floor_sd_pct)
        capped = sum(1 for q in pcts_unc if q >= 100.0)
        ci_src += " [floor %s]" % sd_note
        if capped:
            ci_src += "; %d/%d seeds exceed Waiv (uncapped values are used, so the " \
                      "spread is NOT censored)" % (capped, n)
        cell = grade(mean_pct_unc, ci, n)
        cell.update({
            "n": n,
            "raw_mean": sum(vals) / n, "base": RI_BASE[a], "waiv": RI_WAIV[a],
            "selected_step": step_a,
            "per_seed_pct": pcts_unc, "per_seed_pct_uncapped": pcts_unc,
            "per_seed_pct_capped": [cap100(q) for q in pcts_unc],
            "seed_sd_raw": raw_sd, "seed_sd_pct_points": floor_sd_pct,
            "empirical_ci": emp_ci, "floor_ci": floor_ci,
            "ci_source": ci_src,
        })
        report["cells"].setdefault(a, {})["RI"] = cell

    # ---- HEST ---------------------------------------------------------------
    for a in ARMS:
        vals = [x["hest"] for x in by_bb[a] if x.get("hest") is not None]
        n = len(vals)
        if n == 0:
            report["cells"].setdefault(a, {})["HEST"] = {"pct": None, "ci": None, "n": 0, "status": "NO_DATA"}
            continue
        step_a = by_bb[a][0]["selected_step"]
        # F-A: derived from disk, same estimator as scoreboard.NOISE_SD, same step rule
        # as RI above.
        raw_sd, sd_note = _ec.seed_sd_at_step(HEST_SEED_SD.get(a, {}), step_a)
        gain = HEST_WAIV[a] - HEST_BASE[a]

        g = gate_denominator("HEST", a, gain, raw_sd, sd_note)
        if g is not None:
            g.update({"base": HEST_BASE[a], "waiv": HEST_WAIV[a], "selected_step": step_a,
                      "pooling": HEST_POOLING[a], "n_seeds_available": n,
                      "raw_mean": sum(vals) / n})
            report["cells"].setdefault(a, {})["HEST"] = g
            continue

        pcts_unc = [pct_of_waiv_uncapped(v, HEST_BASE[a], HEST_WAIV[a]) for v in vals]
        mean_pct_unc = sum(pcts_unc) / n
        floor_sd_pct = raw_sd / gain * 100.0
        ci, emp_ci, floor_ci, ci_src = _ec.ci95(pcts_unc, floor_sd_pct)
        cell = grade(mean_pct_unc, ci, n)
        cell.update({
            "n": n,
            "raw_mean": sum(vals) / n, "base": HEST_BASE[a], "waiv": HEST_WAIV[a],
            "selected_step": step_a,
            "per_seed_pct": pcts_unc, "per_seed_pct_uncapped": pcts_unc,
            "per_seed_pct_capped": [cap100(q) for q in pcts_unc],
            "pooling": HEST_POOLING[a],
            "step_source": "stopping rule",
            "seed_sd_raw": raw_sd, "seed_sd_pct_points": floor_sd_pct,
            "seed_sd_df": ((_HEST_SD_BLOB["pooled_seed_sd"].get(a, {}) or {})
                           .get(str(step_a), {}) or {}).get("df"),
            "empirical_ci": emp_ci, "floor_ci": floor_ci,
            "ci_source": ci_src + " [floor %s]" % sd_note,
        })
        report["cells"].setdefault(a, {})["HEST"] = cell

        # --- F5 SUPPLEMENTARY HEST cell: rule-selected runs PLUS assumed-step runs ---
        # Reported alongside, never in place of, the primary cell.  It is not scored and
        # does not feed any average; it exists so that HEST evidence sitting on disk is
        # visible instead of silently discarded.
        sup_rows = [x for x in report["runs"]
                    if x["backbone"] == a and x.get("hest") is not None
                    and (x.get("selected_step") or x.get("assumed_step"))]
        if len(sup_rows) > n:
            sv = [x["hest"] for x in sup_rows]
            sn = len(sv)
            s_unc = [pct_of_waiv_uncapped(v, HEST_BASE[a], HEST_WAIV[a]) for v in sv]
            s_ci, _, _, s_src = _ec.ci95(s_unc, floor_sd_pct)
            scell = grade(sum(s_unc) / sn, s_ci, sn)
            scell.update({
                "n": sn,
                "raw_mean": sum(sv) / sn, "base": HEST_BASE[a], "waiv": HEST_WAIV[a],
                "per_seed_pct": s_unc, "per_seed_pct_uncapped": s_unc,
                "pooling": HEST_POOLING[a],
                "step_source": f"stopping rule where available, ASSUMED step "
                               f"{hest_assume_step} for runs with no ri_curve.json",
                "assumed_step_runs": [x["run"] for x in sup_rows
                                      if x.get("hest_step_is_assumed")],
                "ci_source": s_src,
                "NOT_SCORED": "supplementary only; the verdict uses the primary "
                              "rule-selected HEST cell",
            })
            report.setdefault("hest_supplementary", {})[a] = scell

    # ---- THUNDER ------------------------------------------------------------
    for a in ARMS:
        base12 = thunder_base_12ds(a)
        tasks_out: dict[str, dict] = {}
        for task in CLS_TASKS:
            floor = THUNDER_FLOOR[a][task]
            base, base_cov = base12[task]
            waiv_gain = (WAIV_THUNDER[a]["ft"][task] - WAIV_THUNDER[a]["base"][task]) / 100.0
            entry = {
                "waiv_gain": waiv_gain, "floor": floor,
                "our_base": base, "our_base_coverage": f"{base_cov}/{len(PAPER_CLS)}",
                "pct": None, "pct_capped": None, "pct_uncapped": None, "ci": None,
                "lower_uncapped": None, "upper_uncapped": None, "was_capped": False,
            }
            # (1) degenerate denominator -- checked FIRST and generally, so a cell whose
            # denominator is noise never gets a number printed next to it regardless of
            # how complete our own data is.
            #
            # F-B fix (2026-08-26): this test is now the SHARED gate, identical to the
            # one RI and HEST run above, instead of THUNDER's own private
            # `|waiv_gain| < offset_2se`.  Two things changed and both were wrong before:
            #   * the THRESHOLD.  The old test asked only "is Waiv's gain bigger than the
            #     noise", i.e. can a full-gain arm be told from a zero-gain arm.  RI and
            #     HEST have always been held to the harder question the 70/80 criterion
            #     actually poses -- can 80% of the gain be told from 100% -- which is
            #     `2*SD < 20% of the gain`.  A benchmark does not get an easier bar
            #     because it is THUNDER.
            #   * the NOISE ESTIMATE.  offset_2se takes its SD OVER THE 12 DATASETS; the
            #     other two benchmarks supply an across-SEED SD.  The gate now takes the
            #     across-seed SD of the 12-dataset task mean (the same quantity that
            #     builds this cell's CI), so all three benchmarks feed the gate the same
            #     KIND of number.  offset_2se is still reported, as context, below.
            seed_sd = THUNDER_SEED_SD.get(a, {}).get(task)
            entry["resolvability_floor_offset_2se"] = floor
            entry["seed_sd_of_task_mean"] = seed_sd
            g = gate_denominator("THUNDER/%s" % task, a, waiv_gain, seed_sd)
            if g is not None:
                entry.update(g)
                tasks_out[task] = entry
                continue
            if base is None:
                entry.update({"pct": None, "ci": None, "n": 0, "status": "PARTIAL",
                              "reason": f"our BASE covers only {base_cov}/{len(PAPER_CLS)}"})
                tasks_out[task] = entry
                continue
            full = [x for x in by_bb[a] if x["thunder"][task]["mean"] is not None]
            covs = [x["thunder"][task]["coverage"] for x in by_bb[a]]
            n = len(full)
            if n == 0:
                cov_s = ",".join(str(c) for c in covs) or "-"
                entry.update({"pct": None, "ci": None, "n": 0, "status": "PARTIAL",
                              "reason": f"coverage {cov_s}/{len(PAPER_CLS)} per seed; "
                                        f"12ds floor invalid below 12/12"})
                tasks_out[task] = entry
                continue
            pcts_unc = [pct_of_waiv_uncapped(x["thunder"][task]["mean"], base, base + waiv_gain)
                        for x in full]
            mean_pct_unc = sum(pcts_unc) / n
            # F4 fix (2026-08-26).  This used to be abs(floor/waiv_gain)*100/sqrt(n),
            # where `floor` is offset_2se = |mean(d)| + 2*SD(d)/sqrt(12), with the SD
            # taken OVER THE 12 DATASETS.  That is the resolvability floor -- correct
            # for the INDETERMINATE gate just above, which asks whether Waiv's own gain
            # is even bigger than seed noise -- but it is NOT a 95% half-width on our
            # 12-dataset task mean, and dividing an already-sqrt(12)-shrunk
            # dataset-level SD again by sqrt(n_runs) compounds the error.  The correct
            # one-run 95% half-width is 2 * seed_SD_12ds: the SD of the 12-dataset task
            # mean itself across the 5 training seeds.  Averaging n independent runs
            # shrinks that by sqrt(n).
            # F-D: the SAME max(empirical, floor) construction RI and HEST use.  The
            # floor is the measured across-seed SD of the 12-dataset task mean, expressed
            # in pct points; the empirical term is the spread of our own n seeds.  This
            # cell used to be floor-only and never consulted the observed spread.
            floor_sd_pct = (seed_sd / abs(waiv_gain) * 100.0) if seed_sd is not None else None
            ci, emp_ci, floor_ci, ci_src = _ec.ci95(pcts_unc, floor_sd_pct)
            entry["seed_sd_pct_points"] = floor_sd_pct
            entry["empirical_ci"] = emp_ci
            entry["floor_ci"] = floor_ci
            entry["ci_source"] = ci_src + (
                " [floor = 2 * seed_SD_12ds(%.6f) / |waiv_gain|(%.4f) * 100 / sqrt(%d)]"
                % (seed_sd, abs(waiv_gain), n) if seed_sd is not None
                else " [no measured seed SD for this (backbone, task)]")
            entry.update(grade(mean_pct_unc, ci, n))
            if entry["status"] == "UNDERPOWERED":
                tasks_out[task] = entry
                continue
            entry.update({
                "n": n,
                "raw_mean": sum(x["thunder"][task]["mean"] for x in full) / n,
                "per_seed_pct": pcts_unc, "per_seed_pct_uncapped": pcts_unc,
                "per_seed_pct_capped": [cap100(q) for q in pcts_unc],
                "coverage": f"{len(PAPER_CLS)}/{len(PAPER_CLS)}",
            })
            tasks_out[task] = entry

        usable = [t for t in CLS_TASKS if tasks_out[t]["status"] not in UNGRADED or
                  tasks_out[t]["status"] == "NOT RESOLVED"]
        graded = [t for t in CLS_TASKS if tasks_out[t].get("pct") is not None]
        if not graded:
            reasons = sorted({tasks_out[t]["status"] for t in CLS_TASKS})
            cell = {"pct": None, "pct_capped": None, "pct_uncapped": None, "ci": None,
                    "lower_uncapped": None, "upper_uncapped": None, "was_capped": False,
                    "n": 0, "tasks": tasks_out,
                    "status": "PARTIAL" if "PARTIAL" in reasons else "INDETERMINATE",
                    "reason": "; ".join(f"{t}: {tasks_out[t]['status']}" for t in CLS_TASKS)}
        else:
            mean_pct_unc = sum(tasks_out[t]["pct_uncapped"] for t in graded) / len(graded)
            # F3 fix (2026-08-26): the three task CIs are NOT independent.  knn,
            # linear_probing and simple_shot are three readouts of the SAME per-seed
            # checkpoints over the SAME 12 datasets, so a seed that shifts one shifts
            # all three together.  Combining them in quadrature (the old
            # sqrt(sum(ci^2))/len) understated the aggregate half-width by up to
            # sqrt(3) ~ 1.73x.  Under perfect correlation the half-width of the mean is
            # the mean of the half-widths.
            ci = sum(tasks_out[t]["ci"] for t in graded) / len(graded)
            # An aggregate built from a subset of tasks is still a partial instrument:
            # flag it so it is never mistaken for the full 3-task THUNDER read.
            agg_n = max(tasks_out[t]["n"] for t in graded)
            cell = grade(mean_pct_unc, ci, agg_n)
            cell["ci_source"] = ("task CIs combined as the MEAN (perfect correlation: "
                                 "same checkpoints, same 12 datasets), not in quadrature")
            cell.update({"n": agg_n,
                         "tasks": tasks_out,
                         "tasks_graded": graded,
                         "tasks_excluded": {t: tasks_out[t]["status"]
                                            for t in CLS_TASKS if t not in graded}})
        report["cells"].setdefault(a, {})["THUNDER"] = cell
        _ = usable

    # ---- aggregate ----------------------------------------------------------
    benches = ("RI", "HEST", "THUNDER")
    bench_avg = {}
    for b in benches:
        vals = [report["cells"][a][b]["pct"] for a in ARMS
                if report["cells"].get(a, {}).get(b, {}).get("pct") is not None]
        # F1 fix (2026-08-26).  `overall_average` was the unweighted mean of the three
        # benchmark means regardless of how many backbones each rested on -- so a
        # THUNDER mean built from ONE backbone carried the same 1/3 weight as an RI mean
        # built from all three.  A benchmark mean over a subset of backbones is not
        # "the benchmark"; it is a different quantity sharing its name.  It is still
        # reported with its coverage, but it no longer enters the overall average, and
        # if ANY benchmark is short of 3 backbones the overall average is UNDEFINED
        # rather than quietly rebuilt from whatever is left.
        bench_avg[b] = {"mean": (sum(vals) / len(vals)) if vals else None,
                        "n_backbones": len(vals),
                        "coverage_ok": len(vals) == len(ARMS),
                        "eligible_for_overall": len(vals) == len(ARMS)}
    short = [b for b in benches if not bench_avg[b]["eligible_for_overall"]]
    if short:
        overall = None
        overall_note = ("UNDEFINED: %s rest(s) on fewer than %d backbones (%s); the "
                        "overall average is defined only when all three benchmark means "
                        "cover all three backbones"
                        % (", ".join(short), len(ARMS),
                           ", ".join("%s=%d/3" % (b, bench_avg[b]["n_backbones"])
                                     for b in short)))
    else:
        overall = sum(bench_avg[b]["mean"] for b in benches) / len(benches)
        overall_note = "mean of three full-coverage benchmark means"
    # Transparency only -- explicitly NOT the criterion.
    _partial = [bench_avg[b]["mean"] for b in benches if bench_avg[b]["mean"] is not None]
    overall_partial = (sum(_partial) / len(_partial)) if _partial else None

    graded_cells = [(a, b, report["cells"][a][b]) for a in ARMS for b in benches
                    if report["cells"].get(a, {}).get(b, {}).get("pct") is not None]
    ungraded_cells = [(a, b, report["cells"][a][b]) for a in ARMS for b in benches
                      if report["cells"].get(a, {}).get(b, {}).get("pct") is None]
    unresolved = [(a, b, c) for a, b, c in graded_cells if c["status"] == "NOT RESOLVED"]
    failed = [(a, b, c) for a, b, c in graded_cells if c["status"] == "FAIL"]
    worst = min(graded_cells, key=lambda t: t[2]["pct"]) if graded_cells else None

    if ungraded_cells or unresolved:
        verdict = "INDETERMINATE"
        bits = []
        if ungraded_cells:
            bits.append("no gradeable number for " +
                        ", ".join(f"{a}/{b} ({c['status']})" for a, b, c in ungraded_cells))
        if unresolved:
            bits.append("error bar straddles the 70 bar for " +
                        ", ".join(f"{a}/{b} ({c['pct']:.1f}+/-{c['ci']:.1f})"
                                  for a, b, c in unresolved))
        reason = "; ".join(bits)
    elif failed:
        verdict = "FAIL"
        reason = "below the 70 bar: " + ", ".join(
            f"{a}/{b} = {c['pct']:.1f}+/-{c['ci']:.1f}" for a, b, c in failed)
    elif overall is not None and overall <= OVERALL_BAR:
        verdict = "FAIL"
        reason = f"every cell clears 70 but the overall average {overall:.1f} does not exceed {OVERALL_BAR}"
    elif overall is None:
        # F1: with an undefined overall average the 80 bar cannot be tested, so PASS is
        # not available.
        verdict = "INDETERMINATE"
        reason = f"every gradeable cell clears 70 but the overall average is {overall_note}"
    else:
        verdict = "PASS"
        reason = (f"worst cell {worst[0]}/{worst[1]} = {worst[2]['pct']:.1f} "
                  f"(CI lower bound {worst[2]['lower_uncapped']:.1f}) >= {PASS_BAR}; "
                  f"overall average {overall:.1f} > {OVERALL_BAR}")

    # --- floor-quality disclosure (see docs/FORMULA_UNIFICATION_2026-08-26.md, F-A) ---
    weak = []
    for a in ARMS:
        c = report["cells"].get(a, {}).get("HEST") or {}
        df = c.get("seed_sd_df")
        if df is not None and df <= 1:
            st = c.get("selected_step")
            fams = ((_HEST_SD_BLOB["pooled_seed_sd"].get(a, {}) or {}).get(str(st), {}) or {}).get("families") or []
            runs_backing = sorted(r for f in fams for r in f.get("runs", []))
            graded_runs = sorted(x["run"] for x in by_bb[a])
            same = set(runs_backing) == set(graded_runs)
            weak.append({
                "backbone": a, "benchmark": "HEST", "df": df, "step": st,
                "floor_runs": runs_backing,
                "floor_is_the_graded_runs": same,
                "note": ("the ONLY family supporting the floor at step %s IS the family "
                         "being graded (%d runs) -- the 'independent instrument floor' and "
                         "the empirical spread are the same two numbers"
                         % (st, len(runs_backing))) if same else
                        ("floor at step %s rests on a single n=2 family" % st),
            })
    report["weak_floors"] = weak

    report["benchmark_averages"] = bench_avg
    report["overall_average"] = overall
    report["overall_average_note"] = overall_note
    report["overall_average_partial_NOT_THE_CRITERION"] = overall_partial
    report["worst_cell"] = ({"backbone": worst[0], "benchmark": worst[1],
                             "pct": worst[2]["pct"],
                             "pct_uncapped": worst[2]["pct_uncapped"],
                             "lower_uncapped": worst[2]["lower_uncapped"],
                             "ci": worst[2]["ci"],
                             "status": worst[2]["status"]} if worst else None)
    report["ungraded_cells"] = [{"backbone": a, "benchmark": b, "status": c["status"],
                                 "reason": c.get("reason")} for a, b, c in ungraded_cells]
    report["verdict"] = verdict
    report["verdict_reason"] = reason
    return report


def print_report(rep: dict) -> None:
    W = 78
    print("=" * W)
    print("FINAL RECIPE VERDICT -- mask_same_core / cls-bias 3.0 / lr1e-4 / T0.07 / wd0.05")
    print("=" * W)
    print(f"Stopping rule : {rep['stopping_rule']}")
    print(f"Criterion     : pct_of_waiv >= {PASS_BAR:.0f} on EVERY cell (worst cell, not a mean),")
    print(f"                AND mean of the three benchmark means > {OVERALL_BAR:.0f}.")
    print("                ONE quantity is graded and averaged: the UNCAPPED pct_of_waiv.")
    print("                The 100 cap is presentation only and enters no arithmetic (F-C).")
    print("                Scorecard cells read:  uncapped (capped*) +/-CI n=  status")
    print()

    print("-- CHECKPOINT SELECTION (rule applied per run) " + "-" * 31)
    print(f"{'run':<20} {'seed':>4} {'step':>6} {'mean CI':>8}   curve (step:meanCI)")
    for r in rep["runs"]:
        curve = " ".join(f"{t['step']}:{t['mean_ci']:.2f}" if t["mean_ci"] is not None
                         else f"{t['step']}:--" for t in r["ci_trace"])
        step = r["selected_step"] if r["selected_step"] is not None else "none"
        mci = f"{r['mean_confounder_insensitivity']:.3f}" if r.get("mean_confounder_insensitivity") else "  --"
        print(f"{r['backbone']:<20} {r['seed']:>4} {str(step):>6} {mci:>8}   {curve}")
    print()

    not_reportable = rep.get("arms_not_reportable") or {}
    if not_reportable:
        print("-- ARMS IN THE ROSTER BUT NOT GRADEABLE " + "-" * 38)
        for a, info in sorted(not_reportable.items()):
            runs_found = info.get("runs_found") or []
            print(f"  {a:<10} runs found: {len(runs_found)}")
            for why in info["missing_inputs"]:
                print(f"             missing: {why}")
        print("   These arms are absent from every table below because a denominator they")
        print("   need has not been measured yet -- not because they scored badly.")
        print()

    print("-- SCORECARD (pct_of_waiv, +/-95% CI) " + "-" * 40)
    print(f"{'backbone':<10} {'RI':<42} {'HEST':<42} THUNDER")
    for a in ARMS:
        cells = rep["cells"].get(a, {})
        row = f"{a:<10} "
        for b in ("RI", "HEST", "THUNDER"):
            c = cells.get(b, {"pct": None, "status": "NO_DATA"})
            row += f"{fmt_cell(c):<42}"
        print(row.rstrip())
    print()

    print("-- THUNDER task detail " + "-" * 55)
    for a in ARMS:
        c = rep["cells"].get(a, {}).get("THUNDER", {})
        for t in CLS_TASKS:
            e = (c.get("tasks") or {}).get(t)
            if not e:
                continue
            head = f"  {a:<9} {t:<16}"
            if e.get("pct") is not None:
                star = "*" if e.get("was_capped") else " "
                print(f"{head} {e['pct']:6.1f} ({e['pct_capped']:6.1f}{star}) "
                      f"+/-{e['ci']:5.1f}  lo={e['lower_uncapped']:6.1f}  n={e['n']}  "
                      f"cov={e.get('coverage')}  {e['status']}")
            else:
                print(f"{head} {'--':>6}            n={e.get('n', 0)}  {e['status']}"
                      f"  ({e.get('reason', '')})")
    print()

    print("-- AGGREGATE " + "-" * 65)
    for b in ("RI", "HEST", "THUNDER"):
        m = rep["benchmark_averages"][b]
        v = f"{m['mean']:.1f}" if m["mean"] is not None else "n/a"
        print(f"  {b:<8} average over backbones: {v:>6}   (from {m['n_backbones']}/3 backbones)")
    ov = rep["overall_average"]
    print(f"  OVERALL  average of the three : {f'{ov:.1f}' if ov is not None else 'UNDEF':>6}"
          f"   (bar: > {OVERALL_BAR:.0f})")
    if ov is None:
        print(f"           {rep['overall_average_note']}")
        opa = rep.get("overall_average_partial_NOT_THE_CRITERION")
        if opa is not None:
            print(f"           (unweighted mean of whatever means exist = {opa:.1f} -- "
                  f"NOT the criterion, do not quote)")
    w = rep["worst_cell"]
    if w:
        print(f"  WORST cell: {w['backbone']}/{w['benchmark']} = {w['pct']:.1f} "
              f"+/-{w['ci']:.1f} (lower {w['lower_uncapped']:.1f})  [{w['status']}]"
              f"   (bar: >= {PASS_BAR:.0f})")
    weak = rep.get("weak_floors") or []
    if weak:
        print()
        print("-- FLOOR QUALITY WARNING (disclosure, NOT a correction) " + "-" * 22)
        for w in weak:
            print(f"   {w['backbone']}/{w['benchmark']} floor df={w['df']}: {w['note']}")
        print("   A df=1 floor is ONE draw, not a measurement of the instrument.  Where the")
        print("   only recipe family supporting the floor at that step is the family being")
        print("   graded, max(empirical, floor) degenerates to the empirical spread with one")
        print("   degree of freedom and the printed CI understates the true uncertainty.")
        print("   This is REPORTED, not patched: patching it would require either a new")
        print("   tuned constant or a change of estimator, both of which are out of scope.")

    sup = rep.get("hest_supplementary") or {}
    if sup:
        print()
        print("-- SUPPLEMENTARY HEST (assumed step, NOT SCORED) " + "-" * 30)
        print(f"   assumed step for runs with no ri_curve.json: {rep.get('hest_assume_step')}")
        for a, c in sup.items():
            print(f"   {a:<10} {fmt_cell(c)}  step_source={c.get('step_source')}")
            for r_ in c.get("assumed_step_runs", []):
                print(f"        assumed-step run: {r_}")
        print("   These cells are reported for completeness only.  The verdict above is")
        print("   scored on the primary rule-selected HEST cells.")
    if rep["ungraded_cells"]:
        print("  UNGRADED cells (excluded from every average above):")
        for u in rep["ungraded_cells"]:
            print(f"    {u['backbone']}/{u['benchmark']}: {u['status']}"
                  f"{' -- ' + u['reason'] if u.get('reason') else ''}")
    print()
    print("=" * W)
    print(f"FINAL VERDICT: {rep['verdict']}")
    print(f"REASON: {rep['verdict_reason']}")
    print("=" * W)
    print()
    print(f"Waiv THUNDER denominators: {rep['sources']['waiv_thunder']}")
    print(f"THUNDER seed floors      : {rep['sources']['thunder_floors']}")
    print(f"HEST seed SD (raw, disk) : {rep['sources']['hest_seed_sd']}")
    print(f"RI base (disk)           : phikon/midnight/virchow2 = " +
          ", ".join(f"{RI_BASE[a]:.6f}" for a in ARMS))
    print(f"                           {RI_BASE_SOURCE['phikon']}")
    print(f"Denominator gate         : {rep['sources']['denominator_gate']}")
    print(f"CI construction          : {rep['sources']['ci_construction']}")
    print()
    print("-- RETIRED LITERALS vs THE VALUES NOW READ FROM DISK " + "-" * 25)
    for k, v in rep["retired_literal_agreement"].items():
        flag = "ok" if v["agrees_to_4dp"] else "*** DISAGREES ***"
        print(f"   {k:<20} literal {v['retired_literal']:<10} disk {v['from_disk']:.7f} "
              f"delta {v['delta']:+.7f}  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true",
                    help="also write docs/final_recipe_verdict.json")
    ap.add_argument("--json-out", default=str(REPO / "docs" / "final_recipe_verdict.json"))
    ap.add_argument("--hest-assume-step", type=int, default=None, metavar="STEP",
                    help="F5 escape hatch: admit runs that have HEST summaries but NO "
                         "ri_curve.json into a clearly-marked SUPPLEMENTARY HEST cell at "
                         "this explicit step.  They never enter the RI or THUNDER cells, "
                         "and never the primary rule-selected HEST cell the verdict is "
                         "scored on.  Off by default; remove the flag to revert.")
    args = ap.parse_args()

    rep = build_report(hest_assume_step=args.hest_assume_step)
    print_report(rep)
    if args.json:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
