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
from collections import defaultdict
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

# ---------------------------------------------------------------------------
# CLASSIFICATION ROSTER (2026-08-26).  Owned by collect_final5; selected here.
# ---------------------------------------------------------------------------
# DEFAULT IS THE 16-SET WAIV ROSTER.  Waiv's published THUNDER classification numbers
# (arXiv:2607.22861 Table 2) are means over 16 datasets -- the THUNDER paper's 12 plus
# the 4 SPIDER organ subsets, which postdate that paper.  We averaged over 12.  That
# mismatch put our base task means 0.86-3.72 points BELOW Waiv's published bases on all
# 9 (backbone, task) cells, and since pct_of_waiv = (ours - our_base) / waiv_gain, a base
# that is not the same quantity as theirs is not a comparison at all.  On the 16-set
# roster the same 9 gaps collapse to -0.61..+0.43 -- i.e. the base gap IS the roster.
#
# `--cls-roster 12` reproduces every pre-2026-08-26 number for before/after inspection.
#
# CAVEAT THAT MUST TRAVEL WITH THE 16-SET NUMBERS: THUNDER_FLOOR and THUNDER_SEED_SD
# below were measured on the 12-DATASET task mean (n=5 seeds).  A 16-dataset mean
# averages over MORE datasets and is therefore LESS noisy, so reusing the 12ds SD as the
# error floor OVER-states the noise.  That is the safe direction for an error bar, and it
# is used rather than rescaled because no 5-seed SPIDER cohort exists to measure with.
CLS_ROSTERS = {
    "12": _c5.PAPER_CLS_THUNDER12,   # THUNDER paper panel; what the seed floors were measured on
    "16": _c5.PAPER_CLS_WAIV16,      # Waiv's Table-2 panel = 12 + 4 SPIDER
}
CLS_ROSTER_DEFAULT = "16"
PAPER_CLS = CLS_ROSTERS[CLS_ROSTER_DEFAULT]
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

# ---------------------------------------------------------------------------
# THE RUN FAMILY.  Changed 2026-08-31 from the retired `genMASK-c3s-*` pilot (2 seeds,
# 3 ungated backbones, CKPT_EVERY=125) to the FINALISED `genMASK-c50-*` sweep: five
# backbones, CKPT_EVERY=50, ms500, warmup 200, lr 1e-4, rank 32, projdim 512, t900,
# WAIV_BCLS=3.0 / WAIV_BMEAN=-inf, pin `falseneg-gated`  (docs/RUNBOOK.md section 1.2).
# The c3s family is still selectable with --run-glob for before/after inspection.
# ---------------------------------------------------------------------------
RUN_GLOB = "genMASK-c50-*"
CI_TARGET = 0.75
# F13 fix: the stopping rule's mean is only comparable across checkpoints when it is
# taken over the SAME dataset panel every time.
N_CI_DATASETS = 3   # camelyon, tolkach_esca, tcga          # stopping rule: first ckpt with mean confounder_insensitivity >= this

# ---------------------------------------------------------------------------
# CHECKPOINT SELECTION -- the 1-SE RULE  (2026-08-31, supersedes CI >= 0.75)
# ---------------------------------------------------------------------------
# The rule, online, one run, no free parameters:
#
#   walk the checkpoint grid in step order, scoring each checkpoint by PathoROB's
#   published bounded `avg_robustness_index` (NOT `confounder_insensitivity`, which is
#   an unbounded odds with a per-dataset chance level -- see the audit in
#   docs/CAVEATS.md).  Track B = best RI seen so far.
#     1. STOP at the first checkpoint t >= 2 for which  R_t - B <= SE.
#     2. RETURN the EARLIEST checkpoint seen so far whose RI >= B - SE.
#   Step 2 is the rule.  Returning the stalling checkpoint instead overshoots by
#   exactly one checkpoint on every run it was measured on.
#
# SE IS AN INPUT, NOT A CONSTANT.  It is the within-run standard error of the
# 3-dataset mean RI at that checkpoint, and it must be MEASURED, not chosen: the whole
# point of replacing CI>=0.75 was to delete a fitted constant, and swapping in a fitted
# SE would put it straight back.  RI_SE_KEYS below are the per-dataset bootstrap fields
# PathoROB emits under `--compute_bootstrapped_robustness_index`
# (see patches/pathorob-enable-bootstrap-ri.patch, which repairs two crashes on that
# path).  When they are present in a run's ri_curve.json the rule is fully automatic.
#
# THEY ARE NOT PRESENT ANYWHERE ON DISK TODAY.  The bootstrap flag
# (src/waivphaet/eval/pathorob_adapter.py:161 `bootstrap: bool = False`) has never been
# switched on by any caller, and eval_checkpoints.RESULT_KEYS would drop the fields even
# if it had been.  So `--ri-se` exists as an EXPLICIT, operator-supplied override: a
# number the reader can see, argue with, and vary, rather than one buried in the code.
# Without it, or without the fields on disk, NO STEP IS SELECTED and the affected cells
# are reported NOT REPORTABLE with that exact reason.  `--ri-se-sweep` prints the picks
# as a function of SE so the sensitivity is visible instead of asserted.
RI_SE_KEYS = ("robustness_index-std", "robustness_index_se", "ri_se")
RI_SE: float | None = None          # set only by --ri-se; never defaulted to a guess

# The SE the FINAL SCOREBOARD runs the rule with (scripts/final_scoreboard.py, which must
# work with no arguments).  It is still an operator input -- this constant is where the
# assertion is made once, with its source, instead of being retyped on a command line:
#
#   0.0070 = the MEASURED between-seed floor of the 3-dataset avg robustness index,
#   max |ctrl - ctrlseed| over checkpoints, n=2 (docs/RESULTS.md section 12.3, quoted in
#   docs/CAVEATS.md: "the avg-RI floor (0.0070)").  It is the noise of exactly the
#   quantity this rule scores -- the same mean over the same three PathoROB datasets --
#   which is why it is used rather than scoreboard.NOISE_SD's per-(backbone, step) RI SDs
#   (0.0020-0.0048, measured only at steps 250/500, i.e. off this grid entirely).
#
# It is NOT the within-run bootstrap SE the rule was designed around: PathoROB's
# `robustness_index-std` is absent from every ri_curve.json on disk (see RI_SE_KEYS
# above), so no per-checkpoint SE exists to read.  A between-seed |difference| also runs
# ~1.1-1.4x a one-run SD, so this SE is if anything WIDE, and a wider SE selects an
# EARLIER checkpoint -- the direction that under-claims training, not over-claims it.
# report["ri_se_sweep"] prints the pick as a function of this number so the sensitivity
# is visible rather than asserted; at the time of writing the picks are unchanged over
# SE 0.006-0.0085 on four of five backbones.
RI_SE_SCOREBOARD_DEFAULT = 0.0070
RI_SE_SCOREBOARD_SOURCE = (
    "docs/RESULTS.md section 12.3 measured between-seed avg-RI floor 0.0070 "
    "(max |ctrl - ctrlseed| over checkpoints, n=2); OPERATOR INPUT, not a "
    "per-checkpoint bootstrap SE -- PathoROB's bootstrap fields are on no curve on disk")
# The rule cannot declare a plateau off a single checkpoint: at t=1 the "improvement
# over the best so far" is identically zero and the stop condition fires trivially.
MIN_CHECKPOINTS_FOR_STOP = 2
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
    # 2026-08-31: gated backbones.  Source: the "NEW SECTION ... gated backbones"
    # part of docs/thunder_seed_floor_12ds.md, generated by
    # scripts/thunder_seed_floor_gated.py (raw: docs/thunder_seed_floor_gated.json).
    # SAME estimator (offset_2se over the same 12 PAPER_CLS datasets, f1, frozen) --
    # the gated script imports pair_stats/read from thunder_seed_floor_12ds so there is
    # one implementation.  TWO differences the reader must carry:
    #   (a) n_seeds = 2 (one c3s s0/s1 pair at step 125), not 5.  offset_2se still has
    #       df = 11 over datasets, but the across-seed term rests on a single pair.
    #   (b) measured in the pathfm-full-evals corpus (corrected Resize(256,bicubic)
    #       transform), not the old Resize(224,bilinear) corpus the three floors above
    #       come from.  Checked, not assumed: the same c3s pairs for phikon/midnight/
    #       virchow2 give a new/old floor ratio of mean 1.11, range 0.76-1.40, with no
    #       consistent sign -- the transform moves absolute scores, not seed dispersion.
    # Calibration in that section shows a single pair usually lands within +/-25% of the
    # 5-seed value but was off by 2.2x in 1 of 9 cells.  Do not defend a gated verdict
    # whose margin is inside 2x of the floor.
    "hoptimus": {"knn": 0.0170, "linear_probing": 0.0226, "simple_shot": 0.0120},
    "uni2":     {"knn": 0.0107, "linear_probing": 0.0059, "simple_shot": 0.0031},
}
THUNDER_FLOOR_SOURCE = (
    "docs/thunder_seed_floor_12ds.md, offset-2SE, 12/12 coverage; "
    "phikon/midnight/virchow2 n=5 training seeds (old corpus), "
    "hoptimus/uni2 n=2 (single c3s seed pair, pathfm-full-evals corpus)"
)

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

# F-P (2026-08-26): the loader and its literal fallback used to live here.  They now
# live in eval_common (ONE owner, shared with scoreboard), and the fallback literals are
# gone -- a missing measurement must be missing, not substituted.
THUNDER_SEED_SD, THUNDER_SEED_SD_PATH = _ec.load_thunder_seed_sd()

# --- F2 fix (2026-08-26): minimum-n gate -----------------------------------------
# Every CI path in this script will happily manufacture an error bar at n=1:
# RI falls back to floor_ci, HEST computes 2*sd/sqrt(1), THUNDER computes
# floor/gain*100/sqrt(1).  None of those is an across-seed measurement, so an n=1
# cell was being graded PASS/FAIL on a number no data supports.  Cells below this
# many seeds now return UNDERPOWERED and are never scored.
MIN_N_FOR_VERDICT = 2

# F9 fix (2026-08-26): single source of truth for the HEST pooling rule.
HEST_POOLING = {a: _c5.hest_pooling(a) for a in ALL_ARMS}


# ---------------------------------------------------------------------------
# REPORTABILITY IS PER (ARM, BENCHMARK), NOT PER ARM  (2026-08-31)
# ---------------------------------------------------------------------------
# This used to be one all-or-nothing list per arm: an arm missing ANY denominator was
# struck from EVERY table.  The only input the two gated backbones lack is the THUNDER
# seed floor -- their RI base, HEST base and both Waiv targets have been on disk since
# the gated-backbone eval landed -- so the old rule silently deleted ten perfectly
# measurable RI/HEST cells because a third benchmark was unmeasured.  A missing
# denominator now removes exactly the cells it actually blocks.
def _missing_denominators_for(arm: str, bench: str) -> list[str]:
    """Which grading inputs this (arm, benchmark) lacks.  Empty list = gradeable."""
    missing = []
    if bench == "RI":
        if arm not in RI_BASE:
            missing.append("RI base (eval_common.RI_BASE_MODEL_DIRS)")
        if arm not in RI_WAIV:
            missing.append("Waiv RI target (pathorob_adapter.TARGETS)")
    elif bench == "HEST":
        if arm not in HEST_BASE:
            missing.append("HEST base (collect_final5.HEST_BASE_FILES)")
        if arm not in HEST_WAIV:
            missing.append("Waiv HEST target (eval_common.HEST_WAIV)")
    elif bench == "THUNDER":
        if arm not in THUNDER_FLOOR:
            missing.append("THUNDER seed floor (docs/thunder_seed_floor_12ds.md)")
        if arm not in WAIV_THUNDER:
            missing.append("Waiv THUNDER target (docs/waiv_published.json Table 2)")
    return missing


BENCHES = ("RI", "HEST", "THUNDER")
BENCH_ARMS = {b: tuple(a for a in ALL_ARMS if not _missing_denominators_for(a, b))
              for b in BENCHES}
THUNDER_ARMS = BENCH_ARMS["THUNDER"]
# `ARMS` keeps its old name and its old job -- the row order of every table -- but it is
# now the arms gradeable on AT LEAST ONE benchmark, i.e. all five.
ARMS = tuple(a for a in ALL_ARMS
             if any(not _missing_denominators_for(a, b) for b in BENCHES))
# Retained for backwards compatibility with scripts/final_scoreboard.py section 5, which
# lists "section 1 cells for <arm>: NOT REPORTABLE".  Its meaning is now the strict one:
# an arm here has NO gradeable benchmark at all.  Per-benchmark gaps live in
# report["cells_not_reportable"].
ARMS_NOT_REPORTABLE = {a: sorted({m for b in BENCHES
                                  for m in _missing_denominators_for(a, b)})
                       for a in ALL_ARMS if a not in ARMS}
CELLS_NOT_REPORTABLE = {"%s/%s" % (a, b): _missing_denominators_for(a, b)
                        for a in ALL_ARMS for b in BENCHES
                        if _missing_denominators_for(a, b)}

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


def discover_runs(run_glob: str | None = None) -> list[dict]:
    """Glob the recipe's runs and parse (backbone, seed, train job id) out of each name."""
    out = []
    for d in sorted((REPO / "runs").glob(run_glob or RUN_GLOB)):
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


def _ri_points(run_dir: Path) -> list[dict]:
    """Every checkpoint on one run's curve, step-ordered, with its RI and (if measured)
    the within-run SE of that RI.

    A point is USABLE only with all N_CI_DATASETS present: the reported RI is a mean over
    the three PathoROB datasets, and a mean over two of them is a different statistic
    that happens to share the name (the same F13 argument the CI rule already made).
    Short points stay in the trace, flagged, so the skip is auditable.
    """
    p = run_dir / "ri_curve.json"
    if not p.exists():
        return []
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return []
    rows = []
    for pt in blob.get("points", []):
        ds = pt.get("datasets") or {}
        per_ds = {k: v.get("robustness_index") for k, v in ds.items()
                  if isinstance(v, dict) and v.get("robustness_index") is not None}
        # Per-dataset bootstrap SDs, when PathoROB was run with bootstrapping enabled.
        # The SE of the 3-dataset MEAN is sqrt(sum sd_i^2)/n -- the datasets are disjoint
        # cohorts, so their bootstrap draws are independent.
        sds = []
        for v in ds.values():
            if not isinstance(v, dict):
                continue
            for k in RI_SE_KEYS:
                if v.get(k) is not None:
                    sds.append(float(v[k]))
                    break
        se = (math.sqrt(sum(s * s for s in sds)) / len(sds)) if len(sds) == len(per_ds) and sds else None
        rows.append({
            "step": pt.get("step"),
            "ri": pt.get("avg_robustness_index"),
            "n_datasets": len(per_ds),
            "eligible": len(per_ds) == N_CI_DATASETS and pt.get("avg_robustness_index") is not None,
            "measured_se": se,
            "skip_reason": (None if len(per_ds) == N_CI_DATASETS else
                            "only %d/%d RI datasets present" % (len(per_ds), N_CI_DATASETS)),
        })
    rows.sort(key=lambda r: (r["step"] is None, r["step"]))
    return rows


def select_step_1se(run_dir: Path, ri_se: float | None = None):
    """THE RULE.  Returns (selected_step, se_used, trace).

    Online one-standard-error rule (Breiman CART; Hastie & Tibshirani ESL), read as
    "the least-trained checkpoint within one SE of peak robustness":

        B = best RI so far.  STOP at the first checkpoint (index >=
        MIN_CHECKPOINTS_FOR_STOP) where R_t - B <= SE.  RETURN the EARLIEST checkpoint
        seen so far with RI >= B - SE.

    Three outcomes are distinguished and none of them is silently a step:

      * a step, with the SE that produced it;
      * (None, se, trace) with trace[-1]["verdict"] == "UNTERMINATED" -- the run is still
        improving by more than one SE at its last measured checkpoint, so the rule has
        not fired.  Selecting the last checkpoint here would be the stop-at-last rule
        wearing this rule's name;
      * (None, None, trace) -- no SE.  The rule is undefined, not zero.  See RI_SE above.
    """
    trace = _ri_points(run_dir)
    if ri_se is None:
        for r in trace:
            r["verdict"] = "NO_SE"
        return None, None, trace
    usable = [r for r in trace if r["eligible"]]
    best = None            # B
    best_step = None
    for i, r in enumerate(usable):
        se = r["measured_se"] if r["measured_se"] is not None else ri_se
        r["se_used"] = se
        if best is None:
            best, best_step = r["ri"], r["step"]
            r["delta_vs_best"] = 0.0
            r["verdict"] = "SEED"          # first point: no improvement to judge
            continue
        r["delta_vs_best"] = r["ri"] - best
        if i + 1 >= MIN_CHECKPOINTS_FOR_STOP and r["ri"] - best <= se:
            r["verdict"] = "STOP"
            if r["ri"] > best:
                best, best_step = r["ri"], r["step"]
            thresh = best - se
            for q in usable[:i + 1]:
                if q["ri"] >= thresh:
                    q["verdict"] = q.get("verdict", "") + "|RETURNED"
                    return q["step"], se, trace
            # Unreachable: best itself satisfies RI >= best - se.
            return best_step, se, trace
        r["verdict"] = "CONTINUE"
        if r["ri"] > best:
            best, best_step = r["ri"], r["step"]
    if usable:
        usable[-1]["verdict"] = "UNTERMINATED"
    return None, ri_se, trace


def select_step_ci075(run_dir: Path) -> tuple[int | None, float | None, list[dict]]:
    """RETIRED stopping rule, kept selectable with --rule ci075 for before/after work.

    Retired because `confounder_insensitivity` is an unbounded ODDS whose chance level
    differs per dataset (1.03 / 2.21 / 1.05), so its cross-dataset mean is a mixed-unit
    quantity, and 0.75 was a grid point fitted on a script that contained a
    never-fired-rule bug (see scripts/eval_stopping_rules.py's F-H note).

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


RULES = ("1se", "ci075")


def select_step(run_dir: Path, rule: str = "1se", ri_se: float | None = None):
    """Dispatch to the configured checkpoint-selection rule.

    Returns (selected_step, rule_statistic, trace).  `rule_statistic` is the SE that
    produced the pick under `1se`, and the mean confounder_insensitivity under `ci075`.
    """
    if rule == "ci075":
        return select_step_ci075(run_dir)
    if rule == "1se":
        return select_step_1se(run_dir, ri_se)
    raise ValueError("unknown rule %r (choose from %s)" % (rule, ", ".join(RULES)))


def ri_se_sweep(runs: list[dict], grid: list[float] | None = None) -> dict:
    """Which step the 1-SE rule picks, per run, as a function of the SE it is given.

    The rule's only input is the SE.  Printing the pick across a grid of SEs turns "we
    chose step 100" into a checkable statement about how wide the SE would have to be
    wrong for the pick to move, and it is the honest substitute for a single asserted
    number while the bootstrap SE is unmeasured.  Nothing here selects anything: it is a
    diagnostic table.
    """
    grid = grid or [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010,
                    0.015, 0.020]
    out: dict = {"grid": grid, "runs": {}, "consensus": {}}
    for r in runs:
        out["runs"][r["run"]] = {
            "backbone": r["backbone"], "seed": r["seed"],
            "picks": {("%g" % se): select_step_1se(r["run_dir"], se)[0] for se in grid},
        }
    for se in grid:
        k = "%g" % se
        per_bb: dict[str, list] = defaultdict(list)
        for name, v in out["runs"].items():
            per_bb[v["backbone"]].append(v["picks"][k])
        out["consensus"][k] = {bb: sorted({p for p in v if p is not None})
                               for bb, v in per_bb.items()}
    return out


def cell_steps(rows: list[dict]) -> tuple[list[int], str]:
    """The step(s) the rule selected for the runs that build ONE cell.

    The rule is applied PER RUN, so two seeds of the same backbone can plateau at
    different checkpoints (on the 50-step grid this actually happens: uni2 s0 stops at
    100 and s1 at 150 at the cited SE).  That is not an error and it is not a licence to
    mix benchmarks either -- every ROW is still one (run, step) pair, and each run
    contributes the SAME checkpoint to RI, HEST and THUNDER.  What it does mean is that
    the cell cannot be labelled with a single step, so both the set and a printable
    label are returned and the label carries every step that went in.
    """
    steps = sorted({x["selected_step"] for x in rows if x.get("selected_step")})
    return steps, ("/".join(str(s) for s in steps) if steps else "-")


def seed_sd_over_steps(per_step: dict, steps: list[int]):
    """The measured seed SD for a cell whose runs may sit at different steps.

    One SD per distinct step through the SHARED rule (eval_common.seed_sd_at_step), then
    the LARGEST of them -- the same conservative direction that rule already takes when a
    step has no measurement of its own.
    """
    if not steps:
        return None, "no selected step"
    got = [_ec.seed_sd_at_step(per_step, s) for s in steps]
    have = [(sd, note, s) for (sd, note), s in zip(got, steps) if sd is not None]
    if not have:
        return None, got[0][1]
    sd, note, s = max(have, key=lambda t: t[0])
    if len(steps) > 1:
        note = "max over the %d selected steps %s -> step %s: %s" % (
            len(steps), steps, s, note)
    return sd, note


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


def thunder_model_names(backbone: str, seed: int, jobid: str, step: int,
                        run: str | None = None) -> list[str]:
    """Every THUNDER model-dir spelling this (run, step) could have been written under.

    THE HARNESS HAS USED TWO CONVENTIONS and this function used to hardcode one of them.
    `scripts/run_thunder.sbatch` names a model `f5_<run>_s<step>`, which is what the c3s
    gated-backbone cells on disk are called
    (`f5_genMASK-c3s-...-hoptimus-s0-t900-394307_s0000125`), while the older three-backbone
    c3s cells were submitted under the short `f5_ci-<bb>-s<seed>-<jobid>_s<step>` alias to
    stay inside THUNDER's 64-character run-name limit.  Returning only the alias meant a
    run written under the long name looked like "evaluated and empty" instead of
    "evaluated"; returning only the long name would break every existing cell.  Both are
    tried, in order, and the caller records which one it found.
    """
    names = []
    if run:
        names.append(f"f5_{run}_s{step:07d}")
    names.append(f"f5_ci-{backbone}-s{seed}-{jobid}_s{step:07d}")
    return names


def thunder_model_name(backbone: str, seed: int, jobid: str, step: int,
                       run: str | None = None) -> str:
    """The spelling that actually exists on disk; the first candidate if none does."""
    res_root = THUNDER_ROOT / "outputs" / "res"
    cands = thunder_model_names(backbone, seed, jobid, step, run)
    for c in cands:
        for ds in PAPER_CLS:
            if (res_root / ds / c).is_dir():
                return c
    return cands[0]


def thunder_base_12ds(backbone: str) -> dict[str, tuple[float | None, int]]:
    """Our own BASE (unfinetuned) 12-dataset task means, with coverage.

    Read through collect_final5's base-dir mapping so base and FT go through the same
    path and the same pooling convention per backbone.
    """
    per_ds = _c5._thunder_base_per_ds(backbone, cls_datasets=PAPER_CLS)
    out = {}
    for task in CLS_TASKS:
        vals = [v for k, v in (per_ds.get(task) or {}).items() if k in PAPER_CLS and v is not None]
        out[task] = ((sum(vals) / len(vals)) if len(vals) == len(PAPER_CLS) else None, len(vals))
    return out


def thunder_base_gap() -> dict:
    """OUR base task mean MINUS Waiv's PUBLISHED base, on both rosters.

    This is the validity test for every THUNDER pct_of_waiv in this report.  pct_of_waiv
    subtracts OUR base from OUR score and divides by WAIV's gain; if our base is not
    measuring the same quantity as their base, the numerator carries a constant offset
    that has nothing to do with the recipe.  A roster that closes this gap is the only
    evidence that the two sides are like-for-like.

    Nothing here is hardcoded: our side is read from disk through the same
    _thunder_base_per_ds path the cells use, and Waiv's side comes from
    scoreboard.WAIV_THUNDER, which is loaded from docs/waiv_published.json.
    """
    out: dict = {"waiv_source": WAIV_THUNDER_SOURCE, "rosters": {}}
    for label, roster in CLS_ROSTERS.items():
        per_roster: dict = {}
        for a in THUNDER_ARMS:
            per_ds = _c5._thunder_base_per_ds(a, cls_datasets=roster)
            for task in CLS_TASKS:
                vals = [v for k, v in (per_ds.get(task) or {}).items()
                        if k in roster and v is not None]
                ours = (100.0 * sum(vals) / len(vals)) if len(vals) == len(roster) else None
                waiv = WAIV_THUNDER[a]["base"][task]
                per_roster["%s/%s" % (a, task)] = {
                    "our_base_pct": ours,
                    "coverage": "%d/%d" % (len(vals), len(roster)),
                    "waiv_published_base_pct": waiv,
                    "gap_pct_points": (None if ours is None else ours - waiv),
                }
        gaps = [v["gap_pct_points"] for v in per_roster.values()
                if v["gap_pct_points"] is not None]
        out["rosters"][label] = {
            "n_datasets": len(roster),
            "datasets": list(roster),
            "cells": per_roster,
            "n_cells_with_full_coverage": len(gaps),
            "max_abs_gap_pct_points": (max(abs(g) for g in gaps) if gaps else None),
            "mean_gap_pct_points": (sum(gaps) / len(gaps) if gaps else None),
        }
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


def percell_gate_diagnostic(bench: str, arm: str, gain: float | None,
                            seed_sd: float | None, sd_note: str = "") -> dict:
    """The old PER-CELL denominator gate, retained as a DIAGNOSTIC ONLY.

    F-P fix (2026-08-26).  Until now this gate VETOED a cell whenever one seed-SD
    exceeded 10 pct_of_waiv points, i.e. whenever 2*SD_waiv > 20% of Waiv's own gain.
    Two things were wrong with using it as a veto:

      * It is not the test its own docstring describes.  "The denominator is noise"
        means the denominator's SIGN is not determined (|gain| <= 2*SD); the 10-point
        bar instead asks whether Waiv's gain is known to better than +/-20% relative,
        which is a PRECISION question, not an is-it-real question.  The two differ by a
        factor of five, and every THUNDER cell sits between them: phikon/knn's gain is
        5.8 seed-SD -- unambiguously real -- yet the 10-point bar rejected it.
      * A precision shortfall does not need a veto, because it can be CARRIED.  The
        denominator's own uncertainty propagates into the interval on the ratio
        (eval_common.pool_cells uses the delta method on both terms).  Vetoing instead
        of propagating throws away a cell whose interval may still clear the bar --
        phikon/knn graded 116.6 [72.1, 161.1] before the veto was introduced, an
        interval that is wide precisely BECAUSE the denominator is imprecise, and that
        still clears 70.

    So the number below is still computed and still printed -- an honest reader wants to
    know that Waiv's phikon/knn gain is known only to +/-35% -- but it no longer decides
    whether a cell is graded.  Grading is decided by eval_common.pooled_denominator_
    unresolvable applied to the POOLED denominator.
    """
    unres, sd_pct, why = _ec.denominator_unresolvable(gain, seed_sd)
    return {
        "waiv_gain": gain,
        "seed_sd": seed_sd,
        "seed_sd_pct_points": sd_pct,
        "waiv_gain_over_1sd": (abs(gain) / seed_sd) if (gain and seed_sd) else None,
        "percell_precision_flag": unres,
        "percell_precision_note": why,
        "percell_gate": "DIAGNOSTIC ONLY -- one seed-SD > %.0f pct_of_waiv points; "
                        "does NOT withhold the cell (see percell_gate_diagnostic)"
                        % _ec.UNRESOLVABLE_SD_PCT_LIMIT,
        "sd_note": sd_note,
    }


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
def build_report(hest_assume_step: int | None = None,
                 cls_roster: str = CLS_ROSTER_DEFAULT,
                 rule: str = "1se",
                 ri_se: float | None = None,
                 run_glob: str | None = None) -> dict:
    """Build the verdict report.

    `rule` / `ri_se` select the checkpoint rule (see RI_SE above).  `run_glob` selects
    the run family (default RUN_GLOB = the finalised c50 sweep).

    `cls_roster` selects the THUNDER classification panel: "16" (default, Waiv's
    Table-2 roster) or "12" (the THUNDER paper's, which is what the seed floors were
    measured on).  See CLS_ROSTERS above.

    `hest_assume_step` (F5 fix, 2026-08-26) is an explicit, opt-in, reversible escape
    hatch.  Runs that have finished training and have HEST summaries at every step but
    have NO ri_curve.json cannot be assigned a step by the stopping rule, and used to be
    dropped from ALL THREE benchmarks.  With this flag they additionally enter a clearly
    marked SUPPLEMENTARY HEST cell at the given assumed step.  They never enter the RI or
    THUNDER cells (both of which need the curve), and they never enter the primary
    rule-selected HEST cell that the verdict is scored on.
    """
    global PAPER_CLS, RUN_GLOB
    PAPER_CLS = CLS_ROSTERS[cls_roster]
    if run_glob:
        RUN_GLOB = run_glob
    runs = discover_runs()
    report: dict = {
        "run_family": {
            "glob": RUN_GLOB,
            "n_runs_discovered": len(runs),
            "runs": [r["run"] for r in runs],
        },
        "checkpoint_rule": {
            "rule": rule,
            "description": (
                "1-SE (online): B = best avg_robustness_index so far; STOP at the first "
                "checkpoint (index >= %d) with R_t - B <= SE; RETURN the EARLIEST "
                "checkpoint with RI >= B - SE." % MIN_CHECKPOINTS_FOR_STOP
                if rule == "1se" else
                "RETIRED: first checkpoint with mean confounder_insensitivity >= %s"
                % CI_TARGET),
            "metric": ("PathoROB avg_robustness_index (bounded, published)"
                       if rule == "1se" else "confounder_insensitivity (unbounded odds)"),
            "se_source": (
                "per-checkpoint bootstrap SE from ri_curve.json (%s) where present, "
                "otherwise the operator-supplied --ri-se" % "/".join(RI_SE_KEYS)),
            "ri_se_supplied": ri_se,
            "se_measured_on_disk": False,   # overwritten below if any point carries one
            "se_unmeasured_note": (
                "PathoROB's bootstrap fields are absent from every checkpoint in this "
                "corpus: src/waivphaet/eval/pathorob_adapter.py:161 defaults "
                "bootstrap=False and no caller sets it, and "
                "scripts/eval_checkpoints.RESULT_KEYS does not copy the fields into "
                "ri_curve.json.  Until one of those changes, --ri-se is the ONLY way to "
                "run this rule, and it is an operator input, not a measurement."),
        },
        "cells_not_reportable": CELLS_NOT_REPORTABLE,
        "thunder_cls_roster": {
            "selected": cls_roster,
            "n_datasets": len(PAPER_CLS),
            "datasets": list(PAPER_CLS),
            "why": ("Waiv average THUNDER classification over 16 datasets (the THUNDER "
                    "paper's 12 + the 4 SPIDER organ subsets, which postdate that paper). "
                    "Averaging over 12 while comparing to their 16-set figures is a roster "
                    "mismatch that biases every pct_of_waiv numerator."),
            "seed_floor_caveat": ("THUNDER_FLOOR / THUNDER_SEED_SD were measured on the "
                                  "12-dataset task mean (n=5 seeds).  A 16-dataset mean is "
                                  "LESS noisy, so reusing the 12ds SD OVER-states the error "
                                  "bar -- the safe direction.  No 5-seed SPIDER cohort "
                                  "exists to remeasure with, so it is reused, not rescaled."),
            "segmentation": ("still 0/4 -- no SPIDER segmentation task exists and the two "
                             "segpath cells are not run on this cohort.  'THUNDER' in this "
                             "report means CLASSIFICATION ONLY."),
        },
        "thunder_base_gap_validation": thunder_base_gap(),
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
        "stopping_rule": ("1-SE on avg_robustness_index (earliest checkpoint within "
                          "one SE of the best seen so far)" if rule == "1se" else
                          "RETIRED: first checkpoint with mean "
                          "confounder_insensitivity >= %s" % CI_TARGET),
        "sources": {
            "waiv_thunder": WAIV_THUNDER_SOURCE,
            "thunder_floors": THUNDER_FLOOR_SOURCE,
            "hest_seed_sd": HEST_SEED_SD_SOURCE,
            "ri_base": RI_BASE_SOURCE,
            "ri_waiv": RI_WAIV_SOURCE,
            "hest_waiv": HEST_WAIV_SOURCE,
            "aggregation": ("eval_common.pool_cells -- RATIO OF MEANS: aggregate the "
                            "numerator (our raw delta) and the denominator (Waiv's raw "
                            "gain) over the group, then divide ONCE.  Per-cell "
                            "percentages are never averaged.  Applied at both levels: "
                            "3 tasks -> backbone/THUNDER, 3 backbones -> benchmark."),
            "pooling_completeness": ("all-or-nothing: a pooled number requires EVERY "
                                     "cell of its group; a subset is withheld"),
            "denominator_gate": ("eval_common.pooled_denominator_unresolvable, applied "
                                 "to the POOLED denominator (not per cell): %s"
                                 % _ec.POOLED_DENOMINATOR_GATE),
            "denominator_gate_change_2026_08_26": (
                "F-P: the per-cell 'one seed-SD > %.0f pct_of_waiv points' veto is "
                "RETIRED as a gate and kept as a diagnostic. It was a PRECISION test "
                "(is Waiv's gain known to better than +/-20%% relative), not the "
                "is-the-denominator-real test its own docstring described, and it "
                "rejected cells whose denominators are unambiguously real -- "
                "phikon/knn's Waiv gain is 5.8 seed-SD. Denominator imprecision is now "
                "PROPAGATED into the interval (delta method on both terms) instead of "
                "vetoing the cell." % _ec.UNRESOLVABLE_SD_PCT_LIMIT),
            "ci_construction": ("per-cell: eval_common.ci95 -- max(empirical "
                                "2*SD/sqrt(n), measured seed floor 2*SD/sqrt(n)).  "
                                "Pooled: eval_common.pool_cells delta method, "
                                "SE_agg = sqrt(sum SE_cell^2)/k on the numerator and the "
                                "same on the denominator, carried onto the ratio."),
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
        step, stat, trace = select_step(r["run_dir"], rule=rule, ri_se=ri_se)
        if any(t.get("measured_se") is not None for t in trace):
            report["checkpoint_rule"]["se_measured_on_disk"] = True
        rec = {
            "run": r["run"], "backbone": r["backbone"], "seed": r["seed"],
            "train_jobid": r["jobid"],
            "selected_step": step,
            "rule": rule,
            "ci_trace": trace,
        }
        if rule == "1se":
            rec["se_used"] = stat
        else:
            rec["mean_confounder_insensitivity"] = stat
        if step is None:
            has_curve = (r["run_dir"] / "ri_curve.json").exists()
            if not has_curve:
                why = ("no ri_curve.json -- the checkpoint rule cannot be applied "
                       "(RI and THUNDER cells excluded)")
            elif rule == "1se" and ri_se is None:
                why = ("1-SE rule undefined: no per-checkpoint RI SE.  PathoROB's "
                       "bootstrap fields (%s) are absent from this curve and no --ri-se "
                       "was supplied." % "/".join(RI_SE_KEYS))
            elif rule == "1se":
                why = ("1-SE rule UNTERMINATED: RI is still improving by more than one "
                       "SE at the last measured checkpoint (%s), so the run has not "
                       "plateaued.  Returning the last checkpoint here would be the "
                       "stop-at-last rule under this rule's name."
                       % ([t["step"] for t in trace if t.get("eligible")][-1:] or "none"))
            else:
                why = f"no checkpoint on the curve reaches mean CI >= {CI_TARGET}"
            rec["note"] = why
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
        model = thunder_model_name(r["backbone"], r["seed"], r["jobid"], step, r["run"])
        rec["thunder_model"] = model
        rec["thunder_model_candidates"] = thunder_model_names(
            r["backbone"], r["seed"], r["jobid"], step, r["run"])
        rec["thunder_root"] = str(THUNDER_ROOT / "outputs" / "res")
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
        steps_a, step_label = cell_steps(by_bb[a])
        step_a = step_label
        # The measured RI seed SD, per (backbone, step), through the ONE step-selection
        # rule shared with HEST (eval_common.seed_sd_at_step): exact step if measured,
        # otherwise the largest SD measured for that backbone at any step -- an
        # over-estimate by construction, the safe direction for an error bar.
        ri_per_step = {st: v.get("ri") for st, v in (_sb.NOISE_SD.get(a, {}) or {}).items()}
        raw_sd, sd_note = seed_sd_over_steps(ri_per_step, steps_a)
        gain = RI_WAIV[a] - RI_BASE[a]

        # F-P: the per-cell gate is now a DIAGNOSTIC (see percell_gate_diagnostic).
        # Withholding is decided on the POOLED denominator, in the aggregate section.
        diag = percell_gate_diagnostic("RI", a, gain, raw_sd, sd_note)

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
            "selected_steps": steps_a,
            "steps_mixed": len(steps_a) > 1,
            "per_run_step": {x["run"]: x["selected_step"] for x in by_bb[a]},
            "per_seed_pct": pcts_unc, "per_seed_pct_uncapped": pcts_unc,
            "per_seed_pct_capped": [cap100(q) for q in pcts_unc],
            "seed_sd_raw": raw_sd, "seed_sd_pct_points": floor_sd_pct,
            "empirical_ci": emp_ci, "floor_ci": floor_ci,
            "ci_source": ci_src,
            # Raw ABSOLUTE quantities -- these, not the percentage, are what the pooled
            # (ratio-of-means) grading rule aggregates.
            "our_delta": sum(vals) / n - RI_BASE[a],
            "waiv_gain": gain,
            "se_our_delta": (raw_sd / math.sqrt(n)) if raw_sd is not None else None,
            "sd_waiv_gain": raw_sd,
            "percell_denominator_diagnostic": diag,
        })
        report["cells"].setdefault(a, {})["RI"] = cell

    # ---- HEST ---------------------------------------------------------------
    for a in ARMS:
        vals = [x["hest"] for x in by_bb[a] if x.get("hest") is not None]
        n = len(vals)
        if n == 0:
            report["cells"].setdefault(a, {})["HEST"] = {"pct": None, "ci": None, "n": 0, "status": "NO_DATA"}
            continue
        steps_a, step_label = cell_steps(by_bb[a])
        step_a = step_label
        # F-A: derived from disk, same estimator as scoreboard.NOISE_SD, same step rule
        # as RI above.
        raw_sd, sd_note = seed_sd_over_steps(HEST_SEED_SD.get(a, {}), steps_a)
        gain = HEST_WAIV[a] - HEST_BASE[a]

        diag = percell_gate_diagnostic("HEST", a, gain, raw_sd, sd_note)

        pcts_unc = [pct_of_waiv_uncapped(v, HEST_BASE[a], HEST_WAIV[a]) for v in vals]
        mean_pct_unc = sum(pcts_unc) / n
        # 2026-09-01: None-guard, matching the RI cell above.  HEST_SEED_SD is keyed by
        # backbone and the two gated backbones have no measured HEST seed SD at all, so
        # this divided None by a float and killed the whole report the first time the
        # five-backbone family was graded.  A missing floor must leave the cell with the
        # EMPIRICAL interval only (and ci95 already reports which term it used), never
        # crash and never silently substitute another backbone's floor.
        floor_sd_pct = (raw_sd / gain * 100.0) if raw_sd is not None else None
        ci, emp_ci, floor_ci, ci_src = _ec.ci95(pcts_unc, floor_sd_pct)
        cell = grade(mean_pct_unc, ci, n)
        cell.update({
            "n": n,
            "raw_mean": sum(vals) / n, "base": HEST_BASE[a], "waiv": HEST_WAIV[a],
            "selected_step": step_a,
            "selected_steps": steps_a,
            "steps_mixed": len(steps_a) > 1,
            "per_run_step": {x["run"]: x["selected_step"] for x in by_bb[a]},
            "per_seed_pct": pcts_unc, "per_seed_pct_uncapped": pcts_unc,
            "per_seed_pct_capped": [cap100(q) for q in pcts_unc],
            "pooling": HEST_POOLING[a],
            "step_source": "stopping rule",
            "seed_sd_raw": raw_sd, "seed_sd_pct_points": floor_sd_pct,
            "seed_sd_df": ((_HEST_SD_BLOB["pooled_seed_sd"].get(a, {}) or {})
                           .get(str(steps_a[0]) if len(steps_a) == 1 else "",
                                {}) or {}).get("df"),
            "empirical_ci": emp_ci, "floor_ci": floor_ci,
            "ci_source": ci_src + " [floor %s]" % sd_note,
            "our_delta": sum(vals) / n - HEST_BASE[a],
            "waiv_gain": gain,
            "se_our_delta": (raw_sd / math.sqrt(n)) if raw_sd is not None else None,
            "sd_waiv_gain": raw_sd,
            "percell_denominator_diagnostic": diag,
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
    # Only the arms with a measured THUNDER seed floor.  The other arms get an explicit
    # NOT REPORTABLE cell below so the gap is a printed row, not an absent one.
    for a in THUNDER_ARMS:
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
            # F-P: per-cell precision is a DIAGNOSTIC, not a veto.  virchow2's three
            # per-task Waiv gains are +0.0270 / -0.0030 / +0.0030 -- individually two are
            # below the seed floor and one is NEGATIVE, so no per-task ratio means
            # anything -- but their POOLED denominator is +0.0090 and is well
            # conditioned.  Withholding is decided on the pooled denominator below.
            entry["percell_denominator_diagnostic"] = percell_gate_diagnostic(
                "THUNDER/%s" % task, a, waiv_gain, seed_sd)
            entry["waiv_gain"] = waiv_gain
            entry["sd_waiv_gain"] = seed_sd
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
                # Name the DIRECTORIES that were looked for.  "coverage 0/16" alone reads
                # as a half-finished eval; when every seed is at 0 the actual fact is
                # usually that this run family was never submitted to THIS harness at
                # all, and the reader has to be able to tell those two apart without
                # going to the filesystem.
                looked = sorted({m for x in by_bb[a]
                                 for m in (x.get("thunder_model_candidates") or [])})
                where = ("; nothing on disk for this (run, step) under %s -- searched %s"
                         % (str(THUNDER_ROOT / "outputs" / "res"), ", ".join(looked))
                         ) if all(c == 0 for c in covs) and looked else ""
                entry.update({"pct": None, "ci": None, "n": 0, "status": "PARTIAL",
                              "reason": f"coverage {cov_s}/{len(PAPER_CLS)} per seed; "
                                        f"12ds floor invalid below 12/12" + where})
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
                "our_delta": sum(x["thunder"][task]["mean"] for x in full) / n - base,
                "se_our_delta": (seed_sd / math.sqrt(n)) if seed_sd is not None else None,
            })
            tasks_out[task] = entry

        # ---- POOL the three tasks: ONE numerator, ONE denominator, divide ONCE -----
        # F-P fix (2026-08-26).  This used to be the MEAN OF THE THREE PER-TASK
        # PERCENTAGES, which is the wrong aggregation for the user's grading rule and is
        # what made this cell ungradeable on two of three backbones.  A per-task ratio
        # divides by that task's own Waiv gain; when one of those gains is +0.0030 the
        # ratio explodes, and when one is NEGATIVE it rewards regressing.  Pooling first
        # -- mean(our delta) / mean(waiv gain) -- divides by +0.0090 instead, which is a
        # real scale.  ALL THREE tasks are required: a pooled number over two of them
        # silently re-weights the benchmark.
        pooled = _ec.pool_cells(
            [{
                "key": t,
                "delta": tasks_out[t].get("our_delta"),
                "gain": tasks_out[t].get("waiv_gain"),
                "se_delta": tasks_out[t].get("se_our_delta"),
                "sd_gain": tasks_out[t].get("sd_waiv_gain"),
                "complete": tasks_out[t].get("our_delta") is not None
                            and tasks_out[t].get("n", 0) >= MIN_N_FOR_VERDICT,
                "note": tasks_out[t].get("status") or tasks_out[t].get("reason"),
            } for t in CLS_TASKS],
            group="%s/THUNDER (3 tasks)" % a)

        if pooled["status"] != "POOLED":
            cell = {"pct": None, "pct_capped": None, "pct_uncapped": None, "ci": None,
                    "lower_uncapped": None, "upper_uncapped": None, "was_capped": False,
                    "n": 0, "tasks": tasks_out, "pooled": pooled,
                    "status": ("PARTIAL" if pooled["status"] == "WITHHELD_INCOMPLETE"
                               else "INDETERMINATE"),
                    "reason": pooled.get("reason")}
        else:
            agg_n = max(tasks_out[t].get("n") or 0 for t in CLS_TASKS)
            cell = grade(pooled["pct"], pooled["ci"], agg_n)
            cell.update({
                "n": agg_n,
                "tasks": tasks_out,
                "pooled": pooled,
                "our_avg_delta": pooled["our_avg_delta"],
                "waiv_avg_gain": pooled["waiv_avg_gain"],
                "our_delta": pooled["our_avg_delta"],
                "waiv_gain": pooled["waiv_avg_gain"],
                "se_our_delta": pooled["se_our_avg_delta"],
                "sd_waiv_gain": pooled["sd_waiv_avg_gain"],
                "concentration_flags": pooled["concentration_flags"],
                "ci_source": pooled["ci_source"],
                "aggregation": pooled["rule"],
            })
        report["cells"].setdefault(a, {})["THUNDER"] = cell

    # ---- cells with no denominator: PRINTED, not omitted ---------------------
    # A cell that cannot be graded because an INPUT is missing is a different thing from
    # a cell that scored badly, and it is a different thing again from a cell that is
    # simply absent from the table.  Give it a row and the reason.
    for bench in BENCHES:
        for a in ALL_ARMS:
            miss = _missing_denominators_for(a, bench)
            if not miss:
                continue
            report["cells"].setdefault(a, {})[bench] = {
                "pct": None, "pct_capped": None, "pct_uncapped": None, "ci": None,
                "lower_uncapped": None, "upper_uncapped": None, "was_capped": False,
                "n": 0, "status": "NOT REPORTABLE",
                "reason": "missing denominator(s): " + ", ".join(miss),
                "missing_inputs": miss,
            }

    # ---- aggregate ----------------------------------------------------------
    benches = ("RI", "HEST", "THUNDER")
    bench_avg = {}
    for b in benches:
        # ---- POOL across the three backbones: one numerator, one denominator ------
        # F-P fix (2026-08-26).  This was the MEAN OF THE PER-BACKBONE PERCENTAGES.
        # Under the user's grading rule a benchmark mean is the ratio of the two
        # ABSOLUTE averages -- mean(our raw delta) / mean(Waiv's raw gain) -- divided
        # once.  The two spellings differ whenever the per-backbone denominators differ,
        # which they always do (RI gains run 0.337 / 0.061 / 0.021 across the trio, a
        # 16x spread, so a mean of ratios silently weights virchow2 16x heavier than
        # phikon).  All three backbones are required; a benchmark mean over a subset is
        # a different quantity sharing its name (F1), and is now withheld by the same
        # all-or-nothing rule that governs the THUNDER task pool.
        pooled = _ec.pool_cells(
            [{
                "key": a,
                "delta": (report["cells"].get(a, {}).get(b, {}) or {}).get("our_delta"),
                "gain": (report["cells"].get(a, {}).get(b, {}) or {}).get("waiv_gain"),
                "se_delta": (report["cells"].get(a, {}).get(b, {}) or {}).get("se_our_delta"),
                "sd_gain": (report["cells"].get(a, {}).get(b, {}) or {}).get("sd_waiv_gain"),
                "complete": (report["cells"].get(a, {}).get(b, {}) or {}).get("pct") is not None,
                "note": (report["cells"].get(a, {}).get(b, {}) or {}).get("status", "NO_DATA"),
            } for a in BENCH_ARMS[b]],
            group="%s (%d backbones)" % (b, len(BENCH_ARMS[b])))
        graded_bb = [a for a in BENCH_ARMS[b]
                     if (report["cells"].get(a, {}).get(b, {}) or {}).get("pct") is not None]
        bench_avg[b] = {
            "mean": pooled.get("pct"),
            "our_avg_delta": pooled.get("our_avg_delta"),
            "waiv_avg_gain": pooled.get("waiv_avg_gain"),
            "ci": pooled.get("ci"),
            "lower": pooled.get("lower"),
            "upper": pooled.get("upper"),
            "aggregation": pooled["rule"],
            "pooled": pooled,
            "concentration_flags": pooled.get("concentration_flags") or [],
            "n_backbones": len(graded_bb),
            "coverage_ok": len(graded_bb) == len(BENCH_ARMS[b]),
            "n_backbones_required": len(BENCH_ARMS[b]),
            "backbones_without_a_denominator": [a for a in ALL_ARMS if a not in BENCH_ARMS[b]],
            "eligible_for_overall": pooled.get("pct") is not None,
            "withheld_reason": pooled.get("reason") if pooled["status"] != "POOLED" else None,
            # Diagnostic only: what the OLD mean-of-ratios would have printed, so the
            # change in the published number is visible rather than silent.
            "legacy_mean_of_ratios": (
                sum((report["cells"][a][b]["pct"]) for a in graded_bb) / len(graded_bb)
                if graded_bb else None),
        }
    short = [b for b in benches if not bench_avg[b]["eligible_for_overall"]]
    if short:
        overall = None
        overall_note = ("UNDEFINED: %s rest(s) on fewer than %d backbones (%s); the "
                        "overall average is defined only when all three benchmark means "
                        "cover all three backbones"
                        % (", ".join(short), len(ARMS),
                           ", ".join("%s=%d/%d" % (b, bench_avg[b]["n_backbones"],
                                                   len(BENCH_ARMS[b]))
                                     for b in short)))
    else:
        overall = sum(bench_avg[b]["mean"] for b in benches) / len(benches)
        overall_note = "mean of three full-coverage benchmark means"
    # Transparency only -- explicitly NOT the criterion.
    _partial = [bench_avg[b]["mean"] for b in benches if bench_avg[b]["mean"] is not None]
    overall_partial = (sum(_partial) / len(_partial)) if _partial else None

    graded_cells = [(a, b, report["cells"][a][b]) for b in benches for a in BENCH_ARMS[b]
                    if report["cells"].get(a, {}).get(b, {}).get("pct") is not None]
    ungraded_cells = [(a, b, report["cells"][a][b]) for b in benches for a in BENCH_ARMS[b]
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

    # ---- THE CRITERION: PER MODEL, not pooled across models -----------------
    # Corrected 2026-08-26.  The bar is applied to EACH backbone on its own:
    #   * pct_of_waiv >= 70 on EACH of RI, HEST, THUNDER, and
    #   * the mean of that backbone's three benchmark percentages > 80.
    # Pooling still happens WITHIN a cell (the 3 THUNDER tasks are pooled by
    # ratio-of-means), but NOT across backbones.  The pooled-across-backbones block in
    # report["benchmark_averages"] is retained as a SECONDARY view and is explicitly not
    # what decides the verdict -- a benchmark mean over three backbones lets one model's
    # surplus pay for another's shortfall, which is exactly what a per-model bar forbids.
    per_model = {}
    for a in ARMS:
        cells_a = {b: (report["cells"].get(a, {}).get(b) or {}) for b in benches}
        pcts = {b: cells_a[b].get("pct") for b in benches}
        stats = {b: cells_a[b].get("status", "NO_DATA") for b in benches}
        gradeable = [b for b in benches if pcts[b] is not None]
        avg = (sum(pcts[b] for b in gradeable) / len(gradeable)
               if len(gradeable) == len(benches) else None)
        fails = [b for b in gradeable if stats[b] == "FAIL"]
        unres = [b for b in gradeable if stats[b] == "NOT RESOLVED"]
        ungraded = [b for b in benches if pcts[b] is None]
        if ungraded or unres:
            v = "INDETERMINATE"
            why = "; ".join(filter(None, [
                ("no gradeable number for " + ", ".join("%s (%s)" % (b, stats[b])
                                                        for b in ungraded)) if ungraded else "",
                ("error bar straddles the %g bar for " % PASS_BAR
                 + ", ".join("%s (%.1f+/-%.1f)" % (b, pcts[b], cells_a[b]["ci"])
                             for b in unres)) if unres else ""]))
        elif fails:
            v = "FAIL"
            why = "below the %g bar: " % PASS_BAR + ", ".join(
                "%s = %.1f+/-%.1f" % (b, pcts[b], cells_a[b]["ci"]) for b in fails)
        elif avg is not None and avg <= OVERALL_BAR:
            v = "FAIL"
            why = ("every benchmark clears %g but this model's average %.1f does not "
                   "exceed %g" % (PASS_BAR, avg, OVERALL_BAR))
        else:
            v = "PASS"
            wb = min(gradeable, key=lambda b: pcts[b])
            why = ("worst benchmark %s = %.1f (lower bound %.1f) >= %g; model average "
                   "%.1f > %g" % (wb, pcts[wb], cells_a[wb]["lower_uncapped"], PASS_BAR,
                                  avg, OVERALL_BAR))
        per_model[a] = {
            "pct": pcts,
            "ci": {b: cells_a[b].get("ci") for b in benches},
            "lower_uncapped": {b: cells_a[b].get("lower_uncapped") for b in benches},
            "status": stats,
            "n": {b: cells_a[b].get("n") for b in benches},
            "average": avg,
            "average_note": ("mean of this model's three benchmark percentages"
                             if avg is not None else
                             "UNDEFINED: this model has no gradeable number on "
                             + ", ".join(ungraded)),
            "verdict": v,
            "verdict_reason": why,
        }
        # SECONDARY, and labelled as such: the same bars applied to only the benchmarks
        # this model actually has a number on.  NOT the criterion -- a model that clears
        # 70 on two benchmarks has not been shown to clear it on the third -- but the
        # difference between "we measured it and it failed" and "we have not measured
        # it" is the whole content of the verdict right now, and burying both under one
        # INDETERMINATE hides which one applies.
        if gradeable and len(gradeable) < len(benches):
            g_avg = sum(pcts[b] for b in gradeable) / len(gradeable)
            g_fail = [b for b in gradeable if stats[b] == "FAIL"]
            g_unres = [b for b in gradeable if stats[b] == "NOT RESOLVED"]
            if g_fail:
                gv, gwhy = "FAIL", ("below the %g bar: " % PASS_BAR + ", ".join(
                    "%s = %.1f+/-%.1f" % (b, pcts[b], cells_a[b]["ci"]) for b in g_fail))
            elif g_unres:
                gv, gwhy = "NOT RESOLVED", ("error bar straddles %g for " % PASS_BAR
                                            + ", ".join(g_unres))
            elif g_avg <= OVERALL_BAR:
                gv, gwhy = "FAIL", ("every measured benchmark clears %g but their mean "
                                    "%.1f does not exceed %g" % (PASS_BAR, g_avg, OVERALL_BAR))
            else:
                gv, gwhy = "PASS", ("worst measured benchmark %s = %.1f >= %g; mean of "
                                    "measured %.1f > %g"
                                    % (min(gradeable, key=lambda b: pcts[b]),
                                       min(pcts[b] for b in gradeable), PASS_BAR,
                                       g_avg, OVERALL_BAR))
            per_model[a]["measured_only"] = {
                "benchmarks": gradeable,
                "average": g_avg,
                "verdict": gv,
                "verdict_reason": gwhy,
                "NOT_THE_CRITERION": ("the criterion needs all of %s; this is the same "
                                      "bars on %s only"
                                      % ("/".join(benches), "/".join(gradeable))),
            }
    report["per_model"] = per_model
    report["per_model_criterion"] = (
        "THE CRITERION.  Each backbone independently: pct_of_waiv >= %g on EACH of "
        "RI/HEST/THUNDER AND the mean of its three percentages > %g.  No pooling across "
        "backbones." % (PASS_BAR, OVERALL_BAR))
    pm_v = [per_model[a]["verdict"] for a in ARMS]
    report["per_model_verdict"] = ("FAIL" if "FAIL" in pm_v else
                                   "INDETERMINATE" if "INDETERMINATE" in pm_v else "PASS")
    # The rule's sensitivity to its one input, printed rather than asserted.
    if rule == "1se":
        report["ri_se_sweep"] = ri_se_sweep(runs)
    report["pooled_across_backbones_NOT_THE_CRITERION"] = (
        "report['benchmark_averages'], report['overall_average'] and report['verdict'] "
        "pool across the three backbones.  They are SECONDARY.  The criterion is "
        "report['per_model'] / report['per_model_verdict'].")
    return report


def iter_pooled_groups(rep: dict):
    """(group_label, pooled_block) for every pooled group in the report, in print order.

    One walker so the concentration table can never drift out of sync with the set of
    groups the verdict actually rests on.
    """
    for b in ("RI", "HEST", "THUNDER"):
        p = (rep["benchmark_averages"].get(b) or {}).get("pooled")
        if p and p.get("shares"):
            yield ("%s / 3 backbones" % b), p
    for a in ARMS:
        p = ((rep["cells"].get(a, {}) or {}).get("THUNDER", {}) or {}).get("pooled")
        if p and p.get("shares"):
            yield ("%s/THUNDER / 3 tasks" % a), p


def print_report(rep: dict) -> None:
    W = 78
    print("=" * W)
    print("FINAL RECIPE VERDICT -- mask_same_core / cls-bias 3.0 / lr1e-4 / T0.07 / wd0.05")
    print("=" * W)
    cr = rep.get("checkpoint_rule") or {}
    print(f"Run family    : {rep.get('run_family', {}).get('glob')}  "
          f"({rep.get('run_family', {}).get('n_runs_discovered')} runs)")
    print(f"Stopping rule : {rep['stopping_rule']}")
    if cr.get("rule") == "1se":
        print(f"                metric {cr.get('metric')}")
        print(f"                SE     {cr.get('se_source')}")
        print(f"                SE supplied via --ri-se: {cr.get('ri_se_supplied')}  "
              f"| bootstrap SE found on disk: {cr.get('se_measured_on_disk')}")
        if not cr.get("se_measured_on_disk"):
            print(f"                {cr.get('se_unmeasured_note')}")
    print(f"Criterion     : pct_of_waiv >= {PASS_BAR:.0f} on EVERY cell (worst cell, not a mean),")
    print(f"                AND mean of the three benchmark means > {OVERALL_BAR:.0f}.")
    print("                ONE quantity is graded and averaged: the UNCAPPED pct_of_waiv.")
    print("                The 100 cap is presentation only and enters no arithmetic (F-C).")
    print("                Scorecard cells read:  uncapped (capped*) +/-CI n=  status")
    print()

    print("-- CHECKPOINT SELECTION (rule applied per run) " + "-" * 31)
    is_1se = (rep.get("checkpoint_rule") or {}).get("rule") == "1se"
    key, lab, hdr = (("ri", "SE used", "curve (step:RI)") if is_1se
                     else ("mean_ci", "mean CI", "curve (step:meanCI)"))
    print(f"{'backbone':<10} {'seed':>4} {'step':>6} {lab:>9}   {hdr}")
    for r in rep["runs"]:
        curve = " ".join(f"{t['step']}:{t[key]:.4f}" if t.get(key) is not None
                         else f"{t['step']}:--" for t in r["ci_trace"])
        step = r["selected_step"] if r["selected_step"] is not None else "none"
        stat = r.get("se_used") if is_1se else r.get("mean_confounder_insensitivity")
        stat_s = f"{stat:.4f}" if isinstance(stat, (int, float)) else "   --"
        print(f"{r['backbone']:<10} {r['seed']:>4} {str(step):>6} {stat_s:>9}   {curve}")
        if r["selected_step"] is None and r.get("note"):
            print(f"{'':<10} {'':>4} {'':>6} {'':>9}   -> {r['note']}")
    print()

    sweep = rep.get("ri_se_sweep")
    if sweep:
        print("-- 1-SE SENSITIVITY: which step the rule picks, per backbone, vs the SE "
              + "-" * 5)
        print("   The rule has exactly one input.  This is what varying it does; it is a")
        print("   diagnostic, not a selection.  A '-' means the rule did not fire.")
        bbs = sorted({v["backbone"] for v in sweep["runs"].values()})
        print("   " + "SE".ljust(9) + "".join(b_[:11].rjust(13) for b_ in bbs))
        for se in sweep["grid"]:
            k = "%g" % se
            row = "   " + k.ljust(9)
            for b_ in bbs:
                picks = sweep["consensus"][k].get(b_) or []
                row += (",".join(str(x) for x in picks) or "-").rjust(13)
            print(row)
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
    # LEAD WITH THE TWO ABSOLUTE AVERAGES.  The percentage is derived from them and is
    # reported second; a reader who sees only "104%" cannot tell whether that is 104% of
    # a large gain or of a rounding error.
    print("  Pooled (ratio-of-means): ONE numerator and ONE denominator per benchmark,")
    print("  divided ONCE.  Per-cell percentages are NEVER averaged.")
    print()
    print(f"  {'benchmark':<9} {'OUR avg increase':>17} {'WAIV avg increase':>18} "
          f"{'pct':>8} {'+/-95%':>8}  coverage")
    for b in ("RI", "HEST", "THUNDER"):
        m = rep["benchmark_averages"][b]
        ours = f"{m['our_avg_delta']:+.5f}" if m.get("our_avg_delta") is not None else "--"
        waiv = f"{m['waiv_avg_gain']:+.5f}" if m.get("waiv_avg_gain") is not None else "--"
        v = f"{m['mean']:.1f}" if m["mean"] is not None else "WITHHELD"
        ci = f"{m['ci']:.1f}" if m.get("ci") is not None else "--"
        print(f"  {b:<9} {ours:>17} {waiv:>18} {v:>8} {ci:>8}  "
              f"{m['n_backbones']}/3 backbones")
        if m.get("withheld_reason"):
            print(f"            WITHHELD: {m['withheld_reason']}")
        legacy = m.get("legacy_mean_of_ratios")
        if legacy is not None and m["mean"] is not None and abs(legacy - m["mean"]) > 0.05:
            print(f"            (was {legacy:.1f} under the retired mean-of-ratios "
                  f"aggregation; delta {m['mean'] - legacy:+.1f} pts)")
        for f_ in m.get("concentration_flags") or []:
            print(f"            *** CONCENTRATION: {f_}")
    print()
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
    # -- CONCENTRATION (mandatory disclosure) ---------------------------------------
    # Pooling cures a small denominator, but it can also let ONE cell carry the whole
    # pooled result.  Both failure modes have to be visible at once, so every pooled
    # group prints each cell's signed share of its numerator and of its denominator.
    print()
    print("-- CONCENTRATION: each cell's share of its pooled numerator / denominator " + "-" * 3)
    print(f"   {'group':<26} {'cell':<16} {'our delta':>10} {'waiv gain':>10} "
          f"{'num%':>7} {'den%':>7}")
    any_flag = False
    for grp, pooled in iter_pooled_groups(rep):
        for key, s in (pooled.get("shares") or {}).items():
            ns = s.get("numerator_share")
            ds = s.get("denominator_share")
            mark = ""
            if (ns is not None and abs(ns) > _ec.CONCENTRATION_FLAG_SHARE) or \
               (ds is not None and abs(ds) > _ec.CONCENTRATION_FLAG_SHARE):
                mark = "  <== CARRIES THE GROUP"
                any_flag = True
            print(f"   {grp:<26} {key:<16} {s['delta']:>+10.5f} {s['gain']:>+10.5f} "
                  f"{(f'{ns * 100:+.0f}' if ns is not None else '--'):>7} "
                  f"{(f'{ds * 100:+.0f}' if ds is not None else '--'):>7}{mark}")
    if any_flag:
        print(f"   *** A cell above supplies more than "
              f"{_ec.CONCENTRATION_FLAG_SHARE * 100:.0f}% of its group's pooled")
        print("       numerator or denominator.  The pooled number is that cell's result")
        print("       wearing the group's name; read it with the per-cell appendix, not alone.")
    else:
        print(f"   (no cell exceeds {_ec.CONCENTRATION_FLAG_SHARE * 100:.0f}% of any "
              f"pooled numerator or denominator)")
    print()

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
    print(f"Aggregation              : {rep['sources']['aggregation']}")
    print(f"Denominator gate         : {rep['sources']['denominator_gate']}")
    print(f"Gate change 2026-08-26   : {rep['sources']['denominator_gate_change_2026_08_26']}")
    print(f"CI construction          : {rep['sources']['ci_construction']}")
    print()
    r = rep["thunder_cls_roster"]
    print("-- THUNDER CLASSIFICATION ROSTER " + "-" * 45)
    print(f"   panel                 : {r['n_datasets']} datasets ({r['selected']}-set)")
    print(f"   segmentation          : {r['segmentation']}")
    print(f"   seed-floor caveat     : {r['seed_floor_caveat']}")
    print()
    print("-- BASE GAP: OUR BASE minus WAIV'S PUBLISHED BASE (pct points) " + "-" * 15)
    keys = sorted(rep["thunder_base_gap_validation"]["rosters"]["12"]["cells"])
    print(f"   {'cell':<28} {'ours12':>8} {'ours16':>8} {'waiv':>8} {'gap12':>8} {'gap16':>8}")
    for k in keys:
        c12 = rep["thunder_base_gap_validation"]["rosters"]["12"]["cells"][k]
        c16 = rep["thunder_base_gap_validation"]["rosters"]["16"]["cells"][k]
        f = lambda x: "    --  " if x is None else f"{x:8.2f}"
        print(f"   {k:<28} {f(c12['our_base_pct'])} {f(c16['our_base_pct'])} "
              f"{f(c12['waiv_published_base_pct'])} {f(c12['gap_pct_points'])} "
              f"{f(c16['gap_pct_points'])}")
    for lab in ("12", "16"):
        b = rep["thunder_base_gap_validation"]["rosters"][lab]
        print("   %s-set: max |gap| = %.4f  mean gap = %+.4f  (%d/9 cells at full coverage)"
              % (lab, b["max_abs_gap_pct_points"], b["mean_gap_pct_points"],
                 b["n_cells_with_full_coverage"]))
    print()
    print("== THE CRITERION: PER MODEL " + "=" * 50)
    print(f"   {rep['per_model_criterion']}")
    print(f"   {'backbone':<10} {'RI':>16} {'HEST':>16} {'THUNDER':>16} {'avg':>8}  verdict")
    for a in ARMS:
        pm = rep["per_model"][a]
        cells = []
        for b in ("RI", "HEST", "THUNDER"):
            v, ci = pm["pct"][b], pm["ci"][b]
            cells.append("%16s" % (pm["status"][b][:16] if v is None
                                   else "%.1f+/-%s" % (v, "?" if ci is None else "%.1f" % ci)))
        avg = "      --" if pm["average"] is None else "%8.1f" % pm["average"]
        print(f"   {a:<10} {' '.join(cells)} {avg}  {pm['verdict']}")
    for a in ARMS:
        print(f"   {a}: {rep['per_model'][a]['verdict_reason']}")
    print(f"   PER-MODEL VERDICT: {rep['per_model_verdict']}")
    print(f"   ({rep['pooled_across_backbones_NOT_THE_CRITERION']})")
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
    ap.add_argument("--rule", choices=RULES, default="1se",
                    help="checkpoint-selection rule.  1se (default) = the parameter-free "
                         "one-standard-error rule on avg_robustness_index; ci075 = the "
                         "RETIRED first-checkpoint-with-mean-confounder_insensitivity"
                         ">=0.75 rule, kept for before/after inspection.")
    ap.add_argument("--ri-se", type=float, default=None, metavar="SE",
                    help="EXPLICIT, operator-supplied within-run SE of the 3-dataset mean "
                         "RI, used by --rule 1se wherever PathoROB's bootstrap fields are "
                         "not on disk (they are not, anywhere, today).  This is an input "
                         "you are asserting, not a measurement the report made: supply it "
                         "only with a source, and read --ri-se-sweep first.  Omitted, the "
                         "rule does not fire and every affected cell is NOT REPORTABLE.")
    ap.add_argument("--ri-se-sweep", action="store_true",
                    help="print the SE-vs-pick sensitivity table and exit.")
    ap.add_argument("--run-glob", default=None, metavar="GLOB",
                    help=f"run family under runs/ (default {RUN_GLOB} = the finalised "
                         "5-backbone 50-step sweep).")
    ap.add_argument("--cls-roster", choices=sorted(CLS_ROSTERS), default=CLS_ROSTER_DEFAULT,
                    help="THUNDER classification panel: 16 = Waiv's Table-2 roster "
                         "(12 THUNDER-paper sets + 4 SPIDER), the default and the only "
                         "one on which our base is comparable to their published base; "
                         "12 = the THUNDER paper's panel, which reproduces every "
                         "pre-2026-08-26 number.")
    args = ap.parse_args()

    if args.ri_se_sweep:
        sw = ri_se_sweep(discover_runs(args.run_glob))
        print(json.dumps(sw, indent=2))
        return
    rep = build_report(hest_assume_step=args.hest_assume_step,
                       cls_roster=args.cls_roster,
                       rule=args.rule, ri_se=args.ri_se, run_glob=args.run_glob)
    print_report(rep)
    if args.json:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
