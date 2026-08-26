#!/usr/bin/env python3
"""Scoreboard v3 — mechanically enforces one-checkpoint-per-row and raw-scores rules.

RULE 1 (ENFORCED): Every row is ONE (run_name, step).  Never pair best-RI from one
  arm with best-HEST from another.  If a metric is missing for THAT checkpoint,
  print MISSING — never substitute another arm's number.

RULE 2 (ENFORCED): Raw scores are reported as  ours | Waiv | diff.  Gain-vs-base
  may appear as an extra column but never as the headline.

PRIMARY METRIC: pct_of_waiv = (ours - base) / (Waiv - base) * 100  for every
  metric and dataset, CAPPED at 100 (beating Waiv scores 100, not more; the
  uncapped value is retained and printed with a trailing '*').
  The per-checkpoint AVERAGE of the CAPPED pct_of_waiv is the default sort key.

PASS CRITERION (CURRENT): per backbone, EVERY metric (RI, HEST, THUNDER) >= 70%
  of Waiv's gain AND the mean of the three > 80%.  A recipe is scored by its
  WORST backbone.  Printed by verdict_report().  This REPLACES the older
  RI>80 / HEST>70 / THUNDER>80 rule.

NOISE: Every diff is annotated with how many seed-SD it spans.  Differences below
  2 SD are marked [~noise].  Seed-SD is ALSO expressed as a % of Waiv's gain,
  so we can see whether 80% vs 100% is even resolvable.  Metrics where 2SD >
  20% of Waiv's gain are flagged UNRESOLVABLE.

THUNDER PARTIAL GUARD: If a protocol does not have ALL required datasets (12 cls,
  2 seg), the per-protocol mean is printed as PARTIAL(n/total) and the overall
  THUNDER task-mean is suppressed.

RI FLOOR (replaces absolute floors): base + 0.80 * (Waiv - base).  PASS/FAIL
  against this per-checkpoint.

HEST BASE FIX: virchow2 base_hest = 0.40324 (correct; old scoreboard had 0.4034).

Usage:
    python scripts/scoreboard.py                          # all runs @ step 500
    python scripts/scoreboard.py --step 250               # step 250
    python scripts/scoreboard.py --runs final5-midnight-s0-t900-386799
    python scripts/scoreboard.py --only-arm midnight
    python scripts/scoreboard.py --sort-by hest_pct       # sort by pct_of_waiv on HEST
    python scripts/scoreboard.py --only-complete
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Import shared logic from collect_final5 (do not re-implement)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# waivphaet.eval.thunder_protocol is stdlib-only by design (see its docstring) -- it is
# the ONE definition of THUNDER's per-backbone pooling, shared with the runner, which
# itself cannot be imported here because it pulls in the THUNDER package.
if str(_HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "src"))
import collect_final5 as _c5
import eval_common as _ec
from waivphaet.eval import thunder_protocol as _thunder_protocol

# F6 fix (2026-08-26): _c5.HEST_BASE is now READ FROM DISK from the same field
# (hest_perf_per_encoder.custom_encoder) that _c5._hest_score reads for fine-tuned
# runs.  It used to be a hardcoded literal taken from the rounded `results.avg`
# field, so base and FT came from DIFFERENT fields (virchow2 base low by 2.4e-5,
# inflating the virchow2 HEST pct by ~+0.28 points).
HEST_BASE = _c5.HEST_BASE          # see _c5.HEST_BASE_SOURCE for provenance
RI_BASE   = _c5.RI_BASE
THUNDER_BASE_DIRS = _c5.THUNDER_BASE_DIRS
PAPER_CLS = _c5.PAPER_CLS           # 12 classification datasets
PAPER_SEG = _c5.PAPER_SEG           # 2 segmentation datasets
THUNDER_TASKS = _c5.THUNDER_TASKS   # knn, linear_probing, simple_shot, segmentation

# ── Waiv published targets (arXiv:2607.22861 Tables 1+3) ────────────────────
# READ FROM docs/waiv_published.json, not retyped here.  That file is the full
# transcription of all 20 rows of their tables; this module used to hold a
# hand-copied subset of three of them, which is a second source of truth for a
# published constant -- the exact shape of bug that produced the HEST base
# discrepancy (F6).  The literals it used to hold are pinned as a regression
# guard in tests/test_invariants.py, which asserts the derived dicts still equal
# them value-for-value.
#
# Their model NAMES are not our arm tokens, and for two backbones the fine-tuned
# row has a different name from the base row (Phikon-v2 -> Phaet,
# Midnight-12k -> Mascaret), so the correspondence has to be stated.  It is
# stated ONCE, here, as (base_row, fine_tuned_row) keyed by our arm.
#
# TRAP: "H0-mini" is a separate row and a DIFFERENT model -- a distillation of
# H-Optimus-0, with its own numbers and its own (clsmean) THUNDER protocol.  It
# is not an alias for `hoptimus`.
WAIV_PUBLISHED_JSON = _HERE.parent / "docs" / "waiv_published.json"

WAIV_ROWS = {
    "phikon":   (("Phikon-v2", "base"),    ("Phaet", "fine-tuned")),
    "midnight": (("Midnight-12k", "base"), ("Mascaret", "fine-tuned")),
    "virchow2": (("Virchow2", "base"),     ("Virchow2", "fine-tuned")),
    "hoptimus": (("H-Optimus-0", "base"),  ("H-Optimus-0", "fine-tuned")),
    "uni2":     (("UNI2-h", "base"),       ("UNI2-h", "fine-tuned")),
}

#: Waiv's THUNDER task names -> ours.  Their two extra tasks (calibration,
#: adversarial) are not computed by this repo and are dropped here on purpose --
#: see note (a) below on why any mean over these four is not their rank sum.
_WAIV_THUNDER_TASKS = {
    "knn": "knn",
    "linear": "linear_probing",
    "few_shot": "simple_shot",
    "segmentation": "segmentation",
}

#: Their RI per-dataset key -> ours.
_WAIV_RI_DS = {"tcga": "tcga", "camelyon": "camelyon", "tolkach": "tolkach_esca"}


def _load_waiv_published():
    """(WAIV, WAIV_THUNDER) read from docs/waiv_published.json.  ONE formula, all arms."""
    try:
        blob = json.loads(WAIV_PUBLISHED_JSON.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the Waiv published-numbers transcription at "
            f"{WAIV_PUBLISHED_JSON}: {exc}. Every pct_of_waiv denominator comes from "
            "that file; there is no fallback literal to fall back to."
        ) from exc
    index = {(m["name"], m["variant"]): m for m in blob["models"]}
    waiv, waiv_thunder = {}, {}
    for arm, (base_row, ft_row) in WAIV_ROWS.items():
        missing = [r for r in (base_row, ft_row) if r not in index]
        if missing:
            raise RuntimeError(
                f"arm {arm!r} maps to rows {missing} which are not in "
                f"{WAIV_PUBLISHED_JSON.name}; fix WAIV_ROWS or the transcription."
            )
        base, ft = index[base_row], index[ft_row]
        waiv[arm] = {
            "ri": ft["ri"]["avg"],
            "hest": ft["hest_avg"],
            "ri_ds": {ours: ft["ri"][theirs] for theirs, ours in _WAIV_RI_DS.items()},
        }
        waiv_thunder[arm] = {
            "base": {ours: base["thunder"][theirs] for theirs, ours in _WAIV_THUNDER_TASKS.items()},
            "ft": {ours: ft["thunder"][theirs] for theirs, ours in _WAIV_THUNDER_TASKS.items()},
        }
    return waiv, waiv_thunder

# ── Waiv published THUNDER targets ──────────────────────────────────────────
# Source: Filiot, Thaeter, Schmauch, Guillou, "Robustifying pathology foundation
# models via fine-tuning", arXiv:2607.22861v1 (24 Jul 2026), TABLE 2
# ("THUNDER benchmark. Per-task scores with leaderboard rank in parentheses").
# Verified 2026-08-24 against https://arxiv.org/html/2607.22861v1 Table 2; the
# values below are byte-identical to the full transcription in
# docs/waiv_published.json and to PUBLISHED_TASKMEAN in scripts/collect_thunder.py.
#
# Values are PERCENT as printed; divided by 100 at use so they match our
# fractional means.  Waiv's task keys are renamed to ours:
#     knn -> knn | linear -> linear_probing | few_shot -> simple_shot | segmentation -> segmentation
# Waiv's two extra tasks (calibration, adversarial) are NOT computed by this repo
# and are deliberately absent -- any mean here is over 4 tasks, not their 6, and is
# therefore NOT comparable to their rank sum.
#
# READ THIS BEFORE TRUSTING A THUNDER pct_of_waiv:
#
#  (a) TWO-BASE FORMULA.  Unlike RI and HEST -- where our measured base reproduces
#      Waiv's published base to 4 decimals (RI 0.4686/0.7589/0.8582 vs their
#      0.469/0.759/0.858; HEST 0.37470/0.39521/0.40324 vs their 0.3747/0.3952/0.4034)
#      -- our measured THUNDER base does NOT reproduce theirs.  Classification sits
#      ~2-4 points BELOW their base on every backbone (e.g. phikon-v2 kNN: ours
#      0.7028 vs their 74.0), while our 2-dataset segmentation sits ~4 points ABOVE
#      their 4-dataset one.  Our base instead reproduces THUNDER's OWN paper
#      (arXiv:2507.07860v3 Tables S37/S39/S50: kNN 70.14, lin-probe 76.46 vs our
#      70.28 / 76.54).  So Waiv's THUNDER absolutes are on a different scale from
#      both us and the THUNDER authors, and a LEVEL comparison is invalid.
#      Consequently THUNDER pct_of_waiv is computed as a GAIN RATIO with two bases:
#          (ours - OUR_base) / (Waiv_ft - Waiv_base)
#      RI and HEST keep the single-base formula because their bases agree.
#
#  (b) SEGMENTATION SUPPORT MISMATCH.  Waiv's segmentation mean is over FOUR datasets
#      (ocelot, pannuke, segpath_epithelial, segpath_lymphocytes); ours is over TWO
#      (see PAPER_SEG in collect_final5.py).  Every segmentation cell -- and any
#      task-mean containing it -- is flagged support_2v4 and must not be read as a
#      matched comparison.
#
#  (c) NO PER-DATASET BREAKDOWN.  Waiv publish task-level means only; §3.3 states
#      "Each model is evaluated on frozen features following the default protocol"
#      and never names the metric.  Alignment with our F1 (classification) / Dice
#      (segmentation, binary Dice == F1) is INHERITED from THUNDER's defaults, not
#      asserted by Waiv.  There is no appendix per-dataset table to check it against.
#
#  (d) WAIV REGRESS ON SOME TASKS.  Where Waiv_ft < Waiv_base the denominator is
#      negative and pct_of_waiv is meaningless (matching a regression would score
#      100%).  Those cells are guarded to N/A with reason "waiv_regressed".
#  (e) NOT EVERY ARM IS GRADEABLE.  For `hoptimus` and especially `uni2`, Waiv's own
#      THUNDER gain is at or below our measured seed floor, and three of UNI2-h's four
#      task gains are NEGATIVE.  Nothing special is done about that here: (d)'s
#      `waiv_regressed` guard already refuses a ratio against a negative denominator,
#      _cap_pct already handles a near-zero one, and _thunder_2se returns None for an
#      (arm, task) whose floor was never measured -- which the callers already turn into
#      an ungradeable cell.  The generic guards ARE the answer; do not add a per-model
#      case.
WAIV, WAIV_THUNDER = _load_waiv_published()
WAIV_SOURCE = f"docs/waiv_published.json ({WAIV_PUBLISHED_JSON})"
WAIV_THUNDER_SOURCE = (
    "arXiv:2607.22861v1 Table 2 (THUNDER benchmark), verified 2026-08-24, "
    "read from docs/waiv_published.json"
)

# Tasks whose dataset support differs between us (2 seg datasets) and Waiv (4).
THUNDER_SUPPORT_MISMATCH = {"segmentation"}


# Noise floors (seed 1-SD, per backbone per step), in RAW metric units.
#
# HEST values are POOLED WITHIN-RECIPE seed SDs measured 2026-08-24, not borrowed from a
# single family.  Pooling formula: sqrt( sum_f (n_f - 1) * sd_f^2 / sum_f (n_f - 1) ) over
# every recipe family with n>=2 seeds at that (backbone, step), same HEST pooling protocol.
# Using ONE family systematically mis-states the floor -- the per-family spread is large:
#   virchow2@250: final5 n=3 -> 0.00153 | ph2 n=2 -> 0.00039 | ret0.01 n=2 -> 0.00263
#   midnight@250: final5 n=3 -> 0.00227 | lr3e-5 n=3 -> 0.00123 | ph2 n=2 -> 0.00004
# so the pooled value is the defensible one.  df is the pooled degrees of freedom.
NOISE_SD = {
    "midnight": {
        250: {"ri": 0.00482, "hest": 0.00163},   # hest pooled df=4 (final5 n=3, lr3e-5 n=2, ph2 n=2)
        500: {"ri": 0.00211, "hest": 0.00185},   # hest pooled df=5 (final5 n=5, ph2 n=2)
    },
    "virchow2": {
        250: {"ri": 0.00475, "hest": 0.00171},   # hest pooled df=4 (final5 n=3, ph2 n=2, ret0.01 n=2)
        500: {"ri": 0.00196, "hest": 0.00154},   # hest pooled df=6 (final5 n=5, ph2 n=3)
    },
    "phikon": {
        500: {"ri": 0.00453, "hest": 0.00167},   # unchanged: no n>=2 HEST family available
    },
}

# UNRESOLVABLE-BY-CONSTRUCTION limit.  A (backbone, metric) cell is refused a printed
# pct_of_waiv when one seed-SD, expressed in pct_of_waiv points, exceeds this.
# NOTE this is ALGEBRAICALLY THE SAME TEST the module already applied as
# "2*seed_SD > 20% of Waiv's gain" -- 2*sd/gain > 0.20  <=>  sd/gain*100 > 10.
# What changes here is the CONSEQUENCE: the cell used to be annotated and printed anyway;
# now the number is withheld and a raw delta + CI is printed in its place.
# F-B fix (2026-08-26): the threshold AND the test now live in scripts/eval_common.py so
# that final_recipe_report.py applies literally the same gate instead of its own private
# THUNDER-only variant.  The name is re-exported unchanged for every existing caller.
UNRESOLVABLE_SD_PCT_LIMIT = _ec.UNRESOLVABLE_SD_PCT_LIMIT

# THUNDER per-(backbone, task) seed floors, offset-2SE form.
# Source: docs/thunder_seed_floor_12ds.md (raw: docs/thunder_seed_floor_12ds.json,
# script: scripts/thunder_seed_floor_12ds.py) -- n=5 TRAINING seeds from the final5
# t900 family at step 500, metric=f1, adaptation=frozen, mean offset-2SE over the 10
# unordered seed pairs.  Contamination-checked: identical configs across all 15 runs,
# no restart siblings, 12/12 dataset coverage, t450/t1800 siblings excluded.
#
# *** THESE ARE FLOORS FOR THE 12-DATASET PAPER_CLS TASK MEAN. ***
# That is exactly the quantity this scoreboard reports (see _thunder_for(), which
# means over PAPER_CLS).  They are INVALID for a partial-coverage mean: a mean over
# n < 12 datasets averages away less per-dataset noise and is therefore NOISIER, so
# applying a 12-dataset floor to it would understate the noise and manufacture
# resolvability.  Any cell whose coverage is < 12/12 must be reported as
# PARTIAL_COVERAGE and graded by NOTHING -- see the partial guard in score_run().
#
# The floors are keyed by BACKBONE, not by pooling: midnight and virchow2 share
# pooling=clsmean but have materially different floors (e.g. knn 0.0100 vs 0.0083).
# Each backbone has exactly one classification pooling protocol, so (backbone, task)
# is both the finer and the correct key.
#
# *** simple_shot now HAS a measured floor; segmentation still does NOT. ***
# _thunder_2se() returns None for segmentation ON PURPOSE.  There is no safe small
# default: substituting one would re-create exactly the bug this dict fixes.  EVERY
# consumer must treat None as "unmeasured floor" -> the cell is NOT eligible for a
# pass/fail verdict and must be reported as UNMEASURED_FLOOR, never as a clean pct.
THUNDER_TASK_2SE_12DS = {
    ("phikon",   "knn"):             0.0233,
    ("phikon",   "linear_probing"):  0.0097,
    ("phikon",   "simple_shot"):     0.0087,
    ("midnight", "knn"):             0.0100,
    ("midnight", "linear_probing"):  0.0087,
    ("midnight", "simple_shot"):     0.0104,
    ("virchow2", "knn"):             0.0083,
    ("virchow2", "linear_probing"):  0.0088,
    ("virchow2", "simple_shot"):     0.0066,
    # ("*", "segmentation"):  UNMEASURED -- deliberately absent.  Segmentation is a
    # mean over PAPER_SEG, a different (2-dataset) panel; no floor was ever measured
    # for it and none may be borrowed from the classification panel.
}

# SUPERSEDED -- retained for historical comparison ONLY, never consulted by the code.
# Source: docs/thunder_seed_floor.md -- a task mean over just 5 datasets
# (bach/mhist/break_his/bracs/ccrcc) from a SINGLE seed pair (n=2 seeds, df=0) of the
# fast5_ctrl/fast5_ctrlseed family.  It was doubly wrong for what the scoreboard
# reports: wrong n (5 vs 12) and wrong noise level (those 5 are the noisiest datasets
# in the panel, 2-8x noisier than the other 7).  It also had NO simple_shot floor.
# Using it forced phikon/linear_probing, midnight/knn and the three simple_shot cells
# to read UNRESOLVABLE/UNMEASURED when they are in fact gradeable.
THUNDER_TASK_2SE_5DS_LEGACY = {
    ("cls",     "knn"):             0.0297,
    ("cls",     "linear_probing"):  0.0156,
    ("clsmean", "knn"):             0.0468,
    ("clsmean", "linear_probing"):  0.0208,
}

# pct_of_waiv cap.  The scoring criterion treats "exceeded Waiv" as 100, not more:
# a recipe that overshoots on one metric must not launder that overshoot into the
# average and mask a shortfall elsewhere.  The UNCAPPED value is retained and printed
# (as `{metric}_pct_uncapped`, rendered with a trailing `*`) so nothing is hidden.
CAP_PCT_AT_100 = True

# Pass criterion (CURRENT).  Per backbone: every metric >= 70% of Waiv's gain AND the
# mean of the three > 80%.  A recipe is scored by its WORST backbone.
# (This REPLACES the older RI>80 / HEST>70 / THUNDER>80 rule.)
VERDICT_MIN_PCT = 70.0
# F2 fix (2026-08-26): a single run has no error bar, so it cannot support a
# PASS/FAIL claim.  Cells below this many runs return UNDERPOWERED, never PASS.
MIN_N_FOR_VERDICT = 2
VERDICT_MEAN_PCT = 80.0

# One list of arms for the whole repo, defined in collect_final5.ARM_BACKBONE next to
# the tables that are keyed by it.  A second tuple here is how `hoptimus` ends up in the
# collector's aggregates and missing from every scoreboard table.
ARMS = _c5.ARMS
ARM_BACKBONE = _c5.ARM_BACKBONE

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_RUNS = _REPO / "runs"


# ── pct_of_waiv helpers ─────────────────────────────────────────────────────

def _waiv_gain(base, waiv):
    """Return Waiv - base, or None if either is missing."""
    if base is None or waiv is None:
        return None
    return waiv - base


def _fmt(v, spec="%+.5f"):
    """Null-safe numeric formatter.  Returns 'n/a' instead of raising on None.

    print_run_block formats diffs that are deliberately None when a base or a
    Waiv reference is unavailable (e.g. UNKNOWN-BACKBONE runs, where
    RI_BASE.get(arm) is None by design).  Formatting those with %f raised
    TypeError and killed the trailing detail section after ~2880 good lines.
    """
    if v is None:
        return "n/a"
    return spec % v


def _cap_pct(pct):
    """Return (capped, uncapped).  Cap is at 100 -- exceeding Waiv scores 100, not more."""
    if pct is None:
        return None, None
    if CAP_PCT_AT_100 and pct > 100.0:
        return 100.0, pct
    return pct, pct


def _pct_of_waiv(ours, base, waiv):
    """
    Compute (ours - base) / (Waiv - base) * 100.
    Returns (pct_capped, guard_reason, pct_uncapped).  pct is None when uncomputable.
    The CAPPED value is the primary one and is what feeds avg_pct_of_waiv.
    """
    if ours is None:
        return None, "no_ours", None
    gain = _waiv_gain(base, waiv)
    if gain is None:
        if base is None:
            return None, "no_base", None
        if waiv is None:
            return None, "no_waiv", None
        return None, "unknown", None
    if abs(gain) < 1e-10:
        return None, "base>=Waiv", None
    capped, uncapped = _cap_pct((ours - base) / gain * 100.0)
    return capped, None, uncapped


def _pct_of_waiv_two_base(ours, our_base, waiv_base, waiv_ft):
    """
    GAIN-RATIO form of pct_of_waiv for metrics where our base does not reproduce
    Waiv's base (THUNDER).  Numerator uses OUR base, denominator uses THEIRS:

        (ours - our_base) / (waiv_ft - waiv_base) * 100

    Returns (pct_capped, guard_reason, pct_uncapped).  pct is None when uncomputable.
    The CAPPED value is the primary one and is what feeds avg_pct_of_waiv.
    """
    if ours is None:
        return None, "no_ours", None
    if our_base is None:
        return None, "no_base", None
    if waiv_base is None or waiv_ft is None:
        return None, "no_waiv", None
    gain = waiv_ft - waiv_base
    if abs(gain) < 1e-10:
        return None, "waiv_gain_zero", None
    if gain < 0:
        # Waiv REGRESSED on this task.  A ratio against a negative gain rewards
        # getting worse; refuse to print one.
        return None, "waiv_regressed", None
    capped, uncapped = _cap_pct((ours - our_base) / gain * 100.0)
    return capped, None, uncapped


def _thunder_pooling(arm):
    """THUNDER *classification* pooling protocol per backbone.

    Delegates to `waivphaet.eval.thunder_protocol`, which is the same table the
    THUNDER runner itself resolves `WAIV_POOLING=auto` through.  Two earlier
    versions of this function were local re-derivations and both were wrong:
    first `cls unless virchow2` (the HEST rule, wrong for midnight), then
    `cls if phikon else clsmean` -- which is right for the published trio by
    coincidence and wrong for every backbone added after it, including
    H-Optimus-0 and UNI2-h (both `cls`).  The protocol is a transcription of
    arXiv:2607.22861 3 and cannot be inferred from an arm name at all, so this
    reads the transcription instead of paraphrasing it.

    It stays DELIBERATELY separate from collect_final5.hest_pooling(): the two
    benchmarks use different protocols (midnight is cls on HEST, clsmean on
    THUNDER) and those two rules must never be merged.
    """
    return _thunder_protocol.default_pooling(ARM_BACKBONE[arm])


def _thunder_2se(arm, task):
    """Measured 12-dataset offset-2SE seed floor, or None when never measured.

    Keyed by (backbone_arm, task) -- see THUNDER_TASK_2SE_12DS.  None means
    UNMEASURED (segmentation) -- NOT "small".  Callers must refuse a pass/fail
    verdict for such a cell rather than fall back to a default.

    VALIDITY: the returned number is the floor of the mean over ALL 12 PAPER_CLS
    datasets.  It must NOT be applied to a cell with partial coverage (n < 12);
    such cells are PARTIAL_COVERAGE and are ungradeable.
    """
    return THUNDER_TASK_2SE_12DS.get((arm, task))


def _thunder_2se_mean(arm, tasks):
    """Floor for a mean over `tasks`.

    Returns (floor, unmeasured_tasks).  If ANY constituent task lacks a measured
    floor the mean has no floor either -> (None, [those tasks]).  Otherwise the
    conservative choice is the LARGEST constituent floor.
    """
    unmeasured = [t for t in tasks if _thunder_2se(arm, t) is None]
    if unmeasured:
        return None, unmeasured
    return max(_thunder_2se(arm, t) for t in tasks), []


def _thunder_unresolvable(waiv_gain_frac, arm=None, task=None, floor=None):
    """2*seed_SE > 20% of Waiv's gain  ->  80% vs 100% is not distinguishable.

    Returns (unresolvable, noise_pct_of_gain, floor_unmeasured).
    floor_unmeasured=True means the seed floor for this (arm, task) was NEVER
    measured; the cell is then INDETERMINATE -- it is neither pass nor fail, and the
    caller must render it as UNMEASURED_FLOOR.

    This function knows nothing about a run's dataset coverage.  Coverage is a
    SEPARATE gate (PARTIAL_COVERAGE) applied by the caller: the floors here are only
    valid for a full 12/12 PAPER_CLS mean.
    """
    if floor is None and task is not None:
        floor = _thunder_2se(arm, task)
    if floor is None:
        # No measured floor.  Do NOT substitute a small default -- that is the exact
        # bug the old scalar THUNDER_TASK_2SE=0.0025 had.
        return True, None, True
    if waiv_gain_frac is None or abs(waiv_gain_frac) < 1e-10:
        return True, None, False
    noise_pct = floor / abs(waiv_gain_frac) * 100.0
    # NOTE ON THE 20% BAR -- it is STRICTER than "resolvable" in
    # docs/thunder_seed_floor_12ds.md, and deliberately so.  That doc calls a cell
    # resolvable when floor < |Waiv gain| (ratio < 1), i.e. we could tell a full-gain
    # arm from a zero-gain arm.  This scoreboard asks a harder question: can we tell
    # 80% of the gain from 100% of it?  That band is 20% of the gain wide, so the bar
    # is floor < 0.20 * |gain|.  Under the corrected 12-dataset floors the best cell
    # (virchow2/simple_shot) sits at 24% -- so EVERY THUNDER cell still trips this
    # bar on a SINGLE run.  That is the honest state of the instrument, not a bug:
    # grading a 70/80% criterion on THUNDER requires averaging seeds (5-seed mean
    # divides the floor by ~sqrt(5)), not a lower bar.  Do not relax this constant to
    # make cells pass.
    return (noise_pct > 20.0), noise_pct, False


# ── Run-name parsing ────────────────────────────────────────────────────────

def _infer_backbone(name):
    for a in ARMS:
        if a in name:
            return a
    return None


def _parse_trail(name):
    m = re.search(r"-s(\d+)-t(\d+)-(\d+)$", name)
    if m:
        return {"seed": int(m.group(1)), "T": int(m.group(2)), "jobid": m.group(3)}
    return {}


def _parse_arm_desc(name, arm):
    s = name
    s = re.sub(r"-s\d+-t\d+-\d+$", "", s)
    if arm:
        s = re.sub(r"[-_]?%s[-_]?" % re.escape(arm), "-", s).strip("-")
    parts = []
    if s == "final5" or s.startswith("final5"):
        parts.append("final5-plain-recipe")
    else:
        if "genMASK" in name or re.search(r"\bMASK\b", name, re.IGNORECASE):
            parts.append("mask=ON")
        elif re.search(r"\bgen\b", name):
            parts.append("mask=OFF")
        if re.match(r"^ph2[-]", name) or re.search(r"[-_]ph2[-_]", name):
            parts.append("ph2(per-head)")
        m = re.search(r"ret([\d.]+)", s)
        if m: parts.append("retention=%s" % m.group(1))
        m = re.search(r"lr([\de\-.]+)", s)
        if m: parts.append("lr=%s" % m.group(1))
        m = re.search(r"(?<![a-z])-r(\d+)(?![\d])", s)
        if m: parts.append("r=%s" % m.group(1))
        m = re.search(r"kl([\d.]+)", s)
        if m: parts.append("kl=%s" % m.group(1))
        m = re.search(r"(?<![a-z])-t([\d.]+)(?![\d])", s)
        if m: parts.append("temp=%s" % m.group(1))
        m = re.search(r"wd([\d.]+)", s)
        if m: parts.append("wd=%s" % m.group(1))
        m = re.search(r"ms(\d+)", s)
        if m: parts.append("ms=%s" % m.group(1))
    return " ".join(parts) if parts else s


def parse_run_meta(run_dir):
    name = run_dir.name
    arm = _infer_backbone(name)
    trail = _parse_trail(name)
    desc = _parse_arm_desc(name, arm)
    return {"run_name": name, "run_dir": run_dir, "arm": arm, "desc": desc, **trail}


# ── Metric reading (reuse collect_final5 internals) ─────────────────────────

def _ri_point_at(run_dir, step):
    points, _ = _c5._union_ri_curves(run_dir)
    for pt in points:
        if pt.get("step") == step:
            return pt
    return None


def _available_steps_in(run_dir):
    points, _ = _c5._union_ri_curves(run_dir)
    return sorted(pt["step"] for pt in points if pt.get("step") is not None)


def _thunder_for(run_name, step):
    step_str = "%07d" % step
    model = "f5_%s_s%s" % (run_name, step_str)
    per_ds_all = _c5._thunder_per_ds_by_model(model)
    result = {}
    for task in THUNDER_TASKS:
        total = len(PAPER_SEG if task == "segmentation" else PAPER_CLS)
        ds_vals = per_ds_all.get(task, {})
        n = len(ds_vals)
        partial = n < total
        mean = (sum(ds_vals.values()) / n) if (n > 0 and not partial) else None
        result[task] = {"mean": mean, "per_ds": ds_vals, "n": n, "total": total, "partial": partial}
    return result


def _sd_for(arm, step, metric):
    if arm is None:
        return None
    return NOISE_SD.get(arm, {}).get(step, {}).get(metric)


def _noise_tag(diff_abs, sd):
    if sd is None or sd == 0:
        return ""
    n = diff_abs / sd
    return "[~noise]" if n < 2.0 else "[%.1fxSD]" % n


def score_run(meta, step):
    """Score ALL metrics at EXACTLY step for one run.  RULE 1 enforced."""
    run_dir = meta["run_dir"]
    name = meta["run_name"]
    arm = meta.get("arm")

    r = dict(meta)
    r["step"] = step

    # RI
    ri_pt = _ri_point_at(run_dir, step)
    if ri_pt is not None:
        r["ri"] = ri_pt.get("avg_robustness_index")
        ds = ri_pt.get("datasets", {})
        r["ri_ds"] = {
            "tcga":         ds.get("tcga",         {}).get("robustness_index"),
            "camelyon":     ds.get("camelyon",      {}).get("robustness_index"),
            "tolkach_esca": ds.get("tolkach_esca",  {}).get("robustness_index"),
        }
    else:
        r["ri"] = None
        r["ri_ds"] = {}
        r["step_missing_note"] = "step %d not in ri_curve; available: %s" % (step, _available_steps_in(run_dir))

    # HEST
    r["hest"] = _c5._hest_score(name, step, arm) if arm else None

    # THUNDER
    r["thunder"] = _thunder_for(name, step)

    # pct_of_waiv computations
    waiv_dict = WAIV.get(arm, {})
    base_ri = RI_BASE.get(arm)
    base_hest = HEST_BASE.get(arm)

    # RI pct_of_waiv
    r["ri_pct"], r["ri_pct_guard"], r["ri_pct_uncapped"] = _pct_of_waiv(
        r["ri"], base_ri, waiv_dict.get("ri"))

    # HEST pct_of_waiv
    r["hest_pct"], r["hest_pct_guard"], r["hest_pct_uncapped"] = _pct_of_waiv(
        r["hest"], base_hest, waiv_dict.get("hest"))

    # THUNDER pct_of_waiv -- GAIN RATIO with two bases (see WAIV_THUNDER note).
    base_thunder = _c5._thunder_base_score(arm) if arm in ARMS else {}
    r["base_thunder"] = base_thunder

    wt = WAIV_THUNDER.get(arm, {})
    wt_base = wt.get("base", {})
    wt_ft = wt.get("ft", {})
    pooling = _thunder_pooling(arm)
    r["thunder_pooling"] = pooling
    th_pct = {}
    for task in THUNDER_TASKS:
        td = r["thunder"].get(task, {})
        # PARTIAL-SWEEP GUARD: no task mean unless every required dataset is present.
        ours_t = td.get("mean") if not td.get("partial", True) else None
        guard_partial = None
        if td.get("partial", True):
            guard_partial = "partial_%d/%d" % (td.get("n", 0), td.get("total", 0))
        wb = wt_base.get(task)
        wf = wt_ft.get(task)
        wb_f = wb / 100.0 if wb is not None else None
        wf_f = wf / 100.0 if wf is not None else None
        gain = (wf_f - wb_f) if (wb_f is not None and wf_f is not None) else None
        pct, guard, pct_unc = _pct_of_waiv_two_base(ours_t, base_thunder.get(task), wb_f, wf_f)
        if guard_partial:
            pct, guard, pct_unc = None, guard_partial, None
        unres, noise_pct, floor_unmeasured = _thunder_unresolvable(gain, arm, task)
        # COVERAGE GATE, tied to the floor's own validity.  The measured floor is the
        # floor of the FULL 12-dataset task mean; a mean over fewer datasets is
        # noisier, so the floor may not be applied to it.  This reuses the SAME
        # partial flag the pct guard above uses -- one source of truth, not two.
        partial_coverage = bool(td.get("partial", True))
        if partial_coverage:
            noise_pct = None
            unres = True          # ungradeable: no valid floor exists for this n
        th_pct[task] = {
            "pct": pct,
            "pct_uncapped": pct_unc,
            "guard": guard,
            "pooling": pooling,
            "seed_2se": None if partial_coverage else _thunder_2se(arm, task),
            "seed_2se_measured": _thunder_2se(arm, task),
            "partial_coverage": partial_coverage,
            "coverage": "%d/%d" % (td.get("n", 0), td.get("total", 0)),
            "unmeasured_floor": floor_unmeasured,
            "waiv_base": wb_f,
            "waiv_ft": wf_f,
            "waiv_gain": gain,
            "our_base": base_thunder.get(task),
            "unresolvable": unres,
            "noise_pct_of_gain": noise_pct,
            "support_mismatch": task in THUNDER_SUPPORT_MISMATCH,
        }
    r["thunder_pct"] = th_pct

    # THUNDER task-mean pct_of_waiv: only when ALL four protocols are complete.
    complete = [t for t in THUNDER_TASKS
                if not r["thunder"].get(t, {}).get("partial", True)
                and r["thunder"].get(t, {}).get("mean") is not None]
    if len(complete) == len(THUNDER_TASKS) and wt_base and wt_ft:
        ours_m = sum(r["thunder"][t]["mean"] for t in THUNDER_TASKS) / len(THUNDER_TASKS)
        ourb_vals = [base_thunder.get(t) for t in THUNDER_TASKS]
        if all(v is not None for v in ourb_vals):
            ourb_m = sum(ourb_vals) / len(ourb_vals)
            wb_m = sum(wt_base[t] for t in THUNDER_TASKS) / len(THUNDER_TASKS) / 100.0
            wf_m = sum(wt_ft[t] for t in THUNDER_TASKS) / len(THUNDER_TASKS) / 100.0
            pct, guard, pct_unc = _pct_of_waiv_two_base(ours_m, ourb_m, wb_m, wf_m)
            mean_floor, unmeasured_tasks = _thunder_2se_mean(arm, THUNDER_TASKS)
            unres, noise_pct, floor_unmeasured = _thunder_unresolvable(
                wf_m - wb_m, arm, None, floor=mean_floor)
            r["thunder_mean_pct"] = pct
            r["thunder_mean_pct_uncapped"] = pct_unc
            r["thunder_mean_pct_guard"] = guard
            r["thunder_mean_unresolvable"] = unres
            r["thunder_mean_noise_pct_of_gain"] = noise_pct
            # The 4-task mean spans simple_shot/segmentation, which have NO measured
            # floor -> the mean has none either.  It is reported, but never as a clean
            # pct and never as a pass/fail input.
            r["thunder_mean_unmeasured_floor"] = floor_unmeasured
            r["thunder_mean_unmeasured_tasks"] = unmeasured_tasks
        else:
            r["thunder_mean_pct"], r["thunder_mean_pct_guard"] = None, "no_base"
            r["thunder_mean_unmeasured_floor"] = True
    else:
        r["thunder_mean_pct"] = None
        r["thunder_mean_pct_guard"] = "partial_%d/%d_protocols" % (len(complete), len(THUNDER_TASKS))
        r["thunder_mean_unmeasured_floor"] = True

    # VERDICT-GRADE THUNDER number: mean of capped per-task pcts over ONLY the tasks
    # that have a measured seed floor AND are resolvable AND are complete.  None here
    # means the THUNDER cell is INDETERMINATE for verdict purposes.
    elig, elig_tasks, skipped = [], [], []
    for task in THUNDER_TASKS:
        tp = th_pct.get(task, {})
        if tp.get("unmeasured_floor"):
            skipped.append("%s:UNMEASURED_FLOOR" % task)
            continue
        if tp.get("partial_coverage"):
            # The 12-dataset floor does not describe this cell's noise.  Refuse to
            # grade rather than reuse a floor measured on a different denominator.
            skipped.append("%s:PARTIAL_COVERAGE(%s)" % (task, tp.get("coverage")))
            continue
        if tp.get("pct") is None:
            skipped.append("%s:%s" % (task, tp.get("guard") or "no_pct"))
            continue
        if tp.get("unresolvable"):
            skipped.append("%s:unresolvable" % task)
            continue
        elig.append(tp["pct"])
        elig_tasks.append(task)
    r["thunder_verdict_pct"] = (sum(elig) / len(elig)) if elig else None
    r["thunder_verdict_tasks"] = elig_tasks
    r["thunder_verdict_skipped"] = skipped

    # RI floor: base + 0.80 * (Waiv - base)
    ri_gain = _waiv_gain(base_ri, waiv_dict.get("ri"))
    if ri_gain is not None:
        r["ri_floor"] = base_ri + 0.80 * ri_gain
        r["ri_floor_pct"] = 80.0
    else:
        r["ri_floor"] = None
        r["ri_floor_pct"] = None

    if r["ri"] is not None and r["ri_floor"] is not None:
        r["ri_budget"] = "PASS" if r["ri"] >= r["ri_floor"] else "FAIL"
    else:
        r["ri_budget"] = "N/A"

    # Average pct_of_waiv (for sorting)
    pcts = []
    if r["ri_pct"] is not None:
        pcts.append(r["ri_pct"])
    if r["hest_pct"] is not None:
        pcts.append(r["hest_pct"])
    if r.get("thunder_mean_pct") is not None:
        pcts.append(r["thunder_mean_pct"])
    r["avg_pct_of_waiv"] = (sum(pcts) / len(pcts)) if pcts else None

    # Noise as % of Waiv's gain
    sd_ri = _sd_for(arm, step, "ri")
    sd_hest = _sd_for(arm, step, "hest")

    ri_gain_abs = ri_gain if ri_gain is not None else None
    hest_gain = _waiv_gain(base_hest, waiv_dict.get("hest"))
    hest_gain_abs = hest_gain if hest_gain is not None else None

    r["ri_noise_pct_of_gain"] = None
    if sd_ri is not None and ri_gain_abs is not None and abs(ri_gain_abs) > 1e-10:
        r["ri_noise_pct_of_gain"] = (2 * sd_ri) / abs(ri_gain_abs) * 100.0

    r["hest_noise_pct_of_gain"] = None
    if sd_hest is not None and hest_gain_abs is not None and abs(hest_gain_abs) > 1e-10:
        r["hest_noise_pct_of_gain"] = (2 * sd_hest) / abs(hest_gain_abs) * 100.0

    r["ri_unresolvable"] = (r["ri_noise_pct_of_gain"] is not None and r["ri_noise_pct_of_gain"] > 20.0)
    r["hest_unresolvable"] = (r["hest_noise_pct_of_gain"] is not None and r["hest_noise_pct_of_gain"] > 20.0)

    # 1 seed-SD expressed in pct_of_waiv points (half the stored 2SD-of-gain figure).
    r["ri_sd_pct"] = (r["ri_noise_pct_of_gain"] / 2.0) if r["ri_noise_pct_of_gain"] is not None else None
    r["hest_sd_pct"] = (r["hest_noise_pct_of_gain"] / 2.0) if r["hest_noise_pct_of_gain"] is not None else None

    # UNRESOLVABLE-BY-CONSTRUCTION: withhold the pct, publish raw delta + CI instead.
    # The cell is NEVER silently dropped -- it prints as UNRES and is listed in the
    # raw-delta report -- but it is excluded from avg_pct_of_waiv, because averaging a
    # number we just declared unmeasurable would launder it back into the headline.
    for metric, base_v, sd_v in (("ri", base_ri, sd_ri), ("hest", base_hest, sd_hest)):
        sd_pct = r.get("%s_sd_pct" % metric)
        ours_v = r.get(metric)
        r["%s_by_construction" % metric] = False
        r["%s_raw_delta" % metric] = (ours_v - base_v) if (ours_v is not None and base_v is not None) else None
        r["%s_raw_ci95" % metric] = (1.96 * sd_v) if sd_v is not None else None
        unres_bc, _sdp, _why = _ec.denominator_unresolvable(
            _waiv_gain(base_v, (waiv_dict or {}).get(metric)), sd_v)
        if unres_bc and sd_pct is not None:
            r["%s_by_construction" % metric] = True
            r["%s_pct_withheld" % metric] = r.get("%s_pct" % metric)
            r["%s_pct" % metric] = None
            r["%s_pct_guard" % metric] = "unresolvable_by_construction"

    return r


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_runs(runs_dir, filter_names=None, filter_arm=None):
    if not runs_dir.exists():
        return []
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if re.search(r"\.r\d+$", name):
            continue
        if not (d / "ri_curve.json").exists():
            continue
        if filter_names and name not in filter_names:
            continue
        meta = parse_run_meta(d)
        if filter_arm and meta.get("arm") != filter_arm:
            continue
        runs.append(meta)
    return runs


# ── Display ──────────────────────────────────────────────────────────────────

_W = 132

def _hr(char="-"):
    return char * _W


def _pct_str(pct, guard, width=7, uncapped=None):
    if pct is not None and uncapped is not None and uncapped > pct + 1e-9:
        # capped at 100 -- trailing '*' marks that the raw value was higher
        return ("%.0f*" % pct).rjust(width)
    if pct is None:
        if guard == "no_ours":
            return ("MISSING").ljust(width)
        if guard == "base>=Waiv":
            return ("n/a").ljust(width)
        if guard == "unresolvable_by_construction":
            return ("UNRES").rjust(width)
        return ("N/A").ljust(width)
    return ("%.1f" % pct).rjust(width)


def print_denominators():
    print("\n%s" % _hr("="))
    print("  DENOMINATORS (auditable base/Waiv values per backbone)")
    print("%s" % _hr("="))
    hdr = "  %-12s %8s %8s %10s %14s %9s %9s %10s" % (
        "backbone", "base_RI", "Waiv_RI", "gain_RI", "RI_floor(80pct)",
        "base_HEST", "Waiv_HEST", "gain_HEST")
    print(hdr)
    for arm in ARMS:
        b_ri = RI_BASE.get(arm)
        w_ri = WAIV[arm].get("ri")
        b_h = HEST_BASE.get(arm)
        w_h = WAIV[arm].get("hest")
        ri_gain = _waiv_gain(b_ri, w_ri)
        h_gain = _waiv_gain(b_h, w_h)
        ri_floor = (b_ri + 0.80 * ri_gain) if ri_gain is not None else None
        ri_floor_str = "%.5f" % ri_floor if ri_floor is not None else "N/A"
        print("  %-12s %8.5f %8.3f %+10.5f %14s %9.5f %9.4f %+10.5f" % (
            arm, b_ri, w_ri, ri_gain, ri_floor_str, b_h, w_h, h_gain))
    print()
    print("  THUNDER denominators -- %s" % WAIV_THUNDER_SOURCE)
    print("  GAIN-RATIO form: (ours - OUR_base) / (Waiv_ft - Waiv_base).  Our THUNDER base does NOT")
    print("  reproduce Waiv's base (cls ~2-4pp low, seg ~4pp high on 2 datasets vs their 4), so only")
    print("  the GAIN is comparable -- never the level.  Our base reproduces THUNDER's own paper.")
    print("  2SE = measured seed floor from docs/thunder_seed_floor_12ds.md (offset-2SE of the")
    print("  12-dataset PAPER_CLS task mean, n=5 final5 training seeds, keyed by backbone).")
    print("  %-12s %-16s %9s %9s %9s %9s %8s %6s" % (
        "backbone", "task", "our_base", "Waiv_base", "Waiv_ft", "Waiv_gain", "2SE/gain", "flag"))
    for arm in ARMS:
        wt = WAIV_THUNDER.get(arm, {})
        ob = _c5._thunder_base_score(arm)
        for task in THUNDER_TASKS:
            wb = wt.get("base", {}).get(task)
            wf = wt.get("ft", {}).get(task)
            wb_f = wb / 100.0 if wb is not None else None
            wf_f = wf / 100.0 if wf is not None else None
            gain = (wf_f - wb_f) if (wb_f is not None and wf_f is not None) else None
            unres, npct, floor_unm = _thunder_unresolvable(gain, arm, task)
            flags = []
            if floor_unm:
                flags.append("UNMEASURED_FLOOR")
            if gain is not None and gain < 0:
                flags.append("WAIV-REGRESSED")
            if unres:
                flags.append("UNRESOLVABLE")
            if task in THUNDER_SUPPORT_MISMATCH:
                flags.append("support_2v4")
            ob_v = ob.get(task)
            print("  %-12s %-16s %9s %9s %9s %9s %8s %s" % (
                arm, task,
                "%.5f" % ob_v if ob_v is not None else "MISSING",
                "%.5f" % wb_f if wb_f is not None else "N/A",
                "%.5f" % wf_f if wf_f is not None else "N/A",
                "%+.5f" % gain if gain is not None else "N/A",
                "%.0f%%" % npct if npct is not None else "inf",
                ",".join(flags)))
    print()
    print("  Per-dataset RI base values not stored separately; RI pct uses backbone avg base.")


def print_raw_delta_report(scored):
    """Cells whose pct_of_waiv was WITHHELD as unresolvable-by-construction.

    These are reported the only honest way available: as a raw delta vs base with a
    95% CI from the measured seed-SD, alongside the denominator that makes the
    percentage form meaningless.  Read the CI, not a point estimate.
    """
    print("\n%s" % _hr("="))
    print("  UNRESOLVABLE-BY-CONSTRUCTION -- pct_of_waiv WITHHELD, raw delta shown")
    print("%s" % _hr("="))
    print("  Trigger: one seed-SD > %.0f pct_of_waiv points (identical to 2SD > 20%% of gain)." % UNRESOLVABLE_SD_PCT_LIMIT)
    print("  A percentage here divides by a Waiv gain so small that ordinary seed jitter")
    print("  swings the score by tens of points.  The raw delta is the measurement; the")
    print("  percentage is an artefact of the denominator.  Cells are NOT dropped from the")
    print("  scoreboard -- they print as UNRES -- but they are excluded from avg_pct_of_waiv.")
    print()
    hdr = "  %-34s %-9s %-6s %9s %11s %11s %8s %7s" % (
        "run", "arm", "metric", "ours", "raw_delta", "+/-95%CI", "denom", "1SD_pct")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    any_row = False
    for r in scored:
        for metric in ("ri", "hest"):
            if not r.get("%s_by_construction" % metric):
                continue
            if r.get(metric) is None:
                continue      # metric not measured for this checkpoint -- RULE 1, no substitution
            any_row = True
            arm = r.get("arm")
            base_v = (RI_BASE if metric == "ri" else HEST_BASE).get(arm)
            waiv_v = WAIV.get(arm, {}).get(metric)
            denom = _waiv_gain(base_v, waiv_v)
            d = r.get("%s_raw_delta" % metric)
            ci = r.get("%s_raw_ci95" % metric)
            sig = ""
            if d is not None and ci is not None:
                sig = "yes" if abs(d) > ci else "NO"
            print("  %-34s %-9s %-6s %9s %11s %11s %8s %7s   sig>CI:%s" % (
                (r.get("run_name") or "?")[:34], arm or "?", metric.upper(),
                "%.5f" % r.get(metric) if r.get(metric) is not None else "MISSING",
                "%+.5f" % d if d is not None else "N/A",
                "%.5f" % ci if ci is not None else "N/A",
                "%.5f" % denom if denom is not None else "N/A",
                "%.1f" % r.get("%s_sd_pct" % metric) if r.get("%s_sd_pct" % metric) is not None else "N/A",
                sig))
    if not any_row:
        print("  (none -- every cell has a denominator large enough to carry a percentage)")
    print()


def print_unresolvable_report(scored):
    print("\n%s" % _hr("="))
    print("  UNRESOLVABLE METRICS (2*seed_SD > 20% of Waiv's gain)")
    print("%s" % _hr("="))
    print("  A metric is UNRESOLVABLE when noise alone spans more than 20% of Waiv's gain.")
    print("  This means 80% vs 100% CANNOT be distinguished -- any claim is within noise.")
    print()

    seen = set()
    for r in scored:
        arm = r.get("arm")
        step = r.get("step")
        key = (arm, step)
        if key in seen:
            continue
        seen.add(key)

        ri_un = r.get("ri_unresolvable", False)
        h_un = r.get("hest_unresolvable", False)
        ri_n_pct = r.get("ri_noise_pct_of_gain")
        h_n_pct = r.get("hest_noise_pct_of_gain")

        if not ri_un and not h_un:
            continue

        issues = []
        if ri_un:
            if ri_n_pct:
                issues.append("RI: 2SD=%.1f%% of gain -> UNRESOLVABLE" % ri_n_pct)
            else:
                issues.append("RI: UNRESOLVABLE")
        if h_un:
            if h_n_pct:
                issues.append("HEST: 2SD=%.1f%% of gain -> UNRESOLVABLE" % h_n_pct)
            else:
                issues.append("HEST: UNRESOLVABLE")

        print("  %s step=%d: %s" % (arm, step, "; ".join(issues)))

    print()
    print("  THUNDER (backbone-level, independent of run -- from Waiv's published gain):")
    for arm in ARMS:
        wt = WAIV_THUNDER.get(arm, {})
        bad = []
        for task in THUNDER_TASKS:
            wb, wf = wt.get("base", {}).get(task), wt.get("ft", {}).get(task)
            if wb is None or wf is None:
                continue
            gain = (wf - wb) / 100.0
            unres, npct, floor_unm = _thunder_unresolvable(gain, arm, task)
            if gain < 0:
                bad.append("%s: Waiv REGRESSED (%+.1fpp) -> no denominator" % (task, gain * 100))
            elif floor_unm:
                bad.append("%s: no measured seed floor -> UNMEASURED_FLOOR "
                           "(cannot pass or fail)" % task)
            elif unres:
                bad.append("%s: 2SE=%.5f is %.0f%% of gain -> UNRESOLVABLE"
                           % (task, _thunder_2se(arm, task), npct))
        if bad:
            print("    %-10s %s" % (arm, "; ".join(bad)))
    print()
    print("  GUARDED CELLS (pct_of_waiv = N/A):")
    # Count guarded RI/HEST cells
    ri_guards = [(r["run_name"], r.get("ri_pct_guard")) for r in scored if r.get("ri_pct_guard")]
    hest_guards = [(r["run_name"], r.get("hest_pct_guard")) for r in scored if r.get("hest_pct_guard")]
    if ri_guards:
        guards = set(g for _, g in ri_guards if g)
        print("    RI: %d runs with guards: %s" % (len(ri_guards), list(guards)[:3]))
    if hest_guards:
        guards = set(g for _, g in hest_guards if g)
        print("    HEST: %d runs with guards: %s" % (len(hest_guards), list(guards)[:3]))


def _mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _cell_verdict(rows):
    """Collapse the runs of one (recipe, backbone) cell into a verdict record.

    Returns a dict with the three CAPPED pcts, min, mean, and status in
    {PASS, FAIL, INDETERMINATE}.  A metric that is missing, guarded,
    unresolvable-by-construction, or (THUNDER) has no measured seed floor makes the
    whole cell INDETERMINATE -- it can never be PASS.
    """
    reasons = []

    # F2 fix (2026-08-26): minimum-n gate.  This function previously compared a bare
    # point estimate against the 70 bar with NO error bar at all, so an n=1 cell could
    # print PASS.  `n` was computed and then never used.  Now it gates the verdict.
    if len(rows) < MIN_N_FOR_VERDICT:
        return {"pcts": {"RI": None, "HEST": None, "THUNDER": None},
                "min": None, "mean": None, "status": "UNDERPOWERED",
                "reasons": ["n=%d run(s) < MIN_N_FOR_VERDICT=%d; no error bar is "
                            "derivable, so no PASS/FAIL is defensible"
                            % (len(rows), MIN_N_FOR_VERDICT)],
                "n": len(rows)}

    def _collect(key, guard_key, label):
        vals, bad = [], []
        for r in rows:
            v = r.get(key)
            if v is None:
                bad.append(r.get(guard_key) or "missing")
            else:
                vals.append(v)
        if not vals:
            reasons.append("%s: %s" % (label, ",".join(sorted(set(bad))) or "missing"))
            return None
        return sum(vals) / len(vals)

    ri = _collect("ri_pct", "ri_pct_guard", "RI")
    hest = _collect("hest_pct", "hest_pct_guard", "HEST")

    th_vals = [r.get("thunder_verdict_pct") for r in rows]
    th = _mean_or_none(th_vals)
    if th is None:
        skipped = []
        for r in rows:
            skipped.extend(r.get("thunder_verdict_skipped") or [])
        reasons.append("THUNDER: %s" % (",".join(sorted(set(skipped))) or "missing"))
    else:
        # Even when SOME tasks are usable, name the ones that were excluded, so the
        # number is never mistaken for a verdict over all four protocols.
        skipped = []
        for r in rows:
            skipped.extend(r.get("thunder_verdict_skipped") or [])
        if skipped:
            reasons.append("THUNDER excluded: %s" % ",".join(sorted(set(skipped))))

    pcts = {"RI": ri, "HEST": hest, "THUNDER": th}
    if any(v is None for v in pcts.values()):
        return {"pcts": pcts, "min": None, "mean": None,
                "status": "INDETERMINATE", "reasons": reasons, "n": len(rows)}

    mn = min(pcts.values())
    mean = sum(pcts.values()) / 3.0
    ok = (mn >= VERDICT_MIN_PCT) and (mean > VERDICT_MEAN_PCT)
    return {"pcts": pcts, "min": mn, "mean": mean,
            "status": "PASS" if ok else "FAIL", "reasons": reasons, "n": len(rows)}


_VERDICT_RANK = {"PASS": 0, "FAIL": 1, "INDETERMINATE": 2, "UNDERPOWERED": 3}


def verdict_report(scored, step):
    """CURRENT pass criterion, per backbone and rolled up per recipe.

        per backbone: EVERY metric (RI, HEST, THUNDER) >= 70% of Waiv's gain
                      AND mean of the three > 80%
        per recipe:   the WORST backbone decides.

    Percentages are CAPPED at 100 (beating Waiv scores 100, not more), so an
    overshoot on one metric can never mask a shortfall on another.  Any cell that is
    UNMEASURED_FLOOR or unresolvable forces INDETERMINATE -- never PASS.
    """
    print("\n%s" % _hr("="))
    print("  VERDICT -- per backbone: min(RI,HEST,THUNDER) >= %.0f%% AND mean > %.0f%%"
          % (VERDICT_MIN_PCT, VERDICT_MEAN_PCT))
    print("            per recipe:   scored by WORST backbone")
    print("%s" % _hr("="))
    print("  pcts are CAPPED at 100.  THUNDER uses only protocols with a MEASURED seed")
    print("  floor -- knn, linear_probing AND simple_shot, from the 12-dataset floors in")
    print("  docs/thunder_seed_floor_12ds.md (keyed by backbone, not pooling).  Segmentation")
    print("  still has none and is reported as UNMEASURED_FLOOR.  A cell whose THUNDER sweep")
    print("  is incomplete (< 12/12 datasets) is PARTIAL_COVERAGE: the 12-dataset floor")
    print("  understates the noise of a shorter mean, so it is excluded, not graded with it.")
    print("  Any UNMEASURED_FLOOR, PARTIAL_COVERAGE or unresolvable cell -> INDETERMINATE.")
    print()

    recipes = {}
    for r in scored:
        key = r.get("desc") or r.get("run_name")
        recipes.setdefault(key, {}).setdefault(r.get("arm"), []).append(r)

    hdr = "  %-40s %-10s %3s %8s %8s %8s %8s %8s  %s" % (
        "recipe", "backbone", "n", "RI", "HEST", "THUNDER", "min", "mean", "verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for recipe in sorted(recipes):
        per_bb = {}
        for arm in ARMS + (None,):
            rows = recipes[recipe].get(arm)
            if not rows:
                continue
            cv = _cell_verdict(rows)
            per_bb[arm] = cv
            f = lambda v: ("%.1f" % v) if v is not None else "N/A"
            print("  %-40s %-10s %3d %8s %8s %8s %8s %8s  %s" % (
                recipe[:40], str(arm)[:10], cv["n"],
                f(cv["pcts"]["RI"]), f(cv["pcts"]["HEST"]), f(cv["pcts"]["THUNDER"]),
                f(cv["min"]), f(cv["mean"]), cv["status"]))
            for reason in cv["reasons"]:
                print("  %-40s %-10s     -> %s" % ("", "", reason))

        if not per_bb:
            continue
        # WORST backbone: most severe status first (INDETERMINATE > FAIL > PASS),
        # tie-broken by the lowest min pct.
        worst_arm = sorted(per_bb, key=lambda a: (
            -_VERDICT_RANK[per_bb[a]["status"]],
            (per_bb[a]["min"] if per_bb[a]["min"] is not None else -1e9)))[0]
        worst = per_bb[worst_arm]
        print("  %-40s %-10s %s (worst backbone: %s)" % (
            "", ">> RECIPE", worst["status"], worst_arm))
        print()

    if not recipes:
        print("  (no rows)")
    print()


def print_summary_table(scored):
    hdr = "  %-40s %-9s %-5s %-7s %-7s %-7s %-7s %-6s %-10s %-7s %-7s %-5s" % (
        "Run", "BB", "@", "RI", "RI_pct", "noise", "HEST", "H_pct", "THmean", "TH_pct", "avgPct", "RI-fl")
    print("\n%s" % _hr("="))
    print("  SCOREBOARD -- pct_of_waiv headline | Rule 1: one checkpoint/row | Rule 2: ours|Waiv|diff")
    print("%s" % _hr("="))
    print(hdr)
    print("%s" % _hr("-"))

    for r in scored:
        arm = r.get("arm") or "?"
        step = r.get("step", "?")
        ri = r.get("ri")
        hest = r.get("hest")

        ri_s = "%.4f" % ri if ri is not None else "MISSING"
        ri_pct_s = _pct_str(r.get("ri_pct"), r.get("ri_pct_guard"), 7, r.get("ri_pct_uncapped"))

        # Noise annotation on RI vs Waiv
        sd_ri = _sd_for(arm, step, "ri")
        w_ri = WAIV.get(arm, {}).get("ri")
        ri_noise = ""
        if ri is not None and w_ri is not None:
            ri_noise = _noise_tag(abs(ri - w_ri), sd_ri)

        hest_s = "%.4f" % hest if hest is not None else "MISSING"
        hest_pct_s = _pct_str(r.get("hest_pct"), r.get("hest_pct_guard"), 6, r.get("hest_pct_uncapped"))

        # THUNDER mean
        thunder = r.get("thunder", {})
        t_means = [
            thunder[t]["mean"] for t in THUNDER_TASKS
            if not thunder.get(t, {}).get("partial", True)
            and thunder.get(t, {}).get("mean") is not None
        ]
        if len(t_means) == len(THUNDER_TASKS):
            t_s = "%.4f" % (sum(t_means) / len(t_means))
        elif len(t_means) > 0:
            t_s = "P%d/%d" % (len(t_means), len(THUNDER_TASKS))
        else:
            t_s = "MISSING"

        avg_pct = r.get("avg_pct_of_waiv")
        avg_s = "%.1f" % avg_pct if avg_pct is not None else "N/A"

        bud = r.get("ri_budget", "N/A")

        t_pct_s = _pct_str(r.get("thunder_mean_pct"), r.get("thunder_mean_pct_guard"), 7,
                           r.get("thunder_mean_pct_uncapped"))
        if r.get("thunder_mean_pct") is not None and r.get("thunder_mean_unmeasured_floor"):
            # never render a floorless THUNDER mean as a clean pct
            t_pct_s = "UNMEAS".rjust(7)
        elif r.get("thunder_mean_pct") is not None and r.get("thunder_mean_unresolvable"):
            t_pct_s = ("%.1f!" % r["thunder_mean_pct"]).rjust(7)

        print("  %-40s %-9s %-5s %-7s %-7s %-7s %-7s %-6s %-10s %-7s %-7s %-5s" % (
            r["run_name"][:40], arm[:9], str(step), ri_s, ri_pct_s, ri_noise[:7],
            hest_s, hest_pct_s, t_s, t_pct_s, avg_s, bud))


def print_run_block(r):
    arm = r.get("arm") or "?"
    step = r.get("step")
    waiv_dict = WAIV.get(arm, {})
    base_ri = RI_BASE.get(arm)
    base_hest = HEST_BASE.get(arm)

    print("%s" % _hr())
    print("  %s" % r["run_name"])
    print("  backbone=%s  seed=%s  T=%s  jobid=%s  step=%s" % (
        arm, r.get("seed", "?"), r.get("T", "?"), r.get("jobid", "?"), step))
    print("  desc: %s" % r.get("desc", ""))

    note = r.get("step_missing_note", "")
    if note:
        print("  ! %s" % note)

    # RI
    ri = r.get("ri")
    ri_un = r.get("ri_unresolvable", False)
    ri_pct = r.get("ri_pct")
    ri_guard = r.get("ri_pct_guard")
    sd_ri = _sd_for(arm, step, "ri")

    if ri is not None:
        w_ri = waiv_dict.get("ri")
        diff_w = ri - w_ri if w_ri is not None else None
        diff_b = ri - base_ri if base_ri is not None else None
        noise = _noise_tag(abs(diff_w), sd_ri) if diff_w is not None else ""
        pct_s = "%.1f%% of Waiv gain" % ri_pct if ri_pct is not None else "N/A (%s)" % (ri_guard or "")
        unres_str = "  *** UNRESOLVABLE ***" if ri_un else ""
        print("  RI: %s | Waiv %s | diff %s %s  %sbase %s  [%s]%s" % (
            _fmt(ri, "%.5f"), _fmt(w_ri, "%.3f"), _fmt(diff_w), noise, "",
            _fmt(diff_b), pct_s, unres_str))
        waiv_ds = waiv_dict.get("ri_ds", {})
        for ds in ("tcga", "camelyon", "tolkach_esca"):
            val = r.get("ri_ds", {}).get(ds)
            wv = waiv_ds.get(ds)
            if val is not None:
                dws = val - wv if wv is not None else None
                print("    RI.-%12s  %s | Waiv %s | diff %s" % (
                    ds, _fmt(val, "%.5f"), _fmt(wv, "%.3f"), _fmt(dws)))
            else:
                print("    RI.-%12s  MISSING" % ds)
    else:
        print("  RI: MISSING")

    # HEST
    hest = r.get("hest")
    hest_un = r.get("hest_unresolvable", False)
    hest_pct = r.get("hest_pct")
    hest_guard = r.get("hest_pct_guard")
    sd_hest = _sd_for(arm, step, "hest")

    if hest is not None:
        w_h = waiv_dict.get("hest")
        diff_w = hest - w_h if w_h is not None else None
        diff_b = hest - base_hest if base_hest is not None else None
        noise = _noise_tag(abs(diff_w), sd_hest) if diff_w is not None else ""
        pct_s = "%.1f%% of Waiv gain" % hest_pct if hest_pct is not None else "N/A (%s)" % (hest_guard or "")
        unres_str = "  *** UNRESOLVABLE ***" if hest_un else ""
        print("  HEST: %s | Waiv %s | diff %s %s  %sbase %s  [%s]%s" % (
            _fmt(hest, "%.5f"), _fmt(w_h, "%.4f"), _fmt(diff_w), noise, "",
            _fmt(diff_b), pct_s, unres_str))
    else:
        print("  HEST: MISSING")

    # THUNDER
    thunder = r.get("thunder", {})
    base_thunder = r.get("base_thunder", {})
    th_pct = r.get("thunder_pct", {})
    print("  THUNDER  [pct_of_waiv = (ours - OUR_base) / (Waiv_ft - Waiv_base); %s]" % WAIV_THUNDER_SOURCE)
    task_means = []
    all_complete = True
    for task in THUNDER_TASKS:
        td = thunder.get(task, {"mean": None, "n": 0, "total": 0, "partial": True, "per_ds": {}})
        base_mean = base_thunder.get(task)
        n, total = td["n"], td["total"]
        if n == 0:
            print("    %-18s  MISSING (0/%d)" % (task, total))
            all_complete = False
        elif td["partial"]:
            pm = sum(td["per_ds"].values()) / n
            # A partial mean has a DIFFERENT (larger) noise level than the
            # 12-dataset floor describes, so no floor is quoted and no verdict
            # is possible -- see PARTIAL_COVERAGE in score_run().
            print("    %-18s  PARTIAL(%d/%d) partial-mean=%.5f  "
                  "*** PARTIAL_COVERAGE -- 12ds seed floor does not apply, ungradeable ***"
                  % (task, n, total, pm))
            all_complete = False
        else:
            mean = td["mean"]
            delta = mean - base_mean if base_mean is not None else None
            t_floor = _thunder_2se(arm, task)
            if delta is None:
                noise = ""
            elif t_floor is None:
                noise = "[floor?]"   # UNMEASURED_FLOOR: no seed floor for this protocol
            else:
                noise = _noise_tag(delta if delta >= 0 else -delta, t_floor)
            ds_str = "  %sbase %+.5f %s" % ("", delta, noise) if delta is not None else ""
            tp = th_pct.get(task, {})
            wf, gain = tp.get("waiv_ft"), tp.get("waiv_gain")
            if tp.get("pct") is not None:
                pct_s = "%.1f%% of Waiv gain" % tp["pct"]
                if tp.get("pct_uncapped") is not None and tp["pct_uncapped"] > tp["pct"] + 1e-9:
                    pct_s += " (capped; raw %.1f%%)" % tp["pct_uncapped"]
                if tp.get("unmeasured_floor"):
                    pct_s += " *** UNMEASURED_FLOOR -- no pass/fail ***"
                elif tp.get("unresolvable"):
                    pct_s += " *** UNRESOLVABLE (2SE=%.0f%% of gain) ***" % tp["noise_pct_of_gain"]
            else:
                pct_s = "N/A (%s)" % (tp.get("guard") or "")
            wv_s = " | Waiv %.5f (gain %+.5f)" % (wf, gain) if wf is not None and gain is not None else ""
            sup = "  ^support_2v4" if tp.get("support_mismatch") else ""
            print("    %-18s  %.5f (%d/%d)%s%s  [%s]%s" % (
                task, mean, n, total, ds_str, wv_s, pct_s, sup))
            task_means.append(mean)

    if all_complete and len(task_means) == len(THUNDER_TASKS):
        t_mean = sum(task_means) / len(task_means)
        base_means = [v for v in base_thunder.values() if v is not None]
        base_t = sum(base_means) / len(base_means) if base_means else None
        base_str = "  %sbase %+.5f" % ("", t_mean - base_t) if base_t is not None else ""
        tmp = r.get("thunder_mean_pct")
        if tmp is not None:
            pct_s = "%.1f%% of Waiv gain" % tmp
            if r.get("thunder_mean_pct_uncapped") is not None and \
                    r["thunder_mean_pct_uncapped"] > tmp + 1e-9:
                pct_s += " (capped; raw %.1f%%)" % r["thunder_mean_pct_uncapped"]
            if r.get("thunder_mean_unmeasured_floor"):
                pct_s += " *** UNMEASURED_FLOOR (%s) -- no pass/fail ***" % (
                    ",".join(r.get("thunder_mean_unmeasured_tasks") or []) or "no floor")
            elif r.get("thunder_mean_unresolvable"):
                pct_s += " *** UNRESOLVABLE ***"
        else:
            pct_s = "N/A (%s)" % (r.get("thunder_mean_pct_guard") or "")
        print("    %-18s  %.5f%s  [%s]  ^contains segmentation (support_2v4)" % (
            "task-mean", t_mean, base_str, pct_s))
    else:
        n_comp = sum(1 for t in THUNDER_TASKS
                     if not thunder.get(t, {}).get("partial", True)
                     and thunder.get(t, {}).get("mean") is not None)
        print("    %-18s  PARTIAL(%d/%d protocols)" % ("task-mean", n_comp, len(THUNDER_TASKS)))

    # RI floor
    ri_floor = r.get("ri_floor")
    bud = r.get("ri_budget", "N/A")
    floor_str = "%.5f" % ri_floor if ri_floor is not None else "N/A"
    gap = ""
    if ri is not None and ri_floor is not None:
        g = ri - ri_floor
        gap = "  (gap=%+.5f)" % g
    print("  RI-floor: %s (base + 80%% of Waiv gain)  %s%s" % (floor_str, bud, gap))

    # avg pct_of_waiv
    avg = r.get("avg_pct_of_waiv")
    avg_s = "%.1f%%" % avg if avg is not None else "N/A"
    print("  avg_pct_of_waiv (RI + HEST + THUNDER task-mean): %s" % avg_s)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--step", type=int, default=500,
                    help="Checkpoint step (default 500)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Explicit run dir names")
    ap.add_argument("--only-arm", choices=ARMS, default=None,
                    help="Restrict to one backbone")
    ap.add_argument("--only-complete", action="store_true",
                    help="Hide rows where RI or HEST is MISSING")
    ap.add_argument("--sort-by",
                    choices=["avg_pct", "ri", "ri_pct", "hest", "hest_pct", "thunder", "name"],
                    default="avg_pct",
                    help="Sort key (default: avg_pct = average pct_of_waiv across metrics)")
    ap.add_argument("--runs-dir", default=None,
                    help="Override runs directory")
    ap.add_argument("--no-detail", action="store_true",
                    help="Summary table only")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else _DEFAULT_RUNS
    if not runs_dir.exists():
        print("[ERROR] runs directory not found: %s" % runs_dir, file=sys.stderr)
        sys.exit(1)

    print("scoreboard v3  |  step=%d  |  runs_dir=%s" % (args.step, runs_dir))
    print("RULE 1: single (run_name, step) per row -- MISSING if absent")
    print("RULE 2: ours | Waiv | diff visible; pct_of_waiv headline")
    print("FIX: virchow2 base_hest = 0.40324 (was 0.4034)")
    print()

    metas = discover_runs(runs_dir, filter_names=args.runs, filter_arm=args.only_arm)
    print("Discovered %d run(s) with ri_curve.json" % len(metas))
    scored = [score_run(m, args.step) for m in metas]

    if args.only_complete:
        before = len(scored)
        scored = [r for r in scored if r.get("ri") is not None and r.get("hest") is not None]
        print("--only-complete: kept %d/%d" % (len(scored), before))

    # Sort
    def _sort_key(r):
        sb = args.sort_by
        if sb == "avg_pct":
            return -(r.get("avg_pct_of_waiv") or -999)
        if sb == "ri":
            return -(r.get("ri") or -9)
        if sb == "ri_pct":
            return -(r.get("ri_pct") or -999)
        if sb == "hest":
            return -(r.get("hest") or -9)
        if sb == "hest_pct":
            return -(r.get("hest_pct") or -999)
        if sb == "thunder":
            td = r.get("thunder", {})
            ms = [td[t]["mean"] for t in THUNDER_TASKS
                  if td.get(t, {}).get("mean") is not None and not td[t]["partial"]]
            return -(sum(ms) / len(ms) if ms else -9)
        return (r.get("arm") or "z", r["run_name"])

    scored.sort(key=_sort_key)

    print_summary_table(scored)
    verdict_report(scored, args.step)
    print_denominators()
    print_unresolvable_report(scored)
    print_raw_delta_report(scored)

    if not args.no_detail:
        for arm in ARMS:
            arm_runs = [r for r in scored if r.get("arm") == arm]
            if not arm_runs:
                continue
            print("\n%s" % _hr("="))
            print("  DETAIL: %s  (%d runs)" % (arm, len(arm_runs)))
            print("%s" % _hr("="))
            for r in arm_runs:
                print_run_block(r)

        unknown = [r for r in scored if r.get("arm") not in ARMS]
        if unknown:
            print("\n%s" % _hr("="))
            print("  UNKNOWN BACKBONE (%d runs)" % len(unknown))
            print("%s" % _hr("="))
            for r in unknown:
                print_run_block(r)

    print("\n%s" % _hr("="))
    print("Total rows: %d" % len(scored))
    print("pct_of_waiv = (ours - base) / (Waiv - base) * 100, CAPPED at 100")
    print("PASS (per backbone): every metric >= %.0f%% AND mean of the three > %.0f%%"
          % (VERDICT_MIN_PCT, VERDICT_MEAN_PCT))
    print("PASS (per recipe):   scored by the WORST backbone")
    print("RI floor = base + 0.80 * (Waiv - base)   [legacy row, not the verdict]")
    print("UNRESOLVABLE: 2*seed_SD > 20% of Waiv's gain")
    print("THUNDER seed floors: docs/thunder_seed_floor_12ds.md -- offset-2SE of the")
    print("                  12-dataset PAPER_CLS task mean, n=5 final5 training seeds,")
    print("                  keyed by (backbone, task).  Supersedes the 5-dataset n=2 floor.")
    print("UNMEASURED_FLOOR: no measured seed floor (THUNDER segmentation only)")
    print("                  -> INDETERMINATE, never PASS")
    print("PARTIAL_COVERAGE: THUNDER sweep < 12/12 datasets -- the 12ds floor is invalid for")
    print("                  a shorter, noisier mean -> excluded from the verdict, never PASS")


if __name__ == "__main__":
    main()
