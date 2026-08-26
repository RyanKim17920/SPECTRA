#!/usr/bin/env python3
"""ONE definition of every quantity the three verdict/reporting scripts share.

Created 2026-08-26 by the formula-unification audit (docs/FORMULA_UNIFICATION_2026-08-26.md).

Before this module existed the repo carried FIVE independent transcriptions of the same
few numbers (RI base, RI/HEST Waiv targets, the HEST base, the resolvability limit) and
THREE different constructions of a 95% error bar.  Numbers that are supposed to be the
same quantity disagreed by up to 32%, and two scripts printed different statuses for the
same cell.  Everything in here is either (a) read from disk, or (b) a transcription from
a named published table with exactly one owner in the repo.

WHAT IS ALLOWED TO BE A LITERAL HERE
  * A path or a directory name (where a measurement lives).
  * A number transcribed from a paper we cannot recompute (Waiv's published Table 1/2).
    Those carry a `_SOURCE` string and exist in NO other file.
Everything else -- every measured quantity, every noise floor -- is loaded.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# The ONE denominator-resolvability gate  (F-B)
# ---------------------------------------------------------------------------
# A pct_of_waiv cell divides by (waiv - base).  When that denominator is itself within
# the instrument's seed noise, the ratio is not a measurement of anything: matching a
# regression scores 100%, and 80% cannot be told from 100%.  The cell is then WITHHELD.
#
# The test, algebraically identical in both of its historical spellings:
#       2 * seed_SD > 20% of |waiv gain|      <=>      seed_SD / |gain| * 100 > 10
#
# THRESHOLD PROVENANCE: scoreboard.py has applied `sd_pct > 10` since the resolvability
# audit; the 20%-of-gain spelling is the one scoreboard._thunder_unresolvable applies to
# THUNDER.  This module is now the only definition; scoreboard imports it.
#
# n-INDEPENDENT ON PURPOSE.  This is a property of the COMPARISON (is Waiv's own gain
# bigger than the noise of the instrument that measured it), not of how many seeds WE
# ran.  Running more of our own seeds shrinks OUR error bar -- that is the separate
# CI-vs-70-bar test -- but it cannot sharpen a denominator that was never resolved.
UNRESOLVABLE_SD_PCT_LIMIT = 10.0

#: The noise quantity every benchmark must supply to the gate: the across-SEED SD, in
#: RAW metric units, of the exact statistic that is being divided.  RI and HEST supply a
#: per-(backbone, step) pooled seed SD; THUNDER supplies seed_sd_of_task_mean.  Supplying
#: a different variance component (e.g. an SD taken over DATASETS) makes the three
#: benchmarks incomparable again and is the bug this module exists to prevent.
DENOMINATOR_GATE_NOISE = "across-seed SD (1 SD, one run) of the reported statistic, raw units"


def denominator_sd_pct(gain: float | None, seed_sd: float | None) -> float | None:
    """One seed-SD expressed in pct_of_waiv POINTS, or None when it cannot be formed."""
    if gain is None or seed_sd is None or abs(gain) < 1e-12:
        return None
    return seed_sd / abs(gain) * 100.0


def denominator_unresolvable(gain: float | None, seed_sd: float | None):
    """(unresolvable, sd_pct, reason).  The ONLY implementation in the repo.

    unresolvable is True when the gate cannot be formed at all (no measured seed SD, or
    a denominator of zero) as well as when it trips -- "we cannot tell" is never
    silently rendered as "resolved".
    """
    if seed_sd is None:
        return True, None, "no measured seed SD for this cell -- resolvability untestable"
    if gain is None or abs(gain) < 1e-12:
        return True, None, "waiv gain is zero/undefined -- denominator is not a scale"
    sd_pct = seed_sd / abs(gain) * 100.0
    if sd_pct > UNRESOLVABLE_SD_PCT_LIMIT:
        return True, sd_pct, (
            "one seed-SD is %.1f pct_of_waiv points (> %.0f); waiv gain %+.4f is within "
            "%.1fx the instrument's seed noise %.4f -- denominator is noise"
            % (sd_pct, UNRESOLVABLE_SD_PCT_LIMIT, gain, abs(gain) / seed_sd, seed_sd))
    return False, sd_pct, None


# ---------------------------------------------------------------------------
# The ONE 95% CI construction  (F-D)
# ---------------------------------------------------------------------------
def ci95(pcts: list[float], floor_sd_pct: float | None):
    """max(empirical 2*SD/sqrt(n), instrument floor 2*SD/sqrt(n)) -- for ALL benchmarks.

    Rule: NEVER report an error bar narrower than the instrument's known seed noise.
    The empirical spread over n seeds is the natural estimate but is untrustworthy alone
    (at n=2 it has one degree of freedom, and any censoring collapses it toward 0); the
    measured floor is a lower bound on how precise the instrument can possibly be.  The
    wider of the two is the defensible half-width.

    Returns (ci, empirical_ci, floor_ci, source_string).
    """
    n = len(pcts)
    emp = None
    if n >= 2:
        m = sum(pcts) / n
        sd = math.sqrt(sum((q - m) ** 2 for q in pcts) / (n - 1))
        emp = 2.0 * sd / math.sqrt(n)
    flr = (2.0 * floor_sd_pct / math.sqrt(n)) if (floor_sd_pct is not None and n) else None
    cands = [c for c in (emp, flr) if c is not None]
    ci = max(cands) if cands else None
    src = ("max(empirical 2*SD/sqrt(n)=%s, measured seed floor=%s)"
           % ("%.1f" % emp if emp is not None else "n/a",
              "%.1f" % flr if flr is not None else "n/a"))
    return ci, emp, flr, src


# ---------------------------------------------------------------------------
# RI base -- READ FROM DISK  (F-F)
# ---------------------------------------------------------------------------
# The literals `RI_BASE = {phikon 0.4686, midnight 0.7589, virchow2 0.8582}` appeared in
# five scripts, and collect_final5's comment cited `probe_before.json` as their source.
# THAT PROVENANCE IS FALSE: probe_before.json is the PLISM cross-scanner/cross-stain
# probe and contains no robustness_index field at all.  The real source is PathoROB's own
# results_summary.json for the UNFINETUNED feature dirs, which is what is read here.
#
# NOT to be confused with ri_curve.json["targets"]["*_base"], which is WAIV's PUBLISHED
# Table-1 base row (3 decimals), a different quantity that happens to agree to 3 dp.
RI_BASE_MODEL_DIRS = {
    "phikon":   "phikonv2_clsmean_ours",
    "midnight": "midnight_clsmean_ours",
    "virchow2": "virchow2_clsmean_base",
}
PATHOROB_RESULTS = REPO / "third_party" / "PathoROB" / "results" / "robustness_index"
RI_DATASETS = ("tcga", "camelyon", "tolkach_esca")


def load_ri_base():
    """Per-backbone Avg RI of the untuned backbone, averaged over RI_DATASETS.

    Raises when a dataset is missing: an average over a SUBSET of the three datasets is
    a different quantity, and silently returning it would bias every pct_of_waiv that
    divides by it.
    """
    vals, src = {}, {}
    for arm, model in RI_BASE_MODEL_DIRS.items():
        per = {}
        for ds in RI_DATASETS:
            p = PATHOROB_RESULTS / model / ds / "-1_0" / "results_summary.json"
            if not p.exists():
                raise FileNotFoundError(
                    "RI base for %s: missing %s.  Refusing to fall back to a literal -- "
                    "an unavailable number must be unavailable, not substituted." % (arm, p))
            per[ds] = float(json.loads(p.read_text())["robustness_index"])
        vals[arm] = sum(per.values()) / len(per)
        src[arm] = "%s/%s/<%s>/-1_0/results_summary.json" % (
            PATHOROB_RESULTS.relative_to(REPO), model, "|".join(RI_DATASETS))
    return vals, src


# ---------------------------------------------------------------------------
# Published Waiv targets -- ONE transcription each
# ---------------------------------------------------------------------------
# RI: owned by src/waivphaet/eval/pathorob_adapter.TARGETS (already the repo's single
# transcription of Waiv Table 1); re-keyed here from backbone -> value so callers stop
# re-typing the numbers.
def load_ri_waiv():
    from waivphaet.eval.pathorob_adapter import TARGETS  # noqa: PLC0415
    keys = {"phikon": "phaet_target", "midnight": "mascaret_target", "virchow2": "virchow2_target"}
    return ({a: float(TARGETS[k]["avg"]) for a, k in keys.items()},
            "src/waivphaet/eval/pathorob_adapter.TARGETS[%s].avg (Waiv arXiv:2607.22861 Table 1)"
            % "/".join(keys.values()))


#: HEST: Waiv arXiv:2607.22861 Table 1, "HEST" column, fine-tuned rows.  There is no
#: recomputable artifact for these -- they are a paper transcription and this is their
#: only home in the repo.
HEST_WAIV = {"phikon": 0.3943, "midnight": 0.4167, "virchow2": 0.4135}
HEST_WAIV_SOURCE = "Waiv arXiv:2607.22861 Table 1 HEST column (Phaet / Mascaret / Virchow2-FT)"


# ---------------------------------------------------------------------------
# HEST seed SD -- READ FROM DISK  (F-A)
# ---------------------------------------------------------------------------
HEST_SEED_SD_JSON = REPO / "docs" / "hest_seed_sd.json"


def load_hest_seed_sd():
    """{backbone: {step: sd}} pooled within-recipe seed SD, from docs/hest_seed_sd.json.

    Produced by scripts/hest_seed_sd.py.  Raises if absent: there is no literal to fall
    back to any more, by design.
    """
    if not HEST_SEED_SD_JSON.exists():
        raise FileNotFoundError(
            "%s missing -- run `python3 scripts/hest_seed_sd.py --write` first.  "
            "There is deliberately no hardcoded fallback." % HEST_SEED_SD_JSON)
    blob = json.loads(HEST_SEED_SD_JSON.read_text())
    out = {}
    for arm, per_step in (blob.get("pooled_seed_sd") or {}).items():
        out[arm] = {int(k): v["sd"] for k, v in per_step.items() if v.get("sd") is not None}
    return out, str(HEST_SEED_SD_JSON.relative_to(REPO)), blob


def seed_sd_at_step(per_step: dict, step: int):
    """The ONE step-selection rule for a measured seed floor, shared by RI and HEST.

    Prefer the SD measured at exactly `step`.  When no floor was measured there, take the
    LARGEST SD measured for that backbone at any step: an over-estimate by construction,
    which is the safe direction to be wrong in for an error bar.  Returns (sd, note).
    """
    if not per_step:
        return None, "no measured seed SD at any step for this backbone"
    if step in per_step and per_step[step] is not None:
        return per_step[step], "measured at step %s" % step
    avail = [v for v in per_step.values() if v is not None]
    if not avail:
        return None, "no measured seed SD at any step for this backbone"
    return max(avail), ("no floor measured at step %s; conservative max over measured "
                        "steps %s" % (step, sorted(per_step)))
