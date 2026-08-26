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
    # The two GATED backbones, built from local checkpoints (encoder.BACKBONE_LOCAL_DIRS).
    # Both are `clsmean` on PathoROB, matching the trio above.
    #
    # An arm listed here whose results are not on disk YET is ABSENT from load_ri_base()'s
    # output, not an error -- that is how the report learns its RI denominator has not
    # been measured.  A PARTIAL set (some datasets but not all) still raises; see below.
    "uni2":     "uni2h_clsmean",
    "hoptimus": "hoptimus_clsmean",
}
PATHOROB_RESULTS = REPO / "third_party" / "PathoROB" / "results" / "robustness_index"
RI_DATASETS = ("tcga", "camelyon", "tolkach_esca")


def load_ri_base():
    """Per-backbone Avg RI of the untuned backbone, averaged over RI_DATASETS.

    NOT MEASURED vs MEASURED-IN-PART.  An arm with NONE of the three datasets on disk is
    simply absent from the result: its base has not been measured yet, which is a fact the
    caller must be able to see (that is what puts a backbone in the report's "not
    gradeable" block) rather than a crash that takes the whole report down with it.

    An arm with SOME but not all three still RAISES.  An average over a subset of the
    datasets is a different quantity wearing the same name, and silently returning it
    would bias every pct_of_waiv that divides by it -- and, unlike a wholly-missing arm,
    a partial one is evidence that something went wrong rather than that nothing has run.
    """
    vals, src = {}, {}
    for arm, model in RI_BASE_MODEL_DIRS.items():
        paths = {ds: PATHOROB_RESULTS / model / ds / "-1_0" / "results_summary.json"
                 for ds in RI_DATASETS}
        found = [ds for ds, p in paths.items() if p.exists()]
        if not found:
            continue
        per = {}
        for ds in RI_DATASETS:
            p = paths[ds]
            if not p.exists():
                raise FileNotFoundError(
                    "RI base for %s is PARTIAL: %s of %s present, missing %s.  Refusing "
                    "to average over a subset -- that is a different quantity, and "
                    "refusing to fall back to a literal -- an unavailable number must be "
                    "unavailable, not substituted."
                    % (arm, sorted(found), list(RI_DATASETS), p))
            per[ds] = float(json.loads(p.read_text())["robustness_index"])
        vals[arm] = sum(per.values()) / len(per)
        src[arm] = "%s/%s/<%s>/-1_0/results_summary.json" % (
            PATHOROB_RESULTS.relative_to(REPO), model, "|".join(RI_DATASETS))
    return vals, src


# ---------------------------------------------------------------------------
# Published Waiv targets -- READ FROM docs/waiv_published.json, ONE loader  (F-J)
# ---------------------------------------------------------------------------
# docs/waiv_published.json is the full, line-by-line transcription of Waiv Tables 1/2/3
# for all twenty models they rank.  Before this section existed, THREE of its numbers had
# been copied back out into Python literals keyed by our three published arms -- the RI
# target (via pathorob_adapter.TARGETS), the HEST target (HEST_WAIV), and scoreboard's own
# copy of the THUNDER rows -- so adding a fourth or fifth backbone meant hand-typing six
# more numbers into three more places.  All of them now come from the JSON, through the
# loader below, for EVERY arm.  Adding a backbone is a row in WAIV_ROWS and nothing else.
#
# The retired literals survive ONLY as an assertion target (see the *_RETIRED_LITERALS
# dicts and check_retired_literals()): a disagreement between what was published and what
# is on disk is itself a bug and must be measured, not absorbed.
WAIV_PUBLISHED_JSON = REPO / "docs" / "waiv_published.json"

#: arm -> (base row, fine-tuned row) in the published table.  Waiv RENAME the fine-tuned
#: models (Phikon-v2 -> Phaet, Midnight-12k -> Mascaret) while leaving Virchow2, UNI2-h
#: and H-Optimus-0 under their own names, so the correspondence has to be stated; it is
#: stated ONCE, here.
#:
#: TRAP: "H0-mini" is a separate row and a DIFFERENT model -- a distillation of
#: H-Optimus-0, with its own numbers and its own (clsmean) THUNDER protocol.  It is not
#: an alias for `hoptimus`.
WAIV_ROWS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "phikon":   (("Phikon-v2", "base"),   ("Phaet", "fine-tuned")),
    "midnight": (("Midnight-12k", "base"), ("Mascaret", "fine-tuned")),
    "virchow2": (("Virchow2", "base"),    ("Virchow2", "fine-tuned")),
    "hoptimus": (("H-Optimus-0", "base"), ("H-Optimus-0", "fine-tuned")),
    "uni2":     (("UNI2-h", "base"),      ("UNI2-h", "fine-tuned")),
}

#: Waiv's THUNDER task names -> ours.  Their two extra tasks (calibration, adversarial)
#: are not computed by this repo and are dropped on purpose -- any mean over the four
#: below is NOT their six-task rank sum.
WAIV_THUNDER_TASKS = {
    "knn": "knn",
    "linear": "linear_probing",
    "few_shot": "simple_shot",
    "segmentation": "segmentation",
}

#: Their RI per-dataset key -> ours.
WAIV_RI_DS = {"tcga": "tcga", "camelyon": "camelyon", "tolkach": "tolkach_esca"}

WAIV_SOURCE = "docs/waiv_published.json (arXiv:2607.22861v1 Tables 1/2/3, verified 2026-08-24)"


def load_waiv_published():
    """(WAIV, WAIV_THUNDER) read from docs/waiv_published.json.  ONE formula, all arms.

    WAIV[arm]         = {"ri": ft avg RI, "hest": ft HEST avg, "ri_ds": {ds: ft RI}}
    WAIV_THUNDER[arm] = {"base": {our_task: pct}, "ft": {our_task: pct}}

    Raises rather than falling back: every pct_of_waiv denominator comes from this file
    and there is no literal left to fall back to.
    """
    try:
        blob = json.loads(WAIV_PUBLISHED_JSON.read_text())
    except Exception as exc:  # noqa: BLE001 -- any read/parse failure is fatal here
        raise RuntimeError(
            "cannot read the Waiv published-numbers transcription at %s: %s.  Every "
            "pct_of_waiv denominator comes from that file; there is no fallback literal."
            % (WAIV_PUBLISHED_JSON, exc)) from exc
    index = {(m["name"], m["variant"]): m for m in blob["models"]}
    waiv, waiv_thunder = {}, {}
    for arm, (base_row, ft_row) in WAIV_ROWS.items():
        absent = [r for r in (base_row, ft_row) if r not in index]
        if absent:
            raise RuntimeError(
                "arm %r maps to rows %s which are not in %s; fix WAIV_ROWS or the "
                "transcription." % (arm, absent, WAIV_PUBLISHED_JSON.name))
        base, ft = index[base_row], index[ft_row]
        waiv[arm] = {
            "ri": ft["ri"]["avg"],
            "ri_base": base["ri"]["avg"],
            "hest": ft["hest_avg"],
            "hest_base": base["hest_avg"],
            "ri_ds": {ours: ft["ri"].get(theirs) for theirs, ours in WAIV_RI_DS.items()},
        }
        waiv_thunder[arm] = {
            "base": {ours: base["thunder"][theirs] for theirs, ours in WAIV_THUNDER_TASKS.items()},
            "ft": {ours: ft["thunder"][theirs] for theirs, ours in WAIV_THUNDER_TASKS.items()},
        }
    return waiv, waiv_thunder


WAIV, WAIV_THUNDER = load_waiv_published()


def load_ri_waiv():
    """{arm: Waiv's fine-tuned Avg RI} for EVERY arm in WAIV_ROWS, from the JSON.

    Was: re-keyed out of src/waivphaet/eval/pathorob_adapter.TARGETS, which is a second
    transcription of the same Table-1 column and covers only the published trio.  That
    module keeps its per-DATASET targets (the gate script indexes them by dataset); this
    average is now read from the one file that has the whole table.
    """
    return ({a: float(v["ri"]) for a, v in WAIV.items()},
            "%s -> models[<ft row>].ri.avg (Waiv Table 1)" % WAIV_SOURCE)


#: HEST: Waiv arXiv:2607.22861 Table 1, "HEST" column, fine-tuned rows -- for every arm.
HEST_WAIV = {a: float(v["hest"]) for a, v in WAIV.items()}
HEST_WAIV_SOURCE = "%s -> models[<ft row>].hest_avg (Waiv Table 1 HEST column)" % WAIV_SOURCE


# --- retired literals, kept ONLY as assertion targets -------------------------------
#: What the three Waiv targets were hand-typed as before they were read from the JSON.
#: Nothing consumes these; check_retired_literals() compares them to what is now loaded
#: so that a transcription drift shows up as a reported number rather than as silence.
RI_WAIV_RETIRED_LITERALS = {"phikon": 0.806, "midnight": 0.924, "virchow2": 0.918}
HEST_WAIV_RETIRED_LITERALS = {"phikon": 0.3943, "midnight": 0.4167, "virchow2": 0.4135}


def check_retired_literals():
    """{name: {retired, from_disk, delta, agrees}} for every retired Waiv literal.

    Also re-checks pathorob_adapter.TARGETS, which remains the owner of the per-DATASET
    Table-1 targets: its averages must equal the JSON's or the two transcriptions have
    drifted apart.
    """
    out = {}

    def _cmp(tag, retired, live, tol=5e-5):
        for a, v in retired.items():
            got = live.get(a)
            d = None if got is None else got - v
            out["%s/%s" % (tag, a)] = {
                "retired_literal": v, "from_disk": got, "delta": d,
                "agrees_to_4dp": d is not None and abs(d) < tol,
            }

    _cmp("RI_WAIV", RI_WAIV_RETIRED_LITERALS, {a: v["ri"] for a, v in WAIV.items()})
    _cmp("HEST_WAIV", HEST_WAIV_RETIRED_LITERALS, HEST_WAIV)
    try:
        from waivphaet.eval.pathorob_adapter import TARGETS  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- the cross-check is a bonus, not a requirement
        return out
    ft_keys = {"phikon": "phaet_target", "midnight": "mascaret_target",
               "virchow2": "virchow2_target"}
    _cmp("pathorob_adapter.TARGETS",
         {a: float(TARGETS[k]["avg"]) for a, k in ft_keys.items() if k in TARGETS},
         {a: v["ri"] for a, v in WAIV.items()})
    return out


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


# ---------------------------------------------------------------------------
# POOLED (ratio-of-means) aggregation  --  the ONE implementation  (F-P)
# ---------------------------------------------------------------------------
# THE GRADING RULE.  Aggregate the NUMERATOR and the DENOMINATOR first, then divide
# ONCE:
#
#       pct = mean_over_cells(our raw delta) / mean_over_cells(Waiv's raw gain) * 100
#
# NEVER average per-cell percentages.  A mean of ratios is dominated by whichever cell
# happens to have the smallest denominator, and it is exactly that pathology that made
# three of the nine (backbone x benchmark) cells ungradeable: virchow2's per-task Waiv
# THUNDER gains are +0.037 / -0.0030 / +0.0030, two of them below the seed floor and one
# NEGATIVE, so no per-task ratio is meaningful -- yet their POOLED denominator is
# +0.0090, which is well conditioned.
#
# The two ABSOLUTE averages (ours, Waiv's) are the primary quantities and must be led
# with; the percentage is the derived, secondary one.
#
# ALL-OR-NOTHING.  A pooled number may only be formed from EVERY cell of its group.  A
# mean over a subset is a different quantity that shares its name -- e.g. a THUNDER mean
# over 2 of 3 tasks silently re-weights the benchmark -- so an incomplete group is
# WITHHELD, never computed from what happens to be on disk.

#: Groups smaller than this are not "pooled" in any meaningful sense.
POOL_MIN_CELLS = 1

#: A pooled cell's share of the numerator or denominator above which the pooled number
#: is carried by one cell rather than by the group.  Disclosure threshold, NOT a gate:
#: a flagged group is still reported, loudly annotated.
CONCENTRATION_FLAG_SHARE = 0.50


def pool_cells(cells, *, group: str = "", require_complete: bool = True,
               denominator_gate: bool = True):
    """Ratio-of-means over `cells`.  The ONLY pooled-aggregation implementation.

    `cells` is a list of dicts, one per (backbone, task) or (backbone,) cell:

        key        str   -- how the cell is named in the concentration table
        delta      float -- OUR raw improvement over OUR base, in raw metric units
        gain       float -- WAIV's raw gain (their FT minus their base), same units
        se_delta   float -- 1 SE of OUR delta = per-seed SD / sqrt(n_our_seeds)
        sd_gain    float -- 1 SD, one run, of the instrument that measured `gain`
        complete   bool  -- False when the cell is missing or PARTIAL
        note       str   -- optional, carried through to the output

    Returns a dict.  `status` is one of:
        POOLED               -- a number was formed; `pct`, `ci`, `lower`, `upper` set
        WITHHELD_INCOMPLETE  -- at least one cell of the group is missing/PARTIAL
        WITHHELD_DENOMINATOR -- the POOLED denominator is itself within seed noise

    ERROR PROPAGATION.  The numerator is a mean of k cell-deltas, so

        SE_agg = sqrt(sum_i SE_i^2) / k

    and the same construction gives the denominator's SD_agg from each cell's one-run
    sd_gain.  The interval on the RATIO is the delta method carried on BOTH terms:

        CI = 2 * 100 * sqrt( (SE_agg/den)^2 + (num*SD_agg/den^2)^2 )

    which is why no separate "the denominator is imprecise" veto is needed for a
    denominator whose SIGN is determined -- the imprecision is IN the interval.

    INDEPENDENCE CAVEAT (reported, not hidden): quadrature assumes the cells' seed noise
    is independent.  For the three THUNDER tasks of one backbone it is not -- they are
    three readouts of the SAME per-seed checkpoints -- so SE_agg is an UNDER-estimate
    there by up to sqrt(k).  `independence_caveat` carries the worst case.
    """
    out = {
        "group": group,
        "rule": "ratio-of-means (pooled): mean(our delta) / mean(waiv gain) * 100",
        "n_cells": len(cells),
        "cells": [dict(c) for c in cells],
    }
    if not cells:
        out["status"] = "WITHHELD_INCOMPLETE"
        out["reason"] = "no cells in group"
        return out

    missing = [c["key"] for c in cells
               if not c.get("complete", True)
               or c.get("delta") is None or c.get("gain") is None]
    if require_complete and missing:
        out["status"] = "WITHHELD_INCOMPLETE"
        out["reason"] = (
            "pooling requires ALL %d cells of %s; missing or PARTIAL: %s.  A pooled "
            "number over a SUBSET re-weights the group and is a different quantity."
            % (len(cells), group or "the group", ", ".join(missing)))
        out["missing_cells"] = missing
        return out

    k = len(cells)
    num = sum(c["delta"] for c in cells) / k
    den = sum(c["gain"] for c in cells) / k
    se_num = math.sqrt(sum((c.get("se_delta") or 0.0) ** 2 for c in cells)) / k
    sd_den = math.sqrt(sum((c.get("sd_gain") or 0.0) ** 2 for c in cells)) / k
    any_se_missing = [c["key"] for c in cells if c.get("se_delta") is None]
    any_sd_missing = [c["key"] for c in cells if c.get("sd_gain") is None]

    out.update({
        "our_avg_delta": num,
        "waiv_avg_gain": den,
        "se_our_avg_delta": se_num,
        "sd_waiv_avg_gain": sd_den,
        "se_missing_cells": any_se_missing,
        "sd_missing_cells": any_sd_missing,
        "independence_caveat": (
            "SE_agg = sqrt(sum SE_i^2)/k assumes independent cell noise; for the 3 "
            "THUNDER tasks of one backbone (same checkpoints, same 12 datasets) the "
            "true SE is up to sqrt(%d) = %.2fx larger, i.e. up to %.6f"
            % (k, math.sqrt(k), se_num * math.sqrt(k))),
    })

    # -- concentration disclosure (MANDATORY: pooling fixes small denominators but can
    #    let one cell carry the whole result).  Signed shares, so a cell pulling the
    #    pooled number the other way shows as negative rather than being hidden by an
    #    absolute value.
    tot_num = sum(c["delta"] for c in cells)
    tot_den = sum(c["gain"] for c in cells)
    shares, flags = {}, []
    for c in cells:
        sn = (c["delta"] / tot_num) if abs(tot_num) > 1e-15 else None
        sd_ = (c["gain"] / tot_den) if abs(tot_den) > 1e-15 else None
        shares[c["key"]] = {
            "delta": c["delta"], "gain": c["gain"],
            "numerator_share": sn, "denominator_share": sd_,
        }
        for what, s in (("numerator", sn), ("denominator", sd_)):
            if s is not None and abs(s) > CONCENTRATION_FLAG_SHARE:
                flags.append("%s carries %.0f%% of the pooled %s" % (c["key"], s * 100, what))
                shares[c["key"]]["flag_%s" % what] = True
    out["shares"] = shares
    out["concentration_flags"] = flags
    out["concentrated"] = bool(flags)

    # -- THE denominator gate, applied to the POOLED denominator, not to each cell.
    #    A group is ungradeable only when its POOLED denominator is noise.
    if denominator_gate:
        unres, reason = pooled_denominator_unresolvable(den, sd_den)
        out["denominator_gate"] = {
            "pooled_gain": den, "pooled_sd": sd_den,
            "gain_over_2sd": (abs(den) / (2 * sd_den)) if sd_den else None,
            "rule": POOLED_DENOMINATOR_GATE,
            "unresolvable": unres, "reason": reason,
        }
        if unres:
            out["status"] = "WITHHELD_DENOMINATOR"
            out["reason"] = reason
            out["pct"] = None
            return out

    pct = num / den * 100.0
    # Delta method on the ratio, propagating BOTH terms.
    rel = math.sqrt((se_num / den) ** 2 + (num * sd_den / den ** 2) ** 2)
    ci = 2.0 * 100.0 * rel
    out.update({
        "status": "POOLED",
        "pct": pct,
        "ci": ci,
        "lower": pct - ci,
        "upper": pct + ci,
        "ci_source": ("delta method on the ratio: 2*100*sqrt((SE_num/den)^2 + "
                      "(num*SD_den/den^2)^2); numerator term %.2f pts, denominator "
                      "term %.2f pts"
                      % (2 * 100 * abs(se_num / den),
                         2 * 100 * abs(num * sd_den / den ** 2))),
    })
    return out


#: Gate applied to a POOLED denominator.  A ratio is a measurement only when the sign
#: and scale of its denominator are determined; when |gain| <= 2*SD the denominator is
#: not distinguishable from zero and the ratio is unbounded (its distribution has no
#: finite mean).  That -- and only that -- is what "the denominator is noise" means.
#: It is the same 2-sigma the CI construction uses; nothing here is tuned.
POOLED_DENOMINATOR_GATE = ("|pooled waiv gain| > 2 * SD(pooled waiv gain) -- the "
                           "denominator's sign/scale must be determined for the ratio "
                           "to be a measurement")


def pooled_denominator_unresolvable(gain: float | None, sd: float | None):
    """(unresolvable, reason) for a POOLED denominator.  See POOLED_DENOMINATOR_GATE."""
    if gain is None:
        return True, "no pooled waiv gain"
    if sd is None:
        return True, "no measured SD for the pooled waiv gain -- resolvability untestable"
    if abs(gain) <= 2.0 * sd:
        return True, ("pooled waiv gain %+.5f is within 2 SD (%.5f) of zero -- the "
                      "denominator's sign is not determined, so the ratio is unbounded"
                      % (gain, 2.0 * sd))
    return False, None


# ---------------------------------------------------------------------------
# THUNDER per-seed SD -- READ FROM DISK, ONE owner
# ---------------------------------------------------------------------------
# `seed_sd_of_task_mean` is the SD, ACROSS THE 5 TRAINING SEEDS, of the 12-dataset task
# mean itself.  It is the quantity the pooled machinery needs: a 1-SD, one-run noise on
# exactly the statistic being pooled.
#
# It must NOT be confused with `offset_2se` (the other column in the same file), which is
# |mean(d)| + 2*SD(d)/sqrt(12) with the SD taken OVER THE 12 DATASETS.  That is a
# different variance component measuring a different thing, and swapping the two inflates
# a noise estimate by roughly 2-4x.  Both live in the same JSON; only this one is loaded
# for pooling and error propagation.
THUNDER_SEED_SD_JSON = REPO / "docs" / "thunder_seed_floor_12ds.json"
THUNDER_SEED_SD_FIELD = "cells[<backbone>/<task>].12ds.seed_sd_of_task_mean"


def load_thunder_seed_sd():
    """{backbone: {task: sd}} from docs/thunder_seed_floor_12ds.json.  (map, source_str).

    Raises when the file is absent: as with the HEST seed SD, an unavailable measurement
    must be unavailable rather than substituted with a literal.
    """
    if not THUNDER_SEED_SD_JSON.exists():
        raise FileNotFoundError(
            "%s missing -- run `python3 scripts/thunder_seed_floor_12ds.py` first.  "
            "There is deliberately no hardcoded fallback." % THUNDER_SEED_SD_JSON)
    blob = json.loads(THUNDER_SEED_SD_JSON.read_text())
    out: dict = {}
    for key, cell in (blob.get("cells") or {}).items():
        bb, _, task = key.partition("/")
        v = (cell.get("12ds") or {}).get("seed_sd_of_task_mean")
        if v is not None:
            out.setdefault(bb, {})[task] = float(v)
    return out, "%s -> %s" % (THUNDER_SEED_SD_JSON.relative_to(REPO), THUNDER_SEED_SD_FIELD)


#: The OTHER column of the same file: the RESOLVABILITY floor.
#: offset_2se = |mean(d)| + 2*SD(d)/sqrt(12) over the 12 per-dataset F1 deltas of a seed
#: pair, averaged over all 10 unordered pairs.  Its job is the denominator gate ("is
#: Waiv's own gain even bigger than seed noise"), NOT the error bar on our task mean --
#: its SD is taken over DATASETS, which is the wrong variance component for that.  Both
#: quantities live in the same JSON and swapping them inflates a noise estimate 2-4x, so
#: each has its own loader and its own docstring saying which is which.
THUNDER_FLOOR_FIELD = "cells[<backbone>/<task>].12ds.offset_2se_mean"


def load_thunder_floor():
    """{backbone: {task: offset_2se_mean}} from docs/thunder_seed_floor_12ds.json.

    Was three hand-typed 4-decimal literals in final_recipe_report.py, keyed by the
    published trio only -- which is why a fourth backbone read as "no THUNDER floor
    measured" even when one had been.  Now every backbone the file has a cell for is
    present, and a backbone it does NOT have a cell for is absent (the correct answer:
    its floor has not been measured, so its THUNDER cell is ungradeable).

    VALID ONLY for a full 12/12 PAPER_CLS task mean; a mean over fewer datasets averages
    away less per-dataset noise and is NOISIER, so applying this floor to it would
    manufacture resolvability.
    """
    if not THUNDER_SEED_SD_JSON.exists():
        raise FileNotFoundError(
            "%s missing -- run `python3 scripts/thunder_seed_floor_12ds.py` first.  "
            "There is deliberately no hardcoded fallback." % THUNDER_SEED_SD_JSON)
    blob = json.loads(THUNDER_SEED_SD_JSON.read_text())
    out: dict = {}
    for key, cell in (blob.get("cells") or {}).items():
        bb, _, task = key.partition("/")
        v = (cell.get("12ds") or {}).get("offset_2se_mean")
        if v is not None:
            out.setdefault(bb, {})[task] = float(v)
    return out, "%s -> %s (n=5 training seeds, offset-2SE, 12/12 coverage)" % (
        THUNDER_SEED_SD_JSON.relative_to(REPO), THUNDER_FLOOR_FIELD)
