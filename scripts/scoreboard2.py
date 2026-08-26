#!/usr/bin/env python3
"""scoreboard2.py — Rule-enforcing checkpoint scoreboard.

RULE 1 — ONE ROW = ONE (run_name, step). Never pair best-RI from one run with
          best-HEST from another. Every row is a single (run_name, step) and
          reports ALL metrics for THAT checkpoint. Missing metrics print MISSING,
          never substituted from another arm or step.

RULE 2 — Raw scores as  ours | Waiv | diff.  Gain-vs-base appears as an EXTRA
          column, never as the headline figure.

Usage:
    python scripts/scoreboard2.py                          # all runs, step 500
    python scripts/scoreboard2.py --step 250               # at step 250
    python scripts/scoreboard2.py --sort-by hest
    python scripts/scoreboard2.py --only-complete          # hide rows missing any metric
    python scripts/scoreboard2.py --backbones virchow2 midnight
    python scripts/scoreboard2.py --runs ret0.01-midnight-s0-t900-391057
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: put scripts/ on path so we can import collect_final5 helpers
# without duplicating them. This is the canonical way to reuse that logic.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import collect_final5 as _c5  # noqa: E402
import eval_common as _ec    # noqa: E402
from collect_final5 import (  # noqa: E402 — intentional after sys.path insert
    HEST_WORK_DIR,
    THUNDER_ROOT,
    PAPER_CLS,
    PAPER_SEG,
    THUNDER_TASKS,
    _union_ri_curves,
    _hest_score,
    _thunder_ft_per_ds,
)

# ---------------------------------------------------------------------------
# Waiv arXiv:2607.22861 Tables 1+3  (base → Waiv fine-tuned)
# NOTE: virchow2 base_hest was 0.4034 in the old scoreboard.py — that was
# rounded incorrectly. The correct value is 0.40324, which is what
# collect_final5.HEST_BASE["virchow2"] carries (source: vbase_clsmean_summary.json).
# ---------------------------------------------------------------------------
WAIV: dict[str, dict] = {
    "phikon": {
        "base_ri":   _c5.RI_BASE["phikon"],      # F-E/F-F: read from PathoROB results on disk
        "waiv_ri":   _c5.RI_WAIV["phikon"],
        "base_hest": _c5.HEST_BASE["phikon"],    # F-E: loader, not a literal
        "waiv_hest": _ec.HEST_WAIV["phikon"],
        "base_ds": {"tcga": 0.619, "camelyon": 0.019, "tolkach_esca": 0.768},
        "waiv_ds":  {"tcga": 0.785, "camelyon": 0.702, "tolkach_esca": 0.932},
        "pool": "cls",
    },
    "midnight": {
        "base_ri":   _c5.RI_BASE["midnight"],    # F-E/F-F: read from PathoROB results on disk
        "waiv_ri":   _c5.RI_WAIV["midnight"],
        "base_hest": _c5.HEST_BASE["midnight"],  # F-E: loader, not a literal
        "waiv_hest": _ec.HEST_WAIV["midnight"],
        "base_ds": {"tcga": 0.858, "camelyon": 0.478, "tolkach_esca": 0.941},
        "waiv_ds":  {"tcga": 0.893, "camelyon": 0.907, "tolkach_esca": 0.972},
        "pool": "cls",
    },
    "virchow2": {
        "base_ri":   _c5.RI_BASE["virchow2"],    # F-E/F-F: read from PathoROB results on disk
        "waiv_ri":   _c5.RI_WAIV["virchow2"],
        "base_hest": _c5.HEST_BASE["virchow2"],  # F-E: was the ROUNDED results.avg 0.40324; the loader reads custom_encoder 0.4032685
        "waiv_hest": _ec.HEST_WAIV["virchow2"],
        "base_ds": {"tcga": 0.822, "camelyon": 0.799, "tolkach_esca": 0.954},
        "waiv_ds":  {"tcga": 0.849, "camelyon": 0.935, "tolkach_esca": 0.969},
        "pool": "clsmean",
    },
}

# ---------------------------------------------------------------------------
# RI-budget floors -- QUARANTINED 2026-08-26 (F-I).  OFF BY DEFAULT.
# ---------------------------------------------------------------------------
# 0.9134 / 0.9140 are TUNED CONSTANTS.  They exist in no other file, they have no
# producing script and no on-disk source, they are not derived from any measured seed
# floor -- and they drove a PASS/FAIL column.  A hand-chosen threshold that decides
# pass/fail is exactly the thing this repo's criterion is supposed to exclude, and it is
# not comparable across backbones (phikon simply had none, so phikon could never fail it).
#
# The column is retained ONLY behind an explicit opt-in, so that any historical output
# quoting it can be reproduced, and it prints its own provenance warning when enabled.
# Do NOT re-enable it to make a checkpoint pass.
RI_BUDGET_FLOOR: dict[str, float | None] = {
    "virchow2": 0.9134,
    "midnight": 0.9140,
    "phikon":   None,   # never defined -- phikon could not fail this column
}
RI_BUDGET_ENABLED = bool(os.environ.get("WAIV_ENABLE_RI_BUDGET_COLUMN"))
RI_BUDGET_WARNING = ("RI-budget column is a TUNED constant with no on-disk source; "
                     "quarantined 2026-08-26, enable with WAIV_ENABLE_RI_BUDGET_COLUMN=1")

# ---------------------------------------------------------------------------
# Seed-SD noise floors, per backbone × step.
# Source: memory waiv-ri-seed-noise-floor; measured from on-disk seed replicates.
# "~noise" = |diff| < 2 SD.
# ---------------------------------------------------------------------------
SEED_FLOORS: dict[str, dict[int, dict[str, float]]] = {
    "midnight": {
        250: {"ri": 0.00482, "hest": 0.00227},
        500: {"ri": 0.00211, "hest": 0.00191},
    },
    "virchow2": {
        250: {"ri": 0.00475, "hest": 0.00153},
        500: {"ri": 0.00196, "hest": 0.00115},
    },
    "phikon": {
        500: {"ri": 0.00453, "hest": 0.00167},
    },
}

# ---------------------------------------------------------------------------
# THUNDER task-mean seed SD -- MEASURED, per (backbone, task).   F-I fix 2026-08-26.
# ---------------------------------------------------------------------------
# This was `_THUNDER_MEAN_1SD = 0.0025`, a single flat number the comment itself
# described as an eyeballed proxy ("2SE ≈ 0.20–0.30pp → treat 0.25pp as 1 SD").  The
# measured per-(backbone, task) floors span 0.0066–0.0233, i.e. up to 9.3x larger, so
# every "~noise" / "Xσ" label this script printed on a THUNDER cell was computed against
# a noise scale that was too small by up to an order of magnitude -- which turns real
# noise into an apparently significant sigma count.  The measured values are read from
# the same artifact final_recipe_report and scoreboard use.
_TSF = json.loads((Path(__file__).resolve().parents[1] /
                   "docs" / "thunder_seed_floor_12ds.json").read_text())
THUNDER_SEED_SD: dict[tuple[str, str], float] = {}
THUNDER_OFFSET_2SE: dict[tuple[str, str], float] = {}
for _k, _cell in (_TSF.get("cells") or {}).items():
    _bb, _, _task = _k.partition("/")
    _12 = _cell.get("12ds") or {}
    if _12.get("seed_sd_of_task_mean") is not None:
        THUNDER_SEED_SD[(_bb, _task)] = _12["seed_sd_of_task_mean"]
    if _12.get("offset_2se") is not None:
        THUNDER_OFFSET_2SE[(_bb, _task)] = _12["offset_2se"]
THUNDER_SEED_SD_SOURCE = ("docs/thunder_seed_floor_12ds.json "
                          "cells[bb/task].12ds.seed_sd_of_task_mean (n=5 seeds, 12/12)")


# ---------------------------------------------------------------------------
# Backbone detection
# ---------------------------------------------------------------------------
_BACKBONES = ("phikon", "midnight", "virchow2")


def _detect_backbone(name: str) -> str | None:
    for b in _BACKBONES:
        if b in name:
            return b
    return None


# ---------------------------------------------------------------------------
# Run-name metadata parsing (best-effort; never fatal on unknown pattern)
# ---------------------------------------------------------------------------

def _parse_run_meta(run_name: str) -> dict:
    """Extract key hyperparams from the run name."""
    meta: dict = {"run_name": run_name}

    parts = run_name.split("-")

    # Job ID — trailing all-digit segment
    meta["job_id"] = parts[-1] if (parts and parts[-1].isdigit()) else ""

    # Backbone
    meta["backbone"] = _detect_backbone(run_name) or "unknown"

    # Seed: -s<N>- pattern
    m = re.search(r"(?:^|-)(s)(\d+)(?:-|$)", run_name)
    meta["seed"] = int(m.group(2)) if m else None

    # Temperature schedule: -t<N>-
    m = re.search(r"(?:^|-)t(\d+(?:\.\d+)?)(?:-|$)", run_name)
    meta["temp_sched"] = m.group(1) if m else None

    # Learning rate
    m = re.search(r"-lr([^-]+)-", run_name)
    meta["lr"] = m.group(1) if m else None

    # KL weight
    m = re.search(r"-kl([^-]+)-", run_name)
    meta["kl"] = m.group(1) if m else None

    # max_steps
    m = re.search(r"-ms(\d+)-", run_name)
    meta["max_steps"] = int(m.group(1)) if m else None

    # Flags
    meta["mask"]     = "MASK" in run_name or "genMASK" in run_name
    meta["per_head"] = (run_name.startswith("ph-") or run_name.startswith("ph2-")
                        or "-ph-" in run_name or "-ph2-" in run_name)

    # Recipe prefix: everything up to (but not including) the backbone token
    recipe = run_name
    for b in _BACKBONES:
        idx = recipe.find(b)
        if idx != -1:
            recipe = recipe[:idx].rstrip("-")
            break
    meta["recipe"] = recipe or run_name

    return meta


def _arm_desc(meta: dict) -> str:
    """One-line arm description for the printed header."""
    parts = [meta["recipe"]]
    if meta.get("lr"):
        parts.append(f"lr={meta['lr']}")
    if meta.get("kl") and meta["kl"] != "0":
        parts.append(f"kl={meta['kl']}")
    if meta.get("mask"):
        parts.append("mask=on")
    if meta.get("per_head"):
        parts.append("per-head")
    if meta.get("max_steps"):
        parts.append(f"ms={meta['max_steps']}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Noise annotation
# ---------------------------------------------------------------------------

def _noise_label(diff: float | None, metric: str, backbone: str, step: int) -> str:
    """Return 'Xσ' or '~noise' for a difference vs Waiv (or vs base).

    metric is 'ri', 'hest', or 'thunder_mean'.
    Returns empty string when the floor is unknown (don't annotate what we
    can't measure).
    """
    if diff is None:
        return ""
    if metric == "thunder_mean":
        # No single scalar: the floor is per (backbone, task).  A task-mean label is only
        # honest against the LARGEST constituent floor, and only when every constituent
        # has one -- otherwise the label is withheld, exactly as for an unknown SEED_FLOOR.
        cand = [v for (bb, _t), v in THUNDER_SEED_SD.items() if bb == backbone]
        sd = max(cand) if cand else None
    else:
        sd = SEED_FLOORS.get(backbone, {}).get(step, {}).get(metric)
    if sd is None or sd == 0:
        return ""
    n_sd = abs(diff) / sd
    return "~noise" if n_sd < 2.0 else f"{n_sd:.1f}σ"


# ---------------------------------------------------------------------------
# THUNDER helpers
# ---------------------------------------------------------------------------

def _thunder_task_mean_guarded(
    per_ds: dict[str, dict[str, float | None]],
    task: str,
) -> tuple[float | None, str]:
    """Return (mean, display_label) for one THUNDER task.

    If coverage < expected, mean is None and label is PARTIAL(n/N).
    RULE 1 enforcement: we refuse to print a mean when coverage is incomplete.
    """
    expected = PAPER_SEG if task == "segmentation" else PAPER_CLS
    found = [ds for ds in expected if ds in per_ds.get(task, {})]
    n_found, n_total = len(found), len(expected)
    if n_found == 0:
        return None, "MISSING"
    if n_found < n_total:
        return None, f"PARTIAL({n_found}/{n_total})"
    vals = [per_ds[task][ds] for ds in found if per_ds[task][ds] is not None]
    if not vals:
        return None, "MISSING"
    mean = sum(vals) / len(vals)
    return mean, f"{mean:.4f}"


def _thunder_overall_mean(task_means: dict[str, float | None]) -> tuple[float | None, str]:
    """Mean across the 4 THUNDER task means. Suppressed if any task is missing."""
    vals = [v for v in task_means.values() if v is not None]
    if len(vals) < len(THUNDER_TASKS):
        n_missing = len(THUNDER_TASKS) - len(vals)
        return None, f"PARTIAL({len(vals)}/{len(THUNDER_TASKS)} tasks)"
    mean = sum(vals) / len(vals)
    return mean, f"{mean:.4f}"


# ---------------------------------------------------------------------------
# Per-run data collection
# ---------------------------------------------------------------------------

def _ri_point_at(points: list[dict], step: int) -> dict | None:
    for pt in points:
        if pt.get("step") == step:
            return pt
    return None


def collect_run(run_dir: Path, step: int) -> dict:
    """Collect all metrics for a single (run_name, step). This is the atomic unit.

    Never looks at a different step or a different run to fill in a missing value.
    """
    name = run_dir.name
    meta = _parse_run_meta(name)
    backbone = meta["backbone"]
    pool = WAIV.get(backbone, {}).get("pool", "cls")

    result: dict = {
        "name":              name,
        "meta":              meta,
        "backbone":          backbone,
        "step":              step,
        # RI
        "ri":                None,   # avg_robustness_index at `step`
        "ri_datasets":       {},     # {ds: float} per dataset
        "ri_steps_avail":    [],     # for diagnostic notes
        "preempted":         False,
        # HEST
        "hest":              None,
        # THUNDER
        "thunder_means":     {},     # {task: float | None}
        "thunder_labels":    {},     # {task: display str}
        "thunder_overall":   None,
        "thunder_overall_label": "MISSING",
    }

    # --- RI ----------------------------------------------------------------
    points, preempted = _union_ri_curves(run_dir)
    result["preempted"]      = preempted
    result["ri_steps_avail"] = [p.get("step") for p in points]

    pt = _ri_point_at(points, step)
    if pt is not None:
        result["ri"] = pt.get("avg_robustness_index")
        for dsname in ("tcga", "camelyon", "tolkach_esca"):
            ds_block = pt.get("datasets", {}).get(dsname, {})
            v = ds_block.get("robustness_index")
            if v is not None:
                result["ri_datasets"][dsname] = v

    # --- HEST --------------------------------------------------------------
    # _hest_score derives pooling from arm internally (exactly as collect_final5 does)
    if backbone in WAIV:
        result["hest"] = _hest_score(name, step, backbone)

    # --- THUNDER -----------------------------------------------------------
    per_ds = _thunder_ft_per_ds(name, step)
    task_means: dict[str, float | None] = {}
    for task in THUNDER_TASKS:
        mean, label = _thunder_task_mean_guarded(per_ds, task)
        result["thunder_means"][task]  = mean
        result["thunder_labels"][task] = label
        task_means[task] = mean

    overall, overall_label = _thunder_overall_mean(task_means)
    result["thunder_overall"]       = overall
    result["thunder_overall_label"] = overall_label

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _raw_line(ours: float | None, waiv: float | None, base: float | None,
              metric: str, backbone: str, step: int,
              extra_vs_waiv_label: str = "") -> str:
    """Format:  ours=X | Waiv=Y | diff=Z(Nσ) | Δbase=W"""
    def _f4(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "MISSING"

    o_s  = _f4(ours)
    w_s  = _f4(waiv)
    b_s  = _f4(base)

    if ours is not None and waiv is not None:
        diff_f = ours - waiv
        noise  = _noise_label(diff_f, metric, backbone, step)
        diff_s = f"{diff_f:+.4f}"
        if noise:
            diff_s += f"({noise})"
    else:
        diff_s = "?"

    if ours is not None and base is not None:
        base_d = f"{ours - base:+.4f}"
    else:
        base_d = "?"

    label = f"ours={o_s} | Waiv={w_s} | diff={diff_s} | Δbase={base_d}"
    if extra_vs_waiv_label:
        label += f"  [{extra_vs_waiv_label}]"
    return label


def _budget_flag(ri: float | None, backbone: str) -> str:
    if not RI_BUDGET_ENABLED:
        return "off"            # F-I: quarantined tuned constant
    floor = RI_BUDGET_FLOOR.get(backbone)
    if floor is None:
        return "n/a"
    if ri is None:
        return "?"
    return "PASS" if ri >= floor else "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=500,
                    help="Primary checkpoint step to report (default: 500)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Explicit run dir names (default: scan runs/)")
    ap.add_argument("--backbones", nargs="*", choices=list(_BACKBONES),
                    default=None, help="Filter to these backbones")
    ap.add_argument("--sort-by", choices=["hest", "thunder", "ri", "name"],
                    default="name", dest="sort_by",
                    help="Sort rows within each backbone group (default: name)")
    ap.add_argument("--only-complete", action="store_true",
                    help="Hide checkpoints missing RI, HEST, or any THUNDER task")
    ap.add_argument("--runs-dir", default=None,
                    help="Path to runs/ directory (default: repo/runs)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else (_REPO / "runs")

    # -- Discover run directories ------------------------------------------
    if args.runs:
        run_dirs = [runs_dir / r for r in args.runs if (runs_dir / r).is_dir()]
    else:
        run_dirs = []
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir()):
                if not d.is_dir():
                    continue
                n = d.name
                if re.search(r"\.r\d+$", n):
                    continue           # restart sibling — handled by _union_ri_curves
                if not (d / "ri_curve.json").exists():
                    continue
                if not _detect_backbone(n):
                    continue
                run_dirs.append(d)

    if args.backbones:
        run_dirs = [d for d in run_dirs if _detect_backbone(d.name) in args.backbones]

    if not run_dirs:
        print("No runs found.")
        return

    print(f"Collecting {len(run_dirs)} run(s) at step {args.step} …", flush=True)

    rows = [collect_run(d, args.step) for d in run_dirs]

    # -- Filter ----------------------------------------------------------
    if args.only_complete:
        def _complete(r: dict) -> bool:
            if r["ri"] is None or r["hest"] is None:
                return False
            return all(r["thunder_means"].get(t) is not None for t in THUNDER_TASKS)
        before = len(rows)
        rows = [r for r in rows if _complete(r)]
        print(f"  --only-complete: {before - len(rows)} row(s) hidden, {len(rows)} remaining.")

    if not rows:
        print("No rows to display.")
        return

    # -- Sort within each backbone group ---------------------------------
    def _sort_key(r: dict):
        if args.sort_by == "ri":
            return -(r["ri"] or 0)
        if args.sort_by == "hest":
            return -(r["hest"] or 0)
        if args.sort_by == "thunder":
            v = r["thunder_overall"] or 0
            return -v
        # default: name (then seed)
        return (r["backbone"], r["meta"].get("seed") or 0, r["name"])

    rows.sort(key=_sort_key)

    # Backbone ordering
    _bb_order = {"phikon": 0, "midnight": 1, "virchow2": 2}
    backbones_present: list[str] = []
    seen: set[str] = set()
    for r in sorted(rows, key=lambda r: _bb_order.get(r["backbone"], 9)):
        if r["backbone"] not in seen:
            backbones_present.append(r["backbone"])
            seen.add(r["backbone"])

    # ====================================================================
    # Print
    # ====================================================================
    W = 110
    print()
    print("=" * W)
    print(f"  WAIV SCOREBOARD  |  step={args.step}  |  {len(rows)} row(s)")
    print(f"  RULE 1: every row is one (run, step); MISSING = not available, never substituted.")
    print(f"  RULE 2: raw scores as  ours | Waiv | diff.  Δbase is an extra column.")
    print("=" * W)

    for backbone in backbones_present:
        w      = WAIV[backbone]
        bb_rows = [r for r in rows if r["backbone"] == backbone]

        print()
        print(f"  ┌─ {backbone.upper()}  (HEST pool={w['pool']}) ─────────────────────────────")
        print()

        for r in bb_rows:
            meta   = r["meta"]
            budget = _budget_flag(r["ri"], backbone)
            pre    = " [PREEMPTED]" if r["preempted"] else ""
            avail  = ""
            if r["ri"] is None and r["ri_steps_avail"]:
                avail = f"  [ri available at steps: {r['ri_steps_avail']}]"

            print(f"  ┃ {r['name']}")
            print(f"  ┃   job={meta.get('job_id','')}  s={meta.get('seed','?')}  "
                  f"step={args.step}  desc={_arm_desc(meta)}  "
                  f"budget={budget}{pre}{avail}")

            # RI
            ri_line = _raw_line(r["ri"], w["waiv_ri"], w["base_ri"],
                                 "ri", backbone, args.step)
            print(f"  ┃   RI:    {ri_line}")

            # RI per-dataset
            ds_parts = []
            for dsname in ("tcga", "camelyon", "tolkach_esca"):
                v  = r["ri_datasets"].get(dsname)
                wv = w["waiv_ds"].get(dsname)
                bv = w["base_ds"].get(dsname)
                if v is not None:
                    d_waiv = f"Δ{v - wv:+.4f}" if wv else ""
                    d_base = f"(Δbase{v - bv:+.4f})" if bv else ""
                    ds_parts.append(f"{dsname}={v:.4f}{d_waiv}{d_base}")
                else:
                    ds_parts.append(f"{dsname}=MISSING")
            print(f"  ┃     ds: {' | '.join(ds_parts)}")

            # HEST
            hest_line = _raw_line(r["hest"], w["waiv_hest"], w["base_hest"],
                                   "hest", backbone, args.step)
            print(f"  ┃   HEST: {hest_line}")

            # THUNDER per task — display raw scores; σ annotation only applies to
            # differences vs a reference, so we don't annotate absolute task-means.
            t_parts = []
            for task in THUNDER_TASKS:
                lbl  = r["thunder_labels"].get(task, "MISSING")
                t_parts.append(f"{task[:3]}={lbl}")
            overall_lbl = r["thunder_overall_label"]
            print(f"  ┃   THUNDER: {' | '.join(t_parts)} | mean={overall_lbl}")
            print(f"  ┃")

        # -- Summary across this backbone --------------------------------
        print(f"  ┃  ── {backbone.upper()} summary (mean ± SD of rows with data) ──")
        for metric, key, waiv_key, base_key in [
            ("RI",   "ri",   "waiv_ri",   "base_ri"),
            ("HEST", "hest", "waiv_hest", "base_hest"),
        ]:
            vals = [r[key] for r in bb_rows if r[key] is not None]
            if vals:
                mn = sum(vals) / len(vals)
                sd = (sum((v - mn) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                wv = w[waiv_key]
                bv = w[base_key]
                diff_w = mn - wv
                diff_b = mn - bv
                noise  = _noise_label(diff_w, key, backbone, args.step)
                noise_s = f"({noise})" if noise else ""
                print(f"  ┃    {metric}:  mean={mn:.4f} sd={sd:.4f}  "
                      f"ours={mn:.4f} | Waiv={wv:.4f} | diff={diff_w:+.4f}{noise_s} | "
                      f"Δbase={diff_b:+.4f}  n={len(vals)}")
            else:
                print(f"  ┃    {metric}:  -- no data --")
        print()

    # -- Footer ----------------------------------------------------------
    print("─" * W)
    print("Notes:")
    print("  MISSING       = metric not found for this (run, step); never substituted.")
    print("  PARTIAL(n/N)  = checkpoint has n of N expected datasets; task-mean suppressed.")
    print(f"  ~noise / Xσ  = annotates diff vs Waiv in seed-SD units (2SD threshold).")
    if RI_BUDGET_ENABLED:
        print(f"  budget=PASS/FAIL: RI ≥ floor (Virchow2 ≥ 0.9134, midnight ≥ 0.9140).")
        print(f"  !!! {RI_BUDGET_WARNING}")
    else:
        print(f"  budget=off    : {RI_BUDGET_WARNING}")
    print("  THUNDER noise: MEASURED per (backbone, task) seed SD, "
          f"{min(THUNDER_SEED_SD.values())*100:.2f}-{max(THUNDER_SEED_SD.values())*100:.2f}pp "
          "(was a flat 0.25pp eyeballed proxy, too small by up to 9.3x).")
    print(f"  source: {THUNDER_SEED_SD_SOURCE}")
    print()
    print("HEST base fix (vs old scoreboard.py):")
    print("  Virchow2 base_hest: 0.4034 (old, rounded) → 0.40324 (correct, from")
    print("  vbase_clsmean_summary.json via collect_final5.HEST_BASE['virchow2']).")
    print("  This changes Δbase for all Virchow2 HEST cells by +0.00016.")
    print("─" * W)


if __name__ == "__main__":
    main()
