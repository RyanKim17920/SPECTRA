#!/usr/bin/env python
"""Aggregate results for the final5 3-backbone x 5-seed study.

Run from the repo root:
    python scripts/collect_final5.py
    python scripts/collect_final5.py --runs-dir /data/ryan.kim/waiv_runs
    python scripts/collect_final5.py --ri-step 500 --json-out docs/final5_results.json

HEST work dir: /data/ryan.kim/hest_work  (H.DEFAULT_WORK_DIR)
THUNDER base:  /data/ryan.kim/thunder    ($THUNDER_BASE_DATA_FOLDER)

Preemption: if a run dir has sibling dirs <name>.r1, <name>.r2, ... the RI curves
are unioned (deduplication by step, latest restart wins per step).  The run is flagged
PREEMPTED in the output.

Config comparison: every run is checked against the MODAL config for the following keys.
Any run whose value differs from the mode is flagged NOT COMPARABLE and excluded from
aggregates.  The check is the single most important function of this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# BASE HEST SCORES (unfinetuned backbone, per-backbone pooling protocol)
# All values measured by us under our own protocol; source files noted below.
# ---------------------------------------------------------------------------
# IMPORTANT: midnight's HEST pooling (cls) DIFFERS from its THUNDER pooling
# (clsmean for classification, cls for segmentation). These are separate
# protocols for separate benchmarks. Do NOT assume one pooling per backbone
# across both benchmarks.
# NOTE (F6 fix, 2026-08-26): these literals are now only a FALLBACK for when the
# summary JSON is not on disk.  The authoritative base is read from disk by
# _load_hest_base() below, from the SAME field (hest_perf_per_encoder.custom_encoder)
# that _hest_score() reads for fine-tuned runs.  Previously these literals were the
# rounded `results.avg` field while fine-tuned scores came from `custom_encoder`,
# so base and FT came from DIFFERENT fields (virchow2 base was low by 2.4e-5).
HEST_BASE_FALLBACK = {
    # phikon-v2: cls pooling
    # source: results_backup/hest_work_results/base_cls_summary.json
    "phikon":   0.37470,
    # midnight: cls pooling  (NOT clsmean — our HEST protocol for midnight is cls)
    # source: /data/ryan.kim/hest_work/results/midnight_base_cls_9task_v1_summary.json
    "midnight": 0.39521,
    # virchow2: clsmean pooling
    # source: results_backup/hest_work_results/vbase_clsmean_summary.json
    # Note: the file stores 0.40324; docs/FINAL_RESULTS.md quotes 0.40327.
    # We use the value in the file (0.40324) as ground truth.
    "virchow2": 0.40324,
}
# WRONG-PROTOCOL VALUES — MUST NEVER BE USED AS A BASE:
#   phikon-v2 clsmean 0.39144  (base_clsmean_summary.json) — wrong pooling for phikon HEST
#   midnight  clsmean 0.41210  (mbase_clsmean_summary.json) — wrong pooling for midnight HEST
#   any value from results_backup/hest_sub5/ — 5-task subset, wrong task count AND wrong protocol
#   any value read from the `results.avg` field — that field is rounded and is NOT the
#     field the fine-tuned collector reads; mixing the two biases pct_of_waiv.

# ---------------------------------------------------------------------------
# RI BASE (avg_ri for the untuned backbone, checkpoint=None adapter=None)
# Computed over cross_scanner and cross_stain separation; this is NOT the
# mean top1 from the same probe_before.json (~0.75/0.64/0.78), which is a
# different aggregation. Values from the finalgem probe_before.json runs.
# ---------------------------------------------------------------------------
RI_BASE = {
    "phikon":   0.4686,
    "midnight": 0.7589,
    "virchow2": 0.8582,
}

# ---------------------------------------------------------------------------
# THUNDER: base model directory names, per backbone and per task kind.
#
# NOTE: The base results are SPLIT across two model dirs per backbone for
# midnight and Virchow2, because segmentation uses cls pooling while
# classification uses clsmean. The FT runs use ONE model dir for all task
# kinds (the dir name does NOT encode pooling for FT runs). These two layouts
# are ASYMMETRIC and must not be conflated.
#
# IMPORTANT: midnight's THUNDER classification pooling (clsmean) differs from
# its HEST pooling (cls). These are separate protocols for separate benchmarks.
#
# phikon-v2 : classification -> base_cls        , segmentation -> base_cls
# midnight  : classification -> mbase_clsmean   , segmentation -> mbase_cls
# virchow2  : classification -> vbase_clsmean   , segmentation -> vbase_cls
#
# Base numbers are read from disk (same parsing path as FT runs) so that
# base and FT values go through identical code. If a base result file is
# absent, the delta is None / "no base".
# ---------------------------------------------------------------------------
THUNDER_BASE_DIRS: dict[str, dict[str, str]] = {
    "phikon":   {"cls": "base_cls",      "seg": "base_cls"},
    "midnight": {"cls": "mbase_clsmean", "seg": "mbase_cls"},
    "virchow2": {"cls": "vbase_clsmean", "seg": "vbase_cls"},
}

# HEST work dir (run_hest.py default = H.DEFAULT_WORK_DIR)
HEST_WORK_DIR = Path(os.environ.get("HEST_WORK_DIR", "/data/ryan.kim/hest_work"))

# ---------------------------------------------------------------------------
# HEST single source of truth (F6/F9 fix, 2026-08-26)
# ---------------------------------------------------------------------------
# ONE metric field, ONE pooling rule, ONE base loader -- used by every consumer
# (collect_final5, scoreboard, final_recipe_report).  The whole repo must read the
# same field for base and for fine-tuned, or pct_of_waiv is biased.
HEST_METRIC_FIELD = "hest_perf_per_encoder.custom_encoder"

HEST_BASE_FILES = {
    "phikon":   "base_cls_summary.json",
    "midnight": "midnight_base_cls_9task_v1_summary.json",
    "virchow2": "vbase_clsmean_summary.json",
}


def hest_pooling(arm: str) -> str:
    """HEST pooling protocol per backbone.  The ONLY definition in the repo.

    NOT the same as the THUNDER pooling rule -- see _thunder_pooling in scoreboard.py
    and THUNDER_BASE_DIRS below.  midnight is `cls` on HEST but `clsmean` on THUNDER
    classification, so the two rules must never be shared.
    """
    return "clsmean" if arm == "virchow2" else "cls"


def _hest_summary_paths(fname: str):
    repo = Path(__file__).resolve().parents[1]
    return (
        HEST_WORK_DIR / "results" / fname,
        repo / "results_backup" / "hest_work_results" / fname,
    )


def _hest_read_metric(path) -> float | None:
    """Read the ONE authoritative HEST scalar out of a summary JSON."""
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return None
    return (d.get("hest_perf_per_encoder") or {}).get("custom_encoder")


def _load_hest_base():
    """Authoritative per-backbone HEST base, read FROM DISK, same field as FT."""
    vals, src = {}, {}
    for arm, fname in HEST_BASE_FILES.items():
        for cand in _hest_summary_paths(fname):
            if cand.exists():
                v = _hest_read_metric(cand)
                if v is not None:
                    vals[arm], src[arm] = v, str(cand)
                    break
        if arm not in vals:
            vals[arm] = HEST_BASE_FALLBACK[arm]
            src[arm] = "FALLBACK LITERAL (summary JSON absent or unreadable)"
    return vals, src


HEST_BASE, HEST_BASE_SOURCE = _load_hest_base()
# THUNDER root
THUNDER_ROOT = Path(os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder"))

# THUNDER dataset lists (from collect_thunder.py)
PAPER_CLS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist",
    "patch_camelyon", "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]
# ---------------------------------------------------------------------------
# SEGMENTATION ROSTER -- SINGLE OWNER (2026-08-26)
# ---------------------------------------------------------------------------
# This module is the ONLY definition of the segmentation panels.  collect_thunder.py
# imports them from here.  Until today the two modules each defined a symbol spelled
# `PAPER_SEG` with DIFFERENT contents (4 there, 2 here), so which panel a consumer got
# depended on which module it happened to import, and two different quantities were
# printed under one label.
#
# PAPER_SEG_PUBLISHED -- Waiv's published 4-dataset segmentation panel (arXiv:2607.22861).
# PAPER_SEG_SUBMITTED -- the 2 datasets we run on every exploratory checkpoint.
# PAPER_SEG           -- the default panel our collectors average over = SUBMITTED.
#
# WHY THE DEFAULT IS 2, NOT 4 -- CORRECTED JUSTIFICATION.
# The previous comment here claimed segpath was excluded because "midnight has no base
# result for either".  That claim is FALSE and is retracted.  All six segpath base cells
# exist on disk with real F1 and were produced with the mandated epoch overrides:
#   segpath_epithelial  (epochs=9):  base_cls 0.69459  mbase_cls 0.70949  vbase_cls 0.70639
#   segpath_lymphocytes (epochs=21): base_cls 0.60649  mbase_cls 0.63755  vbase_cls 0.63172
# under $THUNDER_BASE_DATA_FOLDER/outputs/res/<ds>/<model>/segmentation/frozen/.
#
# The real reason is COST.  Measured wall-clock of a single segpath segmentation job
# (sacct, jobs 369825/369827/369913/369915/369916/375909/375910) is 27-32 h -- roughly a
# day and a quarter per (checkpoint x dataset) cell, versus ~1 h for ocelot/pannuke.  The
# 2-dataset panel is therefore the deliberate operating point for exploratory cohorts, and
# segpath is run ONCE, on the final locked configuration, as a last-case evaluation.
#
# CONSEQUENCE THAT MUST BE STATED WHEREVER SEGMENTATION IS COMPARED TO WAIV:
# our segmentation mean has 2-dataset support and theirs has 4.  That is a support
# mismatch, not a like-for-like delta, and it stays one until the final segpath run lands.
# Flipping this default to PAPER_SEG_PUBLISHED before then would not fix the comparison --
# it would only mark every exploratory segmentation cell PARTIAL.
PAPER_SEG_PUBLISHED = ["ocelot", "pannuke", "segpath_epithelial", "segpath_lymphocytes"]
PAPER_SEG_SUBMITTED = ["ocelot", "pannuke"]
PAPER_SEG = PAPER_SEG_SUBMITTED
THUNDER_TASKS = ["knn", "linear_probing", "simple_shot", "segmentation"]

# Config keys that MUST be identical across all runs (except seed and backbone).
# Values are read flat; encoder sub-keys are prefixed "encoder.".
CHECKED_CONFIG_KEYS = [
    "grid_conditions", "grid_tiles", "grid_forward_chunk",
    "max_steps", "ckpt_every", "lr", "temperature",
    "warmup_steps", "activation_offload",
    "encoder.split_heads", "encoder.pool_head",
    "cls_weight", "mean_weight",           # TOP-LEVEL keys, NOT inside encoder
    "encoder.grad_checkpointing",
    "encoder.lora_rank", "encoder.lora_alpha",
    # F11 fix (2026-08-26): recipe-DEFINING knobs that were previously unchecked, so
    # two runs with OPPOSITE negative-masking / cls-bias settings were pooled as
    # COMPARABLE.  Any of these differing makes them different experiments.
    "mask_same_core",
    "same_core_logit_bias",       # direct sibling of the two below; unchecked it would
    "same_core_logit_bias_cls",   # leave one masking knob free while checking the rest
    "same_core_logit_bias_mean",
    "weight_decay",
    "center_embeddings",
    "cores_per_batch",
    "grad_accum",
    "grad_clip",
    "ckpt_schedule",
    "core_labels_path",
    "encoder.proj_out_dim",
    "encoder.pooling",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_config(cfg: dict) -> dict:
    """Flatten config.json to the checked keys."""
    enc = cfg.get("encoder", {})
    row: dict = {}
    for k in CHECKED_CONFIG_KEYS:
        if k.startswith("encoder."):
            sub = k[len("encoder."):]
            row[k] = enc.get(sub)
        else:
            row[k] = cfg.get(k)
    return row


def _modal(rows: list[dict]) -> dict:
    """Return the modal value for each key across a list of flat config dicts."""
    result = {}
    for k in CHECKED_CONFIG_KEYS:
        vals = [json.dumps(r[k], sort_keys=True) for r in rows if k in r]
        if not vals:
            result[k] = None
        else:
            result[k] = json.loads(Counter(vals).most_common(1)[0][0])
    return result


def _parse_run_name(name: str) -> tuple[str, int, str] | None:
    """Parse 'final5-<arm>-s<seed>-t<T>-<jobid>' -> (arm, seed, T) or None."""
    # strip any restart suffix like .r1, .r2 for the base name
    base = name.split(".r")[0]
    parts = base.split("-")
    if len(parts) < 5 or parts[0] != "final5":
        return None
    arm = parts[1]
    seed_part = parts[2]
    t_part = parts[3]
    if not seed_part.startswith("s") or not t_part.startswith("t"):
        return None
    try:
        seed = int(seed_part[1:])
        t_val = t_part[1:]
    except ValueError:
        return None
    return arm, seed, t_val


def _union_ri_curves(run_dir: Path) -> tuple[list[dict], bool]:
    """Union ri_curve.json points across the main dir and any .r1/.r2 siblings.

    Returns (points_list, was_preempted).
    Points are deduplicated by step; the restart with the highest suffix wins
    (i.e. later restarts are preferred over earlier ones for the same step).
    """
    parent = run_dir.parent
    base_name = run_dir.name
    # collect restart dirs: base, base.r1, base.r2, ...
    candidates: list[tuple[int, Path]] = [(0, run_dir)]
    i = 1
    while True:
        cand = parent / f"{base_name}.r{i}"
        if cand.is_dir():
            candidates.append((i, cand))
            i += 1
        else:
            break
    preempted = len(candidates) > 1
    # build step -> point mapping; higher restart index wins
    by_step: dict[int, dict] = {}
    for restart_idx, d in candidates:
        ri_file = d / "ri_curve.json"
        if not ri_file.exists():
            continue
        try:
            blob = json.loads(ri_file.read_text())
        except Exception:
            continue
        for pt in blob.get("points", []):
            step = pt.get("step")
            if step is None:
                continue
            if step not in by_step or restart_idx >= by_step[step].get("_restart_idx", -1):
                pt = dict(pt)
                pt["_restart_idx"] = restart_idx
                by_step[step] = pt
    points = [v for v in sorted(by_step.values(), key=lambda x: x.get("step", 0))]
    return points, preempted


def _ri_at_step(points: list[dict], step: int) -> float | None:
    for pt in points:
        if pt.get("step") == step:
            return pt.get("avg_robustness_index")
    return None


def _ri_argmax(points: list[dict]) -> tuple[float | None, int | None]:
    best_ri, best_step = None, None
    for pt in points:
        ri = pt.get("avg_robustness_index")
        s = pt.get("step")
        if ri is not None and (best_ri is None or ri > best_ri):
            best_ri, best_step = ri, s
    return best_ri, best_step


def _hest_score(run_name: str, step: int, arm: str) -> float | None:
    """Read HEST score for a (run, step) from the results summary JSON."""
    pooling = hest_pooling(arm)
    step_str = f"{step:07d}"
    exp_code = f"f5_{run_name}_s{step_str}_{pooling}"
    # Primary: live hest_work dir
    summary_path = HEST_WORK_DIR / "results" / f"{exp_code}_summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            enc = d.get("hest_perf_per_encoder", {})
            return enc.get("custom_encoder")
        except Exception:
            pass
    # Fallback: results_backup dirs
    repo = Path(__file__).resolve().parents[1]
    for backup in (
        repo / "results_backup" / "hest_work_results" / f"{exp_code}_summary.json",
        # results_backup/hest_sub5/ deliberately NOT searched (F12 fix, 2026-08-26):
        # it holds a 5-task subset with a byte-identical schema, so a hit there is
        # silently indistinguishable from a real 9-task score.
    ):
        if backup.exists():
            try:
                d = json.loads(backup.read_text())
                enc = d.get("hest_perf_per_encoder", {})
                return enc.get("custom_encoder")
            except Exception:
                pass
    return None


def _thunder_score_by_model(model: str) -> dict[str, float | None]:
    """Return per-task F1 means for a given model dir name.

    FT runs: model = f5_<run_name>_s<step_str>  (all task kinds in same dir)
    Base runs: caller should pass the correct per-(backbone,task-kind) dir name;
               use _thunder_base_score() for that.
    """
    per_ds = _thunder_per_ds_by_model(model, cls_model=model, seg_model=model)
    results: dict[str, float | None] = {t: None for t in THUNDER_TASKS}
    for task in THUNDER_TASKS:
        vals = [v for v in per_ds[task].values() if v is not None]
        if vals:
            results[task] = sum(vals) / len(vals)
    return results


def _thunder_per_ds_by_model(model: str,
                              cls_model: str | None = None,
                              seg_model: str | None = None,
                              ) -> dict[str, dict[str, float | None]]:
    """Return per-dataset F1 for each task, keyed by dataset name.

    If cls_model/seg_model are provided, they override `model` for those task kinds
    (used for base models which split pooling across two dirs).  When None, `model`
    is used for both.
    Returns dict[task, dict[dataset, float|None]].  Missing dataset files are absent
    from the inner dict (not silently None-filled) so callers can distinguish
    "not evaluated yet" from "evaluated but null".
    """
    res_root = THUNDER_ROOT / "outputs" / "res"
    results: dict[str, dict[str, float | None]] = {t: {} for t in THUNDER_TASKS}
    if not res_root.exists():
        return results
    for task in THUNDER_TASKS:
        m = (seg_model if task == "segmentation" else cls_model) or model
        datasets = PAPER_CLS if task != "segmentation" else PAPER_SEG
        for ds in datasets:
            p = res_root / ds / m / task / "frozen" / "outputs.json"
            if not p.exists():
                continue
            try:
                blob = json.loads(p.read_text())
            except Exception:
                continue
            f1 = _thunder_f1(blob, task)
            if f1 is not None:
                results[task][ds] = f1
    return results


def _thunder_score(run_name: str, step: int) -> dict[str, float | None]:
    """Return per-task F1 means for a FT (run, step).

    FT runs land under ONE model dir for all task kinds (pooling is internal).
    Model dir name: f5_<run_name>_s<step_str>  (7-digit zero-padded step).
    F1 values from outputs.json are fractions; returned as fractions.
    """
    step_str = f"{step:07d}"
    model = f"f5_{run_name}_s{step_str}"
    return _thunder_score_by_model(model)


def _thunder_ft_per_ds(run_name: str, step: int) -> dict[str, dict[str, float | None]]:
    """Return per-dataset F1 for a FT (run, step).  Returns dict[task, dict[ds, float]]."""
    step_str = f"{step:07d}"
    model = f"f5_{run_name}_s{step_str}"
    return _thunder_per_ds_by_model(model)


def _thunder_base_score(arm: str) -> dict[str, float | None]:
    """Return per-task F1 means for the BASE (unfinetuned) checkpoint.

    Base results are split across two model dirs per backbone because segmentation
    uses cls pooling while classification uses clsmean (for midnight and Virchow2).
    The per-(backbone, task-kind) dir mapping is in THUNDER_BASE_DIRS.
    If a result file is absent, the task entry is None (no silent zero-fill).
    """
    per_ds = _thunder_base_per_ds(arm)
    results: dict[str, float | None] = {t: None for t in THUNDER_TASKS}
    for task in THUNDER_TASKS:
        vals = [v for v in per_ds[task].values() if v is not None]
        if vals:
            results[task] = sum(vals) / len(vals)
    return results


def _thunder_base_per_ds(arm: str) -> dict[str, dict[str, float | None]]:
    """Return per-dataset F1 for the BASE checkpoint.  Returns dict[task, dict[ds, float]]."""
    dirs = THUNDER_BASE_DIRS.get(arm)
    if dirs is None:
        return {t: {} for t in THUNDER_TASKS}
    cls_model = dirs["cls"]
    seg_model = dirs["seg"]
    return _thunder_per_ds_by_model(
        model=cls_model,
        cls_model=cls_model,
        seg_model=seg_model,
    )


def _thunder_f1(blob: dict, task: str) -> float | None:
    """Extract F1 from a single outputs.json (mirrors collect_thunder.py logic)."""
    def _flat(d: dict) -> float | None:
        f1 = d.get("f1", {})
        if isinstance(f1, dict):
            return f1.get("metric_score")
        return f1

    if task in ("linear_probing", "segmentation"):
        return _flat(blob)
    keys = [k for k in blob if k.isdigit()]
    if not keys:
        return _flat(blob)
    if task == "knn":
        key = keys[0] if len(keys) == 1 else max(keys, key=int)
    else:  # simple_shot -> shot=16
        key = "16" if "16" in keys else max(keys, key=int)
    return _flat(blob[key])


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _agg(vals: list[float | None]) -> dict:
    good = [v for v in vals if v is not None]
    n = len(good)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "seed_floor_2sd_sqrtn": None}
    mean = sum(good) / n
    # ddof=1: sample SD (unbiased estimator of seed-to-seed variability).
    # ddof=0 (population SD) systematically understates the noise floor — do not use.
    # n < 2: SD and floor are undefined — return None so display shows "--" not 0.00000.
    sd = statistics.stdev(good) if n > 1 else None
    floor = 2 * sd / math.sqrt(n) if (sd is not None and n > 0) else None
    return {"n": n, "mean": mean, "sd": sd, "seed_floor_2sd_sqrtn": floor}


def _delta(val: float | None, base: float | None) -> float | None:
    if val is None or base is None:
        return None
    return val - base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=None,
                    help="Path to runs/ (default: repo/runs, which is a symlink to "
                         "/data/ryan.kim/waiv_runs)")
    ap.add_argument("--ri-step", type=int, default=500,
                    help="Fixed RI step to report as primary (default 500)")
    ap.add_argument("--json-out", default="docs/final5_results.json",
                    help="Output JSON path (relative to repo root)")
    ap.add_argument("--allow-partial-thunder", action="store_true",
                    help="Show THUNDER deltas even when the FT run has fewer datasets "
                         "than the base. By default partial deltas are suppressed and "
                         "labelled PARTIAL n/N to avoid misleading comparisons.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runs_dir = Path(args.runs_dir) if args.runs_dir else (repo / "runs")
    json_out = repo / args.json_out

    if not runs_dir.exists():
        print(f"[WARN] runs dir does not exist: {runs_dir}")
        run_dirs = []
    else:
        # Discover final5 run dirs (exclude restart siblings — they're handled inside)
        all_entries = sorted(runs_dir.iterdir())
        run_dirs = []
        for d in all_entries:
            if not d.is_dir():
                continue
            name = d.name
            # skip restart siblings; they are unioned inside _union_ri_curves
            if ".r" in name and name.split(".r")[-1].isdigit():
                continue
            parsed = _parse_run_name(name)
            if parsed is None:
                continue
            run_dirs.append(d)

    print(f"Discovered {len(run_dirs)} final5 run dir(s) in {runs_dir}")

    if not run_dirs:
        print("No final5 runs found — nothing to aggregate (this is expected if runs "
              "have not started yet).")
        # Still write an empty JSON so downstream tools don't crash.
        payload = {"runs": [], "aggregates": {}, "config_modal": None,
                   "config_failures": [], "ri_step": args.ri_step}
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote empty results to {json_out}")
        return

    # ------------------------------------------------------------------
    # Step 1: Load configs and compute modal config
    # ------------------------------------------------------------------
    run_configs: list[tuple[Path, dict, dict]] = []  # (dir, raw_cfg, flat_cfg)
    for d in run_dirs:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            print(f"[WARN] no config.json in {d.name} -- skipping config check")
            run_configs.append((d, {}, {}))
            continue
        try:
            raw = json.loads(cfg_path.read_text())
        except Exception as e:
            print(f"[WARN] could not parse config.json in {d.name}: {e}")
            run_configs.append((d, {}, {}))
            continue
        run_configs.append((d, raw, _flat_config(raw)))

    flat_cfgs = [fc for _, _, fc in run_configs if fc]
    modal_cfg = _modal(flat_cfgs) if flat_cfgs else {}

    print(f"\nModal config across {len(flat_cfgs)} run(s):")
    none_keys = []
    for k, v in modal_cfg.items():
        print(f"  {k}: {v}")
        if v is None:
            none_keys.append(k)
    if none_keys:
        print()
        print("=" * 70)
        print("  WARNING: THE FOLLOWING CONFIG KEYS ARE ABSENT / None IN ALL RUNS.")
        print("  Their identity check passes VACUOUSLY — they verify NOTHING.")
        print("  Any run deviation on these keys will be silently missed.")
        for k in none_keys:
            print(f"    {k}")
        print("  FIX: Find the correct key path in config.json and update")
        print("  CHECKED_CONFIG_KEYS. Until fixed, these columns are unverified.")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Step 2: Per-run extraction
    # ------------------------------------------------------------------
    run_records: list[dict] = []

    for run_dir, raw_cfg, flat_cfg in run_configs:
        name = run_dir.name
        parsed = _parse_run_name(name)
        arm, seed, t_val = parsed  # already validated above

        record: dict = {
            "run_name": name,
            "arm": arm,
            "seed": seed,
            "T": t_val,
            "status": "PENDING",
            "config_ok": None,
            "config_diffs": [],
            "preempted": False,
        }

        # Config comparison
        if flat_cfg and modal_cfg:
            diffs = []
            for k in CHECKED_CONFIG_KEYS:
                modal_v = modal_cfg.get(k)
                run_v = flat_cfg.get(k)
                if json.dumps(run_v, sort_keys=True) != json.dumps(modal_v, sort_keys=True):
                    diffs.append({"key": k, "run": run_v, "modal": modal_v})
            record["config_ok"] = (len(diffs) == 0)
            record["config_diffs"] = diffs
            if diffs:
                print(f"\n[!!! NOT COMPARABLE !!!] {name}")
                for diff in diffs:
                    print(f"    {diff['key']}: run={diff['run']}  modal={diff['modal']}")

        # RI curve
        points, preempted = _union_ri_curves(run_dir)
        record["preempted"] = preempted
        if preempted:
            print(f"[PREEMPTED] {name}: unioned across restart dirs")

        if not points:
            print(f"[PENDING]   {name}: no ri_curve.json yet")
            run_records.append(record)
            continue

        record["status"] = "OK"

        # Primary RI at fixed step
        ri_fixed = _ri_at_step(points, args.ri_step)
        # Secondary: argmax
        ri_argmax, ri_argmax_step = _ri_argmax(points)

        record["ri_step"] = args.ri_step
        record["ri"] = ri_fixed           # PRIMARY
        record["ri_argmax"] = ri_argmax   # secondary
        record["ri_argmax_step"] = ri_argmax_step

        if ri_fixed is None:
            print(f"[PENDING-RI] {name}: ri_curve.json exists but no step {args.ri_step} yet "
                  f"(steps present: {[p.get('step') for p in points]})")

        # HEST
        record["hest"] = _hest_score(name, args.ri_step, arm)

        # THUNDER — store both the per-task means and the per-dataset breakdown
        thunder = _thunder_score(name, args.ri_step)
        record["thunder"] = thunder
        record["thunder_per_ds"] = _thunder_ft_per_ds(name, args.ri_step)

        run_records.append(record)

    # ------------------------------------------------------------------
    # Step 3: Exclude NOT COMPARABLE runs from aggregates
    # ------------------------------------------------------------------
    comparable = [r for r in run_records if r.get("config_ok") is not False]
    excluded = [r["run_name"] for r in run_records if r.get("config_ok") is False]
    if excluded:
        print(f"\n[EXCLUDED from aggregates - config mismatch]: {excluded}")

    # ------------------------------------------------------------------
    # Step 4: Aggregate per arm
    # ------------------------------------------------------------------
    arms = ["phikon", "midnight", "virchow2"]
    aggregates: dict[str, dict] = {}

    for arm in arms:
        arm_runs = [r for r in comparable if r["arm"] == arm and r["status"] == "OK"]
        ri_vals    = [r.get("ri") for r in arm_runs]
        hest_vals  = [r.get("hest") for r in arm_runs]
        thunder_per_task: dict[str, list[float | None]] = {t: [] for t in THUNDER_TASKS}
        for r in arm_runs:
            for t in THUNDER_TASKS:
                thunder_per_task[t].append(r.get("thunder", {}).get(t))

        ri_agg    = _agg(ri_vals)
        hest_agg  = _agg(hest_vals)
        t_agg     = {t: _agg(thunder_per_task[t]) for t in THUNDER_TASKS}

        hest_base  = HEST_BASE.get(arm)
        ri_base    = RI_BASE.get(arm)
        ri_delta   = _delta(ri_agg["mean"], ri_base)
        hest_delta = _delta(hest_agg["mean"], hest_base)

        # Read base THUNDER from disk (per-dataset, per-(backbone,task-kind) model dirs).
        thunder_base_per_ds = _thunder_base_per_ds(arm)
        # Total dataset counts per task (from the base, which has the full set when done)
        total_ds_counts = {
            t: len(PAPER_SEG if t == "segmentation" else PAPER_CLS)
            for t in THUNDER_TASKS
        }

        # Compute per-dataset deltas: only where BOTH ft and base have a value for that ds.
        # Average those per-dataset deltas. Track coverage (shared_ds / total_ds).
        t_delta: dict[str, float | None] = {}
        t_delta_ds_count: dict[str, int] = {}   # number of datasets that contributed
        t_delta_partial: dict[str, bool] = {}    # True if coverage < total

        for t in THUNDER_TASKS:
            base_ds = thunder_base_per_ds[t]   # dict[ds, float]
            total_ds = total_ds_counts[t]
            # Collect per-run per-dataset deltas, then average across runs then datasets.
            # For each dataset that has both ft and base, compute mean ft value across runs,
            # then delta vs base.
            shared_ds_set: set[str] = set(base_ds.keys())
            # Also intersect with datasets that at least one run has evaluated
            ft_ds_seen: set[str] = set()
            for r in arm_runs:
                ft_ds_seen |= set(r.get("thunder_per_ds", {}).get(t, {}).keys())
            shared_ds_set &= ft_ds_seen

            per_ds_deltas: list[float] = []
            for ds in shared_ds_set:
                base_val = base_ds.get(ds)
                if base_val is None:
                    continue
                # Average ft value across all runs that have this dataset
                ft_vals_for_ds = [
                    r["thunder_per_ds"][t][ds]
                    for r in arm_runs
                    if ds in r.get("thunder_per_ds", {}).get(t, {})
                ]
                if not ft_vals_for_ds:
                    continue
                ft_mean_for_ds = sum(ft_vals_for_ds) / len(ft_vals_for_ds)
                per_ds_deltas.append(ft_mean_for_ds - base_val)

            n_shared = len(per_ds_deltas)
            t_delta_ds_count[t] = n_shared
            t_delta_partial[t] = (n_shared < total_ds)

            if n_shared == 0:
                t_delta[t] = None
            else:
                t_delta[t] = sum(per_ds_deltas) / n_shared

        thunder_base_vals = _thunder_base_score(arm)  # for JSON output only

        aggregates[arm] = {
            "n_runs": len(arm_runs),
            "ri": {**ri_agg, "base": ri_base, "delta_vs_base": ri_delta,
                   "note": ("avg_ri is computed over cross_scanner/cross_stain separation; "
                            "it is NOT the mean top1 from probe_before.json (~0.75/0.64/0.78), "
                            "which is a different aggregation.")},
            "hest": {**hest_agg, "base": hest_base, "delta_vs_base": hest_delta},
            "thunder": {t: {**t_agg[t],
                             "base": thunder_base_vals[t],
                             "delta_vs_base": t_delta[t],
                             "delta_ds_count": t_delta_ds_count[t],
                             "delta_ds_total": total_ds_counts[t],
                             "delta_partial": t_delta_partial[t]}
                        for t in THUNDER_TASKS},
        }

    # ------------------------------------------------------------------
    # Step 5: Print human-readable table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  FINAL5 RESULTS  |  RI step={args.ri_step}  |  "
          f"{len(comparable)} comparable runs  |  {len(excluded)} excluded")
    print("=" * 78)

    print(f"\n{'Run':<45} {'arm':<10} {'s':>2}  {'RI@500':>8}  "
          f"{'RI_argmax':>9}  {'argmax_step':>11}  {'HEST':>7}")
    print("-" * 100)
    for r in sorted(run_records, key=lambda x: (x["arm"], x["seed"])):
        excl = " [EXCL]" if r.get("config_ok") is False else ""
        pend = " [PENDING]" if r["status"] == "PENDING" else ""
        pre  = " [PRE]"  if r.get("preempted") else ""
        ri_str    = f"{r['ri']:.4f}"    if r.get("ri")    is not None else "--"
        rmax_str  = f"{r['ri_argmax']:.4f}" if r.get("ri_argmax") is not None else "--"
        rmaxs_str = str(r.get("ri_argmax_step") or "--")
        hest_str  = f"{r['hest']:.4f}" if r.get("hest") is not None else "--"
        print(f"{r['run_name']:<45} {r['arm']:<10} {r['seed']:>2}  "
              f"{ri_str:>8}  {rmax_str:>9}  {rmaxs_str:>11}  {hest_str:>7}"
              f"{excl}{pend}{pre}")

    # N is shown PER METRIC (non-null count used for that statistic).
    # A single row-level N is intentionally absent: RI, HEST, and each THUNDER
    # task can have different availability, so one N would be misleading.
    print(f"\n\n{'Arm':<12} {'RI n':>5}  {'RI mean':>8}  {'RI SD':>8}  {'RI floor':>9}  "
          f"{'HEST n':>6}  {'HEST mean':>10}  {'HEST Δbase':>10}  {'HEST SD':>8}")
    print("-" * 95)
    for arm in arms:
        ag = aggregates[arm]
        ri = ag["ri"]
        hs = ag["hest"]
        ri_n     = ri["n"]
        hs_n     = hs["n"]
        ri_mean  = f"{ri['mean']:.5f}"  if ri["mean"]  is not None else "--"
        ri_sd    = f"{ri['sd']:.5f}"    if ri["sd"]    is not None else "--"
        ri_fl    = f"{ri['seed_floor_2sd_sqrtn']:.5f}" if ri["seed_floor_2sd_sqrtn"] is not None else "--"
        hs_mean  = f"{hs['mean']:.4f}"  if hs["mean"]  is not None else "--"
        hs_delt  = f"{hs['delta_vs_base']:+.4f}" if hs["delta_vs_base"] is not None else "--"
        hs_sd    = f"{hs['sd']:.4f}"    if hs["sd"]    is not None else "--"
        print(f"{arm:<12} {ri_n:>5}  {ri_mean:>8}  {ri_sd:>8}  {ri_fl:>9}  "
              f"{hs_n:>6}  {hs_mean:>10}  {hs_delt:>10}  {hs_sd:>8}")

    print(f"\n\nTHUNDER per-task means  (n = non-null run count per task; Δ computed per-dataset then averaged)")
    print(f"{'Arm':<12}  {'kNN(n)':>8}  {'LP(n)':>8}  {'SS(n)':>8}  {'Seg(n)':>8}  "
          f"{'kNN Δ':>18}  {'LP Δ':>18}  {'SS Δ':>18}  {'Seg Δ':>18}")
    print("-" * 130)
    for arm in arms:
        ag   = aggregates[arm]
        thr  = ag["thunder"]
        def _fm(d):
            v = d.get("mean")
            n = d.get("n", 0)
            s = f"{v:.4f}" if v is not None else "--"
            return f"{s}({n})"
        def _fd(d):
            v         = d.get("delta_vs_base")
            n_ds      = d.get("delta_ds_count", 0)
            total_ds  = d.get("delta_ds_total", 0)
            partial   = d.get("delta_partial", False)
            coverage  = f"{n_ds}/{total_ds}ds"
            if v is None:
                return f"-- ({coverage})"
            if partial and not args.allow_partial_thunder:
                return f"PARTIAL {coverage}"
            marker = "*" if partial else ""
            return f"{v:+.4f}{marker} ({coverage})"
        print(f"{arm:<12}  "
              f"{_fm(thr['knn']):>8}  "
              f"{_fm(thr['linear_probing']):>8}  "
              f"{_fm(thr['simple_shot']):>8}  "
              f"{_fm(thr['segmentation']):>8}  "
              f"{_fd(thr['knn']):>18}  "
              f"{_fd(thr['linear_probing']):>18}  "
              f"{_fd(thr['simple_shot']):>18}  "
              f"{_fd(thr['segmentation']):>18}")
    if any(aggregates[arm]["thunder"][t].get("delta_partial")
           for arm in arms for t in THUNDER_TASKS):
        print("  NOTE: PARTIAL rows have fewer ft datasets than the base; "
              "deltas suppressed by default. Use --allow-partial-thunder to show them.")

    # ------------------------------------------------------------------
    # Step 6: Write JSON
    # ------------------------------------------------------------------
    payload = {
        "ri_step_primary": args.ri_step,
        "note_ri_argmax": (
            "ri_argmax and ri_argmax_step are SECONDARY -- do not treat them as the "
            "headline. The primary metric is ri at the fixed step above."
        ),
        "config_modal": modal_cfg,
        "config_failures": excluded,
        "runs": run_records,
        "aggregates": aggregates,
        "hest_base": HEST_BASE,
        "ri_base": RI_BASE,
        "thunder_base_dirs": THUNDER_BASE_DIRS,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote results to {json_out}")

    if excluded:
        print(f"\n[SUMMARY] {len(excluded)} run(s) excluded from aggregates due to "
              f"config mismatch. Fix the config or explicitly remove these runs.")


if __name__ == "__main__":
    main()
