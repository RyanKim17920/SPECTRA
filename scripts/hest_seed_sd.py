#!/usr/bin/env python3
"""DERIVE the HEST seed SD from disk.  Emits docs/hest_seed_sd.json.

WHY THIS SCRIPT EXISTS (F-A of docs/FORMULA_UNIFICATION_2026-08-26.md)
---------------------------------------------------------------------
`final_recipe_report.HEST_SD_PCT = {phikon 5.8, midnight 8.3, virchow2 14.2}` was a
hand-set literal with no on-disk source and no producing script, and it solely determined
every HEST confidence interval in the final verdict.  It disagreed with both other
estimates of the same quantity in this repo (phikon's literal was 32% NARROWER than
either).  A hand-set error bar that is narrower than the measurement is not an error bar.

THE ESTIMATOR -- copied verbatim from the provenance comment of scoreboard.NOISE_SD
(scripts/scoreboard.py:161-174), which is the estimator the repo already documents:

    "HEST values are POOLED WITHIN-RECIPE seed SDs [...].  Pooling formula:
     sqrt( sum_f (n_f - 1) * sd_f^2 / sum_f (n_f - 1) ) over every recipe family with
     n>=2 seeds at that (backbone, step), same HEST pooling protocol.  Using ONE family
     systematically mis-states the floor [...] so the pooled value is the defensible one.
     df is the pooled degrees of freedom."

Two things are made explicit here that the literal dict left to the reader:

  * "RECIPE FAMILY" IS THE CONFIG, NOT THE RUN NAME.  Two runs are the same family iff
    their configs are IDENTICAL apart from the seed and pure bookkeeping (BOOKKEEPING_KEYS
    below).  Grouping by name prefix would pool runs whose masking / cls-bias / lr differ,
    which inflates a WITHIN-recipe SD with BETWEEN-recipe spread.

    NOT collect_final5.CHECKED_CONFIG_KEYS, deliberately, and this is a DEFECT REPORT:
    nine recipe-DEFINING config keys vary across runs/ but are absent from that list --
    retention_kl_weight (0.0 .. 0.3), mask_sim_thresh, min_tissue_frac, use_lora, grid,
    group_size, n_groups, resume_from -- and two more, `split_heads` and `pool_head`, are
    looked for at `encoder.split_heads` / `encoder.pool_head` while the writer emits them
    at TOP LEVEL, so those two checks pass vacuously (collect_final5 already prints a
    warning for exactly this).  Keying families on that list would have merged e.g. the
    ret0.01 runs with the kl0 runs into one "family" and reported their BETWEEN-recipe
    spread as seed noise.  The full-config key cannot do that.  Fixing
    CHECKED_CONFIG_KEYS itself is deliberately NOT done here: it would change which runs
    collect_final5 excludes from its published aggregates, i.e. it would move measured
    numbers, which this audit is forbidden to do.
  * "SAME HEST POOLING PROTOCOL" is collect_final5.hest_pooling(backbone) -- cls for
    phikon and midnight, clsmean for virchow2 -- and the scalar is
    hest_perf_per_encoder.custom_encoder, read through collect_final5._hest_score so
    base, fine-tuned and floor all come off ONE loader and ONE field.

WHAT IS EMITTED
    docs/hest_seed_sd.json
      pooled_seed_sd[backbone][step] = {sd, df, n_families, n_runs, families:[...]}
    plus, per (backbone, step), the per-family sds so the pooling is auditable.

A (backbone, step) with no family of n>=2 gets sd=None.  That is a real outcome -- the
floor is UNMEASURED there -- and consumers must handle it, not substitute a neighbour.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import collect_final5 as _c5  # noqa: E402

OUT = REPO / "docs" / "hest_seed_sd.json"

BACKBONE_OF = {
    "owkin/phikon-v2": "phikon",
    "kaiko-ai/midnight": "midnight",
    "paige-ai/Virchow2": "virchow2",
}


#: Config keys that do NOT define the experiment: the seed itself, where the run wrote
#: its output, how chatty it was, and machine-local paths.  Everything else -- every
#: knob that could change the learned model -- participates in the family identity.
#: An INCLUSION list would silently pool any knob someone forgets to add; an EXCLUSION
#: list fails the other way, splitting a family and costing degrees of freedom, which is
#: the safe direction for a noise floor.
BOOKKEEPING_KEYS = {
    "seed",            # the thing whose effect we are measuring
    "out_dir",         # run directory
    "log_every", "eval_every", "num_workers",   # logging / dataloading only
    "packed_dir",      # /data/plism/repacked vs /data/ryan.kim/plism/repacked -- the
                       # SAME corpus, relocated 2026-08-09 (91/91 tiles verified intact)
}


def _flat_signature(cfg: dict) -> str:
    """The recipe identity of a run: its FULL config minus BOOKKEEPING_KEYS."""
    def strip(d):
        return {k: (strip(v) if isinstance(v, dict) else v)
                for k, v in sorted(d.items()) if k not in BOOKKEEPING_KEYS}
    return json.dumps(strip(cfg), sort_keys=True, default=str)


def collect(runs_dir: Path):
    """-> {(backbone, step): {signature: {seed_or_run: hest}}} plus a skip ledger."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    skipped = defaultdict(list)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if ".r" in name.rsplit("-", 1)[-1] or name.split(".")[-1].startswith("r") and "." in name:
            # restart sibling: its checkpoints duplicate the parent run's identity
            skipped["restart_sibling"].append(name)
            continue
        cfgp = d / "config.json"
        if not cfgp.exists():
            skipped["no_config"].append(name)
            continue
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            skipped["unreadable_config"].append(name)
            continue
        bb = BACKBONE_OF.get((cfg.get("encoder") or {}).get("backbone"))
        if bb is None:
            skipped["unknown_backbone"].append(name)
            continue
        sig = _flat_signature(cfg)
        pool = _c5.hest_pooling(bb)
        hits = 0
        for p in sorted((_c5.HEST_WORK_DIR / "results").glob(
                "f5_%s_s*_%s_summary.json" % (name, pool))):
            step_tok = p.name[len("f5_%s_s" % name):][:7]
            if not step_tok.isdigit():
                continue
            step = int(step_tok)
            v = _c5._hest_score(name, step, bb)
            if v is None:
                continue
            cells[(bb, step)][sig][name] = v
            hits += 1
        if hits == 0:
            skipped["no_hest_summary"].append(name)
    return cells, skipped


def pool(cells):
    """Pooled within-family SD per (backbone, step).  sqrt(sum df_f sd_f^2 / sum df_f)."""
    out: dict = defaultdict(dict)
    for (bb, step), by_sig in sorted(cells.items()):
        fams = []
        for sig, runs in by_sig.items():
            if len(runs) < 2:
                continue
            vals = list(runs.values())
            fams.append({
                "n": len(vals),
                "df": len(vals) - 1,
                "sd": statistics.stdev(vals),
                "mean": statistics.fmean(vals),
                "runs": sorted(runs),
                "signature_sha": "%08x" % (hash(sig) & 0xFFFFFFFF),
            })
        if not fams:
            out[bb][str(step)] = {
                "sd": None, "df": 0, "n_families": 0, "n_runs": 0, "families": [],
                "reason": "no recipe family with n>=2 seeds at this (backbone, step)",
            }
            continue
        df = sum(f["df"] for f in fams)
        sd = math.sqrt(sum(f["df"] * f["sd"] ** 2 for f in fams) / df) if df else None
        out[bb][str(step)] = {
            "sd": sd, "df": df, "n_families": len(fams),
            "n_runs": sum(f["n"] for f in fams),
            "families": sorted(fams, key=lambda f: -f["n"]),
        }
    return {k: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--write", action="store_true", help="write docs/hest_seed_sd.json")
    args = ap.parse_args()

    cells, skipped = collect(Path(args.runs_dir))
    pooled = pool(cells)
    blob = {
        "estimator": ("pooled WITHIN-RECIPE-FAMILY across-seed SD of the HEST scalar, "
                      "sqrt(sum_f df_f * sd_f^2 / sum_f df_f), df_f = n_f - 1"),
        "estimator_source": ("scripts/scoreboard.py:161-174 provenance comment for "
                             "NOISE_SD -- matched exactly"),
        "family_definition": ("configs identical apart from %s; NOT the run name, and "
                              "NOT collect_final5.CHECKED_CONFIG_KEYS (9 recipe-defining "
                              "keys vary outside it -- see module docstring)"
                              % sorted(BOOKKEEPING_KEYS)),
        "metric_field": _c5.HEST_METRIC_FIELD,
        "pooling_protocol": {a: _c5.hest_pooling(a) for a in ("phikon", "midnight", "virchow2")},
        "hest_work_dir": str(_c5.HEST_WORK_DIR),
        "runs_dir": args.runs_dir,
        "units": "RAW HEST metric (same units as the scalar), 1 SD, one run",
        "pooled_seed_sd": pooled,
        "skipped": {k: sorted(v) for k, v in skipped.items()},
    }
    for bb, per_step in pooled.items():
        for step, c in per_step.items():
            sd = c["sd"]
            print("%-9s step %-5s sd=%s  df=%-3s families=%-2s runs=%s"
                  % (bb, step, ("%.5f" % sd) if sd is not None else "  --   ",
                     c["df"], c["n_families"], c["n_runs"]))
    if args.write:
        OUT.write_text(json.dumps(blob, indent=2))
        print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
