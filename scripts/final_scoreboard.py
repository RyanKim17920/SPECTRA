#!/usr/bin/env python3
"""final_scoreboard.py -- ONE command that regenerates the paper's final results table.

    ./.venv/bin/python scripts/final_scoreboard.py

Writes  docs/final_scoreboard.md  (and refreshes docs/final_recipe_verdict.json, which
is the machine-readable form of section 1).  Nothing on the page is hand-typed: every
number is read from disk at run time, so re-running after more evals land needs no edit.

WHY THIS EXISTS.  The paper's numbers currently live in three places that no single
command joined up:

  1. scripts/final_recipe_report.py -- RI / HEST / THUNDER{knn,linear,few-shot} for the
     three ungated backbones, checkpoint chosen by the CI>=0.75 RULE, graded as
     pct_of_waiv.  Reads the OLD harness: /data/ryan.kim/hest_work/results/ (HEST),
     /data/ryan.kim/thunder/outputs/res/ (THUNDER), third_party/PathoROB (base RI).
  2. /data/ryan.kim/pathfm-full-evals/ -- a SECOND, newer corpus that no script in this
     repo read: THUNDER with segmentation + calibration + adversarial, PathoROB for all
     five backbones, and CPTAC.  Its own base-controls reproduce Waiv's published base
     far more closely than the old harness does (see section 2's base-gap column), so
     within-corpus deltas are the defensible ones.
  3. docs/waiv_published.json -- the published targets.  NEVER hardcoded here.

RULES CARRIED OVER FROM scoreboard2.py / final_recipe_report.py:
  * ONE ROW = ONE (run, step).  Never best-RI from one checkpoint and best-HEST from
    another.
  * MISSING is printed, never substituted or silently dropped.
  * Raw scores first (ours | our base | Waiv base | Waiv fine-tuned); pct_of_waiv
    ( = (ours - our base) / (Waiv ft - Waiv base) ) is an EXTRA column.
  * A pct is not printed when the two sides do not measure the same thing, or when
    Waiv's own gain -- the denominator -- is at the print precision of their table.
    Section 2 gates both cases explicitly rather than emitting a large ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import eval_common as _ec        # noqa: E402 -- intentional, after sys.path insert
import final_recipe_report as _frr  # noqa: E402

# ---------------------------------------------------------------------------
# The second corpus.  Env-overridable, same convention as collect_final5's paths.
# ---------------------------------------------------------------------------
PFE = Path("/data/ryan.kim/pathfm-full-evals")
PFE_THUNDER_CSV = PFE / "thunder" / "outputs" / "res" / "results.csv"
PFE_PATHOROB = PFE / "pathorob" / "results" / "robustness_index"
PFE_CPTAC = PFE / "cptac"

OUT_MD = _REPO / "docs" / "final_scoreboard.md"

# pathfm-full-evals spells the backbones differently from our arm names.
PFE_BACKBONE = {
    "phikon2": "phikon",
    "midnight": "midnight",
    "virchow2": "virchow2",
    "hoptimus0": "hoptimus",
    "uni2h": "uni2",
}
ARM_ORDER = ["phikon", "midnight", "virchow2", "hoptimus", "uni2"]
ARM_LABEL = {"phikon": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
             "hoptimus": "H-Optimus-0", "uni2": "UNI2-h"}

# THUNDER: the six tasks Waiv publish in Table 2.  Our side reads the harness's own
# benchmark_* roll-up rows, which are the same quantity Waiv tabulate.
#   our (dataset, task, metric, setting)                 -> published key
THUNDER_ROWS = [
    ("knn",           "benchmark_knn",               "knn",               "f1",  "",            "knn",           True),
    ("linear",        "benchmark_linear_probing",    "linear_probing",    "f1",  "",            "linear",        True),
    ("few_shot",      "benchmark_simple_shot",       "simple_shot",       "f1",  "16",          "few_shot",      True),
    ("segmentation",  "benchmark_segmentation",      "segmentation",      "f1",  "",            "segmentation",  True),
    ("calibration",   "benchmark_calibration",       "linear_probing",    "ECE", "",            "calibration",   True),
    ("adversarial",   "benchmark_adversarial_attack", "adversarial_attack", "f1", "drop",       "adversarial",   True),
]
LOWER_IS_BETTER = {"calibration", "adversarial"}

# The candidate recipe's arm name in the second corpus (genMASK-c3s...).
RECIPE_ARM_PREFIX = "c3s-"

# --- Adversarial sanity gate -----------------------------------------------
# The adversarial column is the f1 DROP under attack, and it is comparable to Waiv's
# for four of the five backbones (ours 19-32, Waiv 23-42).  On Virchow2 -- and ONLY on
# Virchow2 -- every model in the corpus reports a drop of 0.1-0.3pp against a published
# 31.1 for the very same base weights, i.e. the attack did not bite at all.  That is a
# broken attack, not a robust model, and quoting it as a 100x win would be the single
# most misleading number in the paper.  So the cell is measured, printed, and gated:
# any drop below this floor while Waiv report a large one is flagged, not scored.
ADVERSARIAL_DEAD_ATTACK_DROP = 5.0

# --- Denominator gate ------------------------------------------------------
# pct_of_waiv divides by Waiv's own gain.  Their Table 2 is printed to 0.1pp, so a gain
# read off two rounded numbers carries +/-0.1pp of pure print error; once |gain| falls to
# that scale the RATIO's relative error exceeds 100% and the percentage is an artefact of
# rounding, not a measurement (it is what turned a +1.0pp linear-probing move into
# "500% of Waiv" in the first draft of this table).  Not a tuned threshold: it is twice
# the print granularity of the source table.
WAIV_PRINT_GRANULARITY_PP = 0.1
DENOMINATOR_FLOOR_PP = 2 * WAIV_PRINT_GRANULARITY_PP
ADVERSARIAL_SUSPECT_NOTE = (
    "attack ineffective: our f1 drop is <%.0fpp where Waiv report a large drop for the "
    "same base weights, so the drop measures the attack, not the model.  Printed, not "
    "scored." % ADVERSARIAL_DEAD_ATTACK_DROP)


def _fmt(v, nd=4, pct=False):
    if v is None:
        return "MISSING"
    return f"{v:.1f}" if pct else f"{v:.{nd}f}"


def _pct_of_waiv(ours, our_base, waiv_base, waiv_ft):
    """(ours - our base) / (Waiv ft - Waiv base) * 100, uncapped.  None if undefined."""
    if None in (ours, our_base, waiv_base, waiv_ft):
        return None
    den = waiv_ft - waiv_base
    if den == 0:
        return None
    return (ours - our_base) / den * 100.0


# ===========================================================================
# Section 1 -- the graded criterion, straight from final_recipe_report
# ===========================================================================
def section1(rep: dict) -> list[str]:
    cr = rep.get("checkpoint_rule") or {}
    rf = rep.get("run_family") or {}
    L = ["## 1. Headline: the graded criterion (RI / HEST / THUNDER)", "",
         "Source: `scripts/final_recipe_report.py` (`build_report()`), re-run by this",
         "command; machine-readable copy in `docs/final_recipe_verdict.json`.", "",
         f"Run family: `{rf.get('glob')}` -- the FINALISED recipe (WAIV_BCLS=3.0, "
         "WAIV_BMEAN=-inf, ms500, warmup 200, lr 1e-4, rank 32, projdim 512, t900,",
         "CKPT_EVERY=50, pin `falseneg-gated`), five backbones, "
         f"{rf.get('n_runs_discovered')} runs discovered.", "",
         f"Checkpoint per run is chosen by the **{cr.get('rule')} rule**: "
         f"{cr.get('description')}",
         f"Metric: {cr.get('metric')}.  This SUPERSEDES the retired "
         "`confounder_insensitivity >= 0.75` rule (and its 250/125/125 picks), which",
         "graded an unbounded odds with a per-dataset chance level -- see `docs/CAVEATS.md`.",
         "",
         f"SE fed to the rule: **{cr.get('ri_se_supplied')}** -- "
         f"{rep.get('ri_se_source', 'operator-supplied')}.",
         f"Per-checkpoint bootstrap SE found on disk: {cr.get('se_measured_on_disk')}.",
         "The sensitivity of every pick to that one number is tabulated below, so the",
         "choice is checkable rather than asserted.", "",
         "ONE ROW = ONE (run, step): the same rule-selected checkpoint feeds RI, HEST and",
         "THUNDER for a given run.  Best-RI from one checkpoint and best-HEST from another",
         "is never combined.  Where two seeds of one backbone plateau at different steps",
         "the `step` cell lists both and the cell's floor is the larger of the two.", "",
         "`pct` = (our mean - our base) / (Waiv ft - Waiv base) x 100, UNCAPPED.", ""]
    L += _checkpoint_tables(rep)
    hdr = ("| backbone | benchmark | step | ours | our base | Waiv base | Waiv ft | "
           "our gain | Waiv gain | pct of Waiv | +/-95% | n | status |")
    L += [hdr, "|" + "---|" * 13]
    for arm in ARM_ORDER:
        cells = rep.get("cells", {}).get(arm)
        if not cells:
            why = rep.get("arms_not_reportable", {}).get(arm, {})
            miss = ", ".join(why.get("missing_inputs", ["no cells built"]))
            L.append(f"| {ARM_LABEL[arm]} | RI / HEST / THUNDER | - | MISSING | MISSING | "
                     f"MISSING | MISSING | MISSING | MISSING | MISSING | - | 0 | "
                     f"NOT REPORTABLE ({miss}) |")
            continue
        for bench in ("RI", "HEST", "THUNDER"):
            c = cells.get(bench) or {}
            ours = c.get("raw_mean")
            base = c.get("base")
            waiv = c.get("waiv")
            if bench == "THUNDER":
                # THUNDER is a pooled ratio over its tasks: raw absolutes are per-task,
                # so the honest absolute columns here are the pooled deltas.
                ours = base = waiv = waiv_base = None
            else:
                waiv_base = rep_waiv_base(arm, bench)
            # Always the graded quantities the report itself divided, never re-derived.
            gain_o = c.get("our_delta", c.get("our_avg_delta"))
            gain_w = c.get("waiv_gain", c.get("waiv_avg_gain"))
            L.append(
                f"| {ARM_LABEL[arm]} | {bench} | {c.get('selected_step', '-')} | "
                f"{_fmt(ours, 5)} | {_fmt(base, 5)} | {_fmt(waiv_base, 5)} | "
                f"{_fmt(waiv, 5)} | {_fmt(gain_o, 5)} | {_fmt(gain_w, 5)} | "
                f"{_fmt(c.get('pct'), pct=True)} | {_fmt(c.get('ci'), pct=True)} | "
                f"{c.get('n', 0)} | {c.get('status', 'MISSING')} |")
    L += ["", "Per-model verdict (THE criterion: pct >= 70 on each of RI/HEST/THUNDER "
          "and mean of the three > 80):", ""]
    L += ["| backbone | RI | HEST | THUNDER | average | verdict |", "|" + "---|" * 6]
    for arm in ARM_ORDER:
        pm = rep.get("per_model", {}).get(arm)
        if not pm:
            L.append(f"| {ARM_LABEL[arm]} | MISSING | MISSING | MISSING | MISSING | NOT REPORTABLE |")
            continue
        # per_model is keyed BY QUANTITY then benchmark: pm["pct"]["RI"], not pm["RI"].
        vals = " | ".join(
            f"{_fmt((pm.get('pct') or {}).get(b), pct=True)}"
            f" +/-{_fmt((pm.get('ci') or {}).get(b), pct=True)}"
            f" [{(pm.get('status') or {}).get(b, 'MISSING')}]"
            for b in ("RI", "HEST", "THUNDER"))
        L.append(f"| {ARM_LABEL[arm]} | {vals} | {_fmt(pm.get('average'), pct=True)} | "
                 f"{pm.get('verdict')} -- {pm.get('verdict_reason', '')} |")
    L += ["", f"**Overall (per-model) verdict: {rep.get('per_model_verdict', 'MISSING')}**", ""]
    return L


def _checkpoint_tables(rep: dict) -> list[str]:
    """The rule's own working: what it picked per run, and how that moves with the SE."""
    L = ["### 1a. Checkpoint the rule selected, per run", "",
         "| backbone | seed | run | selected step | RI curve (step:RI) | note |",
         "|" + "---|" * 6]
    for r in rep.get("runs", []):
        curve = " ".join(
            f"{t['step']}:{t['ri']:.4f}" if t.get("ri") is not None else f"{t['step']}:--"
            for t in r.get("ci_trace", []))
        step = r.get("selected_step")
        L.append("| %s | %s | `%s` | %s | %s | %s |"
                 % (ARM_LABEL.get(r["backbone"], r["backbone"]), r["seed"], r["run"],
                    step if step is not None else "NOT SELECTED", curve or "no curve",
                    (r.get("note") or "").replace("|", "/")))
    sweep = rep.get("ri_se_sweep")
    if sweep:
        bbs = sorted({v["backbone"] for v in sweep["runs"].values()})
        L += ["", "### 1b. Sensitivity of the picks to the SE (diagnostic, not a selection)",
              "", "| SE | " + " | ".join(ARM_LABEL.get(b, b) for b in bbs) + " |",
              "|" + "---|" * (1 + len(bbs))]
        for se in sweep["grid"]:
            k = "%g" % se
            row = [",".join(str(x) for x in (sweep["consensus"][k].get(b) or [])) or "-"
                   for b in bbs]
            L.append(f"| {k} | " + " | ".join(row) + " |")
        L.append("")
        L.append("A cell with two steps means the seeds of that backbone disagree at that "
                 "SE; `-` means the rule did not fire on any seed.")
    L.append("")
    return L


def rep_waiv_base(arm: str, bench: str):
    """Waiv's published base for this arm/benchmark (RI and HEST only)."""
    w = _ec.WAIV.get(arm) or {}
    return w.get("ri_base") if bench == "RI" else w.get("hest_base")


# ===========================================================================
# Section 2 -- THUNDER, all six published tasks, from pathfm-full-evals
# ===========================================================================
def load_pfe_thunder() -> dict:
    """{model: {task_key: score}} from the harness's own benchmark_* roll-up rows."""
    if not PFE_THUNDER_CSV.exists():
        return {}
    want = {(ds, task, metric, setting): key
            for key, ds, task, metric, setting, _pub, _cmp in THUNDER_ROWS}
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with PFE_THUNDER_CSV.open() as fh:
        for row in csv.DictReader(fh):
            k = want.get((row["dataset"], row["task"], row["metric"], row["setting"]))
            if k is None:
                continue
            try:
                out[row["model"]][k] = float(row["metric_score"])
            except (TypeError, ValueError):
                pass
    return dict(out)


def split_model(model: str) -> tuple[str | None, str]:
    """'virchow2-c3s-s0-step125_optimized' -> ('virchow2', 'c3s-s0-step125')."""
    name = model[: -len("_optimized")] if model.endswith("_optimized") else model
    for pfe_bb, arm in PFE_BACKBONE.items():
        if name.startswith(pfe_bb + "-"):
            return arm, name[len(pfe_bb) + 1:]
    return None, name


def section2(pfe: dict) -> list[str]:
    L = ["## 2. THUNDER, all six published tasks (second corpus)", "",
         f"Source: `{PFE_THUNDER_CSV}` -- the harness's own `benchmark_*` roll-up rows,",
         "which are the same quantity Waiv tabulate in their Table 2.  This corpus is",
         "the ONLY place segmentation, calibration (ECE) and adversarial exist on our",
         "side; the old harness computes none of them.  Its base-controls also sit much",
         "closer to Waiv's published base than the old harness does (`base gap` column),",
         "so the within-corpus base-vs-tuned delta is the defensible one.", ""]
    if not pfe:
        return L + [f"**MISSING** -- {PFE_THUNDER_CSV} not readable.", ""]

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for model, scores in pfe.items():
        arm, tuned_arm = split_model(model)
        if arm:
            by_arm[arm][tuned_arm] = scores

    for arm in ARM_ORDER:
        arms = by_arm.get(arm)
        L.append(f"### {ARM_LABEL[arm]}")
        L.append("")
        if not arms:
            L += ["**MISSING** -- no model dirs for this backbone in the corpus.", ""]
            continue
        base = arms.get("base-control")
        pub = _ec.load_waiv_published()  # (WAIV, WAIV_THUNDER) -- only the raw json here
        blob = json.loads(_ec.WAIV_PUBLISHED_JSON.read_text())
        idx = {(m["name"], m["variant"]): m for m in blob["models"]}
        base_row, ft_row = _ec.WAIV_ROWS[arm]
        wb, wf = idx[tuple(base_row)]["thunder"], idx[tuple(ft_row)]["thunder"]
        del pub

        L.append("| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | " +
                 " | ".join(sorted(a for a in arms if a != "base-control")) +
                 " | pct of Waiv (recipe arm, mean over seeds) |")
        L.append("|" + "---|" * (7 + len([a for a in arms if a != "base-control"])))
        tuned_names = sorted(a for a in arms if a != "base-control")
        suspect: set[str] = set()
        gated = False
        for key, _ds, _task, _metric, _setting, pubkey, comparable in THUNDER_ROWS:
            ob = (base or {}).get(key)
            wbv, wfv = wb.get(pubkey), wf.get(pubkey)
            gap = None if (ob is None or wbv is None) else ob - wbv
            tuned_vals = [arms[a].get(key) for a in tuned_names]
            dead_attack = (key == "adversarial" and ob is not None
                           and ob < ADVERSARIAL_DEAD_ATTACK_DROP
                           and wbv is not None and wbv >= ADVERSARIAL_DEAD_ATTACK_DROP)
            # rounded to the table's own precision: 68.0 - 68.2 is -0.2, not -0.20000000000000284
            waiv_gain = None if (wbv is None or wfv is None) else round(wfv - wbv, 1)
            if dead_attack:
                suspect.add(key)
                pct_s = "SUSPECT -- not scored"
            elif waiv_gain is not None and round(abs(waiv_gain), 6) <= DENOMINATOR_FLOOR_PP:
                gated = True
                pct_s = (f"INDETERMINATE (Waiv gain {waiv_gain:+.1f}pp is within "
                         f"{DENOMINATOR_FLOOR_PP:.1f}pp print error)")
            elif not comparable:
                pct_s = "NOT COMPARABLE"
            else:
                # RULE 1: never max-over-arms.  Taking the best of six checkpoints per
                # task would both mix checkpoints across rows of one table and inflate
                # by argmax.  The scored cell is THE RECIPE -- the c3s arm(s) -- averaged
                # over whatever seeds of it are present, and every other arm stays
                # visible as its own column.
                got = [arms[a].get(key) for a in tuned_names if a.startswith(RECIPE_ARM_PREFIX)]
                got = [v for v in got if v is not None]
                if not got or ob is None:
                    pct_s = "MISSING"
                else:
                    p = _pct_of_waiv(sum(got) / len(got), ob, wbv, wfv)
                    pct_s = f"{_fmt(p, pct=True)} (n={len(got)})"
            lo = " (lower is better)" if key in LOWER_IS_BETTER else ""
            L.append(f"| {key}{lo} | {_fmt(ob, 1)} | {_fmt(wbv, 1)} | {_fmt(gap, 1)} | "
                     f"{_fmt(wfv, 1)} | {_fmt(waiv_gain, 1)} | " +
                     " | ".join(_fmt(v, 1) for v in tuned_vals) + f" | {pct_s} |")
        L.append("")
        for k in sorted(suspect):
            L.append(f"* **{k} SUSPECT for {ARM_LABEL[arm]}** -- {ADVERSARIAL_SUSPECT_NOTE}")
            L.append("")
        if gated:
            L.append(f"* INDETERMINATE cells above: Waiv's own published gain for that "
                     f"task is at or below {DENOMINATOR_FLOOR_PP:.1f}pp, twice the "
                     f"{WAIV_PRINT_GRANULARITY_PP:.1f}pp granularity their table is "
                     f"printed to, so the ratio is rounding, not a measurement.  Read the "
                     f"raw columns for these tasks.")
            L.append("")
    return L


# ===========================================================================
# Section 3 -- PathoROB RI, all five backbones, from pathfm-full-evals
# ===========================================================================
def section3() -> list[str]:
    L = ["## 3. PathoROB robustness index, all five backbones (second corpus)", "",
         f"Source: `{PFE_PATHOROB}/<model>_clsmean/<dataset>/-1_0/results_summary.json`,",
         "key `robustness_index`, averaged over tcga / camelyon / tolkach_esca -- the same",
         "three datasets and the same key section 1 uses, but from the newer corpus, which",
         "is the only one carrying the two gated backbones.", ""]
    if not PFE_PATHOROB.exists():
        return L + [f"**MISSING** -- {PFE_PATHOROB} not readable.", ""]
    datasets = ("tcga", "camelyon", "tolkach_esca")
    by_arm: dict[str, dict[str, float | None]] = defaultdict(dict)
    for cell in sorted(PFE_PATHOROB.iterdir()):
        if not cell.is_dir() or not cell.name.endswith("_clsmean"):
            continue
        arm, tuned = split_model(cell.name[: -len("_clsmean")])
        if arm is None:
            continue
        vals = []
        for ds in datasets:
            f = cell / ds / "-1_0" / "results_summary.json"
            if f.exists():
                try:
                    vals.append(json.loads(f.read_text())["robustness_index"])
                except (KeyError, json.JSONDecodeError):
                    pass
        by_arm[arm][tuned] = (sum(vals) / len(vals)) if len(vals) == len(datasets) else None
    L += ["| backbone | our base-control | Waiv base | Waiv ft | " +
          "best tuned (arm) | pct of Waiv |", "|" + "---|" * 6]
    for arm in ARM_ORDER:
        cells = by_arm.get(arm, {})
        base = cells.get("base-control")
        wb = (_ec.WAIV.get(arm) or {}).get("ri_base")
        wf = (_ec.WAIV.get(arm) or {}).get("ri")
        tuned = {k: v for k, v in cells.items() if k != "base-control" and v is not None}
        if tuned:
            bk = max(tuned, key=lambda k: tuned[k])
            best_s = f"{tuned[bk]:.4f} ({bk})"
        else:
            bk, best_s = None, "MISSING"
        pct = _pct_of_waiv(tuned.get(bk) if bk else None, base, wb, wf)
        L.append(f"| {ARM_LABEL[arm]} | {_fmt(base)} | {_fmt(wb)} | {_fmt(wf)} | "
                 f"{best_s} | {_fmt(pct, pct=True)} |")
    L += ["", "Note: `phikon2-base-control_clsmean` is absent from this corpus, so the",
          "phikon base cell above is MISSING; section 1's phikon base RI comes from the",
          "repo-local `third_party/PathoROB` tree instead and is NOT interchangeable.", ""]
    return L


# ===========================================================================
# Section 4 -- CPTAC
# ===========================================================================
def _pub_cptac_key(ours: str) -> str | None:
    """'cptac_coad/KRAS_mutation' -> '[CPTAC COAD][KRAS][AUC]'."""
    cohort, _, task = ours.partition("/")
    if not cohort.startswith("cptac_"):
        return None
    coh = cohort[len("cptac_"):].upper()
    if task.endswith("_mutation"):
        return f"[CPTAC {coh}][{task[: -len('_mutation')]}][AUC]"
    if task == "MSI_H":
        return f"[CPTAC {coh}][MSI-H][AUC]"
    return None   # Immune_class is bAcc for Waiv but AUC for us; subtype has no row


def section4() -> list[str]:
    L = ["## 4. CPTAC / Patho-Bench", "",
         f"Source: `{PFE_CPTAC}/<model>/aggregate.json`, key `classification_macro_ovr_auc`.",
         "Waiv's side is `docs/waiv_published.json -> table4_pathobench`.  Only the",
         "mutation/MSI AUC tasks are metric-compatible: Waiv score `Immune class` as",
         "balanced accuracy while we score it as macro-OvR AUC, and their survival cells",
         "are a C-index we compute per-alpha, so both groups are excluded from the paired",
         "mean and listed as MISSING rather than silently averaged in.", ""]
    if not PFE_CPTAC.exists():
        return L + [f"**MISSING** -- {PFE_CPTAC} not readable.", ""]
    blob = json.loads(_ec.WAIV_PUBLISHED_JSON.read_text())
    t4 = blob["table4_pathobench"]
    cols = t4["column_order"]
    pub_by_task = {t["task"]: t["scores"] for t in t4["tasks"]}

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(PFE_CPTAC.iterdir()):
        f = d / "aggregate.json"
        if not d.is_dir() or not f.exists():
            by_arm.setdefault(split_model(d.name)[0] or "?", {})[split_model(d.name)[1]] = None
            continue
        arm, tuned = split_model(d.name)
        try:
            by_arm[arm][tuned] = json.loads(f.read_text()).get("classification_macro_ovr_auc") or {}
        except json.JSONDecodeError:
            by_arm[arm][tuned] = None

    L += ["| backbone | arm | n matched tasks | our mean AUC (matched) | Waiv base | "
          "Waiv ft | pct of Waiv |", "|" + "---|" * 7]
    for arm in ARM_ORDER:
        cells = by_arm.get(arm)
        if not cells:
            L.append(f"| {ARM_LABEL[arm]} | - | 0 | MISSING | MISSING | MISSING | MISSING |")
            continue
        base_row, ft_row = _ec.WAIV_ROWS[arm]
        try:
            bi = cols.index(f"{base_row[0]}|{base_row[1]}")
            fi = cols.index(f"{ft_row[0]}|{ft_row[1]}")
        except ValueError:
            bi = fi = None
        base_scores = cells.get("base-control")
        for tuned in sorted(cells):
            ours = cells[tuned]
            if not ours:
                L.append(f"| {ARM_LABEL[arm]} | {tuned} | 0 | MISSING (no results on disk) "
                         f"| MISSING | MISSING | MISSING |")
                continue
            pairs = [(k, v, _pub_cptac_key(k)) for k, v in sorted(ours.items())]
            matched = [(k, v, pk) for k, v, pk in pairs if pk in pub_by_task]
            if not matched or bi is None:
                L.append(f"| {ARM_LABEL[arm]} | {tuned} | {len(matched)} | MISSING | "
                         f"MISSING | MISSING | MISSING |")
                continue
            our_mean = sum(v for _k, v, _p in matched) / len(matched) * 100.0
            wbv = sum(pub_by_task[p][bi] for _k, _v, p in matched) / len(matched)
            wfv = sum(pub_by_task[p][fi] for _k, _v, p in matched) / len(matched)
            ob = None
            if base_scores and tuned != "base-control":
                bm = [base_scores[k] for k, _v, p in matched if k in base_scores]
                if len(bm) == len(matched):
                    ob = sum(bm) / len(bm) * 100.0
            pct = _pct_of_waiv(our_mean, ob, wbv, wfv) if tuned != "base-control" else None
            note = "" if (ob is not None or tuned == "base-control") else " (no base-control)"
            L.append(f"| {ARM_LABEL[arm]} | {tuned}{note} | {len(matched)} | "
                     f"{_fmt(our_mean, 2)} | {_fmt(wbv, 2)} | {_fmt(wfv, 2)} | "
                     f"{_fmt(pct, pct=True)} |")
    L += ["", "Coverage caveat: only H-Optimus-0 and UNI2-h have a CPTAC base-control on",
          "disk, so only those two backbones can express a gain-over-base at all.  The",
          "midnight / phikon-v2 / Virchow2 rows are absolutes with no base and therefore",
          "no pct.", ""]
    return L


# ===========================================================================
# Section 5 -- what is missing, and why
# ===========================================================================
def section5(rep: dict, pfe: dict) -> list[str]:
    L = ["## 5. MISSING inventory", "",
         "Everything the paper's table would want that is NOT on disk, stated once so no",
         "reader has to infer it from a blank cell.", "",
         "| item | status | why |", "|---|---|---|"]
    rows = [
        ("THUNDER segmentation (section 1)", "MISSING",
         "`collect_final5.PAPER_SEG` defaults to ocelot+pannuke and those two cells were "
         "not run for every arm; the 16-set roster has no SPIDER segmentation task at all. "
         "Section 2 carries segmentation from the second corpus instead."),
        ("THUNDER calibration / adversarial (section 1)", "NOT COMPUTED",
         "`eval_common.WAIV_THUNDER_TASKS` deliberately covers four tasks; the old harness "
         "never computed ECE or an attack.  Section 2 carries both."),
        ("THUNDER adversarial, Virchow2 only", "SUSPECT",
         ADVERSARIAL_SUSPECT_NOTE + "  All three Virchow2 models report a 0.1-0.3pp drop "
         "against a published 31.1; the other four backbones report 19-32 and are scored "
         "normally."),
        ("PathoROB base-control for phikon-v2 (second corpus)", "MISSING",
         "`phikon2-base-control_clsmean` is absent from all three metric dirs under "
         f"{PFE_PATHOROB.parent}."),
        ("CPTAC base-control for midnight / phikon-v2 / Virchow2", "MISSING",
         "Only hoptimus0 and uni2h have a `base-control` dir under the CPTAC tree."),
        ("CPTAC for 4 hoptimus arms", "EMPTY",
         "hoptimus0-bm3-s0-step100 and hoptimus0-c50-s0-step{50,100,150} have no "
         "`.complete`, no aggregate.json and zero task dirs."),
        ("CPTAC Immune class / survival", "NOT COMPARED",
         "metric mismatch: Waiv report balanced accuracy and C-index, we compute macro-OvR "
         "AUC and a per-alpha C-index."),
        ("Waiv Patho-Bench grand average (63 tasks)", "NOT COMPARABLE",
         "our CPTAC corpus covers 38 tasks, 26 of which map onto their table; their grand "
         "average also spans Hancock / PANDA / BC-Therapy cohorts we never ran."),
    ]
    for arm in ARM_ORDER:
        why = (rep.get("arms_not_reportable") or {}).get(arm)
        if why:
            rows.append((f"section 1 cells for {ARM_LABEL[arm]}", "NOT REPORTABLE",
                         "missing inputs: " + ", ".join(why.get("missing_inputs", []))))
    # Per-(arm, benchmark) grading inputs that are absent.  An arm can be perfectly
    # gradeable on RI and HEST and blocked on THUNDER alone; that used to delete the arm
    # from every table, and it is now printed at the granularity it actually applies to.
    for cell, missing in sorted((rep.get("cells_not_reportable") or {}).items()):
        rows.append((f"section 1 cell {cell}", "NOT REPORTABLE",
                     "missing grading input(s): " + ", ".join(missing)))
    # Whatever section 1's own THUNDER cells are complaining about, quoted verbatim from
    # the report rather than restated here (restating it is how the two drift apart).
    seen: set[str] = set()
    for arm in ARM_ORDER:
        tk = ((rep.get("cells", {}).get(arm) or {}).get("THUNDER") or {}).get("tasks") or {}
        for task, ent in sorted(tk.items()):
            reason = ent.get("reason")
            if not reason or ent.get("status") not in ("PARTIAL", "NO_DATA"):
                continue
            key = f"{ARM_LABEL[arm]} THUNDER"
            if key in seen:
                continue
            seen.add(key)
            rows.append((f"section 1 THUNDER for {ARM_LABEL[arm]}", ent["status"], reason))
    # ... and where those same checkpoints DO have a THUNDER eval: the second corpus.
    # This is the difference between "never evaluated" and "evaluated in the corpus
    # section 1 is not allowed to read", and only one of them is a to-do.
    inv = {v: k for k, v in PFE_BACKBONE.items()}
    for arm in ARM_ORDER:
        steps = (((rep.get("cells", {}).get(arm) or {}).get("RI") or {})
                 .get("selected_steps") or [])
        if not steps:
            continue
        want = {f"{inv[arm]}-c50-s{s}-step{st}"
                for st in steps for s in range(6)}
        have = sorted(m[: -len("_optimized")] if m.endswith("_optimized") else m
                      for m in pfe if (m.split("_optimized")[0]) in want)
        if have:
            rows.append((f"THUNDER at the section-1 checkpoint for {ARM_LABEL[arm]}",
                         "IN THE SECOND CORPUS ONLY",
                         "the rule-selected checkpoint(s) " + "/".join(str(s) for s in steps)
                         + " have THUNDER results as " + ", ".join(have) + " under "
                         + f"`{PFE_THUNDER_CSV}` (section 2), not in the old harness "
                         "section 1 grades against.  They are NOT merged into section 1: "
                         "the 12-dataset seed floors for phikon-v2 / Midnight-12k / "
                         "Virchow2 were measured in the OLD corpus and its "
                         "Resize(224,bilinear) transform, so a numerator from one corpus "
                         "over a floor from the other is not a matched comparison."))
    for item, status, why in rows:
        L.append(f"| {item} | {status} | {why} |")
    L.append("")
    if not pfe:
        L.append(f"**Second corpus unreadable at {PFE}: sections 2-4 are all MISSING.**")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_MD), help=f"markdown path (default {OUT_MD})")
    ap.add_argument("--json-out", default=str(_REPO / "docs" / "final_recipe_verdict.json"),
                    help="also refresh the machine-readable section-1 verdict")
    ap.add_argument("--cls-roster", choices=sorted(_frr.CLS_ROSTERS),
                    default=_frr.CLS_ROSTER_DEFAULT,
                    help="THUNDER classification roster for section 1")
    ap.add_argument("--rule", choices=_frr.RULES, default="1se",
                    help="checkpoint-selection rule for section 1 (default 1se, the "
                         "parameter-free one-standard-error rule; ci075 is the RETIRED "
                         "confounder_insensitivity>=0.75 rule, kept for before/after work)")
    ap.add_argument("--ri-se", type=float, default=_frr.RI_SE_SCOREBOARD_DEFAULT,
                    metavar="SE",
                    help="SE the 1-SE rule is run with.  Default %(default)s -- "
                         + _frr.RI_SE_SCOREBOARD_SOURCE)
    ap.add_argument("--run-glob", default=None, metavar="GLOB",
                    help=f"run family under runs/ (default {_frr.RUN_GLOB})")
    args = ap.parse_args()

    print("building section 1 (final_recipe_report.build_report) ...", flush=True)
    rep = _frr.build_report(cls_roster=args.cls_roster, rule=args.rule,
                            ri_se=(args.ri_se if args.rule == "1se" else None),
                            run_glob=args.run_glob)
    # Provenance of the one number the rule takes, carried into both outputs.
    rep["ri_se_source"] = (_frr.RI_SE_SCOREBOARD_SOURCE
                           if args.ri_se == _frr.RI_SE_SCOREBOARD_DEFAULT
                           else "operator-supplied on the command line (--ri-se)")
    Path(args.json_out).write_text(json.dumps(rep, indent=2, default=str))

    print("reading the pathfm-full-evals corpus ...", flush=True)
    pfe = load_pfe_thunder()

    body = [
        "# Waiv final scoreboard",
        "",
        "**Generated file -- do not hand-edit.**  Regenerate with:",
        "",
        "```",
        "./.venv/bin/python scripts/final_scoreboard.py",
        "```",
        "",
        f"Waiv targets: `{_ec.WAIV_SOURCE}`.",
        "Every number below is read from disk at generation time.  `MISSING` means the",
        "metric is not on disk for that cell; it is never substituted from another",
        "checkpoint, another step, or another arm.",
        "",
    ]
    body += section1(rep)
    body += section2(pfe)
    body += section3()
    body += section4()
    body += section5(rep, pfe)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n")
    print(f"wrote {out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
