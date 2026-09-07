# Docs map

`docs/` holds three kinds of file, and it matters which one you are reading:

- **Current docs** — the recipe, the protocol, the reporting discipline. Maintained by hand.
- **Generated tables** — written by a script, every one carrying a *"do not hand-edit"*
  banner that names its generator. Regenerate rather than edit; `README.md` §Reproducing the
  tables lists the commands.
- **[`archive/`](archive/)** — superseded research logs, kept for provenance. See the note at
  the bottom before quoting anything from there.

---

## Current

| Doc | Contents |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | **Start here.** Current recipe, commands and config — 5 supported arms, 50-step checkpoint grid, parameter-free 1-SE stopping rule, `SPECTRA_BCLS=3.0` / `SPECTRA_BMEAN=-inf`. Single source of truth for how to run this. |
| [`THUNDER_16DS_2026-08-26.md`](THUNDER_16DS_2026-08-26.md) | Current THUNDER protocol: 16-dataset roster, correct transforms. |
| [`thunder_seed_floor_12ds.md`](thunder_seed_floor_12ds.md) | Current per-(backbone, task) THUNDER seed floors — what a delta has to clear to be gradeable. |
| [`CAVEATS.md`](CAVEATS.md) | Reporting discipline the project holds itself to. The rules apply; its own example numbers are historical. |
| [`release_manifest.md`](release_manifest.md) | What each released model repo contains, which checkpoint the 1-SE rule selected, per-backbone licence state. |
| [`reference_published.json`](reference_published.json) | The published comparison targets, transcribed once. Carries its own `_source` citation; every comparison script loads from here rather than restating the numbers. |

## Generated tables

| Doc | Generator |
|---|---|
| [`final_scoreboard.md`](final_scoreboard.md) | `scripts/final_scoreboard.py` — the graded RI / HEST / THUNDER criterion, per run and per cell |
| [`seed_stats.md`](seed_stats.md) | `scripts/seed_stats.py` — per-seed 1-SE-selected checkpoints |
| [`eval_matrix.md`](eval_matrix.md) | `scripts/eval_matrix.py` — what is and is not measured |
| [`pathorob_ranks.md`](pathorob_ranks.md) | `scripts/pathorob_ranks.py` |
| [`pathorob_submetrics.md`](pathorob_submetrics.md) (+ `.tex`) | `scripts/pathorob_submetrics.py` |
| [`hest_ranks.md`](hest_ranks.md), [`hest_per_task.md`](hest_per_task.md) | `scripts/hest_ranks.py` |
| [`thunder_ranks.md`](thunder_ranks.md) | `scripts/thunder_ranks.py` |
| [`thunder_per_ds.md`](thunder_per_ds.md) | `scripts/thunder_per_ds.py` |
| [`thunder16.md`](thunder16.md), [`thunder16_floor.md`](thunder16_floor.md) | `scripts/thunder16.py`, `scripts/thunder16_floor.py` |
| [`thunder_seed_floor.md`](thunder_seed_floor.md) | `scripts/thunder_seed_floor.py` — 5-dataset floor, **superseded** by `thunder_seed_floor_12ds.md` |
| [`cptac.md`](cptac.md) | `scripts/cptac_table.py` |
| [`plism_retrieval.md`](plism_retrieval.md) | `scripts/plism_retrieval.py` |
| [`overall_ranks.md`](overall_ranks.md) | `scripts/overall_ranks.py` |
| [`aggregate_criterion_resolvability.md`](aggregate_criterion_resolvability.md) | `scripts/aggregate_criterion_resolvability.py` |
| [`figure1.html`](figure1.html), [`figure1_data.json`](figure1_data.json) | `scripts/figure1.py` |

The `*.json` files alongside them (`final5_results.json`, `final_recipe_verdict*.json`,
`hest_seed_sd.json`, `stopping_*.json`, `thunder_seed_floor*.json`) are machine-readable
intermediates that the same generators write and read. Do not hand-edit those either.

`projdiag/` holds a standalone projection diagnostic and its inputs.

## Archive

[`archive/`](archive/) is the project's research log: earlier findings, superseded recipes,
audit journals and one-off investigations. They are kept because they record *why* the
current state is what it is, and several current decisions are only justified there.

Three warnings before quoting from `archive/`:

1. **Numbers are stale by construction.** Most predate the 7-backbone roster, the 50-step
   checkpoint grid, the 1-SE stopping rule, or the 16-dataset THUNDER roster.
   `FINAL_RESULTS.md` and `RESULTS.md` in particular contain pre-16-dataset THUNDER "4 tasks"
   rows that are **invalid**, not merely old.
2. **Naming is superseded.** These files were written under an earlier internal project name
   and against an earlier module layout, and were archived unedited rather than rewritten.
3. Several already carry their own prepended status banner. Where a banner and this map
   disagree, the map is newer.

| Doc | Note |
|---|---|
| `FINAL_RECIPE.md` | Carries its own SUPERSEDED banner → `RUNBOOK.md`. |
| `EVAL_FIXES_2026-08-26.md` | Fixes already merged; audit record only. |
| `FINAL5_RESULTS.md` | Only 3 of 5 backbones; not the final table. |
| `FINAL_CANDIDATE.md` | Pre-1-SE (0.75 threshold), 3-backbone, 250/500-grid era. |
| `FINAL_RESULTS.md` | **INVALID sections** — pre-16-dataset THUNDER "4 tasks" rows. |
| `FINDINGS_2026-08-16.md` | Earliest-era findings, 3-backbone, grid to step 1500. |
| `FORMULA_UNIFICATION_2026-08-26.md` | Audit journal; fixes now merged. |
| `NEGATIVE_MASKING.md` | n=1, phikon-v2-only ablation; fed the final bias choice. |
| `NEW_MODEL.md` | 2026-08-07 architecture note, 2-backbone era. Its §2 "reproduce the base first" procedure is still the one the README points at. |
| `REPRODUCING.md` | Superseded by `RUNBOOK.md`. |
| `RESULTS.md` (140 KB) | **HISTORICAL, INVALID sections.** Two non-interchangeable studies; the "4 tasks" THUNDER rows are invalid — flagged inline, see `CAVEATS.md`. |
| `STATE_2026-08-25.md` | Self-claimed canonical 08-25; superseded by `RUNBOOK.md`. |
| `reference_comparison.md` | Comparison snapshot against the published baseline; predates the 7-backbone / 16-dataset THUNDER state. |
| `baseline_comparability_audit.md` | The "THUNDER base offset" finding; dated. |
| `generation_comparison.md` | Three-generation comparison, 3-backbone scope. |
| `internal_stopping_criterion.md` | Refutes the L2-delta rule; superseded by the 1-SE rule. |
| `round_temp_dose.md` | Pre-registered RI round; superseded. |
| `thunder_base_offset_investigation.md` | Superseded by its own follow-up. |
