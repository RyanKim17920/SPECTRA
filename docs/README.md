# Docs index — status as of 2026-08-31

24 documents accumulated in this directory across the project. Most are historical
records of investigations that led to the current state; only a few are current.
This file is the map. It does not replace any of them — nothing below was rewritten
or deleted, several stale docs got a short prepended status banner instead.

## If you are writing the paper, read only these

1. **[`RUNBOOK.md`](RUNBOOK.md)** — current recipe, current commands, current config
   (5 backbones, 50-step checkpoint grid, parameter-free 1-SE stopping rule,
   `WAIV_BCLS=3.0` / `WAIV_BMEAN=-inf`).
2. **[`THUNDER_16DS_2026-08-26.md`](THUNDER_16DS_2026-08-26.md)** — current THUNDER
   protocol/numbers (16-dataset roster, correct transforms).
3. **[`thunder_seed_floor_12ds.md`](thunder_seed_floor_12ds.md)** — current per-backbone
   THUNDER seed floors.
4. Nothing else. Every other doc below is either historical context or explicitly
   invalid for citation — check the per-doc note before pulling any number from it.

## Full classification

| Doc | Status | Note |
|---|---|---|
| RUNBOOK.md | **CURRENT** | How-to-run, single source of truth (written 2026-08-31 from scripts). |
| THUNDER_16DS_2026-08-26.md | **CURRENT** | Correct 16-dataset THUNDER protocol/numbers. |
| thunder_seed_floor_12ds.md | **CURRENT** | Per-(backbone,task) THUNDER seed floors in use today. |
| FINAL_RECIPE.md | HISTORICAL | Already carries its own SUPERSEDED banner → RUNBOOK.md. |
| CAVEATS.md | HISTORICAL | Discipline still applies; example numbers predate 5-backbone/50-grid/1-SE. |
| EVAL_FIXES_2026-08-26.md | HISTORICAL | Fixes already merged; audit record only. |
| FINAL5_RESULTS.md | HISTORICAL | Only 3 of 5 backbones; not the final table. |
| FINAL_CANDIDATE.md | HISTORICAL | Pre-1-SE (0.75 threshold), 3-backbone, 250/500-grid era. |
| FINAL_RESULTS.md | **INVALID sections** | Pre-16-dataset THUNDER "4 tasks" rows — do not cite. |
| FINDINGS_2026-08-16.md | HISTORICAL | Earliest-era findings, 3-backbone, grid to step 1500. |
| FORMULA_UNIFICATION_2026-08-26.md | HISTORICAL | Audit journal; fixes now merged. |
| NEGATIVE_MASKING.md | HISTORICAL | n=1, phikon-v2-only ablation; fed final bias choice. |
| NEW_MODEL.md | HISTORICAL | 2026-08-07 architecture note, 2-backbone era. |
| REPRODUCING.md | HISTORICAL | Superseded by RUNBOOK.md for run instructions. |
| RESULTS.md (140KB) | **HISTORICAL, INVALID sections** | Two non-interchangeable studies; pre-16-dataset THUNDER "4 tasks" rows are INVALID — flagged inline, see CAVEATS.md. |
| STATE_2026-08-25.md | HISTORICAL | Self-claimed canonical 08-25; superseded by RUNBOOK.md and 5-backbone state. |
| WAIV_COMPARISON.md | HISTORICAL | Comparison snapshot, predates 5-backbone/16-dataset THUNDER. |
| aggregate_criterion_resolvability.md | HISTORICAL | Intermediate stopping-criterion analysis, 0.75-threshold framing. |
| baseline_comparability_audit.md | HISTORICAL | Finding "THUNDER pct_of_waiv invalid" still holds; table itself is dated. |
| generation_comparison.md | HISTORICAL | Three-generation retrospective, 3-backbone scope. |
| internal_stopping_criterion.md | HISTORICAL | Refutes L2-delta signal; background for the 1-SE rule. |
| round_temp_dose.md | HISTORICAL | Pre-registered round; own text flags its RI floors SUPERSEDED. |
| thunder_base_offset_investigation.md | HISTORICAL | Own conclusion table marks itself superseded. |
| thunder_seed_floor.md | **SUPERSEDED / INVALID** | 5-dataset floor; superseded by thunder_seed_floor_12ds.md. |

**Numbers that could be pasted into the paper by mistake — do not use:**
- `RESULTS.md` — "THUNDER mean Δ over 4 tasks" rows (pre-16-dataset-port).
- `FINAL_RESULTS.md` — same pre-16-dataset-port THUNDER figures.
- `thunder_seed_floor.md` — 5-dataset seed floor (use `thunder_seed_floor_12ds.md`).
- Any doc using the 0.75 confounder_insensitivity threshold (RUNBOOK.md's 1-SE rule is current).
- Any doc scoped to "three backbones" (current roster is five: phikon-v2, midnight, Virchow2, H-Optimus-0, UNI2-h).
