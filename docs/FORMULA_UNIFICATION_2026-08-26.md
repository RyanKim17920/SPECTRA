# Formula unification audit — 2026-08-26

Mandate: nothing hardcoded, everything empirical, **exactly the same formula applied in
every case**. Where a fix makes a number unavailable or different, the correct outcome is
the unavailable/different number — never preserve a number by keeping a bad estimator.

Commits: `52af1e0` (F-A, F-B, F-C, F-D, F-F) and the follow-up commit carrying F-E, F-G,
F-H, F-I and this document.

---

## 1. Before / after — every published number

Baseline = `python3 scripts/final_recipe_report.py` at `f45901d`.
After = same command at HEAD.

### The nine cells

| backbone | benchmark | BEFORE | AFTER | change |
|---|---|---|---|---|
| phikon | RI | `100.0*` (uncapped 108.9) ±1.9 **PASS** | `108.9` (capped 100.0) ±1.9 **PASS** | graded value is now the uncapped one (F-C) |
| phikon | HEST | `90.5` ±8.2 **PASS** | `90.5` ±**13.3** **PASS** | CI widened 62%: derived seed SD 7.60 pct pts vs hand-set 5.8 (F-A) |
| phikon | THUNDER | `79.7` (uncapped 99.6) ±22.8 **PASS** | **INDETERMINATE** | all 3 tasks fail the shared denominator gate (F-B) |
| midnight | RI | `92.5` ±4.1 **PASS** | `92.5` ±4.1 **PASS** | unchanged |
| midnight | HEST | `92.1` ±11.7 **PASS** | `92.1` ±**0.1** **PASS** | derived floor at step 125 has **df=1** — see §6, disclosed not patched |
| midnight | THUNDER | `100.0*` (uncapped 131.6) ±20.3 **PASS** | **PARTIAL** | knn+lp now denominator-gated; only simple_shot survives and it is PARTIAL coverage (9,11/12) |
| virchow2 | RI | `81.6` ±11.2 **PASS** | `81.6` ±11.2 **PASS** | unchanged |
| virchow2 | HEST | `72.9` ±20.1 **NOT RESOLVED** | **INDETERMINATE (withheld)** | denominator gate: one seed-SD = 17.6 pct pts > 10 (F-B) |
| virchow2 | THUNDER | `100.0*` (uncapped 126.7) ±10.4 **PASS** | `126.7` (capped 100.0) ±10.4 **PASS** | graded value now uncapped (F-C) |

### Benchmark averages, overall average, verdict

| quantity | BEFORE | AFTER |
|---|---|---|
| RI average | 91.4 (3/3 backbones) | **94.3** (3/3) |
| HEST average | 85.2 (3/3) | **91.3 (2/3 — ineligible)** |
| THUNDER average | 93.2 (3/3) | **126.7 (1/3 — ineligible)** |
| OVERALL average | **89.9** | **UNDEFINED** |
| worst cell | virchow2/HEST 72.9 [NOT RESOLVED] | virchow2/RI 81.6 (lower 70.4) [PASS] |
| FINAL VERDICT | INDETERMINATE (error bar straddles 70 on virchow2/HEST) | INDETERMINATE (no gradeable number for phikon/THUNDER, midnight/THUNDER, virchow2/HEST) |

The benchmark averages moved for two independent reasons: (a) F-C — averages now sum the
uncapped value, so RI rose 91.4 → 94.3 purely because phikon's 108.9 stopped being
censored to 100.0; (b) F-B — gated cells left the averages entirely, so HEST and THUNDER
now cover 2/3 and 1/3 backbones and are **ineligible** for the overall average.

### THUNDER task detail (the 9 task cells)

| backbone/task | BEFORE | AFTER | one seed-SD, pct pts |
|---|---|---|---|
| phikon/knn | 97.7* (116.6) ±24.4 PASS | INDETERMINATE | 17.3 |
| phikon/linear_probing | 100.0* (140.9) ±19.1 PASS | INDETERMINATE | 13.5 |
| phikon/simple_shot | 41.2 ±24.9 FAIL | INDETERMINATE | 17.6 |
| midnight/knn | 100.0* (131.6) ±20.3 PASS | INDETERMINATE | 14.4 |
| midnight/linear_probing | INDETERMINATE | INDETERMINATE | 106.3 |
| midnight/simple_shot | PARTIAL | PARTIAL | 6.9 (would be gradeable at 12/12) |
| virchow2/knn | INDETERMINATE | INDETERMINATE | 70.3 |
| virchow2/linear_probing | INDETERMINATE | INDETERMINATE | 79.4 |
| virchow2/simple_shot | 100.0* (126.7) ±10.4 PASS | 126.7 (100.0*) ±10.4 PASS | 7.3 |

### Retired literals vs the values now read from disk

| literal | retired value | from disk | delta |
|---|---|---|---|
| RI_BASE phikon | 0.4686 | 0.4686113 | +1.13e-5 |
| RI_BASE midnight | 0.7589 | 0.7588607 | −3.93e-5 |
| RI_BASE virchow2 | 0.8582 | 0.8582400 | +4.00e-5 |
| RI_WAIV (×3) | 0.806 / 0.924 / 0.918 | identical | 0 |
| HEST_BASE phikon | 0.37470 | 0.37470093 | +9.3e-7 |
| HEST_BASE midnight | 0.39521 | 0.39521019 | +1.9e-7 |
| HEST_BASE virchow2 | 0.40324 | 0.40326852 | **+2.85e-5** |

All agree to 4 dp; the literals were correctly-rounded. virchow2's HEST base is the one
that mattered — three scripts used the rounded `results.avg` 0.40324 as a base while the
numerator came from `custom_encoder`.

---

## 2. Fix status

| fix | status | mechanism |
|---|---|---|
| **F-A** derive HEST seed SD from disk | **DONE** | new `scripts/hest_seed_sd.py` → `docs/hest_seed_sd.json`; `final_recipe_report` loads it via `eval_common.load_hest_seed_sd()`. `HEST_SD_PCT` deleted. |
| **F-B** one denominator-resolvability gate | **DONE** | `eval_common.denominator_unresolvable()` is the only implementation; `final_recipe_report.gate_denominator()` calls it for RI, HEST and every THUNDER task; `scoreboard.py` imports the same function and re-exports `UNRESOLVABLE_SD_PCT_LIMIT` from it. |
| **F-C** cap must not enter arithmetic | **DONE** | `grade()` sets `pct = pct_uncapped`. The ≥70 test, the benchmark means, the overall average and the worst-cell search all read `pct`. `pct_capped` is printed in parentheses and enters nothing. |
| **F-D** one CI construction | **DONE** | `eval_common.ci95()` = `max(empirical 2·SD/√n, floor 2·SD/√n)`; used by RI, HEST **and** THUNDER (both HEST and THUNDER were floor-only before). `eval_common.seed_sd_at_step()` is the single step-selection rule (exact step, else conservative max), shared by RI and HEST. |
| **F-E** one HEST loader, one field | **DONE** | `HEST_BASE_FALLBACK` deleted — `collect_final5._load_hest_base()` now **raises**. `scoreboard2.py`, `aggregate_criterion_resolvability.py`, `build_stopping_dataset.py` import `collect_final5.HEST_BASE` / `RI_BASE` / `eval_common.HEST_WAIV`; the last two also switched their numerator from `results['avg']` to `custom_encoder`. `collect_hest.py`'s `avg` row now reads `custom_encoder`. |
| **F-F** RI_BASE false provenance | **DONE** | `eval_common.load_ri_base()` reads `third_party/PathoROB/results/robustness_index/<model>/<ds>/-1_0/results_summary.json`. `RI_WAIV` comes from `pathorob_adapter.TARGETS`. Retired literals are asserted against disk and the comparison is printed in the report. |
| **F-G** `config_ok` gate admitted unreadable configs | **DONE** | `config_ok is not False` → `config_ok is True`; unverifiable runs are excluded and printed in a loud banner. |
| **F-H** `first_where` sentinel | **DONE** | returns `NEVER` (None); `report()` counts non-firing runs, marks such rules `INCOMPLETE`, and a new eligibility ranking excludes them. Stopping-rule sweep re-run — see §4. |
| **F-I** scoreboard2 tuned constants | **DONE** | `_THUNDER_MEAN_1SD = 0.0025` replaced by measured per-(backbone,task) `seed_sd_of_task_mean` read from `docs/thunder_seed_floor_12ds.json` (range 0.00189–0.00639, and the offset-2SE floors run to 0.0233). `RI_BUDGET_FLOOR` quarantined behind `WAIV_ENABLE_RI_BUDGET_COLUMN=1`, column prints `off` by default with a provenance warning. |

---

## 3. The derived HEST seed SD

Derivation: `scripts/hest_seed_sd.py` → artifact `docs/hest_seed_sd.json`
(mirrors how `docs/thunder_seed_floor_12ds.json` is produced and consumed).

Estimator: **yes, it is `scoreboard.NOISE_SD`'s documented estimator** — pooled
within-recipe across-seed SD, `sqrt(Σ_f df_f · sd_f² / Σ_f df_f)`, `df_f = n_f − 1`,
per `(backbone, step)`, same HEST pooling protocol, scalar read through
`collect_final5._hest_score` (`hest_perf_per_encoder.custom_encoder`).

One deviation, deliberate and documented in the script: **"recipe family" is keyed on the
full config minus bookkeeping, not on `CHECKED_CONFIG_KEYS`.** Nine recipe-defining keys
vary across `runs/` but are absent from that list (`retention_kl_weight`,
`mask_sim_thresh`, `min_tissue_frac`, `use_lora`, `grid`, `group_size`, `n_groups`,
`resume_from`), and two more — `split_heads`, `pool_head` — are looked for under
`encoder.` while the writer emits them at top level, so those two checks pass vacuously.
Keying on that list merged e.g. `ret0.01` with `kl0` and reported their *between*-recipe
spread as seed noise.

Raw SD (metric units) at the rule-selected step, and in pct-of-waiv points:

| backbone | selected step | derived raw SD | df | **derived pct pts** | **old literal** | scoreboard `NOISE_SD`/gain | empirical 5-seed |
|---|---|---|---|---|---|---|---|
| phikon | 250 | 0.00149 | 3 | **7.60** | 5.8 | 8.52 | 8.52 |
| midnight | 125 | 0.0000157 | **1** | **0.073** | 8.3 | 8.61 | 8.88 |
| virchow2 | 125 | 0.00180 | 4 | **17.57** | 14.2 | 15.05 | 11.27 |

phikon's literal was 32% narrower than any measurement; it is now 7.60, which widened the
phikon/HEST CI from ±8.2 to ±13.3. virchow2's is now 17.57, above the 10-point
resolvability limit, which is what withholds the cell.

`docs/hest_seed_sd.json` also carries the SD at every other step; the report picks the
step by `eval_common.seed_sd_at_step`.

---

## 4. F-H: did the stopping-rule answer change?

Yes, materially, though `CI ≥ 0.75` is not overturned as a *threshold*.

`first_where` returned `v[-1]` when the predicate never fired, so every threshold rule
silently degraded into "stop at the LAST checkpoint" on the runs where its threshold was
never crossed, and inherited R0b's numbers there. `scripts/full_grid_rules.py:42` does the
same search correctly (yields `None`), which is how the disagreement surfaced.

With the sentinel fixed:

* **`CI ≥ 0.75` never fires on 4 of the 35 runs** (all phikon). It is therefore not a
  complete stopping rule and is now marked `INCOMPLETE / DISQUALIFIED` in the sweep.
  (`final_recipe_report` already behaves consistently with this — it drops the 3 virchow2
  runs whose curve never reaches 0.75.)
* Its reported per-backbone numbers changed: phikon mean HEST pct 73.4 → **80.7**,
  RI 109.8 → 108.6, n 15 → **11**. The earlier figures were partly the bug.
* Its worst-backbone score is unchanged at 54.6 (virchow2, equal to the oracle), and
  among CI thresholds 0.75 still ties for best — so the **value 0.75 survives**, but the
  claim that it is a *rule applicable to every run* does not.
* No CI threshold is complete. Among rules that always fire, the best worst-backbone
  scores are `first L2 ≥ 0.6` (54.6) and `stop at earliest` (54.5).

---

## 5. Is the criterion now computable by one identical formula across all 9 cells?

**The formula is now identical across all 9 cells. The criterion is NOT computable,
because 3 of the 9 cells have no gradeable number.**

Identical across all nine: the pct definition (uncapped), the denominator gate
(`eval_common.denominator_unresolvable`, one seed-SD > 10 pct pts), the CI
(`eval_common.ci95`, `max(empirical, floor)`), the step rule
(`eval_common.seed_sd_at_step`), the ≥70 resolution test, and the minimum-n gate. No
benchmark has a private construction any more.

What blocks the criterion:

1. **phikon/THUNDER — all 3 tasks denominator-gated.** Waiv's published gains
   (+0.037 knn, +0.014 lp, +0.015 ss) are 13–18% of a seed-SD each.
2. **midnight/THUNDER — knn and lp gated, simple_shot PARTIAL** (9 and 11 of 12 datasets).
3. **virchow2/HEST — denominator gated** (17.6 pct pts per seed-SD).

Consequences: HEST covers 2/3 backbones and THUNDER 1/3, so both benchmark means are
ineligible and the >80 overall average is **UNDEFINED**. The ≥70 per-cell half is
satisfied on all 6 gradeable cells (worst = virchow2/RI at 81.6, lower bound 70.4).

Item 2 is a data-completeness problem and will resolve when the remaining midnight/THUNDER
datasets land. Items 1 and 3 are properties of the comparison, not of our sampling: more
of our seeds cannot sharpen a denominator that was never resolved.

---

## 6. Deliberately NOT fixed / known-inconsistent

1. **POOLED grading is NOT implemented — the biggest remaining gap.** The user's grading
   rule is *aggregate the numerator and the denominator, then divide once*. This code
   still averages per-cell ratios: THUNDER's backbone cell is the mean of three task
   `pct` values, and each benchmark average is the mean of three backbone `pct` values.
   That is a different estimator, and it is what forces per-cell denominator gating in
   the first place — pooling virchow2/THUNDER's three tasks gives a combined denominator
   of +0.0090, which may well be resolvable where the individual +0.037/−0.003/+0.003 are
   not. **This was not on the F-list and was not changed.** It should be the next fix, and
   until it lands the THUNDER verdicts above are per-cell verdicts, not pooled ones.
2. **midnight/HEST's CI of ±0.1 is not credible.** The floor at step 125 rests on df=1,
   and the only recipe family supporting it *is* the family being graded — so
   `max(empirical, floor)` degenerates to the empirical spread of two runs that happened
   to land 1.6e-5 apart. The report prints a `FLOOR QUALITY WARNING` for this and the
   verdict JSON carries `weak_floors`. It is disclosed, not patched: patching would need
   either a new tuned constant (a minimum-df) or a change of estimator (Student-t
   multipliers), both out of scope.
3. **`CHECKED_CONFIG_KEYS` is still defective** (§3). Fixing it changes which runs
   `collect_final5` excludes from published aggregates, i.e. it moves measured numbers,
   which this audit is forbidden to do. Reported, not changed.
4. **RI's seed floor is still a literal** (`scoreboard.NOISE_SD[...]["ri"]`). F-A mandated
   deriving the HEST floor only. The RI floors have no producing script either and should
   get the same treatment.
5. **`scoreboard.py` crashes** with `KeyError: None` at
   `_thunder_protocol.default_pooling(ARM_BACKBONE[arm])` (line 424) on runs with an
   unknown backbone. This is in the concurrently-edited backbone-registration path, not in
   anything this audit touched (the only scoreboard edits here are the
   `UNRESOLVABLE_SD_PCT_LIMIT` import and the gate call). Attribution was not completed
   before reporting; flagging it so it is not mistaken for a regression from this work.
6. **THUNDER's offset-2SE floor is retained but demoted.** It is still recorded per task
   as `resolvability_floor_offset_2se`; it no longer gates anything, because its SD is
   taken over *datasets* while RI and HEST supply an across-*seed* SD, and a shared gate
   requires the same kind of number. Under the offset-2SE quantity the gate would be even
   stricter (all 9 THUNDER task cells fail, including virchow2/simple_shot at 24%).

## 7. Files changed

New: `scripts/eval_common.py`, `scripts/hest_seed_sd.py`, `docs/hest_seed_sd.json`,
this document.
Modified: `scripts/final_recipe_report.py`, `scripts/scoreboard.py`,
`scripts/scoreboard2.py`, `scripts/collect_final5.py`, `scripts/collect_hest.py`,
`scripts/build_stopping_dataset.py`, `scripts/aggregate_criterion_resolvability.py`,
`scripts/eval_stopping_rules.py`.
Regenerated: `docs/final_recipe_verdict.json`, `docs/stopping_criterion_rows.json`.
