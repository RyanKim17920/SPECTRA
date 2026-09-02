> **[STATUS BANNER — added 2026-08-31, see `docs/README.md` for the current doc map]**
>
> HISTORICAL (audit journal). The formula fixes it mandates are already merged into the current scripts. Treat the per-arm numbers here as a snapshot of that audit, not a current results table.

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

---

# F-P — pooled (ratio-of-means) aggregation, and the retirement of the per-cell
# denominator veto

Follow-up to `52af1e0`/`3fe9e1f`. Two defects, one of them introduced by the
unification commit itself.

## F-P1. Aggregation was mean-of-ratios; the grading rule is ratio-of-means

The rule is: **aggregate the numerator and the denominator first, then divide once.**

```
pct = mean_over_cells(our raw delta) / mean_over_cells(Waiv's raw gain) * 100
```

Per-cell percentages are never averaged. A mean of ratios is dominated by whichever cell
has the smallest denominator; the numbers below show how much that mattered.

Implemented ONCE, in `scripts/eval_common.pool_cells`. `final_recipe_report.py` and
`scoreboard.py` both call it; there is no second copy. It is applied at two levels:

* **backbone × THUNDER** — the 3 classification tasks pool into one numerator and one
  denominator.
* **benchmark** — the 3 backbones pool into one numerator and one denominator.

The **overall** figure remains the unweighted mean of the three benchmark percentages,
because that is what the criterion is written over (`>=70% on each of 3, >80% average of
the 3`); RI, HEST and THUNDER are in different units and cannot share a numerator.

### Why it mattered — virchow2/THUNDER

| task | Waiv gain | 1 seed-SD | gain / SD |
|---|---|---|---|
| knn | **−0.0030** | 0.00211 | 1.4 |
| linear_probing | +0.0030 | 0.00238 | 1.3 |
| simple_shot | +0.0270 | 0.00198 | 13.7 |

Two of the three are inside seed noise and one is **negative** — a per-task ratio against
a negative gain rewards regressing. Pooled, the denominator is **+0.0090** at 3.6× its
own 2-SD: a real scale. That is the whole argument for pooling.

## F-P2. The per-cell denominator veto was a precision test wearing a
## denominator-is-noise label

`52af1e0` flipped phikon/THUNDER from PASS to INDETERMINATE on the grounds that "Waiv's
gains are 13–18% of a seed-SD". **That framing is inverted and the gate was
over-rejecting.**

What the gate actually compared is correct in *quantity* — `seed_sd_of_task_mean` (the
across-seed SD of the 12-dataset task mean), not `offset_2se`, and not a per-seed SD where
the SD of a mean was wanted. Both of those confusions were checked for and neither is
present. The defect is the **threshold**.

`denominator_unresolvable` rejects when `seed_SD / |gain| * 100 > 10`, i.e. it demands

```
|Waiv gain|  >=  10 x seed_SD
```

Every real THUNDER cell sits at 5–7×:

| cell | Waiv gain | seed_SD | gain/SD | offset_2se | gain/offset_2se | sd_pct | old gate |
|---|---|---|---|---|---|---|---|
| phikon/knn | +0.0370 | 0.00639 | **5.8** | 0.0233 | 1.59 | 17.3 | REJECT |
| phikon/linear_probing | +0.0140 | 0.00189 | **7.4** | 0.0097 | 1.44 | 13.5 | REJECT |
| phikon/simple_shot | +0.0150 | 0.00264 | **5.7** | 0.0087 | 1.72 | 17.6 | REJECT |
| midnight/knn | +0.0170 | 0.00244 | **7.0** | 0.0100 | 1.70 | 14.4 | REJECT |
| midnight/linear_probing | +0.0020 | 0.00213 | 0.9 | 0.0087 | 0.23 | 106.3 | REJECT |
| midnight/simple_shot | +0.0370 | 0.00254 | 14.6 | 0.0104 | 3.56 | 6.9 | pass |
| virchow2/knn | −0.0030 | 0.00211 | 1.4 | 0.0083 | 0.36 | 70.3 | REJECT |
| virchow2/linear_probing | +0.0030 | 0.00238 | 1.3 | 0.0088 | 0.34 | 79.4 | REJECT |
| virchow2/simple_shot | +0.0270 | 0.00198 | 13.7 | 0.0066 | 4.09 | 7.3 | pass |

A gain at 5.8 seed-SD is not "13–18% of a seed-SD" and is not a denominator made of
noise. It is a denominator that is unambiguously real but known only to about ±35%
relative. Those are different statements, and only the first justifies withholding.

**Two tests were conflated:**

* **(A) Is the denominator real?** `|gain| > 2 x SD`. If it fails, the ratio's sign is
  undetermined and its distribution has no finite mean — no amount of our own seeds can
  fix it. This is genuinely n-independent, and it is what the module's own docstring
  describes.
* **(B) Is the denominator *precise* enough to separate 80% from 100%?**
  `2 x SD < 0.20 x |gain|`, i.e. the 10-point bar. This is a precision question — and a
  precision shortfall does not need a veto, because it can be **carried**.

### The fix: propagate, don't veto

Denominator uncertainty now enters the interval on the ratio (delta method, both terms):

```
CI = 2 * 100 * sqrt( (SE_num/den)^2 + (num*SD_den/den^2)^2 )
```

so the imprecision the old gate was reacting to is visible **in the error bar** rather
than deleting the cell. This is what the pre-`52af1e0` behaviour did implicitly:
phikon/knn graded 116.6 with interval [72.1, 161.1] — an interval that is wide *precisely
because* the denominator is imprecise, and that still clears 70.

The only surviving gate is (A), applied to the **pooled** denominator
(`eval_common.pooled_denominator_unresolvable`). No tuned constant was introduced: the
`2` is the same 2-sigma the CI construction already uses everywhere.

The retired 10-point test is still **computed and printed** per cell, as
`percell_denominator_diagnostic`. It no longer decides anything.

### Pooling alone would NOT have fixed this

Worth recording, because it is the reason both changes were needed:

| backbone | pooled Waiv gain | pooled SD | gain/2SD | sd_pct | gate (A) | old 10-pt gate |
|---|---|---|---|---|---|---|
| phikon | +0.02200 | 0.00239 | 4.60 | 10.9 | pass | **REJECT** |
| midnight | +0.01867 | 0.00137 | 6.80 | 7.3 | pass | pass |
| virchow2 | +0.00900 | 0.00125 | 3.60 | 13.9 | pass | **REJECT** |

Pooling raises phikon from 17.3 to 10.9 and virchow2 from 70.3/79.4 to 13.9 — a large
improvement that still lands on the wrong side of a 10-point bar. Only retiring the bar
makes those cells gradeable.

## Before / after — the published numbers

Aggregation change alone (both at HEAD, mean-of-ratios vs ratio-of-means):

| benchmark | mean-of-ratios | **pooled (ratio-of-means)** | delta |
|---|---|---|---|
| RI | 94.3 | **101.2** | +6.9 |
| HEST | 85.2 | **87.7** | +2.5 |
| THUNDER | 137.8 (2/3 backbones) | **WITHHELD** (needs 3/3) | — |

RI moves most because the per-backbone denominators span 16× (0.337 / 0.165 / 0.060), so
mean-of-ratios was silently weighting virchow2 sixteen times more heavily than phikon.

**Lead with the absolutes, not the percentage:**

| benchmark | our average increase | Waiv's average increase | pct | ±95% |
|---|---|---|---|---|
| RI | **+0.18966** | +0.18743 | 101.2 | ±3.6 |
| HEST | **+0.01500** | +0.01711 | 87.7 | ±10.2 |
| THUNDER | withheld (midnight PARTIAL) | — | — | — |

### The nine cells, vs `52af1e0`

| backbone | benchmark | at `52af1e0` | now | why |
|---|---|---|---|---|
| phikon | RI | 108.9 ±1.9 PASS | 108.9 ±1.9 PASS | unchanged |
| phikon | HEST | 90.5 ±13.3 PASS | 90.5 ±13.3 PASS | unchanged |
| phikon | THUNDER | **INDETERMINATE** | **104.6 ±27.4 PASS** | veto retired + 3 tasks pooled |
| midnight | RI | 92.5 ±4.1 PASS | 92.5 ±4.1 PASS | unchanged |
| midnight | HEST | 92.1 ±0.1 PASS | 92.1 ±0.1 PASS | unchanged (df=1 floor still disclosed) |
| midnight | THUNDER | PARTIAL | **PARTIAL** | simple_shot still 9,11/12 — 5 cells still running |
| virchow2 | RI | 81.6 ±11.2 PASS | 81.6 ±11.2 PASS | unchanged |
| virchow2 | HEST | **INDETERMINATE (withheld)** | **72.9 ±24.8 NOT RESOLVED** | veto retired; cell is now graded and honestly fails to resolve |
| virchow2 | THUNDER | 126.7 ±10.4 PASS (simple_shot only) | **171.0 ±51.3 PASS** | now the pooled 3-task figure, not one task |

Two of the three previously-ungradeable cells are now gradeable. The third,
midnight/THUNDER, is withheld for **coverage**, not for statistics: `simple_shot` has
9 and 11 of 12 datasets on its two seeds and the 12-dataset floor does not describe a
partial cell. The 5 outstanding midnight jobs are still running.

## Concentration — mandatory disclosure

Pooling cures a small denominator but can let one cell carry the group. Every pooled
group reports each cell's signed share of its numerator and denominator; >50% is flagged.

| group | cell | our delta | Waiv gain | num% | den% |
|---|---|---|---|---|---|
| RI / 3 backbones | **phikon** | +0.36749 | +0.33739 | **+65** | **+60** |
| RI / 3 backbones | midnight | +0.15272 | +0.16514 | +27 | +29 |
| RI / 3 backbones | virchow2 | +0.04876 | +0.05976 | +9 | +11 |
| HEST / 3 backbones | phikon | +0.01773 | +0.01960 | +39 | +38 |
| HEST / 3 backbones | midnight | +0.01980 | +0.02149 | +44 | +42 |
| HEST / 3 backbones | virchow2 | +0.00746 | +0.01023 | +17 | +20 |
| phikon/THUNDER / 3 tasks | **knn** | +0.04314 | +0.03700 | **+62** | **+56** |
| phikon/THUNDER / 3 tasks | linear_probing | +0.01972 | +0.01400 | +29 | +21 |
| phikon/THUNDER / 3 tasks | simple_shot | +0.00618 | +0.01500 | +9 | +23 |
| virchow2/THUNDER / 3 tasks | knn | +0.00504 | −0.00300 | +11 | −11 |
| virchow2/THUNDER / 3 tasks | linear_probing | +0.00690 | +0.00300 | +15 | +11 |
| virchow2/THUNDER / 3 tasks | **simple_shot** | +0.03422 | +0.02700 | **+74** | **+100** |

Three flags, and the last one is the important one: **virchow2/THUNDER = 171.0 is
essentially the simple_shot cell wearing the backbone's name.** simple_shot supplies 100%
of that pooled denominator — the other two tasks' Waiv gains (−0.0030 and +0.0030) cancel
exactly. Do not quote 171.0 without this line. The RI/phikon and phikon/knn flags are
milder: those cells dominate because their gains are genuinely the largest, not because
the rest cancelled.

## Error propagation on the pooled quantity

`SE_agg = sqrt(sum_i SE_cell_i^2) / k`, with `SE_cell = per-seed SD / sqrt(n_seeds)` from
`docs/thunder_seed_floor_12ds.json` and `docs/hest_seed_sd.json`. The same construction
gives the denominator's SD from each cell's one-run seed SD; both are carried onto the
ratio by the delta method.

**Disclosed under-estimate:** quadrature assumes independent cell noise. The three
THUNDER tasks of one backbone are three readouts of the *same* per-seed checkpoints over
the *same* 12 datasets, so their noise is correlated and `SE_agg` is too small there by up
to sqrt(3) ≈ 1.73×. This is reported in every pooled block as `independence_caveat`
rather than patched, because patching it means either assuming perfect correlation (also
wrong, in the other direction) or introducing a fitted correlation constant.

## All-or-nothing pooling

A pooled number requires **every** cell of its group. Currently withheld on that rule:

* `midnight/THUNDER` — `simple_shot` is PARTIAL (9,11 of 12 datasets).
* `THUNDER / 3 backbones` — because midnight/THUNDER is withheld above.

THUNDER classification is at 211/216 runs; phikon and virchow2 are COMPLETE at 36/36 on
both seeds. Both withheld groups resolve when the 5 outstanding midnight jobs land.

## Is the criterion computable? **NO — but only just, and only on coverage.**

Criterion: `>=70% on each of the 3 benchmarks` and `>80% average of the 3`.

* RI = **101.2** ✔
* HEST = **87.7** ✔
* THUNDER = **withheld** ✘

Exactly one thing blocks it: `midnight/THUNDER/simple_shot` has 9 and 11 of 12 datasets.
Nothing statistical is in the way — the phikon and virchow2 THUNDER cells are now graded
(104.6 and 171.0), and the pooled denominators all clear the sign gate. When the 5
outstanding midnight jobs land, all three benchmark numbers exist and the criterion is
computable in one command.

`worst cell` is `virchow2/HEST = 72.9 ±24.8` — graded, above the 70 bar on its point
estimate, but its interval straddles it, so the per-cell verdict is NOT RESOLVED. That is
a real, unresolved cell and is not fixed by any aggregation change.

## Also fixed

`scripts/scoreboard.py` crashed with `KeyError: None` at `ARM_BACKBONE[arm]`, from the
concurrent H-Optimus-0/UNI2-h backbone registration. Fixed defensively (`.get`, returning
None for an unknown arm) — the same class as the `RI_BASE.get(None)` fix in `a7ff3eb`. A
second instance of the same class was found immediately behind it: `print_denominators`
formatted `None` RI/HEST denominators with `%f` for those same unregistered arms. Both
now render `N/A`; an unmeasured denominator is never substituted.
