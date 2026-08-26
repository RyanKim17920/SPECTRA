# Eval-pipeline defect fixes — 2026-08-26

Audit of the verdict/reporting path found 15 confirmed defects. All Tier-1 and Tier-2
code defects are fixed. This file records, for each: what was wrong, what changed, and
every published number that moved.

Scripts touched: `scripts/final_recipe_report.py`, `scripts/scoreboard.py`,
`scripts/collect_final5.py`, `scripts/collect_thunder.py`.
Report re-run: `python3 scripts/final_recipe_report.py --json`.

**No SLURM jobs were launched and nothing was pushed.**

---

## Headline: the verdict did not change, but two cells did

`FINAL VERDICT: INDETERMINATE` before and after. The reasons changed, and two
previously-comfortable results are now materially worse:

| | before | after |
|---|---|---|
| `overall_average` | **85.4** | **UNDEFINED** (THUNDER rests on 1/3 backbones) |
| phikon / THUNDER / simple_shot | 41.2 ± 41.0 → NOT RESOLVED | 41.2 ± 24.9 → **FAIL** |
| midnight / THUNDER / knn | n=0 PARTIAL | n=1 → **UNDERPOWERED** (would have graded PASS at 131.8 without the new gate) |
| virchow2 / HEST | 73.1 | 72.9 (n=2, rule-selected) — and **67.8 at n=5** including the three assumed-step seeds |
| phikon / THUNDER (aggregate) | 99.6 ± 13.2 | 99.6 ± **22.8** |

---

## Tier 1

### F1 — `overall_average` weighted every benchmark equally regardless of coverage

`final_recipe_report.py` built `overall_average` as the unweighted mean of the three
benchmark means, no matter how many backbones each rested on. THUNDER's mean comes from
**one** backbone (phikon), yet carried the same 1/3 weight as RI and HEST, which cover
all three. A benchmark mean over a subset of backbones is not "the benchmark"; it is a
different quantity that happens to share the name.

**Fix.** A benchmark mean is `eligible_for_overall` only at 3/3 backbone coverage. If any
benchmark is short, `overall_average` is `None` (UNDEFINED) rather than quietly rebuilt
from what is left, and the `PASS` branch is unreachable because the 80 bar cannot be
tested. The old-style partial mean is still emitted as
`overall_average_partial_NOT_THE_CRITERION` for transparency, labelled do-not-quote.

| number | before | after |
|---|---|---|
| `overall_average` | 85.4 | **UNDEFINED** — "THUNDER rests on fewer than 3 backbones (THUNDER=1/3)" |
| (partial mean, not the criterion) | — | 85.4 |

### F2 — no minimum-n gate anywhere; n=1 cells were graded

Every CI path manufactured an error bar at n=1: RI fell back to `floor_ci`, HEST computed
`2*sd/sqrt(1)`, THUNDER computed `floor/gain*100/sqrt(1)`. `scoreboard.recipe_verdict`
was worse still — it compared a **bare point estimate** against the 70 bar with no CI at
all, and computed `n` only to never use it.

**Fix.** `MIN_N_FOR_VERDICT = 2` in both `final_recipe_report.grade()/resolve()` and
`scoreboard._cell_verdict()`. Below it the status is `UNDERPOWERED`, `pct` is set to
`None`, and the cell feeds no average and no worst-cell search. `UNDERPOWERED` is in the
`UNGRADED` set, so `PASS` can never be printed over one.

This caught a live case on the first run. Between the before- and after-runs a THUNDER
job finished, giving midnight seed 1 full 12/12 knn coverage:

| cell | before | after |
|---|---|---|
| midnight / THUNDER / knn | n=0, PARTIAL (coverage 11,11/12) | n=1, **UNDERPOWERED** |

Without F2 that cell would have been graded **PASS at 131.8 ± 28.7 off a single seed**.

### F3 — three THUNDER task CIs combined in quadrature as if independent

`ci = sqrt(sum(ci^2)) / len(graded)`. knn, linear_probing and simple_shot are three
readouts of the **same per-seed checkpoints** over the **same 12 datasets**; a seed that
shifts one shifts all three. Treating them as independent understated the aggregate
half-width by up to sqrt(3) ≈ 1.73×.

**Fix.** Perfect correlation: the half-width of the mean is the mean of the half-widths.

| cell | CI before | CI after |
|---|---|---|
| phikon / THUNDER (aggregate) | ± 13.2 | **± 22.8** |

Point estimate unchanged at 99.6; status stays PASS.

### F4 — THUNDER per-task CI derived from the wrong variance component

The code used `offset_2se / |waiv_gain| * 100 / sqrt(n)`, where
`offset_2se = |mean(d)| + 2*SD(d)/sqrt(12)` and the SD is taken **over the 12 datasets**.
That is a *resolvability floor* — the right tool for asking whether Waiv's own gain even
exceeds seed noise — but it is not a 95% half-width on our task mean, and dividing an
already-sqrt(12)-shrunk dataset-level SD again by `sqrt(n_runs)` compounds the error.

**Fix.** Use `2 * seed_SD_12ds / |waiv_gain| * 100 / sqrt(n)`, where `seed_SD_12ds` is
`seed_sd_of_task_mean` — the SD of the 12-dataset task mean itself across the 5 training
seeds. Read at import from `docs/thunder_seed_floor_12ds.json`
(`cells[bb/task].12ds.seed_sd_of_task_mean`), with literals as fallback. The
INDETERMINATE resolvability gate still uses `offset_2se`, which is its correct use; both
are now recorded per task (`ci_source`, `resolvability_floor_offset_2se`).

| cell | pct | CI before | CI after | status before | status after |
|---|---|---|---|---|---|
| phikon / knn | 116.6 | ± 44.5 | ± 24.4 | PASS | PASS |
| phikon / linear_probing | 140.9 | ± 49.0 | ± 19.1 | PASS | PASS |
| **phikon / simple_shot** | **41.2** | ± 41.0 | **± 24.9** | NOT RESOLVED | **FAIL** |
| midnight / knn | 131.8 | ± 58.8 | ± 28.7 | PASS | UNDERPOWERED (F2) |

**This is the most consequential single change.** phikon/simple_shot was previously
excused as "not resolved"; on the correct error bar its interval is [16.4, 66.1], which
lies **entirely below the 70 bar**. It is a measured failure, not missing information.
See "Newly revealed" below — the aggregate still reports PASS over it.

### F5 — virchow2 HEST stuck at n=2 when n=5 exists on disk

`runs/genMASK-c3s-*-virchow2-s{2,3,4}` are TRAIN_DONE and have HEST summaries at **every**
step, but have no `ri_curve.json`. `select_step` therefore returned `(None, None, [])`,
`discover_runs`→`by_bb` dropped them, and they were excluded from **all three**
benchmarks — including HEST, which does not need the curve.

**Fix.** A clean, opt-in, reversible flag: `--hest-assume-step STEP`. Runs with HEST
scores but no curve enter a clearly-marked **supplementary** HEST cell at that explicit
step, recording `step_source` and naming every assumed-step run. They never enter the RI
or THUNDER cells (both genuinely need the curve) and never the primary rule-selected HEST
cell the verdict is scored on. Off by default; drop the flag to revert.

Run as `python3 scripts/final_recipe_report.py --hest-assume-step 125` (125 is the step
the rule selected for virchow2 seeds 0 and 1):

| virchow2 HEST | n | pct | 95% CI | interval | status |
|---|---|---|---|---|---|
| rule-selected (scored) | 2 | **72.9** | ± 20.1 | [52.8, 93.0] | NOT RESOLVED |
| + assumed step 125 (supplementary) | 5 | **67.8** | ± 12.7 | [55.1, 80.5] | NOT RESOLVED |

**The three extra seeds pull the point estimate from 72.9 to 67.8 — below the 70 bar.**
Still NOT RESOLVED either way, but the direction matters and the n=2 figure should not be
quoted alone. CI-backfill jobs 393547/548/549 will produce the missing curves; once they
land, drop the flag and the five seeds enter the primary cell on the rule.

---

## Tier 2

### F6 — HEST base and fine-tuned scores came from different JSON fields

Three hardcoded dicts (`collect_final5.py`, `final_recipe_report.py`, `scoreboard.py`)
stored the base from the rounded `results.avg` field, while `collect_final5._hest_score`
and `scoreboard` read the fine-tuned score from
`hest_perf_per_encoder.custom_encoder`. Base and FT from different fields biases every
`pct_of_waiv`.

**Fix.** One field repo-wide (`custom_encoder`, unrounded), one pooling rule
(`collect_final5.hest_pooling`), one loader (`collect_final5._load_hest_base`) reading the
base **from disk** rather than hardcoding it. `HEST_BASE_FALLBACK` literals remain only
for when the summary JSON is absent, and `HEST_BASE_SOURCE` records the provenance of
each value. `final_recipe_report.hest_score` — which had been reading `results.avg` — now
delegates to the shared loader.

| backbone | base before | base after | HEST pct before | after |
|---|---|---|---|---|
| phikon | 0.3747000 | 0.3747009 | 90.481 | 90.480 |
| midnight | 0.3952100 | 0.3952102 | 92.137 | 92.137 |
| **virchow2** | **0.4032400** | **0.4032685** | **72.989** | **72.914** |

Phikon and midnight agree to ~1e-7 as expected; only virchow2 moved (base was low by
2.4e-5, inflating the pct by +0.075). Benchmark average: HEST **85.3 → 85.2**.

### F7 — two modules exported `PAPER_SEG` with different contents

`collect_final5.PAPER_SEG` = 2 datasets (what we submitted); `collect_thunder.PAPER_SEG`
= 4 (Waiv's published panel). Which one a consumer got depended on which module it
imported.

**Fix.** The names are now distinct and self-describing:
`collect_thunder.PAPER_SEG_PUBLISHED` (4) and `PAPER_SEG_SUBMITTED` (2);
`collect_final5.PAPER_SEG` (2) documents the split. The roster print in `collect_thunder`
now reports both panels separately, so a 2-dataset mean can never be read as Waiv's
4-dataset published mean.

### F9 / F10 — pooling rules

The HEST pooling rule was transcribed in two places; it is now
`collect_final5.hest_pooling()` alone, which `final_recipe_report.HEST_POOLING` derives
from.

`scoreboard._thunder_pooling` was **wrong for midnight**: it returned the HEST rule
(`cls` unless virchow2). `collect_final5.THUNDER_BASE_DIRS` — the authority — has
`base_cls` (phikon) / `mbase_clsmean` (midnight) / `vbase_clsmean` (virchow2), matching
`docs/thunder_seed_floor_12ds.md`.

| arm | before | after |
|---|---|---|
| phikon | cls | cls |
| **midnight** | **cls** | **clsmean** |
| virchow2 | clsmean | clsmean |

Display-only field, so **no published number moved** — but a wrong protocol label on a
results table is exactly how protocol-mismatch bugs get reintroduced.

### F11 — `CHECKED_CONFIG_KEYS` omitted every recipe-defining knob

Runs with **opposite** negative-masking and cls-bias settings were being pooled as
COMPARABLE. Added 13 keys: `mask_same_core`, `same_core_logit_bias`,
`same_core_logit_bias_cls`, `same_core_logit_bias_mean`, `weight_decay`,
`center_embeddings`, `cores_per_batch`, `grad_accum`, `grad_clip`, `ckpt_schedule`,
`core_labels_path`, `encoder.proj_out_dim`, `encoder.pooling` (16 → 29). The existing
vacuous-key warning is kept.

**Comparability re-run — no previously-pooled run is newly flagged incomparable:**

| cohort | runs | incomparable (16 keys) | incomparable (29 keys) | vacuous |
|---|---|---|---|---|
| final5-*-t900 | 15 | 0 | **0** | 7 |
| genMASK-c3s (the recipe) | 9 | 0 | **0** | 1 |

The 4 runs `collect_final5` excludes (`final5-phikon-s{0,1}-t{450,1800}`) are excluded on
`grid_tiles`, before and after. **No pooled result is invalidated.**

Caveat, correctly surfaced by the existing vacuous-key warning: 7 of the new keys
(`mask_same_core`, `same_core_logit_bias*`, `center_embeddings`, `cores_per_batch`,
`ckpt_schedule`, `core_labels_path`) are absent from **every** `final5-*` config, because
that generation predates those knobs — so on that cohort they still verify nothing. They
are present and verified on the genMASK-c3s recipe cohort, which is the one the verdict
rests on.

### F12 — `results_backup/hest_sub5/` fallback

`_hest_score` fell back to a 5-task subset with a **byte-identical schema**, so a hit
there was silently indistinguishable from a real 9-task score. The branch is deleted.
No current number came from it (all resolve in the live `hest_work` dir), so nothing moved
— but the trap is gone.

### F13 — stopping rule averaged over "datasets present in this point"

A checkpoint probed without camelyon (which sits at 0.08–0.40 while the others reach 1.87)
would show a mean ~+0.25 too high and trip the 0.75 gate early.

**Fix.** A point is eligible only with all `N_CI_DATASETS = 3` present
(camelyon, tcga, tolkach_esca); short points remain in the trace with an explicit
`skip_reason` so the skip is auditable. Per-dataset CI is now recorded in the trace, not
just the mean.

**Audit result: 0 short points across all 9 runs.** No selected step changed. The guard is
prophylactic.

### F14 — RI empirical SD computed over *capped* pcts

The SD was taken over capped pcts about the capped mean — the censored-SD problem the
function's own docstring warns about. Two seeds both above Waiv both become exactly 100.0
and the SD collapses to 0.

**Fix.** Compute `emp_sd` from the uncapped pcts.

| backbone | empirical CI before | after | reported CI (max with floor) |
|---|---|---|---|
| **phikon** | **0.00** | **0.57** | 1.90 (unchanged) |
| midnight | 1.18 | 1.18 | 4.13 (unchanged) |
| virchow2 | 3.45 | 3.45 | 11.23 (unchanged) |

phikon was exactly the censored case — a literal `±0.0` claim of infinite precision. No
reported CI moved, because the measured seed floor is wider on all three and the code
already takes the max; the fix removes the failure mode rather than a live error.

---

## Newly revealed — reported, not improvised

Per the brief, these were found while fixing and are **not** acted on:

1. **A FAILING task sits inside a PASSING benchmark cell.** With F4's correct error bar,
   phikon/THUNDER/simple_shot is 41.2 [16.4, 66.1] → **FAIL**. But the phikon/THUNDER
   *cell* — which is what the worst-cell criterion scores — averages the three tasks to
   99.6 and reports **PASS**. The criterion is defined at (backbone, benchmark)
   granularity, so a task that measurably fails is being averaged away by two that exceed
   Waiv. The script's own honesty rule ("PASS is never printed when a cell is PARTIAL,
   INDETERMINATE or NOT RESOLVED") has no equivalent for a FAILING sub-task.
   **This needs a decision on criterion granularity, not a code patch.**

2. **`scripts/scoreboard.py` crashes on a full run** (pre-existing, unrelated to these
   fixes): `TypeError: must be real number, not NoneType` at `print_run_block`, the
   `"  RI: %.5f | Waiv %.3f ..."` line, on the first UNKNOWN-BACKBONE run
   (`armd-s1-385294`), whose `RI_BASE.get(None)` is `None`. Verified pre-existing by
   reverting all scoreboard edits in a scratch copy and reproducing the identical
   traceback. Consequence: the scoreboard's verdict table never prints on a full run, so
   the F2 gate added there is covered only by unit test (`_cell_verdict` at n=1 →
   `UNDERPOWERED`; at n=2 → `PASS`), not end-to-end.

3. **Data drift during the audit.** A THUNDER job completed between the before- and
   after-runs, taking midnight seed 1 knn from 11/12 to 12/12 coverage. The
   before/after tables above isolate code effects by recomputing both formulas on the
   *same* current data; only the midnight/knn row reflects drift, and it is labelled.

---

## Not fixed, with reasons

- **Criterion granularity** (item 1 above) — a judgement call about what the 70 bar
  applies to, not a defect with a correct answer.
- **`scoreboard.py` print crash** (item 2) — pre-existing and out of the audit's scope.
- **The `offset_2se` resolvability gate** still uses the dataset-level floor. That is its
  correct use (is Waiv's own gain bigger than seed noise?), distinct from F4's half-width;
  both are now recorded per task so the two can never be confused again.
- **`retention_kl_weight` / `retention_kl_temperature`** are arguably recipe-defining and
  are still unchecked in `CHECKED_CONFIG_KEYS`. Not in the audit's list; flagged here.
- **Dated session logs** (`docs/FINDINGS_2026-08-16.md`) are annotated with supersession
  notes rather than rewritten.

---

## Tier 3 — documentation

Each claim was verified against code/data before editing. Numbers were corrected **in
place** with a bracketed note naming the authority; nothing was deleted. Dated session
logs (`docs/FINDINGS_2026-08-16.md`) were annotated, not rewritten.

### F15 — THUNDER dataset-coverage overclaim

`docs/CAVEATS.md:34-37` and `docs/RESULTS.md:162-163` both claimed **"16 of Waiv's 16
datasets (12/12 classification, 4/4 segmentation)"** and that our segmentation average is
therefore **like-for-like against theirs**.

| | before | after |
|---|---|---|
| total coverage | 16 of 16 | **14 of 16** |
| classification | 12/12 | 12/12 (correct) |
| segmentation | **4/4** | **2/4** |
| comparability claim | "like-for-like against theirs" | "NOT directly comparable — flagged `support_2v4`" |

`segpath_epithelial` and `segpath_lymphocytes` were deliberately never submitted
(`collect_final5.PAPER_SEG` = 2 vs `collect_thunder.PAPER_SEG_PUBLISHED` = 4). The
like-for-like claim was already directly contradicted by
`docs/baseline_comparability_audit.md:138` and `docs/FINAL5_RESULTS.md:130-132`. The "20"
figure suspected in the audit appears nowhere.

### F16 — wrong-protocol Midnight HEST base 0.4121

0.41210 is `mbase_clsmean_summary.json` — Midnight under **clsmean**, the wrong HEST
pooling. The matched-`cls` base is **0.39521**. Corrected where it was used *as if it were
the base*:

| location | before | after |
|---|---|---|
| `RESULTS.md:18` (protocol row) | "`clsmean` on Midnight and Virchow2" | "`cls` on Midnight; `clsmean` on Virchow2 only" |
| `RESULTS.md:19` (base row) | 0.4121 (Midnight) | **0.39521** (Midnight, `cls`) |
| `RESULTS.md:123` | 0.4121 | **0.39521** |
| `RESULTS.md:129` | "our base 0.4121 is **above** their reported base (+0.0169)" | matched base 0.39521 agrees with their 0.3952 essentially exactly; **there is no +0.0169 base excess** |
| `RESULTS.md:290` | Avg 0.4121 → 0.4132, Δ **+0.0011** | base 0.39521; the +0.0011 delta marked **INVALID**; matched 5-seed delta is **+0.0113** |
| `CAVEATS.md:45` | "HEST: … Midnight (`clsmean`)" | "Midnight (`cls`)" — with a note that Midnight IS `clsmean` on THUNDER, a different rule |

Left alone deliberately: `FINAL_RESULTS.md:89`, `WAIV_COMPARISON.md:124/140/145`,
`generation_comparison.md:84/122`, `baseline_comparability_audit.md:90` already *label*
0.41210 as the wrong-protocol value. `RESULTS.md:321` and `:288` describe a sweep that
genuinely ran under `clsmean`, so the number stands and was annotated instead.
`FINAL5_RESULTS.md` and `NEW_MODEL.md` contain no occurrence of 0.4121 — that part of the
audit was a false alarm.

### F17 — stale 5-dataset THUNDER seed floors

`docs/FINAL_RESULTS.md` quoted floors from the superseded 5-dataset, **n=2-seed** study.
Current authority is the 12-dataset, **n=5-seed** `docs/thunder_seed_floor_12ds.md`.

| location | before | after |
|---|---|---|
| `:43`, `:121` phikon LP floor | 0.0156 | **0.0097** |
| `:43`, `:122` midnight LP floor | 0.0208 | **0.0087** |
| `:121` noise units (cls) | 0.61 | **0.98** (0.0095/0.0097) |
| `:122` noise units (clsmean) | 0.60 | **1.43** (0.0124/0.0087) |
| `:128` pointer | `thunder_seed_floor.{py,json,md}` | `thunder_seed_floor_12ds.{py,json,md}` |

Both protocols remain under ~2 noise units, so the "blind" verdicts stand. One reversal
was flagged rather than smoothed over: at `:43` the cls difference 0.0104 now marginally
**exceeds** its 0.0097 floor, where the original prose called both flat. A supersession
banner at `:5` records the two-axis nuance — the old numbers were off on dataset count
(5 vs 12) **and** run family, which move floors in opposite directions
(`thunder_seed_floor_12ds.md:145-152`).

### F18 — single-seed HEST gain quoted as a replicated result

"+0.0166, 77% of Waiv" on Midnight HEST is **one seed** (job 386398). It does not
replicate: the 5-seed result is **+0.0113, ~53%**
(`final5_results.json: aggregates.midnight.hest.delta_vs_base = 0.011289`).

| location | before | after |
|---|---|---|
| `FINAL_RESULTS.md:90` | FT 0.41180, Δ **+0.0166**, "2.2× bar — real, 77%" | FT 0.4065, Δ **+0.0113 (n=5)**, "1.5× bar — marginal, ~53%" |
| `FINAL_RESULTS.md:92` | "flipped from null to a real **77%** recovery" | "**~53%**" |
| `WAIV_COMPARISON.md:141` | single row, +0.0166 | split into an n=1 row and an n=5 row |
| `WAIV_COMPARISON.md:143` | "we close **77%**" | "**~53%** (n=5, +0.0113)" |
| `WAIV_COMPARISON.md:164` | "2.2x — real, **77%** of theirs" | "1.5x — marginal, **~53%** of theirs" |

**A second-pass correction was needed here.** The first pass wrote the n=5 row as
**"6.6× the bar — REAL"**, which is wrong. The 0.0075 bar is 2SE for an **n=5 mean**
already (`RESULTS.md:1794`: per-task SD 0.0084, SE = 0.0084/√5 = 0.0037, 2SE = 0.0075), so
+0.0113 / 0.0075 = **1.5×** — it clears the bar, but marginally, not overwhelmingly.

That correction exposed something sharper about the original claim. The n=1 row was being
graded against a **five-seed** bar. The correct single-run bar is 2 × SD = 2 × 0.0084 =
**0.0168**, against which +0.0166 is **0.99× — inside noise**. The "77% of Waiv" result
was never significant on its own terms, which is exactly why it failed to replicate.

### F19 — FINAL5_RESULTS.md placeholder and sign-test precision

| location | before | after |
|---|---|---|
| `:44` midnight per-seed list | ends in the literal placeholder **`+(s3)`**, and is mis-ordered (printed 4th value +0.01282 is seed 4) | seed order s0..s4: +0.01173, +0.00806, +0.01256, **+0.01128**, +0.01282 (mean 0.01129, matching `delta_vs_base`) |
| `:40` sign test | "15/15 seeds positive. Binomial sign test p ≈ 3×10⁻⁵." | arithmetic confirmed correct; caveat added |

The "15/15, p ≈ 3×10⁻⁵" arithmetic is **correct** (all 15 deltas recomputed from
`final5_results.json`; one-sided 0.5^15 = 3.05e-5) — that part of the audit was a false
alarm. Two things were unstated and are now noted: the test is one-sided (two-sided
6.1e-5), and n=15 is 3 backbones × 5 seeds, not 15 independent units — the 5 deltas within
a backbone share one base constant and one recipe, so the stated p overstates the
precision.

---

## Regenerated artifacts

- `docs/final_recipe_verdict.json` — re-emitted by `final_recipe_report.py --json`.
- `docs/final5_results.json` — re-emitted by `collect_final5.py`. Two changes: the 13 new
  `CHECKED_CONFIG_KEYS` now appear in the config block (7 `null` on this cohort, correctly
  reported by the vacuous-key warning), and THUNDER coverage grew because jobs landed
  during the audit (e.g. midnight gained `esca`, `patch_camelyon`, `tcga_uniform`,
  `wilds`). The F6 base change is below the display precision of the tables in
  `docs/FINAL5_RESULTS.md`, which still quote virchow2 base as 0.40324 (now 0.4032685);
  those tables were not regenerated.
