# Baseline comparability audit — is `pct_of_waiv` measuring what it claims?

Date: 2026-08-25. Read-only audit; no code changed, no jobs touched.

`pct_of_waiv = (ours - base) / (Waiv_ft - base)` is only a statement about *our recipe*
if our base and Waiv's base are the same starting point. This document compares the two,
cell by cell, and states where the percentage survives.

Sources: `scripts/collect_final5.py` (`HEST_BASE` L40-52, `RI_BASE` L64-68,
`THUNDER_BASE_DIRS` L90-94), `scripts/scoreboard.py` (`WAIV_THUNDER` L127-140,
`_pct_of_waiv` L269, `_pct_of_waiv_two_base` L290), `docs/waiv_published.json`,
`docs/final5_results.json` (n=5 seeds, step 500), THUNDER results read live from
`/data/ryan.kim/thunder/outputs/res`.

---

## 1. The table

`our_ft` = final5 5-seed mean at step 500. `gap` = our_base − waiv_base.
`|gap|/gain` = baseline discrepancy as a multiple of the entire effect Waiv reports.
`ft_gap` = our_ft − waiv_ft (the acid test: are we actually above or below their model?).

| backbone | cell | our_base | waiv_base | gap | waiv_gain | **\|gap\|/gain** | our_ft | waiv_ft | our_gain | pct | **ft_gap** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| phikon | RI | 0.46860 | 0.46900 | −0.00040 | +0.33700 | **0.00** | 0.82694 | 0.80600 | +0.35834 | 106.3 | **+0.02094** |
| phikon | HEST | 0.37470 | 0.37470 | +0.00000 | +0.01960 | **0.00** | 0.38709 | 0.39430 | +0.01239 | 63.2 | −0.00721 |
| phikon | THUNDER.knn | 0.70281 | 0.74000 | −0.03719 | +0.03700 | **1.01** | 0.73770 | 0.77700 | +0.03489 | 94.3 | −0.03930 |
| phikon | THUNDER.linear | 0.76541 | 0.79300 | −0.02759 | +0.01400 | **1.97** | 0.78235 | 0.80700 | +0.01694 | 121.0 | −0.02465 |
| phikon | THUNDER.simple_shot | 0.69330 | 0.71800 | −0.02470 | +0.01500 | **1.65** | 0.69571 | 0.73300 | +0.00241 | 16.1 | −0.03729 |
| phikon | THUNDER.seg (2v4) | 0.70401 | 0.66500 | +0.03901 | −0.01200 | **3.25** | 0.69526 | 0.65300 | −0.00876 | GUARD | +0.04226 |
| midnight | RI | 0.75890 | 0.75900 | −0.00010 | +0.16500 | **0.00** | 0.89927 | 0.92400 | +0.14037 | 85.1 | −0.02473 |
| midnight | HEST | 0.39521 | 0.39520 | +0.00001 | +0.02150 | **0.00** | 0.40650 | 0.41670 | +0.01129 | 52.5 | −0.01020 |
| midnight | THUNDER.knn | 0.78254 | 0.80000 | −0.01746 | +0.01700 | **1.03** | 0.78681 | 0.81700 | +0.00427 | 25.1 | −0.03019 |
| midnight | THUNDER.linear | 0.82880 | 0.84400 | −0.01520 | +0.00200 | **7.60** | 0.83458 | 0.84600 | +0.00578 | 289.2→100 | −0.01142 |
| midnight | THUNDER.simple_shot | 0.70639 | 0.71500 | −0.00861 | +0.03700 | **0.23** | 0.74891 | 0.75200 | +0.04252 | 114.9→100 | −0.00309 |
| midnight | THUNDER.seg (2v4) | 0.70116 | 0.66000 | +0.04116 | +0.01600 | **2.57** | 0.70564 | 0.67600 | +0.00448 | 28.0 | +0.02964 |
| virchow2 | RI | 0.85820 | 0.85800 | +0.00020 | +0.06000 | **0.00** | 0.89770 | 0.91800 | +0.03950 | 65.8 | −0.02030 |
| virchow2 | HEST | 0.40324 | 0.40340 | −0.00016 | +0.01010 | **0.02** | 0.40555 | 0.41350 | +0.00231 | 22.9 | −0.00795 |
| virchow2 | THUNDER.knn | 0.80874 | 0.82900 | −0.02026 | −0.00300 | **6.75** | 0.79282 | 0.82600 | −0.01592 | GUARD | −0.03318 |
| virchow2 | THUNDER.linear | 0.83253 | 0.84800 | −0.01547 | +0.00300 | **5.16** | 0.83228 | 0.85100 | −0.00025 | −8.2 | −0.01872 |
| virchow2 | THUNDER.simple_shot | 0.72749 | 0.73900 | −0.01151 | +0.02700 | **0.43** | 0.74491 | 0.76600 | +0.01741 | 64.5 | −0.02109 |
| virchow2 | THUNDER.seg (2v4) | 0.71117 | 0.68200 | +0.02917 | −0.00200 | **14.58** | 0.71458 | 0.68000 | +0.00341 | GUARD | +0.03458 |

VERIFIED from disk: every `our_base` and `our_ft` number. THUNDER bases recomputed this
session through `collect_final5._thunder_per_ds_by_model`, 12/12 dataset coverage on every
classification cell, 2/2 on segmentation, zero missing files.

### Cells where the baseline gap exceeds the effect being measured

**Nine of twelve THUNDER cells**: phikon knn (1.01), phikon linear (1.97), phikon
simple_shot (1.65), phikon seg (3.25), midnight knn (1.03), midnight linear (7.60),
midnight seg (2.57), virchow2 knn (6.75), virchow2 linear (5.16), virchow2 seg (14.58).
In these cells the number that decides the percentage is an unexplained calibration
difference, not our recipe. **Zero RI or HEST cells** are in this condition; the worst is
virchow2 HEST at 0.02.

---

## 2. Why the bases differ, per benchmark

### RI — they don't differ. VERIFIED at per-dataset granularity.

`RI_BASE` (`collect_final5.py:64-68`) = 0.4686 / 0.7589 / 0.8582 vs Waiv Table 1
0.469 / 0.759 / 0.858. Max gap 0.0004 = 0.33% of the smallest Waiv gain.

This is not a rounding coincidence: `docs/RESULTS.md:189-197` records our measured
**per-dataset** base RIs, and all nine match Waiv's published per-dataset values:

| backbone | camelyon (ours/Waiv) | tolkach (ours/Waiv) | tcga (ours/Waiv) |
|---|---|---|---|
| phikon-v2 | 0.0190 / 0.019 | 0.7681 / 0.768 | 0.6188 / 0.619 |
| Midnight-12k | 0.4780 / 0.478 | 0.9411 / 0.941 | 0.8575 / 0.858 |
| Virchow2 | 0.7989 / 0.799 | 0.9541 / 0.954 | 0.8218 / 0.822 |

Virchow2's per-dataset base RIs were **never published** (`waiv_published.json` carries the
avg; `pathorob_adapter.py` holds only `{"virchow2_base": {"avg": 0.858}}`), so those three
values cannot have been copied — they were measured and they reproduce.

**One caveat on corroboration.** The `targets` block inside `runs/*/ri_curve.json`
(172 of 187 curves carry the full 3-backbone version) contains
`phikon_v2_base 0.469 / midnight_base 0.759 / virchow2_base 0.858` — these are a
transcription of Waiv's published numbers, **not** an independent measurement, and must not
be cited as a second source. The independent source is the per-dataset table above.

### HEST — they don't differ *once the pooling is chosen to make them not differ*.

| backbone | pooling used | our base | Waiv base | gap | wrong-pooling value |
|---|---|---|---|---|---|
| phikon-v2 | `cls` | 0.37470 | 0.3747 | 0.00000 | `clsmean` 0.39144 |
| Midnight-12k | `cls` | 0.39521 | 0.3952 | +0.00001 | `clsmean` 0.41210 |
| Virchow2 | `clsmean` | 0.40327 | 0.4034 | −0.00013 | `cls` 0.39791 |

All five values re-read from disk this session. The wrong-pooling column is the point: a
pooling error moves the base by 0.005–0.017, i.e. **0.5× to 1.7× the entire Waiv gain**.
The bases agree only because the protocol was matched.

**This agreement is partly circular and should be described as such.** Per
`docs/WAIV_COMPARISON.md:147`, the pooling was *selected* by scoring the base under both
options and keeping whichever reproduced Waiv's published base. So "our base matches theirs"
is true by construction of the selection rule. What is *not* circular, and is genuine
evidence, is that a match at 0.00013 exists at all — an arbitrary protocol would not land
that close, so the inference "Waiv used clsmean on Virchow2, cls elsewhere" is sound.

**Bug found (cosmetic, 3e-5).** `collect_final5.py:47-50` states the virchow2 base file
"stores 0.40324" and that `docs/FINAL_RESULTS.md`'s 0.40327 is the deviant. The reverse is
true: `results_backup/hest_work_results/vbase_clsmean_summary.json` and
`/data/ryan.kim/hest_work/results/vbase_clsmean_summary.json` both store
**0.4032685185185185**. The constant 0.40324 is off by 3.0e-5 = 0.3% of the 0.0101 gain, so
it does not change any verdict, but the comment justifying it is factually wrong and
propagates to `scripts/aggregate_criterion_resolvability.py:52`, `docs/FINAL_RECIPE.md:240`
and `docs/round_temp_dose.md:136`.

### THUNDER — the bases differ, and **Waiv is the outlier**, not us. VERIFIED.

The reproduction claim at `scoreboard.py:98-107` is **true**. Recomputed this session
against `collect_thunder.PUBLISHED` (THUNDER's own paper, arXiv:2507.07860v3 Tables
S37/S39/S50), phikon-v2 base, 12 datasets:

| task | THUNDER paper 12-ds mean | our base | Δ | Waiv base | Waiv − paper |
|---|---|---|---|---|---|
| knn | 70.14 | **70.28** | +0.14 | 74.0 | **+3.86** |
| linear_probing | 76.46 | **76.54** | +0.08 | 79.3 | **+2.84** |
| segmentation (4-ds) | 67.42 | — (we run 2) | — | 66.5 | −0.92 |
| segmentation (2-ds, ocelot+pannuke) | 69.85 | **70.40** | +0.55 | — | — |

Per-dataset agreement with the THUNDER paper is tight on 9 of 12 (|Δ| ≤ 1.0); the three
outliers are bach +4.05, break_his +5.15, wilds −4.43 and largely cancel.

So on the one backbone with an independent third-party reference, **our base reproduces the
benchmark's own published baseline to 0.14 points, and Waiv's sits ~3 points above both.**
The same sign and rough magnitude holds on the other two backbones (knn gaps −1.75
midnight, −2.03 virchow2; linear −1.52, −1.55).

What it is **not**:
- Not pooling. Pooling is per-backbone and internal to feature extraction; our phikon `cls`
  features reproduce the THUNDER paper *per dataset*, which pins the protocol.
- Not a dataset subset. Both are the same 12 `PAPER_CLS` datasets. (Segmentation *is* a
  subset mismatch — ours 2, theirs 4 — flagged `support_2v4`; that is a separate defect.)
- Not a checkpoint difference. The same untuned weights reproduce Waiv's RI and HEST bases
  to 4 decimals. Whatever differs is THUNDER-side only.

What it might be (**INFERRED, not established**): a different metric or adaptation setting.
Waiv §3.3 says only "frozen features following the default protocol" and never names the
metric; we report F1 / binary Dice, inherited from THUNDER's defaults. Note the direction is
*not* uniform — Waiv's classification sits above the paper while their segmentation sits
below it — which argues against a simple "stronger harness" story and is more consistent
with a per-task-family metric difference. There is no per-dataset appendix in Waiv to check
against. **Unexplained is the honest label.**

---

## 3. Does the two-base fix rescue THUNDER? No.

`_pct_of_waiv_two_base` (`scoreboard.py:290-316`) computes
`(ours − OUR_base) / (Waiv_ft − Waiv_base)`.

**The assumption.** That a gain of X points from *our* base is worth the same as a gain of X
points from *their* base — i.e. that the base gap is an additive measurement offset that
cancels, and that the benchmark's difficulty is locally uniform across the ~2–4 point band
separating the two bases.

**Where it is most defensible.** On THUNDER classification, the offset story has real
support: we reproduce the benchmark's own numbers, RI and HEST prove the weights are
identical, and the offset is same-signed and similar-sized across three backbones and three
tasks. If it truly is a harness offset, then their FT number carries the same offset and
gain-vs-gain is the *only* legal comparison. Under that reading our phikon knn 73.77
"corrected" onto their scale would be ~77.5 against their 77.7 — near parity.

**Where it is least defensible.** In the ten cells where `|gap|/gain > 1`. There, the
formula divides our measured gain by a denominator drawn from a scale we have positively
shown we are not on, and the offset is larger than the thing being divided. Concretely:
midnight linear reads **289%** (capped to 100) off a Waiv gain of **0.2 points** — smaller
than that cell's own seed floor (2SE = 0.87 points, `docs/thunder_seed_floor_12ds.md`).
Virchow2 linear reads −8.2% off a Waiv gain of 0.3 points against a 0.88-point floor. These
are not measurements of anything.

**Optimistic or pessimistic for us?** Both readings are live and they point opposite ways,
which is itself the problem:

- *Offset reading* → two-base is roughly fair, maybe slightly pessimistic (a lower absolute
  level leaves more headroom on a capped metric, but if the level is fictitious so is the
  headroom).
- *Headroom reading* (their base is genuinely better) → two-base **flatters us**. Our base
  is lower on all nine classification cells, so we are climbing the easy part of the curve
  while their gain is fought at 74–85 points where ceiling compression bites. On phikon knn
  we gain 3.49 from 70.28; they gain 3.70 from 74.0. Those are not equally hard.

**The acid test — absolute FT scores.** This is where the two-base percentages fall apart:

- THUNDER classification: **we are below Waiv's fine-tuned absolute on all nine cells**,
  by 0.31 to 3.93 points. Yet three of those cells print pct ≥ 100 (phikon linear 121%,
  midnight linear 289%, midnight simple_shot 115%). A percentage that says "we captured
  100%+ of their improvement" while our model scores lower than theirs on the same
  benchmark is being generated by the base gap, not by the recipe.
- THUNDER segmentation: we are *above* on all three, by +0.30 to +4.23 points — but that is
  a 2-dataset mean against their 4-dataset mean and means nothing.
- RI: phikon we **exceed** (+0.0209); midnight −0.0247, virchow2 −0.0203.
- HEST: below on all three (−0.0072 / −0.0102 / −0.0080), consistent with pcts of
  63% / 53% / 23%.

RI and HEST percentages are *directionally consistent* with the absolutes. THUNDER's are not.

---

## 4. Verdicts

| benchmark | verdict | reason |
|---|---|---|
| **RI (PathoROB)** | **VALID** | Base gap ≤ 0.0004, ≤ 0.33% of the Waiv gain on every backbone; reproduced at per-dataset granularity on all 9 cells including three Virchow2 values Waiv never published. Single-base formula is correct here. Percentages 106% / 85% / 66% stand. |
| **HEST** | **VALID-WITH-CAVEAT** | Base gap ≤ 0.00016 (≤ 1.6% of gain). Two caveats: (i) the agreement is protocol-contingent and the pooling was chosen to produce it — report it as "protocol matched by base-reproduction", not as independent validation; (ii) virchow2's gain (0.0101) is only ~10× the 5-seed SE, so 22.9% carries roughly ±10 pct points of seed noise. Also fix the 0.40324 → 0.40327 constant. Percentages 63% / 53% / 23% may be reported with the pooling protocol stated per backbone. |
| **THUNDER** | **INVALID as a percentage. Do not report per-task `pct_of_waiv`.** | Our base reproduces the *benchmark's own* published baseline to 0.14 pts while Waiv's sits ~3 pts above it — unexplained, and larger than the effect in 10 of 12 cells (up to 14.6×). The two-base fix converts a level mismatch into a ratio but cannot repair it: three cells print ≥ 100% while our absolute is *below* Waiv's on the same task, and the two available readings of the offset (harness offset vs real headroom) push the answer in opposite directions with no evidence to choose between them. Compounding this, most cells are below their own seed floor anyway (midnight linear: Waiv gain 0.20 vs 2SE 0.87; virchow2 linear: 0.30 vs 0.88). Segmentation is additionally a 2-vs-4 dataset support mismatch and three of four segmentation/knn cells are already guarded for `waiv_regressed`. |

### What to report instead for THUNDER

Raw, per (backbone, task): **our_base → our_ft (Δ ± 2SE)** alongside
**waiv_base → waiv_ft (Δ)**, with the explicit note that our base reproduces
arXiv:2507.07860v3 and Waiv's does not, so the two Δ columns are on different scales and
neither a level nor a ratio comparison is licensed. Any aggregate that averages a THUNDER
`pct_of_waiv` into `avg_pct_of_waiv` inherits this invalidity and should be recomputed over
RI and HEST only, with the THUNDER omission stated rather than silent.
