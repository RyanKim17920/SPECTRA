# Is the pass/fail criterion gradeable? Error bars on the AGGREGATE, not the cell

Generated 2026-08-25 by `scripts/aggregate_criterion_resolvability.py`. Sources on disk, no new jobs:

- THUNDER per-seed / per-dataset F1: `docs/thunder_seed_floor_12ds.json` (`per_seed_scores`),
  produced by `scripts/thunder_seed_floor_12ds.py`.
- THUNDER **our** base (12-dataset mean, `frozen`, F1): read this session from
  `/data/ryan.kim/thunder/outputs/res/<ds>/<base_dir>/<task>/frozen/outputs.json`,
  base dirs per `scripts/collect_final5.py::THUNDER_BASE_DIRS`
  (phikon→`base_cls`, midnight→`mbase_clsmean`, virchow2→`vbase_clsmean`). All 12/12 present.
- HEST: `/data/ryan.kim/hest_work/results/f5_final5-<bb>-s{0..4}-t900-*_s0000500_<pool>_summary.json`,
  key `results.avg` (9-task mean). Pooling cls / cls / clsmean per `collect_final5.py::HEST_BASE`.
- RI: `runs/final5-<bb>-s{0..4}-t900-*/ri_curve.json`, `points[step==500].avg_robustness_index`.
- Waiv targets: `scripts/scoreboard.py::WAIV`, `::WAIV_THUNDER`.

Replicates: the **final5** family, 3 backbones × 5 training seeds, checkpoint step 500,
15 independent SLURM jobs, config-verified identical except seed/backbone
(provenance in `docs/thunder_seed_floor_12ds.md`). This is the only place on disk with
≥3 seed replicates on all three benchmarks simultaneously.

---

## 1. The estimator, stated exactly

Per cell, following `scripts/scoreboard.py`:

```
RI       pct(b)    = (ours_RI(b)   - RI_BASE[b])   / (WAIV_RI[b]   - RI_BASE[b])   * 100
HEST     pct(b)    = (ours_HEST(b) - HEST_BASE[b]) / (WAIV_HEST[b] - HEST_BASE[b]) * 100
THUNDER  pct(b,t)  = (ours_F1(b,t) - OUR_BASE(b,t)) / (WAIV_ft(b,t) - WAIV_base(b,t)) * 100
```

THUNDER uses the **two-base gain ratio** because our measured THUNDER base does not
reproduce Waiv's (documented at `scripts/scoreboard.py:~120`); RI and HEST use the
single-base form because their bases agree to 4 decimals. `ours_F1(b,t)` is the mean over
the 12 `PAPER_CLS` datasets. Every cell is **capped at 100**. Segmentation is excluded
throughout (2-vs-4 dataset support mismatch).

Aggregates:

```
pct_RI      = mean over 3 backbones                        (3 cells)
pct_HEST    = mean over 3 backbones                        (3 cells)
pct_THUNDER = mean over 3 backbones x 3 quick tasks         (9 cells, flat)
MEAN3       = mean(pct_RI, pct_HEST, pct_THUNDER)
```

The flat 9-cell mean and the nested (task-mean-per-backbone, then backbone-mean) mean are
**numerically identical** here because every backbone has all 3 tasks; both are reported
below and agree to 0.01 except under cell exclusion.

### The three degenerate THUNDER cells

| cell | Waiv gain (F1) | our 12ds pairdiff-2SD | ratio | status |
|---|---|---|---|---|
| virchow2 / knn | **−0.003** | 0.0060 | 1.99 | denominator NEGATIVE |
| midnight / linear_probing | +0.002 | 0.0060 | 3.01 | denominator < our noise |
| virchow2 / linear_probing | +0.003 | 0.0067 | 2.24 | denominator < our noise |

**virchow2/knn must be dropped unconditionally**, not as a judgement call: Waiv *regressed*
on this cell, and `scripts/scoreboard.py` already guards negative denominators to N/A with
reason `waiv_regressed`. Including it means "matching a regression scores 100%".

**The other two are permanently unresolvable**, and more seeds on our side cannot fix them:

1. The denominator is a **single published point estimate with no error bar**. Waiv report
   +0.002 and +0.003 F1. Our own seed noise on the identical quantity is 0.006–0.007 F1
   (2σ on a single-run difference). Whatever their run-to-run noise is, it is not plausibly
   ≥3× smaller than ours on the same benchmark — so the *numerator target itself* is
   consistent with zero. Dividing by it produces a ratio whose uncertainty is dominated by a
   quantity we cannot measure and they did not report.
2. Even driving our numerator noise to zero leaves `pct = (small ± small) / (0.002 ± ?)`,
   which is unbounded. This is a property of the ratio, not of our sample size.

Empirically this is not academic. Per-seed uncapped pct for these cells:

```
midnight / linear_probing   411, 377, 147, 248, 261     (SD 106 pct points)
virchow2 / knn              474, 553, 442, 572, 611     (SD  70 pct points)
virchow2 / linear_probing    69, -40,  81, -100, -52     (SD  79 pct points)
```

A single training seed moves virchow2/linear_probing from **+81% to −100% of Waiv**.

The **6 surviving cells** — phikon/{knn, lp, simple_shot}, midnight/{knn, simple_shot},
virchow2/simple_shot — are exactly the 6 that `docs/thunder_seed_floor_12ds.md` independently
graded resolvable at cell level (ratio < 1). The two analyses agree without being tuned to.

Aggregates are reported **both ways** below.

---

## 2. Empirical propagation of seed noise to the aggregate

Two estimators, neither assuming independence within a backbone:

**(A) Index-matched.** Build the whole aggregate from seed `s` on all three backbones,
for `s = 0..4`, then take SD across the 5 aggregates. Assumes nothing about correlation
structure; df = 4.

**(B) Variance-decomposed.** Write the aggregate as `sum_b c_b(s)` where `c_b(s)` is
backbone `b`'s weighted contribution. Because the 15 runs are separate jobs with distinct
seeds and distinct backbones, the `c_b` are independent *across backbones*; within a
backbone the tasks share one run and their measured correlation is used implicitly
(the variance of the whole task-sum is estimated directly, never task-by-task).
`Var(agg) = Σ_b Var(c_b)`, each term df = 4, Satterthwaite effective df reported.
No `1/sqrt(9)` is ever assumed.

Measured within-backbone cross-task correlation of the raw 12ds task mean (why `sqrt(9)`
would have been wrong): r spans **−0.03 to +0.91**, mean ≈ +0.43
(virchow2 knn~lp r = +0.91; midnight knn~lp r = +0.03).
Cross-benchmark within a backbone: THUNDER~RI r = −0.55 / +0.97 / +0.81,
THUNDER~HEST r ≈ 0, HEST~RI r = +0.66 / −0.03 / −0.66. Sign-unstable at n = 5 — do not read
these as real couplings, read them as "not safely zero".

### Aggregate 2SE at n = 1 seed per (recipe, backbone), in pct-of-waiv POINTS

Estimator (B), capped cells. 95% CI is on the 2SE itself (chi-square, Satterthwaite df).

| aggregate | cells | SD (n=1) | **2SE (n=1)** | 95% CI on 2SE | eff. df | variance concentration |
|---|---|---|---|---|---|---|
| **RI** | 3 | 1.18 | **2.35** | [1.5, 5.6] | 5.2 | virchow2 87% |
| **HEST** | 3 | 5.54 | **11.08** | [7.9, 18.7] | 11.2 | balanced (26/29/46%) |
| **THUNDER, 9 cells** | 9 | 9.79 | **19.58** | [12.3, 47.2] | 5.1 | virchow2 88% |
| **THUNDER, 6 cells (degen excluded)** | 6 | 5.32 | **10.63** | [7.0, 22.1] | 6.7 | phikon 74% |
| MEAN3, THUNDER 9-cell | — | 3.88 | 7.76 | [5.1, 15.8] | 7.0 | virchow2 73% |
| MEAN3, THUNDER 6-cell | — | 2.76 | **5.51** | [3.9, 9.6] | 10.1 | phikon 53% |

Cross-check with estimator (A), which assumes nothing (df = 4):
RI 2SD = 2.62, HEST 9.29, THUNDER 9-cell 20.39, THUNDER 6-cell 12.28, MEAN3 6-cell 6.31.
Both estimators agree within their (wide) CIs on every line. Uncapped variants:
RI 2SE 2.52, HEST 11.08 (no cell hits the cap), THUNDER 6-cell 14.54.

### final5's own point estimates (mean of its 5 seeds, capped)

| aggregate | value | 2SE at n=5 |
|---|---|---|
| RI | **83.7** | ±1.05 |
| HEST | **46.1** | ±4.96 |
| THUNDER (6 cells) | **66.2** | ±4.75 |
| THUNDER (9 cells) | 65.4 | ±8.75 |
| MEAN3 (6-cell THUNDER) | **65.3** | ±2.47 |

Per-backbone, capped: HEST 63.2 / 52.5 / 22.5; RI 100.0 (phikon, capped from 106.2) / 85.0 / 66.0.

---

## 3. Verdict on gradeability at n = 1

Two different questions, two different bars. For a **threshold test** ("is pct ≥ 70?") the
relevant quantity is 2SE. For an **arm-vs-arm comparison** the relevant quantity is the
minimum detectable difference, MDD = 2·√2·SD = √2 · 2SE.

| aggregate | 2SE (n=1) | MDD (n=1) | resolve 70 vs 100 (30 pts)? | resolve 70 vs 80 (10 pts)? | resolve 60 vs 70? |
|---|---|---|---|---|---|
| RI | 2.35 | 3.3 | **YES**, 9× margin | **YES**, 3× margin | **YES** |
| HEST | 11.08 | 15.7 | yes, ~2× margin | **NO** | **NO** |
| THUNDER 6-cell | 10.63 | 15.0 | yes, ~2× margin | **NO** | **NO** |
| THUNDER 9-cell | 19.58 | 27.7 | **NO** (27.7 ≈ 30) | **NO** | **NO** |
| MEAN3 (6-cell) | 5.51 | 7.8 | YES | marginal (7.8 vs 10) | marginal |

Plainly:

- **RI is fully gradeable from one seed.** Its aggregate 2SE is 2.4 points against a
  20-point decision band. Nothing here is noise-limited.
- **HEST and THUNDER are NOT gradeable to the 70-vs-80 resolution from one seed.**
  Their aggregate 2SE (≈11 points) is *larger than the entire 20-point band* the criterion
  needs to discriminate. A recipe whose true pct_HEST is 70 will be measured anywhere in
  59–81; the 70-floor verdict for it is a coin flip.
- They **are** gradeable to the coarser "did we get most of the gain or almost none"
  resolution: 70 vs 100, and 0 vs 70, both clear ~2×.
- **Aggregation genuinely helped, and it is measured, not assumed.** Per-cell, the best
  THUNDER cell sat at ratio 0.24 and three cells were unresolvable at any margin. The
  6-cell aggregate 2SE of 10.6 points is roughly *half* what naive per-cell reasoning
  suggested — but it stops well short of the 20-point band. The previous conclusion
  ("no THUNDER cell is resolvable to 20%") survives aggregation; the criterion is coarse-
  gradeable, not fine-gradeable.
- **Keeping the 3 degenerate cells nearly doubles the THUNDER error bar** (10.6 → 19.6)
  and pushes it past even the 70-vs-100 test. Excluding them is not cosmetic; it is the
  difference between a usable and an unusable instrument.

**MEAN3 > 80 bar.** final5's MEAN3 is 65.3 ± 2.5 (n=5) — decisively below 80.
At n=1 the MEAN3 2SE is 5.5, so the >80 bar is decidable for anything outside ~77–83.

**Does this change the verdict for final5?** No, and noise cannot rescue it:
pct_HEST = 46.1 with 2SE = 5.0 at n=5, i.e. 41–51, is **decisively below the 70 floor**
by 4 error bars. final5 fails the criterion on HEST regardless of how the THUNDER
ambiguity is resolved. THUNDER 6-cell at 66.2 ± 4.8 also fails the 70 floor, marginally
(upper bound 71.0 grazes it). Only RI (83.7) clears 70.

---

## 4. Seeds required

`2SE(k) = 2·SD₁/√k`, k = seeds per (recipe, backbone). One training run yields RI, HEST and
THUNDER, so seeds are **shared** across benchmarks; cost is `3k` runs total (3 backbones).

| aggregate | SD₁ | k for 2SE < 10 pts | k for 2SE < 5 pts |
|---|---|---|---|
| RI | 1.18 | **1** | **1** |
| HEST | 5.54 | **2** | **5** |
| THUNDER 6-cell | 5.32 | **2** | **5** |
| THUNDER 9-cell | 9.79 | 4 | 16 |
| MEAN3 (6-cell) | 2.76 | 1 | 2 |

Binding requirement across all three benchmarks (degenerate cells excluded):

| target | seeds/backbone | training runs | GPU-hours @ 2 GPU × 7–10 h | wall-clock if 3 runs parallel |
|---|---|---|---|---|
| every aggregate 2SE < **10 pts** | **k = 2** | **6** | 84–120 | ~2 waves, 14–20 h |
| every aggregate 2SE < **5 pts** | **k = 5** | **15** | 210–300 | ~5 waves, 35–50 h |

If the 3 degenerate cells are retained, 2SE < 5 on THUNDER needs **k = 16, i.e. 48 runs**
(672–960 GPU-h). That alone justifies excluding them.

Note that **k = 5 is exactly what final5 already is** — final5 is a fully graded recipe at
the 5-point resolution. What is missing is a *second* recipe measured at the same depth to
compare it against. Any candidate promoted on a single seed is being compared to final5
across a ±10-point fog on HEST and THUNDER.

**Even at k = 5 the 70-vs-80 discrimination is not achieved for HEST/THUNDER**
(2SE ≈ 5, band = 10, so ~2σ — borderline, one-sided). Reaching a clean 2SE < 2.5 would take
k = 20 (60 runs) per recipe. Given that, the 70/80 thresholds should be treated as
**decision heuristics, not measurements**, on HEST and THUNDER — a recipe reading 68 and one
reading 82 are not distinguishable at any realistic budget.

---

## 5. Honesty / confidence

Read these before quoting any number above.

1. **Every SD rests on n = 5 seeds per backbone (df = 4).** Pooling three backbones gives
   Satterthwaite effective df of only 5.1–11.2. The 95% CIs in §2 are wide by a factor of
   2–3 on the upper side; the THUNDER 9-cell 2SE could truly be anywhere in [12, 47].
   No SD here is based on n < 4, but none is based on n ≥ 20 either.
2. **Two lines are effectively single-backbone estimates.** RI (virchow2 = 87% of variance)
   and THUNDER 9-cell (virchow2 = 88%) have eff. df ≈ 5, i.e. barely better than df = 4.
   Treat their error bars as low-confidence. HEST (26/29/46%) and MEAN3 6-cell (53/28/19%)
   are genuinely pooled and are the most trustworthy lines in the table.
3. **The 100-cap suppresses variance, and does so recipe-specifically.** phikon RI is
   uncapped 105–108 on all 5 seeds → capped to exactly 100 → contributes **zero** variance.
   That is why RI's capped SD is so small. A recipe sitting *near* 100 rather than above it
   would have a strictly larger RI aggregate 2SE. The uncapped RI 2SE (2.52) is the safe
   number to carry forward. The same artifact pins midnight/lp and virchow2/knn at 100 in
   the capped 9-cell THUNDER aggregate, hiding SDs of 106 and 70 pct points respectively.
   **All error bars in this document are measured on final5 and are not guaranteed to
   transfer to a recipe at a different operating point.**
4. **Estimator (B) assumes independence across backbones only.** Justified — 15 separate
   SLURM jobs, different seeds and different encoders. It is checked against estimator (A),
   which assumes nothing; the two agree on every line.
5. **The denominators carry unreported error.** Every pct is divided by a Waiv published
   gain quoted to 3–4 significant figures with no error bar. This document propagates *our*
   noise only. For the 6 surviving THUNDER cells the denominators (0.014–0.037 F1) are 2–6×
   our per-cell noise so this is a second-order worry; for HEST (denominators 0.0103–0.0215)
   and for the 3 excluded cells it is not.
6. **HEST's virchow2 cell is the fragile one.** Waiv's virchow2 HEST gain is +0.0103 against
   our per-seed HEST SD of 0.00115 — a ratio of only 9, and it carries 46% of the HEST
   aggregate variance. This matches the existing note
   `waiv-pct-of-waiv-amplifies-noise-on-virchow2-hest`.
7. **Not addressed here:** segmentation (support mismatch), checkpoints other than step 500,
   and whether the seed-noise magnitude is stable across training steps
   (`waiv-seed-floor-is-not-a-constant` says it varies up to 4× by step — so these bars are
   step-500 bars only).

---

## Bottom line

| benchmark | aggregate 2SE @ n=1 (pct pts) | 70-floor / 80-mean gradeable at n=1? | seeds for 2SE < 10 | runs |
|---|---|---|---|---|
| **RI** | **2.4** | **YES** — fully | 1 | 3 |
| **HEST** | **11.1** | **NO** at 70-vs-80; yes at 70-vs-100 | 2 | 6 |
| **THUNDER** (6 cells, degenerates dropped) | **10.6** | **NO** at 70-vs-80; yes at 70-vs-100 | 2 | 6 |
| THUNDER (all 9 cells) | 19.6 | **NO** — not even 70-vs-100 | 4 | 12 |
| MEAN3 > 80 bar (6-cell THUNDER) | 5.5 | yes outside ~77–83 | 1 | 3 |

Aggregation over backbones and tasks does shrink the error bar substantially — roughly 2×
versus per-cell reasoning — and it converts THUNDER from "no cell resolvable" to
"aggregate resolvable at 30-point resolution". It does **not** make the 70-vs-80 distinction
measurable on HEST or THUNDER at any single-seed budget, and only k = 5 seeds per backbone
(15 runs, ~210–300 GPU-h) brings those to 5 points — which is still only ~2σ against a
10-point band.

Practical rule: **grade RI from one seed; never grade HEST or THUNDER from one seed.**
Minimum credible comparison of a candidate against final5 is **k = 2 (6 runs)**, and the
reported number must be the multi-seed mean with its 2SE attached, never a single run.
