# Model-agnostic stopping criterion for the LoRA recipe

**Question.** Is there a rule computable from signals we already log during training that
selects the HEST-optimal checkpoint, without running HEST (~36 min/checkpoint)?

**Answer, short.**

1. The `adapter_rel_l2_delta` hypothesis is **refuted**. Peak L2 is not consistent across
   backbones, L2 barely moves within a run, and the sign of its relationship with HEST flips
   by backbone *and within recipe family*.
2. The best rule found is **stop at the first checkpoint where mean
   `confounder_insensitivity` >= 0.75**. It achieves near-zero regret against a per-run
   HEST oracle on all three backbones (0.1 / 0.3 / 0.0 pct pts). But its evidence base is
   asymmetric: on midnight and virchow2 CI is already above 0.75 at the *first* checkpoint we
   ever evaluate, so there it is indistinguishable from "stop as early as possible". Only
   phikon actually exercises the rule.
3. **No internal signal predicts the HEST *level*.** Every candidate ranks steps within a run
   at best; none ranks recipes. Within-run HEST variation is only 18-39% of across-run
   variation, so the rule recovers the smaller half of the problem.
4. The **>= 70 pct_of_waiv on every backbone** target is **not reachable by any stopping rule**
   on this run set. The cheating per-run oracle tops out at virchow2 mean 49.3; only 3 of 31
   virchow2 checkpoints reach 70 at all. virchow2's shortfall is a *recipe* problem, not a
   stopping problem.

---

## 1. Dataset

Built by `scripts/build_stopping_dataset.py` -> `docs/stopping_criterion_rows.json`.
Joins `runs/<run>/ri_curve.json` `points[]` with
`/data/ryan.kim/hest_work/results/f5_<run>_s<7digit>_<pooling>_summary.json`
(pooling: cls for phikon/midnight, clsmean for virchow2 — per the pooling-protocol note).

| backbone | checkpoint rows | distinct runs | runs with >= 2 HEST'd ckpts |
|---|---|---|---|
| phikon   | 52 | 32 | 13 |
| midnight | 34 | 23 |  8 |
| virchow2 | 31 | 24 |  6 |
| **total**| **117** | **79** | **27** |

All 121 `f5_*` HEST summaries on disk matched a run directory; 117 also had a matching
`ri_curve.json` point. Steps covered: 100, 125, 200, 250 (45), 300, 400, 500 (46), 750,
1000, 1250, 1500 (11), 2000, 2500, 3000.

pct_of_waiv uses base 0.37470 / 0.39521 / 0.40324 and Waiv 0.3943 / 0.4167 / 0.4135
(phikon / midnight / virchow2). RI pct uses base 0.4686 / 0.7589 / 0.8582 and Waiv
0.806 / 0.924 / 0.918.

**Noise floor.** HEST is n=1 per checkpoint; 1 SD = 5.8 / 8.3 / 14.2 pct-of-waiv points.
Everything below must be read against that.

---

## 2. The L2 hypothesis is refuted

### 2a. Correlation sign flips by backbone (confirms the prior result)

| signal | phikon (n=52) | midnight (n=34) | virchow2 (n=31) |
|---|---|---|---|
| `adapter_rel_l2_delta` (mean over datasets) | r=+0.495 | r=-0.117 | r=-0.533 |
| `step` | r=-0.173 | r=-0.373 | r=-0.645 |
| mean `confounder_insensitivity` | r=+0.691 | r=-0.262 | r=-0.245 |
| mean `prediction_performance` | r=-0.558 | r=+0.038 | r=+0.201 |
| `avg_robustness_index` | r=+0.363 | r=-0.271 | r=+0.067 |
| train `loss` | r=+0.415 | r=+0.019 | r=-0.204 |
| train `top1` | r=-0.437 | r=-0.251 | r=+0.077 |

I reduced the per-dataset `adapter_rel_l2_delta` dict to a scalar by **unweighted mean over
the three datasets** (camelyon / tolkach_esca / tcga). Justification: the three values track
each other tightly within a checkpoint (e.g. 0.741 / 0.935 / 0.816) and the quantity is a
property of the adapter, not of the dataset — the per-dataset spread is measurement noise from
the probe batches, not signal. Using the max or the tcga value alone changes no conclusion.

### 2b. Where does HEST peak in L2?

Binned mean HEST pct_of_waiv (bins with n >= 3):

| L2 bin | phikon | midnight | virchow2 |
|---|---|---|---|
| [0.60,0.65) | — | — | n=4 **62.7** |
| [0.70,0.75) | — | — | n=5 33.1 |
| [0.75,0.80) | — | n=3 56.7 | n=4 56.2 |
| [0.85,0.90) | n=21 **73.4** | n=5 57.1 | n=10 25.1 |
| [0.90,0.95) | n=18 66.7 | — | n=5 30.8 |
| [1.00,1.05) | — | n=3 **90.2** | — |
| [1.05,1.10) | — | n=3 77.6 | — |
| [1.10,1.15) | — | n=9 50.8 | — |
| [1.15,1.25) | — | n=8 50.6 | — |

**Peak L2: phikon ~0.875, midnight ~1.025, virchow2 ~0.625.** Ratio max/min = 1.64. A +/-20%
band spans a ratio of at most 1.5, so **no single +/-20% L2 band captures the peak on all
three backbones.**

Caveat in the honest direction: if you instead take the L2 of the *single best row* per
backbone (0.885 / 1.025 / 0.779), those three do fit inside a +/-20% band centred at 0.90
(0.72-1.08). But that band is **non-discriminative** — it also contains virchow2's [0.85,0.90)
bin whose mean HEST is 25.1 and phikon's whose mean is 73.4. A band that contains both the
peaks and the troughs is not a rule.

### 2c. L2 saturates within a run, so it cannot be a stopping signal at all

Median within-run span of each signal vs the full across-run span, per backbone:

| backbone | L2 within / across | CI within / across | RI within / across | HEST within / across |
|---|---|---|---|---|
| phikon   | 0.059 / 0.376 = **0.16** | 0.062 / 0.560 = 0.11 | 0.009 / 0.131 = 0.07 | 12.0 / 67.3 = 0.18 |
| midnight | 0.063 / 0.434 = **0.14** | 0.046 / 0.427 = 0.11 | 0.005 / 0.045 = 0.10 | 28.9 / 92.5 = 0.31 |
| virchow2 | 0.036 / 0.383 = **0.09** | 0.017 / 0.785 = 0.02 | 0.003 / 0.344 = **0.01** | 45.8 / 116.6 = 0.39 |

L2 moves 9-16% as much along a run's step axis as it does across recipes. It is a **recipe
descriptor** (how aggressive is this configuration), not a **trajectory descriptor**. Example:
`final5-phikon-s0` goes 0.831 -> 0.929 over steps 250 -> 1500 while HEST goes 57.0 -> 69.0 ->
67.6 -> 59.0 -> 59.3 -> 60.5.

### 2d. Within recipe family the L2 relationship still flips sign

| backbone | family (n>=4) | n | r(L2, HEST) | r(CI, HEST) | r(step, HEST) |
|---|---|---|---|---|---|
| phikon   | lr1e-4 ms1500        | 15 | +0.20 | +0.32 | -0.08 |
| phikon   | lr1e-4 ms1500 mask   | 23 | +0.23 | +0.60 | -0.50 |
| phikon   | lr1e-4 ms3000 mask   |  4 | +0.16 | +0.20 | +0.45 |
| midnight | lr1e-4 ms1500        |  8 | -0.69 | -0.16 | -0.57 |
| midnight | lr1e-4 ms400         |  4 | -0.99 | -0.87 | -0.90 |
| midnight | lr3e-5 ms1500        |  5 | -0.97 | -0.93 | -0.97 |
| virchow2 | lr1e-4 ms1500        |  8 | -0.52 | -0.05 | -0.68 |

The flip is not a heterogeneous-recipe artefact. It reproduces **inside** matched families.
**L2 hypothesis: refuted.**

---

## 3. Alternative internal signals

### 3a. The underlying shape of the data (this is the real finding)

Reading the 27 multi-checkpoint runs directly:

- **virchow2: HEST falls with step in 6/6 runs.** 60.8->11.0, 32.3->29.1, 38.1->25.2,
  57.3->-4.2, 50.8->7.2->-30.8, 56.6->14.8. 4 of 6 drops exceed the 14.2 SD.
- **midnight: HEST falls in 7/8 runs.** The exception (`final5-s0`, 52.0->54.6) is +2.6,
  well inside the 8.3 SD.
- **phikon: mixed.** HEST *rises* in 6 runs and falls/flat in 7. The runs that rise are
  exactly the ones whose CI at step 250 is low (0.38-0.53); the runs that peak at step 250
  are the ones already at CI 0.78-0.80 there.

So the apparent "phikon's HEST rises, the others fall" contradiction dissolves: it is a
**catch-up** effect. phikon's step-250 checkpoint has not yet done enough confounder removal;
midnight and virchow2 are already past the optimum by step 250.

### 3b. Confounder-insensitivity saturation — the strongest candidate

Pooled binned HEST pct_of_waiv by mean CI, and the fraction of checkpoints clearing 70:

| CI band | n | mean HEST% | >= 70 | phikon | midnight | virchow2 |
|---|---|---|---|---|---|---|
| [0.00,0.70) | 16 | 52.6 | 3/16 | 0/11 | 2/2 | 1/3 |
| [0.70,0.78) | 13 | 57.2 | 0/13 | 0/8 | 0/3 | 0/2 |
| **[0.78,0.90)** | **51** | **64.5** | **25/51** | 19/31 | 4/10 | 2/10 |
| [0.90,0.94) | 21 | 45.1 | 3/21 | 1/2 | 2/8 | 0/11 |
| [0.94,inf)  | 16 | 37.8 | 1/16 | 0/0 | 1/11 | 0/5 |

Two usable facts:

- **CI >= 0.94 is a reliable disqualifier**: 1/16 checkpoints clear 70, mean 37.8. The two
  worst rows in the whole dataset (midnight 2.8 and virchow2 -21.1, both the t=0.15 arm) sit
  at CI 1.07 and 1.06 — CI above 1.0 means the confounder probe has been destroyed, and HEST
  collapses with it. This is model-agnostic and holds on both backbones that reach it.
- **CI in [0.78,0.90) is the best band**, but it is *weakly* predictive: only 25/51, and it is
  carried by phikon (19/31) while virchow2 hits 2/10 inside the same band. As a *screen* on
  recipes it does not work.

### 3c. RI is useless as a stopping signal

`avg_robustness_index` moves 1-10% as much within a run as across runs — on virchow2 the ratio
is **0.01** (within-run span 0.003 vs across-run 0.344). It is flat along the step axis, and it
actively misleads: virchow2's *worst* HEST row (-30.8, `genMASK-lr3e-5` step 1500) has the
*highest* virchow2 RI (0.9224, RI pct 107.3). An argmax-RI rule scores virchow2 mean 35.7
(regret 13.6). RI-flatness (`first step where dRI < 0.002`) is better (42.0, regret 7.3) but
still loses to CI.

### 3d. Training metrics are not landmarks

`top1` saturation is uninformative: within the same run family, `top1` at the HEST peak ranges
0.571 to 0.999 (midnight) and 0.619 to 0.996 (phikon). Two seeds of the same config
(`ph-c3.0m-inf` s1/s2, step 250) have top1 0.619 and 0.961 with HEST 96.6 and 91.6. Loss and
heldout_loss correlate |r| <= 0.42 and flip sign by backbone. `eval_seconds` and `probe.*` add
nothing.

---

## 4. Ranked rules

Evaluated on the 27 runs with >= 2 HEST'd checkpoints. Rules are applied to the run's
checkpoint sequence; H = mean HEST pct_of_waiv at the selected step; reg = mean regret vs the
per-run HEST oracle; RI = mean RI pct_of_waiv at the selected step.

| # | rule | phikon H / reg | midnight H / reg | virchow2 H / reg | worst-BB H | min RI% |
|---|---|---|---|---|---|---|
| — | **ORACLE** (uses HEST, upper bound) | 71.3 / 0.0 | 72.4 / 0.0 | 49.3 / 0.0 | 49.3 | 78.4 |
| **1** | **first ckpt with mean CI >= 0.75** | **71.2 / 0.1** | **72.1 / 0.3** | **49.3 / 0.0** | **49.3** | **78.4** |
| 1b | same, plus a CI < 0.93 ceiling | 71.2 / 0.1 | 72.1 / 0.3 | 49.3 / 0.0 | 49.3 | 78.4 |
| 2 | first ckpt with mean CI >= 0.78 | 70.8 / 0.4 | 70.8 / 1.6 | 42.0 / 7.3 | 42.0 | 79.4 |
| 3 | first ckpt where dRI < 0.002 (RI-flat) | 70.7 / 0.6 | 71.2 / 1.3 | 42.0 / 7.3 | 42.0 | 79.4 |
| 4 | PP-drop guard (stop when mean PP first falls) | 63.4 / 7.9 | 72.1 / 0.3 | 47.2 / 2.1 | 47.2 | 78.3 |
| 5 | stop at earliest checkpoint (step floor) | 62.9 / 8.4 | 72.1 / 0.3 | 49.3 / 0.0 | 49.3 | 78.4 |
| 6 | first ckpt with L2 >= 0.7 | 65.2 / 6.0 | 72.1 / 0.3 | 42.0 / 7.3 | 42.0 | 79.4 |
| 7 | argmax `avg_robustness_index` | 70.6 / 0.7 | 70.7 / 1.7 | 35.7 / 13.6 | 35.7 | 79.5 |
| 8 | argmax CI x PP | 69.6 / 1.7 | 48.7 / 23.7 | 10.2 / 39.1 | 10.2 | 74.9 |
| 9 | argmax CI | 69.6 / 1.7 | 48.7 / 23.7 | 10.2 / 39.1 | 10.2 | 74.9 |
| 10 | L2-flat (first dL2 < 0.02) | 69.2 / 2.1 | 47.2 / 25.2 | 9.6 / 39.7 | 9.6 | 73.3 |
| 11 | stop at last checkpoint | 68.6 / 2.7 | 45.9 / 26.6 | 7.5 / 41.8 | 7.5 | 73.2 |

**Recommended rule (#1), stated precisely:**

> At every eval checkpoint, compute `ci = mean_over_datasets(points[i].datasets[*].confounder_insensitivity)`.
> Stop at the **first** checkpoint with `ci >= 0.75`. If a run reaches `max_steps` without
> crossing, take the final checkpoint. Additionally treat `ci >= 0.94` at *any* checkpoint as a
> recipe failure signal (that configuration is too aggressive), and `ci >= 1.0` as a hard reject.

Steps it selects: phikon 250 or 500 (or the end, for the low-CI recipes); midnight 100-250;
virchow2 250. Its RI at the selected step is 109.8 / 90.5 / 78.4 pct_of_waiv — **the worst
backbone clears the RI >= 70 requirement**. Its HEST at the selected step is 71.2 / 72.1 /
**49.3** — the worst backbone **fails** the HEST >= 70 requirement.

---

## 5. Honesty section — what this rule does not establish

**(a) The rule is only actually tested on phikon.** On all 8 midnight runs and all 6 virchow2
runs, CI is already >= 0.75 at the earliest checkpoint we have (virchow2 step 250: CI
0.77-0.92; midnight step 250: CI 0.76-0.96; the two ms400 runs at step 100: CI 0.83 and 0.89).
The rule therefore fires immediately and is **observationally identical to "stop at your first
eval"** on those two backbones (row 1 and row 5 of the table are the same numbers there). All
of the rule's demonstrated skill — the +8.3 point gain over "stop earliest" — comes from
phikon. Do not report it as validated on three backbones.

**(b) virchow2's true optimum is outside the measured range.** HEST falls monotonically from
the first checkpoint in 6/6 virchow2 runs, and the single best virchow2 row in the entire
dataset is step **100** (85.8). We have exactly one virchow2 checkpoint below step 250. The
rule sits on the boundary of the data on this backbone; the honest statement is "stop earlier
than 250 and we do not know how much earlier".

**(c) On 4 of 13 phikon runs the rule never fires** (`combined`, `falseneg`, `falseneg3k`,
`genMASK-lr3e-5` — CI tops out at 0.60-0.72) and falls through to "train to the end". It
happens to be right on 3 of those 4 because those runs are still climbing at the end. That is
a fallback branch, not a crossing, and it is 4 data points.

**(d) Several "peaks" are unresolved against noise.** With 1 SD = 5.8 / 8.3 / 14.2:
`final5-midnight-s0` (52.0 vs 54.6) and `ph-c3.0m-inf-s0` (83.7 vs 82.0) and
`ph-c3.0m-inf-s2` (91.6 vs 88.9) have no resolved peak at all. Rule #1 wins on those runs by
picking either member of a tie. Only the large within-run drops (virchow2 60.8->11.0,
midnight 95.3->37.4, phikon 40.2->67.2) are resolved.

**(e) LR-schedule confound is real and I did not remove it.** 27 of 117 rows are *annealed*
(step == max_steps, so the cosine schedule has completed); 90 are unannealed mid-run readings.
Annealed rows score lower on average on phikon (57.0 vs 66.4) and midnight (54.1 vs 59.1),
roughly equal on virchow2 (36.3 vs 34.0). The step-250 reading of an ms1500 run is **not** the
same object as an annealed ms250 run, and both appear in the table above. The `annealed` flag
is carried in `docs/stopping_criterion_rows.json` for anyone who wants to re-cut this.

**(f) No signal predicts the HEST level, only the step.** Within-run HEST span is 18-39% of
across-run span. Two midnight checkpoints at essentially identical internals (step 250, CI
~0.92, L2 ~1.10-1.13) score HEST 52.0 and 95.3. The rule cannot tell you which *recipe* to
run; it can only tell you when to stop the one you started.

**(g) The >= 70-on-every-backbone bar is not reachable here.** The cheating oracle gives
virchow2 mean 49.3 across its 6 multi-checkpoint runs, and only 3/31 virchow2 checkpoints in
the whole dataset reach 70 (85.8 at step 100 of the gen ms400 run; 85.8 at the annealed
genMASK t=0.03 ms250 run; 74.7 at ret0.01 step 250). Those three share no internal signature —
their CI values are 0.83, 0.63, 0.88. No stopping rule closes this gap; a different virchow2
recipe is required.

---

## 6. Recommendation

1. **Adopt rule #1 as the default stopping criterion** (`first checkpoint with mean CI >= 0.75`),
   with the `CI >= 0.94` recipe-failure guard. It is the only rule with near-zero oracle regret
   on all three backbones, it costs nothing (CI is already in `ri_curve.json`), and it fails
   gracefully — its worst behaviour is "stop at the first eval", which is itself the second-best
   rule.
2. **Change the eval cadence before trusting it further.** The rule's crossing point is at or
   before step 250 on midnight and virchow2, i.e. inside the first eval interval. Set
   `eval_every` / `ckpt_every` to 50 for the next midnight and virchow2 runs so the crossing is
   actually observed rather than assumed.
3. **Spend the next HEST budget on virchow2 steps 25-200**, not on more step-250/500 arms. The
   single 85.8 at step 100 is the only evidence we have about where virchow2's optimum lives,
   and it is n=1 against a 14.2-point SD.
4. **Do not use `adapter_rel_l2_delta` for stopping.** Keep logging it as a recipe descriptor
   (it does separate aggressive from gentle configs), but it saturates within a run and its
   HEST relationship flips sign by backbone even within a matched recipe family.
5. **Do not use RI for stopping.** It is flat along the step axis (virchow2 within/across ratio
   0.01) and anti-correlates with HEST at the extremes.

## Files

- `scripts/build_stopping_dataset.py` -> `docs/stopping_criterion_rows.json` (117 joined rows)
- `scripts/analyze_stopping.py` — correlations, per-backbone leaderboards
- `scripts/analyze_stopping2.py` — within-run trajectories, dynamic-range table
- `scripts/analyze_stopping3.py` — L2/CI binning, annealing split, per-family correlations
- `scripts/eval_stopping_rules.py` — rule evaluation harness (table in section 4)
- `scripts/full_grid_rules.py` -> `docs/stopping_full_curves.json` — full checkpoint grid,
  CI crossing points per run
