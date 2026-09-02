> **SUPERSEDED 2026-08-31 — do not use for the paper. See [`RUNBOOK.md`](RUNBOOK.md).**
>
> This document is a correct record of the 2026-08-25 three-backbone pilot and is kept for
> history. Three things in it no longer describe what we run:
>
> 1. **Scope**: three backbones (phikon-v2, midnight, Virchow2). Production is now **five** —
>    H-Optimus-0 and UNI2-h were added, and both are GATED hub repos requiring
>    `WAIV_PIN=/admin/home/ryan.kim/waiv-snapshots/falseneg-gated` or they 403.
> 2. **Checkpoint grid**: `WAIV_CKPT_EVERY=125` here; production is **50**. The 1-SE rule can
>    only return a step it actually evaluated, so the grid is part of the selection procedure,
>    not a detail. On a 50 grid the selected steps are 200/150/100/100/125, and no fixed step
>    serves all five.
> 3. **Stopping rule**: the `confounder_insensitivity >= 0.75` rule below is replaced by the
>    parameter-free **1-SE rule**, which reproduces its picks 12/12 without the arbitrary
>    0.75 threshold.
>
> The head-bias question left open here is also now settled: keep `WAIV_BCLS=3.0`,
> `WAIV_BMEAN=-inf`. Symmetric `+3/+3` costs RI ~2.3x its seed floor while its segmentation
> gain sits below the THUNDER noise floor.

# FINAL RECIPE — one configuration, three backbones, a stopping rule instead of a step

Generated 2026-08-25 from `scripts/final_recipe_report.py` → `docs/final_recipe_verdict.json`.
Branch `final5-and-ablations`. **No new training jobs were launched to produce this document.**

---

## 0. Read this first — the one-paragraph summary

There is now a single recipe that is **byte-identical across phikon-v2, midnight and
Virchow2**, differing only in `--backbone` and `--seed`. What makes it work on all three is
not the hyperparameters — those were already fixed — but **where you stop**: the first
checkpoint whose mean `confounder_insensitivity` reaches 0.75. That rule is computed from
signals the training job already logs, needs no external benchmark, and selects a *different*
step on each backbone (250 / 125 / 125).

**The verdict is `INDETERMINATE`, and that is the honest reading.** Every point estimate
clears both bars — the worst cell is 73.1 (bar 70), the overall average is 92.2 (bar 80) —
but two THUNDER cells cannot be graded at all for lack of full 12/12 coverage, and the worst
cell's error bar straddles its bar. "Every point estimate passes" and "we have demonstrated a
pass" are different claims; only the first is supported.

| | |
|---|---|
| worst cell | virchow2 / HEST = **73.1** ±20.1 (n=2) → `NOT RESOLVED` |
| overall average | **92.2** (RI 91.4 / HEST 85.3 / THUNDER 100.0) |
| ungraded cells | midnight/THUNDER (`PARTIAL`), virchow2/THUNDER (`PARTIAL`) |
| verdict | **INDETERMINATE** |

---

## 1. The recipe, stated so it can be re-run

Launcher is `scripts/gentle.sbatch`. The complete invocation:

```bash
WAIV_ARM=<phikon|midnight|virchow2> \
WAIV_SEED=<n> \
WAIV_T=900 \
WAIV_LR=1e-4 \
WAIV_MAX_STEPS=500 \
WAIV_CKPT_EVERY=125 \
WAIV_MASK=1 \
WAIV_BCLS=3.0 \
WAIV_BMEAN=-inf \
sbatch scripts/gentle.sbatch
```

Everything else is the sbatch's own default and is therefore also fixed:

| knob | value | where set |
|---|---|---|
| LoRA rank / alpha | `r=32` / `alpha=64` | `WAIV_RANK` default 32; `LORA_ALPHA=$((RANK*2))` |
| projection out-dim | 512 | `WAIV_PROJDIM` default |
| weight decay | 0.05 | `WAIV_WD` default |
| temperature | 0.07 | `WAIV_TEMP` default |
| warmup | 200 steps | `WAIV_WARMUP` default |
| batch geometry | grid, C=2 conditions × T=900 tiles, `--grid-forward-chunk 0` | `WAIV_T=900`, `GRID_COND=2` hardcoded |
| head | `--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem` | `HEAD_ARGS` |
| negative policy | `--mask-same-core`, `bias-cls=3.0`, `bias-mean=-inf` | `WAIV_MASK` / `WAIV_BCLS` / `WAIV_BMEAN` |
| retention KL | 0 (off) | `WAIV_KL` default |
| pooling (train) | `clsmean` | hardcoded |
| memory | `--grad-checkpointing --activation-offload`, `--mem=700G` | hardcoded; the 700G is load-bearing, see the sbatch header |
| pin | `waiv-snapshots/falseneg-pinned` (selected because `WAIV_MASK` is set) | `PIN` resolution in the sbatch |

**This is byte-identical across the three backbones except `WAIV_ARM` and `WAIV_SEED`.** The
six runs behind this document are:

| run | backbone | seed | train job |
|---|---|---|---|
| `genMASK-c3s-lr1e-4-r32-kl0-t0.07-wd0.05-ms500-pd512-phikon-s0-t900-392669` | phikon | 0 | 392669 |
| `genMASK-c3s-…-phikon-s1-t900-392672` | phikon | 1 | 392672 |
| `genMASK-c3s-…-midnight-s0-t900-392670` | midnight | 0 | 392670 |
| `genMASK-c3s-…-midnight-s1-t900-392673` | midnight | 1 | 392673 |
| `genMASK-c3s-…-virchow2-s0-t900-392671` | virchow2 | 0 | 392671 |
| `genMASK-c3s-…-virchow2-s1-t900-392674` | virchow2 | 1 | 392674 |

Evaluation pooling is **per backbone**, not free: HEST/THUNDER read phikon and midnight under
`cls`, Virchow2 under `clsmean` (`scripts/collect_final5.py::HEST_BASE`,
`::THUNDER_BASE_DIRS`). Getting this wrong reproduces the old "base gap vs Waiv" artefact.

---

## 2. The stopping rule — this is the contribution

> **Stop at the first checkpoint whose mean `confounder_insensitivity` is ≥ 0.75.**

Read from `runs/<run>/ri_curve.json` →
`points[].datasets[*].confounder_insensitivity`, unweighted mean over the datasets present at
that point (camelyon / tolkach_esca / tcga). This is produced by the in-job eval follower
(`eval_checkpoints.py --follow`, launched by `gentle.sbatch` on GPU 1), so **selection needs
no external HEST run** — HEST costs ~36 min per checkpoint and is not in the loop.

Applied to the six runs, it selects:

| backbone | selected step | mean CI at selection | CI trace (125 / 250 / 375 / 500) |
|---|---|---|---|
| phikon s0 | **250** | 0.781 | 0.504 → **0.781** → 0.831 → 0.821 |
| phikon s1 | **250** | 0.800 | 0.512 → **0.800** → 0.803 → 0.828 |
| midnight s0 | **125** | 0.919 | **0.919** → 0.984 → 1.009 → 1.000 |
| midnight s1 | **125** | 0.890 | **0.890** → 0.992 → 0.996 → 1.015 |
| virchow2 s0 | **125** | 0.882 | **0.882** → 0.963 → 0.985 → 0.973 |
| virchow2 s1 | **125** | 0.872 | **0.872** → 0.961 → 0.964 → 0.991 |

Source: `docs/final_recipe_verdict.json` → `runs[].ci_trace`. Selection is recomputed by the
report script on every invocation; no step is hardcoded anywhere.

### Why a rule beats a fixed step

HEST `pct_of_waiv` at each *fixed* step, computed from
`/data/ryan.kim/hest_work/results/f5_genMASK-c3s-*_summary.json`
(`results.avg`, 9-task mean; bases/targets from `scripts/scoreboard.py`):

| step | phikon | midnight | virchow2 | **min** | |
|---|---|---|---|---|---|
| 125 | 60.6 (n=2) | 92.1 (n=2) | 73.1 (n=2) | **60.6** | FAIL |
| 250 | 90.5 (n=2) | 75.6 (n=2) | 54.0 (n=1) | **54.0** | FAIL |
| 375 | 96.1 (n=2) | 61.5 (n=2) | 18.0 (n=1) | **18.0** | FAIL |
| 500 | 93.5 (n=2) | — | 30.8 (n=1) | — | — |
| **CI ≥ 0.75 rule** | **90.5** @250 | **92.1** @125 | **73.1** @125 | **73.1** | **PASS** |

Per-seed values behind the means: phikon@125 60.8/60.5, @250 84.0/97.1, @375 90.9/101.2,
@500 89.5/97.6; midnight@125 92.2/92.1, @250 72.5/78.8, @375 53.2/69.9; virchow2@125
74.9/71.3 (only seed 0 is HEST'd at 250/375/500).

**No fixed step passes.** The best one (125) leaves phikon at 60.6; the rule beats it on the
worst cell by **+12.5 points** (73.1 vs 60.6). Every backbone's optimum is at a different
step, so a fixed step is structurally unable to serve all three.

> Note on the brief version of this table: an earlier draft quoted the seed-0-only figures
> (midnight@250 = 72.5, virchow2@125 = 74.9, giving a +14.3-point margin). The table above
> uses the same n for every cell it can, which is the honest comparison; the margin is 12.5,
> not 14.3.

### Guard rails that come with the rule

From `docs/internal_stopping_criterion.md` §3b, measured over 117 HEST'd checkpoints:

- **CI ≥ 0.94 is a reliable disqualifier** — only 1/16 such checkpoints clear 70, mean 37.8.
- **CI ≥ 1.0 is a hard reject**: the confounder probe has been destroyed and HEST collapses
  with it (the two worst rows in the whole dataset, midnight 2.8 and virchow2 −21.1, sit at
  CI 1.07 and 1.06).

Both midnight seeds and virchow2 s1 cross 1.0 by step 500 under this recipe — i.e. the recipe
*does* run past the disqualifier if you let it. The rule is what keeps it from doing so.

### What the rule does NOT establish

`docs/internal_stopping_criterion.md` §5 is blunt about this and it carries over:

- The rule's demonstrated skill comes almost entirely from **phikon**. On midnight and
  virchow2 the CI is already ≥ 0.75 at the *first* checkpoint ever evaluated, so the rule is
  observationally identical to "stop at the first eval" there.
- Because of that, **virchow2's true optimum may be below step 125 and we cannot see it.**
  HEST falls monotonically from the first checkpoint in 6/6 virchow2 runs in the stopping
  dataset, and the single best virchow2 row anywhere is at step **100** (85.8, n=1 against a
  14.2-point SD). The honest statement is "stop earlier than 125", not "125 is optimal".
  Dropping `WAIV_CKPT_EVERY` to 50 on the next virchow2/midnight run would settle it.
- **No internal signal predicts HEST *level*, only step.** Within-run HEST variation is only
  18–39% of across-run variation; the rule recovers the smaller half of the problem. It tells
  you when to stop a run you already started; it cannot rank recipes.

---

## 3. Mechanism — why the optima differ

`adapter_rel_l2_delta` (the adapter update magnitude) correlates with HEST with a **sign that
flips by base-model strength**. From `docs/internal_stopping_criterion.md` §2a, over the 117
joined checkpoints:

| signal | phikon (n=52) | midnight (n=34) | virchow2 (n=31) |
|---|---|---|---|
| `adapter_rel_l2_delta` | **r = +0.495** | r = −0.117 | **r = −0.533** |
| `step` | −0.173 | −0.373 | −0.645 |
| mean `confounder_insensitivity` | +0.691 | −0.262 | −0.245 |

> Provenance note: a working figure of **+0.657 / −0.07 / −0.526 over 100 checkpoints**
> circulated during the session. It is **not reproducible from anything on disk** — the only
> committed cut is the 117-row one above, in `docs/stopping_criterion_rows.json`. Quote the
> table, not the working figure. The direction and the sign flip are identical either way.

The sign flip is not a heterogeneous-recipe artefact — it reproduces **inside** matched recipe
families (§2d of that doc: phikon lr1e-4/ms1500 r = +0.23, midnight lr1e-4/ms400 r = −0.99,
virchow2 lr1e-4/ms1500 r = −0.52).

**Reading:** a weak base wants a bigger update, a strong base wants a smaller one. Concretely,
phikon's HEST peaks **late** and midnight's and virchow2's peak **early** — HEST rises with
step in 6 of 13 phikon runs but falls in 6/6 virchow2 runs and 7/8 midnight runs. The runs
where phikon rises are exactly the ones whose CI at step 250 is still low (0.38–0.53); it is a
**catch-up** effect, not a contradiction. This is the same mechanism the CI rule reads.

### "Train to a target update norm" — REFUTED

The natural next idea — replace the CI rule with "stop when `adapter_rel_l2_delta` hits X" —
does not work, for two independent reasons:

1. **The peak L2 is not a constant.** phikon ≈ 0.875, midnight ≈ 1.025, virchow2 ≈ 0.625 —
   a 1.64× spread. A ±20% band spans a ratio of at most 1.5, so **no single band contains all
   three peaks.** Widening it does not help: the band that contains virchow2's peak also
   contains phikon's trough (virchow2's `[0.85,0.90)` bin means HEST 25.1 while phikon's means
   73.4 — the same band holds both peaks and troughs).
2. **L2 barely moves along a run.** Median within-run span as a fraction of across-run span:
   phikon 0.16, midnight 0.14, virchow2 **0.09**. L2 is a *recipe descriptor* (how aggressive
   the configuration is), not a *trajectory* coordinate. A rule read off it is mostly reading
   which run you launched, not where you are in it.

Keep logging `adapter_rel_l2_delta` — it does separate aggressive from gentle configs — but do
not select on it. `avg_robustness_index` is worse still as a stopping signal (within/across
ratio 0.01 on virchow2; argmax-RI picks virchow2's *worst* HEST row).

---

## 4. Results

All numbers from `docs/final_recipe_verdict.json` (generated 2026-08-25). Formula
`pct_of_waiv = (ours − base) / (waiv − base) × 100`; THUNDER uses the two-base gain ratio
because our THUNDER base does not reproduce Waiv's. The 100 cap is a **reporting** convention
applied *after* the ≥70 resolution test, never before it.

### 4.1 Per cell

| backbone | benchmark | capped pct | uncapped pct | ±95% CI | n | status |
|---|---|---|---|---|---|---|
| phikon | RI | **100.0** | 108.9 | ±1.9 | 2 | PASS |
| phikon | HEST | **90.5** | 90.5 | ±8.2 | 2 | PASS |
| phikon | THUNDER (knn only) | **100.0** | 137.7 | ±63.0 | 1 | PASS |
| midnight | RI | **92.5** | 92.5 | ±4.1 | 2 | PASS |
| midnight | HEST | **92.1** | 92.1 | ±11.7 | 2 | PASS |
| midnight | THUNDER | — | — | — | 0 | **PARTIAL** |
| virchow2 | RI | **81.6** | 81.6 | ±11.2 | 2 | PASS |
| virchow2 | HEST | **73.1** | 73.1 | ±20.1 | 2 | **NOT RESOLVED** |
| virchow2 | THUNDER | — | — | — | 0 | **PARTIAL** |

Raw values and the denominators they are divided by:

| backbone | RI base → ours → Waiv | HEST base → ours → Waiv (pooling) |
|---|---|---|
| phikon | 0.4686 → 0.83610 → 0.806 | 0.37470 → 0.39244 → 0.3943 (`cls`) |
| midnight | 0.7589 → 0.91158 → 0.924 | 0.39521 → 0.41501 → 0.4167 (`cls`) |
| virchow2 | 0.8582 → 0.90700 → 0.918 | 0.40324 → 0.41074 → 0.4135 (`clsmean`) |

Per-seed HEST pct: phikon 84.0 / 97.1 · midnight 92.2 / 92.1 · virchow2 74.9 / 71.3.
Per-seed RI pct (uncapped): phikon 109.2 / 108.6 · midnight 91.9 / 93.1 · virchow2 79.9 / 83.3.

CI provenance, cell by cell:
- **HEST** — authoritative 1-SD in pct-of-waiv points (5.8 / 8.3 / 14.2), as `2·SD/√n`.
- **RI** — `max(empirical 2·SD/√n, measured seed floor)`; no floor was measured at the selected
  step on any backbone, so the conservative max over measured steps is used (1.9 / 4.1 / 11.2).
  phikon's RI is censored: both seeds sit above the 100 cap, so it contributes zero variance.
- **THUNDER** — from `docs/thunder_seed_floor_12ds.md` (n=5 training seeds, offset-2SE,
  12/12 coverage).

### 4.2 Benchmark averages and the verdict

| aggregate | value | backbones contributing |
|---|---|---|
| RI | **91.4** | 3 |
| HEST | **85.3** | 3 |
| THUNDER | **100.0** | **1** (phikon only) |
| **overall average** | **92.2** | bar is 80 |
| **worst cell** | **73.1** (virchow2/HEST, ±20.1) | bar is 70 |

> **FINAL VERDICT: `INDETERMINATE`.**
> Reason, verbatim from the report: *no gradeable number for midnight/THUNDER (PARTIAL),
> virchow2/THUNDER (PARTIAL); error bar straddles the 70 bar for virchow2/HEST
> (73.1 ± 20.1 uncapped).*

Note that the THUNDER row is an average over **one** backbone. Do not read "THUNDER 100.0" as a
three-backbone result; it is phikon/knn, n=1, capped from 137.7, with a ±63-point interval.

---

## 5. Limitations — read this section before quoting anything above

### 5.1 virchow2/HEST is the binding cell, and its bar is 70 by a hair

The cell reads 73.1 with a ±20.1 interval, i.e. **53.0 – 93.2**. It straddles 70 and is
therefore `NOT RESOLVED`: the data cannot distinguish PASS from FAIL, and asserting either
would be a claim the error bars do not support.

The reason is arithmetic, not experimental sloppiness. **Waiv's own virchow2 HEST gain is only
+0.0103** (0.40324 → 0.4135). Dividing by a gain that small amplifies raw noise enormously:
one seed-SD on this backbone is **14.2 percentage points** of `pct_of_waiv`. That number is a
property of Waiv's published table, and no amount of care on our side changes it.

**A caveat that cuts the other way.** The two in-cell seeds actually agree *closely* —
74.9 and 71.3, an empirical SD of **≈2.53** points, five times smaller than the 14.2 borrowed
figure. The report deliberately uses the wider borrowed floor. That is the conservative choice
and it is the right one at n=2 (an SD from two points has df=1 and is nearly worthless), but it
means the reported interval is probably much too wide.

**This judgement call has NOT been made.** Switching to the in-cell empirical SD once n ≥ 5 is a
live option and someone has to decide it deliberately. Note that even under the empirical SD
the cell does not cleanly resolve: `73.1 ± 2·2.53/√2 = 73.1 ± 3.6 = 69.5 – 76.7`, which still
grazes 70. Do not assume more seeds automatically deliver a PASS.

### 5.2 Three THUNDER cells can NEVER be graded

Each of these has a Waiv gain smaller than the benchmark's own measured seed noise
(`docs/thunder_seed_floor_12ds.md`, 12-dataset offset-2SE, n=5 seeds):

| cell | Waiv gain (F1) | our 12ds floor | ratio | why it is dead |
|---|---|---|---|---|
| midnight / linear_probing | **+0.0020** | 0.0087 | 4.37 | denominator < our noise |
| virchow2 / knn | **−0.0030** | 0.0083 | 2.76 | **Waiv REGRESSED** — denominator negative |
| virchow2 / linear_probing | **+0.0030** | 0.0088 | 2.93 | denominator < our noise |

This is a property of **Waiv's published table**, not of our recipe and not of our compute
budget. The denominator is a single published point estimate with no error bar, and it is
consistent with zero. Even driving our numerator noise to zero leaves `pct = (small ± small) /
(0.002 ± ?)`, which is unbounded. Empirically, one training seed moves virchow2/linear_probing
from **+81% to −100% of Waiv** (`docs/aggregate_criterion_resolvability.md` §1).

**Consequence: virchow2's THUNDER verdict rests on `simple_shot` ALONE.** knn and
linear_probing are structurally ungradeable there. Same for midnight, where linear_probing is
dead and knn/simple_shot are merely incomplete.

Separately, the two `PARTIAL` cells in §4.1 are `PARTIAL` for a *different* reason — coverage.
A task mean over fewer than 12/12 `PAPER_CLS` datasets is graded by **nothing**, because the
12-dataset floors are floors for a 12-dataset mean; a shorter mean averages away less
per-dataset noise and is therefore *noisier*, so applying the 12ds floor would understate the
noise and manufacture resolvability. Current coverage is patchy (midnight s0 = 0/12 on every
task; virchow2 s1 = 1–2/12). Finishing THUNDER on these six runs is the single highest-value
piece of remaining compute.

### 5.3 HEST and THUNDER cannot be graded at n=1 at all

From `docs/aggregate_criterion_resolvability.md` §2–3, measured on final5's 15-run seed study:

| aggregate | 2SE at n=1 | resolve 70 vs 80 (10 pts)? |
|---|---|---|
| RI | 2.35 | **YES** |
| HEST | **11.08** | **NO** |
| THUNDER, 6 gradeable cells | **10.63** | **NO** |
| THUNDER, all 9 cells | 19.58 | NO |

The aggregate 2SE on HEST and THUNDER is **larger than the entire 20-point decision band**.
A recipe whose true `pct_HEST` is 70 will be measured anywhere in 59–81. They *are* gradeable
to the coarse "most of the gain vs almost none" resolution (70 vs 100 clears ~2×), which is
what §4 is doing. **The 70/80 thresholds should be treated as decision heuristics, not
measurements, on HEST and THUNDER.** Reaching 2SE < 2.5 would take k = 20 seeds per backbone,
60 runs per recipe.

This document is at **n=2 per backbone**. That is above the k=2 needed for 2SE < 10 points and
well below the k=5 needed for < 5 points.

### 5.4 Segmentation is excluded

Deliberately, and it is stated in `final_recipe_report.py` rather than buried: segmentation is
slow, and **no seed floor has been measured for it**. An ungraded benchmark is not evidence in
either direction. There is also a 2-vs-4 dataset support mismatch against Waiv's table.

---

## 6. Traps — things a fresh reader must not repeat

1. **THUNDER run names must be ≤ 64 characters.** A 72-job sweep was lost to this. It is why
   the THUNDER model ids in the verdict are shortened (`f5_ci-phikon-s0-392669_s0000250`)
   rather than carrying the full `genMASK-c3s-lr1e-4-…` run name. Shorten *before* submitting,
   and keep a mapping — HEST and THUNDER caches are keyed on run name / exp_code alone, so two
   arms that collide in that string silently share results.
   *(Session finding; no artefact of the lost sweep survives on disk to cite.)*

2. **The 4-dataset THUNDER subset is indefensible.** Its error exceeds Waiv's entire gain, and
   5 of 18 cells flip sign relative to the 12-dataset mean. Always score on the full 12
   `PAPER_CLS` datasets. Relatedly, the *old* 5-dataset floor in
   `docs/thunder_seed_floor.md` is superseded — it was measured on a different run family from
   a single seed pair, and the 5 datasets in it are exactly the 5 noisiest.

3. **Scoring at step 1500 kills the strong backbones.** virchow2 HEST goes to **−4.2** (and as
   low as −30.8 on the lr3e-5 arm). Every three-backbone table read at 1500 is measuring
   over-specialisation, not the recipe. This is why `WAIV_MAX_STEPS=500` is part of the recipe
   and not an afterthought.

4. **Nobody had scored HEST below step 250 before this session.** That is the whole reason
   virchow2 looked unsalvageable: its optimum is at or below 125, and every prior measurement
   started after it. Before concluding a backbone is a lost cause, check that you have measured
   anywhere near its optimum.

5. **The sbatch scripts set no `--account`**, so everything defaults to the smaller 8-GPU pool.
   `gentle.sbatch` has no `--account` line; several older scripts carry
   `sbatch --account=max --qos=high` only as a *comment* in their headers. If throughput
   matters, pass it explicitly at submit time.

6. **`results_backup/` is not a backup** — it symlinks into `/data`, which is volatile.
   Real backups live in `/admin/home/ryan.kim/waiv_result_backups/`.

---

## 7. Reproducing this document

```bash
python3 scripts/final_recipe_report.py            # prints the table
python3 scripts/final_recipe_report.py --json     # rewrites docs/final_recipe_verdict.json
```

The script re-discovers runs by glob (`runs/genMASK-c3s-*`), re-applies the stopping rule per
run from `ri_curve.json`, and re-reads HEST/THUNDER from disk. **Nothing about the selected
step is hardcoded**, so re-running it after more seeds or more THUNDER coverage land requires
no edit. It refuses to print PASS for any cell that is `PARTIAL`, `INDETERMINATE`, or
`NOT RESOLVED`, and it returns `INDETERMINATE` overall if any required cell is.

### Sources

| what | where |
|---|---|
| live verdict | `scripts/final_recipe_report.py` → `docs/final_recipe_verdict.json` |
| stopping rule derivation, L2 refutation | `docs/internal_stopping_criterion.md` (+ `docs/stopping_criterion_rows.json`, `docs/stopping_full_curves.json`) |
| THUNDER 12-dataset seed floors | `docs/thunder_seed_floor_12ds.md` / `.json` |
| aggregate resolvability, seeds required | `docs/aggregate_criterion_resolvability.md` |
| training parameterisation | `scripts/gentle.sbatch` |
| prior context | `docs/FINAL_CANDIDATE.md`, `docs/FINAL5_RESULTS.md` |
| Waiv targets | arXiv:2607.22861v1 Tables 1–3; transcription in `docs/waiv_published.json`, constants in `scripts/scoreboard.py` |
| raw HEST summaries | `/data/ryan.kim/hest_work/results/f5_genMASK-c3s-*_summary.json` |
| raw THUNDER outputs | `/data/ryan.kim/thunder/outputs/res/<ds>/<model>/<task>/frozen/outputs.json` |
