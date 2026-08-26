# FINAL CANDIDATE — cross-backbone generalization

Generated 2026-08-24 from `scripts/scoreboard.py` v3 (steps 250 and 500, 143 runs discovered).
Branch `final5-and-ablations`. **No new training jobs were launched to produce this document.**

---

## 0. Read this first — what this document is and is not

**This document selects a single recipe that generalizes across all three backbones.**
The previous edition picked the best single checkpoint *per backbone*. The criterion has changed:
a recipe must score **>=80% of Waiv's average gain on ALL THREE backbones** (phikon-v2,
midnight, Virchow2). A recipe is scored by its **worst** backbone. Per-backbone arm-picking is
no longer a valid result.

**The 80% bar applies to the AVERAGE across readouts within each backbone** — the user's
instruction was "80% of Waiv's averaged increase (so things like 100% of RI and 90% of others
is good enough)". In practice this means: for each backbone, compute
`avg_pct = mean(RI_pct_of_waiv, HEST_pct_of_waiv, [THUNDER_pct_of_waiv if available])` at that
checkpoint; then take the minimum across backbones.

**Warning: under a stricter per-metric floor reading (each individual metric must be >=80%),
NO candidate currently passes.** The per-metric reading would demand, e.g., HEST >=80% on every
backbone, which no recipe achieves. The per-backbone-average reading described above is the
operative criterion.

**All recipes are scored at a single fixed step: step 250.** This avoids the LR-schedule
confound (an ms1500 run read at step 500 is further annealed than the same run at step 250).
Step 250 is the only step with measurements on all three backbones for the leading recipes.

**Recipe scores use SEED MEANS (with sd and n), not chosen seeds.** For final5-plain with n=5
seeds, the mean is used. For recipes with n=1, the single value is reported with an explicit
`n=1` label. Where seeds were trained at different `max_steps` (the ph2 family: s0 is ms1500
while s1/s2 are ms500), those seeds are NOT replicates and are reported separately rather than
averaged.

**Seed noise is material to the 80% bar.** One seed-SD corresponds to:

| backbone | RI (1 SD in pct_of_waiv) | HEST (1 SD in pct_of_waiv) |
|---|---|---|
| phikon | 1.2% | n=1 (HEST only s0 available) |
| midnight | 2.9% | 10.6% |
| Virchow2 | 7.9% | 14.9% |

On Virchow2 HEST, one seed-SD is ~15 percentage points of `pct_of_waiv`. The gap between the
leading recipe (74.9%) and the 80% bar is less than one seed-SD. **No recipe is statistically
distinguishable from the 80% bar at n=1.** Seed replication is required before any recipe can
be declared a pass or fail.

**Scoring formula:** `pct_of_waiv = (ours - base) / (Waiv - base) * 100`.
THUNDER uses a **two-base ratio** `(ours - our_base) / (Waiv_ft - Waiv_base)` because
Waiv's THUNDER absolutes are on a different scale; see the denominators table below.

| backbone | base RI | Waiv RI | RI gain | base HEST | Waiv HEST | HEST gain |
|---|---|---|---|---|---|---|
| phikon-v2 | 0.46860 | 0.806 | +0.33740 | 0.37470 | 0.3943 | +0.01960 |
| Midnight | 0.75890 | 0.924 | +0.16510 | 0.39521 | 0.4167 | +0.02149 |
| Virchow2 | 0.85820 | 0.918 | +0.05980 | 0.40324 | 0.4135 | +0.01026 |

Waiv targets: Filiot, Thaeter, Schmauch, Guillou, *Robustifying pathology foundation models via
fine-tuning*, arXiv:2607.22861v1, Tables 1-3; verified transcription in `docs/waiv_published.json`.

---

## 1. Recipe-level table (ranked by MIN across backbones)

All recipes scored at **step 250** (the only step with measurements on all three backbones for
the leading recipes). The success criterion is that the MIN column reaches >=80%.
**No recipe currently passes.** `ret0.01` leads at 74.9% (limited by Virchow2), with
`genMASK-lr3e-5` close behind at 74.0%.

Values are **seed means** with sd and n in parentheses. Where n=1, the value is the single
observed value. For ph2, s1/s2 (ms500) are used as replicates; s0 (ms1500) is excluded from
the mean because it has a different LR schedule.

| recipe | phikon | midnight | Virchow2 | MIN |
|---|---|---|---|---|
| **ret0.01** | 105.4% (RI only; HEST@250 pending, job 392044; n=1) | 75.6% (n=1) | 74.9% (n=1) | **74.9** |
| genMASK-lr3e-5 | — (no phikon run) | 80.0% (n=1) | 74.0% (n=1) | 74.0 |
| ph2 | — (no phikon run) | 75.2% (s1/s2, n=2, sd=0.9%) | 72.9% (s1/s2, n=2, sd=1.0%) | 72.9 |
| kl0.003 (ms250) | — (no phikon run) | 82.2% (n=1) | 58.0% (n=1) | 58.0 |
| final5-plain (ms1500) | 80.7% (RI n=5 sd=1.2%, HEST n=1) | 75.0% (RI n=5 sd=2.9%, HEST n=3 sd=10.6%) | 54.8% (RI n=5 sd=7.9%, HEST n=3 sd=14.9%) | 54.8 |

**Distinguishing configs per recipe** (relative to the base final5 recipe):

| recipe | config delta |
|---|---|
| ret0.01 | `retention_kl_weight=0.01`, ms1500, lr1e-4, T0.07, r32/a64, pd512, clsmean, GeM, split cls+mean .5/.5, grid 900x2, seed 0 |
| genMASK-lr3e-5 | `mask_same_core=true`, `lr=3e-5`, ms250, T0.07, r32/a64, pd512 |
| ph2 | `mask_same_core=true`, `same_core_logit_bias_cls=-inf`, `same_core_logit_bias_mean=-inf` (per-head masking) |
| kl0.003 | `retention_kl_weight=0.003`, ms250, lr1e-4, T0.07, r32/a64, pd512 |
| final5-plain | base recipe: lr1e-4, ms1500, T0.07, r32/a64, pd512, clsmean, GeM, grid 900x2 |

**Per-backbone detail (step 250, seed means):**

| recipe | backbone | RI mean (pct) | RI sd (n) | HEST mean (pct) | HEST sd (n) | avg% |
|---|---|---|---|---|---|---|
| ret0.01 | phikon | 0.82416 (105.4%) | n=1 | — (pending 392044) | — | 105.4 (RI only) |
| ret0.01 | midnight | 0.90451 (88.2%) | n=1 | 0.40877 (63.1%) | n=1 | 75.6 |
| ret0.01 | virchow2 | 0.90322 (75.3%) | n=1 | 0.41090 (74.6%) | n=1 | 74.9 |
| genMASK-lr3e-5 | midnight | 0.91331 (93.5%) | n=1 | 0.40951 (66.6%) | n=1 | 80.0 |
| genMASK-lr3e-5 | virchow2 | 0.91764 (99.4%) | n=1 | 0.40823 (48.6%) | n=1 | 74.0 |
| ph2 | midnight | 0.92319 (99.5%) | 0.00107 (n=2) | 0.40616 (51.0%) | n=1 | 75.3 |
| ph2 | virchow2 | 0.92218 (107.0%) | 0.00119 (n=2) | 0.40722 (38.8%) | n=1 | 72.9 |
| kl0.003 | midnight | 0.90554 (88.8%) | n=1 | 0.41147 (75.7%) | n=1 | 82.2 |
| kl0.003 | virchow2 | 0.89755 (65.8%) | n=1 | 0.40839 (50.2%) | n=1 | 58.0 |
| final5-plain | phikon | 0.82119 (104.5%) | 0.00408 (n=5) | 0.38586 (56.9%) | n=1 | 80.7 |
| final5-plain | midnight | 0.90063 (85.8%) | 0.00482 (n=5) | 0.40900 (64.2%) | 0.00227 (n=3) | 75.0 |
| final5-plain | virchow2 | 0.89762 (65.9%) | 0.00475 (n=5) | 0.40773 (43.8%) | 0.00153 (n=3) | 54.8 |

**finalgem is the only other three-backbone recipe and is unusable:** dominated by ret0.01 on RI
at every comparable step, and zero HEST on any backbone. Jobs 384585/384586/384587.

**ph2 s0 (ms1500) values, reported separately (NOT seed replicates of s1/s2):**

| backbone | step | RI | RI% | HEST | HEST% | avg% | note |
|---|---|---|---|---|---|---|---|
| midnight | 250 | 0.91924 (97.1%) | n=1 | 0.40610 (50.7%) | n=1 | 73.9 | s0, ms1500 |
| virchow2 | 250 | 0.92340 (109.0%) | n=1 | 0.40666 (33.4%) | n=1 | 71.2 | s0, ms1500 |

---

## 2. New step-250 HEST scores (n=1 each)

These were added after the prior edition.

| job | recipe | backbone | pooling | HEST | HEST% | RI | RI% |
|---|---|---|---|---|---|---|---|
| 391394 | kl0.003 | midnight | cls | 0.41147 | 75.7% | 0.9055 | 88.8% |
| 391395 | kl0.003 | virchow2 | clsmean | 0.40839 | 50.2% | 0.8976 | 65.8% |
| 391769 | genMASK-lr3e-5 | virchow2 | clsmean | 0.40823 | 48.6% | 0.9176 | 99.4% |
| 391770 | lr3e-5 plain | virchow2 | clsmean | 0.40969 | 62.9% | 0.9004 | 70.9% |
| 391768 | genMASK-lr3e-5 | midnight | cls | 0.40951 | 66.6% | 0.9133 | 93.5% |

Note: `lr3e-5 plain` (391770) is lr3e-5 without masking — not a recipe in the main table,
included for calibration against the genMASK-lr3e-5 arms. The mask effect on Virchow2
(48.6% vs 62.9% HEST) is negative — masking costs HEST on that backbone.

---

## 3. Temperature dose-response (step 250, ms250, mask ON)

RI only; HEST pending. All on Midnight backbone, seed 0, genMASK recipe.

| job | backbone | T | RI | RI% | tcga | camelyon | tolkach |
|---|---|---|---|---|---|---|---|
| 391921 | midnight | 0.03 | 0.87937 | 73.0% | 0.8830 | 0.7872 | 0.9679 |
| 391888 | midnight | 0.04 | 0.88482 | 76.3% | 0.8856 | 0.7999 | 0.9689 |
| 391889 | Virchow2 | 0.04 | 0.89942 | 68.9% | 0.8538 | 0.8818 | 0.9627 |

**Interpretation:** lowering temperature costs RI monotonically, via camelyon
(confounder-insensitivity) collapsing while tcga (prediction performance) stays high — the
inverse of the known T=0.15 trap. Whether HEST/THUNDER capture the tcga-side gain is still
pending. The T=0.07 baseline (ret0.01 midnight at 0.90451 RI, 88.2%) confirms the monotonic
relationship: 0.03 < 0.04 < 0.07 in both temperature and RI.

---

## 4. Auxiliary stain/scanner head — arm A, Virchow2 (job 391929)

C=2, 250 steps, judged against a simulated best-constant-predictor baseline:

- **Scanner axis TRAINED**: CE slope t=-10.23, accuracy 0.229 -> 0.839 vs bar 0.590
- **Stain axis DID NOT TRAIN** but trending: t=-5.67, accuracy 0.129 -> 0.486 vs bar 0.541
- **Consequence**: a downstream null is interpretable for scanner, ambiguous for stain.

---

## 5. Pre-registered selection rule

This rule was written down before per-arm results were ranked, and is retained for audit. It
has been superseded by the cross-backbone generalization criterion above but is kept as-is
for reproducibility of the prior edition.

> **Eligibility.** A candidate is a single `(run_name, step)` for which **both** RI and HEST were
> measured at that exact checkpoint, at the correct per-backbone HEST pooling. THUNDER is *not*
> an eligibility requirement.
>
> **Score.** `avg_pct = mean( RI_pct_of_waiv , HEST_pct_of_waiv )` at that checkpoint, where
> `pct_of_waiv = (ours - base) / (Waiv - base) * 100`.
>
> **Pick.** Per backbone, argmax of `avg_pct` over all eligible checkpoints at steps 250 and 500.
>
> **Noise band.** `band = mean( 2*SD_RI / gain_RI , 2*SD_HEST / gain_HEST ) * 100`, in
> percentage-of-Waiv's-gain units, using the per-backbone per-step seed SDs from the final5 n=5
> study. Two candidates whose `avg_pct` differ by less than `band` are treated as tied.

### Noise bands actually used

| backbone | step | 2SD_RI as % of Waiv's RI gain | 2SD_HEST as % of Waiv's HEST gain | band |
|---|---|---|---|---|
| phikon-v2 | 250 / 500 | 2.7 | 17.0 | **9.9** |
| Midnight | 250 | 5.8 | 21.1 | **13.5** |
| Midnight | 500 | 2.6 | 17.8 | **10.2** |
| Virchow2 | 250 | 15.9 | 29.8 | **22.9** |
| Virchow2 | 500 | 6.6 | 22.4 | **14.5** |

### THUNDER is scored differently — read before quoting any THUNDER cell

Our THUNDER base does **not** reproduce Waiv's base (classification ~2-4 pp low; 2-dataset
segmentation ~4 pp high versus their 4-dataset one). Our base *does* reproduce THUNDER's own
paper. So THUNDER uses a two-base gain ratio: `(ours - OUR_base) / (Waiv_ft - Waiv_base)`.

| backbone | task | our base | Waiv base | Waiv ft | Waiv gain | 2SE/gain | flag |
|---|---|---|---|---|---|---|---|
| phikon | knn | 0.70281 | 0.740 | 0.777 | +0.03700 | 7% | |
| phikon | linear_probing | 0.76541 | 0.793 | 0.807 | +0.01400 | 18% | |
| phikon | simple_shot | 0.69330 | 0.718 | 0.733 | +0.01500 | 17% | |
| phikon | segmentation | 0.70401 | 0.665 | 0.653 | -0.01200 | 21% | WAIV-REGRESSED, UNRESOLVABLE, support_2v4 |
| midnight | knn | 0.78254 | 0.800 | 0.817 | +0.01700 | 15% | |
| midnight | linear_probing | 0.82880 | 0.844 | 0.846 | +0.00200 | 125% | **UNRESOLVABLE** |
| midnight | simple_shot | 0.70639 | 0.715 | 0.752 | +0.03700 | 7% | |
| midnight | segmentation | 0.70116 | 0.660 | 0.676 | +0.01600 | 16% | support_2v4 |
| virchow2 | knn | 0.80874 | 0.829 | 0.826 | -0.00300 | 83% | WAIV-REGRESSED, UNRESOLVABLE |
| virchow2 | linear_probing | 0.83253 | 0.848 | 0.851 | +0.00300 | 83% | **UNRESOLVABLE** |
| virchow2 | simple_shot | 0.72749 | 0.739 | 0.766 | +0.02700 | 9% | |
| virchow2 | segmentation | 0.71117 | 0.682 | 0.680 | -0.00200 | 125% | WAIV-REGRESSED, UNRESOLVABLE, support_2v4 |

**Only 5 of 12 backbone x THUNDER-task cells have a usable denominator.** Virchow2 kNN and
segmentation have **negative** Waiv gains. The only honest THUNDER bar this round is
`simple_shot` over the 12 classification datasets.

---

## 6. Selected checkpoint detail (for the ret0.01 lead recipe)

### 6.1 ret0.01 — phikon-v2, job 389548, seed 0

**Step 250** (the scoring step; step 250 is the only step present on all three backbones):

| metric | ours | base | Waiv | ours - Waiv | % of Waiv's gain | n |
|---|---|---|---|---|---|---|
| RI (avg) | 0.82416 | 0.46860 | 0.806 | +0.01816 | 105.4% | 1 |
| HEST (cls) | **MISSING** (job 392044, pending) | 0.37470 | 0.3943 | — | — | — |

**Step 500** (for reference only — prior doc mixed steps):

| metric | ours | base | Waiv | ours - Waiv | % of Waiv's gain | n |
|---|---|---|---|---|---|---|
| RI (avg) | 0.83551 | 0.46860 | 0.806 | +0.02951 | 107.4% | 1 |
| HEST (cls) | 0.38780 | 0.37470 | 0.3943 | -0.00650 | 79.9% | 1 |

### 6.2 ret0.01 — midnight, job 391057, seed 0, step 250

| metric | ours | base | Waiv | ours - Waiv | % of Waiv's gain | n |
|---|---|---|---|---|---|---|
| RI (avg) | 0.90451 | 0.75890 | 0.924 | -0.01949 | 88.2% | 1 |
| HEST (cls) | 0.40877 | 0.39521 | 0.4167 | -0.00793 | 63.1% | 1 | **UNRESOLVABLE** (2SD = 21.1% of gain) |

### 6.3 ret0.01 — Virchow2, job 391059, seed 0, step 250

| metric | ours | base | Waiv | ours - Waiv | % of Waiv's gain | n |
|---|---|---|---|---|---|---|
| RI (avg) | 0.90322 | 0.85820 | 0.918 | -0.01478 | 75.3% | 1 |
| HEST (clsmean) | 0.41090 | 0.40324 | 0.4135 | -0.00260 | 74.6% | 1 | **UNRESOLVABLE** (2SD = 29.8% of gain) |

---

## 7. Corrections and notes

These corrections must travel with the numbers in this document.

1. **ret0.01 step mismatch resolved.** The earlier edition scored ret0.01 phikon at step 500
   while midnight/Virchow2 were at step 250. Step 250 is now the scoring step across all
   three backbones. Phikon HEST@250 is still pending (job 392044).

2. **"Retention is dead" referred to `kl0.003`/ms250, NOT `ret0.01`/ms1500.** These are
   different recipes (`retention_kl_weight=0.003` vs `0.01`, `max_steps=250` vs `1500`).
   ret0.01 is the best current generalizer (74.9% MIN); kl0.003 is 58.0%.

3. **GeM pool head at inference is vacuous.** 99.8% DC energy; scores are bit-identical to
   arithmetic mean. The pool-head-at-eval axis is closed. Waiv specifies no learned pooling
   head, so arithmetic-mean is the comparable protocol.

4. **Segmentation is out of scope.** We cover 2 of Waiv's 4 segmentation datasets, and Waiv
   **regressed** on Virchow2 and phikon segmentation. The honest THUNDER bar is few-shot over
   the 12 classification datasets.

5. **Only 5 of 12 backbone x THUNDER-task cells have a usable denominator.** Virchow2 kNN and
   segmentation have negative Waiv gains (Waiv regressed), so matching their regression would
   score 100%.

6. **`results_backup/` is a symlink onto `/data` and is NOT a backup.** Durable copies live at
   `/admin/home/ryan.kim/waiv_result_backups/`.

7. **Scored at step 250, not step 500 (for new runs).** The new HEST scores (391394-391395,
   391768-391770) are all at step 250, the only step present on all three backbones for the
   ret0.01 recipe.

8. **THUNDER is excluded from per-backbone averages in the recipe table** because it is not
   available for any of the new runs. When THUNDER lands, the average will be
   `mean(RI_pct, HEST_pct, THUNDER_pct)`.

9. **Recipe table now uses step 250 across all backbones.** The prior edition mixed steps
   (final5-plain used step 500 for phikon and Virchow2, step 250 for midnight). The fixed step
   is 250, the only step with measurements on all three backbones for the leading recipes.
   This corrects the LR-schedule confound.

10. **Recipe scores use seed means, not chosen seeds.** final5-plain is now the mean over its
    available seeds at step 250 (RI n=5, HEST n=3 for midnight/Virchow2, n=1 for phikon).
    The ph2 family uses s1/s2 (ms500) as replicates; s0 (ms1500) is excluded because it has a
    different LR schedule. The ranking is unchanged: ret0.01 (74.9%) > genMASK-lr3e-5 (74.0%)
    > ph2 (72.9%) > kl0.003 (58.0%) > final5-plain (54.8%).

11. **Seed noise is material to the 80% bar.** One seed-SD corresponds to 14.9 percentage
    points of `pct_of_waiv` on Virchow2 HEST and 7.9% on Virchow2 RI. The leading recipe
    (ret0.01 at 74.9%) is 5.1 points from the 80% bar — less than one SD on both metrics.
    **No recipe is statistically distinguishable from the 80% bar at n=1.** Seed replication
    is required before any recipe can be declared a pass or fail.

---

## 8. OPEN / PENDING — could change the conclusion

The following are in flight. Any one of them could move a recipe above or below the 80% bar.

| family | count | what | could it change things? |
|---|---|---|---|
| **ret0.01 THUNDER** | 36 (jobs 392003-392038) | Complete THUNDER sweeps for ret0.01 on all three backbones (prefixes `ret1p-`, `ret1m-`, `ret1v-`) | **Yes** — THUNDER data would add a third readout to the per-backbone average and could lift the MIN |
| ret0.01 phikon HEST@250 | 1 (job 392044) | Missing phikon HEST at the scoring step | **Yes** — phikon avg could drop from 105.4 (RI-only) to ~80, confirming 74.9 as the true MIN |
| ret0.01 virchow2 seed-1 | 1 (job 392045) | Second seed for Virchow2 ret0.01 | Could validate/invalidade the 74.9% figure |
| Mask roster THUNDER | ~22/24 classification cells | PH2 THUNDER on midnight and Virchow2 | Could improve ph2 ranking |
| Temperature-dose HEST | pending | HEST for T=0.03, T=0.04, T=0.07 arms | Could reveal whether lower T recovers HEST |
| Partial-mask b2.0 arms | in flight | Intermediate masking strength | Could find a better generalizer |
| ms1500 lr3e-5 seed replication | s1 (391934), s2 (391935) | lr3e-5 at ms1500 on midnight | Could improve the lr3e-5 baseline |
| Aux stain/scanner heads | 5 remaining arms | Multi-task heads | Could reveal scanner/stain decomposition effects |

---

## 9. Preserved checkpoints

`runs/` in the repo is a **symlink to `/data/ryan.kim/waiv_runs`**, i.e. every checkpoint lives on
volatile storage. Durable copies are maintained at `/admin/home/ryan.kim/waiv_result_backups/`.
(The `results_backup/` directory in the repo is a symlink onto `/data` and is NOT a backup.)

Key candidate checkpoints backed up at `/admin/home/ryan.kim/waiv_final_candidates/`:

| candidate | path | size |
|---|---|---|
| phikon-v2 (ret0.01) | `ret0.01-phikon-s0-t900-389548/step_0000250/` | — |
| Midnight (ret0.01) | `ret0.01-midnight-s0-t900-391057/step_0000250/` | — |
| Virchow2 (ret0.01) | `ret0.01-virchow2-s0-t900-391059/step_0000250/` | — |
| phikon-v2 (final5) | `final5-phikon-s0-t900-386794/step_0000500/` | 73 MB |
| Midnight (final5) | `final5-midnight-s4-t900-386803/step_0000500/` | 160 MB |
| Virchow2 (final5) | `final5-virchow2-s4-t900-386808/step_0000500/` | 102 MB |

Manifest: `/admin/home/ryan.kim/waiv_final_candidates/SHA256SUMS.txt` (60 entries).
Verify with `cd /admin/home/ryan.kim/waiv_final_candidates && sha256sum -c SHA256SUMS.txt`.

**`optim.pt` was deliberately not copied** — it is 188-302 MB per checkpoint and is only needed
to resume training, not to score or serve.

---

## 10. How to reproduce / how to score

All three benchmarks take the **step directory** (the one containing `adapter/`) as the checkpoint
argument. Set `REPO=/admin/home/ryan.kim/waiv` and
`ADAPTER=/admin/home/ryan.kim/waiv_final_candidates/<run>/step_<STEP>` (or the `/data` original).

**The per-backbone pooling and the explicit `--backbone` are both load-bearing.** Virchow2 (and
Midnight) adapters were saved with `base_model_name_or_path = null`; without `--backbone` the
loader silently builds a phikon-v2 and the adapter keys fail to load.

| backbone | HEST pooling | RI pooling | THUNDER pooling | `--backbone` |
|---|---|---|---|---|
| phikon-v2 | `cls` | `clsmean` | `cls` (`auto` resolves it) | `owkin/phikon-v2` |
| Midnight | `cls` | `clsmean` | `clsmean`, `cls` for segmentation | `kaiko-ai/midnight` |
| Virchow2 | **`clsmean`** | `clsmean` | `clsmean`, `cls` for segmentation | **`paige-ai/Virchow2`** |

### 10.1 HEST

```bash
cd $REPO
"$REPO/.venv-hest/bin/python" scripts/run_hest.py \
  --pooling clsmean \
  --exp-code "f5_<run>_s<step>_<pooling>" \
  --backbone <backbone> \
  --num-workers 0 \
  --adapter "$ADAPTER" \
  --lora-rank 32 --lora-alpha 64 --proj-out-dim 512
```

**Trap:** `run_hest.py` keys its embedding cache on `--exp-code` alone — pooling is *not* in the
cache key. Changing `--pooling` without changing `--exp-code` silently rescores stale features.

### 10.2 PathoROB robustness index

```bash
"$REPO/.venv/bin/python" scripts/eval_checkpoints.py \
  --run-dir "$REPO/runs/<run>" \
  --backbone <backbone> \
  --lora-rank 32 --lora-alpha 64 \
  --num-workers 8
```

Do **not** pass `--pooling` or `--pool-head` — `eval_checkpoints.py` defaults to `clsmean`.

### 10.3 THUNDER

```bash
scripts/submit_final5_evals.sh <run> <step> --thunder-only
```

Pass `auto` for pooling. Env: `THUNDER_BASE_DATA_FOLDER=/data/ryan.kim/thunder`.

**The `f5_<run>_s<step>` tag is mandatory** — `collect_final5.py` and `scoreboard.py` resolve
exactly that string.

### 10.4 Re-run the scoreboard

```bash
cd $REPO
.venv/bin/python scripts/scoreboard.py --step 500 --no-detail
.venv/bin/python scripts/scoreboard.py --step 250 --no-detail
.venv/bin/python scripts/scoreboard.py --step 250 --runs ret0.01-phikon-s0-t900-389548
```

---

## 11. Corrections to numbers in circulation

Checked against `docs/WAIV_COMPARISON.md`, `docs/FINAL_RESULTS.md`, `docs/FINAL5_RESULTS.md`,
`docs/CAVEATS.md`, `docs/NEGATIVE_MASKING.md`. Items already covered in section 7 are not
repeated here.

1. **`docs/FINAL_RESULTS.md` section 2 is a mixed-checkpoint table and should not be quoted.**
   RI from `splitgrid-3863xx` runs, baseline from `waiv-real-369043` / `waiv-virchow2-375367` /
   `waiv-midnight-369159` — different config (`pool_head = mean`), different `T`, different step
   per backbone.

2. **The "101% / 90.4% / 75.5% of Waiv's gap closed" line is unusable.**
   `FINAL5_RESULTS.md` section 6 already retracts it.

3. **Midnight HEST has four different FT values in circulation** — 0.41322 (retracted), 0.40949,
   0.41180 (job 386398, no step recorded), 0.4065 (final5 n=5 mean). The +0.0166 / "77% of
   Waiv" figure does not replicate.

4. **Three mutually incompatible phikon RI headlines exist** (0.8080 @s1000; 0.8335 @s500
   argmax-selected on no-GEM run; 0.82694 @s500 n=5).

5. **`docs/CAVEATS.md` and `docs/WAIV_COMPARISON.md` disagree on HEST pooling.**
   `cls` for phikon and Midnight, `clsmean` for Virchow2 is correct.

6. **`docs/CAVEATS.md` claims THUNDER covers "4/4 segmentation datasets".** It does not — we
   run `ocelot` + `pannuke` only.

7. **`docs/NEGATIVE_MASKING.md` mixes steps in its summary sentence.**

8. **None of the five selected/fallback runs was preempted.** `sacct --duplicates` shows no
   REQUEUED / PREEMPTED / TIMEOUT.

---

## Appendix A: Complete-metric fallback checkpoints

For Midnight and Virchow2, fully-measured single checkpoints with THUNDER available:

### Midnight fallback — `final5-midnight-s4-t900-386803`, step 500, job 386803, seed 4, n=1

| metric | ours | base | Waiv | % of Waiv's gain | notes |
|---|---|---|---|---|---|
| RI (avg) | 0.89773 | 0.75890 | 0.924 | 84.1% | PASS floor |
| HEST (cls) | 0.40803 | 0.39521 | 0.4167 | 59.6% | 4.5xSD |
| THUNDER simple_shot | 0.74818 | 0.70639 | — | 112.9% | 12/12; **strongest usable THUNDER** |
| THUNDER knn | 0.78369 | 0.78254 | — | 6.8% | ~noise |
| THUNDER linear_probing | 0.83403 | 0.82880 | — | 261.5% | **UNRESOLVABLE** |
| THUNDER segmentation | 0.71316 | 0.70116 | — | 75.0% | support_2v4 |

### Virchow2 fallback — `final5-virchow2-s4-t900-386808`, step 500, job 386808, seed 4, n=1

| metric | ours | base | Waiv | % of Waiv's gain | notes |
|---|---|---|---|---|---|
| RI (avg) | 0.89545 | 0.85820 | 0.918 | 62.3% | **FAILS 80% floor** |
| HEST (clsmean) | 0.40697 | 0.40324 | 0.4135 | 36.3% | UNRESOLVABLE |
| THUNDER simple_shot | 0.74518 | 0.72749 | — | 65.5% | 12/12; 7.1xSD |
| THUNDER knn | 0.79040 | 0.80874 | — | N/A | Waiv regressed |
| THUNDER linear_probing | 0.83097 | 0.83253 | — | -52.0% | UNRESOLVABLE |
| THUNDER segmentation | 0.71996 | 0.71117 | — | N/A | Waiv regressed |
