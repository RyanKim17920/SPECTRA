# Waiv final scoreboard

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/final_scoreboard.py
```

Waiv targets: `docs/waiv_published.json (arXiv:2607.22861v1 Tables 1/2/3, verified 2026-08-24)`.
Every number below is read from disk at generation time.  `MISSING` means the
metric is not on disk for that cell; it is never substituted from another
checkpoint, another step, or another arm.

## 1. Headline: the graded criterion (RI / HEST / THUNDER)

Source: `scripts/final_recipe_report.py` (`build_report()`), re-run by this
command; machine-readable copy in `docs/final_recipe_verdict.json`.

Run family: `genMASK-c50-*` -- the FINALISED recipe (WAIV_BCLS=3.0, WAIV_BMEAN=-inf, ms500, warmup 200, lr 1e-4, rank 32, projdim 512, t900,
CKPT_EVERY=50, pin `falseneg-gated`), five backbones, 22 runs discovered.

Checkpoint per run is chosen by the **1se rule**: 1-SE (online): B = best avg_robustness_index so far; STOP at the first checkpoint (index >= 2) with R_t - B <= SE; RETURN the EARLIEST checkpoint with RI >= B - SE.
Metric: PathoROB avg_robustness_index (bounded, published).  This SUPERSEDES the retired `confounder_insensitivity >= 0.75` rule (and its 250/125/125 picks), which
graded an unbounded odds with a per-dataset chance level -- see `docs/CAVEATS.md`.

SE fed to the rule: **0.007** -- docs/RESULTS.md section 12.3 measured between-seed avg-RI floor 0.0070 (max |ctrl - ctrlseed| over checkpoints, n=2); OPERATOR INPUT, not a per-checkpoint bootstrap SE -- PathoROB's bootstrap fields are on no curve on disk.
Per-checkpoint bootstrap SE found on disk: False.
The sensitivity of every pick to that one number is tabulated below, so the
choice is checkable rather than asserted.

ONE ROW = ONE (run, step): the same rule-selected checkpoint feeds RI, HEST and
THUNDER for a given run.  Best-RI from one checkpoint and best-HEST from another
is never combined.  Where two seeds of one backbone plateau at different steps
the `step` cell lists both and the cell's floor is the larger of the two.

`pct` = (our mean - our base) / (Waiv ft - Waiv base) x 100, UNCAPPED.

### 1a. Checkpoint the rule selected, per run

| backbone | seed | run | selected step | RI curve (step:RI) | note |
|---|---|---|---|---|---|
| H-Optimus-0 | 0 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s0-t900-395391` | 100 | 50:0.8392 100:0.9086 150:0.9140 200:0.9063 |  |
| H-Optimus-0 | 1 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s1-t900-395870` | 100 | 50:0.8455 100:0.9072 150:0.9107 |  |
| H-Optimus-0 | 2 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s2-t900-396382` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| H-Optimus-0 | 3 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s3-t900-396384` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Midnight-12k | 0 | `genMASK-c50-lr1e-4-kl0-ms500-midnight-s0-t900-399166` | 150 | 50:0.8024 100:0.9020 150:0.9108 200:0.9077 250:0.9046 300:0.9027 350:0.9060 400:0.9047 450:0.9019 500:0.9041 |  |
| Midnight-12k | 1 | `genMASK-c50-lr1e-4-kl0-ms500-midnight-s1-t900-407566` | NOT SELECTED | 50:0.8043 | 1-SE rule UNTERMINATED: RI is still improving by more than one SE at the last measured checkpoint ([50]), so the run has not plateaued.  Returning the last checkpoint here would be the stop-at-last rule under this rule's name. |
| Midnight-12k | 2 | `genMASK-c50-lr1e-4-kl0-ms500-midnight-s2-t900-407569` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Phikon-v2 | 0 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165` | 200 | 50:0.4816 100:0.6462 150:0.8032 200:0.8424 250:0.8371 300:0.8388 350:0.8365 400:0.8384 450:0.8368 500:0.8373 |  |
| Phikon-v2 | 1 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s1-t900-407565` | 200 | 50:0.4820 100:0.6577 150:0.8156 200:0.8358 250:0.8351 300:0.8326 350:0.8340 400:0.8340 450:0.8314 |  |
| Phikon-v2 | 2 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s2-t900-407568` | NOT SELECTED | 50:0.4818 | 1-SE rule UNTERMINATED: RI is still improving by more than one SE at the last measured checkpoint ([50]), so the run has not plateaued.  Returning the last checkpoint here would be the stop-at-last rule under this rule's name. |
| UNI2-h | 0 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s0-t900-395390` | 100 | 50:0.8202 100:0.9051 150:0.9094 200:0.9077 250:0.9061 300:0.9035 |  |
| UNI2-h | 1 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s1-t900-395869` | 150 | 50:0.8196 100:0.9043 150:0.9114 200:0.9106 250:0.9057 300:0.9066 |  |
| UNI2-h | 2 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s2-t900-396381` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| UNI2-h | 3 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s3-t900-396383` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Virchow2 | 0 | `genMASK-c50-lr1e-4-kl0-ms500-virchow2-s0-t900-399167` | 100 | 50:0.8758 100:0.9043 150:0.9083 200:0.8990 250:0.9030 300:0.8990 350:0.9004 400:0.9000 450:0.8966 500:0.8996 |  |
| Virchow2 | 1 | `genMASK-c50-lr1e-4-kl0-ms500-virchow2-s1-t900-407567` | NOT SELECTED | 50:0.8779 100:0.9034 | 1-SE rule UNTERMINATED: RI is still improving by more than one SE at the last measured checkpoint ([100]), so the run has not plateaued.  Returning the last checkpoint here would be the stop-at-last rule under this rule's name. |
| Virchow2 | 2 | `genMASK-c50-lr1e-4-kl0-ms500-virchow2-s2-t900-407570` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| H-Optimus-0 | 3 | `genMASK-c50-ms500-hoptimus-s3-t900-436608` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| H-Optimus-0 | 4 | `genMASK-c50-ms500-hoptimus-s4-t900-436613` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Midnight-12k | 3 | `genMASK-c50-ms500-midnight-s3-t900-436609` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Midnight-12k | 4 | `genMASK-c50-ms500-midnight-s4-t900-436614` | NOT SELECTED | no curve | no ri_curve.json -- the checkpoint rule cannot be applied (RI and THUNDER cells excluded) |
| Virchow2 | 3 | `genMASK-c50-ms500-virchow2-s3-t900-436610` | NOT SELECTED | 50:0.8772 | 1-SE rule UNTERMINATED: RI is still improving by more than one SE at the last measured checkpoint ([50]), so the run has not plateaued.  Returning the last checkpoint here would be the stop-at-last rule under this rule's name. |

### 1b. Sensitivity of the picks to the SE (diagnostic, not a selection)

| SE | H-Optimus-0 | Midnight-12k | Phikon-v2 | UNI2-h | Virchow2 |
|---|---|---|---|---|---|
| 0.001 | 150 | 150 | 200 | 150 | 150 |
| 0.002 | 150 | 150 | 200 | 150 | 150 |
| 0.003 | 150 | 150 | 200 | 150 | 150 |
| 0.004 | 100,150 | 150 | 200 | 150 | 100 |
| 0.005 | 100,150 | 150 | 200 | 100,150 | 100 |
| 0.006 | 100 | 150 | 200 | 100,150 | 100 |
| 0.007 | 100 | 150 | 200 | 100,150 | 100 |
| 0.008 | 100 | 150 | 200 | 100 | 100 |
| 0.01 | 100 | 100 | 200 | 100 | 100 |
| 0.015 | 100 | 100 | 200 | 100 | 100 |
| 0.02 | 100 | 100 | 200 | 100 | 100 |

A cell with two steps means the seeds of that backbone disagree at that SE; `-` means the rule did not fire on any seed.

| backbone | benchmark | step | ours | our base | Waiv base | Waiv ft | our gain | Waiv gain | pct of Waiv | +/-95% | n | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phikon-v2 | RI | 200 | 0.83910 | 0.46861 | 0.46900 | 0.80600 | 0.37049 | 0.33739 | 109.8 | 2.0 | 2 | PASS |
| Phikon-v2 | HEST | 200 | 0.39050 | 0.37470 | 0.37470 | 0.39430 | 0.01580 | 0.01960 | 80.6 | 10.7 | 2 | NOT RESOLVED |
| Phikon-v2 | THUNDER | - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0 | PARTIAL |
| Midnight-12k | RI | 150 | 0.91082 | 0.75886 | 0.75900 | 0.92400 | 0.15196 | 0.16514 | MISSING | 5.8 | 1 | UNDERPOWERED |
| Midnight-12k | HEST | 150 | 0.41322 | 0.39521 | 0.39520 | 0.41670 | 0.01801 | 0.02149 | MISSING | 23.6 | 1 | UNDERPOWERED |
| Midnight-12k | THUNDER | - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0 | PARTIAL |
| Virchow2 | RI | 100 | 0.90430 | 0.85824 | 0.85800 | 0.91800 | 0.04606 | 0.05976 | MISSING | 15.9 | 1 | UNDERPOWERED |
| Virchow2 | HEST | 100 | 0.40949 | 0.40327 | 0.40340 | 0.41350 | 0.00622 | 0.01023 | MISSING | 35.1 | 1 | UNDERPOWERED |
| Virchow2 | THUNDER | - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0 | PARTIAL |
| H-Optimus-0 | RI | 100 | 0.90792 | 0.81165 | 0.81100 | 0.91700 | 0.09627 | 0.10535 | 91.4 | 1.4 | 2 | PASS |
| H-Optimus-0 | HEST | 100 | 0.42225 | 0.41500 | 0.41500 | 0.42900 | 0.00725 | 0.01400 | 51.8 | 10.9 | 2 | FAIL |
| H-Optimus-0 | THUNDER | - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0 | PARTIAL |
| UNI2-h | RI | 100/150 | 0.90829 | 0.75667 | 0.75700 | 0.90800 | 0.15162 | 0.15133 | 100.2 | 4.1 | 2 | PASS |
| UNI2-h | HEST | 100/150 | 0.42380 | 0.41380 | 0.41410 | 0.42900 | 0.00999 | 0.01520 | 65.8 | 37.6 | 2 | NOT RESOLVED |
| UNI2-h | THUNDER | - | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0 | PARTIAL |

Per-model verdict (THE criterion: pct >= 70 on each of RI/HEST/THUNDER and mean of the three > 80):

| backbone | RI | HEST | THUNDER | average | verdict |
|---|---|---|---|---|---|
| Phikon-v2 | 109.8 +/-2.0 [PASS] | 80.6 +/-10.7 [NOT RESOLVED] | MISSING +/-MISSING [PARTIAL] | MISSING | INDETERMINATE -- no gradeable number for THUNDER (PARTIAL); error bar straddles the 70 bar for HEST (80.6+/-10.7) |
| Midnight-12k | MISSING +/-5.8 [UNDERPOWERED] | MISSING +/-23.6 [UNDERPOWERED] | MISSING +/-MISSING [PARTIAL] | MISSING | INDETERMINATE -- no gradeable number for RI (UNDERPOWERED), HEST (UNDERPOWERED), THUNDER (PARTIAL) |
| Virchow2 | MISSING +/-15.9 [UNDERPOWERED] | MISSING +/-35.1 [UNDERPOWERED] | MISSING +/-MISSING [PARTIAL] | MISSING | INDETERMINATE -- no gradeable number for RI (UNDERPOWERED), HEST (UNDERPOWERED), THUNDER (PARTIAL) |
| H-Optimus-0 | 91.4 +/-1.4 [PASS] | 51.8 +/-10.9 [FAIL] | MISSING +/-MISSING [PARTIAL] | MISSING | INDETERMINATE -- no gradeable number for THUNDER (PARTIAL) |
| UNI2-h | 100.2 +/-4.1 [PASS] | 65.8 +/-37.6 [NOT RESOLVED] | MISSING +/-MISSING [PARTIAL] | MISSING | INDETERMINATE -- no gradeable number for THUNDER (PARTIAL); error bar straddles the 70 bar for HEST (65.8+/-37.6) |

**Overall (per-model) verdict: INDETERMINATE**

## 2. THUNDER, all six published tasks (second corpus)

Source: `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv` -- the harness's own `benchmark_*` roll-up rows,
which are the same quantity Waiv tabulate in their Table 2.  This corpus is
the ONLY place segmentation, calibration (ECE) and adversarial exist on our
side; the old harness computes none of them.  Its base-controls also sit much
closer to Waiv's published base than the old harness does (`base gap` column),
so the within-corpus base-vs-tuned delta is the defensible one.

### Phikon-v2

| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | c3s-s0-step250 | c3s-s1-step250 | c50-s0-step200 | pct of Waiv (recipe arm, mean over seeds) |
|---|---|---|---|---|---|---|---|---|---|
| knn | 73.9 | 74.0 | -0.1 | 77.7 | 3.7 | 77.4 | 76.3 | 77.3 | 79.7 (n=2) |
| linear | 79.7 | 79.3 | 0.4 | 80.7 | 1.4 | 81.4 | 80.8 | 81.4 | 100.0 (n=2) |
| few_shot | 71.8 | 71.8 | 0.0 | 73.3 | 1.5 | 72.3 | 71.9 | 72.5 | 20.0 (n=2) |
| segmentation | 67.3 | 66.5 | 0.8 | 65.3 | -1.2 | 66.2 | 67.3 | 67.4 | 45.8 (n=2) |
| calibration (lower is better) | 3.8 | 4.5 | -0.7 | 3.0 | -1.5 | 3.6 | 3.7 | 3.8 | 10.0 (n=2) |
| adversarial (lower is better) | 36.2 | 41.9 | -5.7 | 38.8 | -3.1 | 28.1 | 27.5 | MISSING | 271.0 (n=2) |

### Midnight-12k

| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step150 | pct of Waiv (recipe arm, mean over seeds) |
|---|---|---|---|---|---|---|---|---|---|
| knn | 80.0 | 80.0 | 0.0 | 81.7 | 1.7 | 81.9 | 81.8 | 81.4 | 108.8 (n=2) |
| linear | 84.8 | 84.4 | 0.4 | 84.6 | 0.2 | 85.8 | 85.6 | 85.7 | INDETERMINATE (Waiv gain +0.2pp is within 0.2pp print error) |
| few_shot | 71.5 | 71.5 | 0.0 | 75.2 | 3.7 | 77.1 | 77.4 | 77.0 | 155.4 (n=2) |
| segmentation | 68.0 | 66.0 | 2.0 | 67.6 | 1.6 | 68.6 | 68.3 | 68.1 | 28.1 (n=2) |
| calibration (lower is better) | 2.9 | 2.4 | 0.5 | 2.3 | -0.1 | 3.7 | 3.6 | 3.5 | INDETERMINATE (Waiv gain -0.1pp is within 0.2pp print error) |
| adversarial (lower is better) | 29.9 | 35.7 | -5.8 | 23.2 | -12.5 | 23.3 | 22.6 | 21.1 | 55.6 (n=2) |

* INDETERMINATE cells above: Waiv's own published gain for that task is at or below 0.2pp, twice the 0.1pp granularity their table is printed to, so the ratio is rounding, not a measurement.  Read the raw columns for these tasks.

### Virchow2

| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | c3s-s0-step125 | c3s-s1-step125 | pct of Waiv (recipe arm, mean over seeds) |
|---|---|---|---|---|---|---|---|---|
| knn | 82.9 | 82.9 | 0.0 | 82.6 | -0.3 | 82.8 | 82.8 | 33.3 (n=2) |
| linear | 84.7 | 84.8 | -0.1 | 85.1 | 0.3 | 85.3 | 85.6 | 250.0 (n=2) |
| few_shot | 74.0 | 73.9 | 0.1 | 76.6 | 2.7 | 77.8 | 78.1 | 146.3 (n=2) |
| segmentation | 69.0 | 68.2 | 0.8 | 68.0 | -0.2 | 68.9 | 69.1 | INDETERMINATE (Waiv gain -0.2pp is within 0.2pp print error) |
| calibration (lower is better) | 4.0 | 3.6 | 0.4 | 4.2 | 0.6 | 4.3 | 4.2 | 41.7 (n=2) |
| adversarial (lower is better) | 0.3 | 31.1 | -30.8 | 7.7 | -23.4 | 0.2 | 0.1 | SUSPECT -- not scored |

* **adversarial SUSPECT for Virchow2** -- attack ineffective: our f1 drop is <5pp where Waiv report a large drop for the same base weights, so the drop measures the attack, not the model.  Printed, not scored.

* INDETERMINATE cells above: Waiv's own published gain for that task is at or below 0.2pp, twice the 0.1pp granularity their table is printed to, so the ratio is rounding, not a measurement.  Read the raw columns for these tasks.

### H-Optimus-0

| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | bm3-s0-step100 | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step100 | c50-s0-step150 | c50-s0-step50 | pct of Waiv (recipe arm, mean over seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 81.4 | 81.5 | -0.1 | 81.9 | 0.4 | 80.6 | 80.9 | 81.7 | 81.6 | 81.2 | 81.8 | -25.0 (n=2) |
| linear | 83.8 | 83.7 | 0.1 | 84.0 | 0.3 | 83.9 | 84.0 | 83.3 | 84.2 | 83.8 | 83.9 | -50.0 (n=2) |
| few_shot | 76.1 | 76.2 | -0.1 | 77.4 | 1.2 | 76.7 | 77.0 | 77.2 | 77.0 | 77.0 | 76.8 | 83.3 (n=2) |
| segmentation | 64.6 | 63.5 | 1.1 | 68.1 | 4.6 | 65.7 | 65.2 | 65.3 | 64.8 | 65.1 | 64.3 | 14.1 (n=2) |
| calibration (lower is better) | 4.0 | 3.6 | 0.4 | 3.2 | -0.4 | 3.4 | 3.7 | 3.7 | 3.5 | 4.5 | 3.5 | 75.0 (n=2) |
| adversarial (lower is better) | 32.4 | 42.1 | -9.7 | 32.4 | -9.7 | 25.7 | 26.4 | 26.1 | 27.5 | 27.6 | 32.0 | 63.4 (n=2) |

### UNI2-h

| task | our base | Waiv base | base gap | Waiv ft | Waiv gain | bm3-s0-step100 | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step100 | c50-s0-step150 | c50-s0-step50 | pct of Waiv (recipe arm, mean over seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 83.3 | 83.3 | 0.0 | 83.4 | 0.1 | 83.3 | 83.0 | 82.8 | 83.3 | 82.5 | 83.0 | INDETERMINATE (Waiv gain +0.1pp is within 0.2pp print error) |
| linear | 85.7 | 86.3 | -0.6 | 85.5 | -0.8 | 86.0 | 86.1 | 86.2 | 86.1 | 85.9 | 85.7 | -56.3 (n=2) |
| few_shot | 79.8 | 79.8 | 0.0 | 79.5 | -0.3 | 79.1 | 78.8 | 78.7 | 79.2 | 78.7 | 80.0 | 350.0 (n=2) |
| segmentation | 69.2 | 68.1 | 1.1 | 67.6 | -0.5 | 69.1 | 68.7 | 68.4 | 69.0 | 69.3 | 69.0 | 130.0 (n=2) |
| calibration (lower is better) | 3.9 | 3.7 | 0.2 | 2.5 | -1.2 | 3.4 | 4.3 | 4.1 | 4.6 | 4.2 | 3.7 | -25.0 (n=2) |
| adversarial (lower is better) | 26.8 | 31.0 | -4.2 | 24.1 | -6.9 | 21.2 | 19.5 | 19.0 | 22.0 | MISSING | 27.8 | 109.4 (n=2) |

* INDETERMINATE cells above: Waiv's own published gain for that task is at or below 0.2pp, twice the 0.1pp granularity their table is printed to, so the ratio is rounding, not a measurement.  Read the raw columns for these tasks.

## 3. PathoROB robustness index, all five backbones (second corpus)

Source: `/data/ryan.kim/pathfm-full-evals/pathorob/results/robustness_index/<model>_clsmean/<dataset>/-1_0/results_summary.json`,
key `robustness_index`, averaged over tcga / camelyon / tolkach_esca -- the same
three datasets and the same key section 1 uses, but from the newer corpus, which
is the only one carrying the two gated backbones.

| backbone | our base-control | Waiv base | Waiv ft | best tuned (arm) | pct of Waiv |
|---|---|---|---|---|---|
| Phikon-v2 | 0.4701 | 0.4690 | 0.8060 | 0.8421 (c50-s0-step200) | 110.4 |
| Midnight-12k | 0.7589 | 0.7590 | 0.9240 | 0.9126 (c3s-s1-step125) | 93.2 |
| Virchow2 | 0.8610 | 0.8580 | 0.9180 | 0.9088 (c3s-s1-step125) | 79.7 |
| H-Optimus-0 | 0.7997 | 0.8110 | 0.9170 | 0.9124 (c50-s0-step150) | 106.3 |
| UNI2-h | 0.7566 | 0.7570 | 0.9080 | 0.9094 (c50-s0-step150) | 101.2 |

Note: `phikon2-base-control_clsmean` is absent from this corpus, so the
phikon base cell above is MISSING; section 1's phikon base RI comes from the
repo-local `third_party/PathoROB` tree instead and is NOT interchangeable.

## 4. CPTAC / Patho-Bench

Source: `/data/ryan.kim/pathfm-full-evals/cptac/<model>/aggregate.json`, key `classification_macro_ovr_auc`.
Waiv's side is `docs/waiv_published.json -> table4_pathobench`.  Only the
mutation/MSI AUC tasks are metric-compatible: Waiv score `Immune class` as
balanced accuracy while we score it as macro-OvR AUC, and their survival cells
are a C-index we compute per-alpha, so both groups are excluded from the paired
mean and listed as MISSING rather than silently averaged in.

| backbone | arm | n matched tasks | our mean AUC (matched) | Waiv base | Waiv ft | pct of Waiv |
|---|---|---|---|---|---|---|
| Phikon-v2 | c3s-s0-step250 (no base-control) | 25 | 68.58 | 66.43 | 67.86 | MISSING |
| Phikon-v2 | c3s-s1-step250 (no base-control) | 25 | 68.64 | 66.43 | 67.86 | MISSING |
| Phikon-v2 | c50-s0-step200 (no base-control) | 25 | 69.05 | 66.43 | 67.86 | MISSING |
| Midnight-12k | base-control | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |
| Midnight-12k | c3s-s0-step125 | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |
| Midnight-12k | c3s-s1-step125 | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |
| Midnight-12k | c50-s0-step150 (no base-control) | 25 | 68.67 | 65.00 | 67.94 | MISSING |
| Virchow2 | c3s-s0-step125 | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |
| Virchow2 | c50-s0-step100 (no base-control) | 25 | 68.90 | 66.90 | 68.01 | MISSING |
| H-Optimus-0 | base-control | 25 | 67.43 | 67.56 | 70.77 | MISSING |
| H-Optimus-0 | bm3-s0-step100 | 25 | 68.86 | 67.56 | 70.77 | 44.8 |
| H-Optimus-0 | c3s-s0-step125 | 25 | 69.45 | 67.56 | 70.77 | 63.2 |
| H-Optimus-0 | c3s-s1-step125 | 25 | 69.51 | 67.56 | 70.77 | 65.0 |
| H-Optimus-0 | c50-s0-step100 | 25 | 69.33 | 67.56 | 70.77 | 59.2 |
| H-Optimus-0 | c50-s0-step150 | 25 | 69.71 | 67.56 | 70.77 | 71.2 |
| H-Optimus-0 | c50-s0-step50 | 25 | 67.69 | 67.56 | 70.77 | 8.2 |
| UNI2-h | base-control | 25 | 67.75 | 67.47 | 69.01 | MISSING |
| UNI2-h | bm3-s0-step100 | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |
| UNI2-h | c3s-s0-step125 | 25 | 69.95 | 67.47 | 69.01 | 143.1 |
| UNI2-h | c3s-s1-step125 | 25 | 69.89 | 67.47 | 69.01 | 138.9 |
| UNI2-h | c50-s0-step100 | 0 | MISSING (no results on disk) | MISSING | MISSING | MISSING |

Coverage caveat: only H-Optimus-0 and UNI2-h have a CPTAC base-control on
disk, so only those two backbones can express a gain-over-base at all.  The
midnight / phikon-v2 / Virchow2 rows are absolutes with no base and therefore
no pct.

## 5. MISSING inventory

Everything the paper's table would want that is NOT on disk, stated once so no
reader has to infer it from a blank cell.

| item | status | why |
|---|---|---|
| THUNDER segmentation (section 1) | MISSING | `collect_final5.PAPER_SEG` defaults to ocelot+pannuke and those two cells were not run for every arm; the 16-set roster has no SPIDER segmentation task at all. Section 2 carries segmentation from the second corpus instead. |
| THUNDER calibration / adversarial (section 1) | NOT COMPUTED | `eval_common.WAIV_THUNDER_TASKS` deliberately covers four tasks; the old harness never computed ECE or an attack.  Section 2 carries both. |
| THUNDER adversarial, Virchow2 only | SUSPECT | attack ineffective: our f1 drop is <5pp where Waiv report a large drop for the same base weights, so the drop measures the attack, not the model.  Printed, not scored.  All three Virchow2 models report a 0.1-0.3pp drop against a published 31.1; the other four backbones report 19-32 and are scored normally. |
| PathoROB base-control for phikon-v2 (second corpus) | MISSING | `phikon2-base-control_clsmean` is absent from all three metric dirs under /data/ryan.kim/pathfm-full-evals/pathorob/results. |
| CPTAC base-control for midnight / phikon-v2 / Virchow2 | MISSING | Only hoptimus0 and uni2h have a `base-control` dir under the CPTAC tree. |
| CPTAC for 4 hoptimus arms | EMPTY | hoptimus0-bm3-s0-step100 and hoptimus0-c50-s0-step{50,100,150} have no `.complete`, no aggregate.json and zero task dirs. |
| CPTAC Immune class / survival | NOT COMPARED | metric mismatch: Waiv report balanced accuracy and C-index, we compute macro-OvR AUC and a per-alpha C-index. |
| Waiv Patho-Bench grand average (63 tasks) | NOT COMPARABLE | our CPTAC corpus covers 38 tasks, 26 of which map onto their table; their grand average also spans Hancock / PANDA / BC-Therapy cohorts we never ran. |
| section 1 THUNDER for Phikon-v2 | PARTIAL | coverage 0,0/16 per seed; 12ds floor invalid below 12/12; nothing on disk for this (run, step) under /data/ryan.kim/thunder/outputs/res -- searched f5_ci-phikon-s0-399165_s0000200, f5_ci-phikon-s1-407565_s0000200, f5_genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165_s0000200, f5_genMASK-c50-lr1e-4-kl0-ms500-phikon-s1-t900-407565_s0000200 |
| section 1 THUNDER for Midnight-12k | PARTIAL | coverage 0/16 per seed; 12ds floor invalid below 12/12; nothing on disk for this (run, step) under /data/ryan.kim/thunder/outputs/res -- searched f5_ci-midnight-s0-399166_s0000150, f5_genMASK-c50-lr1e-4-kl0-ms500-midnight-s0-t900-399166_s0000150 |
| section 1 THUNDER for Virchow2 | PARTIAL | coverage 0/16 per seed; 12ds floor invalid below 12/12; nothing on disk for this (run, step) under /data/ryan.kim/thunder/outputs/res -- searched f5_ci-virchow2-s0-399167_s0000100, f5_genMASK-c50-lr1e-4-kl0-ms500-virchow2-s0-t900-399167_s0000100 |
| section 1 THUNDER for H-Optimus-0 | PARTIAL | coverage 0,0/16 per seed; 12ds floor invalid below 12/12; nothing on disk for this (run, step) under /data/ryan.kim/thunder/outputs/res -- searched f5_ci-hoptimus-s0-395391_s0000100, f5_ci-hoptimus-s1-395870_s0000100, f5_genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s0-t900-395391_s0000100, f5_genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s1-t900-395870_s0000100 |
| section 1 THUNDER for UNI2-h | PARTIAL | coverage 0,0/16 per seed; 12ds floor invalid below 12/12; nothing on disk for this (run, step) under /data/ryan.kim/thunder/outputs/res -- searched f5_ci-uni2-s0-395390_s0000100, f5_ci-uni2-s1-395869_s0000150, f5_genMASK-c50-lr1e-4-kl0-ms500-uni2-s0-t900-395390_s0000100, f5_genMASK-c50-lr1e-4-kl0-ms500-uni2-s1-t900-395869_s0000150 |
| THUNDER at the section-1 checkpoint for Phikon-v2 | IN THE SECOND CORPUS ONLY | the rule-selected checkpoint(s) 200 have THUNDER results as phikon2-c50-s0-step200 under `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv` (section 2), not in the old harness section 1 grades against.  They are NOT merged into section 1: the 12-dataset seed floors for phikon-v2 / Midnight-12k / Virchow2 were measured in the OLD corpus and its Resize(224,bilinear) transform, so a numerator from one corpus over a floor from the other is not a matched comparison. |
| THUNDER at the section-1 checkpoint for Midnight-12k | IN THE SECOND CORPUS ONLY | the rule-selected checkpoint(s) 150 have THUNDER results as midnight-c50-s0-step150 under `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv` (section 2), not in the old harness section 1 grades against.  They are NOT merged into section 1: the 12-dataset seed floors for phikon-v2 / Midnight-12k / Virchow2 were measured in the OLD corpus and its Resize(224,bilinear) transform, so a numerator from one corpus over a floor from the other is not a matched comparison. |
| THUNDER at the section-1 checkpoint for H-Optimus-0 | IN THE SECOND CORPUS ONLY | the rule-selected checkpoint(s) 100 have THUNDER results as hoptimus0-c50-s0-step100 under `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv` (section 2), not in the old harness section 1 grades against.  They are NOT merged into section 1: the 12-dataset seed floors for phikon-v2 / Midnight-12k / Virchow2 were measured in the OLD corpus and its Resize(224,bilinear) transform, so a numerator from one corpus over a floor from the other is not a matched comparison. |
| THUNDER at the section-1 checkpoint for UNI2-h | IN THE SECOND CORPUS ONLY | the rule-selected checkpoint(s) 100/150 have THUNDER results as uni2h-c50-s0-step100, uni2h-c50-s0-step150 under `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv` (section 2), not in the old harness section 1 grades against.  They are NOT merged into section 1: the 12-dataset seed floors for phikon-v2 / Midnight-12k / Virchow2 were measured in the OLD corpus and its Resize(224,bilinear) transform, so a numerator from one corpus over a floor from the other is not a matched comparison. |

