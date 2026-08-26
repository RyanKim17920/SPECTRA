# Final results — 2026-08-18

Status of every measurement at session end. Evidence tags: **MEASURED** (on disk) · **PENDING** (job queued/running) · **NOT MEASURABLE** (structurally impossible, see reason).

> **SUPERSESSION BANNER (2026-08-26).** All THUNDER seed-floor numbers in this document were recomputed by the 12-dataset study, `docs/thunder_seed_floor_12ds.{json,md}` (via `scripts/thunder_seed_floor_12ds.py`). The old floors (0.0156 / 0.0208) were wrong on **two independent axes**: (1) **dataset count** — they averaged 5 datasets, not 12; and (2) **run family** — they came from a different pair of runs at n=2 seeds rather than the final5 family at n=5. Both axes matter separately: `docs/thunder_seed_floor_12ds.md:145-152` shows that holding the dataset count fixed at 5 and changing only the run family already moves the floors substantially, and in opposite directions (cls/linear_probing 0.0156 → 0.0226; clsmean/linear_probing 0.0208 → 0.0142). Do not treat the change as a pure dataset-count rescaling.

---

## 1. Final configuration

**`--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem`, LoRA r32/α64, grid sampler at saturating negatives (C=2, T≈900–1800), scored at each run's own RI-optimal step.**

How each piece was established:

| choice | evidence |
|---|---|
| split heads, 0.5/0.5 | only axis ever to clear the HEST bar (+0.0091 vs ctrl) |
| GeM pooling | **+0.0025 on the grid path, 12/12 paired deltas positive, paired sd 0.00062** (2×2 at T=900, seeds 0/1). Also +0.002 on the pair path, 17/17 paired — confirmed on two independent samplers |
| grid sampler, C=2 | conditions are null: arms a (C=12) .8070, b (C=24) .7980, c (C=6) .8102, F (C=12/T=800) .8105 — all at or below ctrl |
| T ≈ 900 | negatives saturate: T=900 vs T=1800, same seed, differ by **0.0024** — under the 0.0076 floor |
| step selection on RI | THUNDER cannot arbitrate: 0/22 step deltas clear its corrected floor; RI-vs-THUNDER-LP r = **+0.03** |

**Critical methodological note:** GeM's effect is invisible unpaired. Raw seed spread on grid arms is ~0.009, roughly 15× the paired sd. Comparing arm means would have discarded exactly the statistical power that makes the comparison possible. Every small-effect comparison in this project must be paired by seed.

---

## 2. Three-backbone final results

Each backbone at its max feasible negatives (memory-bound), scored at its own RI-optimal step.

| backbone | T | negatives | final RI | published recon | Δ | vs 0.0076 floor |
|---|---|---|---|---|---|---|
| **phikon-v2** | 1800 | 1799 | **0.8335** @500 | 0.8080 | **+0.0255** | **3.4× — REAL** |
| Virchow2 | 600 | 599 | 0.9078 @250 | 0.9035 | +0.0043 | 0.6× — inside |
| midnight | 450 | 449 | 0.9027 @250 | 0.9080 | −0.0053 | 0.7× — inside |

**HEST on the final checkpoints**, each at its matched pooling protocol (§4):

| backbone | protocol | base | final FT | Δ | vs 0.0075 bar |
|---|---|---|---|---|---|
| **phikon-v2** | cls | 0.37470 | **0.38615** | **+0.0115** | **1.5× — clears** |
| midnight | cls | 0.39521 | 0.40949 | +0.0143 | 1.9× — clears |
| Virchow2 | clsmean | 0.40327 | PENDING | — | — |

**THUNDER on the final checkpoints** (fast-5, LP mean): phikon-v2 0.6978 (cls) / 0.7056 (clsmean), vs ctrl 0.7082 / 0.7127 — flat, differences 0.0104 and 0.0071 against corrected floors of **0.0097** and **0.0087** [corrected 2026-08-26: was "0.0156 and 0.0208", which were 5-dataset, n=2-seed floors from `docs/thunder_seed_floor.md`. The 12-dataset, n=5-seed floors are phikon/linear_probing 0.0097 and midnight/linear_probing 0.0087; authority `docs/thunder_seed_floor_12ds.md`. Note the cls difference 0.0104 now marginally *exceeds* its floor rather than sitting well under it]. This is the predicted outcome, not a regression: THUNDER was independently measured blind on this axis.

### 2.1 The headline finding

**The method delivers on phikon-v2 and does essentially nothing on the two stronger backbones.** phikon-v2 gains on both instruments with dynamic range (RI +0.0255, HEST +0.0115); midnight and Virchow2 sit inside the floor on RI.

This is not a one-off. **Four independent interventions showed the same pattern** — GEM at reference geometry, high negatives, the two composed, and the final split×grid config. It tracks base-model headroom, not architecture: gap-closed against Waiv falls monotonically with base strength (phikon-v2 101% → midnight 90% → Virchow2 76%), and phikon-v2 starts furthest from the ceiling (base RI 0.4686 vs 0.7589 and 0.8582).

---

## 3. Comparison vs published Waiv

Published Waiv (arXiv:2607.22861) released **gated weights and results but no method section** — no loss, algorithm, corpus, hyperparameters, or code. Ours is a reconstruction of their *results*, not their method, so every comparison is Δ-vs-Δ.

| benchmark | verdict |
|---|---|
| PathoROB RI | **TIE on phikon-v2** (101% of their Δ), LOSE on Midnight (90%) and Virchow2 (76%); shortfall localised entirely in camelyon |
| THUNDER classification | **WIN** — our Δ ≥ theirs on 11 of 12 shared task×backbone pairs; sign test 32/36, p ≈ 2×10⁻⁶ |
| THUNDER segmentation | LOSE on Midnight (−0.33 vs +1.6); flat-to-negative on all three backbones |
| THUNDER calibration | LOSE 1–2 (ECE degrades on phikon-v2 and Midnight, improves on Virchow2) |
| THUNDER adversarial | LOSE — the column is `drop/accuracy` (**lower is better**), so their Virchow2 31.1→7.7 is their *strongest* result, not a collapse. We improve on all three without trading clean accuracy, but nowhere near their margin |
| HEST | We improve on all three (see §4 — this was previously mis-measured) |
| Patho-Bench | **NOT MEASURABLE** — see §5 |

---

## 4. The start-metric discrepancy — SOLVED

**Root cause: Waiv's published HEST numbers use a DIFFERENT pooling protocol per backbone, and they never state it.**

Determined empirically by scoring each base under both poolings and keeping whichever reproduces their published base:

| backbone | Waiv's protocol | our base | their base | agreement |
|---|---|---|---|---|
| phikon-v2 | `cls` | 0.37470 | 0.3747 | exact |
| Midnight | `cls` | **0.39521** | **0.3952** | exact |
| Virchow2 | **`clsmean`** | 0.40327 | 0.4034 | 0.00013 |

Virchow2 under `cls` gives 0.39791 — off by 0.0055. **Do not assume one protocol across backbones**; assuming `cls` everywhere would manufacture a −0.0055 discrepancy on Virchow2 where none exists.

### 4.1 Why this mattered scientifically

Measuring Midnight under the wrong protocol **masked a real result**:

| protocol | base | FT | Δ | verdict |
|---|---|---|---|---|
| `clsmean` (wrong) | 0.41210 | 0.41322 | +0.0011 | "no result" |
| **`cls` (matched)** | **0.39521** | **0.4065** | **+0.0113 (n=5)** | **1.5× bar — marginal, ~53% of Waiv's +0.0215** |

[re-corrected 2026-08-26 (second pass): the multiplier first written here was **6.6x**, which is wrong. The 0.0075 bar is 2SE for an **n=5 mean** (RESULTS.md:1794 -- per-task SD 0.0084, SE = 0.0084/sqrt(5) = 0.0037, 2SE = 0.0075). +0.0113 / 0.0075 = **1.5x**. Note this also means the n=1 row above was always being graded against a five-seed bar: the single-run 2SD bar is 2 x 0.0084 = 0.0168, against which +0.0166 is **0.99x -- inside noise**, which is precisely why it did not replicate.]

[corrected 2026-08-26: the row previously read FT **0.41180**, Δ **+0.0166**, "2.2× bar — real, 77% of Waiv's +0.0215". That was a **single-seed** measurement (job 386398) that did **not** replicate at n=5. The 5-seed matched-protocol result is +0.0113 (n=5), ~53% of Waiv's +0.0215. Authority: `docs/FINAL5_RESULTS.md:138-142` and `docs/final5_results.json` (`aggregates.midnight.hest.delta_vs_base`).]

Under `clsmean` the base already sits near where fine-tuning lands, because the mean component supplies information the adapter would otherwise learn. The delta was being measured on an inflated base. **Same checkpoint, same benchmark, only the pooling changed — and the conclusion flipped from null to a real ~53% recovery.** [corrected 2026-08-26: was "a real 77% recovery"; 77% came from the single-seed job 386398 (+0.0166), which did not replicate. The n=5 matched-protocol delta is +0.0113, ~53% of Waiv's +0.0215; authority `docs/FINAL5_RESULTS.md:138-142`.] The earlier verdict "our fine-tuning preserves HEST; Waiv's improves it" is retracted.

### 4.2 Operational traps when testing this

1. `run_hest.py` keys its embedding cache on `--exp-code` **alone** — pooling, backbone and checkpoint are **not** in the key. A protocol test that reuses an exp_code silently rescores stale features and manufactures a null.
2. The HEST dataloader crashes in shared-memory teardown (`c10::Error: could not unlink the shared memory file`) at `--num-workers` 8 **and** 2. Use 0.

---

## 5. Patho-Bench — NOT MEASURABLE

One of the three components of Waiv's Figure-1 composite (`y = (58 − total)/53`, `total` = HEST rank + THUNDER rank + Patho-Bench rank). It cannot be measured for our encoder, for three independent reasons (documented at `src/waivphaet/eval/__init__.py:30-41`):

1. It is **slide-level**; the public precomputed features are **UNI2-h patch embeddings**, useless for scoring a different patch encoder.
2. A real number needs **~7–8 TB of raw TCGA WSIs** plus a full extraction pass.
3. **Waiv's own quoted 54.1 → 55.8 has no traceable source** — the Patho-Bench paper publishes no results table and there is no leaderboard, so there is nothing to reproduce even after paying that cost.

**Consequence for the chart:** a pixel-exact recreation of Waiv's Figure 1 is not attainable. `docs/waiv_figure1.html` therefore plots our models on the **two-component composite** (HEST + THUNDER, renormalised), with the inherited-rank estimate and best/worst bounds shown as visually subordinate, explicitly-flagged annotations. No Patho-Bench rank is invented.

---

## 6. Instrument reliability — what can and cannot rank arms

Define noise units `nu = spread across arms / 2SE seed floor`. Below ~2 nu a readout cannot order arms.

| readout | corrected 2SE | spread | nu | verdict |
|---|---|---|---|---|
| **RI** | ~0.007 | ~0.023 | **2–3** | **only usable ranking metric** |
| HEST 5-task | 0.0075 | 0.00406 | ~0.5 | blind |
| THUNDER LP cls | 0.0097 | 0.0095 | 0.98 | blind |
| THUNDER LP clsmean | 0.0087 | 0.0124 | 1.43 | blind |

[corrected 2026-08-26: the two THUNDER LP `corrected 2SE` values were **0.0156** and **0.0208**, which are 5-dataset, n=2-seed floors from `docs/thunder_seed_floor.md`; `nu` was 0.61 and 0.60. Replaced with the 12-dataset, n=5-seed floors — phikon/linear_probing **0.0097**, midnight/linear_probing **0.0087** — from `docs/thunder_seed_floor_12ds.md`, and `nu` recomputed. Both remain well under the ~2 nu bar, so the "blind" verdicts stand.]
| THUNDER kNN | 0.0297 / 0.0468 | ~0.006 / ~0.010 | ~0.2 | unusable |

Two results that should survive this project:

- **A pure seed replicate of the control outscored all nine real arms** on THUNDER LP clsmean (0.7259 vs best real arm 0.7186). A metric a no-op outranks is measuring seeds, not manipulations.
- **The legacy seed-floor construction was invalid.** `2·SD/√n` measures dispersion *about* the mean delta, so a consistent offset is not penalised at all. For clsmean/LP all five per-task deltas were positive with paired t p = 0.025 — the two seeds genuinely disagree. Use `offset_2se = |mean d| + 2·SD/√n`. Under the corrected bar every THUNDER protocol falls to 0.2–0.6 nu. Computation persisted at `scripts/thunder_seed_floor_12ds.py` → `docs/thunder_seed_floor_12ds.{json,md}` [corrected 2026-08-26: pointer was `scripts/thunder_seed_floor.py` → `docs/thunder_seed_floor.{json,md}`, the superseded 5-dataset/n=2 computation].

---

## 7. Outstanding at session end

All eval-only; no training budget required. Blocked on the Bash safety classifier being rate-limited.

1. **Release 28 held THUNDER jobs** (386596–386623) — full 4-task THUNDER for the phikon-v2 and midnight finals. They are `PENDING (JobHeldUser)` and will not start on their own:
   `scontrol release 386596 ... 386623`
2. **Virchow2 full THUNDER** on `runs/splitgrid-386375/step_0000250` under fresh run names (`vft250sg_clsmean` / `vft250sg_cls`). **Must not reuse `vft250_*`** — those features date from 2026-08-10 and belong to the published-reconstruction checkpoint. Both are step 250, which is how a prior agent wrongly concluded it was already done.
3. **Virchow2 HEST** on that checkpoint at `clsmean`, fresh exp_code, `--num-workers 0`.
4. **Regenerate `docs/waiv_figure1.html`** once 1–3 land.

Note: fast-5 THUNDER (5 datasets × {knn, LP}) is the right instrument for *arm comparison* but cannot feed the chart, whose rank uses Waiv's 4-task mean (knn, linear, few_shot, segmentation) over 12+4 datasets.
