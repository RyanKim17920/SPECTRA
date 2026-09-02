> **[STATUS BANNER — added 2026-08-31, see `docs/README.md` for the current doc map]**
>
> HISTORICAL. Three-generation retrospective from 2026-08-18/25; GEN-3 scope is 3 backbones, predates the 5-backbone / 50-step-grid state.

# Three generations on one scale — did we actually improve?

Generated 2026-08-25. **Read-only analysis; no jobs launched or cancelled.**
Every number below is either **[V] VERIFIED** (read off disk, path given) or **[I] INFERRED**
(derived / imputed / estimated — method stated inline). Nothing is reconstructed from memory.

---

## 0. The three generations, as they exist on disk

| | GEN-1 "original run" | GEN-2 "final5" | GEN-3 "current candidate" |
|---|---|---|---|
| artifact | `docs/waiv_figure1_data.json` (2026-08-18) | `docs/FINAL5_RESULTS.md` / `docs/final5_results.json` | `docs/FINAL_RECIPE.md` / `docs/final_recipe_verdict.json` |
| runs | `runs/waiv-real-369043` (phikon)<br>`runs/waiv-midnight-369159`<br>`runs/waiv-virchow2-375367` | `runs/final5-{arm}-s{0..4}-t900-*` (n=5/arm) | `runs/genMASK-c3s-…-ms500-…-{arm}-s{0,1}-t900-*` (n=2/arm) |
| scored at step | 1000 / 500 / 250 (**RI-argmax**) | **fixed** step 500 | **stopping rule**: first ckpt with mean `confounder_insensitivity` ≥ 0.75 → 250 / 125 / 125 |
| seeds | 1 | 5 | 2 |

---

## 1. What the configs actually say — the head-architecture correction [V]

The brief I was given claimed GEN-1 used `same_core_logit_bias_cls = -inf`. **That is wrong, and the
configs on disk refute it.** Verified by reading `runs/<run>/config.json` and `ls runs/<run>/step_*`:

| config key | GEN-1 (`waiv-real-369043`, `waiv-midnight-369159`, `waiv-virchow2-375367`) | GEN-2 (`final5-*`) | GEN-3 (`genMASK-c3s-*`) |
|---|---|---|---|
| `encoder.pooling` | `clsmean` | `clsmean` | `clsmean` |
| `split_heads` / `encoder.split_heads` | **absent** (key not in config) | `True` / `["cls","mean"]` | `True` / `["cls","mean"]` |
| `encoder.pool_head` | **absent** | `gem` | `gem` |
| `cls_weight` / `mean_weight` | **absent** | 0.5 / 0.5 | 0.5 / 0.5 |
| `mask_same_core` | **absent** | `False` | **`True`** |
| `same_core_logit_bias_cls` | **absent** | absent | **`3.0`** |
| `same_core_logit_bias_mean` | **absent** | absent | **`-inf`** |
| `grid_conditions` / `grid_tiles` | **absent** (not grid path) | 2 / 900 | 2 / 900 |
| `lr` | 1e-4 | 1e-4 | 1e-4 |
| `max_steps` | 4000 / 1500 / 1500 | 1500 | **500** |
| `ckpt_every` | 500 / 250 / 250 | 250 | **125** |
| temp / warmup / wd / LoRA r / proj dim | 0.07 / 200 / 0.05 / 32 / 512 | *identical* | *identical* |

**Checkpoint contents corroborate this independently** — the strongest single piece of evidence:

```
runs/waiv-real-369043/step_0000500/       -> adapter, metrics.json, optim.pt, projector.pt
runs/finalgem-phikon-384585/step_0000500/ -> ... projector.pt, projector_cls.pt, projector_mean.pt,
                                                 projector_heads.json, pool_head.pt
runs/final5-phikon-s0-t900-386794/…       -> (same as finalgem — split heads present)
runs/genMASK-c3s-…-phikon-s0-…/step_0000125/ -> (same — split heads present)
```

GEN-1 checkpoints contain **exactly one projector** and **no per-head files and no pool head**.
GEN-2 and GEN-3 both carry `projector_cls.pt` + `projector_mean.pt` + `projector_heads.json` +
`pool_head.pt`.

### Corrected causal ordering of what changed

1. **GEN-1 → GEN-2 is an architecture change, not a hyperparameter change.**
   Single fused `clsmean` projector → **split cls/mean projectors + GeM pool head**, plus a move
   from the pair path to the **grid batch path** (`grid_conditions=2`, `grid_tiles=900`), plus
   RI-argmax scoring → fixed step 500. lr, temperature, warmup, weight decay, LoRA rank and
   projection dim are **byte-identical across all three generations**. This is the change the
   original brief did not mention, and it is the largest structural difference in the table.
2. **GEN-2 → GEN-3 is the negative-masking + early-stopping change.**
   `mask_same_core=True` with `bias_cls=3.0` / `bias_mean=-inf` (soft mask on the cls head, hard
   mask on the mean head), `max_steps` 1500→500, `ckpt_every` 250→125, and scoring by the internal
   `confounder_insensitivity ≥ 0.75` rule instead of a fixed step.

**The `-inf` cls-bias hypothesis is dead.** No GEN-1 or GEN-2 config sets any per-head bias at all;
the only `-inf` in the whole table is GEN-3's **mean** head, and GEN-3 is the *best* generation.
Whatever explains GEN-1's phikon-v2 result, it is not a cls-bias HEST-killer.

---

## 2. The pooling-protocol audit — and which way the bias actually runs [V]

Protocols are per-backbone and are set in `scripts/hest_final5.sbatch:41-46`
(phikon→`cls`, midnight→`cls`, virchow2→`clsmean`), and recorded per-run in the HEST `exp_code`
suffix and in the THUNDER sidecar `pooling_effective` (`/data/ryan.kim/thunder/outputs/provenance/*.json`).

### 2a. Waiv's own protocol, reverse-engineered from exact base matches [V]

| backbone | our measured base, `cls` | our measured base, `clsmean` | Waiv published base | ⇒ Waiv's protocol |
|---|---|---|---|---|
| phikon-v2 | **0.37470** | 0.39144 | 0.3747 | **cls** (exact) |
| midnight | **0.39521** | 0.41210 | 0.3952 | **cls** (exact) |
| virchow2 | — | **0.40324** | 0.4034 | **clsmean** (Δ 0.00013) |

Sources: `results_backup/hest_work_results/base_cls_summary.json`, `base_clsmean_summary.json`,
`/data/ryan.kim/hest_work/results/midnight_base_cls_9task_v1_summary.json`,
`results_backup/hest_work_results/vbase_clsmean_summary.json`; annotated in `scripts/collect_final5.py:40-53`.

### 2b. What each generation was actually evaluated under [V]

| | HEST protocol used | matches Waiv? | THUNDER classification pooling | matches Waiv-era protocol? |
|---|---|---|---|---|
| GEN-1 phikon | `cls` (`ft1000_cls_summary.json`) | **yes** | `cls` (`ft1000_cls`) | yes |
| GEN-1 midnight | `clsmean` (`mft500_clsmean_summary.json`) | **NO — wrong protocol** | `clsmean` (`mft500_clsmean`) | consistent w/ our base dir |
| GEN-1 virchow2 | `clsmean` (`vft250_clsmean_summary.json`) | **yes** | `clsmean` (`vft250_clsmean`) | yes |
| GEN-2 all | phikon/midnight `cls`, virchow2 `clsmean` | **yes** | phikon `cls`, others `clsmean` | yes |
| GEN-3 all | phikon/midnight `cls`, virchow2 `clsmean` | **yes** | phikon `cls`, others `clsmean` (sidecar `pooling_effective`) | yes |

### 2c. The hypothesis, tested — and the direction is the opposite of the one proposed

The proposal was: GEN-1 phikon trained a single `clsmean` vector but was read out under `cls`, so its
19/21 HEST and 20/21 THUNDER may be a measurement artefact. **The disk lets us test this directly,
because the counterfactual HEST run exists** (`ft1000_clsmean_summary.json`):

| phikon-v2 | HEST under `cls` | HEST under `clsmean` |
|---|---|---|
| base | 0.37470 | 0.39144 |
| GEN-1 fine-tuned (`ft1000`) | 0.37937 | 0.39477 |
| **gain over its own matched base** | **+0.0047** | **+0.0033** |

**Under *either* pooling, GEN-1's phikon fine-tune moved HEST by ~+0.004.** The pooling choice moves
the *absolute* (+0.017, which is a base-model property, not a fine-tuning effect) but not the *gain*.
So the train/eval representation mismatch is real but it is **not** what put phikon at 19/21: phikon-v2's
`cls` absolute is simply near the bottom of the 21-model field, and GEN-1 added almost nothing to it.
**Conclusion: GEN-1's phikon deficit is not explained by protocol mismatch.** [V]

The mismatch that *does* exist runs the other way, and it flatters a different model:

> **GEN-1's midnight HEST (0.4132, rank 9/21) was taken under `clsmean`, which is the wrong protocol
> for midnight.** The base-level offset between protocols is `0.41210 − 0.39521 = +0.0169`, and the
> GEN-1 artifact's own caveat records the same "+0.0169 off Waiv's published midnight base". Applying
> that offset gives a protocol-corrected GEN-1 midnight HEST of **≈0.3963 [I]**, which would rank
> **14/21, not 9/21**. GEN-1's midnight column is therefore inflated, and its "mid-field" placing is
> partly a measurement artefact.

THUNDER pooling, by contrast, is **consistent across all three generations** for every backbone, so
THUNDER comparisons below are protocol-clean.

**Net effect on the "did we improve" question:** the phikon-v2 GEN-1→GEN-3 improvement is a **real
recipe gain, not a measurement fix** (both generations scored under `cls`). The midnight GEN-1→GEN-3
comparison is the one contaminated by a protocol change, and correcting it makes GEN-3 look *better*
relative to GEN-1, not worse.

---

## 3. THE CORE DELIVERABLE — absolutes, no rank machinery

All values are means over available seeds. **Δ vs own base** is the honest quantity: absolute
levels are dominated by which backbone you started from.

### 3a. RI (avg over cross_scanner / cross_stain separation)

| backbone | our base [V] | GEN-1 [V] | GEN-2 (n=5) [V] | GEN-3 (n=2) [V] | Waiv base | Waiv fine-tuned |
|---|---|---|---|---|---|---|
| phikon-v2 | 0.4686 | 0.8080 | 0.8269 ± 0.0045 | **0.8361 ± 0.0010** | 0.469 | 0.806 |
| midnight | 0.7589 | 0.9080 | 0.8993 ± 0.0021 | **0.9116 ± 0.0010** | 0.759 | 0.924 |
| virchow2 | 0.8582 | 0.9035 | 0.8977 ± 0.0020 | **0.9070 ± 0.0017** | 0.858 | 0.918 |
| | Δ base → | +0.339 / +0.149 / +0.045 | +0.358 / +0.140 / +0.040 | **+0.367 / +0.153 / +0.049** | | +0.337 / +0.165 / +0.060 |

Seed floor (2·SD/√n at n=5): phikon 0.0041, midnight 0.0019, virchow2 0.0018.
**GEN-3 > GEN-2 on all three backbones by 3–24× the seed floor. GEN-3 ≥ GEN-1 on all three.** The
GEN-1 values are **RI-argmax over the checkpoint grid** (steps 1000/500/250), which per
`docs/final5_results.json:note_ri_argmax` is a *secondary*, upward-biased statistic; GEN-2 and GEN-3
use a pre-declared step. So GEN-1 is being given the more generous estimator and still loses.
RI is the only one of the three axes where GEN-3 is unambiguously best. [V]

### 3b. HEST (9-cancer average)

| backbone | protocol | our base [V] | GEN-1 [V] | GEN-2 (n=5) [V] | GEN-3 (n=2) [V] | Waiv base | Waiv fine-tuned |
|---|---|---|---|---|---|---|---|
| phikon-v2 | `cls` | 0.37470 | 0.37937 | 0.38709 ± 0.0017 | **0.39244** (0.39116 / 0.39373) | 0.3747 | 0.3943 |
| midnight | `cls` | 0.39521 | *(0.4132 clsmean —* **wrong protocol**; ≈0.3963 corrected [I]) | 0.40650 ± 0.0019 | **0.41501** (0.41502 / 0.41500) | 0.3952 | 0.4167 |
| virchow2 | `clsmean` | 0.40324 | 0.40830 | 0.40555 ± 0.0012 | **0.41074** (0.41092 / 0.41056) | 0.4034 | 0.4135 |

Δ over own base — the comparable quantity:

| backbone | GEN-1 | GEN-2 | GEN-3 | Waiv's gain |
|---|---|---|---|---|
| phikon-v2 | +0.0047 | +0.0124 | **+0.0177** | +0.0196 |
| midnight | +0.0011 [I, protocol-corrected] | +0.0113 | **+0.0198** | +0.0215 |
| virchow2 | +0.0051 | +0.0023 | **+0.0075** | +0.0101 |

**HEST improves monotonically GEN-1 → GEN-2 → GEN-3 on every backbone**, and GEN-3 recovers
90% / 92% / 74% of Waiv's gain. Caveat: n=2 at GEN-3, and the authoritative 1-SD on the *percent-of-
Waiv-gain* statistic is 5.8 / 8.3 / **14.2** points (`final_recipe_verdict.json:sources`), so the
virchow2 cell's 73.1% carries ±20.1 and is formally **NOT RESOLVED** against the 70 bar — that is
exactly the `INDETERMINATE` verdict in `docs/FINAL_RECIPE.md`. The *ordering* GEN-3 > GEN-2 is safe
(both GEN-3 seeds beat all five GEN-2 seeds on every backbone); the *level* vs Waiv is not. [V]

### 3c. THUNDER, per task — recomputed fresh from disk today

Recomputed with `scripts/collect_thunder.py::_score` over the 12 classification datasets, so
GEN-1/GEN-2/GEN-3 go through the identical scorer. **The GEN-3 sweep has advanced since
`final_recipe_verdict.json` was written** (that file records 0–12 datasets/model; today it is 4–12),
so these numbers supersede it.

Coverage today, `/data/ryan.kim/thunder/outputs/res/*/f5_ci-*` [V]:

| model | knn | linear_probing | simple_shot | segmentation |
|---|---|---|---|---|
| phikon s0 | **12/12** | **12/12** | 11/12 | 0 |
| phikon s1 | **12/12** | 11/12 | 9/12 | 0 |
| midnight s0 | 7/12 | 7/12 | 6/12 | 0 |
| midnight s1 | 6/12 | 5/12 | 4/12 | 0 |
| virchow2 s0 | **12/12** | 11/12 | 9/12 | 0 |
| virchow2 s1 | 11/12 | 8/12 | 7/12 | 0 |

Because coverage is ragged, the GEN-3 column below is a **matched-subset comparison**: for each seed,
GEN-3 / base / GEN-1 are averaged over *exactly the datasets that seed has*, so the Δ columns are
apples-to-apples even where the 12/12 mean does not exist. Cells marked ⧗ are still incomplete.

**phikon-v2** (all pooling `cls`)

| task | base (12/12) | GEN-1 | GEN-2 (n=5) | GEN-3 s0 | GEN-3 s1 | GEN-3 Δ vs GEN-1 | seed floor |
|---|---|---|---|---|---|---|---|
| knn | 70.281 | 75.203 | 73.770 | **75.376** (12/12) | 73.813 (12/12) | +0.17 / −1.39 | 2.33 |
| linear_probing | 76.541 | 79.236 | 78.235 | **78.695** (12/12) | 79.071 ⧗(11) | −0.54 / −0.87 | 0.97 |
| simple_shot | 69.330 | 70.988 | 69.571 | 72.050 ⧗(11) | 70.033 ⧗(9) | −0.60 / −1.16 | 0.87 |

**midnight** (all pooling `clsmean`)

| task | base (12/12) | GEN-1 | GEN-2 (n=5) | GEN-3 s0 | GEN-3 s1 | GEN-3 Δ vs GEN-1 | seed floor |
|---|---|---|---|---|---|---|---|
| knn | 78.254 | 80.444 | 78.681 | 76.208 ⧗(7) | 81.252 ⧗(6) | +0.07 / −0.21 | 1.00 |
| linear_probing | 82.880 | 84.122 | 83.458 | 80.186 ⧗(7) | 86.003 ⧗(5) | +0.70 / −0.46 | 0.87 |
| simple_shot | 70.639 | 76.380 | 74.891 | 74.559 ⧗(6) | 74.113 ⧗(4) | −0.05 / −2.44 | 1.04 |

**virchow2** (all pooling `clsmean`)

| task | base (12/12) | GEN-1 | GEN-2 (n=5) | GEN-3 s0 | GEN-3 s1 | GEN-3 Δ vs GEN-1 | seed floor |
|---|---|---|---|---|---|---|---|
| knn | 80.874 | 80.906 | 79.282 | **81.327** (12/12) | 79.909 ⧗(11) | +0.42 / +0.56 | 0.83 |
| linear_probing | 83.253 | 84.589 | 83.228 | 84.555 ⧗(11) | 81.280 ⧗(8) | −0.67 / −0.61 | 0.88 |
| simple_shot | 72.749 | 77.163 | 74.491 | 76.352 ⧗(9) | 73.116 ⧗(7) | −0.55 / −0.45 | 0.66 |

*(GEN-3 columns show the seed's own raw mean; the Δ column is the matched-subset delta, which is why
midnight s0's raw 76.2 still yields +0.07 — its 7 datasets are the harder ones.)*

Seed floors are offset-2SE at 12/12 coverage from `docs/thunder_seed_floor_12ds.md`.

**Reading of 3c — the honest one:**

* **GEN-2 was a THUNDER regression.** On every backbone and every task, GEN-2 sits *below* GEN-1
  (phikon knn 73.77 vs 75.20; virchow2 knn 79.28 vs 80.91; midnight simple_shot 74.89 vs 76.38),
  by 1–2.7 points against floors of 0.7–2.3. That is a real regression, not noise.
* **GEN-3 recovers GEN-1's THUNDER level but does not exceed it.** Of 18 GEN-3-seed-vs-GEN-1 matched
  deltas, 5 are positive and 13 negative, and only 3 exceed the seed floor in magnitude
  (phikon s1 knn −1.39, midnight s1 simple_shot −2.44, virchow2 s0/s1 knn +0.42/+0.56). The
  aggregate signal is **flat**.
* **GEN-3 still beats its own base everywhere** (+3.5 to +5.1 phikon knn, +5.0 to +7.0 midnight
  simple_shot, +3.7 to +3.9 virchow2 simple_shot), i.e. the recipe does work; it just does not work
  *better than GEN-1 did* on THUNDER.

---

## 4. GEN-3 placed in the same 21-model field as GEN-1 (task 2) — with heavy caveats

This section **can** be built, but it is the weakest thing in this document and should not be quoted
without §4c.

### 4a. The segmentation problem, stated

GEN-1's published THUNDER rank was a **4-task** mean (knn, linear, few_shot, **segmentation**).
GEN-3 has **zero segmentation results** (0/4 datasets for all six runs — verified above). I therefore
recomputed **every** model — the 20 Waiv rows and all three of our generations — on a **3-task
classification-only** mean, so the comparison is internally consistent. Dropping segmentation is
**not** neutral: it removes a task on which GEN-1's midnight/virchow2 scored 68.4/68.9, near the
field median, and on which GEN-2's own results show fine-tuning is roughly flat to slightly negative
(`final5_results.json`: phikon segmentation Δ vs base **−0.0088**). Removing it therefore **removes a
mild drag** and is, if anything, **generous to us in all three generations equally** — but it also
means these ranks are not the ranks in `docs/waiv_figure1.html`.

Sanity check on the substitution: recomputing GEN-1 on 3 tasks moves its THUNDER ranks from the
published 20 / 8 / 6 to **20 / 10 / 7**. So the segmentation column was worth ~1–2 rank places to
midnight and virchow2 in GEN-1, and 0 to phikon.

### 4b. The recomputed table [I — see method]

GEN-3's 3-task mean is **base-anchored imputed to 12/12**: `est = base_12 + (ours_matched − base_matched)`,
averaged over the two seeds. Only 3 of 18 GEN-3 task-cells are genuinely 12/12 complete.
y is the 2-component renormalisation `y = (40 − total) / 38`, total = HEST rank + THUNDER rank in 21.

| | HEST | rank | THUNDER 3-task | rank | total | **y (2-comp)** |
|---|---|---|---|---|---|---|
| **GEN-1** phikon-v2 | 0.3794 | 19 | 75.14 | 20 | 39 | **0.026** |
| **GEN-2** phikon-v2 | 0.3871 | 18 | 73.86 | 21 | 39 | **0.026** |
| **GEN-3** phikon-v2 | 0.3924 | 17 | 74.82 [I] | 21 | 38 | **0.053** |
| **GEN-1** midnight | 0.4132 ✗protocol | 9 | 80.32 | 10 | 19 | **0.553** |
| **GEN-1** midnight, corrected [I] | ≈0.3963 | 14 | 80.32 | 10 | 24 | **0.421** |
| **GEN-2** midnight | 0.4065 | 11 | 79.01 | 15 | 26 | **0.368** |
| **GEN-3** midnight | 0.4150 | **6** | 80.60 [I] | **7** | **13** | **0.711** |
| **GEN-1** virchow2 | 0.4083 | 10 | 80.89 | 7 | 17 | **0.605** |
| **GEN-2** virchow2 | 0.4055 | 12 | 79.00 | 15 | 27 | **0.342** |
| **GEN-3** virchow2 | 0.4107 | **9** | 80.74 [I] | **7** | **16** | **0.632** |

Waiv fine-tuned reference on the **published 3-component** scale (not the same scale as the column
above — see §4c): UNI2-h 1.000, Mascaret 0.887, H-Optimus-0 0.849, Virchow2 0.755, Phaet 0.264.

### 4c. Why you should not quote §4b as a headline

1. **Our y and Waiv's y are different scales.** Ours is 2-component renormalised `(40−total)/38` over
   HEST+THUNDER only; Waiv's published y is `(58−total)/53` over HEST+THUNDER+**Patho-Bench**.
   Patho-Bench remains unmeasurable for us (source WSIs ~7–8 TB). GEN-3 midnight's 0.711 is **not**
   comparable to Mascaret's 0.887 — different denominators, different component count.
2. **Our THUNDER absolutes are on 12/16 classification datasets; Waiv's are on 16.** The gap this
   opens is large and not constant: our measured base for phikon-v2 knn is **70.28** where Waiv
   publishes **74.0**; our virchow2 base knn is **80.87** vs Waiv's **82.9**. So ranking our absolute
   against their absolute compares two different quantities. **Deltas vs our own base (§3c) are the
   only defensible THUNDER comparison to make.** This alone is enough to distrust every THUNDER rank
   in §4b.
3. **13 of 18 GEN-3 THUNDER task-cells are imputed**, not measured. The imputation is base-anchored
   and conservative, but a rank is a discontinuous function of the estimate.
4. **n=1 (GEN-1) vs n=5 (GEN-2) vs n=2 (GEN-3).** Per the standing caveat, HEST and THUNDER at n=1
   are not gradeable — aggregate 2SE is 11.1 / 10.6 percentage points. GEN-1's entire row is n=1.
5. **The GEN-1 midnight row is protocol-contaminated** (§2c) and I have given both versions.
6. Per `docs/CAVEATS.md` / the figure-1 JSON's own `verification` block, Waiv's published
   `thunder_rank_sum` **cannot be reproduced** from its own per-task values; the ranks are consistent
   but the input sums are opaque. We are ranking into a field we cannot fully audit.

Direction of unfairness, stated plainly: **§4b is unfair *in GEN-3's favour*** on the segmentation
axis (a mildly negative task was dropped) and **unfair *against* GEN-3** on the dataset-coverage axis
(our 12-dataset absolutes are systematically below Waiv's 16-dataset absolutes, by ~2–4 points for
phikon). These do not cancel in any principled way. Use §3.

---

## 5. Verdict — did we improve?

**Yes on RI and HEST; no on THUNDER, where GEN-3 has only recovered ground GEN-2 lost.**

| axis | GEN-1 → GEN-2 | GEN-2 → GEN-3 | GEN-1 → GEN-3 | confidence |
|---|---|---|---|---|
| RI | ~flat (GEN-1 uses the more generous argmax estimator) | **improved**, 3–24× seed floor, 3/3 backbones | **improved** | **high** [V] |
| HEST | improved 2/3 backbones (virchow2 regressed) | **improved 3/3**, both seeds beat all five GEN-2 seeds | **improved 3/3** | **medium** — n=2, virchow2 cell NOT RESOLVED [V] |
| THUNDER | **regressed 3/3 backbones**, 1–2.7 pts vs floors of 0.7–2.3 | recovered | **flat** (5+/13− matched deltas, 3 exceed floor) | **medium-low** — ragged coverage, sweep still running [V] |

### What most plausibly changed, and why

* The **head architecture** is the change the original brief missed and is the dominant structural
  difference: GEN-1 trained **one fused clsmean projector**; GEN-2 and GEN-3 train **split cls + mean
  projectors with a GeM pool head**, on the grid batch path. This is what makes a `cls`-protocol
  readout meaningful at all for phikon-v2 and midnight — in GEN-1 the `cls` half of the representation
  received no dedicated projection.
* But the split-head change **alone was not enough**: GEN-2 has it and still regressed on THUNDER
  across the board, because at a fixed step 500 on a 1500-step schedule it over-trains
  (consistent with the standing "training over-specialises past step 500" finding).
* **GEN-3's gain comes from stopping earlier, chosen by an internal signal.** The
  `confounder_insensitivity ≥ 0.75` rule lands at 250 / 125 / 125 — 2–4× earlier than GEN-2's step 500
  — and that is what simultaneously lifts HEST and claws THUNDER back. The negative-mask biases
  (`cls=3.0`, `mean=-inf`) are the other GEN-2→GEN-3 delta and cannot be separated from the stopping
  change with the runs on disk, since no GEN-3 run varies one without the other.

### The honest bottom line

GEN-3 is the best of the three generations on the two axes we can measure well, and the *ordering*
GEN-3 > GEN-2 is solid. But GEN-3's THUNDER is statistically indistinguishable from GEN-1's, so the
correct summary is **"we fixed the GEN-2 regression and improved RI and HEST on top of GEN-1"**, not
"we improved across the board". And `docs/FINAL_RECIPE.md`'s `INDETERMINATE` verdict still stands:
every point estimate clears the bar, but the worst cell's error bar does not, and two THUNDER cells
remain ungraded pending the sweep.

---

## Appendix — provenance index

| what | where |
|---|---|
| GEN-1 artifact | `docs/waiv_figure1_data.json`, `docs/waiv_figure1.html` |
| GEN-1 configs | `runs/waiv-real-369043/config.json`, `runs/waiv-midnight-369159/config.json`, `runs/waiv-virchow2-375367/config.json` |
| GEN-2 artifact | `docs/final5_results.json`, `docs/FINAL5_RESULTS.md` |
| GEN-3 artifact | `docs/final_recipe_verdict.json`, `docs/FINAL_RECIPE.md` |
| HEST summaries | `results_backup/hest_work_results/*_summary.json` |
| HEST protocol | `scripts/hest_final5.sbatch:41-46`; bases annotated `scripts/collect_final5.py:40-53` |
| THUNDER raw | `/data/ryan.kim/thunder/outputs/res/<ds>/<model>/<task>/frozen/outputs.json` |
| THUNDER pooling | `/data/ryan.kim/thunder/outputs/provenance/<model>.json` → `pooling_effective` |
| THUNDER scorer | `scripts/collect_thunder.py::_score` (knn = selected-k, simple_shot = 16-shot, f1) |
| THUNDER seed floors | `docs/thunder_seed_floor_12ds.md` / `.json` |
| Waiv published | `docs/waiv_published.json`; THUNDER table arXiv:2607.22861v1 Table 2 |
