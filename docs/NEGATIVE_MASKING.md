# Negative masking, all-91 conditions, and batch geometry — phikon seed 0

Status: **complete at n=1 per arm.** All arms are phikon-v2, seed 0, T=900, C=2 unless
stated, and byte-identical to the `final5` recipe except for the one flag named.
Baseline throughout is `final5-phikon-s0-t900-386794` (the SAME seed, not the 5-seed mean).

Floors, measured at n=5 on final5: **RI 0.00405**, **HEST 0.00152**.

---

## 1. What was tested and why

PLISM is ONE tissue microarray: 46 human tissue types, 16,278 registered tiles, imaged 91
ways (13 stains x 7 scanners). The grid InfoNCE treats every non-matching tile as a negative,
so ~1/46 of each anchor's 899 negatives are the SAME organ. The hypothesis was that these are
false negatives and that removing them would help.

Three interventions, then a dose-response sweep:

| arm | change |
|---|---|
| `all91` | empty heldout split -> 91 training conditions instead of 50 |
| `falseneg` | `--mask-same-core`: mask all same-tissue negatives (2.41% of pairs) |
| `combined` | both |
| `combc4` | `combined` at C=4/T=450 (same batch budget B=1800) |
| `sim025/050/100` | `--mask-sim-thresh`: mask only negatives whose frozen-Virchow2 cosine to the anchor exceeds tau |

## 2. The tile -> tissue map

No organ metadata ships with PLISM (the Owkin derivative carries only `(level, x, y)`;
upstream Ochi et al. 2024 has none per-tile). It was derived instead:

1. 8-connectivity on the ROW/COL occupancy lattice -> **49 spatially separated cores**.
   The cores are physically separated on the slide, so this is exact and model-free.
2. Mean Virchow2 embedding per core, agglomerative merge -> **46 tissues**
   (1.07 cores/tissue, matching the published 46 exactly).

Measured false-negative rate: **2.393%** of all pairs, vs the 1/46 = 2.17% estimate.

A first attempt (phikon k-means, k=30) was discarded: k was chosen by max raw contiguity,
which trivially favours fewer clusters, and it merged ~16 organs. Backed up as
`.plism_core_labels_kmeans30.npy.bak`.

Caveat worth stating: the reported contiguity of 1.0 vs a 0.0245 random-label null is
**trivially true**, since the labels derive from connected components. The real validation is
the 49 -> 46 agreement with the dataset's documented design.

---

## 3. RI dose-response — masked fraction sets the LEVEL

| step | 0% (base) | 0.25% | 0.50% | 1.00% | 2.41% |
|---|---|---|---|---|---|
| 250 | 0.82407 | 0.82141 | 0.82130 | 0.82058 | 0.81547 |
| 500 | 0.83384 | 0.83667 | 0.83895 | 0.84309 | 0.84507 |
| 750 | 0.82830 | 0.83230 | 0.83645 | 0.84070 | **0.85331** |
| 1000 | 0.82671 | 0.82963 | 0.83281 | 0.83731 | 0.85290 |

Strictly monotonic in masked fraction at every step from 500 on. At step 500 the span is
+0.0028 -> +0.0112, i.e. 0.7x -> 2.8x the floor.

## 4. Only near-complete masking changes the SHAPE

| masked | delta RI 500->750 |
|---|---|
| 0% | -0.0055 |
| 0.25% | -0.0044 |
| 0.50% | -0.0025 |
| 1.00% | -0.0024 |
| 2.41% | **+0.0082** |

Every partial dose still decays like the baseline; only whole-core masking inverts decay into
continued improvement. So **degradation past step 500 is driven by same-tissue negatives as a
class** — the residual 1.4% left unmasked at tau=0.7625 is still enough to cause it. The most
similar pairs are not the harmful ones; the bulk of them are.

Confirmed to step 3000 (`falseneg3k`): RI runs 0.85317 @750 -> 0.85468 @3000, i.e. a flat
plateau, not continued climbing. **Masking removes the decline; it does not extend learning.**

---

## 5. HEST — the cost is flat, then jumps

All at step 500, vs the same-seed baseline 0.38823. Floor 0.00152.

| masked | RI delta | HEST delta | RI gained per unit HEST lost |
|---|---|---|---|
| 0.25% | +0.0028 | -0.0022 | 1.3 |
| 0.50% | +0.0051 | -0.0022 | 2.3 |
| 1.00% | +0.0093 | **-0.0019** | **4.8** |
| 2.41% | +0.0112 | -0.0051 | 2.2 |

HEST cost is roughly constant at ~-0.002 across 0.25-1%, then jumps 2.6x at 2.41%. The
exchange rate is therefore best at **tau = 0.7625 (1% masked)**: 83% of the RI gain for 38%
of the HEST cost.

**But every masking level loses on HEST.** The best is -0.0019 (1.3x floor). Nothing here
improves the external readout.

The cost also shrinks with training: falseneg is -0.0051 @500, -0.0026 @1500, and
falseneg3k is **-0.0012 @1500 (inside the floor, i.e. null)**. So at the checkpoint one would
actually ship, masking buys +0.025 RI at no *measurable* HEST cost — but neutral, not positive.

---

## 6. Null results

**all91 is null on both readouts.** RI -0.0006 vs same-seed baseline at step 500; HEST
+0.0004. Training on 91 conditions rather than 50 buys nothing measurable.

**Batch geometry is null at fixed budget.** `combc4` (C=4/T=450) vs `combined` (C=2/T=900) at
step 1500: 0.85234 vs 0.85253, a gap of 0.0002 = 5% of the floor. Six times the
condition-pair coverage and half the negatives produce the same result, so all91's null is
NOT a coverage artifact.

**Masking carries the whole effect in `combined`.** falseneg 0.85428 vs combined 0.85253 at
step 1500 — 0.00175 apart, inside the floor, i.e. indistinguishable.

---

## 7. GEM replicates independently

`fnnogem` = `falseneg` with `--pool-head mean` instead of `gem`, all else identical.

| step | GEM | mean | delta |
|---|---|---|---|
| 250 | 0.81547 | 0.81347 | +0.00201 |
| 500 | 0.84507 | 0.84328 | +0.00179 |
| 750 | 0.85331 | 0.85180 | +0.00151 |
| 1000 | 0.85200 | 0.85001 | +0.00200 |
| 1250 | 0.85358 | 0.85157 | +0.00201 |
| 1500 | 0.85428 | 0.85243 | +0.00185 |

**6/6 positive, mean +0.00186.** The prior evidence was +0.00157 over 17/17 paired deltas on
the unmasked pair path. Two independent experiments, different negative regimes, agreeing to
within 0.0003. Every individual delta is inside the 0.00405 floor, so this is only visible
paired — as it was originally.

---

## 8. Standing conclusion

Masking same-tissue negatives is a **large, real, dose-dependent RI effect** (+0.0112 at
2.41%, 2.8x floor) that also eliminates the RI turnover. It does **not** transfer to HEST: the
cost is negative at every dose, shrinking to null at late checkpoints but never positive.

The mechanism appears to be that same-organ tiles are the **hard negatives**. Removing them
makes the task easier, which yields a coarser representation: better at collapsing acquisition
variation (RI) and no better at resolving biology (HEST). Core-identity masking cannot
separate near-duplicates from genuinely different tissue within one organ, and the similarity
sweep shows the harm is not concentrated in the most-similar pairs.

Not adopted into the recipe. What would change that: seeds at tau=0.7625 to establish whether
-0.0019 HEST is real or noise, and a THUNDER readout, which was never run for these arms.

## 9. Provenance

Runs: `all91-…-387667`, `falseneg-…-387704`, `combined-…-387705`, `combc4-…-387817`,
`falseneg3k-…-387815`, `combined3k-…-387816`, `fnnogem-…-388980`, `sim025/050/100-…-388984/5/6`.
Pin: `waiv-snapshots/falseneg-pinned`. Core map: `scripts/derive_core_map_v2.py`,
labels at `runs/.plism_core_labels.npy`, embeddings at `runs/.plism_ref_emb.npy`.

All numbers are **n=1 per arm**. Floors are borrowed from the n=5 final5 measurement at the
matching backbone and step.
