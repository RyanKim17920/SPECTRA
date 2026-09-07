> **[STATUS BANNER — added 2026-08-31, see `docs/README.md` for the current doc map]**
>
> HISTORICAL / PARTIAL. Covers only 3 of the current 5 backbones (phikon-v2, midnight, Virchow2 — H-Optimus-0 and UNI2-h are not included). Do not treat as the final cross-backbone table; see RUNBOOK.md and RESULTS.md for current scope.

# final5 — three backbones, one recipe, five seeds

Status: **COMPLETE — RI, HEST and THUNDER all at n=5, full coverage.**
Generated from `docs/final5_results.json` via `scripts/collect_final5.py`.

This supersedes the three-backbone table in `FINAL_RESULTS.md` §2, which is confounded —
see §6.

---

## 1. What was held identical

15 runs = 3 backbones × 5 seeds. `collect_final5.py` verifies **16 config keys** are
identical across all 15 and excludes any run that differs; it reports
**"15 comparable runs, 0 excluded"**.

```
--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
--grid --grid-conditions 2 --grid-tiles 900 --grid-forward-chunk 0
--lora-rank 32 --lora-alpha 64 --lr 1e-4 --temperature 0.07 --warmup-steps 200
--max-steps 1500 --ckpt-every 250 --grad-checkpointing --activation-offload
--pooling clsmean --proj-out-dim 512 --amp bfloat16
```
Only `--backbone` and `--seed` vary. Pin: `waiv-snapshots/gemgrid-pinned` (byte-identical
to HEAD for `contrastive.py` and `train_lora.py`).

**Scored step: 500**, fixed and pre-registered. Justification is measured, not assumed —
step 500 lies inside the seed floor of every backbone's RI peak (§3).

---

## 2. HEST (9-task leaderboard mean, per-backbone matched pooling)

| backbone | pooling | base | FT mean (n=5) | **Δ** | SD | 2SE | Δ/floor | seeds > 0 |
|---|---|---|---|---|---|---|---|---|
| phikon-v2 | cls | 0.37470 | 0.3871 | **+0.0124** | 0.0017 | 0.00152 | **8.2×** | 5/5 |
| midnight | cls | 0.39521 | 0.4065 | **+0.0113** | 0.0019 | 0.00170 | **6.6×** | 5/5 |
| Virchow2 | clsmean | 0.40324 | 0.4055 | **+0.0023** | 0.0012 | 0.00107 | **2.1×** | 5/5 |

**15/15 seeds positive.** Binomial sign test p ≈ 3×10⁻⁵.

> **Caveat on that p-value (added 2026-08-26).** The arithmetic is correct but one-sided: 0.5^15 = 3.05×10⁻⁵; the two-sided value is 6.1×10⁻⁵. More importantly, the n=15 is 3 backbones × 5 seeds and these are **not 15 independent units** — the 5 deltas within a backbone share one fixed base constant and one recipe, so they are correlated. The stated p therefore overstates the precision of the evidence.

Per-seed Δ (no negatives anywhere):
- phikon: +0.01353, +0.01013, +0.01204, +0.01446, +0.01180
- midnight (s0..s4): +0.01173, +0.00806, +0.01256, **+0.01128**, +0.01282 [corrected 2026-08-26: the list ended in the literal placeholder `+(s3)` and was mis-ordered — the printed 4th value +0.01282 is seed 4, not seed 3. Seed 3's real delta is +0.01128 (seed-3 FT mean 0.40648703703703704 − base 0.39521). Values are now in seed order s0..s4.]
- Virchow2: +0.00112, +0.00297, +0.00259, +0.00113, +0.00373

**The effect falls monotonically with base-model strength on HEST**: base 0.3747 → +0.0124;
0.3952 → +0.0113; 0.4032 → +0.0023.

**Do NOT read this as "headroom".** An identical recipe removes *configuration* as an
explanation, but it does not remove *tuning*. Every hyperparameter here — GEM, C=2 grid, T,
LoRA r32/a64, lr, tau, step 500 — was selected on phikon-v2 and then applied unchanged.
So "less headroom" and "recipe mistuned for this backbone" are **perfectly confounded** in
this design, and the HEST ordering alone cannot distinguish them. Evidence that the second
explanation is live is in §9. The defensible claim is: *the phikon-tuned recipe delivers
progressively less on stronger backbones, for reasons this design cannot isolate.*

---

## 3. RI (PathoROB avg robustness index)

Mean curves (n=5 except where noted):

| step | phikon | midnight | Virchow2 |
|---|---|---|---|
| 250 | 0.82119 | **0.90063** | 0.89762 |
| 500 | 0.82694 | 0.89927 | **0.89770** |
| 750 | **0.82763** | 0.89805 (n=4) | 0.89586 |
| 1000 | 0.82534 | 0.89308 (n=1) | 0.89284 |
| 1250 | 0.82438 | — | 0.89124 |
| 1500 | 0.82462 | — | 0.89015 (n=4) |

**Turnover confirmed on all three** — every backbone peaks and then declines monotonically:
phikon peaks at 750, midnight at 250, Virchow2 at 500.

At step 500, vs base:

| backbone | base RI | FT RI | Δ | SD | floor(2SE) |
|---|---|---|---|---|---|
| phikon-v2 | 0.4686 | 0.82694 | +0.3583 | 0.00453 | 0.00405 |
| midnight | 0.7589 | 0.89927 | +0.1404 | 0.00211 | 0.00189 |
| Virchow2 | 0.8582 | 0.89770 | +0.0395 | 0.00196 | 0.00176 |

---

## 4. The seed floor is not a constant

This is the most transferable result. Prior work applied a single borrowed floor of 0.0076
everywhere. Measured here, the floor varies along **two** axes:

- **By backbone** (RI @500): phikon 0.00453, midnight 0.00211, Virchow2 0.00196 — a 2.3× span.
- **By step** (phikon RI SD): 0.00408 @250, 0.00453 @500, 0.00171 @750, 0.00120 @1000.
  Early checkpoints are ~3–4× noisier than late ones.

Consequence: the 0.0076 constant was simultaneously **too loose** for Virchow2 (it would
have discarded a real +0.0023 HEST gain as null) and drawn from the noisiest checkpoints.

**Selection bias, measured.** Scoring each run at its own RI-argmax instead of fixed step 500
inflates the phikon mean by **+0.0060 = 1.5× the floor**. Every delta in `FINAL_RESULTS.md`
§2 was computed that way.

---

## 5. THUNDER — COMPLETE (n=5, full coverage)

Deltas are computed **per dataset, then averaged** over datasets present in both the
fine-tuned run and the matching base. Classification = 12 datasets; segmentation = 2
(see caveat below).

| backbone | kNN Δ | LinProbe Δ | SimpleShot Δ | Segmentation Δ |
|---|---|---|---|---|
| phikon-v2 | **+0.0349** | **+0.0169** | +0.0024 | **−0.0088** |
| midnight | +0.0043 | +0.0058 | **+0.0425** | +0.0045 |
| Virchow2 | **−0.0159** | −0.0002 | +0.0174 | +0.0034 |

**There is no single "THUNDER effect".** The per-task picture is heterogeneous and partly
negative:

- Each backbone's *best* task is different — phikon→kNN, midnight→SimpleShot,
  Virchow2→SimpleShot.
- kNN spans **+0.0349 to −0.0159** across backbones and is the only task that declines
  monotonically with base strength.
- phikon, the backbone that gains most overall, is the one that **loses** on segmentation.
- **midnight is the only backbone positive on all four tasks.**

Reporting a single averaged "THUNDER score" would conceal all of this and is not done here.

**Segmentation caveat:** Waiv's published segmentation mean covers 4 datasets; this study
uses only `ocelot` + `pannuke`. `segpath_epithelial` / `segpath_lymphocytes` are excluded
because they require non-default epoch overrides (guidelines.md mandates 9 and 21) and
midnight has no base result for either, so no delta could be formed. Our segmentation
column is therefore NOT directly comparable to Waiv's.

---

## 5b. Comparison with published Waiv

Only one directly verifiable anchor exists: **Waiv reports +0.0215 on midnight HEST.**
We measure **+0.0113 (n=5)** — roughly **53%** of it.

The prior in-house figure of +0.0166 ("77% of Waiv") was a **single seed** and does not
replicate at n=5; our 5-seed interval excludes it.

The "gap-closed: phikon 101% / midnight 90% / Virchow2 76%" line in `FINAL_RESULTS.md` §2.1
should be treated as unusable — it derives from the confounded three-backbone table, and its
midnight component rests on that non-replicating +0.0166.

---

## 6. Why the old §2 table is superseded

Two independent defects:

1. **It is not the config §1 calls final.** The three runs behind it — `splitgrid-386374`
   (phikon), `splitgrid-386380` (midnight), `splitgrid-386375` (Virchow2) — are all
   `pool_head = mean`. No GEM.
2. **T varied 4× across backbones** (1800 / 450 / 600), so "helps phikon, does nothing on
   the others" was inseparable from "phikon got 4× the negatives". The document also
   **transposes** midnight's and Virchow2's T values relative to disk.

Specific number retracted: Virchow2 "0.9078 @250" was a **single seed at its own argmax on
the noisiest checkpoint**. Its own run collapses to 0.8925 at step 500 and spends the rest
of training at 0.888–0.897. The 5-seed mean at step 250 is 0.89762 ± 0.00475; the old value
sits 2.1 SD above it — an unremarkable single draw.

Replicated: the phikon HEST gain (+0.0115 reported → **+0.0124** measured at n=5) and the
per-backbone HEST pooling protocol of §4.

Not replicated: midnight HEST **+0.0166** (single run) → **+0.0113 ± 0.0017** (n=5). The
prior figure lies outside the interval.

---

## 7. T is not flat on the GEM path

A paired T sweep on phikon (seeds 0/1, all else identical):

| comparison | seed 0 | seed 1 | all points | sign |
|---|---|---|---|---|
| T=450 − T=900 | −0.00151 | −0.00306 | −0.00229 (n=12) | 1/12 positive |
| T=1800 − T=900 | +0.00390 | +0.00112 | +0.00297 (n=3) | 3/3 positive |

Monotonic **T=450 < T=900 < T=1800**, both seeds agreeing in both directions. Magnitude
(~0.003) is *below* phikon's 0.00405 floor — the evidence is carried by sign consistency,
not effect size, and effective independent n is 2 (steps within a run are correlated).

This contradicts the earlier "negatives axis is flat at fixed C" finding, which was derived
unpaired, non-GEM, at C=12/24. **Implication: T=900 was forced by midnight's OOM ceiling
(it fails at both 1800 and 1200), and that ceiling may cost ~0.003 RI.** If so, GradCache
(~150 LOC, exact, activations O(chunk) not O(B)) would buy accuracy, not just throughput.
Not established — needs more seeds.

---

## 8. Memory ceilings (measured, 1×H100-80GB, C=2, chunk=0, offload on)

| backbone | T=1800 | T=1200 | T=900 | T=600 |
|---|---|---|---|---|
| Virchow2 | OOM | — | PASS | — |
| midnight | OOM | OOM | **PASS** (peak 64.56/79.19 GiB) | PASS |
| phikon-v2 | — | — | PASS | — |

**midnight is the binding backbone, not Virchow2** — the reverse of what the old table
implied by giving midnight the smallest T. GEM forces `--grid-forward-chunk 0`
(`contrastive.py:896` raises otherwise), so chunking is unavailable and activation offload
is the only lever; it is exact (bit-checked, |dgrad| = 0.0 over 294 gradients).

---

## 9. Does the phikon-tuned recipe transfer? (open)

Every hyperparameter in §1 was chosen on phikon-v2. Three independent signals suggest the
package does not fit the other backbones:

1. **The RI-optimal step differs 3x by backbone** — phikon 750, Virchow2 500, midnight 250.
   A recipe whose optimum sits at 750 applied to a model peaking at 250 is a mistuning, not
   a capacity limit.
2. **midnight cannot reach phikon's preferred geometry** — it OOMs at T=1200 and T=1800,
   while the paired sweep (§7) orders T=1800 > T=900 > T=450.
3. **THUNDER task ordering INVERTS across backbones** (§5). On phikon the gain concentrates
   in kNN (+0.0349) and vanishes on SimpleShot (+0.0024); on Virchow2 it is reversed —
   SimpleShot is the only winner (+0.0174) and kNN actively regresses (-0.0159). A pure
   headroom story predicts the same effect shape, scaled down; it does not predict a sign
   flip on the strongest task or a reordering of tasks.

### GEM was the obvious suspect, and it is EXONERATED
On Virchow2, an older pair-path run with no GEM beats the final5 recipe at all 6 steps with a
widening gap (+0.0113 by step 1500). GEM looked like the phikon-specific choice responsible.
A clean paired ablation says otherwise:

**Virchow2, GEM vs no-GEM, paired by seed (3 seeds x 6 steps = 18 paired points):
no-GEM − GEM = −0.00188, only 1/18 points positive.**

GEM *helps* Virchow2. The hypothesis is dead. Since the old run differed in two ways
(no GEM AND pair sampler), the remaining suspect is the **grid sampler** (C=2, T=900) versus
the pair path — untested, and the obvious next experiment.

Runs: `v2ablate-virchow2-s{0,1,2}-t900-3896{72,73,74}` (identical to final5 except
`--pool-head` omitted).
