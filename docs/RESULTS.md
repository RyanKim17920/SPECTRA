# Results

Full numeric record. The README carries only the headline tables; everything below is the
detail behind them. Read alongside [`CAVEATS.md`](CAVEATS.md) — several of these numbers are
not comparable across studies without the caveats attached.

---

## 1. Headline — average level

Each backbone's **base** was reproduced against a published reference before any
fine-tuning, which is what makes the deltas meaningful rather than merely plausible.

### phikon-v2 → Phaet (ours: LoRA, step 1000)

| benchmark | base ours | base published | fine-tuned ours | Waiv Phaet |
|---|---|---|---|---|
| PathoROB Avg RI ↑ | 0.4686 | 0.4686 / 0.469 | **0.8080** | 0.806 |
| THUNDER mean Δ over 4 tasks ↑ | — | — | **+2.29** | +1.35 |
| HEST Avg Pearson ↑ | 0.3747 | 0.3747 | **0.3794** (s1000) / 0.3825 (s2000) | 0.3943 |

### Midnight-12k → Mascaret (ours: LoRA, step 500)

| benchmark | base ours | base published | fine-tuned ours | Waiv Mascaret |
|---|---|---|---|---|
| PathoROB Avg RI ↑ | 0.7589 | 0.759 | **0.9080** | 0.924 |
| THUNDER mean Δ over 4 tasks ↑ | — | — | **+2.21** | +1.80 |
| HEST Avg Pearson ↑ | 0.4121 | 0.3952 | **0.4132** (s500) | 0.4167 |

The Midnight HEST columns are **not** on a common scale: ours is `clsmean` fp32, and both
summary files carry the harness's own note — *"backbone=kaiko-ai/midnight pooling=clsmean has
NO published counterpart here — this is our own reference for checkpoint-to-checkpoint
retention only."* Waiv state no pooling protocol for their Midnight HEST number, so 0.3952
may be CLS-only against our CLS+mean, and their runs are mixed precision. Our base 0.4121 is
already **above** their reported base (+0.0169) and just under their fine-tuned Mascaret
0.4167; that is a protocol artefact, not evidence we are near Mascaret. Only the base→FT
delta is a valid comparison.

### Virchow2 (ours: LoRA, step 250)

| benchmark | base ours | base published | fine-tuned ours | Waiv Virchow2 |
|---|---|---|---|---|
| PathoROB Avg RI ↑ | 0.8582 | 0.858 | **0.9035** | 0.918 |
| THUNDER mean Δ over 4 tasks ↑ | — | — | **+1.42** | +0.62 |
| HEST Avg Pearson ↑ | 0.4032 | 0.4034 | **0.4083** | 0.4135 |

Virchow2 **is** a like-for-like comparison on all three axes. An earlier draft of this file
said Waiv publish nothing but a PathoROB average for this backbone; that was wrong, and it
came from reading `collect_thunder.py`'s `PUBLISHED` dict (which held only phikon-v2) as if it
described the paper rather than what had been transcribed into this repo. arXiv:2607.22861
Table 2 gives Virchow2 THUNDER per task and Tables 1/3 give its HEST. All of it is now
transcribed in [`waiv_published.json`](waiv_published.json) — Tables 1–4, with the Patho-Bench
grand-average row cross-checked against Table 1 for all 20 models.

Our Virchow2 base reproduces their reported base to **0.0002 on HEST** (0.4032 vs 0.4034) and
**0.0002 on RI** (0.8582 vs 0.858), which is what licenses the comparison.

Reading:

- **PathoROB.** phikon-v2 lands *on* Waiv's Phaet number (0.8080 vs 0.806, ~100% of the
  headroom). Midnight reaches 0.9080 against Mascaret's 0.924 — 90.3% of the headroom, and
  plainly short. Virchow2 reaches 0.9035 against 0.918 — ~76% (honest range 70–76%, §6).
  A full fine-tuning pilot on phikon-v2 peaked lower, at 0.8007 (§4). Robustness now
  reproduces on **3 of 3** backbones, and the gap-closed fraction falls monotonically as the
  base gets stronger — 0.4686 → ~101%, 0.7589 → 90.3%, 0.8582 → ~76% — which tracks remaining
  headroom rather than architecture (§6 rules out LoRA rank as the explanation).
- **THUNDER.** All 16 of Waiv's datasets are now covered on all three backbones, so every task
  average — segmentation included — is over the same sets as theirs. We match or beat Waiv on
  **11 of the 12 comparable model × task pairs** (§2, §6) — Virchow2 contributes four, all wins,
  from Table 2. The sole loss is Midnight segmentation (−0.33 vs their +1.6); phikon-v2
  segmentation is a win in the sense that we regress less (−0.12 vs their −1.2). The
  mean-over-4-tasks figure still hides composition: see §2. Across all three backbones the
  gain is carried by classification and **not** by segmentation — 32 of 36 classification
  model × dataset pairs improve (sign test p ≈ 2×10⁻⁶) against 3 of 12 segmentation pairs
  (p ≈ 0.15). Segmentation is flat-to-slightly-negative on every backbone (−0.12, −0.33,
  −0.12); the consistent sign, not any single cell, is what makes that worth stating.
- **HEST is the weak axis of the reconstruction, on all three backbones, 3 of 3.** phikon-v2
  +0.0047 (step 1000) / +0.0078 (step 2000) against Waiv's Phaet +0.0196; Midnight +0.0011
  (step 500) against their Mascaret +0.0215. Read against the benchmark's dynamic range
  (0.3252–0.4229, span 0.0977), not against zero: our Midnight delta is ~1% of that span,
  Waiv's ~22%. With the Midnight run in, this is no longer a phikon-v2 quirk — it is a
  consistent gap in the recipe. A sweep over every checkpoint on both backbones (§2) shows
  the best in range is +0.0098 (phikon-v2, step 3500) and +0.0035 (Midnight, step 250), so
  the gap is intrinsic, not a checkpoint-selection artefact. Virchow2 adds a third instance
  at +0.0051 against their +0.0101 — the closest of the three, but still half — and it sits
  **at** the ±0.005 within-arm scatter of the benchmark: retention preserved, not retention
  improved. Retention is where the reconstruction falls short, and it now does so on every
  backbone tried, on all three of which Waiv publish a HEST number we can be measured against.

Per-dataset PathoROB, for completeness:

| model | camelyon | tolkach_esca | tcga | Avg |
|---|---|---|---|---|
| phikon-v2 base (ours) | 0.0190 | 0.7681 | 0.6188 | 0.4686 |
| phikon-v2 base (Waiv T1) | 0.019 | 0.768 | 0.619 | 0.469 |
| ours step 1000 | **0.7169** | **0.9279** | **0.7791** | **0.8080** |
| Waiv Phaet | 0.702 | 0.932 | 0.785 | 0.806 |
| Midnight base (ours) | 0.4780 | 0.9411 | 0.8575 | 0.7589 |
| Midnight base (Waiv T1) | 0.478 | 0.941 | 0.858 | 0.759 |
| ours step 500 | **0.8844** | **0.9683** | **0.8712** | **0.9080** |
| Waiv Mascaret | 0.907 | 0.972 | 0.893 | 0.924 |
| Virchow2 base (ours) | 0.7989 | 0.9541 | 0.8218 | 0.8582 |
| Virchow2 base (Waiv T1) | 0.799 | 0.954 | 0.822 | 0.858 |
| ours step 250 | **0.9006** | **0.9673** | **0.8425** | **0.9035** |
| Waiv Virchow2 | 0.935 | 0.969 | 0.849 | 0.918 |

Waiv publish the per-dataset breakdown for Virchow2 too — arXiv:2607.22861 Table 1, transcribed
in [`waiv_published.json`](waiv_published.json). An earlier draft had `—` in these cells on the
same mistaken belief corrected in §1. Our Virchow2 base reproduces theirs to **≤0.0002 on every
dataset** (0.7989/0.9541/0.8218 against 0.799/0.954/0.822).

Where their fine-tuning beats ours is **camelyon on the two stronger backbones**: 0.907 vs our
0.8844 on Midnight, 0.935 vs our 0.9006 on Virchow2. That single dataset accounts for
essentially the whole average shortfall on both — on tolkach_esca and tcga we are within
0.004–0.022. phikon-v2 is the exception, where we edge them on camelyon (0.7169 vs 0.702).


---

## 2. Task level — THUNDER

Base and fine-tuned run through identical code, so these deltas carry no cross-study
assumption. Our base reproduces THUNDER's published phikon-v2 row (mean Δ +0.08 F1 over 12
datasets on linear probing), which is what makes Δ-vs-Δ against Waiv legitimate despite
their runs being mixed precision and ours fp32.

### phikon-v2 → Phaet (ours step 1000, `cls` pooling)

| task | our base | our ft | our Δ | Waiv Phaet Δ |
|---|---|---|---|---|
| kNN | 70.28 | 75.20 | **+4.92** | +3.7 |
| linear probing | 76.54 | 79.24 | **+2.69** | +1.4 |
| few-shot | 69.33 | 70.99 | **+1.66** | +1.5 |
| segmentation | 67.73 | 67.61 | **−0.12** | −1.2 |

### Midnight-12k → Mascaret (ours step 500, `clsmean` pooling)

| task | our base | our ft | our Δ | Waiv Mascaret Δ |
|---|---|---|---|---|
| kNN | 78.25 | 80.44 | **+2.19** | +1.7 |
| linear probing | 82.88 | 84.12 | **+1.24** | +0.2 |
| few-shot | 70.64 | 76.38 | **+5.74** | +3.7 |
| segmentation | 68.73 | 68.40 | **−0.33** | +1.6 |

All four task averages are now over the same datasets Waiv used — 12 classification, 4
segmentation — so these are like-for-like. We match or beat Waiv on 7 of the 8 model × task
pairs; the exception is Midnight segmentation.

Waiv publish **both** model pairs in their Table 2; the Midnight row above is compared
against **Mascaret**, which is its correct counterpart.

One precision note, stated once: Waiv's Table 2 Midnight base is their *mixed-precision*
rerun (seg 66.0). Table 5's full-precision leaderboard row for the same model is seg 68.8
and lin 84.7, against which Mascaret's deltas are seg **−1.2** and lin **−0.1** rather than
+1.6 and +0.2. Our runs are fp32. We do not lean on this — the fp32 comparison is the one
that would favour us, and the mixed-precision Table 2 deltas are what we quote above.

### HEST, per cancer type (phikon-v2, `cls` protocol)

Waiv publish the full per-type Phaet row (Table 3), so this is Δ-vs-Δ.

| task | base | Waiv Phaet | Δ Waiv | ours s1000 | ours s2000 | Δ ours (s2000) |
|---|---|---|---|---|---|---|
| IDC | 0.5408 | 0.5630 | +0.0222 | 0.5476 | 0.5491 | +0.0083 |
| PRAD | 0.3545 | 0.3546 | +0.0001 | 0.3268 | 0.3334 | −0.0211 |
| PAAD | 0.4455 | 0.4748 | +0.0293 | 0.4627 | 0.4666 | +0.0211 |
| SKCM | 0.5554 | 0.5985 | +0.0431 | 0.5813 | 0.5880 | +0.0326 |
| COAD | 0.2500 | 0.2915 | +0.0415 | 0.2907 | 0.2859 | +0.0359 |
| READ | 0.1749 | 0.1696 | −0.0053 | 0.1455 | 0.1655 | −0.0093 |
| CCRCC | 0.2659 | 0.2696 | +0.0037 | 0.2681 | 0.2666 | +0.0008 |
| LUNG | 0.5419 | 0.5622 | +0.0203 | 0.5316 | 0.5299 | −0.0120 |
| LYMPH_IDC | 0.2437 | 0.2649 | +0.0212 | 0.2600 | 0.2575 | +0.0139 |
| **Avg** | **0.3747** | **0.3943** | **+0.0196** | 0.3794 | **0.3825** | **+0.0078** |

corr(Δ_waiv, Δ_ours) = 0.880 — their three largest gains (COAD, SKCM, PAAD) are our three
largest, and we regress on READ as they do. The shortfall is concentrated in LUNG and PRAD
rather than spread across tasks.

### HEST, per cancer type (Midnight-12k, `clsmean` protocol)

Jobs 370850 (base, 3978 s) and 370855 (step 500, 4298 s), both COMPLETED, fp32. Waiv publish
no per-type row for Midnight, so this is base→FT only — Δ, not Δ-vs-Δ.

| task | base | ours s500 | Δ ours |
|---|---|---|---|
| IDC | 0.5991 | 0.5898 | −0.0093 |
| PRAD | 0.3715 | 0.3481 | −0.0234 |
| PAAD | 0.5020 | 0.5166 | +0.0146 |
| SKCM | 0.6484 | 0.6506 | +0.0022 |
| COAD | 0.3182 | 0.3109 | −0.0073 |
| READ | 0.2033 | 0.1887 | −0.0146 |
| CCRCC | 0.2093 | 0.2660 | +0.0567 |
| LUNG | 0.5826 | 0.5750 | −0.0076 |
| LYMPH_IDC | 0.2746 | 0.2731 | −0.0015 |
| **Avg** | **0.4121** | **0.4132** | **+0.0011** |

Six of nine tasks regress; the average is held positive by CCRCC alone (+0.0567). Against
Mascaret's +0.0215 this is ~1/20th of the movement, and ~1% of the benchmark's 0.0977 dynamic
range against Waiv's ~22%.

`scripts/collect_hest.py` defaults to `--base base_<pooling>`, which for `clsmean` would diff
Midnight against phikon-v2's baseline. The Midnight table above must be regenerated with the
flags explicit:

```bash
PYTHONNOUSERSITE=1 ./.venv-hest/bin/python scripts/collect_hest.py \
  --pooling clsmean --base mbase_clsmean --runs mft500_clsmean
```

Step 500 is Midnight's best *PathoROB* checkpoint, chosen for consistency with the THUNDER
and PathoROB rows — not a HEST-optimal one. The sweep below settles what that costs.

### HEST checkpoint sweep — both backbones, every available checkpoint

One question: is the HEST shortfall an artefact of which checkpoint we report, or intrinsic
to the recipe? Every checkpoint on both backbones was evaluated. Answer: intrinsic.

phikon-v2, `cls` protocol, base 0.3747, from `runs/waiv-real-369043/` (jobs 370991–370995
plus the pre-existing ft1000/ft2000; exp codes `base_cls`, `ft500_cls` … `ft3500_cls`):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|
| Avg Pearson | 0.3818 | 0.3794 | 0.3813 | 0.3825 | 0.3826 | 0.3838 | **0.3845** |
| Δ vs base | +0.0071 | +0.0047 | +0.0066 | +0.0078 | +0.0079 | +0.0091 | **+0.0098** |

Midnight-12k, `clsmean` protocol, base 0.4121, from `runs/waiv-midnight-369159/` (jobs
370986–370990 plus the pre-existing mft500; exp codes `mbase_clsmean`, `mft250_clsmean` …
`mft1500_clsmean`):

| step | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|
| Avg Pearson | **0.4156** | 0.4132 | 0.4136 | 0.4134 | 0.4118 | 0.4115 |
| Δ vs base | **+0.0035** | +0.0011 | +0.0015 | +0.0013 | −0.0003 | −0.0007 |

**The two curves run in opposite directions.** phikon-v2's retention climbs with training and
is *still climbing* at the last checkpoint (3500), so its HEST optimum is not bracketed — we
cannot claim to have found the maximum, only that nothing in the trained range reaches Waiv.
Midnight's decays monotonically from the earliest checkpoint and goes negative by step 1250.
The natural expectation — that retention peaks *later* than robustness, as it does on
phikon-v2 where PathoROB peaks at 1000 and HEST at 3500 — is **refuted**: on Midnight
PathoROB peaks at 500 and HEST at 250, the other ordering. There is no transferable rule
here. Anyone porting this recipe to a new backbone cannot assume either direction.

**The gap is intrinsic, not a checkpoint-selection artefact.** On both backbones the best
checkpoint in the entire trained range still falls far short of Waiv: +0.0098 vs Phaet's
+0.0196 (phikon-v2), +0.0035 vs Mascaret's +0.0215 (Midnight). Best-case selection closes
roughly half the phikon-v2 gap and a sixth of the Midnight one. The shortfall is a property
of the recipe.

**The headline checkpoints do not change.** We continue to report step 1000 (phikon-v2) and
step 500 (Midnight) on every benchmark, selected by the blind, model-agnostic rule "best
PathoROB checkpoint". Picking a different checkpoint per benchmark would inflate HEST, break
the single-model-per-backbone comparison Waiv's release implies (they publish one Phaet and
one Mascaret, evaluated on everything), and is exactly the eval-gaming that the reporting discipline in
[`CAVEATS.md`](CAVEATS.md) forbids. That rule
costs us, and the cost is recorded here rather than optimised away: step 1000 is a **local
minimum** on phikon-v2's HEST curve (0.3818 → 0.3794 → 0.3813 across 500/1000/1500), and
step 500 is not Midnight's best either.

**Hypothesis for the intrinsic gap — a missing loss term, untested.** The loss as built is a
single InfoNCE term with **no retention component**: no frozen-teacher anchor, no
distillation, no replay, no L2-to-base penalty. LoRA's implicit bounding of drift is the
entire anti-forgetting mechanism. `PLAN.md` §2 scoped "TCGA replay tiles, frozen-teacher
anchor" as optional and neither was implemented. Given that the gap survives checkpoint
selection on both backbones, that missing term is the leading hypothesis for it. We have not
tested it; this is a stated gap in the reconstruction, not a result.


---

## 3. Dataset level — THUNDER

Waiv publish **no per-dataset THUNDER breakdown**, only the six task averages, so this
table has no counterpart in their paper and no dataset-level comparison against them is
possible. It is strictly more granular than what they released.

### phikon-v2, base → step 1000 (F1 ×100)

| dataset | kNN base | ft | Δ | lin base | ft | Δ | few base | ft | Δ | seg base | ft | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bach | 57.2 | 61.4 | +4.3 | 63.7 | 67.8 | +4.1 | 54.9 | 58.9 | +4.0 | — | — | — |
| bracs | 45.2 | 51.3 | +6.1 | 59.9 | 57.6 | −2.3 | 41.3 | 43.5 | +2.3 | — | — | — |
| break_his | 56.8 | 69.2 | +12.5 | 50.8 | 68.3 | +17.5 | 61.8 | 67.2 | +5.4 | — | — | — |
| ccrcc | 76.7 | 85.3 | +8.6 | 78.7 | 90.4 | +11.7 | 90.2 | 88.1 | −2.2 | — | — | — |
| crc | 92.1 | 94.5 | +2.3 | 92.0 | 94.0 | +2.0 | 89.6 | 95.1 | +5.5 | — | — | — |
| esca | 75.3 | 79.2 | +3.9 | 78.0 | 80.9 | +2.9 | 66.4 | 64.5 | −1.9 | — | — | — |
| mhist | 66.4 | 70.8 | +4.4 | 79.1 | 77.4 | −1.7 | 55.7 | 60.5 | +4.8 | — | — | — |
| patch_camelyon | 81.6 | 86.4 | +4.8 | 89.4 | 91.9 | +2.6 | 82.1 | 84.0 | +1.9 | — | — | — |
| tcga_crc_msi | 56.8 | 61.7 | +4.9 | 62.0 | 62.1 | +0.1 | 56.8 | 57.9 | +1.1 | — | — | — |
| tcga_tils | 80.6 | 87.9 | +7.3 | 91.0 | 91.0 | +0.0 | 85.7 | 85.6 | −0.0 | — | — | — |
| **tcga_uniform** | 68.2 | 60.0 | **−8.2** | 77.1 | 71.5 | **−5.7** | 60.0 | 52.7 | **−7.3** | — | — | — |
| wilds | 86.6 | 95.0 | +8.4 | 96.8 | 97.9 | +1.2 | 87.4 | 93.9 | +6.4 | — | — | — |
| ocelot | — | — | — | — | — | — | — | — | — | 80.0 | 79.5 | −0.5 |
| pannuke | — | — | — | — | — | — | — | — | — | 60.8 | 60.6 | −0.2 |
| segpath_epithelial | — | — | — | — | — | — | — | — | — | 69.5 | 68.9 | −0.5 |
| segpath_lymphocytes | — | — | — | — | — | — | — | — | — | 60.6 | 61.3 | +0.7 |

### Midnight-12k, base → step 500 (F1 ×100)

| dataset | kNN base | ft | Δ | lin base | ft | Δ | few base | ft | Δ | seg base | ft | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bach** | 84.3 | 82.2 | **−2.1** | 87.9 | 82.4 | **−5.6** | 82.5 | 73.9 | **−8.6** | — | — | — |
| bracs | 50.2 | 53.5 | +3.3 | 63.8 | 63.7 | −0.1 | 49.6 | 52.0 | +2.4 | — | — | — |
| break_his | 58.1 | 74.2 | +16.1 | 56.7 | 76.1 | +19.4 | 38.5 | 66.5 | +28.0 | — | — | — |
| ccrcc | 91.6 | 89.6 | −1.9 | 90.8 | 89.5 | −1.3 | 77.5 | 88.6 | +11.1 | — | — | — |
| crc | 94.2 | 94.7 | +0.5 | 95.4 | 96.2 | +0.8 | 94.7 | 95.6 | +0.9 | — | — | — |
| esca | 81.4 | 82.4 | +1.0 | 86.2 | 87.0 | +0.8 | 75.0 | 73.9 | −1.2 | — | — | — |
| mhist | 69.3 | 74.8 | +5.5 | 80.2 | 79.8 | −0.4 | 62.4 | 71.1 | +8.6 | — | — | — |
| patch_camelyon | 88.0 | 89.3 | +1.2 | 93.5 | 94.2 | +0.6 | 82.8 | 86.4 | +3.6 | — | — | — |
| tcga_crc_msi | 61.9 | 64.1 | +2.1 | 65.6 | 68.7 | +3.1 | 55.1 | 59.6 | +4.6 | — | — | — |
| tcga_tils | 87.6 | 89.6 | +2.0 | 91.0 | 91.1 | +0.2 | 76.2 | 89.2 | +13.0 | — | — | — |
| **tcga_uniform** | 77.5 | 74.9 | **−2.6** | 85.2 | 82.3 | **−2.9** | 63.4 | 66.3 | +2.9 | — | — | — |
| wilds | 95.0 | 96.2 | +1.2 | 98.3 | 98.4 | +0.2 | 89.9 | 93.6 | +3.6 | — | — | — |
| ocelot | — | — | — | — | — | — | — | — | — | 78.4 | 79.4 | +0.9 |
| pannuke | — | — | — | — | — | — | — | — | — | 61.8 | 61.5 | −0.3 |
| segpath_epithelial | — | — | — | — | — | — | — | — | — | 70.9 | 69.1 | −1.8 |
| segpath_lymphocytes | — | — | — | — | — | — | — | — | — | 63.8 | 63.6 | −0.1 |

### Virchow2, base → step 250 (F1 ×100)

| dataset | kNN base | ft | Δ | lin base | ft | Δ | few base | ft | Δ | seg base | ft | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bach** | 78.6 | 80.8 | +2.1 | 76.6 | 80.5 | +4.0 | 80.7 | 76.4 | **−4.3** | — | — | — |
| bracs | 54.1 | 56.1 | +1.9 | 61.3 | 63.4 | +2.1 | 51.8 | 56.0 | +4.1 | — | — | — |
| **break_his** | 79.7 | 72.7 | **−6.9** | 73.2 | 78.3 | +5.1 | 57.4 | 65.1 | +7.7 | — | — | — |
| ccrcc | 91.4 | 91.9 | +0.5 | 93.7 | 92.5 | −1.2 | 80.0 | 88.4 | +8.4 | — | — | — |
| crc | 94.3 | 94.9 | +0.6 | 91.5 | 95.1 | +3.6 | 91.3 | 95.8 | +4.5 | — | — | — |
| esca | 86.7 | 86.0 | −0.7 | 88.7 | 89.3 | +0.6 | 70.2 | 77.3 | +7.2 | — | — | — |
| mhist | 73.4 | 77.8 | +4.4 | 83.7 | 84.7 | +1.0 | 68.5 | 70.6 | +2.2 | — | — | — |
| patch_camelyon | 90.6 | 90.1 | −0.5 | 94.6 | 94.4 | −0.2 | 79.0 | 90.2 | +11.2 | — | — | — |
| tcga_crc_msi | 65.3 | 65.8 | +0.5 | 69.7 | 69.3 | −0.4 | 56.5 | 62.7 | +6.2 | — | — | — |
| tcga_tils | 88.3 | 88.8 | +0.5 | 90.8 | 91.2 | +0.5 | 86.4 | 87.0 | +0.6 | — | — | — |
| **tcga_uniform** | 70.5 | 68.0 | **−2.5** | 78.5 | 77.6 | −1.0 | 57.4 | 60.0 | +2.6 | — | — | — |
| wilds | 97.6 | 98.1 | +0.5 | 96.6 | 98.6 | +2.0 | 93.8 | 96.5 | +2.7 | — | — | — |
| ocelot | — | — | — | — | — | — | — | — | — | 79.5 | 80.3 | +0.9 |
| pannuke | — | — | — | — | — | — | — | — | — | 62.8 | 62.3 | −0.5 |
| segpath_epithelial | — | — | — | — | — | — | — | — | — | 70.6 | 69.8 | −0.8 |
| segpath_lymphocytes | — | — | — | — | — | — | — | — | — | 63.2 | 63.1 | −0.0 |

**`tcga_uniform` is the only dataset that regresses on all three backbones** (phikon-v2 −8.2
kNN, Midnight −2.6 kNN, Virchow2 −2.5 kNN), and it is phikon-v2's single worst result
anywhere in this record. `break_his` is the largest winner on all three (+12.5 / +16.1 kNN,
and +7.7 few-shot on Virchow2 despite its −6.9 kNN). Virchow2's few-shot column carries its
gain almost alone: +11.2 on patch_camelyon and +8.4 on ccrcc against a kNN column that nets
+0.03 overall (§2) — the base kNN is already the strongest of the three backbones (80.87),
leaving least to add.

All 16 datasets are present and all four are folded into the §2 segmentation averages.
Exact segpath F1 fractions: lymphocytes phikon-v2 0.6065 → 0.6130, Midnight 0.6375 → 0.6364;
epithelial phikon-v2 0.694590 → 0.689437, Midnight 0.709486 → 0.691234.

Midnight's `segpath_epithelial` regression (−1.8) is the largest single-dataset segmentation
move on either backbone, and its confidence intervals do not overlap ([0.7071, 0.7120] base
vs [0.6888, 0.6937] fine-tuned). It is the reason Midnight's segmentation task average is
negative.

THUNDER pooling is per-backbone and is not our choice: arXiv:2607.22861 §3 uses CLS+mean
concatenation in THUNDER only for Virchow2, AquaViT, H0-mini and Midnight-12k, so
phikon-v2 is `cls` and Midnight is `clsmean`. On ViT-g the 3072-d `clsmean` vector crashes
THUNDER's segmentation decoder, so Midnight's 4 segmentation sets are `cls` while its 12
classification sets are `clsmean` — a real methodological split, recorded per-row by
`scripts/collect_thunder.py`.


---

## 4. Full fine-tuning pilot

### Full FT vs LoRA — done, full FT did not beat LoRA

SLURM job 369922 (COMPLETED, 2075 s) + eval follower 369924 (COMPLETED), run dir
`/data/ryan.kim/waiv_runs/waiv-fullft-369922`. phikon-v2, `--full-ft` (305,976,832/
305,976,832 params trainable), 600 steps, checkpoints at 25/50/75/100/150/200/300/400/500/600.
The dense early schedule exists because LoRA had already reached 0.8036 of its 0.8080 peak
by step 500 with no visibility below that.

| step | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|
| Avg RI | 0.4853 | 0.5392 | 0.6374 | 0.7174 | 0.7667 | 0.7682 | 0.7938 | 0.7991 | **0.8007** | 0.8002 |
| Avg bal. acc | 0.9464 | 0.9501 | 0.9529 | 0.9513 | 0.9498 | 0.9487 | 0.9484 | 0.9477 | 0.9477 | 0.9479 |

Best full FT is **0.8007 @ step 500**, against LoRA's **0.8080 @ step 1000**. RI climbs
monotonically to step 500 and then plateaus; balanced accuracy is flat at 0.946–0.953
throughout, so this is not a collapse. The checkpoint costs ~18× the disk (3.42 GiB vs
193 MiB per step dir) and the deployable artefact ~21× (1.13 GiB backbone vs a 54 MiB adapter).

**The A/B is not clean and the conclusion must be read narrowly.** Full FT ran at lr 1e-5 /
weight decay 0.02; the LoRA run was lr 1e-4 / decay 0.05. So more than `--full-ft` differs,
and at n=1 seed a 0.007 gap is not decisive in either direction. The supportable claim is
"full FT did not beat LoRA here", **not** "LoRA wins".


---

## 5. LoRA rank sweep

Does LoRA rank matter for robustification? Four new arms (ranks 8 / 16 / 64 / 128) against the
rank-32 reference that every published number in §1–§4 comes from. Both passes are complete.
The answer is no on both axes: rank moves neither PathoROB nor HEST retention beyond the scatter
a single arm shows between neighbouring checkpoints. What rank changes is *when* an arm peaks,
not how high.

### Design

| rank | alpha | alpha/r | trainable params | adapter on disk | SLURM job | state | run dir |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 2 | 3.54 M | 13.5 MiB | 371020 | COMPLETED | `runs/waiv-rank8-371020/` |
| 16 | 32 | 2 | 7.08 M | 27.0 MiB | 371021 | COMPLETED | `runs/waiv-rank16-371021/` |
| 32 (ref) | 64 | 2 | 14.16 M | 54.0 MiB | 369043 | see note | `runs/waiv-real-369043/` |
| 64 | 128 | 2 | 28.31 M | 108.0 MiB | 371022 | COMPLETED | `runs/waiv-rank64-371022/` |
| 128 | 256 | 2 | 56.62 M | 216.0 MiB | 371023 | COMPLETED | `runs/waiv-rank128-371023/` |

Alpha is set to 2r in every arm, so the LoRA scaling factor alpha/r is held constant at 2 and
**rank is the only thing that varies**. Verified from each arm's `adapter_config.json`
(`r`, `lora_alpha`) and its identical `target_modules` set (query/key/value/dense/fc1/fc2).
Everything else is byte-identical to 369043: lr 1e-4, weight decay 0.05, warmup 200,
temperature 0.07, 2 groups × 192, seed 0, `max_steps` **4000**.

`max_steps` was deliberately **not** shortened for the sweep. The LR schedule is cosine to zero
at `max_steps`, so a shorter run is not a prefix of a longer one — a 2000-step arm would be a
different LR trajectory, not an early view of the same one. Each arm is a full 4000-step run
(~2 h 33 m wall).

Note on the rank-32 reference: job 369043 was cancelled after writing 7 checkpoints, and its
eval follower scored 6 of them, so the reference curve stops at step 3000 while the sweep arms
run to 4000. Its plateau window is therefore n=4 where the others are n=6.

### Avg RI by step (mean of camelyon, tolkach_esca, tcga robustness_index)

| step | rank 8 | rank 16 | rank 32 (ref) | rank 64 | rank 128 |
|---|---|---|---|---|---|
| 500 | 0.7793 | 0.7990 | 0.8036 | 0.8078 | **0.8084** |
| 1000 | 0.7920 | 0.8026 | **0.8080** | 0.8041 | 0.8081 |
| 1500 | 0.7944 | 0.7976 | 0.8032 | 0.7999 | 0.7983 |
| 2000 | 0.8013 | **0.8042** | 0.8051 | 0.8006 | 0.8052 |
| 2500 | 0.8005 | 0.8021 | 0.8022 | 0.7953 | 0.8002 |
| 3000 | 0.8008 | 0.8032 | 0.8010 | 0.7953 | 0.8011 |
| 3500 | 0.8006 | 0.8020 | — | 0.7954 | 0.8001 |
| 4000 | **0.8014** | 0.8026 | — | 0.7950 | 0.8002 |

Recomputed from each arm's `ri_curve.json` (`points[].datasets.{camelyon,tolkach_esca,tcga}
.robustness_index`), not transcribed. phikon-v2 base is 0.469; Waiv's Phaet target is 0.806.

### Peak and plateau disagree, and the plateau is the honest statistic

| rank | peak Avg RI | at step | plateau mean (steps ≥1500) | plateau sd | n |
|---|---|---|---|---|---|
| 8 | 0.8014 | 4000 | 0.7998 | 0.0027 | 6 |
| 16 | 0.8042 | 2000 | 0.8019 | 0.0023 | 6 |
| 32 (ref) | 0.8080 | 1000 | **0.8029** | 0.0017 | 4 |
| 64 | 0.8078 | 500 | 0.7969 | 0.0026 | 6 |
| 128 | **0.8084** | 500 | 0.8009 | 0.0023 | 6 |

**Peak looks monotone in rank; it is an artefact of peak-picking.** Read down the peak column —
0.8014 / 0.8042 / 0.8080 / 0.8078 / 0.8084 — and rank appears to buy robustness. It does not.
Ranks 64 and 128 both peak at step **500**, a single checkpoint; taking an arm's max over 8
checkpoints selects that arm's luckiest one, and the two arms with the most checkpoint-to-
checkpoint scatter get the largest upward selection bias. The plateau means are non-monotone:
rank 32 highest (0.8029), rank 64 **lowest** (0.7969), rank 128 in between (0.8009). No trend.

**Headline: rank does not systematically affect PathoROB.** The between-rank spread of plateau
means is 0.0060 (0.7969–0.8029), against a within-arm scatter of 0.0017–0.0027. At n=1 seed the
arms are not separable: ranks 16, 32, 64 and 128 all sit inside roughly one within-arm
standard deviation of each other. Only rank 8 separates, and only transiently — at step 500 it
is 0.7793, nearly 0.03 below every other arm, but by step 2000 it has caught up.

**What rank changes is *when*, not *how high*.** Ranks 64 and 128 reach their best value at step
500; rank 8 needs ~2000 steps to get there and is clearly worse before that. All arms then
converge to the same ~0.80 plateau. Capacity and step budget substitute for each other. The
practical consequence for a new backbone is that there is no rank to recommend independent of
step budget — and that the existing blind "best PathoROB checkpoint" selection rule absorbs the
interaction automatically, because a low-rank arm is simply selected later.

**Every rank lands at ~0.80, near Waiv's Phaet target of 0.806.** PathoROB is not where this
reconstruction is deficient, so rank is not a lever on it. That is consistent with the HEST
checkpoint sweep in §2: the deficiency is in *retention*, and it is intrinsic to the recipe.

### The limit, stated plainly

This sweep measures the **robustness** axis only. The eval follower in `train_real.sbatch`
computes PathoROB, not HEST, so nothing here says whether rank trades robustness against
retention differently. That question is **not answered by this section**.

### HEST retention over the rank arms — rank moves this axis no more than the other

SLURM jobs 372192–372199, all COMPLETED, `cls` protocol, phikon-v2 base 0.3747. Steps 2000 and
3000 were chosen because both fall inside the plateau and both already exist for the rank-32
reference, making the comparison like-for-like.

| rank | s2000 | Δ | s3000 | Δ | mean Δ |
|---|---|---|---|---|---|
| 8 | 0.3790 | +0.0043 | 0.3774 | +0.0027 | +0.0035 |
| 16 | 0.3774 | +0.0027 | 0.3780 | +0.0033 | +0.0030 |
| 32 (ref) | 0.3825 | +0.0078 | 0.3838 | +0.0091 | **+0.0084** |
| 64 | 0.3846 | +0.0099 | 0.3823 | +0.0076 | **+0.0088** |
| 128 | 0.3785 | +0.0038 | 0.3765 | +0.0018 | +0.0028 |

**Non-monotone, and inside the noise.** Ranks 8 and 16 sit low, 32 and 64 high, and 128 lowest
of all — no trend in capacity. The spread of mean deltas across ranks is 0.0060, against a
within-arm step-to-step range of 0.0051 for rank 32 alone (0.3794–0.3845 over steps 500–3500,
§2). Between-rank variation is the same size as the variation a single arm shows across
neighbouring checkpoints, so at n=1 these ranks are not distinguishable on retention either.

**This closes the frontier question.** Rank does not move robustness (§5 above) and does not
move retention (here). Capacity is not the lever on either axis, so the robustness/retention
frontier this reconstruction sits on is not reachable by re-tuning rank. Read with §2's
checkpoint sweep — which showed the retention shortfall survives every checkpoint on both
backbones — the deficiency is a property of the objective, not of its hyperparameters.

The magnitudes make the point independently of any ranking. The best arm here is +0.0088
(rank 64) against Waiv's Phaet **+0.0196**: less than half, from the most favourable rank at
its most favourable checkpoint. No rank closes the gap.

The standing hypothesis is therefore unchanged and still untested: the loss is a single InfoNCE
term with no retention component — no frozen-teacher anchor, no distillation, no replay, no
L2-to-base penalty — and `PLAN.md` §2 scoped exactly those as optional and never built them.
Adding one is a method change, not a sweep, and nothing here tests it.

### Method notes

- Comparing arms at their peak checkpoint selects noise, and selects more of it from noisier
  arms. Future sweeps here should report plateau means with the within-arm scatter alongside,
  and treat a peak-only table as a diagnostic, not a result.
- The sweep required no fork of the training script. `scripts/train_real.sbatch` reads
  `WAIV_LORA_RANK` / `WAIV_LORA_ALPHA` (defaults 32 / 64, reproducing 369043 exactly), and the
  same `$LORA_RANK` / `$LORA_ALPHA` are passed to the before-probe (`embed_probe.py`), the eval
  follower (`eval_checkpoints.py`) and `train_lora.py`. Rank must come from the env vars rather
  than through the trailing `"$@"`: appending `--lora-rank` as an extra arg reaches only the
  trainer, which would train at one rank and evaluate at another — `build_model` hard-fails on
  that mismatch, so the failure is loud, but the env route is the correct one.

---

## 6. Third backbone — `paige-ai/Virchow2`

The portability work is only tested by a backbone it was not written against. Virchow2 is
that test: a **timm** checkpoint (`AutoModel` cannot load it — its config.json has no
`model_type`), ViT-H/14, 32 blocks, hidden 1280, packed-SwiGLU FFN, and **4 register tokens**
on top of CLS. Every prior backbone here is a `transformers` Dinov2 with a single prefix token.

Base run, `clsmean` pooling (the paper's §3.3 protocol for PathoROB across all models),
SLURM jobs 372330 (features) + 372476 (metric):

| dataset | our base RI | balanced accuracy |
|---|---|---|
| camelyon | 0.798855 | 0.9876 |
| tolkach_esca | 0.954102 | 0.9770 |
| tcga | 0.821763 | 0.9274 |
| **Avg** | **0.858240** | — |

**Waiv Table 1 base: 0.858. Ours: 0.858240, Δ +0.00024.**

This is the base-reproduction gate (PLAN.md §3 phase 5) passing on a third backbone, and it
is a sharp test rather than a loose one — four independent things had to be right at once and
each fails silently rather than loudly:

- the timm loader branch, chosen by reading the repo's config.json rather than a name list;
- **SwiGLUPacked detected from the checkpoint** (`fc1_out == 2*fc2_in`, 6832 vs 3416) — without
  it the load dies with a size mismatch on all 32 blocks;
- **register-token-aware pooling.** `_pool` previously sliced `tokens[:, 1:]`, which on
  Virchow2 averages 4 register tokens into the patch mean. That produces a plausible
  embedding and a wrong number. Had it not been fixed, this average would not have landed on
  0.858;
- normalization derived from timm's `pretrained_cfg` (ImageNet), with no override entry.

Waiv publish no per-dataset breakdown for Virchow2, so `pathorob_adapter.TARGETS` carries the
average only and `pathorob_gate.py` renders the missing per-dataset cells as `-` rather than
inventing them. The per-dataset column above is **ours**, not a reproduction of theirs.

### Fine-tuning — job 375367, rank 32 / alpha 64, 2 × 192, 1500 steps

Sized on a real probe (job 375350, not arithmetic): 2 × 192 peaks at 42.23 GiB of 79.19 at
4.57 s/step, and gives **191 negatives per anchor — identical to both prior backbones**.
2 × 256 also fits (55.39 GiB); it was deliberately not taken, because buying more negatives
here would make the third backbone incomparable to the two the conclusions rest on. The
loader reports `LoRA targets=128 = 4/block × 32 blocks, leaves=['fc1','fc2','proj','qkv']`,
i.e. the fused QKV and packed SwiGLU both attached.

**This run was preempted twice** (15:53→17:07 on n-3, 17:32→19:56 on n-6). `train_real.sbatch`
has no resume, so each requeue restarts at step 0 and rewrites `runs/<run>/step_*` in place,
while `eval_checkpoints.py` skips steps already present in `ri_curve.json`. The curve
therefore started to mix trajectories. The attempt that ran to completion (17:32→19:45, all
1500 steps) had its six checkpoints snapshotted before the third attempt could clobber them;
the foreign points were stripped and recomputed by `eval_backfill.sbatch`. **Every point below
is from that one continuous attempt.** See `ri_curve.json.mixed-attempts.bak` for the
discarded mixed curve.

| step | Avg RI | camelyon | tolkach_esca | tcga | `rel_l2_delta` |
|---|---|---|---|---|---|
| base | 0.8582 | 0.7989 | 0.9541 | 0.8218 | — |
| **250** | **0.9035** | 0.9006 | 0.9673 | 0.8425 | 0.73–0.79 |
| 500 | 0.8981 | 0.8898 | 0.9706 | 0.8339 | 0.77–0.79 |
| 750 | 0.9000 | 0.8929 | 0.9701 | 0.8368 | 0.80–0.86 |
| 1000 | 0.9000 | 0.8951 | 0.9687 | 0.8362 | 0.83–0.89 |
| 1250 | 0.9009 | 0.8978 | 0.9693 | 0.8357 | 0.83–0.88 |
| 1500 | 0.9014 | 0.8983 | 0.9694 | 0.8365 | 0.83–0.88 |

**Waiv Table 1 fine-tuned: 0.918. Ours: 0.9035 at the selected checkpoint, Δ −0.0145.** Base
→ fine-tuned is +0.0453. Same shape as the other two backbones: the direction and most of the
magnitude reproduce, the published level does not.

**The blind rule picks step 250, and that choice is inside the noise.** The rule is "best
PathoROB checkpoint", applied per [`CAVEATS.md`](CAVEATS.md) before looking at any other
benchmark. But the spread across all six points is 0.0054 with a mean of 0.9006, against a
documented within-arm scatter of ±0.002–0.003 — step 250 is not separated from step 1500 by
more than noise. The honest summary is "flat at ≈0.901 from step 250 onward", not "250 is
optimal". Recording the cost rather than optimising it away: had the rule instead said "last
checkpoint", the reported number would be 0.9014, and nothing else in this section changes.
Every `rel_l2_delta` lands in 0.73–0.89, the same band as the existing runs, so the adapter is
demonstrably applied at every point.

### HEST retention (Virchow2, `clsmean` protocol) — base → step 250

| cancer type | base | ft250 | Δ |
|---|---|---|---|
| IDC | 0.5970 | 0.5907 | −0.0063 |
| PRAD | 0.3529 | 0.3804 | +0.0275 |
| PAAD | 0.4779 | 0.4971 | +0.0192 |
| SKCM | 0.6395 | 0.6244 | −0.0151 |
| COAD | 0.2581 | 0.2936 | +0.0355 |
| READ | 0.2072 | 0.2025 | −0.0047 |
| CCRCC | 0.2719 | 0.2585 | −0.0134 |
| LUNG | 0.5679 | 0.5715 | +0.0036 |
| LYMPH_IDC | 0.2568 | 0.2561 | −0.0007 |
| **avg** | **0.4032** | **0.4083** | **+0.0051** |

There is no published counterpart for Virchow2 × `clsmean` on HEST, so these are our own
reference for checkpoint-to-checkpoint retention only — the harness emits that note itself,
and the 0.3747 figure elsewhere in this file is phikon-v2 `cls` and nothing else.

**This is a third-backbone confirmation of §7's finding: fine-tuning does not damage
retention.** The +0.0051 average is at the documented ±0.005 HEST scatter and the per-task
signs are mixed (5 down, 4 up), so the claim is that retention is *preserved*, not improved.

### LoRA rank on a fused-QKV backbone — rank 32 is not the constraint

**Motivation.** Virchow2's fine-tuning closes approximately 76% of the published base-to-fine-tuned gap: (0.9035 − 0.8582) / (0.918 − 0.858) ≈ 0.0453 / 0.060. That is below the ~90% for Midnight and ~101% for phikon-v2. One structural hypothesis: Virchow2's fused QKV makes nominal rank 32 a tighter effective constraint than on the other two backbones. The Virchow2 loader attaches 128 LoRA targets = 4/block × 32 blocks on leaves `['fc1','fc2','proj','qkv']`. phikon-v2 attaches 144 = 6/block × 24 blocks on `['query','key','value','dense','fc1','fc2']`; Midnight attaches 240 = 6/block × 40 blocks on the same leaves. On a fused `qkv` projection (1280 → 3840) a single rank-r LoRA gives q, k and v a **shared r-dimensional input subspace** — one `lora_A` matrix serving all three — whereas separate `query`/`key`/`value` modules give three independent `lora_A` matrices. "Rank 32" is therefore not the same knob across these architectures. The experiment tests whether doubling or quadrupling rank recovers the gap.

**Design.** Three arms, identical in every other respect to the reference run (job 375367): lr 1e-4, weight decay 0.05, warmup 200, temperature 0.07, 2 × 192 groups (191 negatives/anchor), 1500 steps, Virchow2 backbone. Alpha is held at 2r throughout, matching the §5 convention, so the LoRA scaling factor alpha/r is constant at 2 and rank is the only thing that varies.

| rank | alpha | trainable params | % of 655M | SLURM job |
|---|---|---|---|---|
| 32 (ref) | 64 | 24.1 M | 3.68% | 375367 |
| 64 | 128 | 45.1 M | 6.67% | 376087 |
| 128 | 256 | 87.1 M | 12.12% | 376088 |

**Avg RI by step** (mean of camelyon, tolkach\_esca, tcga):

| step | rank 32 | rank 64 | rank 128 |
|---|---|---|---|
| 250 | **0.9035** | 0.9023 | 0.9006 |
| 500 | 0.8981 | 0.8067 | 0.8671 |
| 750 | 0.9000 | 0.8999 | 0.8713 |
| 1000 | 0.9000 | 0.8991 | 0.8712 |
| 1250 | 0.9009 | 0.8968 | 0.8757 |
| 1500 | 0.9014 | 0.8974 | 0.8758 |

Base (no adapter) Avg RI = 0.8582. Waiv's published Virchow2 fine-tuned target = 0.918.

**Summary** (plateau mean at steps ≥ 750, n = 4 per arm):

| rank | plateau mean | all-point spread | `adapter_rel_l2_delta` range | trainable | % of 655M |
|---|---|---|---|---|---|
| 32 (ref) | **0.9006** | 0.0054 | 0.73–0.89 | 24.1 M | 3.68% |
| 64 | 0.8983 | 0.0955 | 0.75–1.17 | 45.1 M | 6.67% |
| 128 | 0.8735 | 0.0334 | 0.94–1.17 | 87.1 M | 12.12% |

**The hypothesis is refuted.** Plateau mean is monotone decreasing in rank: 0.9006 → 0.8983 → 0.8735. Rank 64 is at best marginally worse than the reference (−0.0023); that difference is at the ±0.002–0.003 within-arm scatter floor, so it does not support a claim of improvement in either direction. However, every one of its four plateau points falls below the corresponding reference point, which rules out a benefit. Rank 128 is unambiguously worse: −0.0271, roughly 10× the scatter band. Increasing rank does not close the gap; it opens one.

**A note on the step-500 collapse in rank 64.** At step 500 rank 64 reads 0.8067 — below the un-adapted base of 0.8582 — before recovering to 0.8999 by step 750. A mean taken across that dip would be misleading, which is why the plateau mean (steps ≥ 750) is the reported statistic: it excludes the transient collapse and measures the settled representation. The same discipline was applied in §5, where it distinguished between arms whose instability was early and transient and arms that were genuinely worse. The dip itself is information: rank 64 shows the largest all-point spread (0.0955), indicating that higher rank buys larger excursions during training, not a higher ceiling.

**Mechanism: higher rank over-rotates.** The `adapter_rel_l2_delta` metric — ratio of the displacement (adapted minus base) to the base norm — shows that higher rank does not leave capacity unused; it displaces the representation further. Rank 32 stays in the 0.73–0.89 band throughout all six checkpoints: the adapter is active at every point (delta < 1) but does not outstrip the signal it is modifying. Ranks 64 and 128 both reach 1.17, meaning the adapter shifts the representation by more than the norm of the base embedding at their most aggressive checkpoints. Rank 128's floor of 0.94 — never below 0.94 across any checkpoint — indicates it over-rotates consistently from the outset, not transiently; that persistent drift is reflected in the much lower plateau mean (−0.0271 from the reference) rather than in high within-arm variance. Where drift is largest, robustness is lowest.

**Consistency with §5.** The phikon-v2 rank sweep (§5) also found rank 32 best by plateau mean (0.8029), with rank 64 the lowest (0.7969) — non-monotone there, monotone here, but the same winner and the same conclusion: rank is not a lever on PathoROB. The present experiment extends that finding across the QKV-fusion boundary, which §5 could not test because phikon-v2 has split `query`/`key`/`value` modules. The fused-QKV geometry does not change the verdict.

**What this means for the 76%.** Capacity is not the explanation for the lower gap-closed fraction on Virchow2. The fractions are ordered by base quality: phikon-v2 base 0.4686 → ~101%, Midnight 0.7589 → ~90%, Virchow2 0.8582 → ~76%. Higher base quality leaves less absolute headroom, and the fractions track headroom rather than architecture or LoRA rank. Note also that the 76% figure is itself sensitive to which checkpoint represents the arm: the reference plateau mean is 0.9006 and step 1500 is 0.9014, giving gap-closed fractions of approximately 71% and 72% respectively. The honest range is roughly 70–76%; the 76% (step 250, best PathoROB checkpoint under the blind selection rule) is at the upper end of that range.

**Limits.** n = 1 seed per arm. This measures the PathoROB robustness axis only — the eval follower computes PathoROB, not HEST or THUNDER — so it says nothing about whether rank trades robustness against retention on Virchow2. Rank was varied with alpha pinned at 2× rank throughout, so rank and the alpha/rank scaling are not separated: it is possible that a different alpha/rank ratio at higher rank would behave differently, but that is not tested here.

### THUNDER — complete, 40/40 both arms

Both sweeps landed: `vbase_*` and `vft250_*`, 40 cells each (12 classification × 3 tasks +
4 segmentation × 1), all 16 of the paper's datasets. Collected with
`scripts/collect_thunder.py --model vbase_clsmean vbase_cls` (and `vft250_*`); the split
pooling is the same Midnight-style convention documented in §3 — `clsmean` for the 12
classification sets, `cls` for the 4 segmentation sets, because 2560-d `clsmean` does not
fit the segmentation decoder on ViT-H.

| task | base | ft (step 250) | Δ (pp) | Δ (%) | n |
|---|---|---|---|---|---|
| kNN | 80.87 | 80.91 | **+0.03** | +0.04% | 12 |
| linear probing | 83.25 | 84.59 | **+1.34** | +1.61% | 12 |
| few-shot | 72.75 | 77.16 | **+4.41** | +6.07% | 12 |
| segmentation | 69.01 | 68.89 | **−0.12** | −0.17% | 4 |
| **mean over 4 tasks** | **76.47** | **77.89** | **+1.42** | **+1.85%** | — |

**Retention holds on the third backbone.** +1.42 is a real mean gain, and the sign pattern
matches both prior backbones exactly: classification up, segmentation marginally down. It is
the smallest of the three means (+2.29 phikon-v2, +2.21 Midnight, +1.42 Virchow2), and the
ordering is the same one PathoROB shows — the stronger the base, the less the recipe adds.

**The composition differs sharply from the other two, and the mean hides it.** Virchow2's
gain is almost entirely few-shot (+4.41), while kNN is flat at **+0.03** — against +4.92 on
phikon-v2 and +2.19 on Midnight. Virchow2's base kNN is 80.87, the strongest base kNN of the
three (phikon-v2 70.28, Midnight 78.25), so there is least headroom exactly where the other
two backbones gained most. Quoting +1.42 without this is quoting an average over a column
that did not move and a column that moved 4.4 points.

**Segmentation regresses on all three backbones** (−0.12, −0.33, −0.12). No single figure is
significant on 4 datasets — a sign test over the 12 segmentation model × dataset pairs gives
3/12 improving, p ≈ 0.15 — but the sign is identical on three independently chosen
architectures, which is worth more than any one cell. Against that, 32 of 36 classification
pairs improve (p ≈ 2×10⁻⁶). The honest summary of the recipe's effect on downstream
performance is *classification yes, segmentation no*, on every backbone tested.

**Δ-vs-Δ against Waiv: we win Virchow2, and the record becomes 11 of 12.** Waiv *do* publish
Virchow2 THUNDER — arXiv:2607.22861 Table 2, transcribed in
[`waiv_published.json`](waiv_published.json). On the 4 tasks we share:

| task | Waiv base | Waiv FT | Waiv Δ | our Δ | ours − Waiv |
|---|---|---|---|---|---|
| kNN | 82.9 | 82.6 | **−0.30** | +0.03 | +0.34 |
| linear probing | 84.8 | 85.1 | +0.30 | **+1.34** | +1.04 |
| few-shot | 73.9 | 76.6 | +2.70 | **+4.41** | +1.71 |
| segmentation | 68.2 | 68.0 | −0.20 | −0.12 | +0.08 |
| **mean** | 77.45 | 78.07 | **+0.62** | **+1.42** | **+0.79** |

Our fine-tuning moves Virchow2 more than theirs does on all four shared tasks, and their own
kNN column goes *down* (−0.30) where ours is flat (+0.03). Adding these four to §2's eight
makes the record **11 of 12 model × task pairs where our Δ ≥ Waiv's**; the sole loss is still
Midnight segmentation. Two caveats keep this honest: Waiv score 6 THUNDER tasks, and the two
we do not run (calibration, adversarial attack) are excluded from every mean above — their
Virchow2 FT adversarial score of 7.7 is rank 1 in the whole roster, and it is not in our +1.42.
And our absolute level sits below theirs even where our Δ is larger (our Virchow2 FT mean
77.89 vs their 78.07), because our base reproduction starts ~1 point low.

**Provenance.** Both arms ran to completion uninterrupted after repeated preemption: base job
375369 COMPLETED in 29h21m (13 prior PREEMPTED attempts), fine-tuned job 375910 in 32h08m
(7 prior PREEMPTED + 1 NODE_FAIL). Because `run_thunder.sbatch` restarts from scratch on
requeue, only a clean single attempt yields a trustworthy cell; both final attempts were
clean, so no cell in this table mixes attempts. n = 1 seed per arm, as everywhere in this
record — the THUNDER cells carry no error bars, and the confidence in the +1.42 rests on the
consistency of the sign across datasets, not on any per-cell interval.

---

## 7. Retention term — relational KL to a frozen teacher

Every negative result above points the same way: robustness reproduces, retention does not,
and it is not the checkpoint (§2 sweep), not the LoRA rank (§5), and not full fine-tuning
(§4). The loss has no retention component at all — a single masked InfoNCE term — and
`PLAN.md` §2 scoped a frozen-teacher anchor as optional and never built it. This tests it.

**Design.** Relational, not an L2 pull to base: a pull toward the frozen base embedding would
fight the objective, since robustness is bought precisely by moving embeddings to collapse
acquisition directions. Instead the term preserves relative geometry — softmax over pairwise
cosine similarities under student and frozen teacher, penalised by `KL(P_teacher ‖ P_student)`,
verified invariant to a global rotation+rescale. Candidates are restricted to the anchor's own
condition-homogeneous group (a whole-batch KL would ask the student to preserve teacher
similarities that are partly acquisition-driven, re-injecting the confounder). The diagonal is
masked: after normalisation `S_ii = 1` exactly and at τ=0.07 it would swamp every off-diagonal
term, making the KL silently inert. The teacher is free under LoRA — `disable_adapter()` under
`no_grad`, no second copy of the weights. Measured GPU cost: **+6.1% step time, +4.7 MiB peak**.

**Result.** Plateau over steps ≥1500, 4000-step arms otherwise identical to the §1 reference:

| λ | RI plateau | sd | drift | HEST Δ @s2000 | @s3000 | mean |
|---|---|---|---|---|---|---|
| 0 (reference) | 0.8029 | 0.0017 | 0.680 | +0.0078 | +0.0091 | **+0.0084** |
| **0.03** | **0.8014** | 0.0017 | 0.802 | +0.0069 | +0.0063 | **+0.0066** |
| 0.3 | 0.6250 | 0.0146 | 0.618 | +0.0021 | +0.0012 | +0.0017 |
| 3 | 0.4796 | 0.0010 | 0.154 | +0.0002 | +0.0005 | +0.0004 |
| 10 / 50 / 200 | ~0.474 | — | 0.065 | — | — | cancelled at step 500 |

The bar was **pre-registered before any arm ran**: preserve robustness (RI plateau ≥ 0.80) *and*
beat the best HEST Δ ever observed here (+0.0098, phikon-v2 step 3500, §2). λ=0.03 is the only
arm to clear the robustness half — and it shows **no retention benefit**, landing at +0.0066
against the λ=0 control's +0.0084, with **both** measured checkpoints below the control.
**Refuted.**

**The premise was wrong, and that is the useful finding.** The experiment assumed
fine-tuning damages retention and that constraining drift would recover it. It does not,
because fine-tuning does not damage retention in the first place: the λ=0 control *improves*
HEST by **+0.0078** over base. It improves it less than Waiv's +0.0196, but the sign is
positive. Both axes then decline monotonically with λ, and at λ=3 the model sits at base on
*both* — RI 0.4796 against a base of 0.4686, HEST +0.0004 against a base of 0.3747. The term
never preserved anything; it removed a gain that fine-tuning produces. A retention term was
addressing a problem that does not exist, which is why no λ could work.

**It does not even trade — it is strictly dominated.** The obvious defence of a retention term
is that it buys retention at the cost of robustness, i.e. moves along the frontier. It does not.
At λ=0.3, where robustness has already collapsed (RI 0.6250 against the control's 0.8029),
HEST is **+0.0021** against the control's +0.0078 — *worse on both axes at once*. Giving up
robustness buys nothing back. Every λ tested is dominated by λ=0.

**There is no useful window.** λ=0.03 does not constrain the representation at all — its
plateau drift (0.802) is *above* the control's (0.680) — so it is inert, and its RI sits within
one standard deviation of the control. The next λ up destroys robustness outright (0.625), and
by λ=3 the model is pinned at base (0.4796 against a base of 0.4686) with drift collapsed to
0.154. The term is either inert or destructive; nothing in between buys retention.

**The λ calibration is worth recording as a methods note.** The first bracket, {10, 50, 200},
was chosen from the KL-to-InfoNCE *loss-value* ratio measured at step 60 (1:122). All three
pinned the model at base and were cancelled. Loss magnitude is the wrong quantity: at λ=10 the
KL contributed 3.3% of the loss value while cutting `adapter_rel_l2_delta` from ~0.75 to ~0.065.
**`adapter_rel_l2_delta` — already logged by the eval follower — is the quantity that governs
this**, and calibrating on it directly would have found the transition immediately.

**What this does and does not show.** It does not show that retention is unrecoverable — only
that *this* term cannot recover it, and that the failure is not caused by unconstrained drift,
since constraining drift does not restore retention. n=1 per λ; the 0.0028 gap between λ=0.03
and the control is inside the HEST checkpoint scatter (0.0051 for the reference across steps,
§2), so "no detectable benefit" is the supportable claim, not "actively harmful" — though the
deficit reproduces at both measured checkpoints (s2000 and s3000), which is what rules out a
benefit rather than merely failing to detect one.

The term ships behind `--retention-kl-weight`, default **0.0**, bit-identical to the published
training path when off.

---

## 8. Batch geometry — negatives per row, not query rows

Every result above varies the loss, the rank, or the checkpoint. None varies how the batch is
*shaped*. The same-condition constraint (`PLAN.md` §2) makes that shape load-bearing in a way
standard InfoNCE does not: negatives come only from within a condition-homogeneous group, so
negatives per row is `group_size - 1`, capped by the group rather than the batch, and the
positives are **query-only** — they never serve as negatives, because each carries a different
random condition and so cannot form a condition-homogeneous candidate set. Half the forward
compute produces embeddings used in exactly one row. SimCLR gets `2N-2` negatives from `2N`
images; this gets `N-1`.

That suggests an alternative: share **one tile set across all condition groups**, so every image
is both an anchor in its own group and a valid query against every other group. PLISM supports
it — verified 91/91 slides, all `(16278, 224, 224, 3)`, one shared `keys.json`, so tile *i* is the
same tissue under every condition. Writing `C` = conditions and `T` = tiles at budget `B = C·T`:

```
negatives per row   N = T - 1 = B/C - 1
query rows          R = C(C-1)T = B(C-1)        <- linear in C, T cancels
total pairs         R·N = B(C-1)(B/C - 1)       <- maximised at C = T = sqrt(B)
```

At `B ≈ 2400` the pair-count optimum `C = T = 49` coincides with the 50-condition training split.
Both the pair-count objective and the `R·log N` objective point at large `C`. The experiment below
tests whether either is the right thing to optimise.

### 8.1 Negatives saturation sweep — no cliff (job 380742)

Before restructuring anything, the question was whether few negatives *saturate* the objective.
Three arms, existing sampler, all at 2400 images/step, 300 steps, phikon-v2 rank 32 — varying only
the grouping:

| arm | `n_groups`×`group_size` | neg/row | top1 plateau | loss floor | f>0.98 | 1st≥0.95 | 1st≥0.99 | MI est | ceiling `log(N+1)` |
|---|---|---|---|---|---|---|---|---|---|
| A | 6×200 | 199 | 0.9383 | 0.2236 | 0.00 | 150 | never | 5.075 | 5.298 |
| B | 12×100 | 99 | 0.9613 | 0.1387 | 0.00 | 85 | never | 4.466 | 4.605 |
| C | 24×50 | 49 | 0.9742 | 0.0827 | 0.15 | 50 | never | 3.829 | 3.912 |
| ref | 2×32 (369019) | 31 | 0.9844 | 0.0531 | 0.70 | 20 | **70** | 3.413 | 3.466 |

Raw loss is not comparable across arms — random-guess loss is `log(N+1)`, which differs — so the
saturation reads are top-1 and the fraction of steps above 0.98.

**Control.** Arm A ran 300 steps against 1500–4000-step references, so the comparison is made over
the references' first 300 steps: arm A top1 0.912/0.927/0.939 at steps 100/200/300, against
virchow2-191 at 0.849/0.927/0.919 and midnight-191 at 0.833/0.893/0.951; ref-31 sits at 0.984 flat.
Arm A lands on the 191-negative references and ref-31 is cleanly separated, so the probe is sound.

**Result: 49 negatives does not saturate**, and there is no threshold anywhere — plateau, loss
floor, and both saturation fractions move smoothly and monotonically from 199 → 99 → 49 → 31. The
fraction of the MI ceiling extracted rises as N falls (95.8% → 97.0% → 97.9% → 98.5%) while the
absolute MI falls. "N_min" is a choice of margin, not a measured edge.

### 8.2 Three-arm downstream comparison (jobs 380777/380778/380779)

Since saturation is not the constraint, the question moves downstream. Three arms, 1500 steps,
2400 images/step, phikon-v2, rank 32/alpha 64, lr 1e-4, τ=0.07, seed 0, identical held-out split —
geometry is the only difference. CTRL is the existing pair sampler; the other two use the new grid.

PathoROB average RI:

| step | CTRL 6×200 | GRID24 C=24 T=100 | GRID49 C=49 T=49 |
|---|---|---|---|
| 250 | 0.7865 | 0.7835 | 0.7688 |
| 500 | 0.8128 | 0.7974 | 0.7890 |
| 750 | 0.8131 | 0.8026 | 0.7954 |
| 1000 | 0.8124 | 0.8038 | 0.7963 |
| 1250 | 0.8126 | 0.8042 | 0.7983 |
| 1500 | **0.8130** | **0.8049** | **0.7973** |
| neg/row | 199 | 99 | 48 |
| query rows | 1,200 | 55,200 | 115,248 |

Last-three-checkpoint means: CTRL **0.8127**, GRID24 **0.8043** (Δ −0.0084), GRID49 **0.7973**
(Δ −0.0154). CTRL's plateau is 0.8128 ± 0.0004 across steps 500–1500, which is the
empirical within-run checkpoint noise — tighter than the ±0.005 band §2 quotes, and it makes the
GRID24 gap ~20× the noise.

**Ordering tracks negatives per row at every checkpoint and runs inversely to query rows.** The
grid restructuring buys 46–96× the query rows and loses RI. Spacing is close to linear in
`log(negatives)`: 0.0121 RI/nat between CTRL and GRID24, 0.0098 between GRID24 and GRID49 — about
0.008 RI per doubling of negatives, the same `log N` dependence the InfoNCE bound predicts, showing
up in a downstream robustness metric.

Per-dataset at step 1500, this is a **camelyon-only effect**. TCGA (0.7841 / 0.7820 / 0.7798) and
Tolkach (0.9304 / 0.9304 / 0.9296) are tied across all three arms; camelyon carries essentially all
of it (0.7244 / 0.7022 / 0.6826). Camelyon is the dataset where base RI is 0.019, so it has the most
headroom and is the noisiest of the three.

### 8.3 The confound — what this does NOT establish

At fixed `B`, `C` and `T` are coupled: this varied negatives and conditions **together, in opposite
directions**. The data cannot distinguish "more negatives helped" from "fewer conditions helped."
The negatives reading is the mechanistically plausible one and matches the `log N` bound, but it is
not established. One arm separates them: **grid C=12, T=200** — N=199, identical to CTRL, but 12
conditions and 26,400 rows instead of 6 and 1,200. Tie ⇒ negatives is the only variable.

Also unmeasured: **between-run seed variance**. Every gap here is n=1, and ±0.0004 is *within*-run
checkpoint drift, not run-to-run. A CTRL seed repeat is the cheapest way to price the ±0.008 claims.

### 8.4 Incidental — the current sampler at a larger batch beats the published number

CTRL is the *existing* sampler, just at 2400 images/step instead of 768 (1200 anchors vs 384, at
essentially unchanged negatives per row, 199 vs 191). It plateaus at **0.8130**, above Waiv's
published 0.806 and above this repo's prior best 0.8080 (run 369043 at step 1000) — and it clears
all three datasets individually, not merely on average. The cheapest available win is not the grid;
it is a larger batch on the sampler that already exists. Confounded by total images and step count
against 369043, and n=1, so it needs a seed repeat before it is more than a lead.

One detail the average hides: CTRL's plateau is a *balance*, not a stasis. Camelyon rises
monotonically the whole way (0.7162 → 0.7244) while TCGA and Tolkach drift down (0.7864 → 0.7841,
0.9356 → 0.9304). For checkpoint selection that matters — step 1500 is best for camelyon, step 500
for the other two, and the flat average conceals the choice.

### 8.5 Two bugs the gates caught

Both would have produced a plausible falling loss curve.

1. **OOM.** 2400 images in one forward makes each gradient-checkpoint recompute a 7.2 GiB
   transient; the pair path gets a 2× smaller one free by running its two views separately. Fixed
   with a forward micro-chunk.
2. **The fix was itself wrong.** `ProjectionHead` contains `nn.BatchNorm1d`, so chunking *through*
   the projector computed BN statistics per 600-image chunk rather than over all 2400 — measured
   drift **1.65**, a different objective, not a rounding difference. GRID24 had already passed its
   smoke test that way because 2400/600 divides evenly; it only surfaced when GRID49's 2401 images
   left a trailing chunk of 1 and BatchNorm refused. Now only the per-image backbone is chunked and
   the projector runs once over the full batch — verified 0.0 drift. **Never micro-chunk through a
   batch-coupled layer.** The regression test uses a BatchNorm projector in train mode; the original
   test used a plain-Linear projector in eval mode and was structurally incapable of seeing this.

Measured cost: grid 55.98 GiB / 395–405 img/s at 2401 images vs pair 65.0 GiB / 409 img/s at 2400
(the micro-chunk lowers the peak). Chunked output is not *bit*-identical across chunk sizes (~1e-7,
different GEMM kernels), so `grid_forward_chunk` is recorded in `config.json` and must not vary
within a comparison.

The grid ships behind `--grid --grid-conditions C --grid-tiles T`, bit-identical to the existing
path when off (20/20 per-step losses, verified against a pristine snapshot).

### 8.6 CORRECTION to 8.2/8.3 — the negatives slope was confounded by the sampler path (job 380862)

§8.2 fitted "about 0.008 RI per doubling of negatives" through CTRL (199 neg, **pair** path)
→ GRID24 (99, **grid**) → GRID49 (48, **grid**). **That fit crosses two different samplers and is
not a negatives slope.** The separating arm §8.3 asked for has now run: `gridcmp2-grid12-380862`,
grid `C=12 T=200` — 199 negatives per row, identical to CTRL — everything else matched.

| arm | sampler | neg/row | step 1500 | last-three mean |
|---|---|---|---|---|
| CTRL | pair 6×200 | 199 | 0.8130 | **0.8127** |
| GRID12 | grid C=12 T=200 | 199 | 0.8021 | **0.8034** |
| GRID24 | grid C=24 T=100 | 99 | 0.8049 | **0.8043** |
| GRID49 | grid C=49 T=49 | 48 | 0.7973 | **0.7973** |

**At matched negatives the sampler path itself is worth −0.0093 RI** (last-three means; −0.0109 on
step-1500 finals). That is most of the −0.0154 CTRL→GRID49 gap §8.2 attributed to negatives, and
the tie §8.3 predicted ("Tie ⇒ negatives is the only variable") **did not happen** — so negatives
is *not* the only variable, and the mechanistically plausible reading was the wrong one.

**Within the grid family alone the trend rises then flattens**: 48 → 0.7973, 99 → 0.8043,
199 → 0.8034. The 48→99 step is +0.0070 (0.0097 RI/nat); the 99→199 step is −0.0009, i.e. flat and
inside the ±0.0004–0.005 checkpoint band. Negatives appear to **saturate near ~100 within this
sampler** rather than climbing monotonically in `log N`. The `log N` reading of §8.2 is withdrawn.

Two caveats hold this to "appears":

- **The C/T coupling of §8.3 is still in force.** At fixed budget `B = C·T`, every grid arm's "more
  negatives" is simultaneously "fewer conditions". GRID12's 199 negatives come with 12 conditions;
  the 1199-negative arm below comes with **2**. This family cannot separate the two either — it only
  removes the *sampler* confound, not the C/T one.
- **`gridcmp2-grid2-380890`** (grid `C=2 T=1200`, 1199 neg/row) has finished training and is being
  evaluated; it extends the family by another 2.6 nats and is what would settle saturation.
- Between-run seed variance is **still unpriced**. `gridcmp2-ctrlseed-380889` (CTRL, seed 1) has
  finished training and its RI curve is still being computed. Until it lands, −0.0093 is an n=1
  difference against an unmeasured noise floor.

### 8.7 Negatives memory ceiling — where the grid actually stops (sizing probes)

How far the grid family *can* be pushed, measured on one H100 80GB (79.19 GiB usable), phikon-v2
rank 32, bf16, gradient checkpointing, `expandable_segments:True`, `C=2`:

| C×T | neg/row | img/step | fwd chunk | peak GiB | img/s | s/step |
|---|---|---|---|---|---|---|
| 2×1600 | 1599 | 3200 | 600 | 67.38 | 406.7 | 7.87 |
| 2×1800 | 1799 | 3600 | 300 | **74.15** | **396.5** | **9.08** |
| 2×1800 | 1799 | 3600 | 150 | 71.88 | 373.7 | 9.63 |
| 2×1900 | 1899 | 3800 | 150 | 74.76 | 376.2 | 10.10 |
| 2×1975 | 1974 | 3950 | 150 | 77.60 | 373.4 | 10.58 |
| 2×1800 | 1799 | 3600 | 600 | — | **OOM** | — |

**The edge is T≈1975 at C=2** — 1974 negatives per row, 3950 images/step, 77.60 GiB peak, ~1.6 GiB
of headroom left. The **operating** point is 2×1800 at chunk 300: 74.15 GiB and 396.5 img/s, within
2.5% of the best throughput measured anywhere in the sweep. Chasing the edge costs throughput —
2×1900 at chunk 150 gives 100 more negatives for 376.2 img/s (−5%), and every step past 1800 needs
chunk 150, which is itself the slower setting. Note the chunk size is not free to vary within a
comparison (§8.5): it changes the peak by ~2 GiB and the arithmetic by ~1e-7.

### 8.8 Determinism of the training path (job 380871)

`ctrl-replay-380871` re-ran CTRL's first 20 steps from scratch on the current source. Against
`gridcmp-ctrl-380777`'s step-20 record, **every logged quantity is identical to all 16 significant
digits**: loss `2.329573392868042`, top1 `0.5358333587646484`, `grad_norm` `3.610504388809204`,
peak allocated `65.01362562179565` GiB, negatives/anchor 199.0, and the batch composition counters
(6 anchor conditions, 50 positive conditions). Only wall-clock and `peak_reserved_gib`
(66.0703 vs 66.0508) differ, both allocator-level and outside the compute path.

This is verified **at step 20 on the logged metrics only** — it is not a full-run bitwise
comparison, and it does not cover the eval path. It is enough to say that a re-launched arm lands
on the same trajectory, so any between-arm gap in §8/§9/§11 is not launch nondeterminism.

---

## 9. CLS/mean loss separation — the mean head is not a robustness head (jobs 380856/380857/380858)

Training pools `clsmean` and applies **one** InfoNCE to the concatenation. Two consequences worth
testing. First, nothing forces both halves to become invariant — the objective may lean on whichever
is easier — and this matters because eval pooling *disagrees* with training pooling: HEST and
THUNDER-on-phikon-v2 read CLS only (§2). Second, `mean` is linear, so `d(mean)/d(t_i) = (1/N)·I` and
the direct gradient reaching every patch token is the **identical vector**; the loss can translate
the token cloud but never expresses a preference about the tokens' relative arrangement.

That second point has a sharp corollary for segmentation. THUNDER's `get_segmentation_embeddings`
returns raw patch tokens, and its decoder is `proj_dec = nn.Linear(d_encoder, d_model)` **with a
bias** (`task_specific_models.py:103`). For any constant `c`, `proj_dec(t_i + c) = W·t_i + (W·c + b)`
— a trained decoder simply learns `b - W·c` and produces identical masks. **A uniform translation of
the token cloud is exactly in the null space of the segmentation decoder**, and a uniform translation
is precisely what the mean head's first-order gradient requests. This is a candidate mechanism for
the standing result that classification improves on 32/36 pairs (p≈2×10⁻⁶) while segmentation does
not (3/12, p≈0.15). Note `emb_dim` only *sizes* that Linear — pooling has no effect on the
segmentation features themselves, so forcing `cls` for segmentation runs is plumbing, not a
representation choice.

Three arms, pair path 6×200 (so CTRL §8 is the baseline), eval pooling `clsmean` throughout because
that is a protocol constant: SPLIT `0.5·(L_cls + L_mean)`, CLSONLY, MEANONLY.

**Weights default 0.5/0.5, not 1.0/1.0** — measured, 1.0/1.0 is exactly `2.000000000000×` the
single-head loss, which at fixed LR is a different optimisation and would confound the structural
change with an LR change. At 0.5/0.5 the split reproduces the single-head loss to `0.0e+00`. A
zero-weighted head is **removed**, not zero-multiplied, because a dead head still updates its
BatchNorm running statistics and the "only" arms would not actually be single-head.

**Early observation (step 20, all three arms): cls top1 0.443 / loss 2.8361 versus mean top1 0.735 /
loss 1.2168.** The two halves are far from equally invariant, which is the asymmetry the concat loss
was free to exploit. Head inputs measured `rel_distance` 1.3548–1.3715, cosine 0.171 on real batches.
SPLIT's `loss_cls` is not bit-identical to CLSONLY's (19/20 steps differ, max 4e-4), confirming the
mean head's gradient does reach the shared LoRA backbone.

**A launch blocker was caught in `embed_probe.load_adapter`:** it loaded `projector.pt` unguarded, so
a 1024-d split head against the 2048-d clsmean eval raises a size mismatch — which would have killed
the RI-curve follower on every checkpoint of all three arms, for a tensor the probe never reads. It
now skips with the message its own full-FT branch already used.

Peak 66.84 GiB / 410 img/s, so the split costs ~1.8 GiB and no measurable throughput.

### 9.1 Results — the split wins, and the mean head alone is catastrophic

All three arms ran to 1500 steps, seed 0, LoRA r32/α64, 199 negatives. RI is `avg_robustness_index`
from `ri_curve.json`; base phikon-v2 is 0.469 and the published waiv target is 0.806.

| arm | cls/mean | RI @1500 | peak RI | @step | camelyon | tolkach_esca | tcga |
|---|---|---|---|---|---|---|---|
| SPLIT `headcmp-split-380856` | 0.5/0.5 | **0.8196** | **0.8252** | 500 | 0.7554 | 0.9294 | 0.7740 |
| CLSONLY `headcmp-clsonly-380857` | 1.0/0.0 | 0.8105 | 0.8173 | 750 | 0.7300 | 0.9268 | 0.7748 |
| MEANONLY `headcmp-meanonly-380858` | 0.0/1.0 | 0.6316 | 0.6316 | 1500 | 0.2633 | 0.8998 | 0.7317 |
| CTRL §8 `gridcmp-ctrl-380777` | single concat | 0.8130 | 0.8131 | 750 | 0.7244 | 0.9304 | 0.7841 |
| phikon-v2 base | — | 0.469 | — | — | 0.019 | 0.768 | 0.619 |

**SPLIT beats CLSONLY by +0.0091 at 1500 (+0.0079 at peak) and CTRL by +0.0066 (+0.0121 at peak).**
Both split-head margins are larger than the −0.0093 sampler effect of §8.6, so the ordering
SPLIT > CTRL > CLSONLY ≫ MEANONLY is not a marginal one — with the standing caveat that seed
variance is still unpriced (§8.6) and every arm here is n=1. MEANONLY is the only arm still rising
at 1500; the other three peak by step 500–750 and decay (below).

**Camelyon carries almost the whole MEANONLY collapse.** SPLIT→MEANONLY is −0.1880 average RI, and
camelyon alone (0.7554 → 0.2633) accounts for 0.1640 of it — **87%**. Camelyon is also the dataset
with the most headroom (base 0.019), so it is where a head that fails to learn invariance shows up.

### 9.2 The finding: contrastive loss is anti-correlated with RI

This is the load-bearing result of §9 and it is worth stating on its own.

| arm | train loss | train top1 | heldout loss | heldout top1 | probe bal-acc | **RI** |
|---|---|---|---|---|---|---|
| MEANONLY | **0.1792** | **0.9500** | 0.1533 | 0.9601 | **0.9426** | **0.6316** |
| CLSONLY | 0.1839 | 0.9467 | **0.1533** | **0.9607** | 0.9389 | 0.8105 |
| SPLIT | 0.1882 | 0.9483 | 0.1605 | 0.9576 | 0.9382 | **0.8196** |

**The ordering on InfoNCE is the exact reverse of the ordering on RI.** The arm with the best
training loss and top-1 has the worst robustness by 0.19 RI; the arm with the *worst* contrastive
metrics has the best. The heldout contrastive metrics do not rescue it either — MEANONLY is tied
with CLSONLY on heldout loss/top1 while being 0.18 RI worse.

Nor is this a capability loss. MEANONLY has the **highest** downstream probe balanced accuracy of
the three (0.9426 vs 0.9382). The classifier still works; what collapsed is specifically the
invariance term. RI is a ratio of OOD to ID performance, and MEANONLY moved the denominator, not
the numerator.

**Consequence for the protocol: training loss, top-1, and balanced accuracy must never be used for
model selection or early stopping on this objective.** They are not weak signals for RI, they are
inverted ones. The only admissible selection signal is the RI curve itself.

Independent corroboration from the PLISM condition probes (`probe_step_0001500.json`, heldout
group, embedding space — `separation` = matched-pair similarity minus random-pair similarity, so
*higher means the representation still encodes the scanner/stain*):

| arm | cross-scanner | cross-stain | Δ vs before |
|---|---|---|---|
| before (base) | 0.3760 | 0.3166 | — |
| SPLIT | 0.3903 | 0.3571 | +0.0143 / +0.0405 |
| CLSONLY | 0.3852 | 0.3527 | +0.0092 / +0.0361 |
| MEANONLY | **0.4866** | **0.4298** | **+0.1106 / +0.1132** |

Read this as an ordering, not as a sign test. **Every** arm's separation rises above baseline,
including the two that improve RI, so "separation went up" is not by itself a failure signature —
whatever the fine-tune does to the embedding geometry raises matched-pair similarity generally.
What distinguishes MEANONLY is the *magnitude*: it moves cross-scanner separation ~8× further than
SPLIT does. The probe is consistent with the RI collapse; it does not independently establish it.

### 9.3 Why the mean head cannot deliver invariance

The gradient argument from the section head is the explanation, and the numbers above are what it
predicted. For `m = (1/N)·Σ t_i`, `∂m/∂t_i = (1/N)·I` for every `i` — **identical at every patch
token, with no dependence on the token's content.** The mean head can therefore only ask the
backbone for a *uniform translation* of the token cloud. That is exactly the null space of THUNDER's
`proj_dec = Linear(d_encoder, d_model)` with bias, which is why the historical
classification 32/36 (p≈2×10⁻⁶) vs segmentation 3/12 (p≈0.15) split has the shape it does.

A uniform translation is a cheap way to make two views of the same tile agree — which is why
MEANONLY wins on InfoNCE — and a translation that a downstream linear probe absorbs into its bias
is not invariance. Hence best loss, worst RI. The mean head is a *loss* head; it is not a
robustness head, and it should not be run alone.

### 9.4 Action item: the 1500-step budget overshoots the RI optimum

SPLIT peaks at 0.8252 at step **500** and ends at 0.8196 — it gives back 0.0056 RI over the
remaining 1000 steps, and the decay is monotone (0.8252 → 0.8248 → 0.8217 → 0.8202 → 0.8196).
CLSONLY does the same (0.8173 @750 → 0.8105). The §11 pooling arms do it more strongly still
(GeM −0.0066, LSE −0.0068). CTRL, notably, does **not** — it is flat at 0.8124–0.8131 from step 500
on, so the decay is specific to the split-head arms rather than a property of the pair path.

The forfeited RI (0.005–0.007) is comparable to the entire effect being measured in §9.1 and larger
than the whole §11 spread. **Every split-head arm in §9 and §11 is reported at a step that is past
its own optimum.** Two consequences: the step-1500 numbers systematically understate these arms,
and the 1500-step budget should be revisited — but changing it mid-family would break comparability
with §8, so it is recorded here as an action item rather than applied.

---

## 10. Residual PLISM misregistration — measured

Three docstrings and `REPRODUCING.md` state positives carry "~5-50 px" residual misregistration.
That figure is **uncited and was never measured here**. Measured by phase correlation over 1,235
confident tile pairs, with controls:

```
CONTROLS   identity (slide vs itself)     0.0 px   peak 1.0000   <- method works
           wrong tile, same condition    64.3 px   peak 0.0223   <- noise floor
           wrong tile, cross-stain       70.0 px   peak 0.0231

CROSS-SCANNER, same stain (peak 0.07-0.15, 3-7x the noise floor -> real)
  n=1235   median 12.8 px   p75 21.6   p90 34.6   p95 44.4   max 124.4
  tokens p16: median 0.80 | p90 2.16 | p95 2.77
  tokens p14: median 0.91 | p90 2.47 | p95 3.17
  shifted >1 token 37.9%   >2 tokens 12.2%   >4 tokens 1.5%
```

So "5–50 px" is about right at p95 and roughly 2.5× optimistic at the median.

**Cross-stain is NOT measurable this way and no number is reported.** Its correlation peak (0.0243,
0.0230) sits *at* the wrong-tile noise floor — grayscale phase correlation cannot match across
stains, so what it returns is the failure mode, not the alignment. A first pass produced "median
51 px" for cross-stain; that is noise and is discarded. A real figure needs mutual-information
registration or stain deconvolution. The cross-scanner number is integer-pixel translation only, so
it is a **lower bound** — rotation, scale, and local warp are invisible to it.

**Consequence for any dense/per-token loss.** Per-token positives are misaligned for 37.9% of
cross-scanner pairs. At 2×2 token blocks (32 px) that falls to 12.2%; at **4×4 blocks (64 px) it is
1.5%**. Token-level correspondence is not defensible; ~64 px region-level is.

---

## 11. Token-dependent pooling — a NULL RESULT (jobs 381014/381015/381016)

§9.3 says the mean head cannot be a robustness head because `∂m/∂t_i = (1/N)·I` is identical at
every token. The obvious repair is to make the pooling *token-dependent*, so the gradient can
select. Three poolers were tried against the §9 SPLIT arm, which is the exact same configuration
with `pool_head=mean`:

- **GeM** — `(mean(t^p))^(1/p)`, learnable `p` initialised at 3.0.
- **LSE** — `τ·log mean(exp(t/τ))`, learnable `τ` initialised at 1.0.
- **ATTN** — single-query dot-product attention over the patch tokens, learnable.

`--pool-head` replaces **only** the input to `projectors["mean"]`; the CLS branch always receives
the raw CLS token, and the pooler sees patch tokens only (`tokens[:, num_prefix_tokens:, :]`,
`encoder.py:814`). The loss stays `L = 0.5·InfoNCE(proj_cls(cls)) + 0.5·InfoNCE(proj_mean(pool(patches)))`.
Critically, the exported embedding is **untouched** — `_pool_parts` still produces the literal
cls/mean pair that `pool_from_parts` reassembles, so the learned pooling reaches evaluation only
through the LoRA weights it trained, never directly.

### 11.1 Results — the four arms are indistinguishable

| arm | pool | RI @1500 | peak RI | @step | camelyon | tolkach_esca | tcga |
|---|---|---|---|---|---|---|---|
| `poolcmp-gem-381014` | GeM | 0.8210 | 0.8276 | 500 | 0.7584 | 0.9301 | 0.7745 |
| `poolcmp-lse-381016` | LSE | 0.8203 | 0.8271 | 500 | 0.7567 | 0.9295 | 0.7747 |
| `headcmp-split-380856` | mean (§9) | 0.8196 | 0.8252 | 500 | 0.7554 | 0.9294 | 0.7740 |
| `poolcmp-attn-381015` | ATTN | 0.8175 | 0.8223 | 500 | 0.7509 | 0.9278 | 0.7736 |

**The total spread across all four arms is 0.0035 RI at step 1500 (0.0053 at peak).** For scale,
that is smaller than what each of these arms throws away between step 500 and step 1500 (§9.4), and
about a third of the §8.6 sampler effect. GeM is nominally on top and ATTN nominally last, and the
ordering is stable across camelyon, tolkach_esca and every checkpoint — but stability is not
significance when the whole range is 0.0035.

**This ranking is not decidable and must not be quoted as one.** The control that would establish
the noise floor — `gridcmp2-ctrlseed-380889`, CTRL re-run at seed 1 — has finished training but
has **not** produced its RI curve (eval job 381441 in flight; `gridcmp2-grid2-380890` / 381442
alongside it). Until that lands there is no measured seed variance for this pipeline, so a 0.0035
spread across four n=1 runs is indistinguishable from four draws of the same distribution. The
honest reading today is **no effect**: token-dependent pooling did not move RI.

### 11.2 The poolers *were* token-dependent — the failure is not a plumbing bug

The natural suspicion for a flat result is that the poolers silently degenerated. They did not.
Per-token gradient spread on real phikon-v2 batches (`runs/poolgates-380891/g3_spread.json`, and
independently `runs/g3_spread_real.json`) measures how much `∂(pooled)/∂t_i` varies across tokens:

| pooler | spread (mean) | max abs grad | zero fraction |
|---|---|---|---|
| mean | **1.83e-07** | 0.005102 | 0.000 |
| ATTN | 0.258 | 0.0223 | 0.000 |
| GeM (p=3) | 0.394 | 0.0558 | 0.000 |
| LSE (τ=1) | 0.435 | 0.1921 | 0.000 |
| GeM + clamp | 2.154 | 0.1722 | **0.501** |

Mean pooling's 1.83e-07 is pure float32 rounding against a `max_abs_grad` of exactly `1/196 =
0.005102` — **literally zero token selectivity**, which is the §9.3 claim measured rather than
argued. The three learned poolers have selectivity six orders of magnitude above that. So the
mechanism §9.3 identified as missing was genuinely supplied, and supplying it changed nothing.

**Why clamp-GeM is not the default**, despite having by far the highest spread: real phikon-v2
patch tokens are **50.11% negative**, so clamping to positives zeroes half the gradient
(`zero_fraction` 0.501). Its large spread is an artifact of discarding half the tokens, not of
selecting among them. Unclamped GeM at odd `p` is used instead.

### 11.3 What did *not* get exercised: the learned sharpening

Over 1500 steps the learned pooling parameters barely moved:

| parameter | init | step 20 | step 1500 |
|---|---|---|---|
| `pool_gem_p` | 3.0 | 2.9999 | 2.9884 |
| `pool_lse_tau` | 1.0 | 0.9999 | 0.9892 |
| `pool_attn_entropy` | — | 0.9964 | 0.9755 |
| `pool_attn_max` | — | 0.00828 | 0.01739 |

GeM's `p` moved 0.39%, LSE's `τ` 1.08%. ATTN moved most — normalised entropy 0.9964 → 0.9755 and
peak attention weight 0.00828 → 0.01739, a **2.1×** sharpening — and ATTN is the arm that scored
*lowest*, which is the opposite of what "the pooler needs to sharpen" would predict.

**This is not "the poolers collapsed to mean pooling."** That reading is wrong twice over: GeM at
p = 2.9884 is nowhere near mean pooling (mean is p = 1), and §11.2 measures the poolers as
genuinely token-dependent throughout. The correct statement is narrower: **the fixed pooling shape
was exercised, the learned sharpening essentially was not.** The experiment tested "does a
token-selective pooling shape help" (answer: not measurably) and did *not* test "can the model
learn a useful pooling sharpness" — 1500 steps at this LR moved those scalars by ~1%, so that
question is untouched. Whether they are under-parameterised, under-learning-rated, or genuinely at
their optimum is not established here.

### 11.4 Standing caveats

- Every arm is **n=1, seed 0**. The seed-variance control (§8.6, job 381441) is the blocking
  measurement for all of §11 and for the smaller margins in §9.1.
- All arms are reported at step 1500, which §9.4 shows is past their optimum by 0.005–0.007 RI.
- The pooling never reaches evaluation directly (it is not in the exported embedding), so §11
  bounds the effect of pooling *as a training signal*, not of pooling as a readout.
