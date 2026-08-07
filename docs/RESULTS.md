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

Reading:

- **PathoROB.** phikon-v2 lands *on* Waiv's Phaet number (0.8080 vs 0.806, ~100% of the
  headroom). Midnight reaches 0.9080 against Mascaret's 0.924 — 90.3% of the headroom, and
  plainly short. A full fine-tuning pilot on phikon-v2 peaked lower, at 0.8007 (§4).
- **THUNDER.** All 16 of Waiv's datasets are now covered, so every task average — segmentation
  included — is over the same sets as theirs. We match or beat Waiv on **7 of the 8
  model × task pairs** (§2). The sole loss is Midnight segmentation (−0.33 vs their +1.6);
  phikon-v2 segmentation is a win in the sense that we regress less (−0.12 vs their −1.2).
  The mean-over-4-tasks figure still hides composition: see §2.
- **HEST is the weak axis of the reconstruction, on both backbones, 2 of 2.** phikon-v2
  +0.0047 (step 1000) / +0.0078 (step 2000) against Waiv's Phaet +0.0196; Midnight +0.0011
  (step 500) against their Mascaret +0.0215. Read against the benchmark's dynamic range
  (0.3252–0.4229, span 0.0977), not against zero: our Midnight delta is ~1% of that span,
  Waiv's ~22%. With the Midnight run in, this is no longer a phikon-v2 quirk — it is a
  consistent gap in the recipe. A sweep over every checkpoint on both backbones (§2) shows
  the best in range is +0.0098 (phikon-v2, step 3500) and +0.0035 (Midnight, step 250), so
  the gap is intrinsic, not a checkpoint-selection artefact. Retention is where the
  reconstruction falls short.

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

## 6. Third backbone — `paige-ai/Virchow2` base reproduction

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

Not yet run: Virchow2 fine-tuning. The submitter's adapter path is a placeholder and `--go`
refuses on it, so no THUNDER sweep can file base numbers under a fine-tuned run name.
