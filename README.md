# waivphaet

A reproduction attempt of Waiv's robustness fine-tuning (arXiv:2607.22861), which released
**Phaet** (from `owkin/phikon-v2`) and **Mascaret** (from `kaiko-ai/midnight`) as gated
weights with **no method section** — no loss, no algorithm, no corpus, no hyperparameters,
no code. The recipe here is reconstructed from the one gap their related-work section
leaves open: they dismiss "contrastive losses over co-registered scanner pairs" because
"these downstream methods all keep the backbone frozen", so we apply that loss family *to
the backbone* — masked InfoNCE over PLISM co-registered pairs (positives = same tile index,
different acquisition condition; negatives drawn from the anchor's own condition), LoRA on
all transformer blocks. PathoROB is the primary metric and is never seen in training.
[`PLAN.md`](PLAN.md) is the spec; every module docstring cites its section.

---

## 1. Headline — average level

Each backbone's **base** was reproduced against a published reference before any
fine-tuning, which is what makes the deltas meaningful rather than merely plausible.

### phikon-v2 → Phaet (ours: LoRA, step 1000)

| benchmark | base ours | base published | fine-tuned ours | Waiv Phaet |
|---|---|---|---|---|
| PathoROB Avg RI ↑ | 0.4686 | 0.4686 / 0.469 | **0.8080** | 0.806 |
| THUNDER mean Δ over 4 tasks ↑ | — | — | **+2.24** | +1.35 |
| HEST Avg Pearson ↑ | 0.3747 | 0.3747 | **0.3794** (s1000) / 0.3825 (s2000) | 0.3943 |

### Midnight-12k → Mascaret (ours: LoRA, step 500)

| benchmark | base ours | base published | fine-tuned ours | Waiv Mascaret |
|---|---|---|---|---|
| PathoROB Avg RI ↑ | 0.7589 | 0.759 | **0.9080** | 0.924 |
| THUNDER mean Δ over 4 tasks ↑ | — | — | **+2.37** | +1.80 |
| HEST Avg Pearson ↑ | not run | 0.3952 | not run | 0.4167 |

Reading:

- **PathoROB.** phikon-v2 lands *on* Waiv's Phaet number (0.8080 vs 0.806, ~100% of the
  headroom). Midnight reaches 0.9080 against Mascaret's 0.924 — 90.3% of the headroom, and
  plainly short. A full fine-tuning pilot on phikon-v2 peaked lower, at 0.8007 (§5).
- **THUNDER.** Our deltas match or exceed Waiv's on both backbones, but the mean-over-4-tasks
  figure hides composition: see §2. Our segmentation average is over 2 datasets, Waiv's over 4.
- **HEST.** We are clearly behind on phikon-v2 (+0.0047 at step 1000, +0.0078 at step 2000,
  against their +0.0196), and we have no Midnight HEST run at all. Read against the
  benchmark's dynamic range (0.3252–0.4229, span 0.0977), not against zero.

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
| linear probing | 76.54 | 79.24 | **+2.70** | +1.4 |
| few-shot | 69.33 | 70.99 | **+1.66** | +1.5 |
| segmentation (2/4 sets) | 70.40 | 70.09 | **−0.31** | −1.2 |

### Midnight-12k → Mascaret (ours step 500, `clsmean` pooling)

| task | our base | our ft | our Δ | Waiv Mascaret Δ |
|---|---|---|---|---|
| kNN | 78.25 | 80.44 | **+2.19** | +1.7 |
| linear probing | 82.88 | 84.12 | **+1.24** | +0.2 |
| few-shot | 70.64 | 76.38 | **+5.74** | +3.7 |
| segmentation (2/4 sets) | 70.12 | 70.42 | **+0.30** | +1.6 |

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

---

## 3. Dataset level — THUNDER

Waiv publish **no per-dataset THUNDER breakdown**, only the six task averages, so this
table has no counterpart in their paper and no dataset-level comparison against them is
possible. It is strictly more granular than what they released.

### phikon-v2, base → step 1000 (F1 ×100)

| dataset | kNN base | ft | Δ | lin base | ft | Δ | few base | ft | Δ | seg base | ft | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bach | 57.1 | 61.4 | +4.3 | 63.7 | 67.8 | +4.1 | 54.9 | 58.9 | +4.0 | — | — | — |
| bracs | 45.2 | 51.3 | +6.1 | 59.9 | 57.6 | −2.3 | 41.3 | 43.5 | +2.3 | — | — | — |
| break_his | 56.8 | 69.2 | +12.5 | 50.8 | 68.3 | +17.5 | 61.8 | 67.2 | +5.4 | — | — | — |
| ccrcc | 76.7 | 85.3 | +8.6 | 78.7 | 90.4 | +11.7 | 90.2 | 88.1 | −2.2 | — | — | — |
| crc | 92.1 | 94.5 | +2.3 | 92.0 | 94.0 | +2.0 | 89.6 | 95.1 | +5.5 | — | — | — |
| esca | 75.3 | 79.2 | +3.9 | 78.0 | 80.9 | +2.9 | 66.4 | 64.5 | −1.9 | — | — | — |
| mhist | 66.4 | 70.8 | +4.4 | 79.1 | 77.4 | −1.7 | 55.7 | 60.5 | +4.8 | — | — | — |
| patch_camelyon | 81.6 | 86.4 | +4.8 | 89.3 | 91.9 | +2.6 | 82.1 | 84.0 | +1.9 | — | — | — |
| tcga_crc_msi | 56.8 | 61.7 | +4.9 | 62.0 | 62.1 | +0.1 | 56.8 | 57.9 | +1.1 | — | — | — |
| tcga_tils | 80.6 | 87.9 | +7.3 | 91.0 | 91.0 | +0.0 | 85.7 | 85.7 | −0.0 | — | — | — |
| **tcga_uniform** | 68.2 | 60.0 | **−8.2** | 77.1 | 71.5 | **−5.7** | 60.0 | 52.7 | **−7.3** | — | — | — |
| wilds | 86.6 | 95.0 | +8.4 | 96.8 | 97.9 | +1.2 | 87.4 | 93.9 | +6.4 | — | — | — |
| ocelot | — | — | — | — | — | — | — | — | — | 80.0 | 79.5 | −0.5 |
| pannuke | — | — | — | — | — | — | — | — | — | 60.8 | 60.6 | −0.2 |
| segpath_lymphocytes | — | — | — | — | — | — | — | — | — | 60.6 | 61.3 | +0.7 |

### Midnight-12k, base → step 500 (F1 ×100)

| dataset | kNN base | ft | Δ | lin base | ft | Δ | few base | ft | Δ | seg base | ft | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bach** | 84.3 | 82.2 | **−2.1** | 87.9 | 82.4 | **−5.6** | 82.5 | 73.9 | **−8.6** | — | — | — |
| bracs | 50.2 | 53.5 | +3.3 | 63.8 | 63.7 | −0.1 | 49.6 | 52.0 | +2.4 | — | — | — |
| break_his | 58.1 | 74.2 | +16.1 | 56.7 | 76.1 | +19.4 | 38.5 | 66.5 | +28.0 | — | — | — |
| ccrcc | 91.6 | 89.6 | −2.0 | 90.8 | 89.5 | −1.3 | 77.5 | 88.6 | +11.1 | — | — | — |
| crc | 94.2 | 94.7 | +0.5 | 95.4 | 96.2 | +0.8 | 94.7 | 95.6 | +0.9 | — | — | — |
| esca | 81.4 | 82.4 | +1.0 | 86.2 | 87.0 | +0.8 | 75.0 | 73.9 | −1.1 | — | — | — |
| mhist | 69.3 | 74.8 | +5.5 | 80.2 | 79.8 | −0.4 | 62.4 | 71.0 | +8.6 | — | — | — |
| patch_camelyon | 88.0 | 89.3 | +1.2 | 93.5 | 94.2 | +0.6 | 82.8 | 86.4 | +3.6 | — | — | — |
| tcga_crc_msi | 61.9 | 64.1 | +2.1 | 65.6 | 68.7 | +3.1 | 55.1 | 59.6 | +4.6 | — | — | — |
| tcga_tils | 87.6 | 89.6 | +2.0 | 91.0 | 91.1 | +0.1 | 76.2 | 89.2 | +13.0 | — | — | — |
| **tcga_uniform** | 77.5 | 74.9 | **−2.6** | 85.2 | 82.3 | **−2.9** | 63.4 | 66.3 | +2.9 | — | — | — |
| wilds | 95.0 | 96.2 | +1.2 | 98.3 | 98.5 | +0.2 | 89.9 | 93.6 | +3.6 | — | — | — |
| ocelot | — | — | — | — | — | — | — | — | — | 78.4 | 79.4 | +0.9 |
| pannuke | — | — | — | — | — | — | — | — | — | 61.8 | 61.5 | −0.3 |
| segpath_lymphocytes | — | — | — | — | — | — | — | — | — | 63.8 | 63.6 | −0.1 |

`segpath_lymphocytes` landed after the §2 task averages were computed and is **not** folded
into them — those stay 2-of-4-set averages until `segpath_epithelial` completes (§5). Exact
F1 fractions: phikon-v2 0.6065 → 0.6130, Midnight 0.6375 → 0.6364.

THUNDER pooling is per-backbone and is not our choice: arXiv:2607.22861 §3 uses CLS+mean
concatenation in THUNDER only for Virchow2, AquaViT, H0-mini and Midnight-12k, so
phikon-v2 is `cls` and Midnight is `clsmean`. On ViT-g the 3072-d `clsmean` vector crashes
THUNDER's segmentation decoder, so Midnight's 2 segmentation sets are `cls` while its 12
classification sets are `clsmean` — a real methodological split, recorded per-row by
`scripts/collect_thunder.py`.

---

## 4. Caveats

- **n=1 seed** throughout, no error bars, on both backbones and every benchmark.
- **fp32 vs their mixed precision.** Each Δ is internally precision-consistent, so Δ-vs-Δ
  is valid; **absolute levels are not comparable** across the two studies.
- **Checkpoint selection is not neutral.** THUNDER and HEST use step 1000 (phikon-v2),
  chosen because it was the best *PathoROB* checkpoint. Step 2000 is better on HEST
  (+0.0078 vs +0.0047). Robustness and retention peak at different steps, so any
  single-checkpoint headline understates one axis.
- **Coverage.** THUNDER: 15 of Waiv's 16 datasets (12/12 classification, 3/4 segmentation);
  `segpath_epithelial` is the missing one and 3 of its 4 arms are still running (§5). The §2
  segmentation task averages remain over 2 datasets, Waiv's over 4. HEST: phikon-v2 only, no
  Midnight run. Patho-Bench dropped (~8 TB of WSIs, no traceable target number).
- **The Tier-1 probe tripwire is not an early-stopping signal.** `scripts/probe_follow.py`
  detects collapse; it does not predict PathoROB RI, in either direction. On the full-FT run
  it fired its scanner-regression signal from step 300 while RI was still climbing — acting on
  it would have discarded the best checkpoint (0.8007 @ 500). On the LoRA run the opposite
  held: heldout cross-scanner separation kept rising (base 0.376 → 0.394 @ 500 → 0.439 @ 2500)
  while RI fell after step 1000. Its thresholds are diagnostics, not actionable. Relatedly, **matched cosine and
  top-1 mislead under collapse**: full FT drove heldout cross-scanner matched cosine
  0.741 → 0.963, but the random-pair baseline climbed just as fast, 0.365 → 0.597. Report
  rank-based top-1 and separation, never matched cosine alone.
- **Regressions.** `tcga_uniform` regresses on both backbones and on every probe
  (phikon-v2 −8.2 kNN / −5.7 lin / −7.3 few-shot; Midnight −2.6 / −2.9). Consistency
  across independent probes rules out a local-neighbourhood artefact — information is
  genuinely lost. `bach` regresses on Midnight only (−2.1 / −5.6 / −8.6) while it *improves*
  on phikon-v2. Both are unexplained and should be understood before any publication claim.
- **PathoROB curves are plateaus, not climbs.** Both backbones reach their peak by the
  first or second checkpoint and then flatten (phikon-v2 0.803–0.808 across steps
  500–3000; Midnight 0.898–0.908 across 250–1500). Training longer buys nothing on the
  headline metric.
- **camelyon's base RI of 0.019** is pathologically low, so its headline jump is the least
  surprising of the three PathoROB datasets.

---

## 5. Full fine-tuning pilot, and what is still running

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

### Still running — `segpath_epithelial`, 3 of 4 arms

`segpath_epithelial` runs at 9 epochs and `segpath_lymphocytes` at 21, per THUNDER's own
`docs/guidelines.md`; the generic 200-epoch default is wrong for these two sets and is why
earlier attempts were abandoned. `segpath_lymphocytes` is complete (all 4 arms, §3), and
`segpath_epithelial`'s Midnight fine-tuned arm finished (job 369916, F1 0.6912).

The other three — 369913 (phikon-v2 `base_cls`), 369914 (phikon-v2 `ft1000_cls`), 369915
(Midnight `mbase_cls`) — were preempted and **restarted from scratch** on 2026-08-05
(369913/369915 at 02:32 UTC, 369914 at 04:00 UTC), because THUNDER does not checkpoint
segmentation training. As of 2026-08-05 18:34 UTC they are at epoch 4–5 of 9 at
~11,500 s/epoch, i.e. roughly 13–16 h of training left plus eval.

Coverage is therefore 15 of Waiv's 16 datasets. The full 16-dataset like-for-like against
their Table 2 — and a 4-dataset segmentation average matching theirs — is still blocked on
these three jobs.

---

## 6. Reproducing

### Environments — four venvs, deliberately separate

`.venv` (training), `.venv-pathorob`, `.venv-hest`, `.venv-thunder`. Each harness hard-pins
versions that conflict with ours and with each other (PathoROB pins `numpy==2.2.6`,
`pandas==2.3.2`, `transformers==4.56.1`; we run numpy 2.5 / pandas 3.0 / transformers 5.14).
Downgrading the training venv to match would be the wrong trade — only their *metrics* need
those pins.

**`PYTHONNOUSERSITE=1` is required** for every harness call:
`~/.local/lib/python3.12/site-packages` on this machine holds a pandas that otherwise
shadows the venv's and makes the pins meaningless. **Always**
`export HF_HOME=/data/ryan.kim/hf_home` — `/admin` must not fill.

```bash
export UV_CACHE_DIR=/data/ryan.kim/uv_cache
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[pathorob]'
# torch MUST come from the cu128 index: this cluster's driver reports CUDA 12.8 and the
# default cu130 wheel fails CUDA init with "driver is too old".
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0

uv venv --python 3.12 .venv-pathorob
uv pip install --python .venv-pathorob/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-pathorob/bin/python -e third_party/PathoROB
```

Harnesses are cloned, not vendored (their source is gitignored):

```bash
git clone --depth 1 https://github.com/bifold-pathomics/PathoROB third_party/PathoROB
git clone --depth 1 https://github.com/owkin/plism-benchmark third_party/plism-benchmark
```

### Data

PLISM is 91 `.h5` slides (7 scanners × 13 stains × 16,278 Elastix-registered tiles,
~224 GB). Each file is flat — 16,278 datasets named `tile_{level}_{x}_{y}`, key order
byte-identical across all 91, which is what makes index-aligned positives possible. Repack
before training: 16,278 tiny HDF5 datasets per slide makes random pair sampling I/O bound
(7.4k tiles/s), and a contiguous memmap turns a tile read into one `pread` (measured
830k–920k tiles/s, page-cache-warm so treat it as an upper bound).

```bash
./.venv/bin/python -m waivphaet.data.repack \
  --h5-dir /data/ryan.kim/plism --out-dir /data/ryan.kim/plism/repacked --verify --benchmark
```

Residual Elastix misregistration is ~5–50 px, so positives are near-identical *shifted*
crops, never pixel-exact. Do not add augmentations that assume otherwise.

### Train

```bash
./.venv/bin/python scripts/smoke_test.py --steps 4 --device cpu   # wiring check
sbatch scripts/train_real.sbatch                                  # LoRA, rank 32/alpha 64
sbatch scripts/train_full_ft.sbatch                               # full FT pilot
```

The split holds out scanners `GT450`, `S210` and stains `HRH`, `KR`, `MY` → 50 train / 41
held-out conditions. It is *named*, not sampled, so it reproduces without a seed.

### Evaluate

```bash
# PathoROB (primary). Login nodes have no GPU; extraction goes through SLURM.
sbatch scripts/extract_pathorob.sbatch phikonv2_clsmean_ours "camelyon tolkach_esca tcga"
PYTHONNOUSERSITE=1 ./.venv-pathorob/bin/python scripts/pathorob_gate.py \
  --model phikonv2_clsmean_ours --datasets camelyon tolkach_esca tcga

# THUNDER
sbatch scripts/run_thunder.sbatch <dataset> <task> <run_name> [ckpt]
PYTHONNOUSERSITE=1 THUNDER_BASE_DATA_FOLDER=/data/ryan.kim/thunder \
  ./.venv-thunder/bin/python scripts/collect_thunder.py --model ft1000_cls

# HEST
sbatch scripts/run_hest.sbatch <run_name> <pooling> [ckpt]
PYTHONNOUSERSITE=1 ./.venv-hest/bin/python scripts/collect_hest.py --pooling cls
```

`collect_thunder.py` deliberately does **not** compute THUNDER's rank sum: it is a rank over
whatever roster is on the leaderboard, and three sources give three different numbers for
phikon-v2 (Waiv 97, the THUNDER paper 77, the live leaderboard 89). Absolute per-dataset F1
means the same thing on every run.

---

## 7. Verification the numbers rest on

- **PathoROB harness.** Base phikon-v2 through our extractor + their metric reproduces
  PathoROB's own committed `phikonv2_clsmean` row to 6+ decimals (camelyon bit-identical;
  µ-scale drift elsewhere is float summation order), and that committed reference
  independently agrees with Waiv's Table-1 base row. Two separate sources corroborating.
  Base Midnight likewise reproduces Waiv's published 0.759 (0.7589).
- **Backbone-agnostic refactor.** `scripts/regression_bitcheck.py` compares raw feature
  arrays across worktrees: base phikon-v2 pre- vs post-refactor is bit-identical at both
  `cls` (1024-d) and `clsmean` (2048-d), max abs delta 0.000e+00.
- **The adapter is proven applied at every eval**, not assumed: the extractor compares
  against `disable_adapter()` and exits non-zero below 1e-4. Observed `rel_l2_delta` is
  0.73–0.93 across all datasets and checkpoints, recorded per-point in `ri_curve.json`.
- **Two silent-failure modes the Midnight port would have hit.** (1) The old LoRA target
  list was a fixed tuple containing `fc1`/`fc2`, which match nothing on Midnight's SwiGLU
  FFN — `query/key/value/dense` still match in every block, so the "LoRA covers all blocks"
  assertion would have *passed* with the entire FFN frozen. Targets are now discovered by
  leaf name and the per-block count is asserted non-empty and uniform. (2) Midnight requires
  (0.5,0.5,0.5) normalization, not ImageNet; the wrong stats change no shape and throw no
  error, they just cost base accuracy — and would have surfaced as our base-Midnight row
  disagreeing with Waiv's 0.759, i.e. as false evidence that our harness is unfaithful.
  `BACKBONE_NORMALIZATION` is now table-driven and warns on unregistered backbones.

## 8. Reporting discipline (`PLAN.md` §6)

- Cross-stain and cross-scanner **separately** — the composite hides the hard axis.
- Never cosine similarity alone (PLIP: 0.878 cosine at 0.054 top-10).
- Any PLISM number is a **training diagnostic**. Label it; never print it next to a
  leaderboard number.
- Retention (HEST / THUNDER) alongside **every** robustness claim, as a pair. Forgetting is
  the default outcome here, not a tail risk.
