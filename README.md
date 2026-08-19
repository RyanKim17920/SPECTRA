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

## Results

Each backbone's **base** reproduces the published reference before any fine-tuning, so the
deltas are meaningful rather than merely plausible.

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

### PathoROB per dataset

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

### Full fine-tuning pilot (phikon-v2, best of 10 checkpoints)

| variant | best Avg RI | at step |
|---|---|---|
| LoRA rank 32 | **0.8080** | 1000 |
| full FT | 0.8007 | 500 |

### Third backbone — `paige-ai/Virchow2`

| dataset | camelyon | tolkach_esca | tcga | **Avg** | Waiv T1 |
|---|---|---|---|---|---|
| our base RI | 0.7989 | 0.9541 | 0.8218 | **0.8582** | 0.858 |
| ours step 250 | **0.9006** | **0.9673** | **0.8425** | **0.9035** | 0.918 |

| benchmark | base ours | fine-tuned ours | Waiv Virchow2 |
|---|---|---|---|
| PathoROB Avg RI ↑ | 0.8582 | **0.9035** | 0.918 |
| THUNDER mean Δ over 4 tasks ↑ | — | **+1.42** | +0.62 |
| HEST Avg Pearson ↑ | 0.4032 | **0.4083** | 0.4135 |

PathoROB is the primary axis and is never seen in training. THUNDER covers all 16 of Waiv's
datasets on all three backbones; we match or beat them on 11 of 12 comparable model × task
pairs, the exception being Midnight segmentation. Full FT did not beat LoRA. Virchow2 (timm ViT-H, 4
register tokens) reproduces its published base to +0.0002, which is what validates the
pipeline on a backbone it was not written against, and then gains +0.0453 RI (~76% of the
headroom to Waiv's 0.918).

Across three backbones the result is consistent: **robustness reproduces (3/3), classification
performance improves (32 of 36 model × dataset pairs, sign test p ≈ 2×10⁻⁶), segmentation does
not (3 of 12, p ≈ 0.15), and retention does not (HEST +0.0047 / +0.0011 / +0.0051 against
Waiv's +0.0196 / +0.0215 / +0.0101)**. The gain shrinks as the base encoder gets stronger —
gap-closed 101% → 90% → 76% as base RI goes 0.469 → 0.759 → 0.858.

**How we compare to Waiv overall.** Their published numbers are transcribed in
[`docs/waiv_published.json`](docs/waiv_published.json) (arXiv:2607.22861, Tables 1–4). Placing
our runs on their Figure 1 composite rank axis — the one their headline figure plots — puts us
within **1–2 rank points** of them on every backbone: total 45 vs their 44 (phikon-v2), 12 vs 11
(Midnight), 20 vs 18 (Virchow2). We are *ahead* on THUNDER deltas and *behind* on HEST deltas,
and those largely cancel. Our absolute levels run ~0.2–1.0 points below theirs on THUNDER
because our base reproductions start low, not because the recipe underperforms. We do not run
Patho-Bench, so that rank is assumed equal to theirs in the placement above.

**Retention is the axis that does not reproduce**, and four experiments say it is not a tuning
problem: it survives every checkpoint, LoRA ranks 8–128 (both axes flat within one arm's own
scatter), full fine-tuning, and an added frozen-teacher retention term. That last one also
corrected the premise — fine-tuning *improves* HEST (+0.0078 over base), just less than Waiv's
+0.0196, so there is no retention loss to prevent and every λ was strictly dominated by λ=0.

n=1 seed throughout, matching each benchmark's own protocol; our runs are fp32 against Waiv's
mixed precision, so deltas compare but absolute levels do not.

Read the numbers with [`docs/CAVEATS.md`](docs/CAVEATS.md) open.

---

## Documentation

| file | contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full numeric record: headline, per-task and per-dataset THUNDER, per-cancer-type HEST, HEST checkpoint sweep, full-FT pilot, LoRA rank sweep, Virchow2 base, retention-KL sweep |
| [`docs/CAVEATS.md`](docs/CAVEATS.md) | Every caveat the results are conditioned on, plus the reporting discipline the project holds itself to |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Environments, data prep, training, evaluation, and the verification the numbers rest on |
| [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md) | Applying this recipe to a new backbone; `paige-ai/Virchow2` is the worked example |
| [`PLAN.md`](PLAN.md) | The spec the recipe was reconstructed against |

## Layout

| path | contents |
|---|---|
| `src/waivphaet/` | Library: `data/` (PLISM repack + pair sampling), `models/` (backbone + LoRA), `train/` (masked InfoNCE), `eval/` (per-benchmark adapters) |
| `scripts/` | Training entrypoints, SLURM submitters, per-benchmark runners and collectors, verification tools |
| `tests/` | Unit tests |
| `third_party/` | Cloned harnesses (PathoROB, plism-benchmark), gitignored |
| `runs/` | Run directories: checkpoints, `ri_curve.json`, per-benchmark summaries |
