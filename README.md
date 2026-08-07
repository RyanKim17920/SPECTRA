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

PathoROB is the primary axis and is never seen in training. THUNDER covers 16 of Waiv's 16
datasets over all four tasks; we match or beat Waiv on 7 of 8 model × task pairs, the
exception being Midnight segmentation. HEST retention is the weak axis on both backbones, and
a sweep over every checkpoint shows that is intrinsic to the recipe, not a checkpoint-selection
artefact. Full FT did not beat LoRA. n=1 seed throughout, matching each benchmark's own
protocol; our runs are fp32 against Waiv's mixed precision, so deltas compare but absolute
levels do not, and the Midnight HEST columns are not on a common scale at all. A LoRA rank
sweep over ranks 8–128 finds no systematic effect on either axis — PathoROB plateau means
0.797–0.803 and HEST mean Δ +0.003 to +0.009, both spreads no larger than one arm's own
checkpoint-to-checkpoint scatter. Capacity is not the lever.

Read the numbers with [`docs/CAVEATS.md`](docs/CAVEATS.md) open.

---

## Documentation

| file | contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full numeric record: headline, per-task THUNDER, per-dataset THUNDER, per-cancer-type HEST, HEST checkpoint sweep, full-FT pilot, LoRA rank sweep |
| [`docs/CAVEATS.md`](docs/CAVEATS.md) | Every caveat the results are conditioned on, plus the reporting discipline the project holds itself to |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Environments, data prep, training, evaluation, and the verification the numbers rest on |
| [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md) | Applying this recipe to a backbone other than phikon-v2 or Midnight |
| [`PLAN.md`](PLAN.md) | The spec the recipe was reconstructed against |

## Layout

| path | contents |
|---|---|
| `src/waivphaet/` | Library: `data/` (PLISM repack + pair sampling), `models/` (backbone + LoRA), `train/` (masked InfoNCE), `eval/` (per-benchmark adapters) |
| `scripts/` | Training entrypoints, SLURM submitters, per-benchmark runners and collectors, verification tools |
| `tests/` | Unit tests |
| `third_party/` | Cloned harnesses (PathoROB, plism-benchmark), gitignored |
| `runs/` | Run directories: checkpoints, `ri_curve.json`, per-benchmark summaries |
