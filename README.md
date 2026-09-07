# SPECTRA

**SPECTRA** takes a frozen public pathology foundation model and trains a LoRA adapter
(rank 32, alpha 64, every transformer block) with a masked InfoNCE loss over
**co-registered PLISM tiles**: a positive pair is the *same physical tile* imaged under a
different scanner or stain, negatives come from the anchor's own acquisition condition, and
same-core negatives are masked out so the loss cannot reward memorising the slide. The
encoder is read through two heads — a CLS view and a softplus-GeM pooled view — and the
loss is reweighted per pseudo-tissue so frequent conditions do not dominate the batch.

Training is 500 steps; the shipped checkpoint is chosen by a **parameter-free 1-SE rule** on
the PathoROB average-RI curve, not by taking the best number after the fact. The result is a
small, mergeable adapter that leaves the base architecture and inference contract untouched.

This repo is both the method and the harness that evaluates it: **7 backbones × 3 seeds**
across **PathoROB** (robustness index — the primary metric, never seen in training),
**HEST**, **THUNDER**, **CPTAC**, and **held-out PLISM retrieval**.

---

## Results

All numbers below are copied from the generated tables in `docs/` — regenerate them rather
than trusting this table (see [Reproducing the tables](#reproducing-the-tables)). Fine-tuned
values are the mean over 3 seeds, each at its own 1-SE-selected checkpoint.

| backbone | PathoROB mean RI | HEST mean *r* | CPTAC AUC | PLISM cross-stain top-1 |
|---|---|---|---|---|
| Phikon-v2      | 0.470 → **0.836** ± 0.014 | 0.3747 → **0.3906** | 0.6491 → **0.6870** | 0.696 → **0.897** |
| Midnight-12k   | 0.759 → **0.908** ± 0.005 | 0.3952 → **0.4122** | 0.6643 → **0.6898** | 0.560 → **0.883** |
| Virchow2       | 0.861 → **0.909** ± 0.007 | 0.4032 → **0.4089** | 0.6789 → **0.6879** | 0.696 → **0.903** |
| H-optimus-0    | 0.800 → **0.906** ± 0.004 | 0.4150 → **0.4226** | 0.6728 → **0.6895** | 0.830 → **0.915** |
| UNI2-h         | 0.757 → **0.907** ± 0.007 | 0.4138 → **0.4238** | 0.6750 → **0.6980** | 0.761 → **0.913** |
| Virchow        | 0.815 → **0.890** ± 0.004 | 0.4061 → **0.4083** | 0.6608 → **0.6839** | 0.597 → **0.931** |
| OpenMidnight   | 0.618 → **0.879** ± 0.007 | 0.3902 → **0.4048** | 0.6561 → **0.6844** | 0.486 → **0.782** |

RI rises on 7/7 backbones and HEST, CPTAC and PLISM retrieval rise on 7/7 alongside it — the
robustness gain is not paid for out of the base model's downstream accuracy.

Leaderboard position, base → fine-tuned (lower is better; PathoROB is a 23-model field, HEST
26 models, THUNDER 32 models):

| backbone | PathoROB | HEST | THUNDER |
|---|---|---|---|
| Phikon-v2 | 21 → 6 | 15 → 10 | 18 → 14 |
| Midnight-12k | 10 → 2 † | 8 → 5 | 5 → 2 |
| Virchow2 | 3 → 2 | 6 → 5 | 3 → 2 |
| H-optimus-0 | 9 → 2 | 3 → 2 | 8 → 5 |
| UNI2-h | 10 → 2 | 4 → 1 | 1 → 1 |
| Virchow | 6 → 2 | 5 → 5 | 11 → 6 |
| OpenMidnight | 14 → 3 † | 9 → 6 | 12 → 5 |

† estimated: the model is not in the published PathoROB field and is inserted into it.

Full tables, including the per-dataset and per-task breakdowns and every error bar:
[`docs/pathorob_ranks.md`](docs/pathorob_ranks.md),
[`docs/pathorob_submetrics.md`](docs/pathorob_submetrics.md),
[`docs/hest_ranks.md`](docs/hest_ranks.md),
[`docs/hest_per_task.md`](docs/hest_per_task.md),
[`docs/thunder_ranks.md`](docs/thunder_ranks.md),
[`docs/thunder_per_ds.md`](docs/thunder_per_ds.md),
[`docs/cptac.md`](docs/cptac.md),
[`docs/plism_retrieval.md`](docs/plism_retrieval.md),
[`docs/overall_ranks.md`](docs/overall_ranks.md).

---

## Released models

Adapters are published one Hugging Face repo per backbone, with `seed0/`, `seed1/`, `seed2/`
as subfolders — all three seeds of a backbone share one base model, one revision pin and one
inference contract.

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "<org>/spectra-midnight-12k-lora",
                                  subfolder="seed0")
```

| repo | contents |
|---|---|
| `<org>/spectra-<backbone>-lora` | the LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`), 3 seeds |
| `<org>/spectra-<backbone>-merged` | the same delta already merged into the base weights, for callers who do not want a PEFT dependency |

`<org>` is a placeholder — substitute the account the weights are actually pushed to.
Backbone slugs are `phikon-v2`, `midnight-12k`, `virchow2`, `virchow`, `h-optimus-0`,
`uni2h`, `openmidnight`. Merged variants exist only for the permissively-licensed bases.

**Not all seven are published.** `spectra-virchow2-lora` and `spectra-uni2h-lora` are built,
verified and staged but **withheld pending a licensing decision**: both bases are
CC-BY-NC-ND-4.0, and whether a LoRA delta is a "derivative" under NoDerivatives is not ours
to assume. `spectra-phikon-v2-lora` is held under Owkin's **non-commercial, non-profit-only**
terms. [`docs/release_manifest.md`](docs/release_manifest.md) records exactly what each repo
contains, which checkpoint the 1-SE rule selected, and the per-backbone licence state.

---

## Install

Python 3.12+. The cu128 wheel index is required on this cluster's driver; use whatever
matches yours.

```bash
python -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
./.venv/bin/pip install -e ".[dev,pathorob,plism]"
```

## Quickstart

```bash
cp .env.example .env && $EDITOR .env    # point SPECTRA_DATA at your scratch volume
./.venv/bin/python scripts/_config.py   # print every resolved root

./.venv/bin/python scripts/acquire_plism.py       # fetch + repack the PLISM corpus
sbatch scripts/gentle.sbatch                      # the recipe (SLURM)
```

`scripts/gentle.sbatch` is the one training entrypoint; only `SPECTRA_ARM` and
`SPECTRA_SEED` vary across backbones:

```bash
SPECTRA_ARM=midnight SPECTRA_SEED=0 SPECTRA_T=900 \
SPECTRA_MASK=1 SPECTRA_BCLS=3.0 SPECTRA_MAX_STEPS=500 SPECTRA_CKPT_EVERY=50 \
  sbatch scripts/gentle.sbatch
```

Everything else — LR `1e-4`, warmup 200, LoRA rank 32 / alpha 64, projection dim 512, weight
decay 0.05, temperature 0.07 — is left at its default.
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) §1.1 is the full variable table and the single source of
truth for how to run this.

## Configuration

No script in this repo carries an absolute path. Every root is an environment variable with a
repo-relative default, resolved in one place — `src/spectra/paths.py` for Python,
`scripts/_env.sh` for shell and sbatch.

| variable | what it points at | default |
|---|---|---|
| `SPECTRA_REPO` | the checkout | auto-detected |
| `SPECTRA_RUNS` | run dirs: checkpoints, `ri_curve.json`, summaries | `<repo>/runs` |
| `SPECTRA_DATA` | the root every corpus below defaults under | `<repo>/data_root` |
| `SPECTRA_PLISM` / `SPECTRA_PLISM_PACKED` | PLISM corpus, and the repacked tiles training reads | `<data>/plism`, `<plism>/repacked` |
| `SPECTRA_THUNDER` | THUNDER's `THUNDER_BASE_DATA_FOLDER` | `<data>/thunder` |
| `SPECTRA_HEST_BENCH` / `SPECTRA_HEST_WORK` | HEST benchmark data, and its work dir | `<data>/hest_bench`, `<data>/hest_work` |
| `SPECTRA_EVALS` | output root of the per-checkpoint cell harness | `<data>/full-evals` |
| `SPECTRA_HF_HOME` | HuggingFace cache (set it, or `~/.cache` fills your home volume) | `<data>/huggingface` |
| `SPECTRA_INPUTS` | local base-model weight dirs, one per gated/converted backbone | `<data>/inputs` |
| `SPECTRA_CELLS` | per-checkpoint eval cells | `<repo>/cells` |
| `SPECTRA_PAPER` | LaTeX tree the table/figure generators write into | `<repo>/paper` |
| `SPECTRA_SNAPSHOTS` / `SPECTRA_BACKUPS` | pinned code snapshots; durable result copies | `<repo>/snapshots`, `<repo>/result_backups` |

A variable set in the real environment always beats `.env`, so a one-off override is
`SPECTRA_RUNS=/tmp/x ./.venv/bin/python scripts/final_scoreboard.py`. `.env` is gitignored
because it describes one machine; [`.env.example`](.env.example) is committed and documents
every root.

The per-run knobs the launchers read (`SPECTRA_ARM`, `SPECTRA_SEED`, `SPECTRA_T`,
`SPECTRA_MASK`, …) live in the same namespace. The frozen code snapshots under
`$SPECTRA_SNAPSHOTS` were pinned under an earlier internal package name and still read
`WAIV_*` spellings; a pin must never be rewritten, so `scripts/_env.sh` and
`spectra.paths.alias_legacy_env()` mirror the two namespaces onto each other instead. Set
either name and both reach the job. That mirroring is legacy scaffolding and goes away with
the last pin that needs it.

## Reproducing the tables

Every table in `docs/` is generated and carries a *"do not hand-edit"* banner naming its
generator. Each one reads measurements off disk at generation time: `MISSING` means the
metric is not on disk for that cell and is never substituted from another checkpoint, step or
arm.

```bash
./.venv/bin/python scripts/final_scoreboard.py    # docs/final_scoreboard.md — the graded criterion
./.venv/bin/python scripts/seed_stats.py          # docs/seed_stats.md — per-seed selected checkpoints
./.venv/bin/python scripts/pathorob_ranks.py      # docs/pathorob_ranks.md
./.venv/bin/python scripts/pathorob_submetrics.py # docs/pathorob_submetrics.md
./.venv/bin/python scripts/thunder_ranks.py       # docs/thunder_ranks.md
./.venv/bin/python scripts/thunder_per_ds.py      # docs/thunder_per_ds.md
./.venv/bin/python scripts/cptac_table.py         # docs/cptac.md
./.venv/bin/python scripts/overall_ranks.py       # docs/overall_ranks.md
./.venv/bin/python scripts/eval_matrix.py         # docs/eval_matrix.md — what is and is not measured
./.venv-hest/bin/python scripts/hest_ranks.py     # docs/hest_ranks.md, docs/hest_per_task.md
./.venv-hest/bin/python scripts/plism_retrieval.py # docs/plism_retrieval.md
```

Set `SPECTRA_PAPER` to write the LaTeX copies somewhere other than `<repo>/paper`.

Published comparison targets are transcribed **once**, in
[`docs/reference_published.json`](docs/reference_published.json), which carries its own
`_source` citation; every comparison script loads them from there rather than restating them.

## Adding a backbone

**One entry in `src/spectra/models/backbones.py`.** That file is the single source for every
per-backbone fact — normalisation override, local weight directory, timm construction kwargs,
published THUNDER pooling, LoRA target leaves, and the shape facts the tests assert. The
tables in `models/encoder.py` and `eval/thunder_protocol.py` are views over it, so there is no
second place to remember.

```python
"vendor/NewModel": Backbone(
    repo_id="vendor/NewModel",
    loader="timm",                    # recorded; dispatch still reads the repo's config.json
    weights_file="model.safetensors",
    architecture="vit_large_patch16_224",
    timm_kwargs={"img_size": 224, "init_values": 1e-5},
    local_subdir="NewModel",          # under $SPECTRA_INPUTS; omit for hub-served
    normalization=(IMAGENET_MEAN, IMAGENET_STD),   # None = ask the repo's own config
    thunder_readout="cls",            # None = no published protocol; scoring it will RAISE
    lora_target_suffixes=_TIMM_VIT,
    embed_dim=1024, num_prefix_tokens=1, num_blocks=24, patch_size=16,
),
```

Then:

1. `./.venv/bin/python -m pytest tests/test_new_backbones.py` — builds the model and asserts
   the shape fields against the real checkpoint, so a wrong kwarg fails as a number rather
   than drifting silently.
2. Reproduce the base against a published row **before** training anything
   ([`docs/archive/NEW_MODEL.md`](docs/archive/NEW_MODEL.md) §2). A base that does not
   reproduce is a harness bug, not a result.
3. `./.venv/bin/python scripts/check_backbone_registry.py` if you use the eval cells — they
   are standalone copies of this table (they run outside this repo and hash their own
   `model.py` for provenance), and this proves a copy has not drifted from the registry.

Leaving `thunder_readout=None` is the safe default for a model with no published THUNDER
protocol: `default_pooling` then raises instead of guessing, because a guessed pooling
produces a number that is not comparable to the leaderboard and looks exactly like one that
is.

## Repo layout

| path | contents |
|---|---|
| `src/spectra/` | Library: `data/` (PLISM repack + pair sampling), `models/` (backbone + LoRA), `train/` (masked InfoNCE), `eval/` (per-benchmark adapters), `paths.py` (every filesystem root) |
| `scripts/` | Training entrypoints, SLURM submitters, per-benchmark runners and collectors, table generators, verification tools |
| `tests/` | Unit tests |
| `docs/` | Current docs and generated result tables — start at [`docs/README.md`](docs/README.md) |
| `docs/archive/` | Superseded research logs, kept for provenance |
| `snapshots/` | Frozen code pins a run was trained against — never edited |
| `third_party/` | Cloned harnesses (PathoROB, plism-benchmark), gitignored |
| `runs/` | Run directories: checkpoints, `ri_curve.json`, per-benchmark summaries |

## Where to read next

| file | contents |
|---|---|
| [`docs/README.md`](docs/README.md) | **The doc map.** Which documents are current, which are archived, and which numbers must never be pasted into a paper. Read before opening anything else in `docs/`. |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Current recipe, commands and config — the single source of truth for how to run this. |
| [`docs/final_scoreboard.md`](docs/final_scoreboard.md) | Generated scoreboard: the graded RI / HEST / THUNDER criterion, per run and per cell. |
| [`docs/CAVEATS.md`](docs/CAVEATS.md) | Reporting discipline the project holds itself to (its own example numbers are historical). |
| [`docs/release_manifest.md`](docs/release_manifest.md) | What each released repo contains, and its licence state. |

## Standing caveats

- fp32 here against mixed-precision published baselines: deltas compare, absolute levels do
  not.
- Every benchmark has a measured seed floor and several sit near it — a single-seed HEST or
  THUNDER delta is not gradeable. See [`docs/thunder_seed_floor_12ds.md`](docs/thunder_seed_floor_12ds.md).
- PLISM is training data here, so PLISM retrieval is a held-out-condition diagnostic, not a
  leaderboard-comparable number.
- Patho-Bench is not run; CPTAC is the Patho-Bench subset only.

## Licence

MIT — see [`LICENSE`](LICENSE). The released adapters inherit the licence of the base model
they are trained on; see [`docs/release_manifest.md`](docs/release_manifest.md).

## Citation

```bibtex
@misc{kim2026spectra,
  title  = {SPECTRA: contrastive robustness fine-tuning for pathology foundation models},
  author = {Kim, Ryan},
  year   = {2026},
  note   = {TODO: fill in venue and year on acceptance --
            submitted to the ASCI workshop at NeurIPS 2026}
}
```
