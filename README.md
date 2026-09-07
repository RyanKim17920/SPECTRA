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

---

## Results are not in this file

The tables that used to live here were written 2026-08-19, against three backbones, a
250/500/1000 checkpoint grid, the since-retired `confounder_insensitivity >= 0.75` stopping
rule, and a 4-task THUNDER roster that has since been shown to be the wrong denominator.
`docs/README.md` classifies every one of those numbers as uncitable. Rather than leave a
plausible-looking stale table at the front door, it is gone.

The current state is five backbones (`phikon-v2`, `midnight`, `Virchow2`, `H-Optimus-0`,
`UNI2-h`), a 50-step checkpoint grid, a parameter-free 1-SE stopping rule on PathoROB
avg RI, and THUNDER on Waiv's full 16-dataset roster. Several cells are still filling in.

**For any number, read the generated scoreboard, not prose:**

```
./.venv/bin/python scripts/final_scoreboard.py    # writes docs/final_scoreboard.md
```

Every cell in it is read from disk at generation time; `MISSING` means the metric is not on
disk for that cell and is never substituted from another checkpoint, step, or arm. Runs that
have not plateaued under the stopping rule are reported `NOT SELECTED` rather than silently
graded at their last checkpoint.

## Start here

| file | contents |
|---|---|
| [`docs/README.md`](docs/README.md) | **The doc map.** 24 documents accumulated here; this says which three are current and which numbers must never be pasted into the paper. Read before opening anything else in `docs/`. |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Current recipe, commands and config — the single source of truth for how to run this. |
| [`docs/final_scoreboard.md`](docs/final_scoreboard.md) | Generated scoreboard: the graded RI / HEST / THUNDER criterion, per run and per cell. |
| [`docs/CAVEATS.md`](docs/CAVEATS.md) | Reporting discipline the project holds itself to (its own example numbers are historical). |
| [`PLAN.md`](PLAN.md) | The spec the recipe was reconstructed against; every module docstring cites its section. |

Waiv's published targets are transcribed once, in
[`docs/waiv_published.json`](docs/waiv_published.json) (Tables 1–4), and loaded from there by
every comparison script.

## Configuration: where everything lives

No script in this repo carries an absolute path. Every root is an environment variable
with a repo-relative default, resolved in one place -- `src/waivphaet/paths.py` for
Python, `scripts/_env.sh` for shell and sbatch. Copy `.env.example` to `.env`, set the
handful your machine needs, and everything downstream follows:

```
cp .env.example .env && $EDITOR .env
./.venv/bin/python scripts/_config.py     # print every resolved root
```

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
`SPECTRA_RUNS=/tmp/x ./.venv/bin/python scripts/final_scoreboard.py`. `.env` is
gitignored because it describes one machine; `.env.example` is committed.

## Adding a backbone

**One entry in `src/waivphaet/models/backbones.py`.** That file is the single source for
every per-backbone fact -- normalisation override, local weight directory, timm
construction kwargs, published THUNDER pooling, LoRA target leaves, and the shape facts
the tests assert. The tables in `models/encoder.py` and `eval/thunder_protocol.py` are
views over it, so there is no second place to remember.

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

1. `./.venv/bin/python -m pytest tests/test_new_backbones.py` -- builds the model and
   asserts the shape fields against the real checkpoint, so a wrong kwarg fails as a
   number rather than drifting silently.
2. Reproduce the base against a published row **before** training anything
   (`docs/NEW_MODEL.md` §2). A base that does not reproduce is a harness bug, not a result.
3. `./.venv/bin/python scripts/check_backbone_registry.py` if you use the eval cells --
   they are standalone copies of this table (they run outside this repo and hash their own
   `model.py` for provenance), and this proves a copy has not drifted from the registry.

Leaving `thunder_readout=None` is the safe default for a model that postdates
arXiv:2607.22861: `default_pooling` then raises instead of guessing a protocol, because a
guessed pooling produces a number that is not comparable to the leaderboard and looks
exactly like one that is.

## Layout

| path | contents |
|---|---|
| `src/waivphaet/` | Library: `data/` (PLISM repack + pair sampling), `models/` (backbone + LoRA), `train/` (masked InfoNCE), `eval/` (per-benchmark adapters) |
| `scripts/` | Training entrypoints, SLURM submitters, per-benchmark runners and collectors, verification tools |
| `tests/` | Unit tests |
| `third_party/` | Cloned harnesses (PathoROB, plism-benchmark), gitignored |
| `runs/` | Run directories: checkpoints, `ri_curve.json`, per-benchmark summaries |

## Standing caveats

- fp32 here against Waiv's mixed precision: deltas compare, absolute levels do not.
- Every benchmark has a measured seed floor and several are near it — a single-seed
  HEST or THUNDER delta is not gradeable. See `docs/thunder_seed_floor_12ds.md`.
- Patho-Bench is not run.
