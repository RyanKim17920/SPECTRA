# SPECTRA v2

[![arXiv](https://img.shields.io/badge/arXiv-2611.10093-b31b1b.svg)](https://arxiv.org/abs/2611.10093)

**Paper:** [Contrastive fine-tuning against stain and scanner bias improves robustness and downstream performance across pathology foundation models](https://arxiv.org/abs/2611.10093)

This branch holds **SPECTRA v2**, a simpler recipe developed after the paper. The code and
numbers of the paper are on the [`main`](https://github.com/RyanKim17920/SPECTRA/tree/main) branch.

SPECTRA takes a frozen pathology foundation model and trains a LoRA adapter with a contrastive
loss over **registered PLISM tiles**: the same tissue location imaged under different scanners
and stains. Each step samples 2 acquisition conditions and T shared tissue locations. For a
query tile, the positive is the tile at the same location under the other condition, and the
negatives are the other T-1 tiles of that condition, so the acquisition alone cannot identify
the match.

## What changed from the paper

| | paper (`main`) | v2 (this branch) |
|---|---|---|
| batch | 2 conditions × 900 locations | 2 conditions × 256 locations, 224 px |
| loss | InfoNCE with pseudo-tissue reweighting of negatives | symmetric InfoNCE, τ = 0.1, no reweighting |
| adapter | LoRA r32 / α64, attention + MLP | LoRA r16 / α32, attention + MLP |
| heads | CLS + GeM | CLS 0.5 + GeM 0.5, each Linear-BN-GELU-Linear → 512 |
| schedule | 500 steps, lr 1e-4, 200-step warmup | lr 1e-4, wd 0.05, 200-step warmup + cosine over 400 steps, truncated at the stop step |
| stop step | tolerance rule on the **PathoROB RI** curve | **R1**: held-out PLISM retrieval only, no test benchmark |

**Stop rule R1.** Train a selection run with the GT450 scanner, the MY stain and 7 tissue cores
held out. Every 25 steps, record held-out PLISM retrieval P on those cores. The stop step is
argmax over s ≥ 25 of P(s). Then train the release model on all 91 conditions and all cores to
that step. No labels and no test benchmark are used to choose it. Steps used for the reported
models: Virchow2 100, UNI2-h 175, H-optimus-0 100, Virchow 225, Phikon-v2 225, Midnight-12k
225, OpenMidnight-1.1 225.

## Results

Each cell is base / paper / v2: the frozen backbone, the paper's checkpoints, and the 3-seed
mean of the v2 release runs. All three PathoROB / CRoMa columns use the same evaluation code and
preprocessing. Full tables with SDs and THUNDER: [docs/v2.md](docs/v2.md).

| backbone | PathoROB RI | CRoMa median | CRoMa LTM10 | HEST | CPTAC AUC |
|---|---|---|---|---|---|
| Virchow2 | 0.858 / 0.908 / 0.906 | 0.225 / 0.249 / 0.278 | -0.106 / -0.051 / -0.064 | 0.4033 / 0.4089 / 0.4110 | 0.6789 / – / 0.6898 |
| UNI2-h | 0.757 / 0.908 / 0.916 | 0.128 / 0.214 / 0.287 | -0.127 / -0.050 / -0.057 | 0.4138 / 0.4238 / 0.4224 | 0.6750 / 0.6980 / 0.7005 |
| H-optimus-0 | 0.812 / 0.909 / 0.908 | 0.134 / 0.193 / 0.217 | -0.103 / -0.033 / -0.052 | 0.4150 / 0.4225 / 0.4220 | 0.6728 / 0.6895 / 0.6854 |
| Virchow | 0.812 / 0.890 / 0.900 | 0.202 / 0.215 / 0.270 | -0.146 / -0.060 / -0.058 | 0.4061 / 0.4083 / 0.4066 | 0.6607 / – / 0.6824 |
| Phikon-v2 | 0.469 / 0.836 / 0.826 | 0.002 / 0.173 / 0.196 | -0.277 / -0.126 / -0.138 | 0.3747 / 0.3906 / 0.3868 | 0.6491 / – / 0.6926 |
| Midnight-12k | 0.759 / 0.908 / 0.920 | 0.363 / 0.279 / 0.333 | -0.214 / -0.047 / -0.045 | 0.3952 / 0.4122 / 0.4089 | 0.6643 / – / 0.6881 (n=1) |
| OpenMidnight † | 0.600 / 0.887 / 0.899 | 0.185 / 0.275 / 0.297 | -0.219 / -0.102 / -0.082 | 0.3834 / 0.4048 / 0.3915 | 0.6624 / – / pending |

† The paper used the original OpenMidnight; v2 uses OpenMidnight-1.1, a different base model
(the base values in this row are OpenMidnight-1.1).

Notes:

* Paper RI and CRoMa were re-scored on the paper's checkpoints with the v2 evaluation code
  (CRoMa is not in the paper). The paper's own RI used a different preprocessing (fp16, bicubic
  resize 256 → center crop 224) and reads 0.9091 / 0.9074 / 0.9055 / 0.8904 / 0.8356 / 0.9082 /
  0.8794 in the row order above.
* Paper HEST values are the paper's numbers. The current code reproduces them for UNI2-h and
  H-optimus-0; the other backbones were not re-checked.
* Paper CPTAC is shown for UNI2-h and H-optimus-0, where the current pipeline reproduces the
  paper's values exactly; for the other backbones it was not re-verified ("–").
* THUNDER results so far (in docs/v2.md): Virchow2 (2 of 3 seeds), UNI2-h and H-optimus-0; it
  is still running for the other four.

In short:

* v2 has a higher CRoMa median than the paper on all 7 backbones.
* PathoROB RI, same protocol: v2 is higher on UNI2-h (+0.008), Virchow (+0.010) and
  Midnight-12k (+0.012), and on OpenMidnight (+0.012; not like-for-like, different base model).
  It is lower on H-optimus-0 (-0.002), Virchow2 (-0.002) and Phikon-v2 (-0.009). The paper chose
  its checkpoints by PathoROB RI; v2 does not use PathoROB, or any other test benchmark, to pick
  its step.
* The paper is better on CRoMa LTM10 for UNI2-h, H-optimus-0, Virchow2 and Phikon-v2, and on
  HEST for Midnight-12k and Phikon-v2 (by 0.003-0.004). Among the other backbones with the same
  base model, HEST differs by at most 0.002.
* v2 improves CPTAC over the frozen base on every backbone measured. Against the paper (verified
  for two backbones) it is higher on UNI2-h (+0.003) and lower on H-optimus-0 (-0.004).
* v2 trains for 100-225 steps on 256 locations per step with rank-16 LoRA, and needs no
  benchmark data to choose when to stop.

## Install

Python 3.12+. Use the torch wheel index that matches your CUDA driver (cu128 shown).

```bash
python -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
./.venv/bin/pip install -e ".[dev,pathorob,plism]"
python scripts/acquire_plism.py        # fetch and repack the PLISM corpus
```

## Train

Paths come from the environment:

| variable | meaning |
|---|---|
| `SPECTRA_PLISM_DIR` | PLISM repack directory (or `run.packed_dir` in the config) |
| `SPECTRA_RUNS_DIR` | run root (default `<repo>/runs/sweeps`) |
| `SPECTRA_CACHE_DIR` | tile caches (default `<runs root>/_cache`) |
| `SPECTRA_SBATCH_ARGS` | extra sbatch options, e.g. `--account=X --partition=Y` |
| `SPECTRA_BACKBONE_LOCAL_DIRS` | optional `repo=/dir,...` for gated backbones stored locally |

The whole recipe is one file, [`configs/v2/spectra_v2.yaml`](configs/v2/spectra_v2.yaml). As
written it is the held-out selection run. Backbones: `virchow2`, `uni2`, `hoptimus`, `virchow`,
`phikon`, `midnight`, `om11` (OpenMidnight-1.1), or any Hugging Face repo id the encoder can
load.

```bash
# 1. selection run (held-out GT450 / MY / 7 cores, P logged every 25 steps)
python scripts/sweep/submit.py configs/v2/spectra_v2.yaml --sweep selection --seeds 1 -o backbone.name=uni2

# 2. R1 stop step from the run's logged P
python scripts/sweep/select.py --sweep selection

# 3. all-data release config at that step, then 3 seeds
python scripts/sweep/make_final.py --backbone uni2 --step 175     # writes configs/v2/final/uni2.yaml
python scripts/sweep/submit.py configs/v2/final/uni2.yaml --sweep final --seeds 0,1,2
```

`make_final.py --backbone <b>` without `--step` uses the step of the reported model. Without
SLURM, train one config directly with `python -m spectra.sweep.train --config <yaml> [key=value ...]`
(`--dry-run` resolves the config and the data split without a GPU). Runs resume from
`resume.pt` after preemption; `scripts/sweep/status.py <sweep>` lists and resubmits dead runs.

## Evaluate

Each benchmark has a self-submitting runner in `scripts/evals/` (pinned environments in
`scripts/evals/env/`). Point it at a run directory and step:

```bash
bash scripts/evals/run_pathorob.sbatch --backbone uni2 --ckpt $SPECTRA_RUNS_DIR/final/<run> --step 175  # RI + CRoMa
bash scripts/evals/run_hest.sbatch     --backbone uni2 --ckpt $SPECTRA_RUNS_DIR/final/<run> --step 175
bash scripts/evals/run_thunder.sbatch  --backbone uni2 --ckpt $SPECTRA_RUNS_DIR/final/<run> --step 175
bash scripts/evals/run_cptac.sbatch    --backbone uni2 --ckpt $SPECTRA_RUNS_DIR/final/<run> --step 175
```

The header of each script lists its required environment variables. The readout used for each
benchmark and backbone is fixed in `src/spectra/evals/protocol.py`.

## Checkpoints

A run writes `ckpt/step_XXXXXXX.safetensors` (bfloat16) holding only the trained tensors: the
LoRA weights, the GeM exponent and the two projectors. Rebuilding the model needs the run's
`config.json` (backbone, LoRA rank / alpha / targets) next to the `ckpt/` directory.
`spectra.evals.member.load_member` loads a checkpoint onto the pretrained backbone and folds the
LoRA update into its weights; the projectors are only used for training.

The v2 weights are to be released. Previously released SPECTRA checkpoints are listed in the
[`main` branch README](https://github.com/RyanKim17920/SPECTRA/tree/main#released-weights)
([`medarc/openmidnight-and-spectra`](https://huggingface.co/collections/medarc/openmidnight-and-spectra)).

## Repo layout

| path | contents |
|---|---|
| `src/spectra/sweep/` | v2 trainer: config, PLISM sampler and held-out split, model and heads, loss, in-run PLISM eval, R1 rule |
| `src/spectra/evals/` | model specs and the checkpoint loader shared by the benchmark runners |
| `configs/v2/` | the v2 recipe |
| `scripts/sweep/` | submit, status, R1 selection, release config, post-hoc in-run eval |
| `scripts/evals/` | PathoROB / CRoMa, HEST, THUNDER and CPTAC runners and their pinned environments |
| `docs/v2.md` | full v2 recipe and results |
| `tests/` | unit tests; `tests/test_v2_*.py` cover the v2 trainer on CPU |
| `paper/`, `snapshots/`, `scripts/gentle.sbatch`, ... | the paper code, as on `main` |

## License

MIT, see `LICENSE`. Fine-tuned weights carry the license of their base model.

## Citation

```bibtex
@article{kim2026contrastive,
  title   = {Contrastive fine-tuning against stain and scanner bias improves robustness and downstream performance across pathology foundation models},
  author  = {Kim, Ryan and Muhammad, Hassan and Dulal, Anish and Scotti, Paul S. and Abraham, Tanishq Mathew},
  journal = {arXiv preprint arXiv:2611.10093},
  year    = {2026}
}
```
