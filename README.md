# SPECTRA

**SPECTRA** takes a frozen public pathology foundation model and trains a LoRA adapter
(rank 32, alpha 64, on every attention and MLP layer) with a contrastive InfoNCE loss over
**registered PLISM tiles**. Each step samples C = 2 acquisition conditions (scanner, stain) and
T = 900 tissue locations. For a query tile, the positive is the registered tile at the same
location under the other condition (the same section across scanners, an adjacent serial
section across stains), and the negatives are the other T-1 tiles of that same condition, so
acquisition identity alone cannot identify the match. The encoder is read through two heads,
the CLS token and a softplus generalized-mean (GeM) pool of the patch tokens. Negatives that
share the query's pseudo-tissue group are dropped from the GeM loss and up-weighted by e^3 in
the CLS loss.

Training is 500 steps with a checkpoint every 50. The shipped checkpoint is chosen by a
**tolerance rule (ε = 0.007)** on the PathoROB average-RI curve, not by taking the best number
after the fact. This repo is the method and its evaluation harness: 7 backbones × 3 seeds on PathoROB
(robustness index, the primary metric, never seen in training), HEST, THUNDER, CPTAC and
held-out PLISM retrieval.

## SPECTRA v2

A simplified post-paper recipe (code: `src/spectra/sweep`, `src/spectra/evals`, `scripts/sweep`,
`scripts/evals`). The paper trained 500 steps and selected the checkpoint with the ε = 0.007
tolerance rule on the PathoROB RI curve. v2 instead uses:

- one CLS head (no GeM head); two temperatures τ {0.1, 0.6}, weights 0.5 / 0.5
- same-core negatives kept as ordinary negatives (no +3 bias)
- queue adversary: linear stain and scanner heads, gradient reversal, 8192-feature memory bank
- slope-rule stopping on held-out PLISM selection runs (no benchmark data read)

```bash
python scripts/sweep/submit.py configs/v2/q8_selection.yaml --sweep q8_selection --seeds 0,1,2 -o backbone.name=uni2
python scripts/sweep/select.py --sweep q8_selection        # slope-rule pick per run
python scripts/sweep/make_final.py --backbone uni2 --picks <p0,p1,p2> --out configs/v2/q8_final_uni2.yaml
python scripts/sweep/submit.py configs/v2/q8_final_uni2.yaml --sweep q8_final --seeds 0,1,2
bash scripts/evals/run_hest.sbatch --backbone uni2 --ckpt $SPECTRA_RUNS_DIR/q8_final/<run> --step <pick>
```

Set `SPECTRA_PLISM_DIR`, `SPECTRA_RUNS_DIR`, `SPECTRA_CACHE_DIR` and `SPECTRA_SBATCH_ARGS`
first. The other benchmarks use `run_pathorob.sbatch`, `run_thunder.sbatch`, `run_cptac.sbatch`.

Results: pending.

## Results

Fine-tuned values are the mean over 3 seeds, each at its own tolerance-rule-selected
checkpoint; ± is two sample SDs across seeds.

| backbone | PathoROB mean RI | HEST mean *r* | CPTAC AUC | PLISM cross-stain top-1 |
|---|---|---|---|---|
| Phikon-v2      | 0.470 → **0.836** ± 0.014 | 0.3747 → **0.3906** | 0.6491 → **0.6870** | 0.696 → **0.897** |
| Midnight-12k   | 0.759 → **0.908** ± 0.005 | 0.3952 → **0.4122** | 0.6643 → **0.6898** | 0.560 → **0.883** |
| Virchow2       | 0.861 → **0.909** ± 0.007 | 0.4032 → **0.4089** | 0.6789 → **0.6879** | 0.696 → **0.903** |
| H-optimus-0    | 0.800 → **0.906** ± 0.004 | 0.4150 → **0.4226** | 0.6728 → **0.6895** | 0.830 → **0.915** |
| UNI2-h         | 0.757 → **0.907** ± 0.007 | 0.4138 → **0.4238** | 0.6750 → **0.6980** | 0.761 → **0.913** |
| Virchow        | 0.815 → **0.890** ± 0.004 | 0.4061 → **0.4083** | 0.6608 → **0.6839** | 0.597 → **0.931** |
| OpenMidnight   | 0.618 → **0.879** ± 0.007 | 0.3902 → **0.4048** | 0.6561 → **0.6844** | 0.486 → **0.782** |

RI rises on 7/7 backbones, and HEST, CPTAC and PLISM retrieval rise on 7/7 alongside it.

## Released weights

Collection: [`medarc/openmidnight-and-spectra`](https://huggingface.co/collections/medarc/openmidnight-and-spectra). Weights are full checkpoints with the LoRA update merged.

| repo | what it is | base model | licence |
|---|---|---|---|
| [`medarc/OpenMidnight-1.1`](https://huggingface.co/medarc/OpenMidnight-1.1) | the OpenMidnight 1.1 pretrained encoder (a base model, not a SPECTRA output) | n/a | Apache-2.0 |
| [`medarc/OpenMidnight-1.1-SPECTRA`](https://huggingface.co/medarc/OpenMidnight-1.1-SPECTRA) | SPECTRA-adapted | `medarc/OpenMidnight-1.1` | Apache-2.0 |
| [`medarc/H-optimus-0-SPECTRA`](https://huggingface.co/medarc/H-optimus-0-SPECTRA) | SPECTRA-adapted | `bioptimus/H-optimus-0` | Apache-2.0 |
| [`medarc/Virchow-SPECTRA`](https://huggingface.co/medarc/Virchow-SPECTRA) | SPECTRA-adapted | `paige-ai/Virchow` | Apache-2.0 |
| [`medarc/Midnight-12k-SPECTRA`](https://huggingface.co/medarc/Midnight-12k-SPECTRA) | SPECTRA-adapted | `kaiko-ai/midnight` | MIT |

Load with `timm.create_model("hf-hub:medarc/<repo>", pretrained=True)`; model cards give the exact
preprocessing for each repo.

## Install

Python 3.12+. Use the torch wheel index that matches your CUDA driver (cu128 shown).

```bash
python -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
./.venv/bin/pip install -e ".[dev,pathorob,plism]"
cp .env.example .env    # set SPECTRA_DATA and the other roots for your machine
```

## Reproduce

Roots are environment variables (`.env.example`, resolved in `src/spectra/paths.py`):
`SPECTRA_DATA` (corpora, caches), `SPECTRA_RUNS` (run dirs, `ri_curve.json`), `SPECTRA_INPUTS`
(local weights of gated backbones), `SPECTRA_EVALS`, `SPECTRA_CELLS`, `SPECTRA_HEST_WORK`
(eval outputs), `SPECTRA_PAPER` (generated figures and tables).

Train one backbone and seed with the paper settings (arms: `phikon`, `midnight`, `virchow2`,
`hoptimus`, `uni2`, `virchow`, `openmidnight`; seeds 0-2). Launcher defaults differ from the
paper, so set them explicitly; C = 2, LoRA r32/alpha 64, lr 1e-4 and both heads are defaults.

```bash
python scripts/acquire_plism.py                   # fetch and repack the PLISM corpus
SPECTRA_ARM=virchow2 SPECTRA_SEED=0 SPECTRA_T=900 SPECTRA_MASK=1 SPECTRA_BCLS=3.0 \
SPECTRA_BMEAN=-inf SPECTRA_MAX_STEPS=500 SPECTRA_CKPT_EVERY=50 \
SPECTRA_RUN_NAME=genMASK-c50-ms500-virchow2-s0-t900 \
  sbatch scripts/gentle.sbatch
```

Keep `SPECTRA_RUN_NAME` under 52 characters (THUNDER key limit). A follower scores PathoROB
RI at each checkpoint into `runs/<run>/ri_curve.json`; the ε = 0.007 tolerance rule
(`select_step_1se` in `scripts/final_recipe_report.py`) picks the checkpoint from it.

Evaluate:

```bash
python scripts/eval_checkpoints.py --help         # PathoROB RI per checkpoint
SPECTRA_RUN=<run> SPECTRA_STEP=0000100 sbatch scripts/hest_final5.sbatch   # HEST
bash scripts/submit_thunder.sh --backbone <b> --go                         # THUNDER
```

Regenerate paper figures (from frozen extracts in `paper/data/`, no GPU) and tables (needs eval outputs):

```bash
paper/make_all.sh            # figures into $SPECTRA_PAPER/figures
paper/make_all.sh --tables   # also every table into $SPECTRA_PAPER/tables
```

Table generators also write summaries to `results/` (gitignored). `data/` holds the published
leaderboards, reference numbers and the selected seed cells they read.

## Repo layout

| path | contents |
|---|---|
| `src/spectra/` | library: PLISM repack and pair sampling, backbones and LoRA, masked InfoNCE, per-benchmark eval adapters |
| `scripts/` | training entrypoint, SLURM submitters, eval runners and collectors, table generators |
| `paper/` | LaTeX source, figure/table scripts, frozen figure extracts (`paper/data/`) |
| `data/` | transcribed public leaderboards and frozen reference numbers |
| `tests/` | unit tests |
| `patches/` | patch enabling bootstrap RI in the PathoROB harness |
| `snapshots/falseneg-gated`, `snapshots/falseneg-pinned` | frozen copies of the training code used for the paper runs |

## License

MIT, see `LICENSE`. Each released checkpoint carries the license of its base model (table above).

## Citation

```bibtex
@inproceedings{kim2026spectra,
  title     = {Contrastive fine-tuning against stain and scanner bias improves robustness and downstream performance across pathology foundation models},
  author    = {Kim, Ryan and Muhammad, Hassan and Dulal, Anish and Scotti, Paul S. and Abraham, Tanishq Mathew},
  booktitle = {NeurIPS 2026 Workshop on AI at Scale for Clinical Impact (ASCI): Cancer Pathology Foundation Models},
  year      = {2026}
}
```
