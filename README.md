# waivphaet

Registered-pair contrastive fine-tuning of [`owkin/phikon-v2`](https://huggingface.co/owkin/phikon-v2)
for scanner/stain robustness. A **PHAET-class reproduction attempt**.

Waiv released PHAET with a paper (arXiv:2607.22861) that has **no method section** — no
loss, no algorithm, no corpus, no hyperparameters, and no code. The only disclosed
constraints are "lightweight, label-free fine-tuning" that "instills scanner invariance
progressively throughout the network", and that it is not teacher distillation.

So this is not a reproduction of a published recipe. It is a bet on the gap their own
related-work section leaves open: they dismiss "contrastive losses over co-registered
scanner pairs" with one sentence — *"These downstream methods all keep the backbone
frozen."* We apply that loss family **to the backbone**.

**Read [`PLAN.md`](PLAN.md) first.** It is the spec; every module docstring cites the
section it comes from.

## Design in one paragraph

PLISM (7 scanners × 13 stains × 16,278 Elastix-registered tiles) is our **training** set —
it is only a qualitative figure in Waiv's paper, so using it does not contaminate the
headline metric. Tile index *i* is the same tissue location in all 91 slides, so a
**positive** is free: same tile, different acquisition condition. **Negatives are drawn
from the anchor's own condition** — the one load-bearing detail (`PLAN.md` §2). LoRA
adapts all 24 transformer blocks, not the head (`PLAN.md` §2, their Fig 4 rules head-only
out). **PathoROB** is the primary metric and is never touched in training; PLISM retrieval
is a diagnostic and is never leaderboard-comparable.

## Layout

```
src/waivphaet/
  data/conditions.py   91 filenames -> (stain, scanner); deterministic held-out split
  data/repack.py       h5 -> contiguous (16278,224,224,3) uint8 memmap  [throughput]
  data/pairs.py        registered-pair batch sampler, same-condition negatives
  models/encoder.py    phikon-v2 + LoRA on all 24 blocks + 512-d projection head
  train/contrastive.py masked InfoNCE, AMP, grad accum, checkpointing
  eval/                thin adapters: PathoROB (primary), plismbench (diagnostic)
scripts/
  smoke_test.py        few real steps on the local slides, CPU or 1 GPU
  train_lora.py        full fine-tune entrypoint
  train_lora.sbatch    SLURM template (partition main/n, 8xH100)
third_party/           PathoROB + plism-benchmark clones (gitignored, not vendored)
```

## Environment

Phase 1 found no project venv with `h5py`, so this repo carries its own.

```bash
cd /admin/home/ryan.kim/waiv

# uv is on PATH at ~/.local/bin/uv
export UV_CACHE_DIR=/data/ryan.kim/uv_cache
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .

# torch MUST come from the cu128 index: this cluster's driver reports CUDA 12.8 and the
# default cu130 wheel fails CUDA init with "driver is too old".
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0

# PathoROB's FeatureDataManager (we call it to write embeddings) needs these:
uv pip install --python .venv/bin/python -e '.[pathorob]'
uv pip install --python .venv/bin/python pyarrow   # read their HF parquet directly
```

Verified imports: `torch 2.8.0+cu128`, `transformers 5.14.1`, `peft 0.20.0`, `h5py 3.16.0`.

### A second venv, on purpose: `.venv-pathorob`

PathoROB hard-pins `numpy==2.2.6`, `pandas==2.3.2`, `transformers==4.56.1`; we run numpy
2.5 / pandas 3.0 / transformers 5.14. Downgrading ours to match would be the wrong trade —
their *metric* is the only thing that needs those pins, and it is a pure-numpy kNN.

```bash
uv venv --python 3.12 .venv-pathorob
uv pip install --python .venv-pathorob/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-pathorob/bin/python -e third_party/PathoROB
```

torch 2.8.0 is the one shared pin and it already agreed (fixed here by the CUDA 12.8
driver, not by them). `run_robustness_index` picks this interpreter up automatically and
sets `PYTHONNOUSERSITE=1` — `~/.local/lib/python3.12/site-packages` on this machine holds
a pandas that otherwise shadows the venv's and makes the pins meaningless.

**Always** `export HF_HOME=/data/ryan.kim/hf_home`. `/admin` has ~1.4 TB free and must not
fill; `/data` has ~7.2 TB (`PLAN.md` §5). All scripts set this default themselves.

### Third-party harnesses

Cloned, not vendored — their source is gitignored:

```bash
git clone --depth 1 https://github.com/bifold-pathomics/PathoROB third_party/PathoROB
git clone --depth 1 https://github.com/owkin/plism-benchmark third_party/plism-benchmark
```

## Data

Only 2 of 91 `.h5` are local (`/data/ryan.kim/plism/`, ~2.3 GB each). Do not download the
rest without a reason — the full set is ~224 GB.

Each `.h5` is *flat*: 16,278 top-level datasets named `tile_{level}_{x}_{y}`, each
`(224,224,3)` uint8, uncompressed. Key order is byte-identical across all 91 files, which
is what makes index-aligned positives possible.

### Repack (do this before any training)

```bash
./.venv/bin/python -m waivphaet.data.repack \
  --h5-dir /data/ryan.kim/plism --out-dir /data/ryan.kim/plism/repacked --verify --benchmark
```

16,278 tiny HDF5 datasets per slide means random pair sampling is I/O bound. Repacking to
a contiguous `.npy` turns a tile read into one `pread` at a fixed offset.

**Measured on the 2 local slides** (2000 random tile reads each):

| slide | h5 | repacked memmap | speedup |
|---|---|---|---|
| `GIVH_AT2` | 7,351 tiles/s | 922,611 tiles/s | **125×** |
| `GIVH_GT450` | 7,371 tiles/s | 827,660 tiles/s | **112×** |

Repack cost ≈ 14 s and 2.28 GiB per slide (≈ 21 min / 208 GiB for all 91). Byte-exactness
verified on a random 64-tile sample per slide (0 mismatches).

Caveat: the memmap side is page-cache-warm because it was just written, so 112–125× is an
upper bound. The *structural* win — one seek instead of a B-tree lookup plus a seek — holds
cold, and either way the h5 number (7.4k tiles/s) is the one that would have bottlenecked
an 8×H100 node.

Residual Elastix misregistration is ~5–50 px, so positives are near-identical *shifted*
crops, never pixel-exact. Do not add augmentations that assume otherwise.

## Run

```bash
# wiring check (CPU is fine, ~11-17 s/step)
./.venv/bin/python scripts/smoke_test.py --steps 4 --device cpu

# full run
sbatch scripts/train_lora.sbatch --lora-rank 16 --temperature 0.07
```

## Evaluation — PathoROB (primary)

Login nodes have no GPU, so extraction goes through SLURM. One H100 does the whole
benchmark (99,392 patches) in ~5 min at ~340 img/s; the metric itself is a CPU kNN.

```bash
# 1. features -> third_party/PathoROB/data/features/{model}/{dataset}/{center}.npz
sbatch scripts/extract_pathorob.sbatch phikonv2_clsmean_ours "camelyon tolkach_esca tcga"
# a fine-tuned checkpoint is the same call with a third argument:
#   sbatch scripts/extract_pathorob.sbatch waiv_step5000 camelyon runs/.../ckpt.pt

# 2. robustness index + the three-way comparison
PYTHONNOUSERSITE=1 ./.venv-pathorob/bin/python scripts/pathorob_gate.py \
  --model phikonv2_clsmean_ours --datasets camelyon tolkach_esca tcga
```

The image data streams from the ungated HF parquet repos and is small — 0.47 GB
(camelyon) + 0.32 GB (tolkach_esca) + 1.12 GB (tcga) = **1.91 GB**, cached under
`$HF_HOME`. There is no need to stage anything by hand.

### Phase-2 gate: PASS

Base `owkin/phikon-v2`, clsmean 2048-d, fp32, our extractor + their metric. SLURM job
369018, 1×H100, 10m32s wall for all 99,392 patches; the metric then took ~30 s (camelyon),
~40 s (tolkach_esca) and 6m49s (tcga, paired, 112,800 rows) on the login node.

| dataset | ours | PathoROB committed ref | Waiv Table 1 | Δ vs ref |
|---|---|---|---|---|
| camelyon | 0.018951 | 0.018951 | 0.019 | **+0.000000** |
| tolkach_esca | 0.768112 | 0.768114 | 0.768 | −0.000002 |
| tcga | 0.618771 | 0.618772 | 0.619 | −0.0000005 |
| **Avg** | **0.468611** | 0.468612 | **0.469** | −0.000001 |

Two things worth separating. **(a)** Our pipeline reproduces PathoROB's own committed
`phikonv2_clsmean` row to 6+ decimals — camelyon is bit-identical, and the µ-scale drift
on the other two is float summation order, not a pipeline difference. **(b)** PathoROB's
committed reference independently agrees with Waiv's quoted Table-1 base row to the 3
decimals Waiv printed. Those are two separate sources and they corroborate each other, so
the 0.019 / 0.469 targets in `PLAN.md` are sound.

The harness is therefore trustworthy: a Camelyon RI that moves after fine-tuning is a real
effect, not harness drift.

Default split holds out scanners `GT450`, `S210` and stains `HRH`, `KR`, `MY` → 50 train /
41 held-out conditions (`PLAN.md` §4 phase 7). It is *named*, not sampled, so it is
reproducible without carrying a seed.

## Reporting discipline (`PLAN.md` §6)

- Cross-stain and cross-scanner **separately** — the composite hides the hard axis.
- Never cosine similarity alone (PLIP: 0.878 cosine at 0.054 top-10).
- Any PLISM number is a **training diagnostic**. Label it. Never print it next to
  H0-mini's 0.541.
- Retention (HEST / THUNDER / Patho-Bench) alongside **every** robustness claim, as a
  pair. Forgetting is the default outcome here, not a tail risk.
