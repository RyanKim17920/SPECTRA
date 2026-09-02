> **[STATUS BANNER — added 2026-08-31, see `docs/README.md` for the current doc map]**
>
> HISTORICAL — superseded by RUNBOOK.md for how to run the project. Environment/venv layout described here may still be accurate; treat RUNBOOK.md as authoritative for commands.

# Reproducing

Environments, data preparation, training and evaluation for the two backbones reported in
[`RESULTS.md`](RESULTS.md). To run the same recipe on a *different* backbone, see
[`NEW_MODEL.md`](NEW_MODEL.md).

---

## 1. Reproducing

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
  --h5-dir /data/plism --out-dir /data/plism/repacked --verify --benchmark
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

# THUNDER. segpath_epithelial runs at 9 epochs and segpath_lymphocytes at 21, per THUNDER's
# own docs/guidelines.md — the generic 200-epoch default is wrong for these two sets.
sbatch scripts/run_thunder.sbatch <dataset> <task> <run_name> [ckpt]
PYTHONNOUSERSITE=1 THUNDER_BASE_DATA_FOLDER=/data/ryan.kim/thunder \
  ./.venv-thunder/bin/python scripts/collect_thunder.py --model ft1000_cls

# HEST
sbatch scripts/run_hest.sbatch <run_name> <pooling> [ckpt]
PYTHONNOUSERSITE=1 ./.venv-hest/bin/python scripts/collect_hest.py --pooling cls
# Midnight: --base/--runs are MANDATORY. The default --base base_<pooling> would silently
# diff Midnight's clsmean run against phikon-v2's baseline.
PYTHONNOUSERSITE=1 ./.venv-hest/bin/python scripts/collect_hest.py \
  --pooling clsmean --base mbase_clsmean --runs mft500_clsmean
```

`collect_thunder.py` deliberately does **not** compute THUNDER's rank sum: it is a rank over
whatever roster is on the leaderboard, and three sources give three different numbers for
phikon-v2 (Waiv 97, the THUNDER paper 77, the live leaderboard 89). Absolute per-dataset F1
means the same thing on every run.


---

## 2. Verification the numbers rest on

- **PathoROB harness.** Base phikon-v2 through our extractor + their metric reproduces
  PathoROB's own committed `phikonv2_clsmean` row to 6+ decimals (camelyon bit-identical;
  µ-scale drift elsewhere is float summation order), and that committed reference
  independently agrees with Waiv's Table-1 base row. Two separate sources corroborating.
  Base Midnight likewise reproduces Waiv's published 0.759 (0.7589).
- **THUNDER segmentation harness, phikon-v2 only.** With all 4 segmentation sets in, our
  **base phikon-v2** run cross-checks against THUNDER's own published leaderboard
  (arXiv:2507.07860v3) at mean Δ **+0.30** (max |Δ| +1.3). On that backbone the harness
  reproduces published numbers on exactly the sets Waiv used, independently of any
  fine-tuning claim. `scripts/collect_thunder.py` prints this as a `# cross-check` line.

  **There is no equivalent check for Midnight.** An earlier revision of this file claimed a
  Midnight base mean Δ of +1.31 against the same leaderboard. That figure was real output but
  it was not what the label said: `collect_thunder.py` held published numbers for phikon-v2
  only, so the Midnight rows were being differenced against *phikon-v2's* published row. That
  is a comparison between two backbones, not a reproduction check, and it supported no claim
  about our harness. The collector now withholds the published column unless it holds numbers
  for the backbone in question, so the misleading line can no longer be produced. A real
  Midnight check would require transcribing that model's published rows into a `MIDNIGHT` key
  of `PUBLISHED`; the structure accepts it with no other edit. Whether the THUNDER leaderboard
  carries a Midnight-12k row at the granularity we would need has not been checked — that is
  the first thing to confirm before attempting it.
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
