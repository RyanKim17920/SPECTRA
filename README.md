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
  models/encoder.py    ANY HF ViT + LoRA on all blocks + 512-d projection head
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

## Second backbone: the pipeline is model-agnostic

The recipe is a reconstruction (Waiv published no method), so a single-backbone result is
weak evidence. `kaiko-ai/midnight` is the second test: **ungated and MIT**, and its Waiv
counterpart MASCARET carries the *largest* published gain in their Table 1
(Avg RI 0.759 → **0.924**; THUNDER rank sum 70 → 34). H0-mini / UNI2-h / Virchow2 /
Prov-GigaPath are all gated and are not options.

`EncoderConfig.backbone` is now a parameter and everything geometric is read off the
loaded config — nothing about ViT-L/16 survives in the code:

| | phikon-v2 | midnight |
|---|---|---|
| arch | Dinov2 ViT-L/16 | Dinov2 ViT-g/14 |
| hidden / blocks | 1024 / 24 | 1536 / 40 |
| `embed_dim` cls / clsmean | 1024 / **2048** | 1536 / **3072** |
| tokens @224 | 196 | 256 |
| FFN | `mlp.fc1` / `mlp.fc2` | `mlp.weights_in` / `mlp.weights_out` (SwiGLU) |
| normalisation | ImageNet | **(0.5,0.5,0.5) / (0.5,0.5,0.5)** |
| LoRA targets | 144 = 6/block × 24 | 240 = 6/block × 40 |

### Two ways this could have failed silently

Both would have produced a running pipeline, plausible numbers, and no warning anywhere.
They are the reason the refactor is worth more than a `backbone=` kwarg.

1. **SwiGLU FFN.** The old LoRA target list was the fixed tuple
   `(query, key, value, dense, fc1, fc2)`. Midnight sets `use_swiglu_ffn=True`, so `fc1`
   and `fc2` match *nothing* — but `query/key/value/dense` still match in every block, so
   the existing "LoRA covers all blocks" assertion would have **passed** while the entire
   FFN stayed frozen (4/block instead of 6/block, ~⅓ of each block's parameters adapted).
   Downstream that reads as "LoRA is just weaker on ViT-g", not as a bug. Targets are now
   *discovered* by leaf name, and the per-block count is asserted non-empty **and uniform**
   and printed at build time.
2. **Normalisation.** Midnight's model card: *"trained on 224x224 images normalized with a
   mean of (0.5, 0.5, 0.5) and a standard deviation of (0.5, 0.5, 0.5). Please ensure you
   apply these exact normalization parameters."* Our stack hardcoded ImageNet stats —
   correct for phikon-v2 (it is what PathoROB's own `Phikonv2ModelWrapper` uses, which is
   why we reproduce their RI to 6 decimals) and wrong here. It changes no shape and throws
   no error; it just costs base accuracy. It would have surfaced as our base-Midnight row
   *disagreeing with Waiv's published 0.759* — i.e. as false evidence that our harness is
   unfaithful, which is the exact question that comparison exists to answer.
   `BACKBONE_NORMALIZATION` is now table-driven; an unregistered backbone warns loudly.

### THUNDER pooling is per-backbone and is not our choice

arXiv:2607.22861 §3 (line 106): CLS+mean-pool concatenation was used for **all** models in
PathoROB, but in THUNDER only for Virchow2, AquaViT, H0-mini and **Midnight-12k**. So
phikon-v2 is `cls` in THUNDER (which is also THUNDER's own published `phikon2` protocol)
and midnight is `clsmean`. `thunder_model._default_pooling` resolves it from the backbone;
`run_thunder.sbatch` takes `auto`. Hardcoding either one makes the base-vs-fine-tuned rank
sums non-comparable to their table.

### Regression: base phikon-v2 is bit-identical across the refactor

`scripts/regression_bitcheck.py` loads the extractor from two worktrees and compares raw
feature arrays. Pre-refactor `861cacf` vs post-refactor HEAD, base phikon-v2, 64 camelyon
patches, fp32 CPU:

| pooling | dim | inputs identical | features identical | max abs delta |
|---|---|---|---|---|
| clsmean | 2048 | yes | **yes** | 0.000e+00 |
| cls | 1024 | yes | **yes** | 0.000e+00 |

`robustness_index` is a deterministic function of the written features, so byte-identical
features imply a byte-identical RI. That is sharper than the RI comparison itself, which
has to explain away µ-scale float-summation drift on tolkach and tcga. It does not replace
the full-benchmark rerun (that also covers the parquet read, the npz layout and the
metric) — which is why the gate is still rerun end-to-end.

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

### First fine-tuned readout: 300-step rank-8 smoke checkpoint

Job 369019 (rank 8, all 24 blocks, 2 groups × 32, 300 steps). Features re-extracted
through the *identical* pipeline — same Resize(224)→CenterCrop(224)→ToTensor→ImageNet
Normalize, same clsmean 2048-d, same fp32, same npz layout — with the LoRA adapter as the
only difference (job 369036, 1×H100, 5m30s for 99,392 patches).

| dataset | base | +LoRA | Δ RI | Waiv T1 target | base bal-acc | +LoRA bal-acc | Δ bal-acc |
|---|---|---|---|---|---|---|---|
| camelyon | 0.018951 | **0.525523** | +0.506572 | 0.702 | 0.9536 | 0.9653 | +0.0118 |
| tolkach_esca | 0.768112 | **0.910295** | +0.142181 | 0.932 | 0.9588 | 0.9684 | +0.0097 |
| tcga | 0.618771 | **0.772558** | +0.153786 | 0.785 | 0.9202 | 0.9177 | −0.0026 |
| **Avg** | **0.468611** | **0.736125** | **+0.267514** | **0.806** | 0.9442 | 0.9505 | +0.0063 |

The decomposition says *why*, and it is the mechanism we wanted rather than a shortcut.
The entire gain lands on `confounder_insensitivity` (camelyon 0.0011→0.0541, tolkach
0.261→0.729, tcga 0.238→0.532) while `prediction_performance` is flat (0.941→0.954,
0.929→0.937, 0.873→0.868) and `generalization_index` barely moves. Balanced accuracy is
flat-to-up on all three, so this is not a robustness gain bought by destroying the
biological signal. `k_opt` is unchanged per dataset (11 / 46 / 61), so the kNN operating
point is the same one the base model was scored at.

Read it with three caveats. **(1)** The checkpoint was trained *after* the
`masked_info_nce` orientation fix (`5907f54`, committed 00:32:45; job started 00:33:20,
and its artifacts — `embed_probe.py` output, `negatives_per_anchor` in `metrics.json` —
exist only in post-fix code), so the condition-spanning-negatives shortcut is **not** the
explanation. **(2)** It is still a 300-step smoke run on a saturated objective (top-1 1.0
by step 150), single seed, no error bars. **(3)** Camelyon's base RI of 0.019 is
pathologically low, so its headline jump is the least surprising of the three. The
movement is far outside harness noise (the Phase-2 gate reproduces to 6 decimals) — but it
is one run, and the honest claim is "the backbone moved the metric in the right way", not
"we hit the PHAET targets". Avg is 0.736 against a 0.806 target.

Default split holds out scanners `GT450`, `S210` and stains `HRH`, `KR`, `MY` → 50 train /
41 held-out conditions (`PLAN.md` §4 phase 7). It is *named*, not sampled, so it is
reproducible without carrying a seed.

### The first real run: job 369043 — Avg RI plateaus **on** the 0.806 target

`scripts/train_real.sbatch`, 1×H100 for training + 1×H100 running
`scripts/eval_checkpoints.py` as a live follower. Rank 32 / alpha 64, all 24 blocks,
**2 groups × 192 = 191 negatives per anchor**, 4000 steps, all **91** slides, lr 1e-4,
T 0.07, bf16, 1.91 s/step.

Three things changed from the smoke run, each fixing a defect it exposed:

1. **191 negatives, up from 31.** The smoke objective was *solved* — top-1 1.0 by step
   70, loss 0.038 — so its late gradient was noise. Negatives come only from the
   anchor's condition-homogeneous group, so more negatives requires a bigger *forward*
   batch (768 images/step); grad accumulation cannot buy them. At the measured
   0.21 GiB/image that is ~161 GiB, so this run enables gradient checkpointing
   (21.77 GiB peak, ~+35% step time). **2 × 192 is not reachable without it.**
2. **All 91 slides.** The smoke saw 59–72 mid-stream.
3. **Rank 32 and 4000 steps** vs rank 8 and 300 (3.07M tiles vs 38k).

| step | camelyon | tolkach_esca | tcga | **Avg RI** | Avg bal-acc | sep cross-scanner | top-1 cross-scanner |
|---|---|---|---|---|---|---|---|
| base | 0.018951 | 0.768112 | 0.618771 | **0.4686** | 0.9442 | 0.3760 | 0.8579 |
| 500 | 0.698211 | 0.928967 | 0.783647 | **0.8036** | 0.9454 | 0.3936 | 0.9951 |
| 1000 | 0.716945 | 0.927932 | 0.779072 | **0.8080** | 0.9437 | 0.4164 | 0.9964 |
| 1500 | 0.704983 | 0.927470 | 0.777217 | **0.8032** | 0.9425 | 0.4267 | 0.9965 |
| 2000 | 0.713128 | 0.925482 | 0.776589 | **0.8051** | 0.9418 | 0.4270 | 0.9964 |
| Waiv T1 | 0.702 | 0.932 | 0.785 | **0.806** | — | — | — |

**The curve is a plateau, not a climb.** Avg RI is 0.803–0.808 across every checkpoint
from 500 on — it lands on the 0.806 target by step 500 and then does not move. Steps
500→2000 buy nothing on the headline metric. That is the single most useful result
here: the next run should sweep *away* from this operating point rather than train
longer, and the cost of a data point is ~500 steps, not 4000.

**Read the two probe columns separately, because they disagree.** Rank-based top-1
jumps and stays (0.858 → 0.995); mean-cosine **separation** improves far less
(0.3760 → 0.4270) because *matched* and *random* rise together — the smoke run's
"matched up, random up" signature persists. Separation did move monotonically this
time, which the smoke run could not show: its before/after probes scored *different*
condition sets (60 vs 67) because the pin landed in `ffc6e4c` after that job's probes
ran. Both probes here are pinned to the identical 91-condition set. Matched cosine is
still never the claim — it rises under collapse — but the collapse gauge
`within_condition_random` *fell* from 500 → 1000 (0.5771 → 0.5535) while separation
rose, which is the opposite of the collapse signature.

**Objective saturation: fixed, but not eliminated.** Top-1 oscillates 0.90–0.99 through
step 2000 (held-out-condition top-1 0.949 → 0.965) where the smoke run was pinned at
1.0 from step 70. There is still gradient signal at step 2000. It is drifting upward
though, so **2 × 256 (255 negatives) is the next escalation** — measured to fit at
28.55 GiB, 2.91 s/step.

**Retention:** balanced accuracy 0.9454 → 0.9418 against a 0.9442 base — flat, with a
slow downward drift worth watching. `k_opt` is unchanged (11 / 46 / 61), so the kNN
operating point is the base model's. HEST / THUNDER / Patho-Bench are still not wired,
so this is the only forgetting detector in play (`PLAN.md` §3 risk 1).

**The adapter is proven applied at every eval**, not assumed: the extractor compares
against `disable_adapter()` and exits non-zero below 1e-4. Observed `rel_l2_delta` is
0.73–0.93 across all datasets and checkpoints, recorded per-point in `ri_curve.json`.

Caveats unchanged: one seed, no error bars, and camelyon's 0.019 base is pathologically
low so its jump is the least surprising of the three.

## Reporting discipline (`PLAN.md` §6)

- Cross-stain and cross-scanner **separately** — the composite hides the hard axis.
- Never cosine similarity alone (PLIP: 0.878 cosine at 0.054 top-10).
- Any PLISM number is a **training diagnostic**. Label it. Never print it next to
  H0-mini's 0.541.
- Retention (HEST / THUNDER / Patho-Bench) alongside **every** robustness claim, as a
  pair. Forgetting is the default outcome here, not a tail risk.

### Note: `adapter_rel_l2_delta` provenance (resolved)

`ri_curve.json` for run `waiv-real-369043` records correct, dataset-distinct
adapter deltas for steps 500-2500 (camelyon ~0.74-0.76, tolkach ~0.92-0.93,
tcga ~0.78-0.80). **Step 3000+ is corrupted**: all three datasets read an
identical 0.2520.

Cause: commit `ea7fa1f` moved the adapter-applied assertion into the shared
`build_model()` so HEST/THUNDER would inherit it. From that commit on, every
extraction emits *two* `[adapter-check]` lines -- a deterministic synthetic-batch
check inside `build_model()`, then the real-tile check in `main()`. The collector's
`_REL_L2.search()` took the first (synthetic) match. Fixed to `findall()[-1]`.

Identical across datasets because the synthetic batch is fixed (seed 1234);
different across checkpoints (0.2520 @ 3000, 0.2539 @ 3500) because the weights
change. **No RI value is affected** -- RI is computed by PathoROB over the written
.npz features, independent of this diagnostic. All deltas, synthetic or real, are
orders of magnitude above the 1e-4 abort threshold, so the adapter was applied in
every extraction.

### THUNDER segmentation coverage (2 of 4 datasets)

THUNDER's 16-dataset suite includes 4 segmentation sets. We can report 2:

| dataset | status |
|---|---|
| ocelot | usable |
| pannuke | usable |
| segpath_epithelial | **absent** -- not present in any tree on this cluster |
| segpath_lymphocytes | **infeasible** -- see below |

`segpath_lymphocytes` was measured at **5,048 s/epoch x 200 epochs = ~281 h** (job 369061,
2/200 epochs in 2h50m) against a 20 h wall limit, so it can only ever be killed having
produced nothing. Cause: segmentation is forced to `online_loading` -- the decoder needs
raw masks, so the h5 embedding cache that rescues kNN/linear-probing cannot be used -- and
this is the largest set (23 GB of raw PNG pairs), so the frozen ViT is re-run over all of
it every epoch. Both the base and fine-tuned jobs were cancelled.

Consequence: THUNDER's `benchmark_segmentation` aggregate is not reportable, and our
segmentation numbers cover ocelot + pannuke only. **Classification (12 datasets) is
unaffected**, which is what Waiv's Table 2 kNN / linear-probe / few-shot columns measure.

## Second model: Midnight-12k (generalization validated)

The recipe was re-run unchanged on `kaiko-ai/midnight` (ViT-g/14, 1.18B, SwiGLU FFN,
40 blocks, 3072-d clsmean, (0.5,0.5,0.5) normalization) to test whether it is a
*recipe* rather than a phikon-v2-specific result.

**Harness validated first.** Base Midnight through our pipeline reproduces Waiv's
published base row: camelyon 0.4780 (0.478), tolkach 0.9411 (0.941), tcga 0.8575
(0.858), **Avg 0.7589 vs 0.759**. Separately, the backbone-agnostic refactor left
phikon-v2 bit-identical (Avg RI 0.4686113 vs 0.468611).

**Training.** Identical config to the phikon-v2 run -- rank 32/alpha 64, 2x192 =
191 negatives/anchor, 768 images/step with gradient checkpointing, all 91 PLISM
slides, held-out scanners GT450+S210 and stains HRH+KR+MY. LoRA targets 240 =
6/block x 40 blocks, leaves `[dense, key, query, value, weights_in, weights_out]`
-- the SwiGLU FFN projections are adapted, which the old hardcoded target list
would have silently skipped while still passing the block-coverage assertion.
1500 steps, 3h26m on 1xH100. Held-out top-1 rose 0.943 -> 0.972 while train top-1
stayed flat.

**Result (PathoROB Avg RI):**

| step | Avg RI | bal. acc |
|---|---|---|
| base | 0.7589 | -- |
| 250 | 0.8981 | 0.9652 |
| **500** | **0.9080** | 0.9663 |
| 750 | 0.9068 | 0.9664 |
| 1000 | 0.8996 | 0.9637 |
| 1250 | 0.8995 | 0.9632 |
| 1500 | 0.8997 | 0.9633 |
| MASCARET (Waiv) | 0.924 | -- |

**0.7589 -> 0.9080 = 90.3% of the available headroom**, 0.016 short of MASCARET,
balanced accuracy flat across all six checkpoints (0.963-0.966).

Same shape as phikon-v2: the gain arrives by the first checkpoint and then
plateaus (six checkpoints span 0.010). Waiv published no curves, so this
early-plateau behaviour is not visible in their paper -- and it means their
"lightweight" claim is, if anything, understated: a few hundred steps suffice.
