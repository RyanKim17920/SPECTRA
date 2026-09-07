# Release manifest -- Spectra LoRA adapters

**Scope.** 21 LoRA adapters: 7 backbones x 3 seeds. Every adapter is rank `r=32`,
`lora_alpha=64` (scaling `alpha/r = 2.0`), `lora_dropout=0.0`, `bias="none"`,
`use_dora=false`, `use_rslora=false`, `fan_in_fan_out=false`, `modules_to_save=null`,
`init_lora_weights=true`, written by PEFT 0.20.0, tensors stored fp32.

**Only the 21 final 1-SE-rule-selected checkpoints are released.** Each training run
wrote checkpoints every 50 steps (`step_0000050 ... step_0000500`); the released step per
seed is the one `scripts/seed_stats.py:rule_selected` /
`final_recipe_report.select_step_1se(run_dir, 0.007)` picks off that run's own
`ri_curve.json`. No intermediate step is published. The selection is per seed, so seeds of
the same backbone are not all at the same step.

Facts below are extracted from `/admin/home/ryan.kim/pathfm-cells/<cell>/model.py`
(the frozen evaluation cells that produced the paper numbers),
`/admin/home/ryan.kim/waiv/src/waivphaet/models/encoder.py`, the on-disk
`adapter_config.json` files, and the run `config.json` files.

Licence and gating status per backbone is **TODO -- under separate investigation.** No
licence claim is made anywhere in this document. See the flagged hazard in the checklist.

---

## 1. Summary

| Backbone (slug) | HF base repo | Pinned revision | Seeds | Selected steps | Adapter `.safetensors` | Adapted modules | Proposed release repo |
|---|---|---|---|---|---|---|---|
| `phikon2` | `owkin/phikon-v2` | `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd` | s0, s1, s2 | 200, 200, 200 | 56,662,696 B (54.0 MiB) | 144 (24 blocks x 6) | `spectra-phikon-v2-lora` |
| `midnight` | `kaiko-ai/midnight` | `adc6b15679c981cce6f9b018bbad09d16eeeda9f` | s0, s1, s3 | 150, 100, 100 | 141,625,400 B (135.1 MiB) | 240 (40 x 6) | `spectra-midnight-lora` |
| `virchow2` | `paige-ai/Virchow2` | `3158645804b69e3f3bc4439d4116edddf0840a72` | s0, s1, s3 | 100, 150, 100 | 83,949,648 B (80.1 MiB) | 128 (32 x 4) | `spectra-virchow2-lora` |
| `hoptimus0` | `bioptimus/H-optimus-0` | `b145cc1e6c6b30d3251aa8b1f844e6974188a743` | s0, s1, s3 | 100, 100, 100 | 125,867,792 B (120.0 MiB) | 160 (40 x 4) | `spectra-h-optimus-0-lora` |
| `uni2h` | `MahmoodLab/UNI2-h` | `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` | s0, s1, s2 | 100, 150, 100 | 75,520,552 B (72.0 MiB) | 96 (24 x 4) | `spectra-uni2-h-lora` |
| `virchow1` | `paige-ai/Virchow` (v1) | `19eebc84ae33e79f1b2d866e6ff90ae50e522f9a` | s0, s1, s2 | 150, 150, 150 | 83,949,648 B (80.1 MiB) | 128 (32 x 4) | `spectra-virchow-lora` |
| `openmidnightsq` | `SophontAI/OpenMidnight` | **no SHA** -- local conversion, see §3.7 | s0, s1, s2 | 150, 150, 150 | 125,867,792 B (120.0 MiB) | 160 (40 x 4) | `spectra-openmidnight-lora` |

**Repo layout decision: 7 repos, not 21.** One HuggingFace repo per backbone, with the
three seeds as subfolders `seed0/`, `seed1/`, `seed2/` inside it. Reasons: (a) all three
seeds of a backbone share one base model, one revision pin, one inference contract and one
model card, so 21 repos would triplicate identical documentation; (b) `PeftModel.from_pretrained(base, repo, subfolder="seed0")`
loads a subfolder natively, so nothing is lost at load time; (c) seed variance is a
reported result of the paper, so the seeds belong together as one artefact.

Proposed per-repo tree:

```
spectra-<backbone>-lora/
  README.md                 # real model card (see checklist)
  seed0/adapter_config.json
  seed0/adapter_model.safetensors
  seed0/training_config.json    # copy of the run dir config.json, JSON-repaired
  seed0/ri_curve.json           # provenance: the curve the 1-SE rule selected from
  seed1/...
  seed2/...
```

All 21 source adapters are present on `/admin` and were verified to exist, parse, and
carry the expected tensor count at the time this manifest was written.

Common training configuration -- **identical across all 21 runs** (verified by reading each
run's `config.json`): `lr=1e-4`, `weight_decay=0.05`, `warmup_steps=200`, `max_steps=500`,
`temperature=0.07`, `grad_clip=1.0`, `n_groups=4`, `group_size=64`, `grid=true`,
`grid_conditions=2`, `grid_tiles=900`, `amp_dtype=bfloat16`, `ckpt_every=50`,
`retention_kl_weight=0.0`, `mask_same_core=true`, `same_core_logit_bias_cls=3.0`,
`same_core_logit_bias_mean=-Infinity`, `split_heads=["cls","mean"]`, `cls_weight=0.5`,
`mean_weight=0.5`, `pool_head="gem"`, `proj_out_dim=512`, `pooling="clsmean"`,
`grad_checkpointing=true`, training corpus `packed_dir=/data/plism/repacked`. The `seed`
field equals the seed label (0/1/2/3). Note that at the selected steps (100--200) training
is still **inside** the 200-step warmup, i.e. the released checkpoints are un-annealed.

---

## 2. Inference contract common to all 21

The paper's numbers come from a hand-rolled merge, not from `peft` at eval time (`peft` is
not installed in the shared benchmark venvs). The reference implementation is
`merge_lora_` in each cell's `model.py`, e.g.
`/admin/home/ryan.kim/pathfm-cells/phikon2-c50-s0-step200/model.py`.

Contract, in order:

1. Build the base architecture from the **local** `config.json` (never the hub default cfg)
   with `num_classes=0, global_pool="token"` for timm rows, or
   `AutoModel.from_config(AutoConfig.from_pretrained(dir))` for HF rows.
2. `load_state_dict(..., strict=True)` from the raw local weight file. No
   `from_pretrained` -- a partial match would leave tensors randomly initialised.
3. Cast the whole backbone to **fp32**, then merge LoRA in fp32, before any device move
   or dtype cast: `W <- W + (alpha/r) * (B @ A)`, with `A` shaped `(r, in)` and `B` shaped
   `(out, r)`; peft key prefix `base_model.model.` is stripped.
4. `backbone.eval().cuda()`, all parameters `requires_grad=False`.
5. Feature extraction runs under `torch.inference_mode()` +
   `torch.autocast("cuda", torch.float16)` (fp16). Exception: the PathoROB adversarial
   (PGD) path runs in **fp32 with no autocast**; the OpenMidnight/H-optimus cells set
   `ATTACK_BATCH=32` for that reason (`EXTRACTION_BATCH=1024` throughout;
   `ATTACK_BATCH=128` on the fp16 cells).
6. **Nothing else is loaded.** `projector.pt`, `projector_cls.pt`, `projector_mean.pt`,
   `projector_heads.json` and `pool_head.pt` are training-only InfoNCE machinery and are
   explicitly not read by any feature path.

Merge invariants asserted by the reference code (worth keeping in the model card): every
targeted module is an `nn.Linear`; every merged delta has non-zero norm; the merged count
must equal `num_blocks * per_block`.

Readout is **per backbone** and matches THUNDER's `extract_embedding`
(`thunder/src/thunder/models/pretrained_models.py`); using one rule for all seven would
silently halve or double the feature width against the published leaderboards.

---

## 3. Per-backbone sections

Adapter path pattern for every entry below:

```
/admin/home/ryan.kim/waiv/runs/<RUN>/<STEP>/adapter/adapter_config.json
/admin/home/ryan.kim/waiv/runs/<RUN>/<STEP>/adapter/adapter_model.safetensors
```

### 3.1 phikon2 -- `owkin/phikon-v2`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 200 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165/step_0000200/adapter/` |
| s1 | 200 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-phikon-s1-t900-407565/step_0000200/adapter/` |
| s2 | 200 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-phikon-s2-t900-407568/step_0000200/adapter/` |

- Base: `owkin/phikon-v2` @ `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`.
  Local pin used for the paper: `/admin/home/ryan.kim/pathfm-inputs/phikon-v2/`
  (`config.json`, `model.safetensors`, `preprocessor_config.json`).
- Loader: **transformers**. `Dinov2Model` (`model_type: dinov2`); timm cannot build it.
  Asserted: `hidden_size=1024`, `num_hidden_layers=24`, `num_attention_heads=16`,
  `patch_size=16`, `use_swiglu_ffn=false`, 439 state tensors.
- Target modules (from the real `adapter_config.json`): `{query, key, value, dense, fc1, fc2}`
  -- **6 per block**, resolved as
  `encoder.layer.<i>.attention.attention.{query,key,value}`,
  `encoder.layer.<i>.attention.output.dense`, `encoder.layer.<i>.mlp.{fc1,fc2}`.
  144 modules / 288 tensors.
- Transform: `Resize(224, bicubic)` -> `CenterCrop(224)` -> `ToTensor` ->
  `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`. Asserted against the
  checkpoint's own `preprocessor_config.json` (`shortest_edge=224`, `resample=3`,
  `crop_size=224`, `do_center_crop`, `do_normalize`).
- Readout: **CLS token alone**, `last_hidden_state[:, 0]`. `embed_dim = 1024`;
  classification dim **1024**. Segmentation features: `last_hidden_state[:, 1:]`
  (`num_prefix_tokens = 1`), 196 spatial tokens (224/16)^2.
- `clsmean` (PathoROB pooling) dim: 2048.
- Unusual: this is the only 16-patch backbone in the set (196 tokens, not 256), and the
  only one whose `adapter_config.json` already carries a correct
  `base_model_name_or_path`.

### 3.2 midnight -- `kaiko-ai/midnight`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-midnight-s0-t900-399166/step_0000150/adapter/` |
| s1 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-midnight-s1-t900-407566/step_0000100/adapter/` |
| s3 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-midnight-s3-t900-436609.r5/step_0000100/adapter/` |

- Base: `kaiko-ai/midnight` @ `adc6b15679c981cce6f9b018bbad09d16eeeda9f`.
  Local pin: `/admin/home/ryan.kim/pathfm-inputs/midnight/`.
- Loader: **transformers**. `Dinov2Model`, ViT-g/14 with SwiGLU FFN. Asserted:
  `hidden_size=1536`, `num_hidden_layers=40`, `num_attention_heads=24`, `patch_size=14`,
  `use_swiglu_ffn=true`, 727 state tensors.
- Target modules: `{query, key, value, dense, weights_in, weights_out}` -- **6 per block**;
  the FFN leaves are `mlp.weights_in` / `mlp.weights_out`, **not** `fc1`/`fc2`, because the
  FFN is SwiGLU. 240 modules / 480 tensors.
- Transform: `Resize(224, BILINEAR)` -> `CenterCrop(224)` -> `ToTensor` ->
  `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`. **Bilinear, not bicubic** --
  copied from THUNDER `get_midnight`, which calls `transforms.Resize(224)` with no
  `interpolation=` and therefore gets torchvision's bilinear default. The repo ships no
  `preprocessor_config.json`, so the (0.5,)*3 stats cannot be asserted against a local
  file; feeding ImageNet stats would neither crash nor warn.
- Readout: **CLS ++ mean(patch tokens)**, `cat([h[:,0], h[:,1:].mean(1)])`.
  `embed_dim = 1536`; classification dim **3072**. `num_prefix_tokens = 1`,
  256 spatial tokens.
- Largest adapter in the set (135 MiB), because 40 blocks x 6 targets.
- Seed labels are `s0, s1, s3` -- there is no s2.

### 3.3 virchow2 -- `paige-ai/Virchow2`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-virchow2-s0-t900-399167/step_0000100/adapter/` |
| s1 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-virchow2-s1-t900-407567/step_0000150/adapter/` |
| s3 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-virchow2-s3-t900-436610/step_0000100/adapter/` |

- Base: `paige-ai/Virchow2` @ `3158645804b69e3f3bc4439d4116edddf0840a72`.
  Local pin: `/admin/home/ryan.kim/pathfm-inputs/Virchow2/` (`model.safetensors`).
- Loader: **timm** (a timm checkpoint published on the hub; its `config.json` has
  `architecture`/`model_args`/`pretrained_cfg` and no `model_type`, so `AutoModel` has
  nothing to dispatch on). Architecture `vit_huge_patch14_224`; kwargs
  `img_size=224, init_values=1e-5, reg_tokens=4, mlp_ratio=5.3375, dynamic_img_size=True,
  mlp_layer=SwiGLUPacked, act_layer=SiLU`; 455 state tensors. `model_args` is checked for
  whole-dict equality against the repo config.
- Target modules: `{qkv, attn.proj, fc1, fc2}` -- **4 per block**, resolved as
  `blocks.<i>.attn.qkv`, `blocks.<i>.attn.proj`, `blocks.<i>.mlp.fc1`,
  `blocks.<i>.mlp.fc2`. 128 modules / 256 tensors.
- Transform: rebuilt exactly as THUNDER does,
  `create_transform(**resolve_data_config(config["pretrained_cfg"], model=model))`, over
  the **local** `config.json`. That cfg carries `interpolation=bicubic`, `crop_pct=1.0`,
  `crop_mode=center`, `input_size=(3,224,224)`, ImageNet mean/std -- i.e.
  `Resize(224, bicubic)` -> `CenterCrop(224)` -> `ToTensor` -> `Normalize(ImageNet)`.
- Readout: **CLS ++ mean(patch tokens)** where the patch tokens start at index 5
  (`num_prefix_tokens = 5` = 1 CLS + 4 registers). `embed_dim = 1280`; classification dim
  **2560**. 256 spatial tokens.
- Unusual: fp16 PGD under-attacks this backbone specifically; the PathoROB adversarial
  numbers for Virchow2 were produced by a separate fp32 control cell
  (`virchow2-basectrl-fp32adv`), not the ordinary base control.

### 3.4 hoptimus0 -- `bioptimus/H-optimus-0`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s0-t900-395391/step_0000100/adapter/` |
| s1 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s1-t900-395870/step_0000100/adapter/` |
| s3 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-hoptimus-s3-t900-436608.r5/step_0000100/adapter/` |

- Base: `bioptimus/H-optimus-0` @ `b145cc1e6c6b30d3251aa8b1f844e6974188a743`.
  Local pin: `/admin/home/ryan.kim/pathfm-inputs/H-optimus-0/` (`pytorch_model.bin`).
- Loader: **timm**. Architecture `vit_giant_patch14_reg4_dinov2`; kwargs
  `img_size=224, init_values=1e-5, dynamic_img_size=False`; 567 state tensors.
  `img_size=224` is mandatory -- the architecture default is 518, whose `pos_embed` is
  `(1, 1369, 1536)` against this checkpoint's `(1, 256, 1536)`.
- Target modules: `{qkv, attn.proj, fc1, fc2}` -- 4 per block, 40 blocks =
  160 modules / 320 tensors.
- Normalisation: **H&E-specific, not ImageNet** -- `mean=(0.707223, 0.578729, 0.703617)`,
  `std=(0.211883, 0.230117, 0.177517)`, read off the local `config.json`'s
  `pretrained_cfg`. timm's built-in cfg for this architecture is DINOv2's LVD-142M cfg,
  whose ImageNet-ish stats have the same shapes, raise no warning, and quietly cost
  accuracy on every downstream row.
- Transform: `create_transform(**resolve_data_config(config["pretrained_cfg"]))`. That
  `pretrained_cfg` carries **no** `crop_pct` and **no** `interpolation`, so timm's defaults
  apply: bicubic, `crop_pct=0.875` -> `Resize(256, bicubic)` -> `CenterCrop(224)` ->
  `ToTensor` -> `Normalize(H&E stats above)`. (Derived from the same cfg shape that the
  OpenMidnight cell documents explicitly as "bicubic Resize(256)+CenterCrop(224)".)
- Readout: **CLS token alone**. `embed_dim = 1536`; classification dim **1536**.
  `num_prefix_tokens = 5` (1 CLS + 4 registers); 256 spatial tokens.
- Seed labels are `s0, s1, s3` -- no s2.

### 3.5 uni2h -- `MahmoodLab/UNI2-h`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-uni2-s0-t900-395390/step_0000100/adapter/` |
| s1 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-uni2-s1-t900-395869/step_0000150/adapter/` |
| s2 | 100 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-lr1e-4-kl0-ms500-uni2-s2-t900-396381.r30/step_0000100/adapter/` |

- Base: `MahmoodLab/UNI2-h` @ `d517a8dd47902dd7c308b3c36f63bce47e7b9a43`.
  Local pin: `/admin/home/ryan.kim/pathfm-inputs/UNI2-h/` (`pytorch_model.bin`).
- Loader: **timm**. Architecture name in the config is `vit_giant_patch14_224`, but the
  card's depth/heads/SwiGLU/registers are **all explicit kwargs** and none of them follow
  from that name:
  `img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
  mlp_ratio=2.66667*2, no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=SiLU,
  reg_tokens=8, dynamic_img_size=True`. 343 state tensors.
- Target modules: `{qkv, attn.proj, fc1, fc2}` -- 4 per block, 24 blocks =
  96 modules / 192 tensors. Smallest adapter in the set.
- Transform: `create_transform(**resolve_data_config(config["pretrained_cfg"]))`. The cfg
  carries `crop_pct=1`, `interpolation=bilinear`, ImageNet mean/std -> `Resize(224, bilinear)`
  -> `CenterCrop(224)` -> `ToTensor` -> `Normalize(ImageNet)`. **Bilinear**, unlike the
  other timm rows.
- Readout: **CLS token alone**. `embed_dim = 1536`; classification dim **1536**.
  `num_prefix_tokens = 9` (1 CLS + 8 registers) -- the largest prefix in the set; getting
  this slice wrong silently mixes register tokens into segmentation and `clsmean`.
  256 spatial tokens.

### 3.6 virchow1 -- `paige-ai/Virchow` (v1)

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-virchow-s0-t900-438673/step_0000150/adapter/` |
| s1 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-virchow-s1-t900-438674/step_0000150/adapter/` |
| s2 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-virchow-s2-t900-438675/step_0000150/adapter/` |

- Base: `paige-ai/Virchow` (**v1**, a different repo id from Virchow2) @
  `19eebc84ae33e79f1b2d866e6ff90ae50e522f9a`.
  - Training loaded the hub snapshot
    `/data/Virchow/models--paige-ai--Virchow/snapshots/19eebc84ae33e79f1b2d866e6ff90ae50e522f9a/model.safetensors`
    (a cache root owned by another user).
  - **Evaluation loaded a different file**:
    `/data/Virchow_conv/pytorch_model.bin` + `/data/Virchow_conv/config.json`.
    See the hazard note in §4.
- Loader: **timm**. Architecture `vit_huge_patch14_224`; kwargs
  `img_size=224, init_values=1e-5, mlp_ratio=5.3375, mlp_layer=SwiGLUPacked,
  act_layer=SiLU, dynamic_img_size=True`; 454 state tensors (one fewer than Virchow2 --
  v1 has no `reg_token`). There is **no** `model_args` whole-dict check for this row
  because the local `config.json` has no `model_args`; the kwargs are transcribed from
  the model card.
- Target modules: `{qkv, attn.proj, fc1, fc2}` -- 4 per block, 32 blocks =
  128 modules / 256 tensors.
- Transform: `create_transform(**resolve_data_config(config["pretrained_cfg"]))` over
  `/data/Virchow_conv/config.json`, whose `pretrained_cfg` is
  `{custom_load: false, input_size: [3,224,224], fixed_input_size: true,
  interpolation: "bicubic", crop_pct: 1.0, mean: ImageNet, std: ImageNet}` ->
  `Resize(224, bicubic)` -> `CenterCrop(224)` -> `ToTensor` -> `Normalize(ImageNet)`.
- Readout: **CLS ++ mean(patch tokens)**. `embed_dim = 1280`; classification dim **2560**.
  `num_prefix_tokens = 1` -- **v1 has NO register tokens**, unlike Virchow2 which has 4.
  This is the single most confusable difference between the two Virchow rows: same
  architecture name, same embed dim, same adapter size, different prefix slice.
  256 spatial tokens.

### 3.7 openmidnightsq -- `SophontAI/OpenMidnight`

| Seed | Selected step | Adapter directory (absolute) |
|---|---|---|
| s0 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-openmidnight-s0-t900-438670/step_0000150/adapter/` |
| s1 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-openmidnight-s1-t900-438671/step_0000150/adapter/` |
| s2 | 150 | `/admin/home/ryan.kim/waiv/runs/genMASK-c50-ms500-openmidnight-s2-t900-438672/step_0000150/adapter/` |

- Base: **not a hub SHA.** `MODEL_REVISION` in all three cells is the literal string
  `local:scripts/convert_openmidnight.py from /data/OpenMidnight_ckpts/openmidnight_checkpoint.pth`.
  The base weights used for both training and evaluation are
  `/data/OpenMidnight/pytorch_model.bin` + `/data/OpenMidnight/config.json`, produced by
  `/admin/home/ryan.kim/waiv/scripts/convert_openmidnight.py` from a raw DINOv2 **training**
  checkpoint (`{"teacher": {...}}`, `block_chunks=4` nesting, DINOv2 SwiGLU names
  `mlp.w12`/`mlp.w3`). The conversion de-chunks the teacher backbone (asserting 4 chunks of
  10, global indices 0..39), runs timm's own `checkpoint_filter_fn` (`_convert_dinov2`), and
  requires a strict load with 0 missing / 0 unexpected keys before writing.
  **VERIFIED bit-identical to the `SophontAI/OpenMidnight` hub release** at revision
  `87189e6674d397a14a5cd342c97b1a1615a185aa`: the public `teacher_checkpoint_load.pt`
  (HF download metadata intact, blob sha256 `b57121fc...312505be`) was compared both
  pre-conversion (568/568 teacher keys `torch.equal`) and post-conversion
  (567/567 keys, `max_abs = 0`; 568->567 is `mask_token`, which timm's filter drops).
  The conversion script itself still performs no such comparison -- equivalence was
  established by an after-the-fact check, not by the pipeline. Cite the hub revision
  above rather than the local conversion string.
- Loader: **timm**. Architecture `vit_giant_patch14_reg4_dinov2`; kwargs
  `img_size=224, init_values=1e-5, dynamic_img_size=False`; 567 state tensors --
  architecturally the same animal as H-optimus-0.
- Target modules: `{qkv, attn.proj, fc1, fc2}` -- 4 per block, 40 blocks =
  160 modules / 320 tensors. Byte-identical adapter size to H-optimus-0.
- Normalisation: ImageNet `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)` --
  by **statement only**: the SophontAI model card says "# ImageNet normalization"; the
  training checkpoint stores no normalisation, and the `pretrained_cfg` was written by
  `convert_openmidnight.py`. Note this is *not* what `kaiko-ai/midnight` (the model
  OpenMidnight replicates) uses -- midnight demands (0.5,)*3.
- **Two different transforms, by benchmark.** This is the one genuinely unusual inference
  detail in the release:
  - Default (HEST, PathoROB, CPTAC): `create_transform(**resolve_data_config(pretrained_cfg))`.
    The cfg carries no `crop_pct` / `interpolation`, so timm's defaults give
    `Resize(256, bicubic)` -> `CenterCrop(224)`.
  - **THUNDER only:** `Resize((224, 224))` (squash, no crop) -> `ToTensor` ->
    `Normalize(ImageNet)`. THUNDER's `get_openmidnight`
    (`thunder/models/pretrained_models.py`) hand-builds the squash transform; using the
    generic resize+crop path put the base 6.2 F1 below the leaderboard on segmentation and
    ~1 point below on kNN / linear probing.
- Readout: **CLS token alone** -- `self.backbone(images)` under the checkpoint's
  `global_pool="token", num_classes=0` config. `embed_dim = 1536`; classification dim
  **1536**. `num_prefix_tokens = 5` (1 CLS + 4 registers); 256 spatial tokens.
- `ATTACK_BATCH = 32` (fp32 PGD OOMs at 128 on an 80 GB H100); `EXTRACTION_BATCH = 1024`.
- Unusual: these three cells use an **older, timm-only variant** of `model.py` (~19.7 KB,
  3-backbone table, `LORA_TARGET_SUFFIXES` as a module-level constant) rather than the
  ~33 KB 6-backbone file the other 18 cells share. The merge arithmetic and the assertions
  are equivalent; the file is not.

---

## 4. Packaging checklist -- known pre-upload fixes

### Must fix before upload

| # | Item | Affects | Action |
|---|---|---|---|
| 1 | `base_model_name_or_path: null` in `adapter_config.json` | **15 of 21** -- all timm-loader backbones: `virchow2` (3), `hoptimus0` (3), `uni2h` (3), `virchow1` (3), `openmidnightsq` (3) | Patch per repo to the correct hub id. `phikon2` (3) and `midnight` (3) already carry the right value (`owkin/phikon-v2`, `kaiko-ai/midnight`) -- do not touch those. |
| 2 | `README.md` in every adapter dir is the empty PEFT auto-stub | all 21 | Replace with a real model card. The timm-row stubs have an even emptier front-matter (`library_name: peft`, `tags: [lora]`, **no `base_model:` field**) than the two HF-row stubs, which do carry `base_model: <repo>`. |
| 3 | Seed label normalisation | `midnight` (s0,s1,**s3**), `virchow2` (s0,s1,**s3**), `hoptimus0` (s0,s1,**s3**) | Publish as `seed0/seed1/seed2`, i.e. map `s3 -> seed2`. The other four backbones are already s0/s1/s2. Record the original training seed value (`config.json:"seed"` = 3) in the model card so provenance is not lost. |
| 4 | `config.json` contains `-Infinity` | all 21 run configs (`same_core_logit_bias_mean`) | This is valid Python-`json` output but **invalid strict JSON** -- `JSON.parse` and most non-Python parsers reject it. Emit `null` (or the string `"-inf"`) in the published `training_config.json` and document the substitution. |
| 5 | Licence / gating status | all 7 backbones | **TODO -- under separate investigation.** No licence is asserted here. Do not copy the `license` field out of the local `config.json` files (see hazard H2). |

### Ship / do not ship

**DO ship** (per seed):

| File | Source |
|---|---|
| `adapter_model.safetensors` | `<RUN>/<STEP>/adapter/` |
| `adapter_config.json` | `<RUN>/<STEP>/adapter/` (after fix #1) |
| `training_config.json` | `<RUN>/config.json` (after fix #4) |
| `ri_curve.json` | `<RUN>/` -- the curve the 1-SE rule selected the step from; present for all 21 |

Optional, useful provenance also present in every run dir: `conditions_used.json`,
`metrics.json` (per checkpoint, 537 B).

**DO NOT ship** (training-only; explicitly not loaded at inference per each cell's
`model.py`):

| File | Location | Size |
|---|---|---|
| `optim.pt` | `<RUN>/<STEP>/` | ~139 MB each (138,693,775 B on phikon2) |
| `projector.pt`, `projector_cls.pt`, `projector_mean.pt` | `<RUN>/<STEP>/` | ~6.3 MB each |
| `projector_heads.json` | `<RUN>/<STEP>/` | 557 B |
| `pool_head.pt` | `<RUN>/<STEP>/` | 1,655 B (GeM head -- vacuous at inference) |
| `embedding_shift_cache.npz` | `<RUN>/` (3 runs only) | analysis cache |
| `retrieval_qualitative_cache.npz` | `<RUN>/` (1 run only) | analysis cache |
| `gpu.csv`, `eval_follow.log`, `TRAIN_DONE`, `probe_step_*.json`, `summary.json` | `<RUN>/` | internal |

### `history.json` -- training curves are NOT publishable for 8 of 21

`history.json` is **missing on 8 of the 21 runs**, all of which also lack `TRAIN_DONE` and
`summary.json` (i.e. they were stopped before the 500-step schedule finished). Training
curves can only be published for the other 13.

| Backbone | Seed | Run | `history.json` |
|---|---|---|---|
| phikon2 | s0 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165` | present |
| phikon2 | s1 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s1-t900-407565` | present |
| phikon2 | s2 | `genMASK-c50-lr1e-4-kl0-ms500-phikon-s2-t900-407568` | **MISSING** |
| midnight | s0 | `genMASK-c50-lr1e-4-kl0-ms500-midnight-s0-t900-399166` | present |
| midnight | s1 | `genMASK-c50-lr1e-4-kl0-ms500-midnight-s1-t900-407566` | **MISSING** |
| midnight | s3 | `genMASK-c50-ms500-midnight-s3-t900-436609.r5` | present |
| virchow2 | s0 | `genMASK-c50-lr1e-4-kl0-ms500-virchow2-s0-t900-399167` | present |
| virchow2 | s1 | `genMASK-c50-lr1e-4-kl0-ms500-virchow2-s1-t900-407567` | **MISSING** |
| virchow2 | s3 | `genMASK-c50-ms500-virchow2-s3-t900-436610` | **MISSING** |
| hoptimus0 | s0 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s0-t900-395391` | **MISSING** |
| hoptimus0 | s1 | `genMASK-c50-lr1e-4-kl0-ms500-hoptimus-s1-t900-395870` | **MISSING** |
| hoptimus0 | s3 | `genMASK-c50-ms500-hoptimus-s3-t900-436608.r5` | **MISSING** |
| uni2h | s0 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s0-t900-395390` | **MISSING** |
| uni2h | s1 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s1-t900-395869` | present |
| uni2h | s2 | `genMASK-c50-lr1e-4-kl0-ms500-uni2-s2-t900-396381.r30` | present |
| virchow1 | s0/s1/s2 | `...-virchow-s{0,1,2}-t900-4386{73,74,75}` | present (3/3) |
| openmidnightsq | s0/s1/s2 | `...-openmidnight-s{0,1,2}-t900-4386{70,71,72}` | present (3/3) |

Missing: 8 runs -- phikon2 s2, midnight s1, virchow2 s1, virchow2 s3, hoptimus0 s0,
hoptimus0 s1, hoptimus0 s3, uni2h s0. **Note: all three `hoptimus0` seeds lack
`history.json`,** so no training curve at all can be published for that backbone.

### OpenMidnight base-weight sign-off (RESOLVED -- no longer blocking)

`scripts/convert_openmidnight.py` converts
`/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth` -- a raw DINOv2 training checkpoint
obtained on this machine, **not** the `SophontAI/OpenMidnight` hub release -- into
`/data/OpenMidnight/{pytorch_model.bin,config.json}`. What the script *does* guarantee:

- the teacher backbone is de-chunked with the chunk structure asserted (4 chunks x 10
  blocks, global indices 0..39 exactly once);
- DINO/iBOT heads are dropped (training-only);
- the remap is timm's own `checkpoint_filter_fn`, not a hand-rolled key map;
- the load into `timm.create_model("vit_giant_patch14_reg4_dinov2", ...)` is strict, with
  `assert not missing and not unexpected` before anything is written;
- it refuses to overwrite an existing `/data/OpenMidnight/pytorch_model.bin`.

What it does **not** do, and what therefore remains open:

- it never downloads or opens the `SophontAI/OpenMidnight` hub artefact, and never compares
  tensors against it. Bit-identity was therefore established OUT OF BAND, not by the
  pipeline -- see the resolution note below.
- the `pretrained_cfg` it writes (normalisation, `input_size`, and a `license` string) is
  authored by the script, not carried over from any upstream file.

**RESOLVED.** The public artefact was already on this machine at
`/data/OpenMidnight_ckpts/hf_official/teacher_checkpoint_load.pt` (HF download metadata
intact: commit `87189e6674d397a14a5cd342c97b1a1615a185aa`, blob sha256
`b57121fc...312505be`). Two independent tensor-by-tensor comparisons both came back exact:
pre-conversion, the local training checkpoint's teacher backbone vs the public file
(568/568 keys `torch.equal`); post-conversion, the public file run through the same
`checkpoint_filter_fn` vs `/data/OpenMidnight/pytorch_model.bin` (567/567 keys,
`max_abs = 0`, the 568->567 delta being `mask_token`, which timm's filter drops).
Cite `SophontAI/OpenMidnight` @ `87189e66...` as the base revision. No owner sign-off is
required. The `license` string the script writes into `pretrained_cfg` is hand-authored but
matches the hub repo's actual `apache-2.0`.

---

## 5. Open hazards (not resolved by this manifest)

- **H1 -- Virchow v1 local conversion: RESOLVED, bit-identical.**
  `/data/Virchow_conv/pytorch_model.bin` was compared tensor-by-tensor against the pinned
  hub snapshot: 454/454 keys, no key renaming, identical shapes and dtypes, every tensor
  `torch.equal`, `max_abs_diff = 0.0`. The 95,219 B size delta is zip/pickle container
  overhead vs the safetensors header. The blob's sha256 matches its HF filename and
  `paige-ai/Virchow` main is `19eebc84...`, so the paper may cite
  "paige-ai/Virchow at revision 19eebc84". It is a repackaging, not a re-derivation.
  The repackaging STEP remains unreproduced (no script in the repo produces it), but its
  OUTPUT is verified. Original finding, retained for context:
  `encoder.py:BACKBONE_LOCAL_DIRS` binds `paige-ai/Virchow` to the pinned hub snapshot
  `/data/Virchow/models--paige-ai--Virchow/snapshots/19eebc84.../model.safetensors`
  (2,524,960,072 B), but all three evaluation cells load
  `/data/Virchow_conv/pytorch_model.bin` (2,525,055,291 B) instead. **No script in the waiv
  repo produces or references `/data/Virchow_conv`** (`rg "Virchow_conv"` over the repo
  returns nothing outside the cells), so the conversion remains undocumented -- but its
  output is now verified bit-identical to the pinned snapshot (see above), so this changes
  no number and blocks no release.
- **H2 -- `license` literals in the two locally written configs: CHECKED, both correct.**
  Virchow v1's is NOT hand-authored -- the pinned hub `config.json` carries the identical
  string, and the HF card says `apache-2.0`. OpenMidnight's IS written by the conversion
  script, but matches the hub repo's actual `apache-2.0`. The CC-BY-NC-ND comparison below
  is to Virchow2 and UNI2-h, which are different repos with different licences. Original
  finding, retained for context:
  `/data/Virchow_conv/config.json` and `/data/OpenMidnight/config.json` both carry
  `"license": "Apache 2.0"` inside `pretrained_cfg`. For OpenMidnight this string is
  hand-written by `convert_openmidnight.py`. For comparison, the hub-derived
  `/admin/home/ryan.kim/pathfm-inputs/Virchow2/config.json` says `CC-BY-NC-ND-4.0` and
  `/admin/home/ryan.kim/pathfm-inputs/UNI2-h/config.json` says `CC BY-NC-ND 4.0`. Do not
  propagate these strings into any release artefact; licence is TODO.
- **H3 -- three released runs are SLURM requeues.**
  `...-midnight-s3-t900-436609.r5`, `...-hoptimus-s3-t900-436608.r5` and
  `...-uni2-s2-t900-396381.r30` carry requeue suffixes (`.r5`, `.r30`). Confirm with
  `sacct --duplicates` that the released checkpoint sits on a single uninterrupted
  trajectory before publishing those three.
- **H4 -- `/data` is volatile.** Four of the seven backbones' base weights live under
  `/data` (`/data/Virchow_conv`, `/data/OpenMidnight`, `/data/Virchow/...`) or are
  hardlinked pins under `/admin/home/ryan.kim/pathfm-inputs/`. Snapshot anything the
  release depends on to `/admin` before upload.
