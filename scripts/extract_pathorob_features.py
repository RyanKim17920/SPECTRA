#!/usr/bin/env python
"""Extract PathoROB features with *our* encoder, in *their* npz layout (PLAN.md §4 phase 5).

This is the extraction half of the Phase-2 gate. The metric half is PathoROB's own
``robustness_index`` module, which we never reimplement (see
``waivphaet.eval.pathorob_adapter``).

Why this script exists rather than ``python -m pathorob.features.extract_features``
-----------------------------------------------------------------------------------
Their extractor hardcodes a registry of *their* two models. Ours has to run an arbitrary
``PhikonEncoder`` checkpoint (base today, LoRA-fine-tuned later) through the identical
pipeline, so that the only thing that ever changes between the baseline row and a
fine-tuned row is the weights. Everything else here is mirrored from
``pathorob/models/phikon.py`` on purpose:

* preprocessing = ``Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize(IMAGENET)``
  applied to the **PIL** image. The PathoROB tiles are 256x256, so the resize is not a
  no-op -- dropping it silently changes the field of view and the baseline drifts.
* pooling = ``clsmean``: ``cat([cls, mean(patch_tokens)])`` -> **2048-d**, not the 1024-d
  cls-only vector. Their reference row is literally named ``phikonv2_clsmean``.
* features are written by *their* ``FeatureDataManager.save_features``, so the on-disk
  layout ``{features_dir}/{model}/{dataset}/{medical_center}.npz`` is theirs by
  construction rather than by our reading of their docs.

Images come from the ungated HF parquet (``bifold-pathomics/PathoROB-{dataset}``). We read
the parquet directly with pyarrow instead of ``datasets.load_dataset`` -- the row set is
identical (the metric looks rows up by ``f"{slide_id}-{patch_id}"``, never by position) and
it keeps ``datasets`` out of our venv.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/huggingface")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
PATHOROB_ROOT = REPO / "third_party" / "PathoROB"

DATASETS = ("camelyon", "tcga", "tolkach_esca")
#: Which metadata CSV each HF image dataset is scored against. ``tcga`` ships two
#: griddings; ``tcga_4x4`` is the one PathoROB's committed reference row uses.
META_COLUMNS = ("slide_id", "patch_id", "biological_class", "medical_center")


# --------------------------------------------------------------------------------------
# data


def _parquet_files(dataset: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        f"bifold-pathomics/PathoROB-{dataset}",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    files = sorted((Path(snap) / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {snap}/data")
    return files


class PathoRobParquet(Dataset):
    """Decode-on-access view over the HF parquet shards.

    The whole table is held in memory as arrow (JPEG bytes, ~0.5 GB for camelyon); only
    the decode + transform happens per item, which is what we want the dataloader workers
    for.
    """

    def __init__(self, dataset: str, transform):
        import pyarrow.parquet as pq

        import pyarrow as pa

        tables = [pq.read_table(f) for f in _parquet_files(dataset)]
        self.table = pa.concat_tables(tables)
        self.transform = transform
        self.images = self.table.column("image")
        self.meta = {c: self.table.column(c).to_pylist() for c in META_COLUMNS}

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, i: int):
        raw = self.images[i]["bytes"].as_py()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img), i


def build_preprocess(backbone: str | None = None):
    """Byte-for-byte mirror of ``Phikonv2ModelWrapper.get_preprocess`` -- except that the
    normalisation follows the *backbone*.

    Resize/crop are shared (both kaiko-ai/midnight's card and PathoROB's phikon wrapper
    specify Resize(224) -> CenterCrop(224)), but the stats are not: phikon-v2 is ImageNet,
    midnight is (0.5,0.5,0.5). See ``BACKBONE_NORMALIZATION``.
    """
    import torchvision.transforms as T

    from waivphaet.models.encoder import normalization_for

    mean, std = normalization_for(backbone)
    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


# --------------------------------------------------------------------------------------
# model


def assert_adapter_applied(model, batch: torch.Tensor | None = None, tol: float = 1e-4) -> dict:
    """Prove the LoRA adapter actually changes embeddings.

    A silently-unloaded adapter reproduces the base model exactly, which on a retention
    benchmark reads as "perfect retention" -- the most dangerous possible false result.
    Every eval entry point that takes an adapter MUST call this.

    ``batch`` defaults to a deterministic synthetic batch so this works with no dataset
    on hand (HEST / THUNDER load their data inside third-party harnesses). Any input
    distinguishes an applied adapter from an absent one.
    """
    if batch is None:
        g = torch.Generator().manual_seed(1234)
        batch = torch.randn(4, 3, 224, 224, generator=g)
    device = next(model.parameters()).device
    batch = batch.to(device)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        emb_lora = model.embed(batch)
        with model.backbone.disable_adapter():
            emb_base = model.embed(batch)
    if was_training:
        model.train()
    rel = float((emb_lora - emb_base).norm() / emb_base.norm())
    mean_cos = float(
        torch.nn.functional.cosine_similarity(emb_lora.float(), emb_base.float(), dim=-1).mean()
    )
    print(f"[adapter-check] rel_l2_delta={rel:.6e}  mean_cosine_base_vs_lora={mean_cos:.6f}",
          flush=True)
    if rel < tol:
        raise SystemExit(
            f"adapter did not change embeddings (rel_l2_delta={rel:.3e} < {tol}); "
            "adapter likely not applied"
        )
    return {"rel_l2_delta": rel, "mean_cosine_base_vs_lora": mean_cos}


def assert_checkpoint_applied(model, checkpoint_path: str | Path,
                               batch: torch.Tensor | None = None,
                               tol: float = 1e-4) -> dict:
    """Prove a full-FT checkpoint actually changed the backbone weights.

    Replaces ``assert_adapter_applied`` for full-FT checkpoints (no PEFT adapter,
    no ``disable_adapter()``). Loads the base backbone (same ``backbone`` id), runs
    the same input, and asserts the checkpoint embeddings differ from the base.

    A silently-unloaded full-FT checkpoint reproduces the base numbers exactly and reads
    as "perfect retention" -- the most dangerous false result in this project. Every eval
    entry point that loads a full-FT checkpoint MUST call this.

    ``batch`` defaults to a deterministic synthetic batch. Returns the same shape dict
    as ``assert_adapter_applied`` for callers that consume both.
    """
    if batch is None:
        g = torch.Generator().manual_seed(1234)
        batch = torch.randn(4, 3, 224, 224, generator=g)

    device = next(model.parameters()).device
    batch_dev = batch.to(device)
    backbone_id = model.cfg.backbone

    # Build a fresh base model (no checkpoint) for comparison.
    from waivphaet.models.encoder import EncoderConfig, WaivEncoder
    base_cfg = EncoderConfig(backbone=backbone_id, pooling=model.cfg.pooling, use_lora=False)
    base_model = WaivEncoder(base_cfg).to(device).eval()

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        emb_ckpt = model.embed(batch_dev)
        emb_base = base_model.embed(batch_dev)
    if was_training:
        model.train()

    rel = float((emb_ckpt - emb_base).norm() / emb_base.norm())
    mean_cos = float(
        torch.nn.functional.cosine_similarity(emb_ckpt.float(), emb_base.float(), dim=-1).mean()
    )
    print(f"[checkpoint-check] rel_l2_delta={rel:.6e}  mean_cosine_base_vs_ckpt={mean_cos:.6f}",
          flush=True)
    if rel < tol:
        raise SystemExit(
            f"checkpoint at {checkpoint_path} did not change embeddings "
            f"(rel_l2_delta={rel:.3e} < {tol}); checkpoint likely not loaded or "
            "contains unmodified base weights"
        )
    return {"rel_l2_delta": rel, "mean_cosine_base_vs_lora": mean_cos}


def _restore_pool_head(model, ckpt_dir: Path) -> dict:
    """Load a checkpoint's trained ``pool_head.pt`` into an ``infer_pool_head`` encoder.

    Only called when the caller explicitly asked for inference-time pooling. The head is
    tiny (GeM is a single scalar ``p``), but it is the ENTIRE intervention: without this
    the encoder would pool through a freshly-initialised head, i.e. score p=3.0 instead of
    the trained value and quietly report it as "the trained pooling".
    """
    pool = getattr(model, "pool_head", None)
    if pool is None:
        raise SystemExit("infer_pool_head requested but the encoder built no pool head")
    path = Path(ckpt_dir) / "pool_head.pt"
    if not path.exists():
        raise SystemExit(
            f"infer_pool_head requested but {path} does not exist. Only --pool-head "
            "gem/attn/lse runs write one; the mean arm has no learned pooling."
        )
    sd = torch.load(path, map_location="cpu")
    pool.load_state_dict(sd)
    restored = {k: v.flatten()[:1].item() if v.numel() else float("nan")
                for k, v in sd.items() if v.ndim == 0 or v.numel() <= 4}
    print(f"[build_model] restored pool_head from {path}: {restored}", flush=True)
    return restored


def build_model(checkpoint: str | None, pooling: str, adapter: Path | None = None,
                lora_rank: int = 16, lora_alpha: int = 32, proj_out_dim: int = 512,
                backbone: str | None = None, pool_head: str | None = None,
                infer_pool_head: bool = False):
    """Single loader shared by PathoROB, HEST and THUNDER (see ``hest_adapter``).

    ``backbone`` defaults to ``DEFAULT_BACKBONE`` (owkin/phikon-v2) so every existing
    call site keeps its exact behaviour. When an adapter directory is given, the backbone
    is taken from the adapter's own ``base_model_name_or_path`` unless explicitly
    overridden -- evaluating a midnight adapter on top of phikon-v2 would load, produce
    numbers, and be silently meaningless.
    """
    from waivphaet.models.encoder import DEFAULT_BACKBONE, EncoderConfig, WaivEncoder

    backbone = backbone or os.environ.get("WAIV_BACKBONE") or None

    if adapter is not None:
        # The saved adapter_config.json is the source of truth for rank/alpha. Passing the
        # wrong rank is an easy CLI mistake and would either blow up deep inside peft or,
        # worse, load a differently-scaled adapter -- check up front and say so plainly.
        acfg_path = adapter / "adapter" / "adapter_config.json"
        if acfg_path.exists():
            acfg = json.loads(acfg_path.read_text())
            saved_r, saved_alpha = int(acfg["r"]), int(acfg["lora_alpha"])
            if (saved_r, saved_alpha) != (lora_rank, lora_alpha):
                raise SystemExit(
                    f"adapter at {adapter} was saved with r={saved_r} alpha={saved_alpha} "
                    f"but was asked to load with r={lora_rank} alpha={lora_alpha}; "
                    "pass --lora-rank/--lora-alpha (or WAIV_LORA_RANK/WAIV_LORA_ALPHA) to match"
                )
            saved_base = acfg.get("base_model_name_or_path")
            if saved_base:
                if backbone is None:
                    backbone = saved_base
                elif backbone != saved_base:
                    raise SystemExit(
                        f"adapter at {adapter} was trained on {saved_base!r} but the "
                        f"backbone was set to {backbone!r}; a cross-backbone load either "
                        "crashes in peft or silently scores the wrong model"
                    )
        # Load a PEFT adapter directory (save_checkpoint output).
        cfg = EncoderConfig(backbone=backbone or DEFAULT_BACKBONE,
                            pooling=pooling, use_lora=True, lora_rank=lora_rank,
                            lora_alpha=lora_alpha, proj_out_dim=proj_out_dim,
                            pool_head=pool_head or "mean",
                            infer_pool_head=infer_pool_head)
        model = WaivEncoder(cfg)
        if infer_pool_head:
            _restore_pool_head(model, adapter)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(str(adapter / "adapter" / "adapter_model.safetensors"))
        out = set_peft_model_state_dict(model.backbone, state)
        if getattr(out, "unexpected_keys", None):
            raise RuntimeError(f"adapter keys not consumed: {list(out.unexpected_keys)[:5]}")
        # The projector is training-only: InfoNCE is applied to its output, but every eval
        # path here reads model.embed() and never touches it. Its input width is tied to the
        # *training* pooling (2048 for clsmean, 1024 for cls), so loading it under a different
        # eval pooling is both impossible and pointless. Load it when the widths agree (keeps
        # the artifact faithful); skip loudly when they don't, rather than crashing a run whose
        # features don't depend on it.
        proj_sd = torch.load(adapter / "projector.pt", map_location="cpu")
        saved_in = proj_sd["net.0.weight"].shape[1]
        if saved_in == model.embed_dim:
            model.projector.load_state_dict(proj_sd)
        else:
            print(
                f"[build_model] skipping projector: trained with a {saved_in}-d input, "
                f"evaluating at {model.embed_dim}-d (pooling={pooling}). "
                "Projector is unused for feature extraction; LoRA backbone weights are unaffected.",
                flush=True,
            )
        # Cheap CPU-side proof before the caller ever sees the model. Callers that have real
        # tiles on hand (this script's main()) re-run it on those for a sharper number.
        assert_adapter_applied(model.eval())
    elif checkpoint is None:
        # Base backbone: no LoRA, no adapter deltas -- this is the Phase-2 gate model.
        cfg = EncoderConfig(backbone=backbone or DEFAULT_BACKBONE,
                            pooling=pooling, use_lora=False)
        model = WaivEncoder(cfg)
    else:
        # --- Checkpoint loading: handle both legacy torch.load format and full-FT safetensors ---
        ckpt_path = Path(checkpoint)

        # Try to load encoder_config from the checkpoint or from the run's config.json.
        # Full-FT checkpoints may not carry encoder_config inside the weight file.
        cfg_from_ckpt = None

        # Check for the run's top-level config.json. For a checkpoint DIR
        # (step_NNNNNNN/) the run dir is ckpt_path.parent; for a checkpoint FILE it is
        # ckpt_path.parent.parent. Try both -- getting this wrong used to send a
        # perfectly good full-FT dir into the torch.load branch below, where
        # torch.load(<a directory>) raises IsADirectoryError and the bare `except`
        # reported it as "not a recognised format" (job 369922's follower died here on
        # its first checkpoint).
        for run_config in (ckpt_path / "config.json",
                           ckpt_path.parent / "config.json",
                           ckpt_path.parent.parent / "config.json"):
            if run_config.exists():
                run_cfg = json.loads(run_config.read_text())
                enc_cfg = run_cfg.get("encoder", {})
                if isinstance(enc_cfg, dict) and enc_cfg:
                    cfg_from_ckpt = EncoderConfig(**enc_cfg)
                    cfg_from_ckpt.pooling = pooling
                    break

        # Try to load as a single torch.load archive containing encoder_config + model.
        # Only meaningful for a checkpoint FILE; a directory is handled by the
        # safetensors path below.
        if cfg_from_ckpt is None and ckpt_path.is_file():
            try:
                raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    saved_cfg = raw.get("encoder_config")
                    if isinstance(saved_cfg, dict):
                        cfg_from_ckpt = EncoderConfig(**saved_cfg)
                    if "model" in raw:
                        model_state = raw["model"]
                        cfg_from_ckpt = cfg_from_ckpt or EncoderConfig()
                        cfg_from_ckpt.pooling = pooling
                        backbone_id = cfg_from_ckpt.backbone
                        if backbone is not None and backbone != backbone_id:
                            raise SystemExit(
                                f"checkpoint {checkpoint} carries backbone {backbone_id!r}; "
                                f"refusing the requested override {backbone!r}"
                            )
                        model = WaivEncoder(cfg_from_ckpt)
                        model.load_state_dict(model_state, strict=False)
                    else:
                        # raw is a plain state_dict (legacy full-FT format).
                        cfg_from_ckpt = cfg_from_ckpt or EncoderConfig(
                            backbone=backbone or DEFAULT_BACKBONE,
                            pooling=pooling, use_lora=False,
                        )
                        model = WaivEncoder(cfg_from_ckpt)
                        model.load_state_dict(raw, strict=False)
            except Exception as e:
                raise SystemExit(
                    f"checkpoint {checkpoint} is not a recognised format ({type(e).__name__}: {e}). "
                    "Expected a step_NNNNNNN/ dir containing backbone.safetensors "
                    "(full FT) or an adapter/ dir (LoRA)."
                )

        if cfg_from_ckpt is None:
            # No config found — treat as full-FT safetensors checkpoint dir.
            backbone_id = backbone or DEFAULT_BACKBONE
            cfg_from_ckpt = EncoderConfig(backbone=backbone_id, pooling=pooling, use_lora=False)

        model = WaivEncoder(cfg_from_ckpt)

        # Load full-FT backbone weights from safetensors if present.
        backbone_safetensors = ckpt_path / "backbone.safetensors"
        if backbone_safetensors.exists():
            from safetensors.torch import load_file
            backbone_sd = load_file(str(backbone_safetensors))
            model.backbone.load_state_dict(backbone_sd)
        else:
            # Try backbone.pt fallback (legacy format).
            backbone_pt = ckpt_path / "backbone.pt"
            if backbone_pt.exists():
                backbone_sd = torch.load(str(backbone_pt), map_location="cpu", weights_only=False)
                model.backbone.load_state_dict(backbone_sd)

        # Load projector when the input width matches.
        proj_path = ckpt_path / "projector.pt"
        if proj_path.exists():
            try:
                proj_sd = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                saved_in = proj_sd["net.0.weight"].shape[1]
                if saved_in == model.embed_dim:
                    model.projector.load_state_dict(proj_sd)
                else:
                    print(
                        f"[build_model] skipping projector: trained with a {saved_in}-d input, "
                        f"evaluating at {model.embed_dim}-d (pooling={pooling}). "
                        "Projector is unused for feature extraction; backbone weights are unaffected.",
                        flush=True,
                    )
            except Exception as e:
                print(f"[build_model] WARNING: could not load projector: {e}", flush=True)

        # Full-FT checkpoint guard: prove the loaded weights differ from the base model.
        model.eval()
        assert_checkpoint_applied(model, checkpoint)

    return model.eval()


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="camelyon", choices=DATASETS)
    ap.add_argument(
        "--model-name",
        required=True,
        help="free-form features dir name, e.g. phikonv2_clsmean_ours. The metric scripts "
        "never call load_model(), so this needs no registry entry.",
    )
    ap.add_argument("--checkpoint", default=None,
                    help="Full-FT checkpoint dir (step_NNNNNNN/ with backbone.safetensors + "
                         "projector.pt) or legacy torch.load archive. Mutually exclusive "
                         "with --adapter. Omit for the base backbone.")
    ap.add_argument("--backbone", default=None,
                    help="HF id of the base backbone (default: owkin/phikon-v2, or "
                         "WAIV_BACKBONE, or whatever --adapter/--checkpoint was trained "
                         "on). e.g. kaiko-ai/midnight")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA checkpoint dir written by save_checkpoint (contains adapter/ + "
                         "projector.pt); mutually exclusive with --checkpoint")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--pooling", default="clsmean", choices=("cls", "mean", "clsmean"))
    ap.add_argument("--features-dir", default=str(PATHOROB_ROOT / "data" / "features"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="none", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    args = ap.parse_args()

    if args.checkpoint and args.adapter:
        raise SystemExit("--checkpoint and --adapter are mutually exclusive")

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(PATHOROB_ROOT))
    from pathorob.features.data_manager import FeatureDataManager

    import pandas as pd

    t0 = time.time()
    print(f"[extract] dataset={args.dataset} device={args.device} HF_HOME={os.environ['HF_HOME']}")
    # Model FIRST: the preprocessing normalisation is a property of the backbone, and the
    # backbone may come from the adapter's own config rather than the CLI.
    model = build_model(
        args.checkpoint, args.pooling, args.adapter, args.lora_rank,
        args.lora_alpha, args.proj_out_dim, args.backbone,
    ).to(args.device)
    print(f"[extract] normalization mean={model.norm_mean} std={model.norm_std}")
    ds = PathoRobParquet(args.dataset, build_preprocess(model.cfg.backbone))
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[extract] {len(ds)} rows loaded in {time.time() - t0:.1f}s (using {n})")
    print(f"[extract] backbone={model.cfg.backbone} hidden={model.hidden_size} "
          f"embed_dim={model.embed_dim} pooling={args.pooling}")
    # Derived, not literal: 2048 on phikon-v2 (1024x2), 3072 on midnight (1536x2). The
    # check still earns its keep -- it catches a pooling/embed_dim desync, which is what
    # would silently write half-width features into PathoROB's npz layout.
    expected = model.hidden_size * (2 if args.pooling == "clsmean" else 1)
    if model.embed_dim != expected:
        raise RuntimeError(
            f"pooling={args.pooling} on hidden={model.hidden_size} must give {expected}-d, "
            f"got {model.embed_dim}"
        )

    # Adapter-applied check on REAL tiles (build_model already ran it on synthetic input).
    if args.adapter is not None:
        check_sz = min(8, len(ds))
        assert_adapter_applied(model, torch.stack([ds[i][0] for i in range(check_sz)]))

    indices = list(range(n))
    loader = DataLoader(
        torch.utils.data.Subset(ds, indices) if args.limit else ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    amp_dtype = {"none": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.amp]
    feats = np.empty((n, model.embed_dim), dtype=np.float32)
    t1 = time.time()
    done = 0
    with torch.inference_mode():
        for images, idx in loader:
            images = images.to(args.device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=args.device.split(":")[0], dtype=amp_dtype):
                    out = model.embed(images)
            else:
                out = model.embed(images)
            feats[idx.numpy()] = out.float().cpu().numpy()
            done += len(idx)
            if done % (args.batch_size * 20) == 0:
                rate = done / (time.time() - t1)
                print(f"[extract] {done}/{n}  {rate:.0f} img/s  eta {(n - done) / rate:.0f}s",
                      flush=True)
    dt = time.time() - t1
    print(f"[extract] {n} embeddings in {dt:.1f}s ({n / dt:.0f} img/s)")

    # Sanity: a zero row would make PathoROB raise, but a *near*-zero row would not.
    norms = np.linalg.norm(feats, axis=1)
    if not np.isfinite(feats).all():
        raise RuntimeError("non-finite features")
    print(f"[extract] |f| min={norms.min():.4f} mean={norms.mean():.4f} max={norms.max():.4f}")

    metadata = pd.DataFrame({c: ds.meta[c][:n] for c in META_COLUMNS})
    dm = FeatureDataManager(
        features_dir=args.features_dir, metadata_dir=str(PATHOROB_ROOT / "data" / "metadata")
    )
    out_dir = Path(args.features_dir) / args.model_name / args.dataset
    if out_dir.exists():
        # save_features *merges* into existing npz; a stale 1024-d run would survive.
        raise SystemExit(f"{out_dir} already exists -- remove it before re-extracting")
    dm.save_features(args.model_name, args.dataset, feats, metadata)
    written = sorted(p.name for p in out_dir.glob("*.npz"))
    print(f"[extract] wrote {len(written)} npz -> {out_dir}: {written}")
    print(json.dumps({
        "dataset": args.dataset, "model": args.model_name, "rows": int(n),
        "dim": int(model.embed_dim), "seconds": round(dt, 1),
        "centers": written,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
