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

os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf_home")

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


def build_preprocess():
    """Byte-for-byte mirror of ``Phikonv2ModelWrapper.get_preprocess``."""
    import torchvision.transforms as T

    from waivphaet.models.encoder import IMAGENET_MEAN, IMAGENET_STD

    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------------------
# model


def build_model(checkpoint: str | None, pooling: str, adapter: Path | None = None,
                lora_rank: int = 16, lora_alpha: int = 32, proj_out_dim: int = 512):
    from waivphaet.models.encoder import EncoderConfig, PhikonEncoder

    if adapter is not None:
        # Load a PEFT adapter directory (save_checkpoint output).
        cfg = EncoderConfig(pooling=pooling, use_lora=True, lora_rank=lora_rank,
                            lora_alpha=lora_alpha, proj_out_dim=proj_out_dim)
        model = PhikonEncoder(cfg)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(str(adapter / "adapter" / "adapter_model.safetensors"))
        out = set_peft_model_state_dict(model.backbone, state)
        if getattr(out, "unexpected_keys", None):
            raise RuntimeError(f"adapter keys not consumed: {list(out.unexpected_keys)[:5]}")
        model.projector.load_state_dict(torch.load(adapter / "projector.pt", map_location="cpu"))
    elif checkpoint is None:
        # Base phikon-v2: no LoRA, no adapter deltas -- this is the Phase-2 gate model.
        cfg = EncoderConfig(pooling=pooling, use_lora=False)
        model = PhikonEncoder(cfg)
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("encoder_config")
        cfg = EncoderConfig(**cfg) if isinstance(cfg, dict) else cfg
        cfg.pooling = pooling
        model = PhikonEncoder(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
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
    ap.add_argument("--checkpoint", default=None, help="omit for base phikon-v2")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="checkpoint dir written by save_checkpoint (contains adapter/ + projector.pt); mutually exclusive with --checkpoint")
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
    ds = PathoRobParquet(args.dataset, build_preprocess())
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[extract] {len(ds)} rows loaded in {time.time() - t0:.1f}s (using {n})")

    model = build_model(args.checkpoint, args.pooling, args.adapter, args.lora_rank,
                      args.lora_alpha, args.proj_out_dim).to(args.device)
    print(f"[extract] embed_dim={model.embed_dim} pooling={args.pooling}")
    if model.embed_dim != 2048 and args.pooling == "clsmean":
        raise RuntimeError(f"clsmean must be 2048-d, got {model.embed_dim}")

    # Adapter-applied check: verify the LoRA adapter actually changed embeddings.
    if args.adapter is not None:
        check_sz = min(8, len(ds))
        check_batch = torch.stack([ds[i][0] for i in range(check_sz)]).to(args.device)
        with torch.inference_mode():
            emb_lora = model.embed(check_batch)
            with model.backbone.disable_adapter():
                emb_base = model.embed(check_batch)
        rel = float((emb_lora - emb_base).norm() / emb_base.norm())
        cos = torch.nn.functional.normalize(emb_lora.float(), dim=-1) @ torch.nn.functional.normalize(emb_base.float(), dim=-1)
        mean_cos = float(cos.diagonal().mean())
        print(f"[adapter-check] rel_l2_delta={rel:.6e}  mean_cosine_base_vs_lora={mean_cos:.6f}")
        if rel < 1e-4:
            raise SystemExit("adapter did not change embeddings (rel_l2_delta < 1e-4); adapter likely not applied")

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
