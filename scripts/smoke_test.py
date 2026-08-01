#!/usr/bin/env python
"""End-to-end smoke test on whatever PLISM slides are local.

Runs a handful of real training steps through the real sampler, the real encoder and the
real loss, on CPU or 1 GPU. Its job is to catch wiring bugs, NOT to learn anything --
with only 2 of 91 conditions present there is exactly one positive direction available,
which is the degenerate case PLAN.md 2 warns about.

    python scripts/smoke_test.py --steps 4 --device cpu

Assumes the local ``.h5`` have been repacked::

    python -m waivphaet.data.repack --h5-dir /data/ryan.kim/plism \
        --out-dir /data/ryan.kim/plism/repacked --verify
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from waivphaet.data.conditions import available_conditions, all_conditions, default_split
from waivphaet.data.pairs import build_pair_loader
from waivphaet.models.encoder import build_encoder
from waivphaet.train.contrastive import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5-dir", type=Path, default=Path("/data/ryan.kim/plism"))
    ap.add_argument("--packed-dir", type=Path, default=Path("/data/ryan.kim/plism/repacked"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-groups", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf_home")  # never fill /admin

    present = [p.name for p in Path(args.h5_dir).glob("*.tif.h5")]
    conds = available_conditions(all_conditions(), present)
    print(f"[smoke] {len(conds)} local condition(s): {[c.key for c in conds]}")
    if len(conds) < 2:
        raise SystemExit(
            f"need >=2 repacked conditions for a positive pair; found {len(conds)} in {args.h5_dir}"
        )
    split = default_split()
    print(f"[smoke] full-run split would be: {split.summary()}")
    print("[smoke] NOTE: with 2 conditions there is 1 positive direction -- wiring check only.")

    loader = build_pair_loader(
        args.packed_dir,
        conditions=conds,
        n_groups=args.n_groups,
        group_size=args.group_size,
        batches_per_epoch=args.steps,
        num_workers=args.workers,
    )

    model = build_encoder(lora_rank=args.lora_rank)
    print("[smoke] params:", model.trainable_parameter_summary())

    cfg = TrainConfig(
        packed_dir=str(args.packed_dir),
        out_dir=str(args.out_dir),
        max_steps=args.steps,
        warmup_steps=1,
        n_groups=args.n_groups,
        group_size=args.group_size,
        log_every=1,
        ckpt_every=args.steps,
        eval_heldout=False,
        amp_dtype="bfloat16" if args.device.startswith("cuda") else "none",
        num_workers=args.workers,
    )
    res = train(model, loader, cfg, device=args.device)
    print("[smoke] history:", res["history"])
    print(f"[smoke] OK -- checkpoint under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
