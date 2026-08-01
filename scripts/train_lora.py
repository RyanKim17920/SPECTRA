#!/usr/bin/env python
"""Full contrastive fine-tune entrypoint (PLAN.md 3, phase 7).

LoRA-all-blocks + masked InfoNCE over PLISM registered pairs, holding out 2 scanners and
3 stains. Retention/robustness evaluation hangs off ``on_checkpoint`` -- PLAN.md 3 phase 8
requires it at *every* checkpoint, and PLAN.md 6 requires it reported alongside every
robustness claim.

    python scripts/train_lora.py --out-dir runs/lora_r16_t007 --lora-rank 16 --temperature 0.07
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from waivphaet.data.conditions import all_conditions, available_conditions, make_split
from waivphaet.data.pairs import build_pair_loader
from waivphaet.data.repack import present_filenames
from waivphaet.models.encoder import build_encoder
from waivphaet.train.contrastive import TrainConfig, train


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed-dir", type=Path, default=Path("/data/ryan.kim/plism/repacked"))
    ap.add_argument("--out-dir", type=Path, required=True)
    # split (PLAN.md 3 phase 7): 2 of 7 scanners, 3 of 13 stains
    ap.add_argument("--heldout-scanners", nargs="*", default=["GT450", "S210"])
    ap.add_argument("--heldout-stains", nargs="*", default=["HRH", "KR", "MY"])
    # the unknown hyperparameters (PLAN.md 3 risk 4) -- these are what phase 8 sweeps
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    # batching: negatives per anchor == group_size - 1, so prefer few large groups
    ap.add_argument("--n-groups", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", default="bfloat16", choices=["bfloat16", "float16", "none"])
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--symmetric", action="store_true",
                    help="ABLATION ONLY: adds the anchor->positive direction, whose candidate "
                         "row spans conditions and reintroduces the acquisition shortcut")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute block activations in backward. The negative count is "
                         "group_size-1, so more negatives means a bigger forward batch; "
                         "without this, 80 GiB caps out around 340 images/step.")
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--pooling", default="clsmean", choices=["cls", "mean", "clsmean"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf_home")
    torch.manual_seed(args.seed)

    split = make_split(args.heldout_scanners, args.heldout_stains)
    # verified-complete slides only: the acquisition job may still be streaming in
    present = present_filenames(args.packed_dir)
    train_conds = available_conditions(split.train, present)
    heldout_conds = available_conditions(split.heldout, present)
    print(f"[train] {split.summary()}")
    print(f"[train] repacked & usable: {len(train_conds)} train / {len(heldout_conds)} heldout")
    print(f"[train] train conditions:   {sorted(c.key for c in train_conds)}")
    print(f"[train] heldout conditions: {sorted(c.key for c in heldout_conds)}")
    if len(train_conds) < 2:
        raise SystemExit("need >=2 repacked training conditions; run `waiv-repack` first")

    # PLAN.md 3 risk 3: held-out *conditions* are the only in-training check against
    # tile-identity memorisation, so prove the exclusion instead of trusting the split.
    assert set(train_conds).isdisjoint(heldout_conds), "train/heldout condition sets overlap"
    leaked = [c.key for c in train_conds
              if c.scanner in set(args.heldout_scanners) or c.stain in set(args.heldout_stains)]
    assert not leaked, f"held-out scanner/stain present in the training conditions: {leaked}"
    print(f"[train] held-out exclusion verified: no {args.heldout_scanners} scanner and no "
          f"{args.heldout_stains} stain among the {len(train_conds)} training conditions")

    train_loader = build_pair_loader(
        args.packed_dir, conditions=train_conds, n_groups=args.n_groups,
        group_size=args.group_size, batches_per_epoch=args.max_steps,
        num_workers=args.workers, seed=args.seed,
    )
    heldout_loader = (
        build_pair_loader(
            args.packed_dir, conditions=heldout_conds, n_groups=args.n_groups,
            group_size=args.group_size, batches_per_epoch=1000,
            num_workers=max(args.workers // 4, 1), seed=args.seed + 1,
        )
        if len(heldout_conds) >= 2
        else None
    )
    if heldout_loader is None:
        print("[train] WARNING: <2 held-out conditions repacked -- no held-out-condition eval. "
              "PLAN.md 3 risk 3 says this is the only in-training check against tile memorisation.")

    model = build_encoder(
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        proj_out_dim=args.proj_out_dim, pooling=args.pooling,
        grad_checkpointing=args.grad_checkpointing,
    )
    print("[train] params:", model.trainable_parameter_summary())

    cfg = TrainConfig(
        packed_dir=str(args.packed_dir), out_dir=str(args.out_dir),
        lr=args.lr, temperature=args.temperature, max_steps=args.max_steps,
        warmup_steps=args.warmup_steps, n_groups=args.n_groups, group_size=args.group_size,
        grad_accum=args.grad_accum, num_workers=args.workers, amp_dtype=args.amp,
        ckpt_every=args.ckpt_every, eval_every=args.eval_every, seed=args.seed,
        weight_decay=args.weight_decay, log_every=args.log_every, symmetric=args.symmetric,
        encoder={"lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                 "pooling": args.pooling, "proj_out_dim": args.proj_out_dim,
                 "grad_checkpointing": args.grad_checkpointing},
    )

    def on_checkpoint(model, step, metrics, ckpt_dir):
        """PLAN.md 3 phase 8 wants the eval at *every* checkpoint, not just the end.

        It is NOT run inline. Extraction + the CPU kNN is ~15-20 min per checkpoint,
        which would roughly double wall time and stall the GPU. ``eval_checkpoints.py``
        follows this directory on a second GPU instead and evaluates each ``step_*`` as
        it lands, so the RI-vs-step curve is live while training keeps the first GPU
        saturated. ``metrics.json`` is written last by ``save_checkpoint``, so its
        presence is the "this checkpoint is complete" sentinel the follower waits on.
        """
        print(f"[ckpt] step {step} -> {ckpt_dir}  {json.dumps(metrics)}", flush=True)

    summary = train(
        model, train_loader, cfg, heldout_loader=heldout_loader,
        device=args.device, on_checkpoint=on_checkpoint,
        # condition indices are positions in `train_conds`; anything outside that range
        # would mean a held-out condition reached the batch (it cannot, and we check).
        allowed_conditions=set(range(len(train_conds))),
    )
    print(f"[train] done -> {args.out_dir}  {json.dumps({k: v for k, v in summary.items() if k != 'history'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
