#!/usr/bin/env python
"""Full contrastive fine-tune entrypoint (PLAN.md 3, phase 7).

LoRA-all-blocks + masked InfoNCE over PLISM registered pairs, holding out 2 scanners and
3 stains. Retention/robustness evaluation hangs off ``on_checkpoint`` -- PLAN.md 3 phase 8
requires it at *every* checkpoint, and PLAN.md 6 requires it reported alongside every
robustness claim.

    python scripts/train_lora.py --out-dir runs/lora_r16_t007 --lora-rank 16 --temperature 0.07

Full fine-tuning (no LoRA):

    python scripts/train_lora.py --out-dir runs/full_ft --full-ft --lr 1e-5 \
        --ckpt-schedule "50,100,150,200,300,400,500,750,1000,1500,2000,3000,5000"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

from waivphaet.data.conditions import all_conditions, available_conditions, make_split
from waivphaet.data.pairs import build_pair_loader
from waivphaet.data.repack import present_filenames
from waivphaet.models.encoder import DEFAULT_BACKBONE, build_encoder
from waivphaet.train.contrastive import TrainConfig, train


def parse_ckpt_schedule(value: str) -> list[int]:
    """Parse a comma-separated checkpoint schedule, e.g. '50,100,150,300,500'."""
    try:
        steps = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--ckpt-schedule must be comma-separated integers, got: {value!r}")
    if not steps:
        raise argparse.ArgumentTypeError("--ckpt-schedule is empty")
    if any(s <= 0 for s in steps):
        raise argparse.ArgumentTypeError("--ckpt-schedule must contain positive integers")
    # Allow non-sorted input but normalise for predictability.
    return sorted(set(steps))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed-dir", type=Path, default=Path("/data/plism/repacked"))
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
    ap.add_argument("--ckpt-schedule", type=parse_ckpt_schedule, default=None,
                    help="Non-uniform checkpoint schedule (e.g. '25,50,75,100,125,150,175,"
                         "200,300,400,600,800,1000'). Overrides --ckpt-every. Useful for "
                         "full FT where the representation can degrade within tens of steps.")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--symmetric", action="store_true",
                    help="ABLATION ONLY: adds the anchor->positive direction, whose candidate "
                         "row spans conditions and reintroduces the acquisition shortcut")
    # Retention term (PLAN.md 2 "frozen-teacher anchor"). Both are ALSO unknown
    # hyperparameters (PLAN.md 3 risk 4), same class as --lr / --temperature / --lora-rank.
    # Default 0.0 == OFF == the exact training path every published number was produced on.
    ap.add_argument("--retention-kl-weight", type=float, default=0.0,
                    help="lambda for the relational-KL retention term: "
                         "total = infonce + lambda * KL(P_base || P_lora) over the pairwise "
                         "similarity matrix of the batch's anchor embeddings. 0.0 = off "
                         "(default, bit-identical to the pre-retention loss). Requires LoRA: "
                         "under --full-ft there is no adapter to disable, so the teacher "
                         "would be the student and the KL identically 0 -- that combination "
                         "is an error, not a warning.")
    ap.add_argument("--retention-kl-temperature", type=float, default=0.07,
                    help="distillation temperature for the retention KL. Defaults to the "
                         "contrastive --temperature: the similarity matrix is cosine, so at "
                         "tau=1 the row softmax over ~group_size candidates is near-uniform "
                         "and the term mostly measures noise.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute block activations in backward. The negative count is "
                         "group_size-1, so more negatives means a bigger forward batch; "
                         "without this, 80 GiB caps out around 340 images/step.")
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--pooling", default="clsmean", choices=["cls", "mean", "clsmean"])
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="HF id of the base backbone. Default owkin/phikon-v2 (ViT-L/16, "
                         "24 blocks, 1024-d). kaiko-ai/midnight is ViT-g/14, 40 blocks, "
                         "1536-d -- ~3x the parameters, so re-measure the batch that fits "
                         "before reusing a phikon-v2 --n-groups/--group-size.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Full FT mode
    ap.add_argument("--full-ft", action="store_true",
                    help="Full fine-tuning: train all backbone weights instead of LoRA. "
                         "Use a lower learning rate (e.g. 1e-5 instead of 1e-4). "
                         "Checkpoints are ~3.4 GiB each for ViT-L (backbone.safetensors "
                         "1.13 GiB + optim.pt ~2.26 GiB of AdamW moments + projector).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/data/huggingface")
    torch.manual_seed(args.seed)

    use_lora = not args.full_ft

    # --- Retention term requires a frozen teacher, which only LoRA gives us for free ---
    # Under --full-ft the "frozen base" and the student are the same weights, so the KL
    # would be identically 0: a run that looks retention-regularised but regularises
    # nothing. Refuse rather than compute the degenerate loss. Checked here (before the
    # backbone download / build) as well as in train(), so the failure is instant.
    if args.retention_kl_weight > 0 and args.full_ft:
        raise SystemExit(
            "--retention-kl-weight > 0 is incompatible with --full-ft: there is no adapter "
            "to disable, so the frozen teacher would BE the student and the relational KL "
            "would be identically 0. Drop --full-ft, or set --retention-kl-weight 0."
        )
    if args.retention_kl_weight < 0:
        raise SystemExit("--retention-kl-weight must be >= 0")

    # --- LR warning for full FT ---
    if args.full_ft and args.lr >= 5e-5:
        print(
            f"[train] WARNING: --full-ft with lr={args.lr} >= 5e-5. Full FT at high LR "
            "destroys the representation in tens of steps. Waiv's Phaet used ~1e-5 class. "
            "Consider --lr 1e-5. Proceeding anyway.", flush=True,
        )

    # --- Disk space warning for full FT (~3.4 GiB/ckpt for ViT-L) ---
    if args.full_ft:
        import shutil
        try:
            _, _, free = shutil.disk_usage(str(args.out_dir.parent))
            # Estimate checkpoint count: schedule-based or ckpt_every-based
            if args.ckpt_schedule:
                ckpt_count = len(args.ckpt_schedule)
            else:
                ckpt_count = math.ceil(args.max_steps / max(args.ckpt_every, 1))
            # +1 for final checkpoint
            ckpt_count += 1
            # A full-FT checkpoint is NOT just the backbone: backbone.safetensors is
            # 1.13 GiB (303M fp32 params) but optim.pt carries AdamW's two moment
            # tensors over the same params (~2.26 GiB), plus the projector. ~3.4 GiB.
            est_gb = ckpt_count * 3.4
            free_gb = free / 2**30
            if free_gb < est_gb * 1.5:
                print(
                    f"[train] WARNING: full FT with ~{ckpt_count} checkpoints "
                    f"estimates {est_gb:.0f} GB. Free space: {free_gb:.0f} GB. "
                    f"Consider a sparser --ckpt-schedule or --ckpt-every.", flush=True,
                )
        except OSError:
            pass  # can't stat output dir parent, skip warning

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
        backbone=args.backbone,
        use_lora=use_lora,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        proj_out_dim=args.proj_out_dim, pooling=args.pooling,
        grad_checkpointing=args.grad_checkpointing,
    )
    print("[train] params:", model.trainable_parameter_summary())

    # Full FT guard: trainable params should ~= total params.
    # A full-FT run that silently trains only the projector looks like a weak result, not an error.
    if args.full_ft:
        summary = model.trainable_parameter_summary()
        total = summary["total"]
        trainable = summary["trainable"]
        # Allow a small tolerance for the projector + any unavoidable non-trainable params
        pct = trainable / max(total, 1)
        if pct < 0.95:
            raise RuntimeError(
                f"FULL FT GUARD FAILED: trainable={trainable}, total={total}, "
                f"pct={pct:.1%} < 95%%. The backbone appears frozen or LoRA is active. "
                "This will produce a weak result silently. "
                "Check --full-ft is setting use_lora=False and the backbone is unfrozen."
            )
        print(
            f"[train] FULL FT verified: {trainable}/{total} params trainable "
            f"({pct:.1%}). Back to training.", flush=True,
        )

    # Resolve checkpoint schedule
    if args.ckpt_schedule is not None:
        ckpt_schedule = args.ckpt_schedule
        # Filter to steps within max_steps
        ckpt_schedule = [s for s in ckpt_schedule if s <= args.max_steps]
        if not ckpt_schedule:
            print(f"[train] WARNING: --ckpt-schedule has no steps <= --max-steps {args.max_steps}; "
                  "falling back to --ckpt-every", flush=True)
            ckpt_schedule = None
    else:
        ckpt_schedule = None

    cfg = TrainConfig(
        packed_dir=str(args.packed_dir), out_dir=str(args.out_dir),
        lr=args.lr, temperature=args.temperature, max_steps=args.max_steps,
        warmup_steps=args.warmup_steps, n_groups=args.n_groups, group_size=args.group_size,
        grad_accum=args.grad_accum, num_workers=args.workers, amp_dtype=args.amp,
        ckpt_every=args.ckpt_every, eval_every=args.eval_every, seed=args.seed,
        weight_decay=args.weight_decay, log_every=args.log_every, symmetric=args.symmetric,
        retention_kl_weight=args.retention_kl_weight,
        retention_kl_temperature=args.retention_kl_temperature,
        encoder={"backbone": args.backbone, "use_lora": use_lora,
                 "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                 "pooling": args.pooling, "proj_out_dim": args.proj_out_dim,
                 "grad_checkpointing": args.grad_checkpointing},
    )
    if ckpt_schedule is not None:
        cfg.ckpt_schedule = ckpt_schedule

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
