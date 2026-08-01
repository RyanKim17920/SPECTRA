"""InfoNCE over PLISM registered pairs, with the same-condition negative constraint.

The loss (PLAN.md 2)
----------------------
Standard InfoNCE would use *every* other sample in the batch as a negative. We must not:
negatives have to share the anchor's (scanner, stain), or "different scanner" becomes a
partially-correct shortcut for "different tile" and the objective starts *rewarding*
retained acquisition signal.

:mod:`waivphaet.data.pairs` already lays the batch out in condition-homogeneous groups,
so enforcing the constraint here is one mask::

    valid_negative[i, j]  <=>  group_id[i] == group_id[j]

**Which way the softmax runs is not cosmetic.** The *anchors* are the condition-homogeneous
side: within a group every anchor shares one (scanner, stain). The *positives* are not --
each is drawn from its own randomly chosen different condition. So the query has to be the
positive and the candidate set has to be the anchors::

    logits[k, j] = positive_k . anchor_j / T      for j in group(k),  target j = k

Read the candidate row: all ``group_size`` anchors come from one single condition, so
acquisition carries **zero** discriminative information among the candidates and the only
way to find the match is tissue identity. That is ScanGen's "different specimen, same
scanner" repulsion.

Run it the other way (``anchor_k`` against all ``positive_j``) and the candidates span
conditions again -- a negative can be pushed away because it was scanned differently
rather than because it is different tissue, which is precisely the shortcut PLAN.md 2
forbids. Worse, in a mixed row containing the anchor's own same-condition siblings, "pick
the candidate whose acquisition differs from mine" *is* the correct answer, so the
objective would actively reward retaining scanner signal. ``symmetric=True`` adds that
direction and therefore defaults to **False**; it is kept only for ablation.

Note what the constraint costs: the effective negative count is ``group_size - 1``, not
``batch - 1``. Contrastive learning likes many negatives, so prefer *fewer, larger*
groups (e.g. 4 groups x 128) over many small ones.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from waivphaet.data.pairs import assert_same_condition_negatives

NEG_INF = float("-inf")


def masked_info_nce(
    anchor_z: torch.Tensor,
    positive_z: torch.Tensor,
    group_id: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE restricted to same-condition negatives.

    Args:
        anchor_z: ``(B, D)`` projections of the anchors. All anchors sharing a
            ``group_id`` come from the *same* condition -- that is the invariant the
            sampler guarantees and this loss relies on. They are the **candidates**.
        positive_z: ``(B, D)`` projections of the registered positives (different
            condition, same tile). They are the **queries**.
        group_id: ``(B,)`` group membership.
        temperature: softmax temperature. Unknown hyperparameter (PLAN.md 3 risk 4).
        symmetric: also average in the anchor->positive direction. Its candidate row
            spans conditions, so it reintroduces the acquisition shortcut; off by
            default, kept for ablation only (see the module docstring).

    Returns:
        (loss, metrics) where metrics carries top-1 pair-retrieval accuracy within the
        batch -- the cheap online proxy for the PLISM retrieval diagnostic.
    """
    a = F.normalize(anchor_z.float(), dim=-1)
    p = F.normalize(positive_z.float(), dim=-1)

    same_group = group_id[:, None] == group_id[None, :]
    if not bool(same_group.diagonal().all()):  # pragma: no cover - defensive
        raise ValueError("group_id mask does not cover the diagonal (its own positive)")

    # query = positive (its own condition), candidates = the group's anchors (ONE shared
    # condition). Condition is constant down the candidate row, so it cannot be used to
    # find the match. This orientation is the whole point -- see the module docstring.
    logits = ((p @ a.t()) / temperature).masked_fill(~same_group, NEG_INF)
    target = torch.arange(a.shape[0], device=a.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), target))

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean().item()
        n_neg = float(same_group.sum(dim=1).float().mean().item() - 1.0)
    return loss, {"loss": float(loss.detach()), "top1": acc, "negatives_per_anchor": n_neg}


@dataclass
class TrainConfig:
    """Every value here is a guess. PLAN.md 3 risk 4: "no recipe means hyperparameter
    search, not a single run" -- LR / steps / LoRA rank / temperature are all unknown."""

    packed_dir: str = "/data/ryan.kim/plism/repacked"
    out_dir: str = "runs/dev"
    # optimisation
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    max_steps: int = 5000
    temperature: float = 0.07
    #: see the module docstring -- the anchor->positive direction has cross-condition
    #: candidates and reintroduces the acquisition shortcut. Ablation knob, not a default.
    symmetric: bool = False
    grad_accum: int = 1
    grad_clip: float = 1.0
    # batching -- prefer FEWER, LARGER groups: negatives per anchor is group_size - 1
    n_groups: int = 4
    group_size: int = 32
    num_workers: int = 8
    seed: int = 0
    # precision
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "none"
    # checkpointing / eval
    ckpt_every: int = 500
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 20
    #: PLAN.md 3 phase 8: "evaluate retention at every checkpoint, not just at the end".
    #: A robustness win that costs retention is a failed reproduction (risk 1). Point this
    #: at a callable (or leave None and let the caller hook `on_checkpoint`).
    eval_heldout: bool = True
    encoder: dict = field(default_factory=dict)


def _amp_dtype(name: str) -> torch.dtype | None:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[name]


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    prog = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate_heldout(model, loader, cfg: TrainConfig, device, n_batches: int) -> dict[str, float]:
    """Same objective, held-out *conditions*.

    PLAN.md 3 risk 3: 16,278 tile locations is a small instance-discrimination set and
    "memorization of tile identity won't show in training loss" -- held-out-condition
    splits are the only in-training check. This is NOT a robustness metric; PathoROB is
    (PLAN.md 1).
    """
    model.eval()
    tot, n = {"loss": 0.0, "top1": 0.0}, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        _, az = model(anchor)
        _, pz = model(positive)
        _, m = masked_info_nce(
            az, pz, batch["group_id"].to(device), cfg.temperature, cfg.symmetric
        )
        tot["loss"] += m["loss"]
        tot["top1"] += m["top1"]
        n += 1
    model.train()
    return {f"heldout_{k}": v / max(n, 1) for k, v in tot.items()}


def save_checkpoint(model, optimizer, step: int, cfg: TrainConfig, metrics: dict) -> Path:
    out = Path(cfg.out_dir) / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)
    # LoRA adapters only when using PEFT -- a few MB instead of 1.2 GB per checkpoint,
    # which is what makes "checkpoint often" (PLAN.md 3 phase 8) affordable.
    if getattr(model.cfg, "use_lora", False):
        model.backbone.save_pretrained(out / "adapter")
    else:
        torch.save(model.backbone.state_dict(), out / "backbone.pt")
    torch.save(model.projector.state_dict(), out / "projector.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, out / "optim.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out


def train(
    model,
    train_loader,
    cfg: TrainConfig,
    *,
    heldout_loader=None,
    device: str | torch.device = "cuda",
    on_checkpoint=None,
    allowed_conditions: set[int] | None = None,
) -> dict:
    """Run the contrastive fine-tune. Returns the final metrics dict.

    ``on_checkpoint(model, step, metrics, ckpt_dir)`` is where the retention evals
    (HEST / THUNDER / Patho-Bench) and PathoROB hang -- PLAN.md 3 phase 8 requires them
    at *every* checkpoint, and PLAN.md 6 requires retention reported alongside every
    robustness claim, always as a pair.
    """
    device = torch.device(device)
    model.to(device).train()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = _amp_dtype(cfg.amp_dtype)
    scaler = torch.amp.GradScaler(enabled=(amp is torch.float16))

    history: list[dict] = []
    step = 0
    t0 = time.time()
    tiles_seen = 0
    win_tiles, win_t0 = 0, time.time()
    pbar = tqdm(total=cfg.max_steps, desc="train", unit="step")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps:
                break
            # PLAN.md 2's load-bearing detail, asserted rather than assumed. Runs every
            # step: it is a few microseconds on CPU-side index tensors, and a violation
            # here would still produce a perfectly plausible falling loss curve.
            batch_stats = assert_same_condition_negatives(
                batch, allowed_conditions=allowed_conditions
            )
            lr = cosine_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            gid = batch["group_id"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                # one forward per view; both views must see the same weights, so this is
                # a single graph, not two independent steps
                _, az = model(anchor)
                _, pz = model(positive)
            loss, metrics = masked_info_nce(az, pz, gid, cfg.temperature, cfg.symmetric)

            scaler.scale(loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            # two forward views per anchor, so a "step" moves 2 * batch tiles
            n_tiles = int(anchor.shape[0]) * 2
            tiles_seen += n_tiles
            win_tiles += n_tiles
            if step % cfg.log_every == 0:
                now = time.time()
                rec = {
                    "step": step,
                    "lr": lr,
                    "elapsed_s": now - t0,
                    "tiles_seen": tiles_seen,
                    "tiles_per_s": win_tiles / max(now - win_t0, 1e-9),
                    **metrics,
                    **{f"batch_{k}": v for k, v in batch_stats.items()},
                }
                if device.type == "cuda":
                    rec["gpu_mem_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                    rec["gpu_mem_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
                win_tiles, win_t0 = 0, now
                history.append(rec)
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    top1=f"{metrics['top1']:.3f}",
                    tps=f"{rec['tiles_per_s']:.0f}",
                )

            if step % cfg.eval_every == 0 and heldout_loader is not None and cfg.eval_heldout:
                metrics.update(
                    evaluate_heldout(model, heldout_loader, cfg, device, cfg.eval_batches)
                )

            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                ck = save_checkpoint(model, optimizer, step, cfg, {"step": step, **metrics})
                if on_checkpoint is not None:
                    on_checkpoint(model, step, metrics, ck)

    pbar.close()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "step": step,
        "elapsed_s": time.time() - t0,
        "tiles_seen": tiles_seen,
        "mean_tiles_per_s": tiles_seen / max(time.time() - t0, 1e-9),
    }
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        summary.update(
            peak_alloc_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            device_total_gib=total_b / 2**30,
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {**summary, "history": history}
