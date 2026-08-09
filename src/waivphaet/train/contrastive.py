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
import os
import time
from contextlib import contextmanager
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


# --------------------------------------------------------------------------------------
# Retention term: relational KL against the frozen base model (PLAN.md 2, "frozen-teacher
# anchor" -- scoped there as optional, never built until now). OFF by default.
#
# Why RELATIONAL and not an L2/cosine pull toward the base embedding
# ------------------------------------------------------------------
# The robustness gain this repo reproduces comes precisely from MOVING embeddings: the
# fine-tune collapses the scanner/stain directions that the base model happily encodes.
# A pull toward the frozen base embedding opposes that move directly -- it would buy
# retention by giving back robustness, which is the one trade PLAN.md 6 forbids.
#
# So we constrain the RELATIVE geometry instead of the absolute position. Take the
# pairwise similarity matrix over the batch's anchors under the student and under the
# frozen base, softmax each row, and penalise KL(P_base || P_student). A global rotation,
# translation or rescaling of the embedding space leaves both distributions untouched and
# costs nothing; deleting a whole confounder direction is also (near) free, because it
# moves all tiles in a group the same way. What it DOES punish is shredding the
# tile-to-tile ordering that the downstream HEST/THUNDER probes read.
#
# Which embeddings
# ----------------
# BACKBONE POOLED output (``WaivEncoder.embed`` / the first element of ``forward``), NOT
# the projector output. The projector is randomly initialised and discarded at eval time
# -- PathoROB, HEST and THUNDER all read the pooled embedding -- so "preserve the
# projector's geometry" would be preserving the geometry of a random map. It is also the
# SAME projector on both sides here (only the backbone carries adapters), which would make
# the teacher/student comparison partly an artefact of that shared random head.
#
# Diagonal / masking
# ------------------
# * Self-similarity is REMOVED from both distributions. After L2 normalisation S_ii == 1
#   exactly for teacher and student alike, so it carries zero information -- but at a
#   distillation temperature of 0.07, exp(1/0.07) swamps every off-diagonal term, both
#   rows become ~one-hot on the diagonal, and the KL would collapse to ~0 no matter what
#   the model did to the geometry. Masking it is what keeps the term from being inert.
# * Candidates are restricted to the anchor's own group (the same ``same_group`` mask
#   InfoNCE uses), when ``group_id`` is supplied. Within a group every anchor shares one
#   (scanner, stain), so the relative geometry there is pure tissue signal with the
#   acquisition variable held constant. Letting the KL span groups would ask the student
#   to preserve teacher similarities that are partly acquisition-driven -- i.e. it would
#   quietly reintroduce the confounder the InfoNCE term exists to remove.
# * Teacher and student are masked identically, always.


def relational_kl(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    group_id: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL(P_teacher || P_student) over row-wise softmaxed cosine-similarity matrices.

    Equations (``s`` = student, ``t`` = teacher, both L2-normalised, tau = temperature)::

        S^s_ij = <s_i, s_j> / tau        S^t_ij = <t_i, t_j> / tau
        valid_ij = (i != j) and (group_id[i] == group_id[j] if group_id is given)
        P^x_i.   = softmax_j( S^x_ij )   over valid j only
        L        = mean_i  sum_j P^t_ij * ( log P^t_ij - log P^s_ij )

    Non-negative by Gibbs' inequality, and exactly 0 iff every row distribution matches
    (in particular whenever ``student_emb`` and ``teacher_emb`` agree up to a global
    rotation/scale).

    Args:
        student_emb: ``(B, D)`` pooled backbone embeddings from the adapted model.
        teacher_emb: ``(B, D)`` pooled backbone embeddings from the frozen base model.
        group_id: optional ``(B,)`` group membership; restricts candidates to the anchor's
            own condition-homogeneous group. ``None`` = whole batch.
        temperature: distillation temperature. Unknown hyperparameter (PLAN.md 3 risk 4).

    Returns:
        (loss, metrics). ``loss`` is a 0-dim tensor carrying gradient only through
        ``student_emb``.
    """
    if temperature <= 0.0:
        raise ValueError(f"retention_kl_temperature must be > 0, got {temperature}")

    s = F.normalize(student_emb.float(), dim=-1)
    t = F.normalize(teacher_emb.float(), dim=-1)
    if s.shape != t.shape:
        raise ValueError(f"teacher/student embedding shape mismatch: {t.shape} vs {s.shape}")

    n = s.shape[0]
    valid = ~torch.eye(n, dtype=torch.bool, device=s.device)
    if group_id is not None:
        valid = valid & (group_id[:, None] == group_id[None, :])
    # A row with no candidate left (group of one) has no relational content and would
    # softmax to NaN. Drop such rows rather than poisoning the batch mean.
    keep = valid.any(dim=1)
    if not bool(keep.any()):
        zero = student_emb.sum() * 0.0
        return zero, {"loss_retention_kl": 0.0, "retention_kl_neighbours": 0.0}

    s_logits = ((s @ s.t()) / temperature).masked_fill(~valid, NEG_INF)[keep]
    t_logits = ((t @ t.t()) / temperature).masked_fill(~valid, NEG_INF)[keep]

    log_p_student = F.log_softmax(s_logits, dim=-1)
    log_p_teacher = F.log_softmax(t_logits, dim=-1)
    p_teacher = log_p_teacher.exp()
    # masked_fill'd entries give p=0 and log_p=-inf; 0 * -inf is NaN, so zero them out
    # explicitly instead of relying on the product.
    per_row = torch.where(
        valid[keep], p_teacher * (log_p_teacher - log_p_student), torch.zeros_like(p_teacher)
    ).sum(dim=-1)
    loss = per_row.mean()

    with torch.no_grad():
        n_neigh = float(valid[keep].sum(dim=1).float().mean().item())
    return loss, {
        "loss_retention_kl": float(loss.detach()),
        "retention_kl_neighbours": n_neigh,
    }


def assert_retention_teacher_available(model) -> None:
    """Fail loudly when the retention term is requested but there is no frozen teacher.

    Under full fine-tuning there is no adapter to switch off, so "the frozen base model"
    and "the student" are the same weights: the KL would be identically 0 and the run
    would look like a clean retention-regularised fine-tune while regularising nothing.
    That is a silent-degenerate-result failure of exactly the kind this repo's guards
    exist to prevent, so it is an error, not a warning.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is not None and not getattr(cfg, "use_lora", True):
        raise ValueError(
            "retention_kl_weight > 0 requires LoRA: with --full-ft there is no adapter to "
            "disable, so the frozen teacher would be the student itself and the KL would "
            "be identically 0. Either drop --full-ft or set --retention-kl-weight 0."
        )
    backbone = getattr(model, "backbone", None)
    if not callable(getattr(backbone, "disable_adapter", None)):
        raise ValueError(
            "retention_kl_weight > 0 requires a PEFT-wrapped backbone exposing "
            "`disable_adapter()` (the frozen teacher is the adapter-disabled student). "
            f"Got backbone of type {type(backbone).__name__}."
        )


@contextmanager
def frozen_teacher(model):
    """Run the SAME model as the frozen base: adapters off, no grad, eval mode, RNG-neutral.

    Deliberately does NOT load a second copy of the backbone (~1.2 GB for ViT-L) -- the
    teacher is free under LoRA, exactly the idiom ``assert_adapter_applied`` uses in
    ``scripts/extract_pathorob_features.py``.

    Three properties this has to guarantee:

    * **No gradient.** ``torch.no_grad()`` -- the teacher must be a constant target.
    * **Deterministic.** ``eval()`` for the duration, so dropout / stochastic depth are
      identity and the target does not jitter step to step.
    * **RNG-neutral.** The global CPU and CUDA RNG states are saved and restored, so
      adding this forward pass cannot shift the random stream that the InfoNCE path (or
      any sampler seeded from it) consumes. Combined with eval() this is belt and braces,
      but it makes the guarantee independent of whether the backbone happens to have a
      non-zero dropout rate.

    Caveat, same one ``evaluate_heldout`` already carries: restoring the mode calls
    ``model.train()`` on the whole module, so a submodule deliberately pinned to eval
    would be un-pinned.
    """
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()
    try:
        with torch.no_grad(), model.backbone.disable_adapter():
            yield
    finally:
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def retention_teacher_embed(model, images: torch.Tensor) -> torch.Tensor:
    """Pooled backbone embedding of ``images`` under the frozen base model.

    Returned tensor is detached: it is a target, never a path to gradients.
    """
    with frozen_teacher(model):
        return model.embed(images).detach()


@dataclass
class TrainConfig:
    """Every value here is a guess. PLAN.md 3 risk 4: "no recipe means hyperparameter
    search, not a single run" -- LR / steps / LoRA rank / temperature are all unknown."""

    #: ``$WAIV_PACKED_DIR``, read per-instantiation rather than at import so the library
    #: carries no machine-specific path; the fallback is this cluster's repacked PLISM so
    #: every existing launcher keeps working unchanged.
    packed_dir: str = field(
        default_factory=lambda: os.environ.get("WAIV_PACKED_DIR", "/data/plism/repacked")
    )
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
    #: Optional non-uniform checkpoint schedule (e.g. [50, 100, 200, 500, 1000]).
    #: Overrides ckpt_every when set. Useful for full FT where representation can
    #: degrade within tens of steps.
    ckpt_schedule: list[int] | None = None
    #: PLAN.md 3 phase 8: "evaluate retention at every checkpoint, not just at the end".
    #: A robustness win that costs retention is a failed reproduction (risk 1). Point this
    #: at a callable (or leave None and let the caller hook `on_checkpoint`).
    eval_heldout: bool = True
    #: Retention term (PLAN.md 2 "frozen-teacher anchor", optional). 0.0 = OFF and the
    #: training path is bit-identical to the pre-retention implementation -- the published
    #: numbers are all at 0.0. Lambda in ``total = infonce + lambda * relational_kl``.
    #: Unknown hyperparameter (PLAN.md 3 risk 4), same class as lr / temperature / rank.
    retention_kl_weight: float = 0.0
    #: Distillation temperature for the relational KL. Defaults to the CONTRASTIVE
    #: temperature rather than 1.0: the similarity matrix is cosine, so it lives in
    #: [-1, 1], and at tau=1 a softmax over ~30-60 candidates is nearly uniform -- the
    #: term would mostly measure noise. Matching ``temperature`` makes the retention term
    #: resolve neighbourhood structure at the same scale as the objective it trades
    #: against, and avoids inventing a second arbitrary constant. Also unknown (risk 4).
    retention_kl_temperature: float = 0.07
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
    """Save a training checkpoint.

    LoRA mode: saves PEFT adapter dir (a few MB) + projector.pt + optim.pt + metrics.json.
    Full FT mode: saves backbone.safetensors (~1.2 GB for ViT-L) + projector.pt + optim.pt
        + metrics.json. No adapter directory.

    ``metrics.json`` is written LAST -- it is the completeness sentinel that
    ``eval_checkpoints.py:discover`` waits on before attempting to load the checkpoint.
    """
    out = Path(cfg.out_dir) / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)

    if getattr(model.cfg, "use_lora", False):
        # LoRA mode: save PEFT adapter (small, ~few MB).
        model.backbone.save_pretrained(out / "adapter")
    else:
        # Full FT mode: save full backbone weights via safetensors (~1.2 GB for ViT-L).
        backbone_sd = model.backbone.state_dict()
        from safetensors.torch import save_file
        save_file(backbone_sd, str(out / "backbone.safetensors"))

    torch.save(model.projector.state_dict(), out / "projector.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, out / "optim.pt")
    # metrics.json is the completeness sentinel -- written LAST.
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out


def _should_checkpoint(step: int, cfg: TrainConfig) -> bool:
    """Determine whether to checkpoint at *step* using schedule or periodic interval."""
    if cfg.ckpt_schedule is not None:
        return step in cfg.ckpt_schedule
    return step % cfg.ckpt_every == 0


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
    if cfg.retention_kl_weight < 0.0:
        raise ValueError(f"retention_kl_weight must be >= 0, got {cfg.retention_kl_weight}")
    use_retention = cfg.retention_kl_weight > 0.0
    if use_retention:
        # Fail here, before any compute, rather than silently optimising a KL that is
        # structurally 0 (see assert_retention_teacher_available).
        assert_retention_teacher_available(model)
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
    # pre-clip grad norm from the most recent optimizer step; None until the first one
    # (and stays None entirely when cfg.grad_clip is falsy, since nothing computes it)
    last_grad_norm: float | None = None
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
                anchor_emb, az = model(anchor)
                _, pz = model(positive)
                # Frozen-teacher forward for the retention term. Guarded so that at the
                # default weight of 0.0 NOTHING here runs -- no extra forward, no extra
                # tensor, no RNG touched -- and the step stays bit-identical.
                teacher_emb = (
                    retention_teacher_embed(model, anchor)
                    if use_retention else None
                )
            loss, metrics = masked_info_nce(az, pz, gid, cfg.temperature, cfg.symmetric)
            if teacher_emb is not None:
                # Same embeddings the model exports at eval time (pooled backbone), and
                # the same group mask InfoNCE uses -- see the relational_kl block comment.
                kl, kl_metrics = relational_kl(
                    anchor_emb, teacher_emb, group_id=gid,
                    temperature=cfg.retention_kl_temperature,
                )
                # total = infonce + lambda * kl
                loss = loss + cfg.retention_kl_weight * kl
                # Log the two terms SEPARATELY so the trade-off is visible, not just the
                # sum. "loss" keeps its pre-retention meaning (the InfoNCE term) so
                # history.json stays comparable across runs.
                metrics["loss_infonce"] = metrics["loss"]
                metrics.update(kl_metrics)
                metrics["loss_total"] = float(loss.detach())

            scaler.scale(loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    # The return is the PRE-clip total norm. Keep it: under full FT a
                    # grad-norm spike is the earliest visible collapse signal -- it leads
                    # the loss and the top1 by several steps.
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                    )
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
                    "grad_norm": last_grad_norm,
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

            if _should_checkpoint(step, cfg) or step == cfg.max_steps:
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
