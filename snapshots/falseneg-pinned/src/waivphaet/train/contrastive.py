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
import random
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from waivphaet.data.grid import assert_grid_batch
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


def _grid_pair_block_loss(
    zn: torch.Tensor,
    a_blk: torch.Tensor,
    b_blk: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy (and hit count) for ONE block of ordered condition pairs.

    Materialises only ``(P_blk, T, T)`` logits. Returns a SUM rather than a mean so the
    caller can divide once at the end -- a mean of per-block means would be wrong the
    moment the last block is short.
    """
    n_tiles = zn.shape[1]
    zq = zn[a_blk]                                   # (P_blk, T, D) queries
    zk = zn[b_blk]                                   # (P_blk, T, D) candidates
    logits = torch.matmul(zq, zk.transpose(1, 2)) / temperature   # (P_blk, T, T)
    flat = logits.reshape(-1, n_tiles)
    target = torch.arange(n_tiles, device=zn.device).repeat(a_blk.numel())
    loss_sum = F.cross_entropy(flat, target, reduction="sum")
    with torch.no_grad():
        correct = (flat.argmax(dim=1) == target).sum()
    return loss_sum, correct


def grid_info_nce_blocked(
    z: torch.Tensor,
    n_cond: int,
    n_tiles: int,
    temperature: float = 0.07,
    pair_block: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Memory-lean :func:`grid_info_nce`. EXACT -- a numerics refactor, not an approximation.

    Why. The dense path materialises the full ``(C, C, T, T)`` logit tensor, then an
    advanced-index COPY of its off-diagonal, then ``cross_entropy`` saves another for
    backward: roughly 3x live at ``4*C^2*T^2`` bytes. That is negligible today (~187 MB at
    T=1975) but it is QUADRATIC in T, so it binds around T~6000-8000 -- which is where the
    offload work is heading.

    How. Iterate the ``C*(C-1)`` ordered pairs in blocks, gathering each block's queries and
    candidates directly instead of slicing a dense tensor, so the off-diagonal copy never
    exists either. Each block is wrapped in ``torch.utils.checkpoint``: pass 1 computes the
    block's contribution and DISCARDS its intermediates, pass 2 recomputes one block at a
    time during backward. Peak logit memory becomes ``4*P_blk*T^2`` regardless of C.

    Blocking alone would NOT have been enough -- autograd would have retained every block's
    saved tensors and the total would be the dense figure again. The recompute is what makes
    it bounded, and it is the second pass the name promises.

    Exactness. Every block runs the same matmul and the same ``cross_entropy`` on the same
    inputs as the dense path; only the ORDER of the final float summation differs, so the
    two agree to float tolerance rather than bit-exactly. The recompute is safe because the
    block function is pure: it consumes no RNG and depends on nothing but ``zn``.

    Args:
        pair_block: ordered pairs per block. Peak logit bytes are ``4*pair_block*T^2``;
            memory falls and kernel-launch overhead rises as it shrinks.
    """
    if pair_block < 1:
        raise ValueError(f"pair_block must be >= 1, got {pair_block}")
    _check_grid_shape(z, n_cond, n_tiles)

    zn = F.normalize(z.float(), dim=-1).view(n_cond, n_tiles, -1)
    off = ~torch.eye(n_cond, dtype=torch.bool, device=zn.device)
    a_idx, b_idx = torch.where(off)
    n_pairs = int(a_idx.numel())

    loss_sum = zn.new_zeros(())
    correct = zn.new_zeros((), dtype=torch.long)
    for start in range(0, n_pairs, pair_block):
        a_blk = a_idx[start:start + pair_block]
        b_blk = b_idx[start:start + pair_block]
        if zn.requires_grad:
            blk_loss, blk_correct = torch.utils.checkpoint.checkpoint(
                _grid_pair_block_loss, zn, a_blk, b_blk, temperature, use_reentrant=False,
            )
        else:
            # No graph to bound under eval/no_grad -- checkpoint would only buy a
            # pointless second forward.
            blk_loss, blk_correct = _grid_pair_block_loss(zn, a_blk, b_blk, temperature)
        loss_sum = loss_sum + blk_loss
        correct = correct + blk_correct

    n_rows = n_pairs * n_tiles
    loss = loss_sum / n_rows
    # Same keys and same meanings as the dense path -- a caller must not be able to tell
    # which one produced its metrics.
    return loss, {
        "loss": float(loss.detach()),
        "top1": float(correct) / n_rows,
        "negatives_per_anchor": float(n_tiles - 1),
        "n_rows": float(n_rows),
        "n_cond": float(n_cond),
        "n_tiles": float(n_tiles),
    }


def _check_grid_shape(z: torch.Tensor, n_cond: int, n_tiles: int) -> None:
    """Shared preconditions for the dense and blocked grid losses."""
    if n_cond < 2:
        raise ValueError(
            f"grid_info_nce needs n_cond >= 2, got {n_cond}: a row's candidates come from a "
            "DIFFERENT condition group, so C=1 produces zero query rows"
        )
    if n_tiles < 2:
        raise ValueError(f"grid_info_nce needs n_tiles >= 2, got {n_tiles}: a row has no negatives")
    if z.shape[0] != n_cond * n_tiles:
        raise ValueError(
            f"z has {z.shape[0]} rows but the grid is {n_cond} x {n_tiles} = "
            f"{n_cond * n_tiles}; the (C*T,) flatten does not match the declared geometry"
        )


def grid_info_nce(
    z: torch.Tensor,
    n_cond: int,
    n_tiles: int,
    temperature: float = 0.07,
    core_mask: torch.Tensor | None = None,
    core_bias: float | None = None,
    center: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE over a shared-tile GRID batch (:mod:`waivphaet.data.grid`).

    Every image is BOTH an anchor in its own condition group AND a query against every
    other condition group, so ``C*T`` forward passes yield ``C*(C-1)*T`` query rows --
    where :func:`masked_info_nce` spends ``2*G*S`` forwards on ``G*S`` rows and throws the
    positives' embeddings away after one use each.

    Args:
        z: ``(C*T, D)`` projections, laid out ROW-MAJOR as ``(C, T, D)``: index ``a*T + t``
            is condition ``a``, tile position ``t``. ``collate_grid_batch`` guarantees this
            and ``assert_grid_batch`` re-checks it via ``tile_pos``.
        n_cond: ``C``, the number of distinct condition groups.
        n_tiles: ``T``, the shared tile count. Negatives per row is ``T-1``.
        temperature: softmax temperature.

    **Orientation is load-bearing, exactly as in** :func:`masked_info_nce`. For the ordered
    pair ``(a, b)`` the query is ``z[a, t]`` and the CANDIDATES are the whole of ``z[b, :]``
    -- one single condition. Acquisition is constant down the candidate row and therefore
    carries zero discriminative information; the only way to find the match is tissue
    identity. Transposing this would make the candidate row span conditions and hand the
    objective back the acquisition shortcut PLAN.md 2 forbids. ``a == b`` is excluded: its
    "positive" is the image itself, at similarity 1 by construction.

    Both orderings ``(a,b)`` and ``(b,a)`` ARE included -- unlike ``masked_info_nce``'s
    ``symmetric`` flag, which is unsafe precisely because its reverse direction has
    cross-condition candidates. Here the reverse direction's candidate row is condition
    ``a``: still homogeneous, still safe.

    The full ``(C, C, T, T)`` logit tensor is materialised in one einsum -- 5.76M floats
    (~23 MB fp32) at both C=24,T=100 and C=49,T=49, i.e. negligible next to the activations.

    Returns:
        (loss, metrics). ``loss`` is the mean over the ``C*(C-1)`` ordered pairs.
    """
    _check_grid_shape(z, n_cond, n_tiles)

    # Batch-centering. Measured 2026-08-21: ~50% of the energy of the fine-tuning update
    # is a SINGLE SHARED SHIFT common to every tile. Cosine is shift-sensitive, so under
    # the plain loss that shift is a FREE way to raise same-tile similarity -- and HEST
    # scores via StandardScaler -> PCA -> Ridge, which CENTERS, so it cannot see it.
    # Subtracting the batch mean makes a shared shift give EXACTLY ZERO loss reduction,
    # forcing capacity into centered structure, which is what a linear probe reads.
    _z = z.float()
    if center:
        _z = _z - _z.mean(dim=0, keepdim=True)
    zn = F.normalize(_z, dim=-1).view(n_cond, n_tiles, -1)
    # logits[a, b, t, s] = <z[a,t], z[b,s]> / tau -- query (a,t), candidate (b,s)
    logits = torch.einsum("atd,bsd->abts", zn, zn) / temperature

    # keep the ORDERED off-diagonal pairs only; a == b scores the image against itself
    off = ~torch.eye(n_cond, dtype=torch.bool, device=zn.device)
    a_idx, b_idx = torch.where(off)
    pair_logits = logits[a_idx, b_idx]  # (C*(C-1), T, T)
    n_pairs = int(a_idx.numel())

    # Apply same-core false-negative mask. core_mask is (T, T) bool where
    # core_mask[i, j] == True means tile j shares the same core as tile i and j != i.
    # The positive (diagonal j==i) is NEVER masked -- core_mask has False on diagonal.
    # The (T,T) mask is IDENTICAL for every (a,b) pair (shared tile list), so we
    # build it once and broadcast over the n_pairs dimension.
    masked_count = 0
    if core_mask is not None:
        # pair_logits: (n_pairs, T, T) -- mask shape (T, T) broadcasts.
        # core_bias generalises the hard mask to a REWEIGHTING: adding log(beta) to a
        # negative's logit multiplies its softmax weight by beta. beta=0 (-inf) is the
        # original hard mask; beta=1 (bias 0) is the untouched baseline; beta>1 UPWEIGHTS
        # same-tissue negatives, which the masking sweep identified as the ones carrying
        # the HEST-relevant gradient. One knob spans mask -> baseline -> emphasise.
        if core_bias is None or core_bias == float("-inf"):
            pair_logits = pair_logits.masked_fill(core_mask.unsqueeze(0), float("-inf"))
        else:
            pair_logits = pair_logits + core_mask.unsqueeze(0).to(pair_logits.dtype) * core_bias
        with torch.no_grad():
            # PER-ROW average, not the (T,T) total: each of the T rows of a pair block
            # has its own candidate set, and the sanity target is ~T/k.
            masked_count = float(core_mask.sum().item()) / float(n_tiles)

    flat = pair_logits.reshape(-1, n_tiles)  # (C*(C-1)*T, T)
    target = torch.arange(n_tiles, device=zn.device).repeat(n_pairs)
    # every pair contributes exactly T rows, so the flat mean IS the mean over pairs
    loss = F.cross_entropy(flat, target)

    with torch.no_grad():
        acc = (flat.argmax(dim=1) == target).float().mean().item()
    return loss, {
        "loss": float(loss.detach()),
        "top1": acc,
        "negatives_per_anchor": float(n_tiles - 1),
        "n_rows": float(n_pairs * n_tiles),
        "n_cond": float(n_cond),
        "n_tiles": float(n_tiles),
        "masked_per_row": float(masked_count),
    }


def grid_info_nce_split(
    z_dict: dict[str, torch.Tensor],
    n_cond: int,
    n_tiles: int,
    weights: dict[str, float],
    temperature: float = 0.07,
    grid_blocked: bool = False,
    pair_block: int = 8,
    core_mask: torch.Tensor | None = None,
    core_bias=None,
    center: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Split-head grid InfoNCE: ``sum_h w_h * grid_info_nce(head h)``.

    Args:
        z_dict:       ``{"cls": (C*T, D_cls), "mean": (C*T, D_mean)}`` projected embeddings.
        n_cond:       C, distinct conditions per batch.
        n_tiles:      T, shared tiles per condition.
        weights:      ``{"cls": w_cls, "mean": w_mean}``. Weights should sum to 1.0 to keep
                      gradient scale comparable to a single-head arm (same convention as
                      :func:`split_head_info_nce`). NOT renormalized here -- callers own it.
        temperature:  Softmax temperature (shared across heads).
        grid_blocked: If True use :func:`grid_info_nce_blocked` (recompute path).
        pair_block:   Block size when ``grid_blocked`` is True.

    Returns:
        ``(loss, metrics)`` where ``loss`` is the weighted sum and ``metrics`` carries
        per-head keys ``loss_cls``, ``top1_cls``, ``loss_mean``, ``top1_mean`` plus a
        top-level ``loss`` (the weighted total) and ``top1`` (weight-normalised mean).
    """
    active = [h for h in ("cls", "mean") if h in z_dict]
    if not active:
        raise ValueError("grid_info_nce_split: z_dict has no recognised head keys")

    grid_fn = grid_info_nce_blocked if grid_blocked else grid_info_nce

    total: torch.Tensor | None = None
    metrics: dict[str, float] = {}
    w_sum = sum(weights[h] for h in active)
    top1 = 0.0
    for h in active:
        if grid_blocked:
            loss_h, m_h = grid_info_nce_blocked(
                z_dict[h], n_cond, n_tiles, temperature, pair_block=pair_block
            )
        else:
            # PER-HEAD bias. HEST(phikon) reads the CLS half only, while RI reads
            # clsmean (both halves). beta>1 on cls buys HEST, beta<1 on mean buys RI,
            # so the two metrics get separate knobs instead of fighting over one.
            _cb = core_bias
            if isinstance(core_bias, dict):
                _cb = core_bias.get(h)
            loss_h, m_h = grid_info_nce(z_dict[h], n_cond, n_tiles, temperature, core_mask=core_mask, core_bias=_cb, center=center)
        term = weights[h] * loss_h
        total = term if total is None else total + term
        metrics[f"loss_{h}"] = m_h["loss"]
        metrics[f"top1_{h}"] = m_h["top1"]
        metrics[f"weight_{h}"] = float(weights[h])
        top1 += weights[h] * m_h["top1"]
        # Propagate negatives/row count from the last head (same geometry for all heads).
        for k in ("negatives_per_anchor", "n_rows", "n_cond", "n_tiles"):
            metrics[k] = m_h[k]
    metrics["loss"] = float(total.detach())
    metrics["top1"] = top1 / max(w_sum, 1e-12)
    metrics["n_heads"] = float(len(active))
    return total, metrics


# --------------------------------------------------------------------------------------
# Split loss heads: CLS and mean scored SEPARATELY (PLAN.md 2's pooling mismatch)
# --------------------------------------------------------------------------------------
# Training pools ``clsmean`` = cat([CLS, mean(patch_tokens)]) -> 2048-d -> ONE
# ProjectionHead -> ONE InfoNCE. Two things that hides:
#
# 1. A single concat loss does not force BOTH halves to become invariant. The objective
#    can satisfy itself through whichever half is easier, and the eval pooling DISAGREES
#    with the training pooling -- HEST and THUNDER-on-phikon-v2 read CLS only. So a run
#    can look converged on the training objective while the half the downstream probes
#    actually read barely moved.
# 2. ``mean`` is linear: d(mean)/d(t_i) = (1/N) I, so the direct gradient reaching every
#    patch token is the IDENTICAL vector. The loss can translate the token cloud but
#    cannot express a preference about the tokens' relative arrangement.
#
# The split objective is
#
#     L = w_cls * InfoNCE(proj_cls(cls_vec)) + w_mean * InfoNCE(proj_mean(mean_vec))
#
# with a SEPARATE ProjectionHead per pool, each ``hidden``-d in (1024 on phikon-v2), not
# 2048.
#
# WHY THE WEIGHTS DEFAULT TO 0.5 / 0.5 AND NOT 1.0 / 1.0
# ------------------------------------------------------
# Loss scale is a CONTROLLED VARIABLE here. At 1.0/1.0 the total gradient magnitude is
# roughly twice the single-head baseline's, which for AdamW-with-warmup on the same LR
# schedule is a silent learning-rate change -- and the experiment would then be measuring
# "structural split PLUS 2x LR" against "baseline". 0.5/0.5 keeps the total comparable to
# the single concat head, so the arm isolates the structural change. It is also exactly
# the convex combination that makes the degenerate case check out: feed the SAME vector to
# two heads holding the SAME weights and 0.5*L + 0.5*L == L, which is gate G5.
#
# WHY A ZERO-WEIGHT HEAD IS REMOVED RATHER THAN MULTIPLIED BY ZERO
# ----------------------------------------------------------------
# ``--cls-weight 1.0 --mean-weight 0.0`` must be a genuinely single-head run. A head that
# is built and run at weight 0 still (a) burns a projector forward on every step and
# (b) -- the part that actually corrupts the arm -- updates its ``nn.BatchNorm1d`` RUNNING
# STATS every step, because BatchNorm's running-mean/var update happens in the forward
# pass and is not gated by the loss weight. So ``build_split_head_names`` drops any
# zero-weight head and the encoder never constructs it; ``forward_split`` then cannot run
# what does not exist.


def build_split_head_names(cls_weight: float, mean_weight: float) -> tuple[str, ...]:
    """Which heads to BUILD, given the weights. A zero-weight head is not built at all."""
    if cls_weight < 0.0 or mean_weight < 0.0:
        raise ValueError(
            f"split-head weights must be >= 0, got cls={cls_weight} mean={mean_weight}"
        )
    names = tuple(
        n for n, w in (("cls", cls_weight), ("mean", mean_weight)) if w > 0.0
    )
    if not names:
        raise ValueError(
            "both split-head weights are 0: there would be no loss at all. Set at least "
            "one of --cls-weight / --mean-weight above 0."
        )
    return names


def assert_split_head_inputs(
    parts: dict[str, torch.Tensor], min_rel_distance: float = 1e-3
) -> dict[str, float]:
    """THE assertion this feature lives or dies on: the two heads get DIFFERENT inputs.

    If a wiring bug handed both heads the same pooled vector -- ``embed()`` twice, a
    copy-pasted key, a ``parts["cls"]`` that is really the concat -- then
    ``L_cls + L_mean`` is just ``2x`` one loss. That is a perfectly plausible falling
    curve which measures nothing at all, and NO shape check catches it: both vectors are
    ``(B, hidden)`` either way.

    So it is asserted numerically, on the real batch, every step (a few microseconds on
    ``(B, 1024)``), exactly like ``assert_same_condition_negatives``.

    Returns the measured separation so it lands in ``history.json`` and the claim is
    auditable after the fact rather than only at crash time.
    """
    if "cls" not in parts or "mean" not in parts:
        # Defensive only. ``embed_parts`` always returns BOTH pools -- pooling is free
        # next to the backbone forward -- so the train loop records this diagnostic on
        # every arm, single-head ones included. That is deliberate: it makes the measured
        # pool separation comparable across the three arms instead of existing only on the
        # one arm that could have the wiring bug.
        return {}
    # ``"pool"`` is present only when an alternative pooling (gem/attn/lse) replaced the
    # mean for the LOSS head; ``"mean"`` then still carries the true arithmetic mean,
    # because ``pool_from_parts`` must keep reassembling the eval-time clsmean for the
    # retention term. The wiring assertion has to run on what the head ACTUALLY received,
    # so it reads "pool" when it exists -- otherwise a gem head fed a copy of CLS would
    # sail past a check that was busy comparing CLS to an unused mean.
    cls_vec = parts["cls"].detach().float()
    mean_vec = parts.get("pool", parts["mean"]).detach().float()
    if cls_vec.shape != mean_vec.shape:
        raise ValueError(
            f"cls/mean head inputs differ in shape: {tuple(cls_vec.shape)} vs "
            f"{tuple(mean_vec.shape)}; both heads take one pool, i.e. (B, hidden)"
        )
    scale = 0.5 * (cls_vec.norm(dim=-1) + mean_vec.norm(dim=-1)).clamp_min(1e-12)
    rel = float(((cls_vec - mean_vec).norm(dim=-1) / scale).mean())
    cos = float(F.cosine_similarity(cls_vec, mean_vec, dim=-1).mean())
    if rel < min_rel_distance:
        raise ValueError(
            "SPLIT-HEAD WIRING BUG: the cls and mean heads received effectively the SAME "
            f"input (mean relative distance {rel:.3e} < {min_rel_distance:.0e}, mean "
            f"cosine {cos:.6f}). L_cls + L_mean would then be 2x one loss -- a falling "
            "curve that measures nothing. Check that forward_split() pools the token "
            "sequence twice (cls = tokens[:, 0], mean = tokens[:, num_prefix_tokens:]"
            ".mean(1)) rather than handing both heads embed()."
        )
    return {"split_input_rel_distance": rel, "split_input_cosine": cos}


def split_head_info_nce(
    anchor_z: dict[str, torch.Tensor],
    positive_z: dict[str, torch.Tensor],
    group_id: torch.Tensor,
    weights: dict[str, float],
    temperature: float = 0.07,
    symmetric: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``sum_h w_h * masked_info_nce(head h)``, with every head logged separately.

    Per-head ``loss_<h>`` and ``top1_<h>`` go into ``history.json`` alongside the total.
    Without them the three arms are uninterpretable: a falling total tells you nothing
    about which pool moved.

    ``loss`` is the WEIGHTED TOTAL (the quantity that is actually optimised) and ``top1``
    is the weight-normalised mean of the per-head top-1s, so on a single-head arm both
    reduce to that head's own numbers and the column keeps its usual meaning.
    """
    if set(anchor_z) != set(positive_z):
        raise ValueError(f"head mismatch: anchors {sorted(anchor_z)} vs positives {sorted(positive_z)}")
    active = [h for h in ("cls", "mean") if h in anchor_z]
    if not active:
        raise ValueError("no split heads to score")

    total: torch.Tensor | None = None
    metrics: dict[str, float] = {}
    w_sum = sum(weights[h] for h in active)
    top1 = 0.0
    for h in active:
        loss_h, m_h = masked_info_nce(
            anchor_z[h], positive_z[h], group_id, temperature, symmetric
        )
        term = weights[h] * loss_h
        total = term if total is None else total + term
        metrics[f"loss_{h}"] = m_h["loss"]
        metrics[f"top1_{h}"] = m_h["top1"]
        metrics[f"weight_{h}"] = float(weights[h])
        metrics["negatives_per_anchor"] = m_h["negatives_per_anchor"]
        top1 += weights[h] * m_h["top1"]
    metrics["loss"] = float(total.detach())
    metrics["top1"] = top1 / max(w_sum, 1e-12)
    metrics["n_heads"] = float(len(active))
    return total, metrics


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
    #: GRID sampler (:mod:`waivphaet.data.grid`). False = the pair sampler above, which is
    #: the path every published number was produced on and is left bit-identical.
    #: True swaps in C x T shared-tile batches: C*T images/step (no separate positive
    #: tensor), T-1 negatives per row, C*(C-1)*T query rows. Mutually exclusive with
    #: n_groups/group_size, which are ignored (and left at their defaults) when grid=True.
    grid: bool = False
    grid_conditions: int = 0  # C -- distinct conditions per batch; 0 when grid is off
    grid_tiles: int = 0  # T -- shared tiles per condition; 0 when grid is off
    #: Micro-chunk for the grid forward pass. 0 = one forward over all C*T images.
    #:
    #: This is a MEMORY knob, not a maths knob: the chunks stay in ONE autograd graph and
    #: the loss still sees every one of the C*T embeddings at once, so the objective is
    #: unchanged (a ViT is per-image; there is no cross-batch normalisation to break).
    #: It exists because gradient checkpointing's cost is not the stored inputs -- those
    #: scale with C*T either way -- but the TRANSIENT buffer when a block is recomputed in
    #: backward, which scales with the size of one forward. The pair path gets a 2x
    #: smaller transient for free by running anchors and positives as two separate
    #: forwards of B/2; the grid path has a single tensor and must be told. Measured:
    #: 2400 images in one forward OOMs an 80 GiB H100 (7.21 GiB recompute allocation on
    #: top of 70.4 GiB already live), where 2 x 1200 fits in 65 GiB.
    #:
    #: Caveat, measured rather than assumed: equivalent is not BIT-identical. A different
    #: chunk size is a different GEMM shape, so cuBLAS picks a different kernel and the
    #: reductions associate differently -- agreement is ~1e-7, not exact. At a FIXED chunk
    #: size a run still replays exactly, which is why this is recorded in config.json and
    #: must not be varied within a comparison.
    grid_forward_chunk: int = 0
    # ACTIVATION OFFLOAD. Independent of grid_forward_chunk and composes with it:
    # the chunk shrinks the per-block RECOMPUTE buffer, this moves the tensors
    # gradient checkpointing SAVES (the per-block inputs, the bulk of the residual
    # footprint) out to pinned host RAM until backward asks for them.
    #
    # This is EXACT, not an approximation: torch.autograd.graph.save_on_cpu is a
    # save/restore hook pair, so the same bytes come back and the objective is
    # unchanged. It buys memory at roughly 40% throughput (PCIe round trips).
    # Default OFF so no existing arm changes behaviour.
    activation_offload: bool = False
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
    #: Resume: absolute path to a ``step_*`` checkpoint dir belonging to a PREVIOUS attempt
    #: of this same job, or None (the default -- every fresh run starts at step 0).
    #: Set by ``train_lora.py --resume-from-prior-attempt``, which discovers it; a preempted
    #: arm otherwise restarts at step 0 and re-derives a curve it already has (10, 8 and 7
    #: requeues on arms a/b/c on 2026-08-14, ~25 attempts, none reaching step 1500).
    #: Exempt from the same-config guard for the obvious reason that it names the thing
    #: being resumed FROM and so must differ between attempts.
    resume_from: str | None = None
    #: Compute the grid InfoNCE with the blocked/recompute path instead of materialising the
    #: full (C,C,T,T) logit tensor. EXACT -- same objective, same gradient to float32 noise
    #: (tests/test_blocked_loss.py) -- but it trades a second forward over the logits for
    #: bounded memory. Default OFF: at today's T=1975 the dense tensor is only ~187 MB, so
    #: the recompute is not yet worth paying for. Turn it on when T grows past ~6000, where
    #: the quadratic term starts to bind.
    grid_blocked_loss: bool = False
    #: Ordered condition pairs per block when ``grid_blocked_loss`` is on. Peak logit bytes
    #: are ``4 * grid_pair_block * T^2``, independent of C.
    grid_pair_block: int = 8
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
    #: Split loss heads (see the block comment above :func:`build_split_head_names`).
    #: False = the single concat projector, i.e. the path every published number was
    #: produced on, left bit-identical. True scores CLS and mean SEPARATELY, each through
    #: its own ``hidden``-d ProjectionHead.
    split_heads: bool = False
    #: Weights in ``L = w_cls * InfoNCE(cls) + w_mean * InfoNCE(mean)``. They sum to 1 by
    #: default ON PURPOSE: loss scale is a controlled variable, and 1.0/1.0 would double
    #: the total gradient magnitude relative to the single-head baseline, which is a
    #: silent LR change rather than the structural comparison this experiment is for.
    #: A weight of exactly 0 REMOVES its head (not built, never run, its BatchNorm running
    #: stats never updated) -- that is what makes the "only" arms actually single-head.
    cls_weight: float = 0.5
    mean_weight: float = 0.5
    #: Pooling for the NON-CLS split head. ``"mean"`` = the incumbent, bit-identical to
    #: the arm already running. See :mod:`waivphaet.models.pooling`: ``mean`` is linear, so
    #: its per-token gradient is the same vector for every token and the loss can only
    #: TRANSLATE the token cloud -- which THUNDER's biased segmentation decoder absorbs
    #: into its bias. ``gem`` / ``attn`` / ``lse`` have token-dependent gradients.
    #: Recorded here (not only in ``encoder``) so ``config.json`` names the objective.
    pool_head: str = "mean"
    #: Tissue fraction threshold for the GRID sampler tile pool. 0.0 = no filtering
    #: (identical to every prior run). When > 0, tiles from the reference condition
    #: with tissue fraction below this threshold are excluded from the sampler pool.
    min_tissue_frac: float = 0.0
    #: Mask same-core tiles from the negative set. When True, for every anchor at
    #: tile position i, all other tiles j where core[j] == core[i] are masked to
    #: -inf in the logits. The positive (j==i) is NEVER masked. Requires
    #: core_labels_path to be set. Default OFF; recorded in config.json.
    #: Tissue-concentrated sampling: tiles per step drawn from only this many cores,
    #: raising the same-tissue negative fraction. 0 = OFF. Recorded so runs are comparable.
    cores_per_batch: int = 0
    #: log(beta) added to same-tissue negative logits. None/-inf = hard mask (old
    #: behaviour), 0.0 = baseline, >0 = upweight the hard negatives. Recorded in config.
    same_core_logit_bias: float | None = None
    #: Subtract the batch mean before L2-normalising in the loss. Kills the free
    #: shared-shift direction a centered linear readout (HEST) cannot see.
    center_embeddings: bool = False
    #: Per-head same-core logit bias, e.g. {"cls": 2.0, "mean": -2.0}. Overrides the
    #: scalar same_core_logit_bias when set. Lets HEST (cls-only readout) and RI
    #: (clsmean readout) be tuned in opposite directions from one batch.
    same_core_logit_bias_cls: float | None = None
    same_core_logit_bias_mean: float | None = None
    mask_same_core: bool = False
    #: Absolute path to the per-tile core label array (int32, len 16278).
    core_labels_path: str = "/admin/home/ryan.kim/waiv/runs/.plism_core_labels.npy"
    #: Similarity masking. When > 0, a negative j is masked for anchor i iff the FROZEN
    #: base-model cosine between tiles i and j exceeds this threshold. Unlike
    #: mask_same_core (which masks a whole organ core indiscriminately), this targets
    #: only near-duplicates and leaves genuinely-different same-organ tiles as negatives.
    #: 0.0 = OFF. Mutually exclusive with mask_same_core.
    mask_sim_thresh: float = 0.0
    sim_emb_path: str = "/admin/home/ryan.kim/waiv/runs/.plism_ref_emb.npy"
    encoder: dict = field(default_factory=dict)


def _chunked_forward(model, images: torch.Tensor, chunk: int, offload: bool = False):
    """``model(images)`` split into micro-chunks, still as ONE autograd graph.

    Returns the same ``(pooled, projected)`` pair a single call would, so downstream
    indexing (the ``(C, T, D)`` view in :func:`grid_info_nce`) is unaffected.
    ``chunk <= 0`` or a chunk at least as large as the batch is a plain single forward.

    See ``TrainConfig.grid_forward_chunk`` for why this is needed: with gradient
    checkpointing the peak is the per-block RECOMPUTE buffer, which scales with one
    forward's size, not with the number of images kept in the graph.

    **Only the BACKBONE is chunked, and the projector then runs ONCE over the whole
    concatenated batch.** That split is not an optimisation, it is a correctness
    requirement: ``ProjectionHead`` contains ``nn.BatchNorm1d``, which couples the batch
    together. Chunking through it would compute BatchNorm statistics per chunk instead of
    over all C*T images -- a genuine change to the objective, silently different from the
    unchunked run -- and a trailing chunk of size 1 (2401 images at chunk 600) makes
    BatchNorm raise outright, which is how this was caught. The backbone is per-image
    (a ViT's LayerNorms are per-token), so splitting it is exact, and it is where
    essentially all of the activation memory lives anyway.

    ``offload=True`` additionally routes the backbone's SAVED tensors through
    ``torch.autograd.graph.save_on_cpu(pin_memory=True)``. Under gradient checkpointing
    those saved tensors are the per-block inputs, which dominate the residual footprint
    once the recompute buffer has been chunked down. The hook is exact -- identical bytes
    are restored in backward -- so this changes the memory profile and the step time, and
    nothing else. ``offload=False`` is byte-for-byte the original code path.
    """
    # save_on_cpu wraps the BACKBONE ONLY, for the same reason the chunking does: the
    # projector must stay one un-split, un-hooked call over the whole concatenated batch.
    # (Offload would not actually corrupt BatchNorm the way chunking does -- it is a pure
    # save/restore hook and moves no boundaries -- but keeping the projector outside the
    # context keeps "the projector runs exactly once, plainly" true by inspection, and its
    # saved tensors are megabytes, so there is nothing to win by offloading them.)
    ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
    if chunk <= 0 or chunk >= images.shape[0]:
        if not offload:
            return model(images)
        with ctx:
            emb = model.embed(images)
        return emb, model.projector(emb)
    with ctx:
        emb = torch.cat([model.embed(part) for part in images.split(chunk)], dim=0)
    return emb, model.projector(emb)


def _chunked_forward_split(
    model,
    images: torch.Tensor,
    chunk: int = 0,
    offload: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split-head variant of :func:`_chunked_forward`.

    Chunks the BACKBONE (per-image, exact) and then runs EACH HEAD'S projector ONCE over
    the full concatenated batch -- outside both the chunk loop and any offload context.
    This is the same correctness requirement as the single-head path: ``ProjectionHead``
    contains ``nn.BatchNorm1d``, which couples the batch together, so chunking through
    it would compute per-chunk BatchNorm statistics rather than over all C*T images.

    Returns ``(parts, zs)`` where:
      - ``parts``: ``{"cls": (C*T, hidden), "mean": (C*T, hidden)}`` from ``embed_parts``
      - ``zs``:    ``{"cls": (C*T, D), "mean": (C*T, D)}`` projected, one key per built head

    ``anchor_emb`` for the retention term should be reconstructed via
    ``model.pool_from_parts(parts)`` -- NOT from any projected vector.
    """
    ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
    must_chunk = 0 < chunk < images.shape[0]
    if getattr(model, "pool_head", None) is not None:
        # A learned pool head (GeM / LSE / attention) pools over the raw PATCH-TOKEN
        # sequence, which ``embed_parts`` has already reduced away -- so the code below
        # genuinely cannot serve it.  ``model.forward_split`` keeps the tokens and still
        # runs each projector exactly ONCE over the full C*T batch (see its body), i.e.
        # it satisfies the same BatchNorm-coupling requirement this function does.  So
        # when no chunking was asked for, delegate rather than refuse: the guard's own
        # advice ("use model.forward_split() directly (no chunking)") is exactly this.
        if must_chunk:
            raise RuntimeError(
                "_chunked_forward_split cannot chunk with a non-default pool_head "
                "(GeM / LSE / attention) because chunking discards the raw token "
                "sequence the pool head needs. Re-run with --grid-forward-chunk 0 "
                "(model.forward_split() is then used directly), or use pool_head=mean."
            )
        # ``ctx`` covers the projectors here as well as the backbone.  Unlike chunking,
        # save_on_cpu is an exact save/restore hook -- it moves no batch boundaries and
        # cannot perturb BatchNorm statistics -- so this is byte-identical maths, only a
        # different memory/step-time profile.
        with ctx:
            return model.forward_split(images)
    if not must_chunk:
        with ctx:
            raw_parts = model.embed_parts(images)
    else:
        # Chunk the backbone; each chunk returns {"cls": (chunk, H), "mean": (chunk, H)}.
        chunk_parts_list: list[dict[str, torch.Tensor]] = []
        with ctx:
            for part in images.split(chunk):
                chunk_parts_list.append(model.embed_parts(part))
        raw_parts = {
            k: torch.cat([cp[k] for cp in chunk_parts_list], dim=0)
            for k in chunk_parts_list[0]
        }
    # Projectors run ONCE over the whole concatenated batch -- outside the chunk loop and
    # outside any offload context, exactly as the single-head path does.
    # Every non-default ``pool_head`` already returned above (delegated when unchunked,
    # raised when chunking was genuinely required), so by here the head input is the
    # arithmetic mean from ``embed_parts`` and needs no token sequence.
    head_in = raw_parts
    zs = {name: model.projectors[name](head_in[name]) for name in model.split_heads}
    return raw_parts, zs


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
        if cfg.grid:
            # Grid batches carry one image tensor and their own geometry; the held-out
            # loader may run a SMALLER C than training (there are fewer held-out
            # conditions), so read C and T off the batch rather than off cfg.
            images = batch["image"].to(device, non_blocking=True)
            n_cond, n_tiles = int(batch["n_cond"]), int(batch["n_tiles"])
            if cfg.split_heads:
                _, gz = _chunked_forward_split(model, images, cfg.grid_forward_chunk)
                _, m = grid_info_nce_split(
                    gz, n_cond, n_tiles,
                    {"cls": cfg.cls_weight, "mean": cfg.mean_weight},
                    cfg.temperature,
                    grid_blocked=cfg.grid_blocked_loss,
                    pair_block=cfg.grid_pair_block,
                )
            else:
                # same memory guard as the train step; harmless under no_grad
                _, gz = _chunked_forward(model, images, cfg.grid_forward_chunk)
                _, m = grid_info_nce(gz, n_cond, n_tiles, cfg.temperature)
            tot["loss"] += m["loss"]
            tot["top1"] += m["top1"]
            n += 1
            continue
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        gid = batch["group_id"].to(device)
        if cfg.split_heads:
            _, az = model.forward_split(anchor)
            _, pz = model.forward_split(positive)
            _, m = split_head_info_nce(
                az, pz, gid,
                {"cls": cfg.cls_weight, "mean": cfg.mean_weight},
                cfg.temperature, cfg.symmetric,
            )
        else:
            _, az = model(anchor)
            _, pz = model(positive)
            _, m = masked_info_nce(az, pz, gid, cfg.temperature, cfg.symmetric)
        tot["loss"] += m["loss"]
        tot["top1"] += m["top1"]
        n += 1
    model.train()
    return {f"heldout_{k}": v / max(n, 1) for k, v in tot.items()}


def save_projectors(model, out: Path) -> dict:
    """Write the projector artifact(s), single-head or split, and return the manifest.

    Single head: ``projector.pt``, exactly as before -- byte-for-byte the same call.

    Split heads: one file PER HEAD (``projector_cls.pt`` / ``projector_mean.pt``), because
    a single ``projector.pt`` would either silently save one of the two or crash on a
    ModuleDict state_dict that no reader expects.

    A ``projector.pt`` is written as well, aliasing the first built head. That is not
    decoration: ``build_model`` in ``scripts/extract_pathorob_features.py`` loads
    ``projector.pt`` UNCONDITIONALLY on the LoRA-adapter path, so omitting it turns the
    eval follower into a FileNotFoundError. With it present, that reader finds a 1024-d
    input against a 2048-d ``clsmean`` eval and prints its "skipping projector" line --
    which is correct and benign: every eval path reads ``model.embed()`` and never a
    projector, and the LoRA backbone weights are what carry the fine-tune.
    ``projector_heads.json`` records which head the alias is, so the artifact is
    unambiguous rather than merely present.
    """
    heads = tuple(getattr(model, "split_heads", ()) or ())
    if not heads:
        torch.save(model.projector.state_dict(), out / "projector.pt")
        return {"split_heads": False, "heads": [], "projector_pt": "projector.pt"}
    _save_pool_head(model, out)
    for name in heads:
        torch.save(model.projectors[name].state_dict(), out / f"projector_{name}.pt")
    alias = heads[0]
    torch.save(model.projectors[alias].state_dict(), out / "projector.pt")
    manifest = {
        "split_heads": True,
        "heads": list(heads),
        "files": {n: f"projector_{n}.pt" for n in heads},
        "projector_pt_alias": alias,
        "note": (
            "projector.pt is a COPY of projector_%s.pt, present only so readers that load "
            "it unconditionally do not crash. It is a one-pool head, so its input width is "
            "`hidden` (1024 on phikon-v2) and any clsmean (2048-d) eval will -- correctly "
            "-- skip loading it. Eval reads model.embed(); the projector is training-only."
        ) % alias,
    }
    pool_name = str(getattr(model, "pool_head_name", "mean"))
    manifest["pool_head"] = pool_name
    if getattr(model, "pool_head", None) is not None:
        manifest["pool_head_pt"] = "pool_head.pt"
    (out / "projector_heads.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _save_pool_head(model, out: Path) -> None:
    """Write ``pool_head.pt`` when the non-CLS head pools with learnable parameters.

    GeM's ``p``, LSE's ``tau`` and the attention query/key are TRAINED, so a checkpoint
    without them cannot be resumed or re-run faithfully -- and, unlike the projector, they
    are not reconstructible from anything else in the directory. Only written when the
    module exists, so the ``--pool-head mean`` and single-head artifacts are byte-for-byte
    what they always were.

    No eval path reads this file: every eval reads ``model.embed()``, whose pooling is the
    protocol constant ``clsmean``. Both probe readers glob nothing and load by name, so an
    extra file is inert to them (verified in ``test_probe_readers_ignore_pool_head_pt``).
    """
    pool = getattr(model, "pool_head", None)
    if pool is None:
        return
    sd = pool.state_dict()
    if sd:
        torch.save(sd, out / "pool_head.pt")


def load_projectors(model, out: Path) -> None:
    """Inverse of :func:`save_projectors`: restore every head from ``out``. Round-trip."""
    heads = tuple(getattr(model, "split_heads", ()) or ())
    if not heads:
        model.projector.load_state_dict(torch.load(out / "projector.pt", map_location="cpu"))
        return
    for name in heads:
        path = out / f"projector_{name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint {out} has no {path.name}: it was not written by a run with "
                f"the {name} head built"
            )
        model.projectors[name].load_state_dict(torch.load(path, map_location="cpu"))
    pool = getattr(model, "pool_head", None)
    if pool is not None and pool.state_dict():
        path = out / "pool_head.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint {out} has no pool_head.pt, but this model pools with "
                f"{getattr(model, 'pool_head_name', '?')!r}, whose parameters are TRAINED "
                "and are not recoverable from anything else in the directory"
            )
        pool.load_state_dict(torch.load(path, map_location="cpu"))


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg: TrainConfig,
    metrics: dict,
    *,
    scaler=None,
    rng_state: dict | None = None,
) -> Path:
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

    save_projectors(model, out)
    payload = {"optimizer": optimizer.state_dict(), "step": step}
    if scaler is not None:
        # GradScaler carries a live scale factor and a growth counter. Under bfloat16 (the
        # default) the scaler is DISABLED and this is inert, but under float16 restoring a
        # run without it restarts at the initial scale of 65536 and burns a handful of
        # skipped steps re-converging -- which would make a "resumed" curve quietly differ
        # from a continuous one. Cheap to save, so always save it.
        payload["scaler"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng"] = rng_state
    torch.save(payload, out / "optim.pt")
    # metrics.json is the completeness sentinel -- written LAST.
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out


def capture_rng_state() -> dict:
    """Snapshot every RNG the training step could consume.

    Today the grid/pair training step consumes NO torch RNG: there is no stochastic
    augmentation (the positives are real registered pairs, not synthetic views -- see
    :mod:`waivphaet.data.pairs`), and ``lora_dropout`` defaults to 0.0, at which peft
    substitutes ``nn.Identity`` rather than a dropout layer. The batch PLAN sequence is
    numpy, seeded per epoch inside the sampler, and is restored by rewinding the sampler
    rather than from here.

    It is captured anyway because that "consumes no RNG" property is an invariant nobody
    is currently asserting, and the day someone adds dropout or a random augmentation the
    resumed curve would silently diverge from a continuous one. Restoring costs nothing
    and removes the trap. ``assert_resume_rng_is_inert`` is the matching check.
    """
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Inverse of :func:`capture_rng_state`. Missing keys are skipped, not an error."""
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu().to(torch.uint8))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        saved = state["torch_cuda"]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in saved])


#: Config fields that are ALLOWED to differ between a prior attempt and the resuming one.
#: Everything else differing means the two attempts are not the same experiment, and
#: continuing would produce a single curve stitched from two different objectives.
RESUME_CONFIG_EXEMPT = frozenset({"out_dir", "resume_from", "max_steps"})


def assert_resume_config_matches(prior_cfg: dict, cfg: "TrainConfig") -> None:
    """Refuse to resume across a config change.

    This is the guard that makes resume safe to leave on. A preempted arm restarts with
    whatever sbatch/argv the queue hands it; if that differs at all from the attempt whose
    weights we are about to adopt, the resulting ``ri_curve.json`` would be one continuous
    line drawn from two DIFFERENT experiments -- exactly the class of silent-correctness
    failure this project keeps hitting. Fail loudly instead.

    ``max_steps`` is exempt but checked separately below: shortening the horizon is a real
    change (it reshapes the cosine LR schedule for every remaining step), so it is only
    tolerated when identical -- the exemption exists so the error message can be specific.
    """
    current = asdict(cfg)
    diffs = {
        k: (prior_cfg.get(k, "<missing>"), current[k])
        for k in current
        if k not in RESUME_CONFIG_EXEMPT and prior_cfg.get(k, "<missing>") != current[k]
    }
    if prior_cfg.get("max_steps") != current["max_steps"]:
        diffs["max_steps"] = (prior_cfg.get("max_steps", "<missing>"), current["max_steps"])
    if diffs:
        lines = "\n".join(f"    {k}: prior={p!r} current={c!r}" for k, (p, c) in sorted(diffs.items()))
        raise ValueError(
            "refusing to resume: the prior attempt's config.json differs from this run's "
            f"config in {len(diffs)} field(s):\n{lines}\n"
            "Resuming across a config change stitches one curve out of two different "
            "experiments. Re-run from step 0, or fix the launcher so the attempts agree."
        )


def load_checkpoint_weights(model, ckpt: Path) -> None:
    """Restore the TRAINED weights of *ckpt* into an already-built *model*.

    The inverse of the weight half of :func:`save_checkpoint`. The model must already have
    been built with the same encoder config (LoRA rank, split heads, pool head) -- this
    loads tensors into existing modules, it does not reshape anything, and a mismatch
    raises out of ``load_state_dict`` rather than loading silently.

    LoRA mode restores the adapter tensors IN PLACE via peft's
    ``set_peft_model_state_dict`` rather than re-wrapping the backbone, so the module tree
    the optimizer's saved state refers to is left untouched. Re-wrapping would produce
    fresh parameter objects and the restored AdamW moments would land on the wrong tensors.
    """
    if getattr(model.cfg, "use_lora", False):
        adapter_dir = ckpt / "adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"{ckpt} has no adapter/ directory, but this model was built with LoRA"
            )
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model.backbone, sd)
        missing = list(getattr(result, "unexpected_keys", []) or [])
        if missing:
            raise RuntimeError(
                f"resuming from {ckpt}: peft rejected {len(missing)} adapter key(s) "
                f"({missing[:5]}). The built LoRA config does not match the checkpoint's."
            )
    else:
        from safetensors.torch import load_file
        path = ckpt / "backbone.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"{ckpt} has no backbone.safetensors, but this model was built for full FT"
            )
        model.backbone.load_state_dict(load_file(str(path)))
    load_projectors(model, ckpt)


def find_resumable_checkpoint(out_dir: Path) -> Path | None:
    """Highest COMPLETE ``step_*`` checkpoint under *out_dir*, or None.

    "Complete" means ``metrics.json`` exists: :func:`save_checkpoint` writes it last
    precisely so a checkpoint half-flushed by a preemption SIGKILL is not mistaken for a
    finished one. ``eval_checkpoints.discover`` uses the same sentinel.
    """
    if not out_dir.is_dir():
        return None
    done = [
        d for d in out_dir.glob("step_*")
        if d.is_dir() and (d / "metrics.json").exists() and (d / "optim.pt").exists()
    ]
    if not done:
        return None
    return max(done, key=lambda d: int(d.name.split("_")[1]))


#: Matches the ``.r<N>`` attempt suffix that gridcmp2.sbatch appends from
#: ``$SLURM_RESTART_COUNT`` (commit 42990ba): ``RUN_NAME=gridcmp2-<arm>-<jobid>[.r<N>]``.
_ATTEMPT_SUFFIX = re.compile(r"\.r(\d+)$")


def attempt_number(d: Path) -> int:
    """Attempt index of a run dir: ``<base>`` is 0, ``<base>.r<N>`` is N."""
    m = _ATTEMPT_SUFFIX.search(Path(d).name)
    return int(m.group(1)) if m else 0


def prior_attempt_dirs(out_dir: Path) -> list[Path]:
    """Sibling run dirs that are EARLIER attempts of the same job, oldest attempt first.

    Attempts are ``<base>`` (attempt 0) and ``<base>.r<N>`` (requeue N), where ``<base>``
    already contains ``$SLURM_JOB_ID``. SLURM keeps the job id across a requeue, so this
    glob is scoped to THIS job and cannot stray to a different arm or an unrelated run.
    *out_dir* itself is always excluded -- resume READS prior attempts and WRITES its own.
    """
    out_dir = Path(out_dir)
    base = _ATTEMPT_SUFFIX.sub("", out_dir.name)
    here = out_dir.resolve()
    found = [
        d for d in out_dir.parent.glob(f"{base}*")
        if d.is_dir()
        and d.resolve() != here
        and (d.name == base or _ATTEMPT_SUFFIX.fullmatch(d.name[len(base):]) is not None)
        and attempt_number(d) < attempt_number(out_dir)
    ]
    return sorted(found, key=attempt_number)


def find_prior_attempt_checkpoint(out_dir: Path) -> Path | None:
    """Best checkpoint from a PREVIOUS attempt of the same job, or None.

    Attempt directories are siblings named ``<base>`` (attempt 0) and ``<base>.r<N>``
    (requeue N), where ``<base>`` already contains ``$SLURM_JOB_ID`` -- and SLURM keeps the
    job id across a requeue, so this glob is scoped to THIS job and cannot pick up a
    different arm or an unrelated run.

    Returns the highest complete ``step_*`` across all prior attempts, preferring the later
    attempt on a tie (its weights are the ones that trained most recently). Returns None
    when there is no prior attempt or none of them reached a checkpoint -- that is the
    ordinary attempt-0 case and the caller should start from step 0, not fail.
    """
    best: tuple[int, int, Path] | None = None
    for d in prior_attempt_dirs(out_dir):
        ck = find_resumable_checkpoint(d)
        if ck is None:
            continue
        key = (int(ck.name.split("_")[1]), attempt_number(d))
        if best is None or key > best[:2]:
            best = (key[0], key[1], ck)
    return best[2] if best else None


def _should_checkpoint(step: int, cfg: TrainConfig) -> bool:
    """Determine whether to checkpoint at *step* using schedule or periodic interval."""
    if cfg.ckpt_schedule is not None:
        return step in cfg.ckpt_schedule
    return step % cfg.ckpt_every == 0


def _resume_into(
    ckpt: Path,
    *,
    cfg: TrainConfig,
    model,
    optimizer,
    scaler,
    train_loader,
    history: list[dict],
) -> int:
    """Restore a prior attempt's checkpoint into this attempt and return the step to resume at.

    READS *ckpt* (a ``step_*`` dir belonging to a PREVIOUS attempt) and writes nothing to
    it. The caller's ``cfg.out_dir`` is this attempt's own fresh ``.r<N>`` directory, and
    every checkpoint from here lands there. That separation is the correctness property
    commit 42990ba bought -- resume must make requeue CHEAP without making it UNSAFE.

    What has to be restored for the resumed curve to equal the continuous one, and where
    each piece comes from:

    * model weights  -- ``load_checkpoint_weights`` (adapter/ or backbone.safetensors,
      plus projector(s) and pool head).
    * optimizer state -- ``optim.pt``; AdamW's exp_avg/exp_avg_sq and per-param step.
      Dropping these is the classic silent resume bug: the first few hundred steps after
      a restart take effectively unconditioned updates.
    * LR schedule    -- NO state to restore. ``cosine_lr(step, cfg)`` is a pure function of
      the step and the config, so restoring the step restores the LR exactly.
    * step counter   -- ``optim.pt["step"]``.
    * data order     -- the batch plan sequence is a pure function of (seed, epoch,
      position), so the sampler is rewound to position ``step % batches_per_epoch``
      instead of the loader being spun forward (which would re-read every skipped image).
    * RNG            -- restored when present; see ``capture_rng_state`` for why it is
      currently inert but still worth carrying.
    * history        -- carried forward so ``history.json`` is one continuous record.
    """
    if not ckpt.is_dir():
        raise FileNotFoundError(f"--resume-from checkpoint does not exist: {ckpt}")
    if not (ckpt / "metrics.json").exists():
        raise ValueError(
            f"{ckpt} has no metrics.json, so it was half-written when its job died. "
            "Resuming from a torn checkpoint is exactly the failure this sentinel exists "
            "to prevent -- pick an earlier step."
        )

    prior_attempt = ckpt.parent
    if prior_attempt.resolve() == Path(cfg.out_dir).resolve():
        raise ValueError(
            f"refusing to resume from {ckpt}: it lives in this attempt's OWN out_dir "
            f"({cfg.out_dir}). Resume must READ a prior attempt and WRITE a fresh one."
        )
    prior_cfg_path = prior_attempt / "config.json"
    if not prior_cfg_path.exists():
        raise FileNotFoundError(
            f"{prior_attempt} has no config.json, so the guard that this is the same "
            "experiment cannot run. Refusing to resume."
        )
    assert_resume_config_matches(json.loads(prior_cfg_path.read_text()), cfg)

    load_checkpoint_weights(model, ckpt)

    blob = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    step = int(blob["step"])
    if step >= cfg.max_steps:
        raise ValueError(
            f"{ckpt} is already at step {step} of max_steps {cfg.max_steps}: "
            "there is nothing left to resume."
        )
    if cfg.grad_accum > 1 and step % cfg.grad_accum != 0:
        # A checkpoint taken mid-accumulation has un-applied gradients that were never
        # saved. Resuming there drops a fraction of one update -- small, but it is a
        # silent difference from a continuous run, so refuse rather than absorb it.
        raise ValueError(
            f"{ckpt} is at step {step}, which is not a multiple of grad_accum "
            f"{cfg.grad_accum}: that checkpoint has un-applied gradients that were never "
            "saved. Resume from a step aligned to grad_accum."
        )
    if scaler is not None and "scaler" in blob:
        scaler.load_state_dict(blob["scaler"])
    if "rng" in blob:
        restore_rng_state(blob["rng"])

    # Rewind the batch-plan sequence rather than replaying images through the loader.
    sampler = getattr(train_loader, "batch_sampler", None) or getattr(train_loader, "sampler", None)
    if not hasattr(sampler, "set_start_index"):
        raise TypeError(
            f"cannot resume: the train loader's sampler ({type(sampler).__name__}) has no "
            "set_start_index, so the data stream cannot be rewound and the resumed run "
            "would silently re-see the same batches it already trained on."
        )
    sampler.set_start_index(step % sampler.batches_per_epoch)

    prior_history = prior_attempt / "history.json"
    if prior_history.exists():
        history.extend(
            row for row in json.loads(prior_history.read_text())
            if int(row.get("step", 0)) <= step
        )

    print(
        f"[resume] step {step} <- {ckpt}\n"
        f"[resume]   optimizer state restored ({len(blob['optimizer'].get('state', {}))} tensors)\n"
        f"[resume]   sampler rewound to plan {step % sampler.batches_per_epoch}\n"
        f"[resume]   history carried forward: {len(history)} row(s)\n"
        f"[resume]   writing to {cfg.out_dir} (prior attempt is read-only)",
        flush=True,
    )
    return step


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
    if cfg.split_heads:
        # Fail before any compute if the config and the model disagree about which heads
        # exist -- a mismatch here is exactly the "plausible curve measuring nothing"
        # failure mode this feature is guarding against.
        want = build_split_head_names(cfg.cls_weight, cfg.mean_weight)
        got = tuple(getattr(model, "split_heads", ()) or ())
        if got != want:
            built = list(got) if got else "the single concat projector"
            raise ValueError(
                f"cls_weight={cfg.cls_weight} / mean_weight={cfg.mean_weight} require the "
                f"heads {list(want)}, but the encoder was built with {built}. A "
                "zero-weight head must not be BUILT (its BatchNorm running stats would "
                "still update every step), so the two must agree exactly."
            )
    # Same rule for the pooling: a config that says "gem" against an encoder that pools
    # with the mean is a run whose name, config.json and objective all disagree, and
    # nothing downstream could ever tell. Fail before any compute.
    got_pool = str(getattr(model, "pool_head_name", "mean"))
    if got_pool != cfg.pool_head:
        raise ValueError(
            f"cfg.pool_head={cfg.pool_head!r} but the encoder pools with {got_pool!r}. "
            "The pooling is a property of the built encoder (it owns the learnable p / "
            "tau / attention query), so the two must be built from the same value."
        )
    if cfg.pool_head != "mean" and not cfg.split_heads:
        raise ValueError(
            f"pool_head={cfg.pool_head!r} requires split_heads: with the single concat "
            "projector the pooled vector IS the eval-time embedding, and the eval pooling "
            "is a protocol constant. The alternative poolings are TRAINING-TIME loss heads."
        )
    use_retention = cfg.retention_kl_weight > 0.0
    if use_retention and bool(getattr(model, "infer_pool_head", False)):
        # REJECTED COMBINATION -- the retention/KL term is not well defined when the
        # learned pool head is spliced into the mean slot at inference time. Two
        # independent defects, either of which silently corrupts the objective:
        #
        #   1. STUDENT/TEACHER POOLING MISMATCH. The student anchor is
        #      ``model.pool_from_parts(parts)`` where ``parts`` come from
        #      ``_pool_parts``, which is ALWAYS the arithmetic ``patches.mean(dim=1)``.
        #      The teacher goes through ``retention_teacher_embed`` -> ``model.embed()``
        #      -> ``_pool``, which DOES honour ``infer_pool_head`` and therefore pools
        #      with the learned head. Student and teacher would be compared across two
        #      different pooling operators, so the KL would penalise the pooling gap
        #      rather than representational drift.
        #
        #   2. NON-STATIONARY "FROZEN" TEACHER. ``frozen_teacher`` freezes only
        #      ``model.backbone`` (via ``disable_adapter()``). The pool head is a
        #      SEPARATE nn.Module and is NOT covered by that context manager, so the
        #      supposedly frozen teacher would pool with the LIVE, still-training pool
        #      head. The retention target would move every optimiser step, which is
        #      exactly what a retention anchor must not do.
        #
        # Fixing the asymmetry properly means threading the pool head through both
        # _pool_parts and the teacher freeze; until that exists, refuse the combination
        # loudly instead of training against a meaningless anchor.
        raise ValueError(
            "infer_pool_head=True is incompatible with retention_kl_weight="
            f"{cfg.retention_kl_weight!r} (>0). (1) The KL student anchor pools with "
            "the arithmetic mean (_pool_parts) while the teacher pools with the learned "
            "pool head (_pool), so the two sides are not comparable. (2) frozen_teacher "
            "freezes only model.backbone; the pool head is a separate module and stays "
            "live, so the 'frozen' retention target would drift every step. Set "
            "retention_kl_weight=0 or build the encoder with infer_pool_head=False."
        )
    if use_retention:
        # Fail here, before any compute, rather than silently optimising a KL that is
        # structurally 0 (see assert_retention_teacher_available).
        assert_retention_teacher_available(model)
    model.to(device).train()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Load core labels for same-core false-negative masking.
    # The (T,T) mask is built fresh each step from the batch tile_idx.
    _core_labels_np: np.ndarray | None = None
    if cfg.mask_same_core:
        _clp = Path(cfg.core_labels_path)
        if not _clp.exists():
            raise FileNotFoundError(
                f"--mask-same-core requires {_clp}; run scripts/derive_core_map.py first"
            )
        _core_labels_np = np.load(str(_clp)).astype(np.int32)
        print(f"[train] CORE MASK: loaded {len(_core_labels_np)} labels from {_clp}", flush=True)
    _sim_emb_t: torch.Tensor | None = None
    if cfg.mask_sim_thresh > 0:
        if cfg.mask_same_core:
            raise SystemExit("mask_sim_thresh and mask_same_core are mutually exclusive")
        _sep = Path(cfg.sim_emb_path)
        if not _sep.exists():
            raise SystemExit(f"sim_emb_path missing: {_sep}")
        _e = np.load(str(_sep)).astype(np.float32)
        _e /= (np.linalg.norm(_e, axis=1, keepdims=True) + 1e-8)
        _sim_emb_t = torch.from_numpy(_e)
        print(f"[train] SIM MASK: {_e.shape} embeddings from {_sep}, tau={cfg.mask_sim_thresh}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = _amp_dtype(cfg.amp_dtype)
    scaler = torch.amp.GradScaler(enabled=(amp is torch.float16))

    history: list[dict] = []
    step = 0
    resumed_from: Path | None = None
    if cfg.resume_from:
        resumed_from = Path(cfg.resume_from)
        step = _resume_into(
            resumed_from,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            history=history,
        )
        pbar_initial = step
    else:
        pbar_initial = 0
    t0 = time.time()
    tiles_seen = 0
    win_tiles, win_t0 = 0, time.time()
    # pre-clip grad norm from the most recent optimizer step; None until the first one
    # (and stays None entirely when cfg.grad_clip is falsy, since nothing computes it)
    last_grad_norm: float | None = None
    pbar = tqdm(total=cfg.max_steps, initial=pbar_initial, desc="train", unit="step")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps:
                break
            # PLAN.md 2's load-bearing detail, asserted rather than assumed. Runs every
            # step: it is a few microseconds on CPU-side index tensors, and a violation
            # here would still produce a perfectly plausible falling loss curve. The grid
            # path adds one more silent-failure mode -- tile lists that differ between
            # condition groups -- so it gets its own assertion, called just as often.
            if cfg.grid:
                batch_stats = assert_grid_batch(
                    batch, allowed_conditions=allowed_conditions
                )
            else:
                batch_stats = assert_same_condition_negatives(
                    batch, allowed_conditions=allowed_conditions
                )
            lr = cosine_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if cfg.grid:
                # ONE tensor, ONE forward: every image is both an anchor and a query.
                images = batch["image"].to(device, non_blocking=True)
                gid = batch["group_id"].to(device, non_blocking=True)
                n_cond, n_tiles = int(batch["n_cond"]), int(batch["n_tiles"])
                if cfg.split_heads:
                    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                        # Backbone chunked (memory), projectors run ONCE over the full C*T
                        # batch per head -- BatchNorm coupling requirement, same as single
                        # head. See _chunked_forward_split docstring.
                        parts, gz = _chunked_forward_split(
                            model, images, cfg.grid_forward_chunk, cfg.activation_offload
                        )
                        # anchor_emb is the eval-time pooled embedding (clsmean), NOT a
                        # projected vector, so the retention KL measures what is actually
                        # exported at eval time. Same contract as the pair-path split branch.
                        anchor_emb = model.pool_from_parts(parts)
                        teacher_emb = (
                            retention_teacher_embed(model, images) if use_retention else None
                        )
                    split_stats = assert_split_head_inputs(parts)
                    # Build same-core mask for this batch's tile set (once per step).
                    _core_mask = None
                    if cfg.mask_same_core and _core_labels_np is not None:
                        # collate flattens tile_idx to (C*T,) row-major, so the first
                        # n_tiles entries are condition 0's tile list -- which every
                        # condition shares (the grid invariant). Reshape-safe either way.
                        _tile_idx = (
                            batch["tile_idx"].reshape(-1)[:n_tiles].cpu().numpy()
                        )  # (T,)
                        _tc = torch.from_numpy(_core_labels_np[_tile_idx]).to(gz["cls"].device)
                        _core_mask = (_tc.unsqueeze(1) == _tc.unsqueeze(0))  # (T,T)
                        _core_mask.fill_diagonal_(False)  # positive never masked
                    if _sim_emb_t is not None:
                        _ti = batch["tile_idx"].reshape(-1)[:n_tiles].cpu()
                        _ze = _sim_emb_t[_ti].to(gz["cls"].device)
                        _core_mask = (_ze @ _ze.t()) > cfg.mask_sim_thresh  # (T,T)
                        _core_mask.fill_diagonal_(False)  # positive never masked
                    loss, metrics = grid_info_nce_split(
                        gz, n_cond, n_tiles,
                        {"cls": cfg.cls_weight, "mean": cfg.mean_weight},
                        cfg.temperature,
                        grid_blocked=cfg.grid_blocked_loss,
                        pair_block=cfg.grid_pair_block,
                        core_mask=_core_mask,
                        core_bias=(
                            {"cls": cfg.same_core_logit_bias_cls,
                             "mean": cfg.same_core_logit_bias_mean}
                            if cfg.same_core_logit_bias_cls is not None
                            or cfg.same_core_logit_bias_mean is not None
                            else cfg.same_core_logit_bias
                        ),
                        center=cfg.center_embeddings,
                    )
                    metrics.update(split_stats)
                else:
                    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                        # ONE graph, one backward -- the chunks are a memory device only, and
                        # the loss below still sees all C*T embeddings simultaneously (which
                        # is the entire point of the grid).
                        anchor_emb, gz = _chunked_forward(
                            model, images, cfg.grid_forward_chunk, cfg.activation_offload
                        )
                        teacher_emb = (
                            retention_teacher_embed(model, images) if use_retention else None
                        )
                    if cfg.grid_blocked_loss:
                        loss, metrics = grid_info_nce_blocked(
                            gz, n_cond, n_tiles, cfg.temperature,
                            pair_block=cfg.grid_pair_block,
                        )
                    else:
                        _core_mask2 = None
                        if cfg.mask_same_core and _core_labels_np is not None:
                            _tile_idx2 = batch["tile_idx"][0].cpu().numpy()
                            _tc2 = torch.from_numpy(_core_labels_np[_tile_idx2]).to(gz.device)
                            _core_mask2 = (_tc2.unsqueeze(1) == _tc2.unsqueeze(0))
                            _core_mask2.fill_diagonal_(False)
                        loss, metrics = grid_info_nce(gz, n_cond, n_tiles, cfg.temperature, core_mask=_core_mask2, core_bias=cfg.same_core_logit_bias)
                n_images = int(images.shape[0])
            else:
                anchor = batch["anchor"].to(device, non_blocking=True)
                positive = batch["positive"].to(device, non_blocking=True)
                gid = batch["group_id"].to(device, non_blocking=True)

                if cfg.split_heads:
                    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                        # ONE backbone forward per view, pooled two ways. Each head runs on
                        # its own view's batch, exactly as the single-head path calls the
                        # projector once per view -- so BatchNorm sees the same batches it
                        # always did, just through per-pool heads.
                        anchor_parts, az = model.forward_split(anchor)
                        _, pz = model.forward_split(positive)
                        anchor_emb = model.pool_from_parts(anchor_parts)
                        teacher_emb = (
                            retention_teacher_embed(model, anchor)
                            if use_retention else None
                        )
                    # The load-bearing check, run every step for the same reason
                    # assert_same_condition_negatives is: a violation still produces a
                    # plausible falling curve. See assert_split_head_inputs.
                    split_stats = assert_split_head_inputs(anchor_parts)
                    loss, metrics = split_head_info_nce(
                        az, pz, gid,
                        {"cls": cfg.cls_weight, "mean": cfg.mean_weight},
                        cfg.temperature, cfg.symmetric,
                    )
                    metrics.update(split_stats)
                    # GeM's learned p / LSE's tau / the attention entropy. A pooling that
                    # has quietly collapsed back to the mean (p -> 1, entropy -> 1.0) is
                    # invisible in the loss curve and obvious here.
                    metrics.update(model.pool_head_metrics())
                else:
                    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                        # one forward per view; both views must see the same weights, so this
                        # is a single graph, not two independent steps
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
                # two forward views per anchor, so a "step" moves 2 * batch tiles
                n_images = int(anchor.shape[0]) * 2
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
            tiles_seen += n_images
            win_tiles += n_images
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
                ck = save_checkpoint(
                    model, optimizer, step, cfg, {"step": step, **metrics},
                    scaler=scaler, rng_state=capture_rng_state(),
                )
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
