"""The Q8 objective terms, as pure tensor functions.

Layout convention (shared with ``spectra.data.grid``): a batch is ``(C*T, D)``, ROW-MAJOR
``(C, T, D)`` -- row ``a*T + t`` is condition ``a``, location ``t``. Every condition group
holds the SAME T locations, so ``(a, t)`` and ``(b, t)`` are a registered positive pair.

All loss functions return ``(loss, metrics)``; metrics are python floats.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


def _shape(z: torch.Tensor, n_cond: int, n_tiles: int) -> None:
    if n_cond < 2 or n_tiles < 2 or z.shape[0] != n_cond * n_tiles:
        raise ValueError(f"bad grid geometry: z {tuple(z.shape)} vs C={n_cond} T={n_tiles}")


# --------------------------------------------------------------------------------------
# contrastive
# --------------------------------------------------------------------------------------


def infonce_grid(z, n_cond, n_tiles, temperature=0.1):
    """Grid InfoNCE. For every ORDERED condition pair (a, b), a != b: query (a, t),
    candidates = all T locations of condition b (ONE acquisition -> no scanner shortcut),
    target (b, t). Other locations of the same tissue core stay ordinary negatives.
    Loss = mean over the C*(C-1)*T rows."""
    _shape(z, n_cond, n_tiles)
    zn = F.normalize(z.float(), dim=-1).view(n_cond, n_tiles, -1)
    logits = torch.einsum("atd,bsd->abts", zn, zn) / temperature
    off = ~torch.eye(n_cond, dtype=torch.bool, device=zn.device)
    a_idx, b_idx = torch.where(off)
    flat = logits[a_idx, b_idx].reshape(-1, n_tiles)  # (P*T, T)
    target = torch.arange(n_tiles, device=zn.device).repeat(int(a_idx.numel()))
    loss = F.cross_entropy(flat, target)
    with torch.no_grad():
        top1 = float((flat.argmax(1) == target).float().mean())
    return loss, {"loss": float(loss.detach()), "top1": top1}


def multi_tau(nce: Callable[[float], tuple], taus: Sequence[float], weights: Sequence[float] = ()):
    """``nce(t) -> (loss, mets)`` at one temperature -> sum_k w_k nce(taus[k]) with
    w = ``weights`` (empty = 1 / K each). Metrics: the first tau's, plus ``loss_tau<t>`` per tau
    and ``loss`` = the weighted total."""
    taus = [float(t) for t in taus]
    ws = [float(w) for w in weights] if weights else [1.0 / len(taus)] * len(taus)
    if len(ws) != len(taus):
        raise ValueError(f"{len(ws)} tau weights for {len(taus)} taus")
    total, mets = None, {}
    for k, (t_, w) in enumerate(zip(taus, ws)):
        l, m = nce(t_)
        total = w * l if total is None else total + w * l
        if k == 0:
            mets.update(m)
        mets[f"loss_tau{t_:g}"] = float(l.detach())
    mets["loss"] = float(total.detach())
    return total, mets


# --------------------------------------------------------------------------------------
# adversary
# --------------------------------------------------------------------------------------


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Identity forward; backward multiplies the gradient by ``-lam``."""
    return _GRL.apply(x, lam)


def dann_ramp(step: int, total: int) -> float:
    """DANN (Ganin & Lempitsky 2015) GRL schedule factor 2 / (1 + exp(-10 p)) - 1, p = step / total in [0, 1]."""
    p = min(max(step / max(total, 1), 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def adversarial_ce(logits, labels):
    """Cross-entropy over the rows with a label (``-100`` = unlabelled, e.g. empty queue slots)."""
    keep = labels != -100
    if not bool(keep.any()):
        z = logits.sum() * 0.0
        return z, {"adv_loss": 0.0, "adv_acc": float("nan")}
    logits, labels = logits[keep], labels[keep]
    loss = F.cross_entropy(logits.float(), labels)
    with torch.no_grad():
        acc = float((logits.argmax(1) == labels).float().mean())
    return loss, {"adv_loss": float(loss.detach()), "adv_acc": acc}
