"""The v2 objective, as a pure tensor function.

Layout convention (shared with ``spectra.data.grid``): a batch is ``(C*T, D)``, ROW-MAJOR
``(C, T, D)`` -- row ``a*T + t`` is condition ``a``, location ``t``. Every condition group
holds the SAME T locations, so ``(a, t)`` and ``(b, t)`` are a registered positive pair.

Returns ``(loss, metrics)``; metrics are python floats.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _shape(z: torch.Tensor, n_cond: int, n_tiles: int) -> None:
    if n_cond < 2 or n_tiles < 2 or z.shape[0] != n_cond * n_tiles:
        raise ValueError(f"bad grid geometry: z {tuple(z.shape)} vs C={n_cond} T={n_tiles}")


def infonce_grid(z, n_cond, n_tiles, temperature=0.1):
    """Symmetric grid InfoNCE. For every ORDERED condition pair (a, b), a != b: query (a, t),
    candidates = all T locations of condition b (ONE acquisition -> no scanner shortcut),
    target (b, t). Other locations of the same tissue core stay ordinary negatives.
    Loss = mean over the C*(C-1)*T rows (both directions of every pair)."""
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
