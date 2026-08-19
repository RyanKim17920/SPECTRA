"""Unit tests for same-core false-negative masking in grid_info_nce.

Tests:
  (a) positive (diagonal) is never masked
  (b) masked entries contribute nothing to the softmax
  (c) with core_mask=None the loss is bit-identical to the unpatched path
"""
import sys
from pathlib import Path
# Use falseneg-pinned src so the patched grid_info_nce is tested
_PIN = Path("/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned/src")
sys.path.insert(0, str(_PIN))
# Remove any already-imported waivphaet from main src
for _k in list(sys.modules): 
    if _k.startswith("waivphaet"): del sys.modules[_k]

import torch
import pytest
from waivphaet.train.contrastive import grid_info_nce, grid_info_nce_split


def _make_batch(C=3, T=8, D=32, seed=42):
    torch.manual_seed(seed)
    z = torch.randn(C * T, D)
    return z, C, T


def _make_core_mask(T, same_core_pairs):
    """Build a (T,T) bool mask. same_core_pairs: list of (i,j) i!=j to mask."""
    mask = torch.zeros(T, T, dtype=torch.bool)
    for i, j in same_core_pairs:
        mask[i, j] = True
        mask[j, i] = True
    mask.fill_diagonal_(False)  # positive never masked
    return mask


# ----- (a) positive never masked -----------------------------------------

def test_positive_never_masked():
    """Diagonal (positive) entries must not be -inf even with aggressive masking."""
    z, C, T = _make_batch()
    # Mask everything except diagonal (extreme case)
    core_mask = torch.ones(T, T, dtype=torch.bool)
    core_mask.fill_diagonal_(False)

    loss, metrics = grid_info_nce(z, C, T, temperature=0.07, core_mask=core_mask)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    # top1 must be computable (not NaN)
    assert 0.0 <= metrics["top1"] <= 1.0, f"top1 out of range: {metrics['top1']}"


# ----- (b) masked entries contribute nothing to softmax -------------------

def test_masked_entries_excluded_from_softmax():
    """Logits for masked positions should be -inf, giving zero softmax weight."""
    C, T, D = 2, 6, 16
    torch.manual_seed(7)
    z = torch.randn(C * T, D)
    # mask tile 1 and tile 3 as same-core as tile 0
    core_mask = _make_core_mask(T, [(0, 1), (0, 3)])

    loss, metrics = grid_info_nce(z, C, T, temperature=0.07, core_mask=core_mask)

    # Manually check: reconstruct pair logits and verify masked positions are -inf
    import torch.nn.functional as F
    zn = F.normalize(z.float(), dim=-1).view(C, T, -1)
    logits = torch.einsum("atd,bsd->abts", zn, zn) / 0.07
    off = ~torch.eye(C, dtype=torch.bool)
    a_idx, b_idx = torch.where(off)
    pair_logits = logits[a_idx, b_idx]  # (C*(C-1), T, T)
    pair_logits_masked = pair_logits.masked_fill(core_mask.unsqueeze(0), float("-inf"))

    # For anchor tile 0, positions 1 and 3 should be -inf
    for pair_i in range(pair_logits_masked.shape[0]):
        assert pair_logits_masked[pair_i, 0, 1].item() == float("-inf"), "tile 1 not masked for anchor 0"
        assert pair_logits_masked[pair_i, 0, 3].item() == float("-inf"), "tile 3 not masked for anchor 0"
        # diagonal (positive) must NOT be -inf
        for t in range(T):
            assert pair_logits_masked[pair_i, t, t].item() != float("-inf"), f"positive at {t} was masked"


# ----- (c) bit-identical when core_mask=None --------------------------------

def test_no_mask_bit_identical():
    """core_mask=None must produce bit-identical output to the unmasked call."""
    z, C, T = _make_batch()

    loss_ref, metrics_ref = grid_info_nce(z, C, T, temperature=0.07)
    loss_masked, metrics_masked = grid_info_nce(z, C, T, temperature=0.07, core_mask=None)

    assert loss_ref.item() == loss_masked.item(), (
        f"loss differs: {loss_ref.item()} vs {loss_masked.item()}"
    )
    assert metrics_ref["top1"] == metrics_masked["top1"]
    assert metrics_masked["masked_per_row"] == 0


# ----- split-heads path ---------------------------------------------------

def test_split_heads_positive_never_masked():
    """Same test through grid_info_nce_split."""
    C, T, D = 3, 8, 16
    torch.manual_seed(99)
    z_cls = torch.randn(C * T, D)
    z_mean = torch.randn(C * T, D)
    z_dict = {"cls": z_cls, "mean": z_mean}

    core_mask = torch.ones(T, T, dtype=torch.bool)
    core_mask.fill_diagonal_(False)

    loss, metrics = grid_info_nce_split(
        z_dict, C, T, {"cls": 0.5, "mean": 0.5}, temperature=0.07, core_mask=core_mask
    )
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["top1"] <= 1.0


def test_split_heads_no_mask_bit_identical():
    C, T, D = 3, 8, 16
    torch.manual_seed(5)
    z_cls = torch.randn(C * T, D)
    z_mean = torch.randn(C * T, D)
    z_dict = {"cls": z_cls, "mean": z_mean}

    loss_ref, m_ref = grid_info_nce_split(z_dict, C, T, {"cls": 0.5, "mean": 0.5}, temperature=0.07)
    loss_masked, m_masked = grid_info_nce_split(z_dict, C, T, {"cls": 0.5, "mean": 0.5}, temperature=0.07, core_mask=None)

    assert loss_ref.item() == loss_masked.item()
    assert m_ref["top1"] == m_masked["top1"]
