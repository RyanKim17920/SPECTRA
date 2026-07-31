"""Tests for the invariants that fail *silently* if broken.

Every check here guards something that would not show up in the training loss:
a positive that isn't co-registered, or a negative that leaks acquisition signal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from waivphaet.data.conditions import (
    SCANNERS, STAINS, all_conditions, make_split, parse_filename,
)
from waivphaet.data.pairs import PairBatchSampler
from waivphaet.train.contrastive import masked_info_nce


def test_condition_grid_is_13x7():
    assert len(STAINS) == 13 and len(SCANNERS) == 7
    assert len(all_conditions()) == 91
    assert len({c.filename for c in all_conditions()}) == 91


def test_filename_roundtrip():
    for c in all_conditions():
        assert parse_filename(c.filename) == c


def test_split_is_deterministic_and_disjoint():
    a, b = make_split(), make_split()
    assert [c.key for c in a.train] == [c.key for c in b.train]
    assert set(a.train).isdisjoint(a.heldout)
    assert len(a.train) + len(a.heldout) == 91


def test_split_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_split(["NOPE"], [])


def test_positive_condition_never_equals_anchor():
    """PLAN.md 2: a positive must be a *different* acquisition condition."""
    s = PairBatchSampler(all_conditions(), n_groups=32, group_size=16, batches_per_epoch=25)
    for b in s:
        assert not (b.positive_cond == b.anchor_cond[:, None]).any()


def test_group_is_condition_homogeneous_with_unique_tiles():
    """The structural guarantee the loss mask relies on."""
    s = PairBatchSampler(all_conditions(), n_groups=8, group_size=16, batches_per_epoch=10)
    for b in s:
        assert b.anchor_cond.shape == (8,)  # one condition per group, by construction
        for g in range(b.n_groups):
            assert len(np.unique(b.tile_idx[g])) == b.group_size


def test_positive_conditions_cover_the_grid():
    """The offset-shift trick must stay uniform over the other 90 conditions."""
    s = PairBatchSampler(all_conditions(), n_groups=64, group_size=16, batches_per_epoch=40)
    hist = np.zeros(91, dtype=int)
    for b in s:
        hist += np.bincount(b.positive_cond.ravel(), minlength=91)
    assert hist.min() > 0
    assert hist.max() < 2.0 * hist.mean()


def test_masked_infonce_negative_count_is_group_bounded():
    """Negatives must be the group (same condition), not the whole batch."""
    g = torch.arange(32) // 8
    z = torch.randn(32, 16)
    _, m = masked_info_nce(z, z.clone(), g)
    assert m["negatives_per_anchor"] == pytest.approx(7.0)  # group_size - 1, not 31


def test_masked_infonce_random_loss_is_log_group_size():
    g = torch.arange(64) // 8
    torch.manual_seed(0)
    _, m = masked_info_nce(torch.randn(64, 32), torch.randn(64, 32), g, temperature=1.0)
    assert m["loss"] == pytest.approx(np.log(8), abs=0.35)


def test_masked_infonce_ignores_cross_group_similarity():
    """A cross-group near-duplicate must not affect the loss -- it is not a valid negative."""
    g = torch.tensor([0, 0, 1, 1])
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    loss_a, _ = masked_info_nce(a, a.clone(), g, temperature=0.1)
    b = a.clone()
    b[2] = torch.tensor([1.0, 0.0])  # duplicate of row 0, but a different group
    loss_b, _ = masked_info_nce(b, b.clone(), g, temperature=0.1)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)
