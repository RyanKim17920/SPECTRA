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
from waivphaet.data.pairs import (
    PairBatch, PairBatchSampler, assert_same_condition_negatives, collate_pair_batch,
)
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


# --- the negative constraint, asserted rather than eyeballed ---------------------------


def _fake_collated(n_groups=3, group_size=8, n_cond=91, seed=0):
    """A collated batch dict with the pixel tensors omitted (the checks never read them)."""
    s = PairBatchSampler(
        all_conditions()[:n_cond], n_groups=n_groups, group_size=group_size,
        batches_per_epoch=1, seed=seed,
    )
    b = next(iter(s))
    anchor_cond = np.broadcast_to(b.anchor_cond[:, None], b.tile_idx.shape)
    item = {
        "tile_idx": torch.from_numpy(b.tile_idx),
        "anchor_cond": torch.from_numpy(np.ascontiguousarray(anchor_cond)),
        "positive_cond": torch.from_numpy(b.positive_cond),
        "group_id": torch.arange(n_groups).repeat_interleave(group_size),
    }
    return b, collate_pair_batch(item)


def test_collated_batch_satisfies_the_negative_constraint():
    """Every group the loss will mask over must be condition-homogeneous."""
    _, batch = _fake_collated()
    stats = assert_same_condition_negatives(batch, allowed_conditions=set(range(91)))
    assert stats["negatives_per_anchor"] == 7.0
    assert stats["n_groups"] == 3.0


def test_assertion_catches_a_condition_mixed_group():
    """The check must FAIL on the bug it exists to catch, or it is decoration."""
    _, batch = _fake_collated()
    batch["anchor_cond"] = batch["anchor_cond"].clone()
    batch["anchor_cond"][0] = (batch["anchor_cond"][0] + 1) % 91  # one intruder
    with pytest.raises(AssertionError, match="mixes anchor conditions"):
        assert_same_condition_negatives(batch)


def test_assertion_catches_a_heldout_condition_leak():
    _, batch = _fake_collated()
    with pytest.raises(AssertionError, match="held-out condition has leaked"):
        assert_same_condition_negatives(batch, allowed_conditions={0})


def test_assertion_catches_a_repeated_tile_in_a_group():
    _, batch = _fake_collated()
    batch["tile_idx"] = batch["tile_idx"].clone()
    batch["tile_idx"][1] = batch["tile_idx"][0]  # anchor 1 is now anchor 0's own tile
    with pytest.raises(AssertionError, match="repeats a tile index"):
        assert_same_condition_negatives(batch)


def test_pairbatch_validate_rejects_a_same_condition_positive():
    b = PairBatch(
        tile_idx=np.array([[0, 1]]),
        anchor_cond=np.array([3]),
        positive_cond=np.array([[3, 5]]),  # first "positive" is the anchor's own condition
    )
    with pytest.raises(AssertionError, match="cross-acquisition"):
        b.validate(91)


def test_infonce_queries_positives_against_the_condition_homogeneous_anchors():
    """Orientation is load-bearing, not cosmetic.

    The candidate row must be the ANCHORS (one shared condition, so acquisition carries
    no signal), with the positives as queries. Running it the other way makes the
    candidate row span conditions and reintroduces the acquisition shortcut.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    g = torch.arange(16) // 8
    a, p = torch.randn(16, 12), torch.randn(16, 12)
    an, pn = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
    mask = g[:, None] == g[None, :]
    tgt = torch.arange(16)

    want = F.cross_entropy(((pn @ an.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)
    wrong = F.cross_entropy(((an @ pn.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)

    got, _ = masked_info_nce(a, p, g, temperature=0.07)
    assert torch.allclose(got, want, atol=1e-5)
    assert not torch.allclose(want, wrong, atol=1e-3)  # the two really are different


def test_symmetric_is_off_by_default():
    """PLAN.md 2: the anchor->positive direction has cross-condition candidates."""
    from waivphaet.train.contrastive import TrainConfig
    assert TrainConfig().symmetric is False
