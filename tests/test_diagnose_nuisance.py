"""Unit tests for the nuisance-spectrum diagnostics (CPU, synthetic).

Tests:
  (a) the offset/interaction split sums to 1, and separates a synthesised PURE
      per-condition offset (interaction ~0) from a synthesised tile x condition
      interaction (interaction share large)
  (b) the condition-mean spectrum of a 91-condition pure-offset corpus is exhausted by
      rank 90 -- the correctness check the whole framing rests on
  (c) the subspace-overlap helper is ~1 for identical subspaces and ~0 for orthogonal ones
  (d) the shift helper translates by exactly the requested pixel count
  (e) the one-axis-at-a-time delta helper isolates the axis it is told to

No GPU, no PLISM, no backbone. Everything here runs off np.random.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from diagnose_nuisance import (  # noqa: E402
    axis_deltas,
    gram_eigh,
    offset_interaction_split,
    shift_tiles,
    subspace_overlap,
)


def _pure_offset(C, N, D, seed=0, noise=0.0):
    """``F[c, i] = tissue[i] + m[c]`` -- acquisition is a per-condition shift and nothing else."""
    rng = np.random.default_rng(seed)
    tissue = rng.standard_normal((N, D)).astype(np.float32) * 3.0
    m = rng.standard_normal((C, D)).astype(np.float32)
    F = tissue[None, :, :] + m[:, None, :]
    if noise:
        F = F + rng.standard_normal((C, N, D)).astype(np.float32) * noise
    return F.astype(np.float32)


def _pure_interaction(C, N, D, seed=1):
    """``F[c, i] = tissue[i] + a[c] * g[i] * u`` -- the shift's SIZE depends on the tile.

    Constructed so the per-condition mean of the acquisition term is ~0 (``g`` is centred
    over tiles), i.e. essentially all of the nuisance is tile x condition interaction.
    """
    rng = np.random.default_rng(seed)
    tissue = rng.standard_normal((N, D)).astype(np.float32) * 3.0
    a = rng.standard_normal(C).astype(np.float32)
    g = rng.standard_normal(N).astype(np.float32)
    g -= g.mean()
    u = rng.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    F = tissue[None, :, :] + a[:, None, None] * g[None, :, None] * u[None, None, :]
    return F.astype(np.float32)


# ----- (a) the split ------------------------------------------------------

def test_split_energies_sum_to_one():
    out = offset_interaction_split(_pure_offset(8, 20, 16, noise=0.5))
    e = out["energy"]
    assert e["offset_share"] + e["interaction_share"] == pytest.approx(1.0, abs=1e-9)


def test_pure_per_condition_offset_has_no_interaction():
    """The whole point: if acquisition is one vector per condition, the residual is zero."""
    out = offset_interaction_split(_pure_offset(12, 30, 24))
    assert out["energy"]["offset_share"] == pytest.approx(1.0, abs=1e-6)
    assert out["energy"]["interaction_share"] < 1e-6


def test_tile_dependent_nuisance_shows_up_as_interaction():
    """And if the shift's size depends on the tile, the offset part cannot capture it."""
    out = offset_interaction_split(_pure_interaction(12, 30, 24))
    assert out["energy"]["interaction_share"] > 0.9
    assert out["energy"]["offset_share"] < 0.1


def test_split_rejects_bad_shape():
    with pytest.raises(ValueError):
        offset_interaction_split(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        offset_interaction_split(np.zeros((1, 5, 3), dtype=np.float32))


# ----- (b) rank <= C-1 for the offset part --------------------------------

def test_offset_spectrum_of_91_conditions_is_exhausted_by_rank_90():
    """PLISM's exact shape: 91 conditions, D well past 90.

    A per-condition offset is 91 rank-1 terms with one linear constraint (the deltas are
    zero-mean over conditions), so its Gram has rank <= 90 and rank 90 must already carry
    100% of the variance. This is what makes any mass past index 90 in the real fit
    evidence of interaction rather than of a bigger offset.
    """
    out = offset_interaction_split(_pure_offset(91, 12, 128, seed=7))
    ev = out["offset"]["explained_variance"]
    assert ev["90"] == pytest.approx(1.0, abs=1e-6)
    assert ev["64"] < 0.999, "a rank-90 spectrum should not already be saturated at 64"
    assert out["offset"]["tail_past_rank_90"] < 1e-9
    # 128 and 256 exist in the ladder but must add nothing.
    assert ev["128"] == pytest.approx(1.0, abs=1e-6)


def test_interaction_spectrum_can_exceed_rank_90():
    """The converse: a tile-dependent nuisance is NOT confined to 90 directions."""
    out = offset_interaction_split(_pure_offset(91, 200, 128, seed=8, noise=1.0))
    assert out["interaction"]["tail_past_rank_90"] > 0.1


# ----- (c) subspace overlap -----------------------------------------------

def _orthonormal_rows(D, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    return q.T.copy().astype(np.float32)  # ROWS, matching gram_eigh's layout


def test_subspace_overlap_identical_is_one():
    V = _orthonormal_rows(64, seed=3)
    for k in (8, 32):
        assert subspace_overlap(V, V, k) == pytest.approx(1.0, abs=1e-5)


def test_subspace_overlap_orthogonal_is_zero():
    V = _orthonormal_rows(64, seed=4)
    # Rows 32..63 are orthogonal to rows 0..31 by construction of the QR basis.
    assert subspace_overlap(V[:32], V[32:], 32) == pytest.approx(0.0, abs=1e-5)
    assert subspace_overlap(V[:8], V[32:40], 8) == pytest.approx(0.0, abs=1e-5)


def test_subspace_overlap_is_symmetric_and_bounded():
    Va, Vb = _orthonormal_rows(32, seed=5), _orthonormal_rows(32, seed=6)
    for k in (4, 16):
        ab = subspace_overlap(Va, Vb, k)
        assert ab == pytest.approx(subspace_overlap(Vb, Va, k), abs=1e-6)
        assert 0.0 <= ab <= 1.0 + 1e-6


def test_subspace_overlap_rejects_too_large_k():
    V = _orthonormal_rows(8, seed=7)
    with pytest.raises(ValueError):
        subspace_overlap(V, V, 16)


# ----- (d) the shift helper -----------------------------------------------

@pytest.mark.parametrize("px", [1, 2, 4, 8])
def test_shift_translates_by_exactly_the_requested_pixels(px):
    """A single bright pixel must land exactly ``px`` further along BOTH axes."""
    tiles = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    tiles[:, 10, 12, :] = 255
    out = shift_tiles(tiles, px)
    assert out.shape == tiles.shape and out.dtype == np.uint8
    assert out[0, 10 + px, 12 + px, 0] == 255
    assert out[0].sum() == tiles[0].sum(), "roll must not create or destroy signal"
    assert out[0, 10, 12, 0] == 0


def test_shift_zero_is_identity():
    tiles = np.random.default_rng(0).integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
    assert np.array_equal(shift_tiles(tiles, 0), tiles)


def test_shift_wraps_rather_than_padding():
    """Circular, not crop-and-pad: content leaving one edge reappears at the other, so no
    constant band is introduced for the encoder to react to."""
    tiles = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    tiles[0, 7, 7, :] = 255
    out = shift_tiles(tiles, 1)
    assert out[0, 0, 0, 0] == 255


def test_shift_rejects_bad_shape():
    with pytest.raises(ValueError):
        shift_tiles(np.zeros((8, 8, 3), dtype=np.uint8), 2)


# ----- (e) one axis at a time ---------------------------------------------

def test_axis_deltas_isolates_the_named_axis():
    """A grid whose only nuisance is a per-scanner offset: the scanner-axis deltas must
    carry all of it and the stain-axis deltas essentially none."""
    n_stain, n_scan, N, D = 4, 3, 10, 8
    rng = np.random.default_rng(11)
    tissue = rng.standard_normal((N, D)).astype(np.float32) * 2.0
    per_scanner = rng.standard_normal((n_scan, D)).astype(np.float32)
    G = tissue[None, None] + per_scanner[None, :, None, :]

    _, tot_scan = axis_deltas(G, axis=1)   # mean over scanners -> isolates scanner
    _, tot_stain = axis_deltas(G, axis=0)  # mean over stains -> isolates stain (none here)
    assert tot_scan > 0
    assert tot_stain < 1e-6 * max(tot_scan, 1.0)


def test_axis_deltas_gram_matches_a_direct_computation():
    n_stain, n_scan, N, D = 3, 2, 5, 4
    G = np.random.default_rng(12).standard_normal((n_stain, n_scan, N, D)).astype(np.float32)
    gram, total = axis_deltas(G, axis=1)
    d = (G - G.mean(axis=1, keepdims=True)).astype(np.float64).reshape(-1, D)
    assert np.allclose(gram, d.T @ d, atol=1e-8)
    assert total == pytest.approx((d ** 2).sum(), rel=1e-9)
    evals, V = gram_eigh(gram)
    assert np.all(np.diff(evals) <= 1e-9), "eigenvalues must come back descending"
    assert np.allclose(V @ V.T, np.eye(D), atol=1e-5)


def test_axis_deltas_rejects_bad_shape():
    with pytest.raises(ValueError):
        axis_deltas(np.zeros((3, 4, 5), dtype=np.float32), axis=1)
