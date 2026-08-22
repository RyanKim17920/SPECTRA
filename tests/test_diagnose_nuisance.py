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
  (f) the linear probe decodes a condition identity planted in ONE known, LOW-VARIANCE
      direction, and stops decoding it the moment that direction is projected out --
      the decodability-is-not-variance-share point, made on data where the answer is known

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
    linear_probe_diagnostic,
    offset_interaction_split,
    probe_design,
    probe_tile_split,
    shift_tiles,
    subspace_overlap,
)
from fit_svd_nuisance import nuisance_basis  # noqa: E402


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


# ----- (f) the linear probe -----------------------------------------------

def _one_direction_corpus(n_class=4, n_rep=3, N=60, D=16, gap=1.0, seed=21):
    """Class identity lives in ONE direction that holds almost none of the variance.

    ``F[c, i] = tissue[i] + a[class(c)] * u``, with ``tissue`` built entirely inside the
    orthogonal complement of ``u``. So along ``u`` the within-class scatter is only the
    noise floor while the classes sit ``gap`` apart -- perfectly separable -- yet ``u``
    carries a vanishing share of the total embedding variance -- ``gap``-scale against a
    tissue direction 5x wider in each of D-1 dimensions. That is the whole point of
    diagnostic 5 in synthetic form: the two quantities are not the same quantity.

    Returns ``(F, labels, u)`` with ``n_class * n_rep`` conditions.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    u = Q[:, 0].astype(np.float32)

    # Tissue coordinates in the complement of u: column 0 (the u coordinate) is exactly 0.
    coords = rng.standard_normal((N, D)).astype(np.float32) * 5.0
    coords[:, 0] = 0.0
    tissue = (coords @ Q.T).astype(np.float32)

    labels = np.repeat(np.arange(n_class), n_rep)
    a = (labels.astype(np.float32) - labels.mean()) * gap
    F = tissue[None, :, :] + a[:, None, None] * u[None, None, :]
    F = F + rng.standard_normal(F.shape).astype(np.float32) * (gap * 1e-2)
    return F.astype(np.float32), labels, u


def test_probe_tile_split_is_disjoint_and_covers_every_tile():
    """Held-out must mean held-out: a random ROW split would leak, since all 91 conditions
    image the same tile."""
    tr, te = probe_tile_split(50, 0.5, seed=0)
    assert len(tr) == 25 and len(te) == 25
    assert set(tr).isdisjoint(set(te))
    assert set(tr) | set(te) == set(range(50))
    assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0), "positions stay sorted"


def test_probe_tile_split_keeps_both_sides_non_empty():
    for frac in (0.01, 0.99):
        tr, te = probe_tile_split(4, frac, seed=1)
        assert len(tr) >= 1 and len(te) >= 1


def test_probe_tile_split_rejects_a_degenerate_fraction():
    for frac in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            probe_tile_split(10, frac, seed=0)


def test_probe_design_labels_every_row_by_its_condition():
    F = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
    labels = np.array([7, 8, 9])
    X, y = probe_design(F, np.array([0, 2, 4]), labels, max_rows=0, seed=0)
    assert X.shape == (9, 2) and y.shape == (9,)
    assert list(y) == [7, 7, 7, 8, 8, 8, 9, 9, 9]
    assert np.array_equal(X[4], F[1, 2])


def test_probe_design_subsamples_when_capped():
    F = np.zeros((4, 20, 3), dtype=np.float32)
    X, y = probe_design(F, np.arange(20), np.arange(4), max_rows=10, seed=0)
    assert len(X) == len(y) == 10


def test_probe_decodes_a_low_variance_direction_and_loses_it_at_k1():
    """The reconciliation, on data where the answer is known.

    k=0: near-perfect decoding of a class that occupies a direction holding well under 1%
    of the embedding variance. k=1: that one direction is the whole nuisance basis's top
    eigenvector, and once it is projected out the probe is at chance.
    """
    F, labels, u = _one_direction_corpus()
    V, eigvals = nuisance_basis(F)

    # The planted direction IS the nuisance top-1 ...
    assert abs(float(V[0] @ u)) > 0.99
    # ... and it is quiet: its share of the total embedding variance is tiny.
    total_var = float(((F - F.mean(axis=(0, 1))) ** 2).sum())
    assert float(eigvals.sum()) / total_var < 0.01

    out = linear_probe_diagnostic(
        F, {"cond": labels}, V, ranks=(0, 1, 2), pc_dims=(1, 2),
        train_frac=0.5, max_rows=0, max_iter=500, seed=0)
    acc = out["cond"]["project_out_k"]
    chance = out["cond"]["chance_balanced_accuracy"]
    assert chance == pytest.approx(0.25)
    assert acc["0"] > 0.95, f"a separable low-variance direction must decode: {acc}"
    assert acc["1"] < 0.45, f"removing the one direction must kill it: {acc}"
    assert acc["2"] <= acc["1"] + 0.05


def test_probe_reports_the_split_sizes_and_chance_level():
    F, labels, _ = _one_direction_corpus(N=40)
    V, _ = nuisance_basis(F)
    out = linear_probe_diagnostic(F, {"cond": labels}, V, ranks=(0,), pc_dims=(1,),
                                  train_frac=0.5, max_rows=0, max_iter=200, seed=0)
    assert out["n_train_tiles"] == 20 and out["n_test_tiles"] == 20
    assert out["cond"]["rows_train"] == 12 * 20 and out["cond"]["rows_test"] == 12 * 20
    assert out["cond"]["subsampled"] is False
    assert out["cond"]["n_classes"] == 4


def test_probe_marks_a_capped_run_as_subsampled():
    """A capped run must SAY it is capped in the JSON -- an accuracy read off 12k of 182k
    rows is a different measurement from one read off all of them."""
    F, labels, _ = _one_direction_corpus(N=40)
    V, _ = nuisance_basis(F)
    out = linear_probe_diagnostic(F, {"cond": labels}, V, ranks=(0,), pc_dims=(1,),
                                  train_frac=0.5, max_rows=50, max_iter=200, seed=0)
    assert out["cond"]["subsampled"] is True
    assert out["cond"]["rows_train"] == 50 and out["cond"]["rows_test"] == 50


def test_probe_top_m_pcs_needs_more_than_one_pc_for_a_quiet_direction():
    """The top principal component of the EMBEDDING is tissue, not the planted class
    direction, so a 1-PC probe is near chance even though the class is linearly separable
    in the full space. Variance ordering and decodability are different orderings."""
    F, labels, _ = _one_direction_corpus()
    V, _ = nuisance_basis(F)
    out = linear_probe_diagnostic(F, {"cond": labels}, V, ranks=(0,), pc_dims=(1,),
                                  train_frac=0.5, max_rows=0, max_iter=500, seed=0)
    assert out["cond"]["top_m_pcs"]["1"] < 0.5
    assert out["cond"]["project_out_k"]["0"] > 0.95
