"""Unit tests for the training-free nuisance-subspace intervention (CPU, synthetic).

Tests:
  (a) an orthonormal projection removes exactly the intended subspace, and k=0 is identity
  (b) the fit recovers a planted nuisance direction from synthetic PLISM-shaped data
  (c) apply_svd_nuisance round-trips a PathoROB-shaped npz tree: keys, dtype, dir wipe

No GPU, no PLISM, no PathoROB features. Everything here runs off np.random.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from apply_svd_nuisance import apply_fit, load_basis, project_out  # noqa: E402
from fit_svd_nuisance import nuisance_basis, spectrum_report  # noqa: E402


def _orthonormal(D, k, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    return q[:, :k].T.copy().astype(np.float32)  # ROWS, matching the fit's V layout


# ----- (a) projection algebra --------------------------------------------

def test_projection_removes_exactly_the_subspace():
    """After projecting out Vk, the residual has zero component along every row of Vk."""
    D, k, N = 32, 4, 50
    Vk = _orthonormal(D, k, seed=1)
    f = np.random.default_rng(2).standard_normal((N, D)).astype(np.float32)

    out = project_out(f, Vk)
    assert np.abs(out @ Vk.T).max() < 1e-5, "residual still has mass in the removed subspace"
    # The complement is untouched: the removed part is exactly the projection.
    removed = f - out
    assert np.allclose(removed, (f @ Vk.T) @ Vk, atol=1e-5)


def test_k0_is_identity():
    """k=0 must be a no-op -- it is the control arm the whole sweep is measured against."""
    D, N = 16, 20
    f = np.random.default_rng(3).standard_normal((N, D)).astype(np.float32)
    V0 = np.zeros((0, D), dtype=np.float32)
    assert np.array_equal(project_out(f, V0), f)


# ----- (b) the fit recovers a planted direction ---------------------------

def test_fit_recovers_planted_nuisance_direction():
    """Synthetic PLISM: tissue varies per tile, one planted direction varies per condition.

    ``F[c, i] = tissue[i] + a[c] * u + small noise``. The tissue term is identical across
    c, so it cancels in the per-tile mean subtraction and the ONLY systematic residual is
    along ``u``. The top eigenvector must therefore be parallel to ``u``.
    """
    C, N, D = 12, 40, 24
    rng = np.random.default_rng(4)
    u = rng.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    tissue = rng.standard_normal((N, D)).astype(np.float32) * 5.0  # deliberately dominant
    a = rng.standard_normal(C).astype(np.float32) * 2.0

    F = (tissue[None, :, :] + a[:, None, None] * u[None, None, :]).astype(np.float32)
    F += rng.standard_normal(F.shape).astype(np.float32) * 1e-3

    V, eigvals = nuisance_basis(F)
    assert V.shape == (D, D)
    assert eigvals.shape == (D,)
    # Descending, and the values were reordered together with their vectors.
    assert np.all(np.diff(eigvals) <= 1e-9)

    cos = abs(float(V[0] @ u))
    assert cos > 0.99, f"top eigenvector is not the planted direction (|cos|={cos:.4f})"
    # A single planted direction is a rank-1 nuisance, so rank 1 must explain nearly all of it.
    assert spectrum_report(eigvals, ranks=(1,))[1] > 0.99


def test_fit_rejects_a_single_condition():
    with pytest.raises(ValueError):
        nuisance_basis(np.zeros((1, 5, 4), dtype=np.float32))


# ----- (c) npz round-trip -------------------------------------------------

def _write_fake_features(root: Path, model: str, datasets, D=8, seed=5):
    rng = np.random.default_rng(seed)
    keys = {}
    for ds in datasets:
        for center in ("Center_A", "Center_B"):
            k = [f"slide{ds[:3]}{center[-1]}{i}-patch{i}" for i in range(4)]
            (root / model / ds).mkdir(parents=True, exist_ok=True)
            np.savez(root / model / ds / f"{center}.npz",
                     **{name: rng.standard_normal(D).astype(np.float32) for name in k})
            keys[(ds, center)] = k
    return keys


def _write_fake_fit(path: Path, D=8, seed=6):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    np.savez(path, V=q.T.astype(np.float32), eigvals=np.arange(D, 0, -1, dtype=np.float32),
             backbone=np.array("owkin/phikon-v2"), pooling=np.array("clsmean"),
             n_tiles=np.array(10), seed=np.array(0), D=np.array(D),
             conditions=np.array(["GM_AT2", "GM_P"]))


def test_apply_roundtrip_preserves_keys_and_dtype(tmp_path):
    D, datasets = 8, ("camelyon",)
    feats = tmp_path / "features"
    keys = _write_fake_features(feats, "src_model", datasets, D=D)
    fit = tmp_path / "fit.npz"
    _write_fake_fit(fit, D=D)

    info = apply_fit(fit, 3, "src_model", "dst_model", feats, datasets)
    assert info["D"] == D
    assert info["vectors"] == {"camelyon": 8}

    Vk, _ = load_basis(fit, 3)
    for (ds, center), names in keys.items():
        src = np.load(feats / "src_model" / ds / f"{center}.npz")
        dst = np.load(feats / "dst_model" / ds / f"{center}.npz")
        assert list(dst.files) == list(src.files) == names
        for name in names:
            assert dst[name].dtype == src[name].dtype
            assert dst[name].shape == src[name].shape
            expected = project_out(src[name][None, :], Vk)[0]
            assert np.allclose(dst[name], expected, atol=1e-6)


def test_apply_k0_is_bit_identical(tmp_path):
    D, datasets = 8, ("camelyon",)
    feats = tmp_path / "features"
    keys = _write_fake_features(feats, "src_model", datasets, D=D)
    fit = tmp_path / "fit.npz"
    _write_fake_fit(fit, D=D)

    apply_fit(fit, 0, "src_model", "dst_model", feats, datasets)
    for (ds, center), names in keys.items():
        src = np.load(feats / "src_model" / ds / f"{center}.npz")
        dst = np.load(feats / "dst_model" / ds / f"{center}.npz")
        for name in names:
            assert np.array_equal(dst[name], src[name])


def test_apply_wipes_destination(tmp_path):
    D, datasets = 8, ("camelyon",)
    feats = tmp_path / "features"
    _write_fake_features(feats, "src_model", datasets, D=D)
    fit = tmp_path / "fit.npz"
    _write_fake_fit(fit, D=D)

    # A stale file from an earlier k, in a dataset dir the new run also writes. PathoROB's
    # save_features would merge it; we must not.
    stale = feats / "dst_model" / "camelyon" / "Center_STALE.npz"
    stale.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stale, junk=np.zeros(D, dtype=np.float32))

    apply_fit(fit, 2, "src_model", "dst_model", feats, datasets)
    assert not stale.exists(), "stale destination file survived; it would be silently scored"


def test_apply_rejects_width_mismatch(tmp_path):
    feats = tmp_path / "features"
    _write_fake_features(feats, "src_model", ("camelyon",), D=8)
    fit = tmp_path / "fit.npz"
    _write_fake_fit(fit, D=16)  # a cls-only-vs-clsmean style mismatch
    with pytest.raises(SystemExit):
        apply_fit(fit, 2, "src_model", "dst_model", feats, ("camelyon",))
