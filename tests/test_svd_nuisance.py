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
from fit_svd_nuisance import (  # noqa: E402
    build_basis,
    nuisance_basis,
    spectrum_report,
    split_half_dim,
)


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


# ----- (d) split vs joint basis -------------------------------------------

def _plism_like(C=8, N=30, H=6, seed=20, cls_scale=1.0, mean_scale=1.0):
    """Synthetic clsmean data: tissue cancels, each half gets its own planted nuisance.

    The two halves are given deliberately different nuisance magnitudes, because that is
    the exact case a joint eigendecomposition gets wrong -- the louder half claims rank the
    other never warranted.
    """
    rng = np.random.default_rng(seed)
    tissue = rng.standard_normal((N, 2 * H)).astype(np.float32) * 4.0
    u_cls = rng.standard_normal(H).astype(np.float32)
    u_cls /= np.linalg.norm(u_cls)
    u_mean = rng.standard_normal(H).astype(np.float32)
    u_mean /= np.linalg.norm(u_mean)
    a = rng.standard_normal(C).astype(np.float32)
    b = rng.standard_normal(C).astype(np.float32)

    F = np.repeat(tissue[None], C, axis=0).copy()
    F[:, :, :H] += cls_scale * a[:, None, None] * u_cls[None, None, :]
    F[:, :, H:] += mean_scale * b[:, None, None] * u_mean[None, None, :]
    F += rng.standard_normal(F.shape).astype(np.float32) * 1e-3
    return F.astype(np.float32), u_cls, u_mean


def test_split_half_dim_rejects_single_site_pooling():
    """The guard: `cls` and `mean` have no halves, so splitting them is refused outright
    rather than silently cutting one representation down its middle."""
    for pooling in ("cls", "mean", "gem"):
        with pytest.raises(ValueError, match="two-site pooling"):
            split_half_dim(pooling, 2048)
    assert split_half_dim("clsmean", 2048) == 1024


def test_split_half_dim_rejects_odd_width():
    with pytest.raises(ValueError):
        split_half_dim("clsmean", 2049)


def test_build_basis_split_hard_errors_on_non_clsmean():
    F, _, _ = _plism_like()
    with pytest.raises(ValueError, match="two-site pooling"):
        build_basis(F, "split", "cls")


def test_build_basis_joint_is_bitwise_the_old_behaviour():
    """Regression guard: --basis joint must not have moved a single bit, or every k-sweep
    already run stops being comparable to the next one."""
    F, _, _ = _plism_like(cls_scale=1.0, mean_scale=8.0)
    V_ref, ev_ref = nuisance_basis(F)
    V, ev, extra = build_basis(F, "joint", "clsmean")
    assert np.array_equal(V, V_ref)
    assert np.array_equal(ev, ev_ref)
    assert extra == {"basis": "joint", "half_dim": 0}


def test_split_basis_is_block_diagonal_and_orthonormal():
    """Each half's sub-basis is supported on its own half and nowhere else."""
    C, N, H = 8, 30, 6
    F, u_cls, u_mean = _plism_like(C=C, N=N, H=H, mean_scale=8.0)
    V, ev, extra = build_basis(F, "split", "clsmean")

    assert V.shape == (2 * H, 2 * H)
    assert extra["basis"] == "split" and extra["half_dim"] == H
    assert np.abs(V[:H, H:]).max() == 0.0, "cls rows must be zero on the mean half"
    assert np.abs(V[H:, :H]).max() == 0.0, "mean rows must be zero on the cls half"
    assert np.allclose(V @ V.T, np.eye(2 * H), atol=1e-5)

    # Row 0 and row H are the leading direction of their OWN half, so each recovers its
    # own planted vector even though the mean half is 8x louder.
    assert abs(float(V[0, :H] @ u_cls)) > 0.99
    assert abs(float(V[H, H:] @ u_mean)) > 0.99
    # eigvals is stored ROW-ALIGNED with V, so it restarts at row H rather than
    # descending globally -- reading a spectrum off it directly would be wrong.
    assert np.allclose(ev, np.concatenate([extra["eigvals_cls"], extra["eigvals_mean"]]),
                       rtol=1e-5)
    assert ev[H] > ev[H - 1], "the merged eigvals array is deliberately not descending"


def test_joint_basis_lets_the_louder_half_claim_the_rank():
    """The fidelity bug the split option exists to fix, demonstrated.

    With one half 8x louder, the joint top-1 lives almost entirely inside that half, so a
    joint rank-1 removal touches the quiet half barely at all. The split basis removes one
    direction from each by construction.
    """
    C, N, H = 8, 30, 6
    F, _, _ = _plism_like(C=C, N=N, H=H, cls_scale=1.0, mean_scale=8.0)
    V_joint, _ = nuisance_basis(F)
    mass_in_mean_half = float((V_joint[0, H:] ** 2).sum())
    assert mass_in_mean_half > 0.95, "the loud half should dominate the joint top-1"

    V_split, _, _ = build_basis(F, "split", "clsmean")
    assert float((V_split[0, :H] ** 2).sum()) == pytest.approx(1.0, abs=1e-5)
    assert float((V_split[H, H:] ** 2).sum()) == pytest.approx(1.0, abs=1e-5)


def _write_split_fit(path: Path, F, pooling="clsmean"):
    V, ev, extra = build_basis(F, "split", pooling)
    np.savez(path, V=V, eigvals=ev.astype(np.float32),
             backbone=np.array("paige-ai/Virchow2"), pooling=np.array(pooling),
             n_tiles=np.array(F.shape[1]), seed=np.array(0), D=np.array(F.shape[2]),
             conditions=np.array(["GM_AT2", "GM_P"]),
             basis=np.array(extra["basis"]), half_dim=np.array(extra["half_dim"]),
             eigvals_cls=extra["eigvals_cls"], eigvals_mean=extra["eigvals_mean"])
    return V, extra["half_dim"]


def test_load_basis_split_selects_k_per_half(tmp_path):
    """Rank k under a split fit is 2k rows: the top k of each half, not the top k overall."""
    H = 6
    F, _, _ = _plism_like(H=H, mean_scale=8.0)
    fit = tmp_path / "split.npz"
    V, half = _write_split_fit(fit, F)
    assert half == H

    for k in (0, 1, 3, H):
        Vk, meta = load_basis(fit, k)
        assert Vk.shape == (2 * k, 2 * H)
        assert meta["basis"] == "split"
        assert meta["directions_removed"] == 2 * k
        if k:
            assert np.array_equal(Vk[:k], V[:k])
            assert np.array_equal(Vk[k:], V[H:H + k])


def test_load_basis_split_caps_k_at_the_half_width(tmp_path):
    """k is capped at H, not D -- asking for H+1 would silently reach into the other half's
    block and remove directions twice."""
    H = 6
    F, _, _ = _plism_like(H=H)
    fit = tmp_path / "split.npz"
    _write_split_fit(fit, F)
    with pytest.raises(ValueError, match="split fit"):
        load_basis(fit, H + 1)


def test_split_projection_leaves_each_half_orthogonal_to_its_own_sub_basis(tmp_path):
    """The block-diagonal packing means the projection acts independently per half."""
    H, k = 6, 2
    F, _, _ = _plism_like(H=H, mean_scale=8.0)
    fit = tmp_path / "split.npz"
    V, _ = _write_split_fit(fit, F)
    Vk, _ = load_basis(fit, k)

    f = np.random.default_rng(31).standard_normal((25, 2 * H)).astype(np.float32)
    out = project_out(f, Vk)
    assert np.abs(out[:, :H] @ V[:k, :H].T).max() < 1e-5
    assert np.abs(out[:, H:] @ V[H:H + k, H:].T).max() < 1e-5
    # And it is genuinely per-half: removing cls directions cannot move the mean half's
    # component along a mean direction that was NOT removed.
    keep = V[H + k, H:]
    assert np.allclose(out[:, H:] @ keep, f[:, H:] @ keep, atol=1e-4)


def test_load_basis_defaults_to_joint_for_a_pre_flag_npz(tmp_path):
    """An npz written before --basis existed has no `basis` field and must still read as
    joint -- k rows, not 2k."""
    D = 8
    fit = tmp_path / "old.npz"
    _write_fake_fit(fit, D=D)
    Vk, meta = load_basis(fit, 3)
    assert Vk.shape == (3, D)
    assert meta["basis"] == "joint"
    assert meta["directions_removed"] == 3
