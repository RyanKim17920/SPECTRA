"""``grid_info_nce_blocked`` must be a NUMERICS refactor, not an approximation.

The dense path materialises the full ``(C, C, T, T)`` logit tensor, then an off-diagonal
COPY, then ``cross_entropy`` saves another for backward -- ~3x live at ``4*C^2*T^2`` bytes.
That is ~187 MB at today's T=1975 and irrelevant, but it is QUADRATIC in T and binds around
T~6000-8000, which is where the offload work is heading.

Swapping a loss is exactly the kind of change that goes wrong silently: the run trains, the
loss curve looks plausible, and the objective is not the one that was reviewed. So these
tests pin the value AND the gradient against the dense path, which stays the default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waivphaet.train.contrastive import grid_info_nce, grid_info_nce_blocked

#: Both paths compute in float32 -- ``grid_info_nce`` casts with ``z.float()`` regardless of
#: the caller's dtype, so a float64 comparison is not available to sharpen these. The two
#: differ only in the ORDER the per-row losses are summed, which in exact arithmetic is no
#: difference at all. Measured worst case across the cases below: relative 1.1e-7 on the
#: loss and 6.0e-8 absolute on the gradient (against a gradient scale of ~0.1-0.4), i.e.
#: right at float32 eps = 1.19e-7. These bounds are tight enough that a genuine method
#: error -- a dropped pair, a wrong axis, a mis-scaled temperature -- would fail loudly.
F32_ATOL = 1e-6
F32_RTOL = 1e-6

CASES = [
    # (C, T, D, pair_block)
    (2, 2, 4, 1),      # smallest legal grid; single pair per block
    (3, 5, 8, 1),      # block size 1 -- the most fragmented path
    (4, 7, 16, 3),     # blocks do NOT divide C*(C-1)=12 evenly -> short final block
    (5, 4, 8, 20),     # one block holds every pair -> degenerates to the dense path
    (6, 3, 12, 7),
]
IDS = [f"C{c}_T{t}_D{d}_blk{b}" for c, t, d, b in CASES]


def make_z(c, t, d, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(c * t, d, generator=g, dtype=torch.float64)
    z.requires_grad_(requires_grad)
    return z


@pytest.mark.parametrize("c,t,d,blk", CASES, ids=IDS)
def test_blocked_loss_matches_dense(c, t, d, blk):
    z = make_z(c, t, d)
    dense, dm = grid_info_nce(z, c, t, temperature=0.07)
    blocked, bm = grid_info_nce_blocked(z, c, t, temperature=0.07, pair_block=blk)
    assert torch.allclose(dense, blocked, atol=F32_ATOL, rtol=F32_RTOL), (
        f"blocked loss {blocked.item()!r} != dense {dense.item()!r} -- the blocked path is "
        "supposed to be the SAME objective, only computed with bounded memory"
    )
    assert bm.keys() == dm.keys(), "a caller must not be able to tell the paths apart"
    assert bm["top1"] == pytest.approx(dm["top1"])
    assert bm["n_rows"] == dm["n_rows"]


@pytest.mark.parametrize("c,t,d,blk", CASES, ids=IDS)
def test_blocked_gradient_matches_dense(c, t, d, blk):
    """The gradient is what actually trains the model, so equal losses is not enough.

    A silently-wrong contrastive gradient is the worst available outcome here: nothing
    crashes and the resulting curve is simply not the experiment anyone approved.
    """
    zd = make_z(c, t, d, requires_grad=True)
    zb = make_z(c, t, d, requires_grad=True)
    grid_info_nce(zd, c, t, temperature=0.07)[0].backward()
    grid_info_nce_blocked(zb, c, t, temperature=0.07, pair_block=blk)[0].backward()
    assert zd.grad is not None and zb.grad is not None
    assert torch.allclose(zd.grad, zb.grad, atol=F32_ATOL, rtol=F32_RTOL), (
        f"max grad delta {(zd.grad - zb.grad).abs().max().item():.3e}"
    )


def test_blocked_matches_dense_in_float32():
    """The real dtype. Tolerance is float32 summation-order noise, not method error."""
    c, t, d = 6, 12, 32
    zd = make_z(c, t, d, requires_grad=True).float().detach().requires_grad_(True)
    zb = zd.detach().clone().requires_grad_(True)
    ld, _ = grid_info_nce(zd, c, t)
    lb, _ = grid_info_nce_blocked(zb, c, t, pair_block=5)
    ld.backward()
    lb.backward()
    assert torch.allclose(ld, lb, atol=F32_ATOL, rtol=F32_RTOL)
    assert torch.allclose(zd.grad, zb.grad, atol=F32_ATOL, rtol=F32_RTOL)


@pytest.mark.parametrize("temperature", [0.01, 0.07, 1.0])
def test_temperature_is_applied_identically(temperature):
    z = make_z(4, 6, 8)
    a, _ = grid_info_nce(z, 4, 6, temperature=temperature)
    b, _ = grid_info_nce_blocked(z, 4, 6, temperature=temperature, pair_block=3)
    assert torch.allclose(a, b, atol=F32_ATOL, rtol=F32_RTOL)


def test_blocked_works_without_grad():
    """Under no_grad there is no graph to bound, so the checkpoint is skipped."""
    z = make_z(4, 5, 8)
    with torch.no_grad():
        a, _ = grid_info_nce(z, 4, 5)
        b, _ = grid_info_nce_blocked(z, 4, 5, pair_block=2)
    assert torch.allclose(a, b, atol=F32_ATOL, rtol=F32_RTOL)


def test_blocked_enforces_the_same_preconditions():
    z = make_z(3, 4, 8)
    with pytest.raises(ValueError, match="n_cond >= 2"):
        grid_info_nce_blocked(z[:4], 1, 4)
    with pytest.raises(ValueError, match="n_tiles >= 2"):
        grid_info_nce_blocked(z[:3], 3, 1)
    with pytest.raises(ValueError, match="does not match the declared geometry"):
        grid_info_nce_blocked(z, 3, 5)
    with pytest.raises(ValueError, match="pair_block must be >= 1"):
        grid_info_nce_blocked(z, 3, 4, pair_block=0)


def test_blocked_never_materialises_the_full_matrix():
    """The point of the exercise: peak logit memory must not scale with C^2.

    Counts elements passed to matmul rather than measuring allocator peak, so it is
    deterministic and CPU-safe.
    """
    c, t, d = 8, 10, 16
    z = make_z(c, t, d, requires_grad=True)
    seen = []
    real = torch.matmul

    def spy(a, b, *args, **kw):
        out = real(a, b, *args, **kw)
        seen.append(out.numel())
        return out

    torch.matmul = spy
    try:
        grid_info_nce_blocked(z, c, t, pair_block=4)[0].backward()
    finally:
        torch.matmul = real

    full = c * c * t * t
    assert seen, "expected the blocked path to go through torch.matmul"
    assert max(seen) <= 4 * t * t, (
        f"largest logit tensor was {max(seen)} elements; pair_block=4 caps it at "
        f"{4 * t * t}, versus {full} for the dense (C,C,T,T) path"
    )
