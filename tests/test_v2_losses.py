"""CPU tests: multi-temperature grid InfoNCE, gradient reversal, DANN ramp, adversary CE."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectra.sweep import losses as L


def _z(c=2, t=16, d=8, seed=0, noise=0.5):
    g = torch.Generator().manual_seed(seed)
    loc = torch.randn(1, t, d, generator=g)
    return (loc + noise * torch.randn(c, t, d, generator=g)).reshape(c * t, d)


def _reference_infonce(z, c, t, tau):
    """Row-by-row definition: query (a, i), candidates = the t locations of condition b != a."""
    zn = F.normalize(z.float(), dim=-1).view(c, t, -1)
    rows = []
    for a in range(c):
        for b in range(c):
            if a != b:
                logits = zn[a] @ zn[b].t() / tau
                rows.append(F.cross_entropy(logits, torch.arange(t), reduction="none"))
    return torch.cat(rows).mean()


@pytest.mark.parametrize("c", [2, 3])
@pytest.mark.parametrize("tau", [0.1, 0.6])
def test_infonce_matches_row_definition(c, tau):
    z = _z(c, 16, 8, seed=c)
    got, m = L.infonce_grid(z, c, 16, tau)
    assert torch.allclose(got, _reference_infonce(z, c, 16, tau), atol=1e-6)
    assert 0.0 <= m["top1"] <= 1.0 and m["loss"] == pytest.approx(float(got))


def test_infonce_perfect_alignment_low_loss_and_top1():
    z = _z(2, 32, 16, noise=0.0)
    loss, m = L.infonce_grid(z, 2, 32, 0.1)
    assert m["top1"] == 1.0 and float(loss) < float(L.infonce_grid(_z(2, 32, 16, noise=3.0), 2, 32, 0.1)[0])


def test_infonce_rejects_bad_geometry():
    with pytest.raises(ValueError):
        L.infonce_grid(torch.randn(10, 4), 2, 4, 0.1)


def test_multi_tau_is_weighted_sum_over_same_logits():
    z = _z(2, 16, 8, seed=1).requires_grad_(True)
    nce = lambda t: L.infonce_grid(z, 2, 16, t)  # noqa: E731
    tot, m = L.multi_tau(nce, [0.1, 0.6], [0.5, 0.5])
    want = 0.5 * L.infonce_grid(z, 2, 16, 0.1)[0] + 0.5 * L.infonce_grid(z, 2, 16, 0.6)[0]
    assert torch.allclose(tot, want)
    assert m["loss"] == pytest.approx(float(want.detach()))
    assert m["loss_tau0.1"] == pytest.approx(float(L.infonce_grid(z, 2, 16, 0.1)[0]))
    assert m["loss_tau0.6"] == pytest.approx(float(L.infonce_grid(z, 2, 16, 0.6)[0]))
    assert m["top1"] == L.infonce_grid(z, 2, 16, 0.1)[1]["top1"]      # metrics of the first tau
    eq, _ = L.multi_tau(nce, [0.1, 0.6])                                # empty weights = 1/K each
    assert torch.allclose(eq, want)
    single, _ = L.multi_tau(nce, [0.1])
    assert torch.allclose(single, L.infonce_grid(z, 2, 16, 0.1)[0])
    tot.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    with pytest.raises(ValueError):
        L.multi_tau(nce, [0.1, 0.6], [1.0])


def test_grl_flips_and_scales_gradient():
    x = torch.ones(3, requires_grad=True)
    y = L.grad_reverse(x, 0.5)
    assert torch.equal(y, x)
    (y * 2).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -1.0))


def test_dann_ramp():
    assert L.dann_ramp(0, 400) == 0.0
    assert L.dann_ramp(400, 400) == pytest.approx(2 / (1 + torch.exp(torch.tensor(-10.0)).item()) - 1)
    assert L.dann_ramp(800, 400) == L.dann_ramp(400, 400)                # clamped at p = 1
    vals = [L.dann_ramp(s, 400) for s in range(0, 401, 50)]
    assert vals == sorted(vals)


def test_adversarial_ce_skips_unlabelled_rows():
    logits = torch.randn(6, 4, requires_grad=True)
    y = torch.tensor([0, 1, -100, 3, -100, 2])
    loss, m = L.adversarial_ce(logits, y)
    keep = y != -100
    assert torch.allclose(loss, F.cross_entropy(logits[keep], y[keep]))
    z, mz = L.adversarial_ce(logits, torch.full((6,), -100))
    assert float(z) == 0.0 and mz["adv_loss"] == 0.0
    z.backward()                                                         # still graph-connected
