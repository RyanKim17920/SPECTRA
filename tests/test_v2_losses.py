"""CPU tests: grid InfoNCE."""

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
