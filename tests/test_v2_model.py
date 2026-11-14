"""CPU tests of the v2 heads, objective and optimizer groups on a tiny fake backbone
(no pretrained weights are loaded)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectra.sweep import config as SC
from spectra.sweep import losses as L
from spectra.sweep import model as M
from spectra.sweep import train as TR

D, N_PREFIX, N_PATCH = 16, 2, 9


class _TinyViT(nn.Module):
    """(B, 3, 8, 8) -> tokens (B, prefix + 9, D): a linear patch embed + one Linear 'block'."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3 * 4, D)
        self.prefix = nn.Parameter(torch.randn(N_PREFIX, D))
        self.blocks = nn.ModuleList([nn.Linear(D, D)])

    def forward(self, x):
        b = x.shape[0]
        p = x.unfold(2, 2, 3).unfold(3, 2, 3)[:, :, :3, :3]          # (B, 3, 3, 3, 2, 2)
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(b, 9, 12)
        t = torch.cat([self.prefix.expand(b, -1, -1), self.embed(p)], 1)
        return self.blocks[0](t)


@pytest.fixture
def tiny(monkeypatch):
    def fake_build(cfg, grad_checkpointing=None):
        bb = _TinyViT()
        for q in bb.parameters():
            q.requires_grad_(False)
        bb.blocks[0].weight.requires_grad_(True)                      # stands in for the LoRA params
        return bb, dict(is_timm=True, hidden=D, num_blocks=1, prefix=N_PREFIX, norm_mean=(0.5,) * 3,
                        norm_std=(0.5,) * 3, repo="fake/tiny", lora_targets=["blocks.0"])

    monkeypatch.setattr(M, "build_backbone", fake_build)
    torch.manual_seed(0)
    heads = ("heads=[{pool: cls, weight: 0.5, proj_hidden: 32, proj_dim: 8},"
             "{pool: gem, weight: 0.5, proj_hidden: 32, proj_dim: 8}]")
    cfg = SC.load_config(None, [heads])
    return cfg, M.SweepModel(cfg)


def test_heads_build_and_pool(tiny):
    _, model = tiny
    assert list(model.pools) == ["gem"] and len(model.projectors) == 2
    x = torch.rand(6, 3, 8, 8)
    tok = model.tokens(x)
    f = model.head_features(x)
    assert set(f) == {"cls", "gem"} and f["cls"].shape == f["gem"].shape == (6, D)
    assert torch.equal(f["cls"], tok[:, 0])
    patches = tok[:, N_PREFIX:].float()                               # register tokens excluded
    want = torch.nn.functional.softplus(patches).clamp_min(1e-6).pow(3.0).mean(1).pow(1 / 3.0)
    assert torch.allclose(f["gem"], want, atol=1e-5)
    assert set(model.pool_tokens(tok, ["cls", "clsmean"])) == {"cls", "clsmean"}


def test_objective_is_weighted_sum_of_head_infonce(tiny):
    cfg, model = tiny
    model.train()
    C, T = 2, 8
    x = torch.rand(C * T, 3, 8, 8)
    obj = TR.Objective(cfg, model, torch.device("cpu"))
    feats = model.head_features(x)
    loss, mets = obj(feats, {"n_cond": C, "n_tiles": T, "step": 0})
    with torch.no_grad():
        want = 0.0
        for i, h in enumerate(cfg.heads):
            with obj.autocast():
                z = model.projectors[i](feats[h.pool].detach())
            want += 0.5 * float(L.infonce_grid(z, C, T, 0.1)[0])
    assert float(loss.detach()) == pytest.approx(want, rel=1e-5)
    assert {"h0_cls/top1", "h1_gem/top1", "loss", "gem_p"} <= set(mets)
    loss.backward()
    assert model.pools["gem"].p.grad is not None
    assert all(p.grad is not None for p in model.projectors.parameters())
    assert model.backbone.blocks[0].weight.grad is not None


def test_param_groups_and_trainable_state(tiny):
    cfg, model = tiny
    groups = TR.param_groups(model, cfg)
    by_id = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert by_id[id(model.pools["gem"].p)] == 0.0                     # scalar exponent: no decay
    assert by_id[id(model.backbone.blocks[0].weight)] == cfg.optim.weight_decay
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in groups) == n_trainable
    sd = TR.trainable_state(model)
    assert "pools.gem.p" in sd and any(k.startswith("projectors.1.") for k in sd)
    TR.load_trainable(model, {k: v.clone() for k, v in sd.items()})


def test_lr_schedule_is_truncated_not_rescaled():
    cfg = SC.load_config(None, ["optim.stop_step=175"])
    assert TR.lr_factor(0, cfg) == pytest.approx(1 / 200)
    assert TR.lr_factor(174, cfg) == pytest.approx(175 / 200)          # still warming up at a 175 stop
    full = SC.load_config(None)
    assert all(TR.lr_factor(s, cfg) == TR.lr_factor(s, full) for s in range(0, 250, 7))
