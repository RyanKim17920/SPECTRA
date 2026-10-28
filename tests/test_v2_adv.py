"""CPU tests: the queue adversary, the optimiser groups and the Q8 objective (stub model)."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectra.sweep import config as SC
from spectra.sweep import losses as L
from spectra.sweep import train as TR
from spectra.sweep.model import N_CLASSES

C, T, D = 2, 12, 16


def _cfg(**over):
    d = SC.to_dict(SC.SweepConfig())
    for k, v in over.items():
        SC.set_by_path(d, k, v)
    return SC.from_dict(d)


def _stub(cfg, d=D):
    torch.manual_seed(0)
    m = types.SimpleNamespace()
    m.adv_heads = nn.ModuleDict({t: nn.Linear(d, N_CLASSES[t]) for t in cfg.adv.targets})
    m.projector = nn.Linear(d, 8)
    return m


def _labels():
    return {"stain": torch.tensor([1] * T + [4] * T), "scanner": torch.tensor([0] * T + [2] * T)}


def test_adv_lambda_dann_ramp_uses_horizon():
    cfg = _cfg(**{"adv.lam": 0.3, "optim.steps": 400})
    assert TR.adv_lambda(cfg, 0) == 0.0
    assert TR.adv_lambda(cfg, 100) == pytest.approx(0.3 * L.dann_ramp(100, 400))


def test_adv_queue_fifo_wrap_and_state_roundtrip():
    q = TR.AdvQueue(5, ["stain", "scanner"])
    assert q.get("stain") == (None, None)
    f = torch.arange(12, dtype=torch.float32).view(6, 2)
    q.push(f[:3], {"stain": torch.tensor([0, 1, 2]), "scanner": torch.tensor([3, 3, 3])})
    qf, qy = q.get("stain")
    assert q.n == 3 and torch.equal(qf, f[:3]) and qy.tolist() == [0, 1, 2]
    q.push(f[3:6], {"stain": torch.tensor([3, 4, 5]), "scanner": torch.tensor([1, 1, -100])})
    qf, qy = q.get("stain")                                          # 6 pushed into 5: oldest dropped
    assert q.n == 5 and q.ptr == 1
    assert sorted(qy.tolist()) == [1, 2, 3, 4, 5]
    assert torch.equal(qf[0], f[5]) and qy[0] == 5                   # slot 0 overwritten by the newest
    assert q.get("scanner")[1].tolist() == [-100, 3, 3, 1, 1]
    s = q.state_dict()
    q2 = TR.AdvQueue(5, ["stain", "scanner"])
    q2.load_state_dict(s, torch.device("cpu"))
    assert q2.n == 5 and q2.ptr == 1 and torch.equal(q2.get("stain")[0], qf)
    big = TR.AdvQueue(4, ["stain"])                                  # a push larger than the queue keeps its tail
    big.push(f, {"stain": torch.arange(6)})
    assert big.n == 4 and sorted(big.get("stain")[1].tolist()) == [2, 3, 4, 5]
    with pytest.raises(AssertionError):
        TR.AdvQueue(6, ["stain", "scanner"]).load_state_dict(s, torch.device("cpu"))


def test_adv_queue_head_trains_on_queue_and_reversal_only_through_batch():
    cfg = _cfg(**{"adv.lam": 0.3, "adv.queue": 64, "optim.steps": 10, "optim.warmup_steps": 2,
                  "optim.stop_step": 0})
    lam = TR.adv_lambda(cfg, 10)
    m = _stub(cfg)
    q = TR.AdvQueue(64, cfg.adv.targets)
    g = torch.Generator().manual_seed(3)
    qf = torch.randn(40, D, generator=g)
    qlab = {"stain": torch.randint(0, N_CLASSES["stain"], (40,), generator=g),
            "scanner": torch.randint(0, N_CLASSES["scanner"], (40,), generator=g)}
    q.push(qf, qlab)
    f = torch.randn(C * T, D, requires_grad=True)
    lab = _labels()
    loss, mets = TR.adv_loss(m, cfg, f, lab, 10, q)
    assert mets["adv/lambda"] == pytest.approx(lam)
    loss.backward()
    hg = {t: (m.adv_heads[t].weight.grad.clone(), m.adv_heads[t].bias.grad.clone()) for t in ("stain", "scanner")}
    # backbone: exactly the batch-only reversed gradient (-lambda x dCE_batch/df, head as a constant)
    f2 = f.detach().clone().requires_grad_(True)
    sum(F.cross_entropy(TR._frozen_call(m.adv_heads[t], f2), lab[t]) for t in ("stain", "scanner")).backward()
    assert torch.allclose(f.grad, -lam * f2.grad, atol=1e-6)
    # heads: exactly dCE(queue + batch)/dtheta (no contribution from the reversed term)
    for t in ("stain", "scanner"):
        h = m.adv_heads[t]
        ref = nn.Linear(D, N_CLASSES[t])
        ref.load_state_dict(h.state_dict())
        F.cross_entropy(ref(torch.cat([qf, f.detach()])), torch.cat([qlab[t], lab[t]])).backward()
        assert torch.allclose(hg[t][0], ref.weight.grad, atol=1e-6)
        assert torch.allclose(hg[t][1], ref.bias.grad, atol=1e-6)
        with torch.no_grad():
            assert mets[f"adv/{t}_qacc"] == pytest.approx(float((h(qf).argmax(1) == qlab[t]).float().mean()))
            assert mets[f"adv/{t}_acc"] == pytest.approx(float((h(f).argmax(1) == lab[t]).float().mean()))
        assert mets[f"adv/{t}_qcls"] == len(set(qlab[t].tolist()))
    want = sum(F.cross_entropy(m.adv_heads[t](torch.cat([qf, f.detach()])), torch.cat([qlab[t], lab[t]]))
               + F.cross_entropy(m.adv_heads[t](f.detach()), lab[t]) for t in ("stain", "scanner"))
    assert float(loss) == pytest.approx(float(want), rel=1e-5)
    assert q.n == 64 and mets["adv/queue_n"] == 64                   # batch enqueued AFTER the loss
    assert torch.equal(q.get("stain")[1][40:], lab["stain"])


def test_adv_queue_empty_first_step_and_unlabelled():
    cfg = _cfg(**{"adv.lam": 0.1, "adv.queue": 32})
    m = _stub(cfg)
    q = TR.AdvQueue(32, cfg.adv.targets)
    f = torch.randn(C * T, D, requires_grad=True)
    lab = {"stain": torch.full((C * T,), -100), "scanner": torch.tensor([0] * T + [2] * T)}
    loss, mets = TR.adv_loss(m, cfg, f, lab, 0, q)
    assert torch.isfinite(loss) and "adv/stain_qacc" not in mets and q.n == C * T
    loss, mets = TR.adv_loss(m, cfg, f, lab, 1, q)
    assert "adv/stain_qacc" not in mets and mets["adv/scanner_qcls"] == 2


def test_adv_queue_makes_a_real_multiclass_head():
    """A batch shows only C=2 of K conditions; through the queue the head is trained over all K."""
    torch.manual_seed(0)
    K, d, n_per = 8, 16, 64
    mu = 4.0 * torch.randn(K, d)

    def sample(ks):
        return (torch.cat([mu[k] + torch.randn(n_per, d) for k in ks]),
                torch.cat([torch.full((n_per,), k) for k in ks]))
    xt, yt = sample(range(K))
    cfg = _cfg(**{"adv.lam": 0.0, "adv.targets": ["stain"], "adv.queue": 4096})
    m = types.SimpleNamespace(adv_heads=nn.ModuleDict({"stain": nn.Linear(d, N_CLASSES["stain"])}))
    opt = torch.optim.AdamW(m.adv_heads.parameters(), lr=1e-2, weight_decay=0.0)
    q = TR.AdvQueue(4096, ["stain"])
    rng = np.random.default_rng(0)
    for step in range(300):
        x, y = sample(rng.choice(K, 2, replace=False))
        loss, _ = TR.adv_loss(m, cfg, x, {"stain": y}, step, q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((m.adv_heads["stain"](xt).argmax(1) == yt).float().mean())
    assert acc > 0.9


def test_grl_adversary_lowers_nuisance_separability():
    """Features = noise + a class-dependent offset; training them against a GRL'd linear head
    shrinks the between-class mean gap."""
    torch.manual_seed(0)
    n = 400
    y = torch.randint(0, 2, (n,))
    f = (torch.randn(n, D) + 3.0 * F.one_hot(y, D).float()).requires_grad_(True)
    head = nn.Linear(D, 2)
    opt_f = torch.optim.SGD([f], lr=0.5 * n)
    opt_h = torch.optim.SGD(head.parameters(), lr=0.5)

    def gap():
        x = f.detach()
        return float((x[y == 1].mean(0) - x[y == 0].mean(0)).norm())
    g0 = gap()
    for _ in range(300):
        opt_f.zero_grad(), opt_h.zero_grad()
        F.cross_entropy(head(L.grad_reverse(f, 1.0)), y).backward()
        opt_f.step(), opt_h.step()
    assert gap() < 0.5 * g0


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.projector = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
        self.adv_heads = nn.ModuleDict({"stain": nn.Linear(4, 5)})


def test_param_groups_wd_split_and_fixed_lr_adversary():
    cfg = _cfg(**{"adv.lr": 1e-3, "optim.lr": 1e-4, "optim.weight_decay": 0.05})
    m = _Tiny()
    groups = TR.param_groups(m, cfg)
    adv = [g for g in groups if g.get("adv")]
    assert len(adv) == 1 and adv[0]["lr"] == 1e-3 and adv[0]["weight_decay"] == 0.0
    assert {id(p) for p in adv[0]["params"]} == {id(p) for p in m.adv_heads.parameters()}
    main = [g for g in groups if not g.get("adv")]
    assert [g["weight_decay"] for g in main] == [0.05, 0.0]
    assert all(p.ndim >= 2 for p in main[0]["params"]) and all(p.ndim <= 1 for p in main[1]["params"])
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))
    opt = torch.optim.AdamW(groups, lr=cfg.optim.lr)
    for f in (0.005, 0.5, 1.0):
        TR.set_lrs(opt, cfg, f)
        for g in opt.param_groups:
            assert g["lr"] == (1e-3 if g.get("adv") else pytest.approx(1e-4 * f))


def test_clip_groups_separate():
    m = _Tiny()
    opt = torch.optim.AdamW(TR.param_groups(m, _cfg()))
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    for p in m.adv_heads.parameters():
        p.grad = 1000 * torch.ones_like(p)
    main = [p for n, p in m.named_parameters() if not n.startswith("adv_heads.")]
    want = float(torch.sqrt(sum((p.grad ** 2).sum() for p in main)))
    gn, gna = TR.clip_groups(opt, 1.0)
    assert gn == pytest.approx(want, rel=1e-5) and gna > 1000      # adversary grads excluded
    assert float(torch.sqrt(sum((p.grad ** 2).sum() for p in main))) == pytest.approx(1.0, rel=1e-4)


def test_lr_schedule_warmup_then_cosine():
    cfg = _cfg(**{"optim.warmup_steps": 200, "optim.steps": 400})
    assert TR.lr_factor(0, cfg) == pytest.approx(1 / 200)
    assert TR.lr_factor(199, cfg) == pytest.approx(1.0)
    assert TR.lr_factor(200, cfg) == pytest.approx(1.0)
    assert TR.lr_factor(300, cfg) == pytest.approx(0.5)
    assert TR.lr_factor(400, cfg) == pytest.approx(0.0, abs=1e-12)


class _StubModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(0)
        self.projector = nn.Linear(D, 8)
        self.adv_heads = nn.ModuleDict({t: nn.Linear(D, N_CLASSES[t]) for t in cfg.adv.targets})


def test_objective_total_is_multitau_nce_plus_adv():
    cfg = _cfg(**{"adv.queue": 64})
    m = _StubModel(cfg)
    obj = TR.Objective(cfg, m, torch.device("cpu"))
    cls = torch.randn(C * T, D, requires_grad=True)
    meta = {"n_cond": C, "n_tiles": T, "step": 50, "adv_targets": _labels()}
    total, mets = obj(cls, meta)
    with torch.no_grad():
        with obj.autocast():                                          # the projector runs under bf16 autocast
            z = m.projector(cls)
        nce = 0.5 * L.infonce_grid(z, C, T, 0.1)[0] + 0.5 * L.infonce_grid(z, C, T, 0.6)[0]
        adv = TR.adv_loss(m, cfg, cls, meta["adv_targets"], 50, TR.AdvQueue(64, cfg.adv.targets))[0]
    assert float(total.detach()) == pytest.approx(float(nce + adv), rel=1e-5)
    assert mets["loss_primary"] == pytest.approx(float(nce), rel=1e-5)
    assert "h0_cls/loss_tau0.1" in mets and "h0_cls/loss_tau0.6" in mets and "adv/stain_loss" in mets
    total.backward()
    assert cls.grad is not None and m.projector.weight.grad is not None
    assert obj.adv_queue.n == C * T
