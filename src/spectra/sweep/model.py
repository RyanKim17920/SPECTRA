"""Backbone + LoRA + the v2 loss heads (CLS and GeM, each with its own projector).

Backbone LOADING is delegated to ``spectra.models.encoder`` (registry normalisation, the
gated local-directory binding via ``SPECTRA_BACKBONE_LOCAL_DIRS``, strict timm loads,
register-token counts).

Construction order fixes the RNG draws: backbone build -> LoRA injection -> learned pools
(GeM) -> projectors in head order.

``.data`` (which pulls h5py through ``spectra.data.grid``) and ``transformers`` are imported
lazily, so inference-only environments can load a trained checkpoint with torch + timm + peft.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from spectra.models import encoder as E
from spectra.models.pooling import GeMPool

from .config import SweepConfig, resolve_backbone

_ATTN_RE = re.compile(r"(?:^|\.)(?:attn|attention|self_attn)(?:\.|$)")


def load_backbone(repo: str) -> tuple[nn.Module, dict]:
    """Raw pretrained backbone + geometry, exactly as ``SpectraEncoder`` builds it."""
    if E.is_timm_backbone(repo):
        import timm

        local = E.local_backbone_dir(repo)
        if local is None:
            bb = timm.create_model(f"hf-hub:{repo}", pretrained=True, **E._timm_extra_kwargs(repo))
        else:
            bb = E.SpectraEncoder._build_local_timm(repo, local)
        info = dict(is_timm=True, hidden=int(bb.embed_dim), num_blocks=len(bb.blocks),
                    prefix=int(getattr(bb, "num_prefix_tokens", 1)))
    else:
        from transformers import AutoModel

        bb = AutoModel.from_pretrained(repo)
        info = dict(is_timm=False, hidden=int(bb.config.hidden_size),
                    num_blocks=int(bb.config.num_hidden_layers), prefix=1)
    info["norm_mean"], info["norm_std"] = E.normalization_for(repo)
    info["repo"] = repo
    return bb, info


def enable_grad_ckpt(bb: nn.Module, is_timm: bool) -> None:
    if is_timm:
        bb.set_grad_checkpointing(True)
        ok = bool(getattr(bb, "grad_checkpointing", False))
    else:
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ok = bool(getattr(bb, "is_gradient_checkpointing", False))
    if not ok:
        raise RuntimeError("gradient checkpointing did not take on the backbone")


def block_index(name: str) -> int | None:
    return E._block_index(name)


def lora_targets(bb: nn.Module, which: str) -> list[str]:
    """In-block Linear module names that get LoRA: attn | mlp | both."""
    out = []
    for n, m in bb.named_modules():
        if not isinstance(m, nn.Linear) or block_index(n) is None:
            continue
        is_attn = bool(_ATTN_RE.search(n))
        if (which == "attn" and not is_attn) or (which == "mlp" and is_attn):
            continue
        out.append(n)
    if not out:
        raise RuntimeError(f"no LoRA targets for lora_targets={which}")
    return out


def build_backbone(cfg: SweepConfig, grad_checkpointing: bool | None = None) -> tuple[nn.Module, dict]:
    """The pretrained backbone with the run's LoRA injected (PEFT model; only LoRA params are
    trainable). Parameter names equal ``SweepModel.backbone``'s, so a run checkpoint's
    ``backbone.*`` tensors load into it with the prefix stripped. ``info`` also lists the LoRA
    target modules (``lora_targets``)."""
    from peft import LoraConfig, get_peft_model

    bb, info = load_backbone(resolve_backbone(cfg.backbone.name))
    gc = cfg.backbone.grad_checkpointing if grad_checkpointing is None else grad_checkpointing
    if gc:
        enable_grad_ckpt(bb, info["is_timm"])
    a = cfg.adapt
    targets = lora_targets(bb, a.lora_targets)
    bb = get_peft_model(bb, LoraConfig(
        r=a.lora_rank, lora_alpha=a.lora_alpha if a.lora_alpha is not None else 2 * a.lora_rank,
        lora_dropout=a.lora_dropout, target_modules=targets, bias="none"))
    info["lora_targets"] = targets
    return bb, info


class ProjHead(nn.Module):
    """Linear -> [BN] -> GELU -> Linear (same construction order as
    ``spectra.models.encoder.ProjectionHead``)."""

    def __init__(self, in_dim: int, hidden: int, out: int, bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.GELU(), nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SweepModel(nn.Module):
    def __init__(self, cfg: SweepConfig):
        super().__init__()
        self.cfg = cfg
        bb, info = build_backbone(cfg)
        self.backbone = bb
        self.info = info
        self.is_timm = info["is_timm"]
        self.hidden = info["hidden"]
        self.num_blocks = info["num_blocks"]
        self.prefix = info["prefix"]
        self.norm_mean, self.norm_std = info["norm_mean"], info["norm_std"]
        self.lora_targets: list[str] = info["lora_targets"]
        # learned pools first, then one projector per head (in head order)
        self.pools = nn.ModuleDict({"gem": GeMPool()} if any(h.pool == "gem" for h in cfg.heads) else {})
        self.projectors = nn.ModuleList([ProjHead(self.hidden, h.proj_hidden, h.proj_dim, h.proj_bn)
                                         for h in cfg.heads])

    # ---------------------------------------------------------------------------------

    def prep(self, x01: torch.Tensor) -> torch.Tensor:
        from .data import normalize

        return normalize(x01, self.norm_mean, self.norm_std)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_timm:
            return self.backbone(x)
        return self.backbone(pixel_values=x).last_hidden_state

    def pool_tokens(self, tok: torch.Tensor, names) -> dict[str, torch.Tensor]:
        """{name: pooled feature} for name in cls | mean | clsmean | gem. Patch tokens exclude the
        CLS and register tokens (``self.prefix``)."""
        cls, patches = tok[:, 0, :], tok[:, self.prefix:, :]
        out: dict[str, torch.Tensor] = {}
        for n in names:
            if n == "cls":
                out[n] = cls
            elif n in ("mean", "clsmean"):
                mean = patches.mean(dim=1)
                out[n] = mean if n == "mean" else torch.cat([cls, mean], dim=1)
            elif n in self.pools:
                out[n] = self.pools[n](patches)
            else:
                raise ValueError(f"readout {n!r}")
        return out

    def head_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pooled features the loss heads read (one backbone pass)."""
        return self.pool_tokens(self.tokens(x), list(dict.fromkeys(h.pool for h in self.cfg.heads)))

    def base_features(self, x: torch.Tensor, names) -> dict[str, torch.Tensor]:
        """Readouts of the ORIGINAL pretrained backbone (LoRA disabled). Training mode and RNG
        state are preserved."""
        was = self.training
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.eval()
        try:
            with torch.no_grad(), self.backbone.disable_adapter():
                out = self.pool_tokens(self.tokens(x), names)
        finally:
            self.train(was)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return {k: v.detach() for k, v in out.items()}

    def trainable_summary(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        bb_tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": tot, "trainable": tr, "backbone_trainable": bb_tr,
                "lora_targets": len(self.lora_targets), "blocks": self.num_blocks,
                "hidden": self.hidden, "prefix_tokens": self.prefix, "repo": self.info["repo"]}
