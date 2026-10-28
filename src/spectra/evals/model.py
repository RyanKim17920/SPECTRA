"""``EvalModel``: one frozen encoder, exactly as the benchmarks must see it.

Built on :func:`spectra.evals.member.load_member` (base backbone or a sweep LoRA checkpoint),
with token-level access added here.

Numerics:

* weights stay **fp32** (the LoRA update is merged into them in fp32); ``precision`` only
  selects an autocast dtype for the forward (``"fp32"`` = no autocast). Outputs are returned
  as float32 (upcast after pooling).
* input ``x`` is the NORMALISED tensor from :meth:`EvalModel.transform` (per-backbone stats
  from the backbone's own pretrained cfg).
* tokens = timm ``forward_features`` (after the final norm) / HF ``last_hidden_state``;
  cls = token 0; mean / patch tokens exclude ALL prefix tokens (cls + registers).
* training heads (projectors, adversaries, pools) are never used.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from . import spec as S

# compute nodes usually have no hub access: default to offline inside SLURM jobs (must happen
# before huggingface_hub is imported, which reads the flag once at import)
if os.environ.get("SLURM_JOB_ID"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
POOLINGS = ("cls", "mean", "clsmean")


def verify_identity(spec: dict) -> None:
    """Re-hash the weights the spec names; raise if they changed since resolution."""
    again = S.resolve_spec(backbone=spec["backbone_repo"], ckpt=spec.get("ckpt_path"), step=spec.get("step"),
                           ema=bool(spec.get("ema")), key=spec["model_key"])
    if again["identity_sha256"] != spec["identity_sha256"]:
        raise RuntimeError(f"{spec['model_key']}: weights changed since the spec was resolved "
                           f"({spec['identity_sha256'][:12]} -> {again['identity_sha256'][:12]})")


def load_eval_model(spec, device: str | torch.device = "cuda", precision: str = "fp32",
                    verify: bool = True) -> "EvalModel":
    """``spec``: dict from :func:`spectra.evals.spec.resolve_spec` or a path to spec.json."""
    from .member import load_member

    spec = S.load_spec(spec)
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r} not in {list(PRECISIONS)}")
    if spec["kind"] not in ("base", "sweep_lora"):
        raise ValueError(f"unsupported spec kind {spec['kind']!r}")
    if verify:
        verify_identity(spec)
    device = torch.device(device)
    member = load_member(spec["backbone_repo"], spec.get("ckpt_path"), device=device)
    return EvalModel(spec, member, precision=precision).to(device).eval()


class EvalModel(nn.Module):
    def __init__(self, spec: dict, member, precision: str = "fp32"):
        super().__init__()
        self.spec = spec
        self.name = spec["model_key"]
        self.backbone_repo = spec["backbone_repo"]
        self.kind = spec["kind"]
        self.precision = precision
        self.autocast_dtype = PRECISIONS[precision]
        info = member.info
        self.raw = member.bb
        self.is_timm = bool(info["is_timm"])
        self.embed_dim = int(info["hidden"])
        self.num_prefix_tokens = int(info["prefix"])
        self.patch_size = member.patch_size
        self.mean = tuple(float(v) for v in info["norm_mean"])
        self.std = tuple(float(v) for v in info["norm_std"])
        self.adapter_rel_delta = member.adapter_rel_delta
        self.lora_merged = member.lora_merged
        if self.is_timm:
            got = int(getattr(self.raw, "num_prefix_tokens", 1))
            assert got == self.num_prefix_tokens, (got, self.num_prefix_tokens)
        for p in self.parameters():
            p.requires_grad_(False)

    # ---- geometry ----------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def feat_dim(self, pooling: str = "cls") -> int:
        return 2 * self.embed_dim if pooling == "clsmean" else self.embed_dim

    def num_patches(self, size: int = 224) -> int:
        return (size // self.patch_size) ** 2

    # ---- forward -----------------------------------------------------------------------
    def _ctx(self, x: torch.Tensor):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        if x.device.type == "cpu" and self.autocast_dtype == torch.float16:
            return contextlib.nullcontext()          # no fp16 autocast on CPU (tests only)
        return torch.autocast(x.device.type, dtype=self.autocast_dtype)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] full token sequence (prefix tokens first), autocast dtype."""
        with self._ctx(x):
            if self.is_timm:
                return self.raw.forward_features(x)
            return self.raw(pixel_values=x).last_hidden_state

    def cls_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens(x)[:, 0].float()

    def mean_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens(x)[:, self.num_prefix_tokens:].mean(1).float()

    def clsmean_features(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokens(x)
        return torch.cat([t[:, 0], t[:, self.num_prefix_tokens:].mean(1)], dim=-1).float()

    def features(self, x: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
        if pooling == "cls":
            return self.cls_features(x)
        if pooling == "mean":
            return self.mean_features(x)
        if pooling == "clsmean":
            return self.clsmean_features(x)
        raise ValueError(f"pooling {pooling!r} not in {POOLINGS}")

    def all_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """cls / mean / clsmean / patch from ONE forward."""
        t = self.tokens(x)
        P = self.num_prefix_tokens
        cls, mean = t[:, 0], t[:, P:].mean(1)
        return {"cls": cls.float(), "mean": mean.float(), "clsmean": torch.cat([cls, mean], -1).float(),
                "patch": t[:, P:].float()}

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] spatial tokens only (prefix / register tokens dropped)."""
        return self.tokens(x)[:, self.num_prefix_tokens:].float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_features(x)

    # aliases used by the THUNDER harness
    def classification_features(self, x):
        return self.cls_features(x)

    def segmentation_features(self, x):
        return self.patch_tokens(x)

    # ---- preprocessing -----------------------------------------------------------------
    def transform(self, resize: bool = True, size: int = 224, interpolation: str = "bicubic",
                  resize_to: int | None = None, timm_style: bool = False):
        """PIL -> normalised CHW float tensor.

        * ``resize=False``: ToTensor + Normalize only (HEST / CPTAC: tiles are already 224).
        * ``resize=True``: Resize(``resize_to or size``, interpolation) + CenterCrop(size) +
          ToTensor + Normalize (PathoROB extractor geometry).
        * ``timm_style=True``: exactly ``timm.data.create_transform(**resolve_data_config(
          backbone.pretrained_cfg))`` with this model's mean/std (THUNDER leaderboard
          preprocessing; H-optimus-0: bicubic Resize(256) + CenterCrop(224)).
        """
        from torchvision import transforms as T

        if timm_style:
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            # The checkpoint's OWN config.json pretrained_cfg + timm defaults -- NOT the
            # built model's pretrained_cfg, which for local gated builds is merged with the
            # architecture default (DINOv2 lvd142m: crop_pct 1.0, 518px) and would give
            # Resize(224) instead of THUNDER's Resize(256).
            repo = self.backbone_repo
            pc = _config_pretrained_cfg(repo)
            cfg = resolve_data_config(dict(pc), pretrained_cfg=dict(pc))
            if cfg.get("mean") is not None:
                own = (tuple(round(v, 6) for v in cfg["mean"]), tuple(round(v, 6) for v in cfg["std"]))
                ours = (tuple(round(v, 6) for v in self.mean), tuple(round(v, 6) for v in self.std))
                if pc.get("mean") is not None and own != ours:
                    print(f"[evals] NOTE {repo}: pretrained_cfg mean/std {own} != backbone {ours}; using the loaded backbone's",
                          flush=True)
            cfg["mean"], cfg["std"] = tuple(self.mean), tuple(self.std)
            cfg["input_size"] = (3, size, size)
            return create_transform(**cfg)
        norm = T.Normalize(mean=self.mean, std=self.std)
        if not resize:
            return T.Compose([T.ToTensor(), norm])
        modes = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR,
                 "nearest": T.InterpolationMode.NEAREST}
        return T.Compose([T.Resize(resize_to or size, interpolation=modes[interpolation]),
                          T.CenterCrop(size), T.ToTensor(), norm])

    # ---- provenance --------------------------------------------------------------------
    def describe(self) -> dict:
        return {"model_key": self.name, "kind": self.kind, "backbone_repo": self.backbone_repo,
                "precision": self.precision, "weights_dtype": "fp32", "embed_dim": self.embed_dim,
                "num_prefix_tokens": self.num_prefix_tokens, "patch_size": self.patch_size,
                "mean": list(self.mean), "std": list(self.std),
                "identity_sha256": self.spec.get("identity_sha256"),
                "adapter_rel_delta": self.adapter_rel_delta, "lora_merged": self.lora_merged}


def _config_pretrained_cfg(repo: str) -> dict:
    """``pretrained_cfg`` from the backbone's config.json (local gated dir or HF cache); {}
    for transformers backbones (no timm cfg -> timm defaults)."""
    ld = S.local_dirs().get(repo)
    p = Path(ld) / "config.json" if ld else None
    if p is None or not p.is_file():
        from huggingface_hub import try_to_load_from_cache

        c = try_to_load_from_cache(repo, "config.json")
        p = Path(c) if isinstance(c, str) else None
    if p is None or not p.is_file():
        return {}
    return dict(json.loads(p.read_text()).get("pretrained_cfg") or {})


def _cli(argv=None) -> int:
    """Smoke: ``python -m spectra.evals.model spec.json [--device cuda] [--precision fp16]``."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", default="fp32")
    a = ap.parse_args(argv)
    m = load_eval_model(a.spec, a.device, a.precision)
    x = torch.randn(2, 3, 224, 224, device=m.device)
    with torch.no_grad():
        f = {p: tuple(m.features(x, p).shape) for p in POOLINGS}
    print(json.dumps({**m.describe(), "shapes": f}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
