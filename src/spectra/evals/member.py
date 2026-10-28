"""Load ONE frozen encoder for evaluation: the pretrained ``base`` backbone or the backbone of a
``spectra.sweep`` LoRA run checkpoint.

Sweep run checkpoint = ``<run>/config.json`` + ``<run>/ckpt/step_XXXXXXX.safetensors`` (or the
``ema_step_*`` variant). Only the ``backbone.*`` tensors are used; the trainer's projectors /
adversary heads / pools are ignored. The LoRA update is folded into the fp32 weights
(``merge_and_unload``) after checking that the adapter really changes the output.

Needs torch + peft (imported lazily); ``spectra.sweep`` supplies the architecture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

PREFIX = "backbone."


@dataclass
class Member:
    bb: object                       # nn.Module (timm model or transformers model), fp32, eval
    info: dict                       # is_timm, hidden, num_blocks, prefix, norm_mean, norm_std, repo
    kind: str                        # "base" | "sweep_lora"
    ckpt: str | None = None
    adapter_rel_delta: float | None = None   # ||tuned - base|| / ||base|| of the CLS token
    lora_merged: bool = False
    load_seconds: float = 0.0

    @property
    def patch_size(self) -> int:
        if self.info["is_timm"]:
            ps = self.bb.patch_embed.patch_size
            return int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        return int(self.bb.config.patch_size)


def _tokens(bb, is_timm: bool, x):
    return bb.forward_features(x) if is_timm else bb(pixel_values=x).last_hidden_state


def load_sweep_backbone(ckpt: Path, repo: str):
    """-> (peft-wrapped backbone with the checkpoint's weights loaded, info)."""
    from safetensors.torch import load_file
    from spectra.sweep import config as C
    from spectra.sweep.model import build_backbone

    run_dir = ckpt.parent.parent
    cfg = C.load_dumped(run_dir / "config.json", tolerant=True)
    if C.resolve_backbone(cfg.backbone.name) != repo:
        raise ValueError(f"spec backbone {repo} != run backbone {cfg.backbone.name} ({run_dir})")
    if cfg.adapt.mode != "lora":
        raise ValueError(f"{run_dir}: adapt.mode={cfg.adapt.mode!r}; only LoRA runs are supported")
    bb, info = build_backbone(cfg, grad_checkpointing=False)
    sd = {k[len(PREFIX):]: v for k, v in load_file(str(ckpt)).items() if k.startswith(PREFIX)}
    if not sd:
        raise RuntimeError(f"{ckpt}: no '{PREFIX}*' tensors")
    res = bb.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"{ckpt}: {len(res.unexpected_keys)} backbone keys match nothing, e.g. {res.unexpected_keys[:3]}")
    missing = [n for n, p in bb.named_parameters() if p.requires_grad and n not in sd]
    if missing:
        raise RuntimeError(f"{ckpt}: {len(missing)} trainable tensors missing, e.g. {missing[:3]}")
    print(f"[member] sweep lora ckpt {ckpt.name} ({len(sd)} backbone tensors)", flush=True)
    return bb, info


def load_member(repo: str, ckpt_path: str | None = None, device="cpu") -> Member:
    """``ckpt_path=None`` -> the frozen pretrained ``repo``; else a sweep ``.safetensors`` file."""
    import torch

    t0 = time.time()
    if ckpt_path is None:
        from spectra.sweep.model import load_backbone

        bb, info = load_backbone(repo)
        kind = "base"
    else:
        bb, info = load_sweep_backbone(Path(ckpt_path), repo)
        kind = "sweep_lora"
    for q in bb.parameters():
        q.requires_grad_(False)
    bb = bb.to(device=device, dtype=torch.float32).eval()
    m = Member(bb=bb, info=dict(info), kind=kind, ckpt=ckpt_path)
    if kind == "sweep_lora":
        m.adapter_rel_delta = _assert_adapter_applied(m, device)
        # fold B@A*scaling into W in fp32: faster than the unmerged LoRA path
        m.bb = m.bb.merge_and_unload()
        m.lora_merged = True
    m.load_seconds = round(time.time() - t0, 1)
    return m


def _assert_adapter_applied(m: Member, device) -> float:
    """A silently-unapplied adapter is the classic wrong-number bug: check tuned != base on a
    fixed random input. Returns the relative L2 difference of the CLS token."""
    import torch

    peft_bb = m.bb
    if not (hasattr(peft_bb, "disable_adapter") and hasattr(peft_bb, "peft_config")):
        raise RuntimeError(f"{m.ckpt}: backbone is not PEFT-wrapped; LoRA was not applied")
    raw = peft_bb.base_model.model
    dev = torch.device(device)
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0)).to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        a = _tokens(raw, m.info["is_timm"], x)[:, 0].float()
        with peft_bb.disable_adapter():
            b = _tokens(raw, m.info["is_timm"], x)[:, 0].float()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-8))
    if rel < 1e-4:
        raise RuntimeError(f"{m.ckpt}: adapter has no effect (rel L2 {rel:.2e}); not applied?")
    return rel
