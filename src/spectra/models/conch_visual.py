"""CONCH v1 (MahmoodLab/conch) image tower as a SPECTRA backbone.

CONCH is not a timm or ``transformers`` checkpoint: it is a CoCa model from the
``conch`` package (``conch.open_clip_custom``). Its visual tower is a timm ViT-B/16 at
448 px (``visual.trunk``) followed by an attentional pooler with ONE contrastive query
(``attn_pool_contrast`` -> ``ln_contrast``), which yields a 512-d vector, and then a
512x512 ``proj_contrast`` into the joint image-text space.

The readout choice (the part that matters)
------------------------------------------
Every downstream consumer of CONCH here -- SlideChat's patch features and our SAE -- reads
``model.encode_image(x, proj_contrast=False, normalize=False)``, i.e. the 512-d
attentional-pooler output. That vector, and nothing else, is therefore what this backbone
EXPORTS and what the contrastive loss is applied through:

* :class:`ConchVisual.forward` returns that vector as a length-1 "token sequence"
  ``(B, 1, 512)``, so ``SpectraEncoder`` runs with ``pooling="cls"`` and its ``_pool`` /
  ``embed`` return exactly ``encode_image(proj_contrast=False, normalize=False)``.
* The CLS / GeM split heads are NOT used for CONCH. The trunk's CLS token and patch tokens
  are not what any consumer reads -- CONCH's own pooler sits between them and the export --
  so a GeM/mean head over trunk tokens would make the loss optimise a vector nobody uses
  and leave the exported one to drift as a side effect. One ``ProjectionHead(512 -> 512)``
  sits on the exported vector instead, exactly like the legacy single-head path.
* ``proj_contrast`` is kept (frozen) so the zero-shot / text-alignment check can map the
  LoRA output into the joint space the UNTOUCHED text tower lives in.

What is trained: LoRA on ``trunk.blocks.N.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2}`` only.
The attentional pooler, ``ln_contrast`` and ``proj_contrast`` stay frozen, and the text
tower / multimodal decoder are never loaded into the trained module at all.

Resolution: PLISM tiles are 224 px; CONCH is a 448 px model (its ``preprocess`` resizes
with bicubic interpolation). Inputs smaller than 448 are bicubic-upsampled here, after
normalisation (bicubic is linear, so this equals resize-then-normalise up to clamping).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Hub spelling, used as the backbone id everywhere (run configs, adapter manifests).
CONCH_REPO_ID = "MahmoodLab/conch"
#: OpenAI CLIP statistics -- what CONCH's own ``create_model_from_pretrained`` preprocess uses.
CONCH_MEAN = (0.48145466, 0.4578275, 0.40821073)
CONCH_STD = (0.26862954, 0.26130258, 0.27577711)
CONCH_IMAGE_SIZE = 448
DEFAULT_CONCH_WEIGHTS = "/data/ryan.kim/ashby/weights/conch/pytorch_model.bin"


#: Backbone id that selects the TRUNK-TOKEN mode (conch-split pin): ConchVisual returns the
#: timm trunk's full ``(B, 1+784, 768)`` token sequence instead of the 512-d export, so the
#: paper's split CLS / GeM heads (and the canonical ``clsmean`` RI readout) see trunk tokens.
#: Same weights, same LoRA target names -- an adapter trained in either mode loads in both.
CONCH_TRUNK_ID = "MahmoodLab/conch-trunk"


def is_conch_trunk_backbone(backbone: str | None) -> bool:
    return bool(backbone) and backbone.lower() in {CONCH_TRUNK_ID.lower(), "conch-trunk"}


def is_conch_backbone(backbone: str | None) -> bool:
    return bool(backbone) and (
        backbone.lower() in {CONCH_REPO_ID.lower(), "conch", "conch_vit-b-16"}
        or is_conch_trunk_backbone(backbone)
    )


def conch_weights_path() -> str:
    """``SPECTRA_CONCH_WEIGHTS`` (or ``WAIV_CONCH_WEIGHTS``), else the machine default."""
    p = os.environ.get("SPECTRA_CONCH_WEIGHTS") or os.environ.get("WAIV_CONCH_WEIGHTS") or DEFAULT_CONCH_WEIGHTS
    if not Path(p).exists():
        raise FileNotFoundError(f"CONCH weights not found at {p}; set SPECTRA_CONCH_WEIGHTS")
    return p


def load_conch_full(weights: str | None = None):
    """The full CoCa model (image + text towers), eval mode, CPU. Used by evals only."""
    from conch.open_clip_custom import create_model_from_pretrained

    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": True})
    try:
        model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=weights or conch_weights_path())
    finally:
        torch.load = orig
    return model.eval()


class ConchVisual(nn.Module):
    """CONCH visual tower -> ``encode_image(proj_contrast=False, normalize=False)``.

    Holds the SAME module objects as ``coca.visual`` (trunk, attentional pooler, LN,
    proj_contrast) -- no copies, no re-implementation of the pooler -- so its output is
    bit-identical to ``coca.visual.forward_no_head`` on the same input.
    """

    def __init__(self, visual: nn.Module, image_size: int = CONCH_IMAGE_SIZE, mode: str = "export"):
        super().__init__()
        if mode not in ("export", "trunk"):
            raise ValueError(f"ConchVisual mode must be 'export' or 'trunk', got {mode!r}")
        #: "export" -> forward() = (B, 1, 512) export pseudo-sequence (conch-anchor pin);
        #: "trunk"  -> forward() = (B, 1+784, 768) trunk tokens (conch-split pin).
        self.mode = mode
        if not getattr(visual, "use_attentional_pool_contrast", False):
            raise RuntimeError("expected a CONCH VisualModel with attentional contrastive pooling")
        self.trunk = visual.trunk
        self.trunk_kwargs = dict(getattr(visual, "trunk_kwargs", {}) or {})
        self.attn_pool_contrast = visual.attn_pool_contrast
        self.ln_contrast = visual.ln_contrast
        self.proj_contrast = visual.proj_contrast  # frozen; zero-shot only
        self.image_size = int(image_size)
        # Geometry, read off the built trunk.
        self.embed_dim = int(self.ln_contrast.normalized_shape[0])  # 512 -- the export width
        self.trunk_dim = int(self.trunk.num_features)  # 768
        self.blocks = self.trunk.blocks
        ps = self.trunk.patch_embed.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        # Never trained: only the trunk carries LoRA. (Base weights are frozen by PEFT
        # anyway; this makes the pooler/proj frozen even with use_lora=False.)
        for m in (self.attn_pool_contrast, self.ln_contrast):
            for p in m.parameters():
                p.requires_grad_(False)
        self.proj_contrast.requires_grad_(False)

    # --- grad checkpointing, timm-style switch the encoder already knows how to call ---
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.trunk.set_grad_checkpointing(enable)

    @property
    def grad_checkpointing(self) -> bool:
        return bool(getattr(self.trunk, "grad_checkpointing", False))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        return x

    def trunk_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised NCHW float -> ``(B, 1+784, 768)``: the (final-LN'd) trunk tokens the
        attentional pooler reads. Token 0 is CLS; CONCH's trunk has no register tokens."""
        return self.trunk(self._resize(x), **self.trunk_kwargs)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised NCHW float -> ``(B, 512)``, == ``encode_image(proj_contrast=False, normalize=False)``."""
        t = self.trunk_tokens(x)
        p = self.attn_pool_contrast(t)[:, 0]
        return self.ln_contrast(p)

    def clsmean(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised NCHW float -> ``(B, 1536)`` = cat(trunk CLS, mean of trunk patch tokens):
        the canonical SPECTRA/PathoROB ``clsmean`` readout, applied to CONCH's trunk."""
        t = self.trunk_tokens(x)
        return torch.cat([t[:, 0], t[:, 1:].mean(dim=1)], dim=1)

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """The joint-space projection ``encode_image(proj_contrast=True)`` applies (un-normalised)."""
        return pooled @ self.proj_contrast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "trunk":
            return self.trunk_tokens(x)
        # (B, 1, 512): a one-token "sequence" so SpectraEncoder's pooling='cls' path returns
        # exactly the exported vector. See the module docstring.
        return self.pooled(x).unsqueeze(1)


def build_conch_visual(weights: str | None = None, mode: str = "export") -> ConchVisual:
    """Load CONCH, keep ONLY the visual tower (text tower dropped), wrap it."""
    full = load_conch_full(weights)
    vis = ConchVisual(full.visual, mode=mode)
    n = sum(p.numel() for p in vis.parameters())
    print(f"[encoder] CONCH visual tower: trunk blocks={len(vis.blocks)} trunk_dim={vis.trunk_dim} "
          f"export_dim={vis.embed_dim} mode={vis.mode} patch={vis.patch_size} img={vis.image_size} params={n}", flush=True)
    del full
    return vis
