"""Phikon-v2 backbone with LoRA across *all* transformer blocks, plus a wide projection head.

Design choices and where they come from
---------------------------------------
* **Base = ``owkin/phikon-v2``** (Dinov2, ViT-L/16, 24 blocks, 1024-d, ungated, 1.21 GB).
  PLAN.md 1: weakest starting point on PathoROB (Avg RI 0.469, Camelyon 0.019) and the
  largest published gain (-> 0.806 / 0.702), so it is the cheapest informative base.

* **LoRA on every block, not head-only.** PLAN.md 2 + 0 (their Fig 4): base H-Optimus-0
  only develops cross-scanner matching in the last few blocks, and fine-tuning pushes
  that ~8 blocks earlier. Invariance has to build *across depth*, so head-only tuning is
  ruled out. We therefore target ``query/key/value/attention.output.dense/mlp.fc1/mlp.fc2``
  in **all 24 blocks**. LoRA rather than full FT is our deliberate anti-forgetting
  divergence (PLAN.md 2): it bounds drift on a backbone that saw 456M tiles, cuts
  memory, and merges back to full weights afterwards. Full FT is the escalation
  (PLAN.md 3 phase 9).

* **Projection width >= 512.** PLAN.md 2: ScanGen used hidden 48/96 for binary MIL,
  far too narrow for retrieval among 16k tiles. Default 1024 hidden / 512 out.

* **Pooling defaults to ``clsmean``** (CLS token concatenated with the mean of patch
  tokens, 2048-d) because that is exactly what PathoROB's own ``phikonv2_clsmean`` entry
  uses -- matching it is what makes our reproduced Avg RI 0.469 gate (PLAN.md 3 phase 5)
  meaningful. ``cls`` gives the plain 1024-d embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DEFAULT_BACKBONE = "owkin/phikon-v2"

# ImageNet stats -- what phikon-v2's own BitImageProcessor uses, and what both PathoROB
# and plismbench feed it. Keep identical or the reproduced baseline drifts for free.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Dinov2 block sub-modules that carry the bulk of the parameters. Named so that
#: ``target_modules`` matches in *every* block, which is the point (PLAN.md 2).
LORA_TARGET_MODULES: tuple[str, ...] = (
    "query", "key", "value", "dense", "fc1", "fc2",
)


@dataclass
class EncoderConfig:
    backbone: str = DEFAULT_BACKBONE
    pooling: str = "clsmean"  # "cls" | "mean" | "clsmean"
    # --- LoRA (PLAN.md 2: all blocks; rank is one of the unknown hyperparameters,
    # PLAN.md 3 risk 4 -> sweep it)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    lora_blocks: tuple[int, ...] | None = None  # None = ALL blocks (the default, on purpose)
    # --- projection head (PLAN.md 2: >= 512, NOT ScanGen's 48)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512
    proj_use_bn: bool = True
    # --- misc
    freeze_backbone: bool = False  # True => frozen-feature probe (PLAN.md 3 phase 6)
    dtype: str = "float32"
    extra: dict = field(default_factory=dict)


def normalize_uint8(x: torch.Tensor) -> torch.Tensor:
    """``(B, 224, 224, 3)`` uint8 -> ``(B, 3, 224, 224)`` normalised float.

    The pair loader hands us raw uint8 NHWC straight off the memmap (no PIL, no resize:
    PLISM tiles are already exactly 224x224), so this is the whole preprocessing stack.
    """
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class ProjectionHead(nn.Module):
    """2-layer MLP projector. InfoNCE is applied on its output, not on the backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 512, use_bn: bool = True):
        super().__init__()
        if out_dim < 512:
            raise ValueError(
                f"proj_out_dim={out_dim} < 512; PLAN.md 2 rules out narrow heads "
                "(ScanGen's 48/96) for 16k-tile retrieval"
            )
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.GELU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _lora_target_names(model: nn.Module, cfg: EncoderConfig) -> list[str]:
    """Resolve full module names so we can (a) hit every block and (b) *prove* we did.

    PEFT's ``target_modules`` accepts bare suffixes, but a bare suffix silently matches
    whatever happens to share the name. We enumerate explicit full names instead, then
    assert the per-block count -- head-only adaptation is the failure mode PLAN.md 2
    explicitly rules out, and it would otherwise be invisible.
    """
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in cfg.lora_target_modules:
            continue
        if cfg.lora_blocks is not None:
            parts = name.split(".")
            try:
                blk = int(parts[parts.index("layer") + 1])
            except (ValueError, IndexError):
                continue
            if blk not in cfg.lora_blocks:
                continue
        names.append(name)
    return names


class PhikonEncoder(nn.Module):
    """Backbone (optionally LoRA-adapted) + projection head.

    ``forward`` returns ``(embedding, projection)``:

    * ``embedding`` -- pooled backbone output. This is what goes to PathoROB / plismbench.
    * ``projection`` -- L2-normalisable head output. This is what InfoNCE sees.
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        backbone = AutoModel.from_pretrained(cfg.backbone)
        self.hidden_size = int(backbone.config.hidden_size)
        self.num_blocks = int(backbone.config.num_hidden_layers)

        if cfg.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        self.lora_target_names: list[str] = []
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            self.lora_target_names = _lora_target_names(backbone, cfg)
            if not self.lora_target_names:
                raise RuntimeError(f"no LoRA targets matched {cfg.lora_target_modules}")
            covered = {
                int(n.split(".")[n.split(".").index("layer") + 1])
                for n in self.lora_target_names
                if "layer" in n.split(".")
            }
            expected = set(cfg.lora_blocks) if cfg.lora_blocks is not None else set(range(self.num_blocks))
            if covered != expected:
                raise RuntimeError(
                    f"LoRA covers blocks {sorted(covered)} but expected {sorted(expected)}; "
                    "PLAN.md 2 requires adaptation across the full depth, not head-only"
                )
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=self.lora_target_names,
                    bias="none",
                ),
            )
        self.backbone = backbone

        self.embed_dim = self.hidden_size * (2 if cfg.pooling == "clsmean" else 1)
        self.projector = ProjectionHead(
            self.embed_dim, cfg.proj_hidden_dim, cfg.proj_out_dim, cfg.proj_use_bn
        )

    # --- pooling ------------------------------------------------------------------

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        cls, patches = tokens[:, 0, :], tokens[:, 1:, :]
        if self.cfg.pooling == "cls":
            return cls
        if self.cfg.pooling == "mean":
            return patches.mean(dim=1)
        if self.cfg.pooling == "clsmean":
            return torch.cat([cls, patches.mean(dim=1)], dim=1)
        raise ValueError(f"unknown pooling {self.cfg.pooling!r}")

    # --- forward ------------------------------------------------------------------

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC (or normalised float NCHW) -> pooled embedding ``(B, embed_dim)``."""
        if images.dtype == torch.uint8 or images.shape[-1] == 3:
            images = normalize_uint8(images)
        out = self.backbone(pixel_values=images)
        return self._pool(out.last_hidden_state)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(images)
        return emb, self.projector(emb)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
        """Inference-time embedding for the eval adapters (PathoROB / plismbench)."""
        self.eval()
        emb = self.embed(images)
        return F.normalize(emb, dim=-1) if l2_normalize else emb

    # --- housekeeping -------------------------------------------------------------

    def trainable_parameter_summary(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
            "lora_targets": len(self.lora_target_names),
            "blocks": self.num_blocks,
        }

    def merge_lora(self) -> nn.Module:
        """Merge LoRA deltas into the base weights -> a plain Dinov2 checkpoint.

        PLAN.md 2: LoRA "merges to full weights afterwards", which is what lets the eval
        adapters and any downstream user load us as an ordinary ``owkin/phikon-v2``.
        """
        if not self.cfg.use_lora:
            return self.backbone
        return self.backbone.merge_and_unload()


def build_encoder(**kwargs) -> PhikonEncoder:
    return PhikonEncoder(EncoderConfig(**kwargs))
