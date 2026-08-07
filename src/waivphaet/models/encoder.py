"""A *model-agnostic* ViT backbone with LoRA across **all** transformer blocks, plus a
wide projection head.

Design choices and where they come from
---------------------------------------
* **Base is a parameter, not a constant.** The default is ``owkin/phikon-v2`` (Dinov2,
  ViT-L/16, 24 blocks, 1024-d, ungated, 1.21 GB) -- PLAN.md 1: weakest starting point
  on PathoROB (Avg RI 0.469, Camelyon 0.019) and the largest published gain
  (-> 0.806 / 0.702), so it is the cheapest informative base. But the pipeline has to
  generalise, so ``hidden_size``, ``num_hidden_layers`` and ``patch_size`` are read off
  the *loaded* config and the LoRA target set is **discovered by module-name pattern**.
  Nothing about ViT-L/16 is baked in. Second validated backbone: ``kaiko-ai/midnight``
  (Dinov2 ViT-g/14, 40 blocks, 1536-d, SwiGLU FFN, MIT, 4.55 GB).

* **LoRA on every block, not head-only.** PLAN.md 2 + 0 (their Fig 4): base H-Optimus-0
  only develops cross-scanner matching in the last few blocks, and fine-tuning pushes
  that ~8 blocks earlier. Invariance has to build *across depth*, so head-only tuning is
  ruled out. LoRA rather than full FT is our deliberate anti-forgetting divergence
  (PLAN.md 2): it bounds drift on a backbone that saw 456M tiles, cuts memory, and
  merges back to full weights afterwards. Full FT is the escalation (PLAN.md 3 phase 9).

  **Why discovery rather than a fixed name list.** ``fc1``/``fc2`` is the HF Dinov2 MLP
  naming, but ``kaiko-ai/midnight`` sets ``use_swiglu_ffn=True`` and its FFN linears are
  ``mlp.weights_in`` / ``mlp.weights_out``. A fixed list would still have matched
  ``query/key/value/dense`` -- so the block-coverage assertion would have *passed* while
  silently adapting attention only and leaving 2/3 of the block parameters frozen. The
  failure mode is invisible in every log line and reads downstream as "LoRA had less
  effect on ViT-g". So we discover, and we assert the per-block match count is uniform
  and non-empty, and we log it.

* **Projection width >= 512.** PLAN.md 2: ScanGen used hidden 48/96 for binary MIL,
  far too narrow for retrieval among 16k tiles. Default 1024 hidden / 512 out. The
  projector's *input* width is ``embed_dim``, i.e. it is tied to the **training**
  pooling -- see ``build_model`` in ``scripts/extract_pathorob_features.py``.

* **Pooling defaults to ``clsmean``** (CLS token concatenated with the mean of patch
  tokens) because that is exactly what PathoROB's own ``phikonv2_clsmean`` entry uses --
  matching it is what makes our reproduced Avg RI 0.469 gate (PLAN.md 3 phase 5)
  meaningful. ``embed_dim`` is *derived*: ``hidden`` for cls/mean, ``2*hidden`` for
  clsmean -- 1024/2048 on phikon-v2, 1536/3072 on midnight.
"""

from __future__ import annotations

import re
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

#: Symmetric [-1, 1] normalisation.
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

#: **Normalisation is a property of the backbone, not of the pipeline.**
#:
#: phikon-v2 wants ImageNet stats (its own ``BitImageProcessor``), and PathoROB's
#: ``Phikonv2ModelWrapper.get_preprocess`` uses exactly those -- which is why our Avg RI
#: reproduces theirs to 6 decimals. ``kaiko-ai/midnight`` does **not**: its model card is
#: explicit -- "trained on 224x224 images normalized with a mean of (0.5, 0.5, 0.5) and a
#: standard deviation of (0.5, 0.5, 0.5). Please ensure you apply these exact
#: normalization parameters."
#:
#: Feeding midnight ImageNet stats does not crash and does not look wrong anywhere: it
#: just shifts and rescales every channel, quietly costing base accuracy. It would make
#: our base-midnight row disagree with Waiv's published 0.759 for a reason that has
#: nothing to do with the harness being faithful -- i.e. exactly the check we are running
#: it for. So it is table-driven and travels with the backbone id.
#:
#: This table is an **override**, not the only source: it wins over whatever the backbone's
#: own HF preprocessor says. These two entries are the ones our published numbers were
#: produced with, and they must never move because a hub config was re-uploaded.
BACKBONE_NORMALIZATION: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "owkin/phikon-v2": (IMAGENET_MEAN, IMAGENET_STD),
    "kaiko-ai/midnight": (HALF_MEAN, HALF_STD),
}


def _hf_preprocessor_normalization(
    backbone: str,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """``(mean, std)`` read off the backbone's own HF image processor, or ``None``.

    Most model repos publish ``image_mean``/``image_std`` in ``preprocessor_config.json``,
    which is the same thing the model card states -- so for a new backbone this is the
    authoritative value, not a guess. Any failure (no processor, offline hub, missing
    fields) returns ``None`` so the caller can still fall through to the override or to a
    hard error; a hub blip must not crash a run whose stats are already pinned above.
    """
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(backbone)
        mean, std = getattr(proc, "image_mean", None), getattr(proc, "image_std", None)
    except Exception as exc:  # noqa: BLE001 -- any hub/config failure is just "not derivable"
        print(f"[encoder] AutoImageProcessor lookup failed for {backbone!r}: {exc}", flush=True)
        return None
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        return None
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def normalization_for(backbone: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std for ``backbone``: explicit override, else derived from HF, else refuse.

    A wrong-but-plausible normalisation is invisible in every log and every shape check --
    the old ImageNet fallback only printed a warning, which is unreadable in a 90k-line
    SLURM log. So there is no fallback any more: either we know the stats or we stop.
    """
    backbone = backbone or DEFAULT_BACKBONE
    if backbone in BACKBONE_NORMALIZATION:
        return BACKBONE_NORMALIZATION[backbone]
    derived = _hf_preprocessor_normalization(backbone)
    if derived is not None:
        print(
            f"[encoder] normalisation for {backbone!r} derived from its HF image processor: "
            f"mean={derived[0]} std={derived[1]} (no BACKBONE_NORMALIZATION override)",
            flush=True,
        )
        return derived
    raise RuntimeError(
        f"no normalisation for backbone {backbone!r}: it has no BACKBONE_NORMALIZATION "
        "entry and its HF image processor did not yield image_mean/image_std. Read the "
        "model card and add an explicit entry to BACKBONE_NORMALIZATION in "
        "src/waivphaet/models/encoder.py -- defaulting to ImageNet stats here would "
        "silently cost accuracy (e.g. kaiko-ai/midnight needs (0.5,0.5,0.5))."
    )

#: Superset of leaf module names that carry the bulk of a transformer block's parameters,
#: across the ViT namings we care about. This is a *candidate* set: the actual target set
#: is the intersection with what the loaded backbone really has, computed per block.
#:
#:   HF Dinov2 / BERT-style : query, key, value, dense, fc1, fc2
#:   SwiGLU FFN (ViT-g)     : weights_in, weights_out
#:   timm / fused-qkv ViTs  : qkv, proj
#:   HF CLIP-style attn     : q_proj, k_proj, v_proj, out_proj
#:   HF ViT, transformers>=5: q_proj, k_proj, v_proj, o_proj  (note o_proj, NOT out_proj)
LORA_CANDIDATE_MODULES: tuple[str, ...] = (
    "query", "key", "value", "dense", "fc1", "fc2",
    "weights_in", "weights_out",
    "qkv", "proj",
    "q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
)

#: Backwards-compatible alias. The old fixed phikon-v2 list; kept so that an explicit
#: ``lora_target_modules=LORA_TARGET_MODULES`` still means what it used to.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "query", "key", "value", "dense", "fc1", "fc2",
)

#: How a transformer block index appears in a module path. Covers HF (``encoder.layer.N``,
#: ``encoder.layers.N``), timm (``blocks.N``) and GPT-style (``h.N``).
_BLOCK_RE = re.compile(r"(?:^|\.)(?:layer|layers|blocks|block|h)\.(\d+)(?:\.|$)")


def _block_index(name: str) -> int | None:
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


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
    #: ``None`` (the default) = **discover** the target leaf names from the loaded
    #: backbone by intersecting ``LORA_CANDIDATE_MODULES`` with the block Linears it
    #: actually has. Pass an explicit tuple only to deliberately narrow the set.
    lora_target_modules: tuple[str, ...] | None = None
    lora_blocks: tuple[int, ...] | None = None  # None = ALL blocks (the default, on purpose)
    # --- projection head (PLAN.md 2: >= 512, NOT ScanGen's 48)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512
    proj_use_bn: bool = True
    # --- memory/compute trade
    #: Recompute block activations in the backward pass instead of storing them. The
    #: same-condition constraint makes the *in-group* negative count ``group_size - 1``,
    #: so the only way to buy more negatives is a bigger forward batch -- and at
    #: ~0.21 GiB/image (measured, 128 img/step -> 27.35 GiB) a plain ViT-L/16 run caps
    #: out near 340 images on an 80 GiB H100. Checkpointing drops that to ~0.02 GiB/image
    #: for roughly +35% step time, which is the trade that makes 384-anchor groups
    #: possible at all. Off by default so the smoke-run numbers stay reproducible.
    grad_checkpointing: bool = False
    # --- misc
    freeze_backbone: bool = False  # True => frozen-feature probe (PLAN.md 3 phase 6)
    dtype: str = "float32"
    extra: dict = field(default_factory=dict)


def normalize_uint8(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """``(B, 224, 224, 3)`` uint8 -> ``(B, 3, 224, 224)`` normalised float.

    The pair loader hands us raw uint8 NHWC straight off the memmap (no PIL, no resize:
    PLISM tiles are already exactly 224x224), so this is the whole preprocessing stack.
    ``mean``/``std`` default to ImageNet for backwards compatibility; callers inside the
    encoder pass the *backbone's* stats (``normalization_for``).
    """
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


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


def _lora_target_names(model: nn.Module, cfg: EncoderConfig) -> tuple[list[str], dict[int, int], tuple[str, ...]]:
    """Resolve full module names so we can (a) hit every block and (b) *prove* we did.

    PEFT's ``target_modules`` accepts bare suffixes, but a bare suffix silently matches
    whatever happens to share the name. We enumerate explicit full names instead, then
    assert the per-block count -- head-only adaptation is the failure mode PLAN.md 2
    explicitly rules out, and it would otherwise be invisible.

    Architecture-agnostic by construction: we walk the backbone's ``nn.Linear`` modules,
    keep the ones that live inside a numbered transformer block, and select by leaf name
    against ``cfg.lora_target_modules`` -- or, when that is ``None``, against the
    *candidate* superset, which is what makes ``fc1/fc2`` (Dinov2 MLP) and
    ``weights_in/weights_out`` (SwiGLU FFN, ViT-g) both resolve without a per-model list.

    Returns ``(names, per_block_counts, resolved_leaf_names)``.
    """
    candidates = cfg.lora_target_modules or LORA_CANDIDATE_MODULES
    names: list[str] = []
    per_block: dict[int, int] = {}
    leaves: set[str] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in candidates:
            continue
        blk = _block_index(name)
        if blk is None:
            # A Linear with a matching leaf name outside any transformer block (a head,
            # a pooler). Never adapt it: LoRA-on-the-head is exactly what PLAN.md 2 rules
            # out, and it would inflate the "targets" count while adapting no depth.
            continue
        if cfg.lora_blocks is not None and blk not in cfg.lora_blocks:
            continue
        names.append(name)
        per_block[blk] = per_block.get(blk, 0) + 1
        leaves.add(leaf)
    return names, per_block, tuple(sorted(leaves))


class WaivEncoder(nn.Module):
    """Backbone (optionally LoRA-adapted) + projection head.

    ``forward`` returns ``(embedding, projection)``:

    * ``embedding`` -- pooled backbone output. This is what goes to PathoROB / plismbench.
    * ``projection`` -- L2-normalisable head output. This is what InfoNCE sees.
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        backbone = AutoModel.from_pretrained(cfg.backbone)
        # Everything geometric is READ OFF THE LOADED CONFIG. Nothing here may be a
        # literal: phikon-v2 is 1024/24/patch16, midnight is 1536/40/patch14.
        bc = backbone.config
        self.hidden_size = int(bc.hidden_size)
        self.num_blocks = int(bc.num_hidden_layers)
        self.patch_size = int(getattr(bc, "patch_size", 0)) or None
        #: Tokens the backbone emits at ``image_size``, per its own config. We feed 224px
        #: everywhere (PathoROB/HEST/THUNDER/PLISM all resize to 224), and Dinov2
        #: interpolates its position embeddings, so the *runtime* patch count is
        #: ``(224/patch_size)**2`` -- 196 on phikon-v2 (16), 256 on midnight (14).
        self.config_image_size = int(getattr(bc, "image_size", 0)) or None
        self.model_type = str(getattr(bc, "model_type", "unknown"))
        self.norm_mean, self.norm_std = normalization_for(cfg.backbone)

        if cfg.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        self.lora_target_names: list[str] = []
        self.lora_target_leaves: tuple[str, ...] = ()
        self.lora_per_block: dict[int, int] = {}
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            self.lora_target_names, self.lora_per_block, self.lora_target_leaves = (
                _lora_target_names(backbone, cfg)
            )
            # A silently-empty target set trains NOTHING and reads downstream as "the
            # method had no effect on this backbone". Refuse, loudly, with the evidence.
            if not self.lora_target_names:
                sample = sorted({
                    n.rsplit(".", 1)[-1]
                    for n, m in backbone.named_modules()
                    if isinstance(m, nn.Linear) and _block_index(n) is not None
                })
                raise RuntimeError(
                    f"no LoRA targets matched on backbone {cfg.backbone!r} "
                    f"(model_type={self.model_type}): candidates="
                    f"{cfg.lora_target_modules or LORA_CANDIDATE_MODULES}, but the block "
                    f"Linears are named {sample}. Add the missing names to "
                    "LORA_CANDIDATE_MODULES -- an empty target set trains nothing."
                )
            covered = set(self.lora_per_block)
            expected = set(cfg.lora_blocks) if cfg.lora_blocks is not None else set(range(self.num_blocks))
            if covered != expected:
                raise RuntimeError(
                    f"LoRA covers blocks {sorted(covered)} but expected {sorted(expected)}; "
                    "PLAN.md 2 requires adaptation across the full depth, not head-only"
                )
            # Uniformity is the second half of the guard. A ragged count means the leaf
            # names differ between blocks, i.e. some blocks are only partly adapted --
            # which the block-coverage check above cannot see.
            counts = set(self.lora_per_block.values())
            if len(counts) != 1:
                ragged = {b: c for b, c in sorted(self.lora_per_block.items())}
                raise RuntimeError(
                    f"LoRA match count is not uniform across blocks: {ragged}"
                )
            # Third guard: an in-block Linear that matched NOTHING. Neither check above
            # can see this -- a leaf dropped uniformly from every block leaves the target
            # set non-empty and the per-block count uniform, so the adapter looks healthy
            # and is quietly missing a whole projection. owkin/phikon under transformers>=5
            # names its attention output o_proj while the candidate list had only out_proj,
            # which would have skipped all 12 attention outputs with no error at all.
            in_block_leaves = {
                n.rsplit(".", 1)[-1]
                for n, m in backbone.named_modules()
                if isinstance(m, nn.Linear) and _block_index(n) is not None
            }
            dropped = in_block_leaves - set(self.lora_target_leaves)
            if dropped:
                raise RuntimeError(
                    f"LoRA silently skipped in-block Linears on {cfg.backbone!r} "
                    f"(model_type={self.model_type}): {sorted(dropped)} matched no "
                    f"candidate. Targeted {sorted(self.lora_target_leaves)}. Add the "
                    "missing names to LORA_CANDIDATE_MODULES, or pass an explicit "
                    "lora_target_modules to state that skipping them is intended -- "
                    "a partly-adapted block reads downstream as a weak method, not a bug."
                )
            print(
                f"[encoder] backbone={cfg.backbone} type={self.model_type} "
                f"hidden={self.hidden_size} blocks={self.num_blocks} "
                f"patch={self.patch_size} | LoRA targets={len(self.lora_target_names)} "
                f"= {counts.pop()}/block x {self.num_blocks} blocks, "
                f"leaves={list(self.lora_target_leaves)}",
                flush=True,
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
        if cfg.grad_checkpointing:
            # use_reentrant=False: the reentrant autograd.Function variant needs at least
            # one input with requires_grad, and with LoRA the embedding output is frozen,
            # so the reentrant path silently produces *no* gradient for the early blocks.
            target = getattr(backbone, "base_model", backbone)
            target = getattr(target, "model", target)
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if not getattr(target, "is_gradient_checkpointing", False):
                raise RuntimeError(
                    "gradient checkpointing did not take on the backbone; refusing to run "
                    "with the batch size it was requested for"
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
            images = normalize_uint8(images, self.norm_mean, self.norm_std)
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
            "backbone": self.cfg.backbone,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
            "lora_targets": len(self.lora_target_names),
            "lora_targets_per_block": (
                max(self.lora_per_block.values()) if self.lora_per_block else 0
            ),
            "lora_target_leaves": list(self.lora_target_leaves),
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


#: The class was phikon-v2-specific when it was written; it no longer is. Alias kept so
#: saved checkpoints, the THUNDER entry point and any external caller keep importing.
PhikonEncoder = WaivEncoder


def build_encoder(**kwargs) -> WaivEncoder:
    return WaivEncoder(EncoderConfig(**kwargs))
