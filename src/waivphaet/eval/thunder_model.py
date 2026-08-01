"""THUNDER custom-model entry point -- our second RETENTION detector (PLAN.md §3 risk 1).

Used as::

    export THUNDER_BASE_DATA_FOLDER=/data/ryan.kim/thunder
    export WAIV_BACKBONE=kaiko-ai/midnight        # omit for owkin/phikon-v2
    export WAIV_POOLING=cls                       # omit: defaults PER BACKBONE, see below
    export WAIV_ADAPTER=/path/to/checkpoint_dir   # omit for the base model
    thunder benchmark custom:src/waivphaet/eval/thunder_model.py break_his knn

**Do not hardcode ``WAIV_POOLING`` in a sweep script.** The correct THUNDER pooling
depends on the backbone (see ``THUNDER_CLSMEAN_BACKBONES`` below): cls for phikon-v2,
clsmean for midnight. Leaving it unset picks the right one.

Why a module of its own rather than a function in ``thunder_adapter``
----------------------------------------------------------------------
THUNDER loads custom models by *file path* (``thunder/models/utils.py:32``,
``load_custom_model_from_file``): it execs the file, walks it with
``inspect.getmembers``, and instantiates **every** ``PretrainedModel`` subclass it finds,
raising if there is more than one. So this file must contain exactly one subclass and
nothing else that could drag a second one in -- which rules out putting it next to helper
classes. It also calls ``obj()`` with no arguments, so every knob has to arrive by
environment variable; there is nowhere to pass a checkpoint path.

Pooling: THUNDER, like HEST, publishes a CLS-only number
---------------------------------------------------------
``thunder/models/pretrained_models.py:303`` -- for ``phikon2`` their linear-probing
embedding is ``out.last_hidden_state[:, 0, :]`` (CLS, 1024-d; ``emb_dim: 1024`` in
``config/pretrained_model/phikon2.yaml``) and their segmentation embedding is
``out.last_hidden_state[:, 1:]`` (patch tokens). Same split as TRIDENT's HEST baseline.
So ``WAIV_POOLING=cls`` is the setting that is comparable to their published phikon2 row,
and ``clsmean`` is ours alone -- see ``hest_adapter`` for the same argument at length.

Segmentation is unaffected: it consumes patch tokens either way, so the two pooling modes
differ only on the five tile-level tasks.

Their transform for phikon2 comes from ``AutoImageProcessor`` (resize 224, rescale,
ImageNet normalise), which is what ``build_transform`` reproduces -- the same transform
already used for PathoROB and HEST, so a retention delta cannot be a preprocessing
artefact.

Normalisation, unlike resize/crop, is **per backbone**: ``build_transform`` takes the
backbone id and looks the stats up in ``BACKBONE_NORMALIZATION``. kaiko-ai/midnight's
card requires (0.5,0.5,0.5)/(0.5,0.5,0.5), not ImageNet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from thunder.models import PretrainedModel  # noqa: E402


#: THUNDER pooling is **per backbone**, and it is not our choice -- it is Waiv's.
#: arXiv:2607.22861 3, line 106: CLS+mean-pool concatenation was used for ALL models in
#: PathoROB, but in THUNDER only for Virchow2, AquaViT, H0-mini and **Midnight-12k**.
#: So phikon-v2 must be scored CLS-only here (which is also THUNDER's own published
#: phikon2 protocol, ``pretrained_models.py:303``) while midnight must be clsmean. Get
#: this backwards and the base-vs-fine-tuned rank sums are not comparable to their table.
THUNDER_CLSMEAN_BACKBONES = frozenset({"kaiko-ai/midnight"})


def _default_pooling(backbone: str | None) -> str:
    from waivphaet.models.encoder import DEFAULT_BACKBONE

    return "clsmean" if (backbone or DEFAULT_BACKBONE) in THUNDER_CLSMEAN_BACKBONES else "cls"


class WaivPhikonEncoder(PretrainedModel):
    """Our ``PhikonEncoder`` behind THUNDER's three-method interface.

    THUNDER validates ``self.name`` and ``self.emb_dim`` after construction and uses
    ``self.name`` as the results directory, so ``WAIV_RUN_NAME`` is what keeps one
    checkpoint's numbers from overwriting another's.
    """

    def __init__(self):
        super().__init__()

        from waivphaet.eval.hest_adapter import build_transform, load_encoder

        backbone = os.environ.get("WAIV_BACKBONE") or None
        pooling = os.environ.get("WAIV_POOLING") or _default_pooling(backbone)
        adapter = os.environ.get("WAIV_ADAPTER") or None
        checkpoint = os.environ.get("WAIV_CHECKPOINT") or None

        self.encoder = load_encoder(
            checkpoint,
            Path(adapter) if adapter else None,
            pooling,
            lora_rank=int(os.environ.get("WAIV_LORA_RANK", 16)),
            lora_alpha=int(os.environ.get("WAIV_LORA_ALPHA", 32)),
            proj_out_dim=int(os.environ.get("WAIV_PROJ_OUT_DIM", 512)),
            backbone=backbone,
        )
        self.t = build_transform(self.encoder.cfg.backbone)

        slug = self.encoder.cfg.backbone.split("/")[-1].replace("-", "").replace(".", "")
        default_name = f"waiv_{slug}_{pooling}" + ("" if not (adapter or checkpoint) else "_ft")
        self.name = os.environ.get("WAIV_RUN_NAME", default_name)
        # Derived from the backbone: phikon-v2 1024/2048, midnight 1536/3072.
        self.emb_dim = int(self.encoder.embed_dim)
        self.vlm = False

    def forward(self, x):
        """Not abstract, but ``adversarial_attack`` backprops through it -- so it has to
        be the differentiable path, not a ``no_grad`` wrapper."""
        return self.encoder.embed(x)

    def get_transform(self):
        return self.t

    def get_linear_probing_embeddings(self, x):
        """``(B, emb_dim)``. Feeds knn / linear_probing / simple_shot / calibration /
        adversarial_attack -- five of the six tasks in their rank sum."""
        return self.encoder.embed(x)

    def get_segmentation_embeddings(self, x):
        """``(B, tokens, hidden)`` patch tokens, matching their phikon2 branch. Pooling
        does not apply here, so this is identical under cls and clsmean."""
        out = self.encoder.backbone(pixel_values=x)
        return out.last_hidden_state[:, 1:]


def _selftest() -> None:
    """``python -m waivphaet.eval.thunder_model`` -- shape check without any dataset."""
    m = WaivPhikonEncoder().eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        lp, seg = m.get_linear_probing_embeddings(x), m.get_segmentation_embeddings(x)
    print(f"name={m.name} emb_dim={m.emb_dim} linear_probing={tuple(lp.shape)} "
          f"segmentation={tuple(seg.shape)}")
    assert lp.shape == (2, m.emb_dim), lp.shape
    assert seg.ndim == 3 and seg.shape[0] == 2, seg.shape
    print("thunder_model selftest OK")


if __name__ == "__main__":
    _selftest()
