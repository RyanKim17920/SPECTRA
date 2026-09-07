"""The backbone registry: **one entry per model, and nothing about a model elsewhere.**

Adding a sixth or seventh backbone used to mean editing four tables in
``models/encoder.py`` (normalisation, local directory, timm kwargs, weight-file names),
a fifth in ``eval/thunder_protocol.py`` (published pooling), and a sixth in every
per-checkpoint eval cell.  Six places, no error if you missed one -- a missing
normalisation row silently costs accuracy, a missing pooling row silently halves the
feature width, and both produce a number that looks like a result.

So every per-backbone fact lives in :data:`BACKBONES` here, and the tables that used to
hold them are now views over it.  **To add a backbone, add one ``Backbone`` entry.**
See ``docs/archive/NEW_MODEL.md`` for the walkthrough and for what each field is proved against.

What this registry does NOT decide
----------------------------------
``loader`` is recorded, but ``encoder.is_timm_backbone`` still dispatches on the repo's
own ``config.json`` and not on this field.  That is deliberate and predates this file: a
name-keyed loader list means every new timm checkpoint takes the ``AutoModel`` path and
dies with "Unrecognized model", naming nothing about the real architecture.  The field is
here so a wrong assumption can be *checked* (``scripts/check_backbone_registry.py``),
never so it can be *assumed*.

Local weight directories
------------------------
``local_subdir`` is relative to ``SPECTRA_INPUTS`` (see :mod:`spectra.paths`), because
the absolute location of a gated checkpoint is a property of the machine, not of the
model.  ``SPECTRA_BACKBONE_LOCAL_DIRS="repo=/dir"`` still overrides any of them per job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectra.paths import INPUTS

#: ImageNet statistics -- what phikon-v2's own ``BitImageProcessor`` and PathoROB use.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Symmetric [-1, 1] normalisation, which kaiko-ai/midnight's model card demands.
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

#: LoRA leaf-module paths inside one transformer block, per FFN family.
_HF_DINOV2_MLP = (
    "attention.attention.query",
    "attention.attention.key",
    "attention.attention.value",
    "attention.output.dense",
    "mlp.fc1",
    "mlp.fc2",
)
_HF_DINOV2_SWIGLU = (
    "attention.attention.query",
    "attention.attention.key",
    "attention.attention.value",
    "attention.output.dense",
    "mlp.weights_in",
    "mlp.weights_out",
)
_TIMM_VIT = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


@dataclass(frozen=True)
class Backbone:
    """Everything this project needs to know about one pretrained backbone.

    Fields, and what each one is *for*:

    ``repo_id``
        HuggingFace repo id. The identity used everywhere -- run configs, adapter
        manifests, cells -- so it must be the hub spelling, never a nickname.
    ``loader``
        ``"hf"`` (transformers ``AutoModel``) or ``"timm"``. Recorded, not dispatched on;
        see the module docstring.
    ``weights_file``
        The file inside a local directory that holds the weights. ``.safetensors`` and
        ``.bin`` checkpoints load by different calls, so this is not cosmetic.
    ``architecture``
        timm architecture name for ``timm.create_model``. ``None`` for HF backbones.
    ``timm_kwargs``
        Architecture kwargs the *name* does not carry. Mandatory for any locally-served
        timm backbone: ``pretrained=False`` on a bare name applies neither the repo's
        ``model_args`` nor its ``pretrained_cfg``, so an omitted ``embed_dim`` or
        ``mlp_layer`` builds a clean model that fails the strict load -- or worse, loads.
        Layer classes are named as STRINGS so importing this module does not drag in timm.
    ``local_subdir``
        Directory under ``SPECTRA_INPUTS`` serving this backbone, or ``None`` for
        hub-served. Set it for gated repos (403s on every hub call, including the
        ``config.json`` fetch that decides the loader) and for checkpoints that only
        exist as a local file.
    ``normalization``
        ``(mean, std)`` OVERRIDE, or ``None`` to let the repo's own config answer. An
        entry here wins over the hub, because these are the statistics our published
        numbers were produced with and a re-uploaded hub config must not move them.
    ``thunder_readout``
        The pooling THUNDER published for this backbone: ``"cls"``, ``"clsmean"``, or
        ``None`` for "not in the paper". ``None`` makes ``default_pooling`` RAISE rather
        than guess -- a guessed protocol produces a number that is not comparable to the
        leaderboard and looks exactly like one that is.
    ``lora_target_suffixes``
        Leaf module paths inside one block that LoRA adapts. Discovered at build time in
        ``encoder``; recorded here so a cell (which merges adapters outside this repo)
        can be checked against it.
    ``embed_dim`` / ``num_prefix_tokens`` / ``num_blocks`` / ``patch_size``
        Shape facts, all re-derivable from a built model. They exist so
        ``tests/test_new_backbones.py`` can assert them, i.e. so a wrong entry above
        surfaces as a failing number rather than as silent drift.
    """

    repo_id: str
    loader: str
    weights_file: str
    architecture: str | None = None
    timm_kwargs: dict = field(default_factory=dict)
    local_subdir: str | None = None
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    thunder_readout: str | None = None
    lora_target_suffixes: tuple[str, ...] = ()
    embed_dim: int | None = None
    num_prefix_tokens: int | None = None
    num_blocks: int | None = None
    patch_size: int | None = None
    notes: str = ""

    @property
    def local_dir(self) -> str | None:
        """Absolute directory serving this backbone's weights, or ``None`` for hub."""
        if self.local_subdir is None:
            return None
        return str(INPUTS / self.local_subdir)


BACKBONES: dict[str, Backbone] = {
    # ---------------------------------------------------------------- transformers ---
    "owkin/phikon-v2": Backbone(
        repo_id="owkin/phikon-v2",
        loader="hf",
        weights_file="model.safetensors",
        normalization=(IMAGENET_MEAN, IMAGENET_STD),
        # THUNDER pretrained_models.py:291-306 reads the CLS token alone for phikon2 --
        # NOT cls++mean. arXiv:2607.22861 3 line 106 agrees: clsmean was used in
        # PathoROB for all models, but in THUNDER only for the models listed there.
        thunder_readout="cls",
        lora_target_suffixes=_HF_DINOV2_MLP,
        embed_dim=1024, num_prefix_tokens=1, num_blocks=24, patch_size=16,
        notes="Dinov2 ViT-L/16, ungated. The project default: weakest PathoROB base "
              "(Avg RI 0.469) with the largest published gain, so the cheapest "
              "informative starting point.",
    ),
    "kaiko-ai/midnight": Backbone(
        repo_id="kaiko-ai/midnight",
        loader="hf",
        weights_file="model.safetensors",
        # Model card: "normalized with mean (0.5, 0.5, 0.5) and standard deviation
        # (0.5, 0.5, 0.5). Please ensure you apply these exact normalization parameters."
        # Feeding ImageNet stats instead does not crash and does not look wrong anywhere.
        normalization=(HALF_MEAN, HALF_STD),
        thunder_readout="clsmean",       # named on line 106 of the paper
        lora_target_suffixes=_HF_DINOV2_SWIGLU,
        embed_dim=1536, num_prefix_tokens=1, num_blocks=40, patch_size=14,
        notes="Dinov2 ViT-g/14 with SwiGLU FFN, so its FFN leaves are "
              "mlp.weights_in/weights_out -- a fixed query/key/value/dense LoRA target "
              "list would match attention only and freeze 2/3 of each block.",
    ),
    # ------------------------------------------------------------------------ timm ---
    "paige-ai/Virchow2": Backbone(
        repo_id="paige-ai/Virchow2",
        loader="timm",
        weights_file="model.safetensors",
        architecture="vit_huge_patch14_224",
        # Hub-served, so timm applies the repo's own model_args and pretrained_cfg; no
        # transcription belongs here. Normalisation likewise resolves from
        # pretrained_cfg (ImageNet), which is what the model card's transform does.
        thunder_readout="clsmean",       # named on line 106 of the paper
        lora_target_suffixes=_TIMM_VIT,
        embed_dim=1280, num_prefix_tokens=5, num_blocks=32, patch_size=14,
        notes="ViT-H/14, packed-SwiGLU FFN, 4 register tokens. NOT gated -- deliberately "
              "absent from the local-directory table.",
    ),
    "bioptimus/H-optimus-0": Backbone(
        repo_id="bioptimus/H-optimus-0",
        loader="timm",
        weights_file="pytorch_model.bin",
        architecture="vit_giant_patch14_reg4_dinov2",
        # img_size is NOT optional: the architecture's DINOv2 default is 518, whose
        # pos_embed is (1, 1369, 1536) against this checkpoint's (1, 256, 1536).
        timm_kwargs={"img_size": 224, "init_values": 1e-5, "dynamic_img_size": False},
        local_subdir="H-optimus-0",
        # The model card publishes its own H&E-corpus statistics. ImageNet's have the
        # same shapes, raise no warning, and quietly cost accuracy on every row.
        normalization=((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
        # In Reference Table 2 and NOT in the line-106 clsmean list. The trap is H0-mini,
        # which IS in that list: it is a distillation of this model and a separate row.
        thunder_readout="cls",
        lora_target_suffixes=_TIMM_VIT,
        embed_dim=1536, num_prefix_tokens=5, num_blocks=40, patch_size=14,
        notes="GATED (403 on every hub call, config.json included), so the local "
              "binding must be consulted before the hub for loader dispatch too.",
    ),
    "MahmoodLab/UNI2-h": Backbone(
        repo_id="MahmoodLab/UNI2-h",
        loader="timm",
        weights_file="pytorch_model.bin",
        architecture="vit_giant_patch14_224",
        # The card's timm_kwargs verbatim. The architecture NAME carries none of this:
        # timm's vit_giant_patch14_224 defaults are embed_dim=1408, depth=40,
        # num_heads=16 and a plain MLP -- a different model that builds without
        # complaint and then fails to load a single block.
        timm_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "no_embed_class": True,
            "mlp_layer": "SwiGLUPacked",
            "act_layer": "SiLU",
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        local_subdir="UNI2-h",
        # Really ImageNet -- timm's pretrained_cfg says so. Pinned anyway so that
        # "ImageNet because the card says so" and "ImageNet because a lookup fell
        # through" are not indistinguishable at the call site.
        normalization=(IMAGENET_MEAN, IMAGENET_STD),
        thunder_readout="cls",
        lora_target_suffixes=_TIMM_VIT,
        embed_dim=1536, num_prefix_tokens=9, num_blocks=24, patch_size=14,
        notes="GATED. 1 class token + 8 register tokens.",
    ),
    "SophontAI/OpenMidnight": Backbone(
        repo_id="SophontAI/OpenMidnight",
        loader="timm",
        weights_file="pytorch_model.bin",
        architecture="vit_giant_patch14_reg4_dinov2",
        timm_kwargs={"img_size": 224, "init_values": 1e-5, "dynamic_img_size": False},
        # NOT gated. Bound locally for the other reason the table exists: the weights we
        # run are a file on this machine, not a hub artefact. The source is a raw DINOv2
        # *training* checkpoint with block_chunks=4 nesting, which nothing in timm can
        # load; scripts/convert_openmidnight.py de-chunks it into this directory.
        local_subdir="OpenMidnight",
        # ImageNet by STATEMENT: the card's own snippet is
        # Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) commented
        # "# ImageNet normalization". Not inherited from kaiko-ai/midnight, the model it
        # replicates, which demands (0.5, 0.5, 0.5) -- "same family" would have been
        # exactly the wrong reason to pick statistics here.
        normalization=(IMAGENET_MEAN, IMAGENET_STD),
        # Postdates arXiv:2607.22861: no published THUNDER protocol, so unlisted.
        thunder_readout=None,
        lora_target_suffixes=_TIMM_VIT,
        embed_dim=1536, num_prefix_tokens=5, num_blocks=40, patch_size=14,
        notes="Architecturally the same animal as H-Optimus-0, which the CHECKPOINT "
              "proves rather than the name (pos_embed (1, 257, 1536) => 224px).",
    ),
    "paige-ai/Virchow": Backbone(
        repo_id="paige-ai/Virchow",
        loader="timm",
        weights_file="model.safetensors",
        architecture="vit_huge_patch14_224",
        # Transcribed verbatim from the repo's config.json model_args. MANDATORY even
        # though the weight-shape probe would find the packed FFN on its own, because
        # the local path builds the bare architecture name and NOTHING applies
        # model_args: timm's defaults are mlp_ratio=4 (fc1 out 5120, not this
        # checkpoint's 6832) and init_values=None (no ls1.gamma to load into).
        timm_kwargs={
            "img_size": 224,
            "init_values": 1e-5,
            "mlp_ratio": 5.3375,
            "dynamic_img_size": True,
            "mlp_layer": "SwiGLUPacked",
            "act_layer": "SiLU",
        },
        # The pinned SNAPSHOT COMMIT, not a cache root: the parent holds refs/ and
        # blobs/ that timm cannot load, and naming the commit means a later re-download
        # of a different revision cannot be picked up silently.
        local_subdir=(
            "Virchow/models--paige-ai--Virchow/snapshots/"
            "19eebc84ae33e79f1b2d866e6ff90ae50e522f9a"
        ),
        # ImageNet by its own pretrained_cfg; pinned so the stats stay fixed if the
        # local binding is ever repointed. v1 ONLY -- Virchow2 is a separate entry and
        # nothing may be shared between the two.
        normalization=(IMAGENET_MEAN, IMAGENET_STD),
        thunder_readout=None,            # not a row of Reference Table 2
        lora_target_suffixes=_TIMM_VIT,
        embed_dim=1280, num_prefix_tokens=1, num_blocks=32, patch_size=14,
        notes="Virchow **v1**: NO register tokens (1 prefix token / 257 tokens), against "
              "Virchow2's 5 / 261. GATED, hence the local binding.",
    ),
}


# ======================================================================================
# Views. Nothing below adds a fact; each one selects one column of the table above.
# ======================================================================================

def normalization_table() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """``repo_id -> (mean, std)`` for backbones that OVERRIDE their repo's own stats."""
    return {b.repo_id: b.normalization for b in BACKBONES.values() if b.normalization}


def local_dir_table() -> dict[str, str]:
    """``repo_id -> absolute directory`` for locally-served backbones."""
    return {b.repo_id: d for b in BACKBONES.values() if (d := b.local_dir) is not None}


def timm_kwargs_table() -> dict[str, dict]:
    """``repo_id -> timm.create_model kwargs`` for backbones that pin them."""
    return {b.repo_id: dict(b.timm_kwargs) for b in BACKBONES.values() if b.timm_kwargs}


def readout_table() -> dict[str, str]:
    """``repo_id -> published THUNDER pooling``, omitting backbones with no published one."""
    return {b.repo_id: b.thunder_readout for b in BACKBONES.values() if b.thunder_readout}


def get(repo_id: str) -> Backbone | None:
    """The registry entry for ``repo_id``, or ``None`` if it is not registered."""
    return BACKBONES.get(repo_id)
