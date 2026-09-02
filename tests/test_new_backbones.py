"""Invariants for the backbones served from local checkpoints.

``bioptimus/H-optimus-0`` and ``MahmoodLab/UNI2-h`` cannot be reached on the hub from this
account (403). ``SophontAI/OpenMidnight`` is ungated but is served locally anyway: the
weights we run are a DINOv2 *training* checkpoint on this machine, remapped once by
``scripts/convert_openmidnight.py``. All three are bound to local directories by
``waivphaet.models.encoder.BACKBONE_LOCAL_DIRS``, and that changes the failure surface in
ways nothing else in the repo has: the config that decides timm-vs-AutoModel, the
normalisation lookup, the FFN-shape probe and the weight load all come from disk rather
than the hub.

Every check below guards something that fails SILENTLY -- a wrong normalisation, a
randomly-initialised block, or four/eight register tokens averaged into the embedding all
produce a right-shaped, right-dtyped, warning-free, WORSE number.

These build real 2.6-4.3 GB models on CPU, so they are marked ``slow`` and skip when the
checkpoints are not on this machine.
"""

from __future__ import annotations

import types

import pytest
import torch

from waivphaet.models.encoder import (
    BACKBONE_NORMALIZATION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderConfig,
    WaivEncoder,
    is_timm_backbone,
    local_backbone_dir,
    normalization_for,
)

H_OPTIMUS = "bioptimus/H-optimus-0"
UNI2 = "MahmoodLab/UNI2-h"
OPENMIDNIGHT = "SophontAI/OpenMidnight"
#: Virchow **v1**. A different model from ``paige-ai/Virchow2``, not a revision of it:
#: same ViT-H/14 skeleton and packed-SwiGLU FFN, but NO register tokens.
VIRCHOW1 = "paige-ai/Virchow"

#: Every locally-served backbone. Parametrising over this list rather than over a literal
#: pair is what makes a newly-registered backbone inherit the whole guard set instead of
#: quietly getting none of it.
LOCAL_BACKBONES = [H_OPTIMUS, UNI2, OPENMIDNIGHT, VIRCHOW1]

#: (backbone, num_prefix_tokens, hidden, depth) measured from the checkpoints themselves.
#: num_prefix_tokens is [CLS] + register tokens: H-Optimus-0 has ``reg_token (1, 4, 1536)``
#: -> 5, UNI2-h has ``reg_token (1, 8, 1536)`` -> 9.
EXPECTED = {
    H_OPTIMUS: {"prefix": 5, "hidden": 1536, "blocks": 40, "patch": 14},
    UNI2: {"prefix": 9, "hidden": 1536, "blocks": 24, "patch": 14},
    # OpenMidnight is DINOv2 ViT-g/14 with 4 registers -> 5 prefix tokens, like
    # H-Optimus-0. Read off the checkpoint: register_tokens (1, 4, 1536),
    # 40 blocks (4 chunks x 10), 1536-d, patch_embed.proj.weight (1536, 3, 14, 14).
    OPENMIDNIGHT: {"prefix": 5, "hidden": 1536, "blocks": 40, "patch": 14},
    # Virchow v1, read off model.safetensors: cls_token (1, 1, 1280) and NO reg_token key
    # at all -> 1 prefix token; pos_embed (1, 257, 1280) -> 16x16 grid at patch 14, i.e.
    # 224px; 32 blocks; mlp.fc1 (6832, 1280) / mlp.fc2 (1280, 3416) -> packed gated FFN.
    VIRCHOW1: {"prefix": 1, "hidden": 1280, "blocks": 32, "patch": 14},
}


def _require_checkpoint(backbone: str):
    try:
        d = local_backbone_dir(backbone)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if d is None:
        pytest.fail(f"{backbone} lost its BACKBONE_LOCAL_DIRS binding; the hub repo is gated")
    return d


# --------------------------------------------------------------------------------------
# Normalisation -- the one that is wrong-but-invisible
# --------------------------------------------------------------------------------------

def test_h_optimus_normalisation_is_pinned_and_is_not_imagenet():
    """H-Optimus-0's model card gives H&E-corpus stats, NOT ImageNet.

    This is the sharpest silent failure in the whole integration: ``normalization_for``
    used to fall through to a hub lookup, the hub 403s for this repo, and an ImageNet
    fallback would have produced a complete, plausible, wrong set of numbers with no log
    line anywhere. The entry must be pinned in the table so no lookup is even attempted.
    """
    mean, std = normalization_for(H_OPTIMUS)
    assert mean == (0.707223, 0.578729, 0.703617), mean
    assert std == (0.211883, 0.230117, 0.177517), std
    assert mean != IMAGENET_MEAN and std != IMAGENET_STD
    # Pinned in the OVERRIDE table, which wins before any config or hub read.
    assert H_OPTIMUS in BACKBONE_NORMALIZATION


def test_uni2_normalisation_is_imagenet_by_statement_not_by_fallthrough():
    """UNI2-h really is ImageNet -- but "ImageNet because we decided" and "ImageNet
    because the lookup failed" are indistinguishable at the call site, so it is pinned."""
    assert UNI2 in BACKBONE_NORMALIZATION
    assert normalization_for(UNI2) == (IMAGENET_MEAN, IMAGENET_STD)


def test_virchow1_normalisation_is_imagenet_and_is_pinned():
    """Virchow v1's own ``pretrained_cfg`` carries ImageNet mean/std and its model card
    resolves the transform through that same cfg. The value is pinned rather than derived
    so it survives a repointed local binding, and so the call site cannot confuse "we read
    the card" with "a lookup fell through"."""
    assert VIRCHOW1 in BACKBONE_NORMALIZATION
    assert normalization_for(VIRCHOW1) == (IMAGENET_MEAN, IMAGENET_STD)


def test_virchow1_is_registered_separately_from_virchow2():
    """The two are different models sharing a name prefix. Virchow2 is hub-served and must
    stay out of both local tables; any entry keyed on a prefix rather than the exact repo
    id would capture it."""
    from waivphaet.models.encoder import BACKBONE_LOCAL_DIRS, BACKBONE_TIMM_KWARGS

    assert "paige-ai/Virchow2" not in BACKBONE_LOCAL_DIRS
    assert "paige-ai/Virchow2" not in BACKBONE_TIMM_KWARGS
    assert "paige-ai/Virchow2" not in BACKBONE_NORMALIZATION
    assert VIRCHOW1 in BACKBONE_LOCAL_DIRS
    assert VIRCHOW1 in BACKBONE_TIMM_KWARGS
    # An explicit kwargs entry is mandatory: the gated/local path builds the bare
    # architecture name, so nothing applies the repo's model_args. timm's
    # vit_huge_patch14_224 defaults to mlp_ratio=4 and init_values=None, neither of which
    # can load this checkpoint.
    kw = BACKBONE_TIMM_KWARGS[VIRCHOW1]
    assert kw["mlp_ratio"] == 5.3375
    assert kw["init_values"] == 1e-5
    assert kw["mlp_layer"] == "SwiGLUPacked" and kw["act_layer"] == "SiLU"


def test_openmidnight_normalisation_is_imagenet_by_statement_not_by_family():
    """OpenMidnight replicates ``kaiko-ai/midnight``, which demands (0.5, 0.5, 0.5) --
    and OpenMidnight does NOT. Its model card's own embedding-extraction snippet uses
    ImageNet mean/std. Inheriting the ancestor's stats would not crash, would not warn,
    and would just cost accuracy on every row, so the value is pinned and this test
    states the contrast explicitly."""
    assert OPENMIDNIGHT in BACKBONE_NORMALIZATION
    assert normalization_for(OPENMIDNIGHT) == (IMAGENET_MEAN, IMAGENET_STD)
    assert normalization_for(OPENMIDNIGHT) != BACKBONE_NORMALIZATION["kaiko-ai/midnight"]


def test_normalization_refuses_an_unknown_backbone():
    """No silent ImageNet default. An unknown backbone must stop the run."""
    with pytest.raises(RuntimeError):
        normalization_for("some-lab/NeverSeenBefore")


# --------------------------------------------------------------------------------------
# Loader dispatch -- both are timm checkpoints, and both would misroute without the
# local-dir escape hatch (403 -> config None -> "not timm" -> AutoModel -> "Unrecognized
# model", an error that names nothing about the real cause).
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_gated_backbones_dispatch_to_timm_via_the_local_config(backbone):
    _require_checkpoint(backbone)
    assert is_timm_backbone(backbone) is True


@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_local_dir_binding_is_loud_when_the_directory_is_gone(backbone, monkeypatch):
    """/data has been swept before. A missing checkpoint must raise, not fall back to a
    hub call that 403s and misroutes the loader."""
    monkeypatch.setenv("WAIV_BACKBONE_LOCAL_DIRS", f"{backbone}=/nonexistent/checkpoint")
    with pytest.raises(RuntimeError, match="no config.json"):
        local_backbone_dir(backbone)


# --------------------------------------------------------------------------------------
# Pooling / register tokens -- mirrors test_pool_drops_virchow2_register_tokens
# --------------------------------------------------------------------------------------

class _PoolOnly:
    """Just enough of WaivEncoder to call the real, unbound ``_pool``."""

    def __init__(self, pooling, num_prefix_tokens):
        self.cfg = types.SimpleNamespace(pooling=pooling)
        self.num_prefix_tokens = num_prefix_tokens

    def pool(self, tokens):
        return WaivEncoder._pool(self, tokens)


@pytest.mark.parametrize(
    "backbone,n_prefix,n_tokens",
    [(H_OPTIMUS, 5, 261), (UNI2, 9, 265), (OPENMIDNIGHT, 5, 261), (VIRCHOW1, 1, 257)],
)
def test_pool_drops_every_register_token(backbone, n_prefix, n_tokens):
    """[CLS] + registers + 256 patches. Averaging the registers in is right-shape,
    right-dtype, no-warning, just a worse number -- so it gets a numerical test.

    UNI2-h is the reason this is parametrised rather than copied: it carries EIGHT
    registers, so a slice hardcoded to Virchow2's 5 would silently keep four of them.
    """
    torch.manual_seed(0)
    tokens = torch.randn(3, n_tokens, 1536, dtype=torch.float64)

    official = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], dim=-1)
    got = _PoolOnly("clsmean", n_prefix).pool(tokens)
    assert got.shape == (3, 3072), got.shape
    assert torch.equal(got, official)

    # The wrong-but-plausible versions are genuinely different, i.e. the test has teeth.
    naive = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
    if n_prefix == 1:
        # Virchow v1 has no registers, so "drop the prefix" and "drop token 0" coincide.
        # The teeth here point the other way: it must NOT inherit Virchow2's 5-token slice.
        assert torch.equal(got, naive)
    else:
        assert not torch.allclose(got, naive)
    virchow_slice = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)
    if n_prefix != 5:
        assert not torch.allclose(got, virchow_slice)


# --------------------------------------------------------------------------------------
# The real build. Slow (multi-GB CPU load), so one encoder per backbone, reused.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoders():
    built = {}
    for backbone in LOCAL_BACKBONES:
        _require_checkpoint(backbone)
        built[backbone] = WaivEncoder(
            EncoderConfig(backbone=backbone, pooling="clsmean", use_lora=True)
        )
    return built


@pytest.mark.slow
@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_geometry_is_read_off_the_loaded_checkpoint(encoders, backbone):
    """Every geometric field comes from the BUILT model, so a kwargs/weights mismatch
    shows up as a number here rather than as a quietly different architecture.

    A strict state_dict load is what makes this meaningful: ``WaivEncoder`` refuses a
    local checkpoint with any missing or unexpected key, so reaching this assertion at
    all proves 0 missing / 0 unexpected.
    """
    enc = encoders[backbone]
    exp = EXPECTED[backbone]
    assert enc.is_timm is True
    assert enc.num_prefix_tokens == exp["prefix"]
    assert enc.hidden_size == exp["hidden"]
    assert enc.num_blocks == exp["blocks"]
    assert enc.patch_size == exp["patch"]
    assert enc.embed_dim == 2 * exp["hidden"]        # clsmean
    assert (enc.norm_mean, enc.norm_std) == normalization_for(backbone)


@pytest.mark.slow
@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_forward_widths_match_the_pooling_protocol(encoders, backbone):
    """cls -> hidden, clsmean -> 2*hidden, on a real forward. The token count must also
    be prefix + 256, i.e. the model was built at img_size 224 and not at its arch default
    (H-Optimus-0's ``vit_giant_patch14_reg4_dinov2`` defaults to 518 -> 1369 patches)."""
    enc = encoders[backbone]
    hidden = EXPECTED[backbone]["hidden"]
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        tokens = enc.tokens(x)
        assert tokens.shape == (2, EXPECTED[backbone]["prefix"] + 256, hidden), tokens.shape

        enc.cfg.pooling = "clsmean"
        assert enc.embed(x).shape == (2, 2 * hidden)
        enc.cfg.pooling = "cls"
        assert enc.embed(x).shape == (2, hidden)
    enc.cfg.pooling = "clsmean"


@pytest.mark.slow
@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_lora_reaches_every_block_with_the_timm_leaf_names(encoders, backbone):
    """LoRA-on-the-head is the failure mode PLAN.md 2 rules out, and an empty target set
    reads downstream as "the method had no effect on this backbone".

    Both models are timm ViTs with fused qkv, so the resolved leaves must be exactly
    attn.qkv / attn.proj / mlp.fc1 / mlp.fc2 -- 4 per block, on every block.
    """
    enc = encoders[backbone]
    n_blocks = EXPECTED[backbone]["blocks"]
    assert enc.lora_target_leaves == ("fc1", "fc2", "proj", "qkv")
    assert len(enc.lora_per_block) == n_blocks
    assert set(enc.lora_per_block.values()) == {4}
    assert len(enc.lora_target_names) == 4 * n_blocks
    for suffix in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
        hits = [n for n in enc.lora_target_names if n.endswith(suffix)]
        assert len(hits) == n_blocks, (suffix, len(hits))


@pytest.mark.slow
@pytest.mark.parametrize("backbone", LOCAL_BACKBONES)
def test_embed_uses_the_register_aware_slice_not_tokens_1(encoders, backbone):
    """End-to-end version of the register-token test: the exported embedding must equal
    the model card's ``cat([t[:, 0], t[:, n_prefix:].mean(1)])`` and must NOT equal the
    naive ``t[:, 1:]`` variant."""
    enc = encoders[backbone]
    n_prefix = EXPECTED[backbone]["prefix"]
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = enc.tokens(x)
        got = enc.embed(x)
    official = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], dim=-1)
    naive = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
    assert torch.equal(got, official)
    if n_prefix == 1:
        # No registers on Virchow v1: the two slices are the same by construction. The
        # live risk is the opposite one -- Virchow2's 5-token slice would eat 4 patches.
        assert torch.equal(got, naive)
        wrong = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)
        assert not torch.allclose(got, wrong)
    else:
        assert not torch.allclose(got, naive)
