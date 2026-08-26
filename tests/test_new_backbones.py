"""Invariants for the two GATED backbones served from local checkpoints.

``bioptimus/H-optimus-0`` and ``MahmoodLab/UNI2-h`` cannot be reached on the hub from this
account (403). They are bound to local directories by
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

#: (backbone, num_prefix_tokens, hidden, depth) measured from the checkpoints themselves.
#: num_prefix_tokens is [CLS] + register tokens: H-Optimus-0 has ``reg_token (1, 4, 1536)``
#: -> 5, UNI2-h has ``reg_token (1, 8, 1536)`` -> 9.
EXPECTED = {
    H_OPTIMUS: {"prefix": 5, "hidden": 1536, "blocks": 40, "patch": 14},
    UNI2: {"prefix": 9, "hidden": 1536, "blocks": 24, "patch": 14},
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


def test_normalization_refuses_an_unknown_backbone():
    """No silent ImageNet default. An unknown backbone must stop the run."""
    with pytest.raises(RuntimeError):
        normalization_for("some-lab/NeverSeenBefore")


# --------------------------------------------------------------------------------------
# Loader dispatch -- both are timm checkpoints, and both would misroute without the
# local-dir escape hatch (403 -> config None -> "not timm" -> AutoModel -> "Unrecognized
# model", an error that names nothing about the real cause).
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
def test_gated_backbones_dispatch_to_timm_via_the_local_config(backbone):
    _require_checkpoint(backbone)
    assert is_timm_backbone(backbone) is True


@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
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
    "backbone,n_prefix,n_tokens", [(H_OPTIMUS, 5, 261), (UNI2, 9, 265)]
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
    for backbone in (H_OPTIMUS, UNI2):
        _require_checkpoint(backbone)
        built[backbone] = WaivEncoder(
            EncoderConfig(backbone=backbone, pooling="clsmean", use_lora=True)
        )
    return built


@pytest.mark.slow
@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
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
@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
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
@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
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
@pytest.mark.parametrize("backbone", [H_OPTIMUS, UNI2])
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
    assert not torch.allclose(got, naive)
