"""Tests for the invariants that fail *silently* if broken.

Every check here guards something that would not show up in the training loss:
a positive that isn't co-registered, or a negative that leaks acquisition signal.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from waivphaet.data.conditions import (
    SCANNERS, STAINS, all_conditions, make_split, parse_filename,
)
from waivphaet.data.pairs import (
    PairBatch, PairBatchSampler, assert_same_condition_negatives, collate_pair_batch,
)
from waivphaet.train.contrastive import masked_info_nce


def test_condition_grid_is_13x7():
    assert len(STAINS) == 13 and len(SCANNERS) == 7
    assert len(all_conditions()) == 91
    assert len({c.filename for c in all_conditions()}) == 91


def test_filename_roundtrip():
    for c in all_conditions():
        assert parse_filename(c.filename) == c


def test_split_is_deterministic_and_disjoint():
    a, b = make_split(), make_split()
    assert [c.key for c in a.train] == [c.key for c in b.train]
    assert set(a.train).isdisjoint(a.heldout)
    assert len(a.train) + len(a.heldout) == 91


def test_split_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_split(["NOPE"], [])


def test_positive_condition_never_equals_anchor():
    """PLAN.md 2: a positive must be a *different* acquisition condition."""
    s = PairBatchSampler(all_conditions(), n_groups=32, group_size=16, batches_per_epoch=25)
    for b in s:
        assert not (b.positive_cond == b.anchor_cond[:, None]).any()


def test_group_is_condition_homogeneous_with_unique_tiles():
    """The structural guarantee the loss mask relies on."""
    s = PairBatchSampler(all_conditions(), n_groups=8, group_size=16, batches_per_epoch=10)
    for b in s:
        assert b.anchor_cond.shape == (8,)  # one condition per group, by construction
        for g in range(b.n_groups):
            assert len(np.unique(b.tile_idx[g])) == b.group_size


def test_positive_conditions_cover_the_grid():
    """The offset-shift trick must stay uniform over the other 90 conditions."""
    s = PairBatchSampler(all_conditions(), n_groups=64, group_size=16, batches_per_epoch=40)
    hist = np.zeros(91, dtype=int)
    for b in s:
        hist += np.bincount(b.positive_cond.ravel(), minlength=91)
    assert hist.min() > 0
    assert hist.max() < 2.0 * hist.mean()


def test_masked_infonce_negative_count_is_group_bounded():
    """Negatives must be the group (same condition), not the whole batch."""
    g = torch.arange(32) // 8
    z = torch.randn(32, 16)
    _, m = masked_info_nce(z, z.clone(), g)
    assert m["negatives_per_anchor"] == pytest.approx(7.0)  # group_size - 1, not 31


def test_masked_infonce_random_loss_is_log_group_size():
    g = torch.arange(64) // 8
    torch.manual_seed(0)
    _, m = masked_info_nce(torch.randn(64, 32), torch.randn(64, 32), group_id=g, temperature=1.0)
    assert m["loss"] == pytest.approx(np.log(8), abs=0.35)


def test_masked_infonce_ignores_cross_group_similarity():
    """A cross-group near-duplicate must not affect the loss -- it is not a valid negative."""
    g = torch.tensor([0, 0, 1, 1])
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    loss_a, _ = masked_info_nce(a, a.clone(), g, temperature=0.1)
    b = a.clone()
    b[2] = torch.tensor([1.0, 0.0])  # duplicate of row 0, but a different group
    loss_b, _ = masked_info_nce(b, b.clone(), g, temperature=0.1)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


# --- the negative constraint, asserted rather than eyeballed ---------------------------


def _fake_collated(n_groups=3, group_size=8, n_cond=91, seed=0):
    """A collated batch dict with the pixel tensors omitted (the checks never read them)."""
    s = PairBatchSampler(
        all_conditions()[:n_cond], n_groups=n_groups, group_size=group_size,
        batches_per_epoch=1, seed=seed,
    )
    b = next(iter(s))
    anchor_cond = np.broadcast_to(b.anchor_cond[:, None], b.tile_idx.shape)
    item = {
        "tile_idx": torch.from_numpy(b.tile_idx),
        "anchor_cond": torch.from_numpy(np.ascontiguousarray(anchor_cond)),
        "positive_cond": torch.from_numpy(b.positive_cond),
        "group_id": torch.arange(n_groups).repeat_interleave(group_size),
    }
    return b, collate_pair_batch(item)


def test_collated_batch_satisfies_the_negative_constraint():
    """Every group the loss will mask over must be condition-homogeneous."""
    _, batch = _fake_collated()
    stats = assert_same_condition_negatives(batch, allowed_conditions=set(range(91)))
    assert stats["negatives_per_anchor"] == 7.0
    assert stats["n_groups"] == 3.0


def test_assertion_catches_a_condition_mixed_group():
    """The check must FAIL on the bug it exists to catch, or it is decoration."""
    _, batch = _fake_collated()
    batch["anchor_cond"] = batch["anchor_cond"].clone()
    batch["anchor_cond"][0] = (batch["anchor_cond"][0] + 1) % 91  # one intruder
    with pytest.raises(AssertionError, match="mixes anchor conditions"):
        assert_same_condition_negatives(batch)


def test_assertion_catches_a_heldout_condition_leak():
    _, batch = _fake_collated()
    with pytest.raises(AssertionError, match="held-out condition has leaked"):
        assert_same_condition_negatives(batch, allowed_conditions={0})


def test_assertion_catches_a_repeated_tile_in_a_group():
    _, batch = _fake_collated()
    batch["tile_idx"] = batch["tile_idx"].clone()
    batch["tile_idx"][1] = batch["tile_idx"][0]  # anchor 1 is now anchor 0's own tile
    with pytest.raises(AssertionError, match="repeats a tile index"):
        assert_same_condition_negatives(batch)


def test_pairbatch_validate_rejects_a_same_condition_positive():
    b = PairBatch(
        tile_idx=np.array([[0, 1]]),
        anchor_cond=np.array([3]),
        positive_cond=np.array([[3, 5]]),  # first "positive" is the anchor's own condition
    )
    with pytest.raises(AssertionError, match="cross-acquisition"):
        b.validate(91)


def test_infonce_queries_positives_against_the_condition_homogeneous_anchors():
    """Orientation is load-bearing, not cosmetic.

    The candidate row must be the ANCHORS (one shared condition, so acquisition carries
    no signal), with the positives as queries. Running it the other way makes the
    candidate row span conditions and reintroduces the acquisition shortcut.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    g = torch.arange(16) // 8
    a, p = torch.randn(16, 12), torch.randn(16, 12)
    an, pn = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
    mask = g[:, None] == g[None, :]
    tgt = torch.arange(16)

    want = F.cross_entropy(((pn @ an.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)
    wrong = F.cross_entropy(((an @ pn.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)

    got, _ = masked_info_nce(a, p, g, temperature=0.07)
    assert torch.allclose(got, want, atol=1e-5)
    assert not torch.allclose(want, wrong, atol=1e-3)  # the two really are different


def test_symmetric_is_off_by_default():
    """PLAN.md 2: the anchor->positive direction has cross-condition candidates."""
    from waivphaet.train.contrastive import TrainConfig
    assert TrainConfig().symmetric is False


# --------------------------------------------------------------------------------------
# Backbone-agnostic encoder (PLAN.md §2: LoRA across the FULL depth, on any backbone).
#
# These are the guards for the failure that would otherwise be invisible: a LoRA target
# set that resolves to *fewer* modules on a new architecture. kaiko-ai/midnight sets
# use_swiglu_ffn=True, so its FFN linears are mlp.weights_in / mlp.weights_out rather
# than mlp.fc1 / mlp.fc2. The old fixed name list still matched query/key/value/dense in
# every block, so the block-coverage assertion PASSED while two thirds of each block's
# parameters stayed frozen -- and downstream that reads as "LoRA is weaker on ViT-g",
# not as a bug.


def test_block_index_parses_the_naming_schemes_we_target():
    from waivphaet.models.encoder import _block_index

    assert _block_index("encoder.layer.17.attention.attention.query") == 17
    assert _block_index("encoder.layers.3.mlp.fc1") == 3
    assert _block_index("blocks.11.attn.qkv") == 11
    # A Linear outside any numbered block must never be adapted: LoRA-on-the-head is
    # exactly what PLAN.md §2 rules out.
    assert _block_index("pooler.dense") is None
    assert _block_index("head.fc1") is None


def _fake_vit(n_blocks: int, ffn_names: tuple[str, str]):
    """Minimal module tree with HF-Dinov2 module *paths* and a swappable FFN naming."""
    import torch.nn as nn

    def block():
        b = nn.Module()
        b.attention = nn.Module()
        b.attention.attention = nn.Module()
        for n in ("query", "key", "value"):
            setattr(b.attention.attention, n, nn.Linear(8, 8))
        b.attention.output = nn.Module()
        b.attention.output.dense = nn.Linear(8, 8)
        b.mlp = nn.Module()
        for n in ffn_names:
            setattr(b.mlp, n, nn.Linear(8, 8))
        return b

    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layer = nn.ModuleList([block() for _ in range(n_blocks)])
    m.pooler = nn.Module()
    m.pooler.dense = nn.Linear(8, 8)  # decoy: right leaf name, outside every block
    return m


@pytest.mark.parametrize(
    "ffn", [("fc1", "fc2"), ("weights_in", "weights_out")],
    ids=["dinov2-mlp", "swiglu-ffn"],
)
def test_lora_discovery_covers_every_block_under_both_ffn_namings(ffn):
    from waivphaet.models.encoder import EncoderConfig, _lora_target_names

    model = _fake_vit(24, ffn)
    names, per_block, leaves = _lora_target_names(model, EncoderConfig())

    assert len(names) == 144, "6 linears x 24 blocks"
    assert set(per_block) == set(range(24))
    assert set(per_block.values()) == {6}, per_block
    assert set(leaves) == {"query", "key", "value", "dense", *ffn}
    assert not any(n.startswith("pooler.") for n in names), "adapted a head-level Linear"


def test_lora_discovery_is_unchanged_on_the_phikon_v2_naming():
    """The regression half: discovery must reproduce the old FIXED list exactly, or the
    PathoROB 0.468611 gate is being re-run against a different model."""
    from waivphaet.models.encoder import (
        LORA_TARGET_MODULES,
        EncoderConfig,
        _lora_target_names,
    )

    model = _fake_vit(24, ("fc1", "fc2"))
    discovered, _, _ = _lora_target_names(model, EncoderConfig())
    fixed, _, _ = _lora_target_names(
        model, EncoderConfig(lora_target_modules=LORA_TARGET_MODULES)
    )
    assert discovered == fixed


# --------------------------------------------------------------------------------------
# timm-native backbones (paige-ai/Virchow2) -- and the guards that the two PUBLISHED
# backbones did not move an inch while they were added.
#
# Three things can silently change a published number here:
#   1. the loader branch mis-routing owkin/phikon-v2 or kaiko-ai/midnight to timm;
#   2. `_pool` slicing patch tokens from a different offset once register tokens exist;
#   3. LORA_CANDIDATE_MODULES growing a name generic enough to catch a Linear on an
#      existing backbone that was previously not adapted.
# Each has a test below. All of them are offline: they exercise pure functions and fake
# module trees, so the suite does not need the hub or a 2.5 GB download to protect the
# invariant.


def _timm_style_config():
    """paige-ai/Virchow2's real config.json, trimmed. No model_type; timm's own shape."""
    return {
        "architecture": "vit_huge_patch14_224",
        "model_args": {
            "img_size": 224, "init_values": 1e-5, "num_classes": 0,
            "reg_tokens": 4, "mlp_ratio": 5.3375, "global_pool": "",
            "dynamic_img_size": True,
        },
        "pretrained_cfg": {
            "tag": "virchow_v2", "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
        },
    }


def test_loader_is_chosen_from_the_config_shape_not_a_model_name_list():
    """A name list would mean every future timm checkpoint takes the HF path and dies with
    an unrelated 'Unrecognized model' error. It would also mean a re-tagged repo keeps
    taking the wrong path forever. The config file is the source of truth."""
    from waivphaet.models.encoder import _is_timm_config

    assert _is_timm_config(_timm_style_config()) is True
    # The two PUBLISHED backbones: transformers configs, must stay on AutoModel.
    assert _is_timm_config({"model_type": "dinov2", "hidden_size": 1024}) is False   # phikon-v2
    assert _is_timm_config({"model_type": "dinov2", "hidden_size": 1536}) is False   # midnight
    assert _is_timm_config({"model_type": "vit", "hidden_size": 768}) is False       # phikon
    # A transformers config that ALSO happens to carry an architectures list is still HF:
    # model_type is decisive, and it is checked first.
    assert _is_timm_config(
        {"model_type": "dinov2", "architectures": ["Dinov2Model"], "architecture": "x"}
    ) is False
    # Unreadable config (offline, gated, 404) => not timm => AutoModel, i.e. the old
    # behaviour. A hub blip must never silently re-route a published backbone.
    assert _is_timm_config(None) is False
    assert _is_timm_config({}) is False


def test_packed_gated_ffn_is_detected_from_checkpoint_shapes():
    """Virchow2 needs mlp_layer=SwiGLUPacked; that is not expressible in config.json, and
    keying it off the repo id is the per-model dispatch this module refuses. The
    checkpoint says it: fc1 packs gate+value, so fc1.out == 2 * fc2.in."""
    from waivphaet.models.encoder import _needs_packed_gated_mlp

    assert _needs_packed_gated_mlp(6832, 3416) is True    # Virchow2, measured
    assert _needs_packed_gated_mlp(4096, 4096) is False   # plain MLP
    assert _needs_packed_gated_mlp(0, 0) is False         # probe found nothing


def test_normalization_override_still_wins_for_both_published_backbones(monkeypatch):
    """The precedence gained a timm branch. The override must still short-circuit FIRST --
    before any hub lookup at all -- or midnight's (0.5,0.5,0.5) could be replaced by
    whatever a re-uploaded config says, and every published number moves."""
    from waivphaet.models import encoder as enc

    def _boom(*a, **k):  # any hub access at all is a failure of precedence
        raise AssertionError("normalization_for consulted the hub for a pinned backbone")

    monkeypatch.setattr(enc, "_hub_config", _boom)
    monkeypatch.setattr(enc, "_hf_preprocessor_normalization", _boom)

    assert enc.normalization_for("owkin/phikon-v2") == (enc.IMAGENET_MEAN, enc.IMAGENET_STD)
    assert enc.normalization_for("kaiko-ai/midnight") == (enc.HALF_MEAN, enc.HALF_STD)
    assert enc.normalization_for(None) == (enc.IMAGENET_MEAN, enc.IMAGENET_STD)


def test_timm_normalization_is_read_from_pretrained_cfg(monkeypatch):
    """timm repos have no preprocessor_config.json, so AutoImageProcessor cannot answer
    for them -- timm publishes mean/std under pretrained_cfg instead."""
    from waivphaet.models import encoder as enc

    monkeypatch.setattr(enc, "_hub_config", lambda b: _timm_style_config())
    monkeypatch.setattr(
        enc, "_hf_preprocessor_normalization",
        lambda b: (_ for _ in ()).throw(AssertionError("HF processor consulted for a timm repo")),
    )
    mean, std = enc.normalization_for("paige-ai/Virchow2")
    assert mean == enc.IMAGENET_MEAN and std == enc.IMAGENET_STD


def _fake_timm_vit(n_blocks: int):
    """timm ViT module *paths*: fused qkv attention + a packed-SwiGLU FFN's leaf names."""
    import torch.nn as nn

    def block():
        b = nn.Module()
        b.attn = nn.Module()
        b.attn.qkv = nn.Linear(8, 24)
        b.attn.proj = nn.Linear(8, 8)
        b.mlp = nn.Module()
        b.mlp.fc1 = nn.Linear(8, 16)
        b.mlp.fc2 = nn.Linear(8, 8)
        return b

    m = nn.Module()
    m.blocks = nn.ModuleList([block() for _ in range(n_blocks)])
    m.patch_embed = nn.Module()
    m.patch_embed.proj = nn.Conv2d(3, 8, 14, 14)  # decoy: leaf "proj", and NOT a Linear
    m.head = nn.Linear(8, 8)                      # decoy: outside every block
    return m


def test_lora_discovery_on_a_timm_vit_needs_no_new_candidate_names():
    """Virchow2's in-block Linears are qkv / proj / fc1 / fc2 -- all four are already in
    LORA_CANDIDATE_MODULES, so nothing had to be added and nothing on the existing two
    backbones could change. 4/block x 32 blocks = 128 targets."""
    from waivphaet.models.encoder import EncoderConfig, _lora_target_names

    model = _fake_timm_vit(32)
    names, per_block, leaves = _lora_target_names(model, EncoderConfig())

    assert len(names) == 128, "4 linears x 32 blocks"
    assert set(per_block) == set(range(32))
    assert set(per_block.values()) == {4}, per_block
    assert set(leaves) == {"qkv", "proj", "fc1", "fc2"}
    assert not any(n.startswith(("patch_embed.", "head")) for n in names)


def test_candidate_module_list_is_frozen_for_the_published_backbones():
    """The third guard in WaivEncoder raises when an in-block Linear matched nothing, so
    adding a name is safe; *removing* or over-generalising one is not. This pins the exact
    counts the published runs used: phikon-v2 6/block, midnight 6/block."""
    from waivphaet.models.encoder import LORA_CANDIDATE_MODULES, EncoderConfig, _lora_target_names

    assert set(LORA_CANDIDATE_MODULES) == {
        "query", "key", "value", "dense", "fc1", "fc2",
        "weights_in", "weights_out",
        "qkv", "proj",
        "q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
    }
    phikon_v2 = _fake_vit(24, ("fc1", "fc2"))
    names, per_block, _ = _lora_target_names(phikon_v2, EncoderConfig())
    assert len(names) == 144 and set(per_block.values()) == {6}

    midnight = _fake_vit(40, ("weights_in", "weights_out"))
    names, per_block, _ = _lora_target_names(midnight, EncoderConfig())
    assert len(names) == 240 and set(per_block.values()) == {6}


# --------------------------------------------------------------------------------------
# split_heads + grid: the six required invariants
# --------------------------------------------------------------------------------------


class _FakeSplitModel(torch.nn.Module):
    """Minimal stand-in for WaivEncoder built with split_heads=('cls','mean').

    No backbone — embed_parts returns random(-ish) tokens per chunk; the projectors
    are real ProjectionHead instances so BatchNorm coupling is real.
    """

    def __init__(self, hidden: int = 32, proj_hidden: int = 512, proj_out: int = 512,
                 seed: int = 0, same_weights: bool = False):
        super().__init__()
        from waivphaet.models.encoder import ProjectionHead
        torch.manual_seed(seed)
        self.hidden_size = hidden
        self.split_heads: tuple[str, ...] = ("cls", "mean")
        self.pool_head = None
        # ProjectionHead needs out_dim >= 512
        self.projectors = torch.nn.ModuleDict({
            "cls": ProjectionHead(hidden, proj_hidden, proj_out),
            "mean": ProjectionHead(hidden, proj_hidden, proj_out),
        })
        if same_weights:
            # Copy cls weights into mean so both heads are identical — used by
            # the degenerate equivalence test.
            self.projectors["mean"].load_state_dict(self.projectors["cls"].state_dict())

    def embed_parts(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fake embed: images already are (B, 2*hidden) float tensors split into cls/mean."""
        B = images.shape[0]
        h = self.hidden_size
        return {"cls": images[:, :h], "mean": images[:, h:]}

    def pool_from_parts(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """clsmean = cat(cls, mean) -- matches the real encoder for pooling='clsmean'."""
        return torch.cat([parts["cls"], parts["mean"]], dim=1)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        parts = self.embed_parts(images)
        return self.pool_from_parts(parts)

    # not used by split path but satisfies retention_teacher_embed
    def forward(self, images: torch.Tensor):
        emb = self.embed(images)
        return emb, emb


def _make_grid_images(C: int, T: int, hidden: int, seed: int = 1) -> torch.Tensor:
    """(C*T, 2*hidden) float tensor: the fake 'images' _FakeSplitModel.embed_parts reads."""
    torch.manual_seed(seed)
    return torch.randn(C * T, 2 * hidden)


def test_split_grid_degenerate_equivalence():
    """w=0.5/0.5, identical head weights, CLS==mean => split loss matches single-head grid loss.

    Under these conditions:
      L_split = 0.5 * grid_info_nce(z_cls) + 0.5 * grid_info_nce(z_mean)
    where z_cls and z_mean are both outputs of the SAME projector applied to the SAME vector
    (since cls==mean when images have identical cls/mean halves), so both terms equal L_single.
    Hence L_split == L_single.
    """
    from waivphaet.train.contrastive import (
        _chunked_forward_split, grid_info_nce, grid_info_nce_split,
    )

    C, T, H = 3, 4, 32
    model = _FakeSplitModel(hidden=H, same_weights=True)
    model.eval()

    # Craft images where cls half == mean half so both pools are identical.
    torch.manual_seed(7)
    base = torch.randn(C * T, H)
    images = torch.cat([base, base], dim=1)  # (C*T, 2*H)

    with torch.no_grad():
        parts, gz = _chunked_forward_split(model, images, chunk=0)

        # Single-head: run the cls projector on base directly.
        z_single = model.projectors["cls"](base)
        _, m_single = grid_info_nce(z_single, C, T)

        _, m_split = grid_info_nce_split(
            gz, C, T,
            {"cls": 0.5, "mean": 0.5},
        )

    assert abs(m_split["loss"] - m_single["loss"]) < 1e-5, (
        f"split loss {m_split['loss']:.6f} != single loss {m_single['loss']:.6f}"
    )


def test_bn_no_chunk_invariant():
    """_chunked_forward_split with chunk=1 vs chunk=C*T gives identical projected outputs in eval."""
    from waivphaet.train.contrastive import _chunked_forward_split

    C, T, H = 3, 5, 32
    model = _FakeSplitModel(hidden=H)
    model.eval()
    images = _make_grid_images(C, T, H)

    with torch.no_grad():
        _, gz_full = _chunked_forward_split(model, images, chunk=0)
        _, gz_chunk = _chunked_forward_split(model, images, chunk=1)

    for k in gz_full:
        assert torch.allclose(gz_full[k], gz_chunk[k], atol=1e-5), (
            f"head '{k}' differs between chunk=0 and chunk=1 in eval mode"
        )


def test_bn_trailing_chunk_guard():
    """C*T=2401, chunk=600 must not raise (no trailing BatchNorm chunk of size 1)."""
    from waivphaet.train.contrastive import _chunked_forward_split

    C, T, H = 49, 49, 32  # 49*49=2401; 2401 % 600 = 1  -- the exact bad case
    model = _FakeSplitModel(hidden=H)
    model.eval()
    images = _make_grid_images(C, T, H, seed=3)

    with torch.no_grad():
        parts, gz = _chunked_forward_split(model, images, chunk=600)

    assert gz["cls"].shape == (C * T, 512)
    assert gz["mean"].shape == (C * T, 512)


def test_per_head_metric_keys_present():
    """After a grid_info_nce_split call, loss_cls/top1_cls/loss_mean/top1_mean must be present."""
    from waivphaet.train.contrastive import _chunked_forward_split, grid_info_nce_split

    C, T, H = 3, 4, 32
    model = _FakeSplitModel(hidden=H)
    model.eval()
    images = _make_grid_images(C, T, H)

    with torch.no_grad():
        _, gz = _chunked_forward_split(model, images)
        _, m = grid_info_nce_split(gz, C, T, {"cls": 0.5, "mean": 0.5})

    for k in ("loss_cls", "top1_cls", "loss_mean", "top1_mean"):
        assert k in m, f"missing metric key: {k}"


def test_gradient_flows_to_both_projectors():
    """Both projectors["cls"].weight.grad and ["mean"].weight.grad must be non-None and non-zero."""
    from waivphaet.train.contrastive import _chunked_forward_split, grid_info_nce_split

    C, T, H = 3, 4, 32
    model = _FakeSplitModel(hidden=H)
    model.train()
    images = _make_grid_images(C, T, H)

    parts, gz = _chunked_forward_split(model, images)
    loss, _ = grid_info_nce_split(gz, C, T, {"cls": 0.5, "mean": 0.5})
    loss.backward()

    for name in ("cls", "mean"):
        w = model.projectors[name].net[0].weight
        assert w.grad is not None, f"projectors['{name}'].weight.grad is None"
        assert w.grad.norm() > 0, f"projectors['{name}'].weight.grad is all-zero"


def test_anchor_emb_shape_with_split_grid_retention():
    """anchor_emb = pool_from_parts(parts) must have shape (C*T, 2*hidden) when split+grid."""
    from waivphaet.train.contrastive import _chunked_forward_split

    C, T, H = 3, 4, 32
    model = _FakeSplitModel(hidden=H)
    model.eval()
    images = _make_grid_images(C, T, H)

    with torch.no_grad():
        parts, _ = _chunked_forward_split(model, images)
        anchor_emb = model.pool_from_parts(parts)

    # clsmean = cat(cls, mean) => (C*T, 2*H) -- the eval-time embedding shape
    assert anchor_emb.shape == (C * T, 2 * H), (
        f"expected ({C * T}, {2 * H}), got {tuple(anchor_emb.shape)}"
    )


class _PoolOnly:
    """Just enough of WaivEncoder to call the real, unbound ``_pool``."""

    def __init__(self, pooling, num_prefix_tokens):
        self.cfg = types.SimpleNamespace(pooling=pooling)
        self.num_prefix_tokens = num_prefix_tokens

    def pool(self, tokens):
        from waivphaet.models.encoder import WaivEncoder

        return WaivEncoder._pool(self, tokens)


@pytest.mark.parametrize("pooling", ["cls", "mean", "clsmean"])
def test_pool_is_bit_identical_for_single_prefix_token_backbones(pooling):
    """`_pool` gained a ``num_prefix_tokens`` slice. On every HF backbone that value is 1,
    so it must reduce EXACTLY to the old ``tokens[:, 1:, :]`` -- not approximately."""
    torch.manual_seed(0)
    tokens = torch.randn(4, 197, 1024, dtype=torch.float64)

    cls, patches = tokens[:, 0, :], tokens[:, 1:, :]          # the pre-change expression
    expected = {
        "cls": cls,
        "mean": patches.mean(dim=1),
        "clsmean": torch.cat([cls, patches.mean(dim=1)], dim=1),
    }[pooling]

    got = _PoolOnly(pooling, 1).pool(tokens)
    assert got.shape == expected.shape
    assert torch.equal(got, expected), "bitwise inequality on a published backbone's pooling"


def test_pool_drops_virchow2_register_tokens():
    """Virchow2 emits [CLS] + 4 registers + 256 patches. The model card: "tokens 1-4 are
    register tokens so we ignore those". Averaging them in is right-shape, right-dtype,
    no-warning, just a worse number -- so it gets a numerical test, not a comment."""
    torch.manual_seed(0)
    tokens = torch.randn(3, 261, 1280, dtype=torch.float64)

    official = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)   # the model card
    got = _PoolOnly("clsmean", 5).pool(tokens)
    assert got.shape == (3, 2560), got.shape
    assert torch.equal(got, official)

    # And the wrong-but-plausible version is genuinely different, i.e. the test has teeth.
    naive = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
    assert not torch.allclose(got, naive)


def test_thunder_pooling_is_resolved_per_backbone_not_hardcoded():
    """arXiv:2607.22861 §3 line 106: in THUNDER, CLS+mean-pool concatenation is used only
    for Virchow2 / AquaViT / H0-mini / Midnight-12k. phikon-v2 is CLS there. Hardcoding
    either one makes the base-vs-fine-tuned rank sums non-comparable to their table."""
    src = Path(__file__).resolve().parents[1] / "src" / "waivphaet" / "eval" / "thunder_model.py"
    spec = importlib.util.spec_from_file_location("_waiv_thunder_model_test", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:  # thunder lives in .venv-thunder, not the default venv
        pytest.skip("thunder not importable in this interpreter")
    sys.modules.pop("_waiv_thunder_model_test", None)

    assert mod._default_pooling(None) == "cls"                 # default = phikon-v2
    assert mod._default_pooling("owkin/phikon-v2") == "cls"
    assert mod._default_pooling("kaiko-ai/midnight") == "clsmean"


def test_thunder_auto_pooling_never_resolves_to_clsmean_for_segmentation():
    """clsmean advertises emb_dim = 2*hidden, but get_segmentation_embeddings returns raw
    hidden-d patch tokens; THUNDER sizes its seg decoder from emb_dim, so on Midnight
    (3072 vs 1536) the job dies at task_specific_models.py:121. The correction must be
    narrow: explicit pooling and every classification run are untouched."""
    import importlib.util
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "waivphaet" / "eval" / "thunder_model.py"
    spec = importlib.util.spec_from_file_location("_waiv_thunder_seg_test", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:  # thunder lives in .venv-thunder, not the default venv
        pytest.skip("thunder not importable in this interpreter")
    sys.modules.pop("_waiv_thunder_seg_test", None)

    mid, phi = "kaiko-ai/midnight", "owkin/phikon-v2"

    # The bug: auto + segmentation + a backbone whose cls dim != patch dim.
    assert mod.resolve_pooling(mid, None, True) == "cls"
    # ... and nothing else moves.
    assert mod.resolve_pooling(mid, None, False) == "clsmean"   # the 24 held cls jobs
    assert mod.resolve_pooling(phi, None, True) == "cls"
    assert mod.resolve_pooling(phi, None, False) == "cls"
    # Explicit pooling always wins, segmentation or not -- the 4 running jobs pass cls.
    for seg in (True, False):
        for backbone in (mid, phi, None):
            for explicit in ("cls", "clsmean", "mean"):
                assert mod.resolve_pooling(backbone, explicit, seg) == explicit

    # Task detection is an EXACT argv token match, so no dataset / task / loading mode /
    # model path used by the classification sweep can trip it.
    assert mod._is_segmentation_run(["benchmark", "custom:/x/thunder_model.py",
                                     "ocelot", "segmentation",
                                     "--loading-mode", "online_loading"])
    for task in ("knn", "linear_probing", "simple_shot", "pre_computing_embeddings"):
        for ds in ("bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist",
                   "patch_camelyon", "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds"):
            assert not mod._is_segmentation_run(
                ["benchmark", "custom:/admin/home/ryan.kim/waiv/src/waivphaet/eval/"
                 "thunder_model.py", ds, task,
                 "--loading-mode", "embedding_pre_loading"]
            )


# --------------------------------------------------------------------------------------
# Virchow2 (paige-ai/Virchow2) registration -- timm ViT-H/14, 5 prefix tokens


def _load_thunder_model(name):
    """Import src/waivphaet/eval/thunder_model.py, stubbing ``thunder`` if it is absent.

    The two tests above skip when thunder is not importable, which is fine for them --
    they assert on protocol tables. These tests assert on the SEGMENTATION SLICE, which is
    the thing that silently degrades, so they must actually run in the default venv.
    thunder is only used for the ``PretrainedModel`` base class and none of these tests
    instantiate the subclass, so a stub base is faithful.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "waivphaet" / "eval" / "thunder_model.py"
    stubbed = []
    if importlib.util.find_spec("thunder") is None:
        pkg = types.ModuleType("thunder")
        pkg.__path__ = []
        models = types.ModuleType("thunder.models")
        models.PretrainedModel = type("PretrainedModel", (object,), {})
        sys.modules["thunder"], sys.modules["thunder.models"] = pkg, models
        stubbed = ["thunder", "thunder.models"]
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
        for m in stubbed:
            sys.modules.pop(m, None)
    return mod


class _HFBackbone:
    """transformers-style: keyword ``pixel_values``, ``.last_hidden_state``."""

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, pixel_values):
        assert pixel_values is not None
        return types.SimpleNamespace(last_hidden_state=self.tokens)


class _TimmBackbone:
    """timm-style: the batch is POSITIONAL and the raw token tensor comes straight back."""

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, images):
        assert images is not None
        return self.tokens


def _seg_shim(mod, backbone, is_timm, num_prefix_tokens):
    """Minimal stand-in for a constructed WaivPhikonEncoder.

    The real class cannot be instantiated without downloading a backbone, so the two real
    methods are bound onto a namespace instead -- the code under test is the shipped one,
    only ``self`` is fake.
    """
    enc = types.SimpleNamespace(
        backbone=backbone, is_timm=is_timm, num_prefix_tokens=num_prefix_tokens
    )
    shim = types.SimpleNamespace(encoder=enc)
    shim._backbone_tokens = types.MethodType(mod.WaivPhikonEncoder._backbone_tokens, shim)
    return shim


@pytest.mark.parametrize("backbone_hidden", [("owkin/phikon-v2", 1024), ("kaiko-ai/midnight", 1536)])
def test_thunder_segmentation_slice_is_bit_identical_on_the_published_backbones(backbone_hidden):
    """``get_segmentation_embeddings`` used to be ``backbone(pixel_values=x).
    last_hidden_state[:, 1:]``. It now dispatches on ``is_timm`` and slices by
    ``num_prefix_tokens``. Both published backbones are HF with num_prefix_tokens == 1,
    so the result must be BITWISE the old expression -- every published segmentation
    number came from it."""
    _, hidden = backbone_hidden
    mod = _load_thunder_model("_waiv_thunder_seg_identity")
    torch.manual_seed(0)
    tokens = torch.randn(2, 197, hidden, dtype=torch.float64)
    x = torch.randn(2, 3, 224, 224, dtype=torch.float64)

    bb = _HFBackbone(tokens)
    expected = bb(pixel_values=x).last_hidden_state[:, 1:]      # the pre-change expression
    got = mod.WaivPhikonEncoder.get_segmentation_embeddings(_seg_shim(mod, bb, False, 1), x)

    assert got.shape == expected.shape
    assert torch.equal(got, expected), "segmentation slice moved on a published backbone"


def test_thunder_segmentation_drops_virchow2_register_tokens_and_calls_timm_positionally():
    """Two separate bugs, both silent-or-fatal on Virchow2:

    1. ``backbone(pixel_values=x)`` is a TypeError on a timm VisionTransformer (the batch
       is positional). Proven directly against the fake, so the test fails if the shim
       ever stops modelling the real signature.
    2. ``[:, 1:]`` would hand THUNDER's segmentation decoder the 4 register tokens as if
       they were image patches -- right shape, right dtype, no warning, worse Dice.
    """
    mod = _load_thunder_model("_waiv_thunder_seg_virchow2")
    torch.manual_seed(0)
    tokens = torch.randn(2, 261, 1280, dtype=torch.float64)     # [CLS] + 4 reg + 256 patch
    x = torch.randn(2, 3, 224, 224, dtype=torch.float64)

    bb = _TimmBackbone(tokens)
    with pytest.raises(TypeError):
        bb(pixel_values=x)                                       # bug 1, demonstrated

    got = mod.WaivPhikonEncoder.get_segmentation_embeddings(_seg_shim(mod, bb, True, 5), x)
    assert got.shape == (2, 256, 1280), got.shape
    assert torch.equal(got, tokens[:, 5:])
    # And the wrong-but-plausible version is genuinely different, i.e. the test has teeth.
    assert got.shape != tokens[:, 1:].shape


def test_thunder_pooling_tables_cover_virchow2_without_moving_the_published_two():
    """Waiv 3.3: CLS+mean is used in THUNDER for Virchow2 / AquaViT / H0-mini /
    Midnight-12k. Virchow2 is therefore clsmean, and -- because clsmean advertises
    emb_dim = 2*hidden (2560) while the segmentation branch returns hidden-d (1280) patch
    tokens -- the existing clsmean->cls segmentation correction applies to it unchanged."""
    mod = _load_thunder_model("_waiv_thunder_virchow2_pooling")
    v, mid, phi = "paige-ai/Virchow2", "kaiko-ai/midnight", "owkin/phikon-v2"

    assert v in mod.THUNDER_CLSMEAN_BACKBONES
    assert mod._default_pooling(v) == "clsmean"
    assert mod.resolve_pooling(v, None, False) == "clsmean"     # 12 classification sets
    assert mod.resolve_pooling(v, None, True) == "cls"          # 2 segmentation sets
    for explicit in ("cls", "mean", "clsmean"):                 # explicit still wins
        for seg in (True, False):
            assert mod.resolve_pooling(v, explicit, seg) == explicit

    # The published two are untouched, including the default-backbone path.
    assert mod._default_pooling(None) == "cls"
    assert mod._default_pooling(phi) == "cls"
    assert mod._default_pooling(mid) == "clsmean"
    assert mod.THUNDER_CLS_BACKBONES == frozenset({phi})
    assert mod.resolve_pooling(mid, None, True) == "cls"
    assert mod.resolve_pooling(mid, None, False) == "clsmean"
    assert mod.resolve_pooling(phi, None, True) == "cls"
    assert mod.resolve_pooling(phi, None, False) == "cls"
    # An unlisted backbone is still a hard error, not a silent "cls".
    with pytest.raises(RuntimeError):
        mod._default_pooling("some-lab/NotInThePaper")


def test_pathorob_virchow2_target_records_only_the_average():
    """Waiv Table 1 gives Virchow2 Avg RI 0.858 -> 0.918 and this repo has no per-dataset
    breakdown behind it. Inventing three numbers that average to 0.858 is undetectable
    once written down, so the per-dataset keys must be ABSENT, and the published rows must
    not have moved."""
    from waivphaet.eval.pathorob_adapter import DATASETS, TARGETS, waiv_target

    assert TARGETS["virchow2_base"] == {"avg": 0.858}
    assert TARGETS["virchow2_target"] == {"avg": 0.918}
    for key in ("virchow2_base", "virchow2_target"):
        for ds in DATASETS:
            assert ds not in TARGETS[key], f"fabricated per-dataset value {key}/{ds}"
            assert waiv_target(key, ds) is None
        assert waiv_target(key, "avg") == TARGETS[key]["avg"]

    # The three rows every published number was gated against are byte-for-byte unchanged.
    assert TARGETS["phikon_v2_base"] == {
        "tcga": 0.619, "camelyon": 0.019, "tolkach_esca": 0.768, "avg": 0.469}
    assert TARGETS["phaet_target"] == {
        "tcga": 0.785, "camelyon": 0.702, "tolkach_esca": 0.932, "avg": 0.806}
    assert TARGETS["midnight_base"] == {
        "tcga": 0.858, "camelyon": 0.478, "tolkach_esca": 0.941, "avg": 0.759}
    assert TARGETS["mascaret_target"] == {
        "tcga": 0.893, "camelyon": 0.907, "tolkach_esca": 0.972, "avg": 0.924}


def test_thunder_run_name_and_job_prefix_conventions_are_consistent_across_the_three_files():
    """The Virchow2 sweep only works if three files agree: submit_thunder.sh names the
    jobs and runs, thunder_pilot.py must recognise the job prefixes (or it declares DONE
    against a held queue -- the first Midnight submission's actual failure), and
    collect_thunder.py must map the run names back to the right backbone (or Virchow2 F1s
    get diffed against phikon-v2's appendix)."""
    repo = Path(__file__).resolve().parents[1]
    submit = (repo / "scripts" / "submit_thunder.sh").read_text()

    pilot = importlib.util.spec_from_file_location(
        "_waiv_pilot", repo / "scripts" / "thunder_pilot.py")
    pmod = importlib.util.module_from_spec(pilot)
    pilot.loader.exec_module(pmod)
    sys.modules.pop("_waiv_pilot", None)

    collect = importlib.util.spec_from_file_location(
        "_waiv_collect", repo / "scripts" / "collect_thunder.py")
    cmod = importlib.util.module_from_spec(collect)
    collect.loader.exec_module(cmod)
    sys.modules.pop("_waiv_collect", None)

    # Job prefixes: declared in the submitter, known to the pilot.
    for prefix in ("thd-", "thdft1k-", "mthd-", "mthdft-", "vthd-", "vthdft-"):
        assert prefix in pmod.DEFAULT_PREFIXES, prefix
    for prefix in ("vthd-", "vthdft-"):
        assert f'"{prefix}"' in submit, f"{prefix} not declared in submit_thunder.sh"
    # The pilot de-prioritises base jobs by looking for "ft" in the prefix.
    assert "ft" in "vthdft-" and "ft" not in "vthd-"
    # Additive only: no existing job name can start with a new prefix and vice versa.
    for new in ("vthd-", "vthdft-"):
        for old in ("thd-", "thdft1k-", "mthd-", "mthdft-"):
            assert not new.startswith(old) and not old.startswith(new)

    # Run names: attributed by the collector. The base rows are literals in the
    # submitter; the FT rows are not, because the fine-tuned step is only known once the
    # blind "best PathoROB checkpoint" rule has run, so it is interpolated from
    # WAIV_VIRCHOW2_FT_STEP. Assert on what the submitter actually EMITS rather than on
    # its source text -- that is the property the sweep depends on, and it keeps the test
    # honest across both a default and an overridden step.
    for run in ("vbase_clsmean", "vbase_cls"):
        assert run in submit, f"{run} not declared in submit_thunder.sh"
    # The submitter consults live SLURM state and on-disk results, and SKIPS anything already
    # queued or complete -- so run it hermetically or this test reports on the cluster's mood
    # rather than on the script. Stub squeue to return nothing and point the results root at
    # an empty dir, so every job takes the emit path.
    with tempfile.TemporaryDirectory() as td:
        bindir = Path(td) / "bin"
        bindir.mkdir()
        (bindir / "squeue").write_text("#!/bin/sh\nexit 0\n")
        (bindir / "squeue").chmod(0o755)
        for step in ("500", "1250"):
            env = dict(os.environ,
                       PATH=f"{bindir}:{os.environ['PATH']}",
                       THUNDER_BASE_DATA_FOLDER=str(Path(td) / "empty_root"),
                       WAIV_VIRCHOW2_FT_STEP=step,
                       WAIV_VIRCHOW2_ADAPTER="runs/PLACEHOLDER-virchow2-adapter")
            emitted = subprocess.run(
                ["bash", str(repo / "scripts" / "submit_thunder.sh"), "--backbone", "virchow2"],
                capture_output=True, text=True, env=env, cwd=repo, check=True).stdout
            assert "SKIP" not in emitted, f"submitter not hermetic:\n{emitted[:400]}"
            for run in (f"vft{step}_clsmean", f"vft{step}_cls"):
                assert f" {run} " in emitted or emitted.rstrip().endswith(f" {run}"), \
                    f"{run} not emitted by submit_thunder.sh at step {step}"
    for run in ("vbase_clsmean", "vft500_clsmean", "vbase_cls", "vft500_cls",
                "vft1250_clsmean", "vft1250_cls"):
        assert cmod.infer_backbone(run) == cmod.VIRCHOW2, run
    # Published-two attribution is unchanged.
    for run in ("base_cls", "ft1000_cls"):
        assert cmod.infer_backbone(run) == cmod.PHIKONV2, run
    for run in ("mbase_clsmean", "mft500_clsmean", "mbase_cls", "mft500_cls"):
        assert cmod.infer_backbone(run) == cmod.MIDNIGHT, run
    assert cmod.infer_backbone("zzz_unknown") is None

    # No transcribed THUNDER leaderboard row for Virchow2 => it must take the same
    # "NO published counterpart" path Midnight takes, never phikon-v2's pub/delta columns.
    assert set(cmod.PUBLISHED) == {cmod.PHIKONV2}
    assert set(cmod.PUBLISHED_SOURCE) == {cmod.PHIKONV2}


# --------------------------------------------------------------------------------------
# Full FT mode (full-ft branch)


def test_full_ft_trainable_param_assertion():
    """A full-FT model (use_lora=False, freeze_backbone=False) must have ~100% trainable params.

    A full-FT run that silently trains only the projector would look like a weak result,
    not an error. The guard in train_lora.py asserts >= 95%.
    """
    import torch.nn as nn

    model = _fake_vit(8, ("fc1", "fc2"))
    # No freezing, no LoRA => all params should be trainable by default.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / max(total, 1)
    assert pct >= 0.95, f"full FT should have ~100% trainable, got {pct:.1%}"


def test_full_ft_checkpoint_roundtrip():
    """Save full FT backbone weights, reload via safetensors, and verify delta vs base.

    A silently-unloaded full-FT checkpoint reproduces base numbers exactly and reads as
    'perfect retention' -- the most dangerous false result. This tests the core guard
    mechanism: load a checkpoint into a model, then compare against the base model.
    """
    import torch.nn as nn
    from safetensors.torch import save_file, load_file

    # Create two models: base and "fine-tuned" (modify weights manually).
    base = _fake_vit(4, ("fc1", "fc2"))
    tuned = _fake_vit(4, ("fc1", "fc2"))

    # Modify tuned weights to simulate fine-tuning.
    for p in tuned.parameters():
        p.data += torch.randn_like(p) * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "backbone.safetensors"
        save_file(tuned.state_dict(), str(ckpt_path))

        # Load into a fresh model.
        loaded = _fake_vit(4, ("fc1", "fc2"))
        loaded_sd = load_file(str(ckpt_path))
        loaded.load_state_dict(loaded_sd)

    # Compare loaded weights vs base — they should differ.
    base_sd = base.state_dict()
    loaded_sd = loaded.state_dict()
    delta = sum((loaded_sd[k] - base_sd[k]).abs().sum().item()
                for k in loaded_sd)
    assert delta > 0, "modified checkpoint should differ from base"


def test_ckpt_schedule_parser():
    """Non-uniform checkpoint schedule parsing."""
    import argparse

    src = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("_train_lora_test", str(src))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("train_lora.py not importable in test environment")

    # Basic parsing.
    assert mod.parse_ckpt_schedule("25,50,75,100") == [25, 50, 75, 100]
    assert mod.parse_ckpt_schedule("1000") == [1000]

    # Unsorted input is sorted and deduped.
    assert mod.parse_ckpt_schedule("200,50,100,75") == [50, 75, 100, 200]
    assert mod.parse_ckpt_schedule("50,50,100,100") == [50, 100]

    # Whitespace tolerance.
    assert mod.parse_ckpt_schedule(" 50 , 100 , 150 ") == [50, 100, 150]

    # Invalid inputs.
    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("abc")

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("")

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("-1,50")


def test_should_checkpoint_uses_schedule_when_set():
    """_should_checkpoint respects ckpt_schedule over ckpt_every."""
    from waivphaet.train.contrastive import TrainConfig, _should_checkpoint

    cfg = TrainConfig(ckpt_every=500, ckpt_schedule=[50, 100, 200])
    assert _should_checkpoint(50, cfg) is True
    assert _should_checkpoint(100, cfg) is True
    assert _should_checkpoint(200, cfg) is True
    assert _should_checkpoint(500, cfg) is False  # not in schedule
    assert _should_checkpoint(75, cfg) is False


def test_should_checkpoint_uses_ckpt_every_when_no_schedule():
    """_should_checkpoint falls back to ckpt_every when schedule is None."""
    from waivphaet.train.contrastive import TrainConfig, _should_checkpoint

    cfg = TrainConfig(ckpt_every=200, ckpt_schedule=None)
    assert _should_checkpoint(200, cfg) is True
    assert _should_checkpoint(400, cfg) is True
    assert _should_checkpoint(100, cfg) is False
    assert _should_checkpoint(300, cfg) is False


# --------------------------------------------------------------------------------------
# Retention term: relational KL against the frozen base model (PLAN.md 2 frozen-teacher
# anchor). OFF by default, and "off" has to mean BIT-IDENTICAL -- every published number
# in this repo was produced by the pre-retention loss, so a default path that merely
# "looks the same" would silently invalidate all of them.


class _FakeAdapterBackbone(torch.nn.Module):
    """Stand-in for a PEFT-wrapped backbone: a base map plus a switchable adapter delta."""

    def __init__(self, d_in: int, d_out: int):
        import torch.nn as nn

        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.delta = nn.Linear(d_in, d_out, bias=False)  # the "LoRA" part: trainable
        self._adapter_on = True

    @contextlib.contextmanager
    def disable_adapter(self):
        prev, self._adapter_on = self._adapter_on, False
        try:
            yield
        finally:
            self._adapter_on = prev

    def save_pretrained(self, out_dir):  # what save_checkpoint calls in LoRA mode
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.delta.state_dict(), out_dir / "adapter.pt")

    def forward(self, x):
        out = self.base(x)
        return out + self.delta(x) if self._adapter_on else out


class _TinyEncoder(torch.nn.Module):
    """Minimal WaivEncoder-shaped model: ``embed`` -> pooled, ``forward`` -> (pooled, proj).

    ``use_lora`` selects an adapter-carrying backbone (retention is possible) or a plain
    Linear (retention must refuse -- the full-FT case).
    """

    def __init__(self, d_in=10, d_emb=8, d_proj=6, use_lora=True):
        import types

        import torch.nn as nn

        super().__init__()
        self.cfg = types.SimpleNamespace(use_lora=use_lora)
        self.backbone = _FakeAdapterBackbone(d_in, d_emb) if use_lora else nn.Linear(d_in, d_emb)
        self.projector = nn.Sequential(nn.Linear(d_emb, d_proj), nn.Dropout(0.5))

    def embed(self, images):
        return self.backbone(images)

    def forward(self, images):
        emb = self.embed(images)
        return emb, self.projector(emb)


def _retention_batches(n_batches=3, n_groups=2, group_size=6, d_in=10, seed=7):
    """Collated batches that satisfy the negative constraint, with pixel tensors attached."""
    out = []
    for i in range(n_batches):
        _, batch = _fake_collated(n_groups=n_groups, group_size=group_size, seed=seed + i)
        n = n_groups * group_size
        g = torch.Generator().manual_seed(1000 + i)
        batch["anchor"] = torch.randn(n, d_in, generator=g)
        batch["positive"] = torch.randn(n, d_in, generator=g)
        out.append(batch)
    return out


def _load_module_from_source(name: str, source: str):
    """Import a module from a source string (used to resurrect the HEAD implementation)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        # Must stay registered: dataclasses resolve `cls.__module__` lazily at
        # instantiation time, so popping it makes TrainConfig() explode.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _run_one_training(train_fn, config_cls, weights, batches, out_dir, **cfg_kwargs):
    """Run `train_fn` on a fresh _TinyEncoder loaded with `weights`; return (params, history)."""
    torch.manual_seed(0)
    model = _TinyEncoder(use_lora=cfg_kwargs.pop("use_lora", True))
    model.load_state_dict(weights)
    cfg = config_cls(
        out_dir=str(out_dir), max_steps=len(batches), warmup_steps=1, log_every=1,
        eval_every=10**9, ckpt_every=10**9, amp_dtype="none", n_groups=2, group_size=6,
        **cfg_kwargs,
    )
    torch.manual_seed(1234)  # fix the dropout stream so the comparison is meaningful
    summary = train_fn(model, batches, cfg, device="cpu")
    params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return params, summary["history"]


def test_retention_weight_zero_is_bit_identical_to_the_head_implementation():
    """(a) The default path must reproduce HEAD *exactly*, parameters and history alike.

    Not "close" -- torch.equal. The PathoROB 0.468611 gate and every HEST/THUNDER number
    on record were produced by the HEAD loss; if adding an optional term perturbs the
    default path by one ULP, those numbers no longer describe this code.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    head_src = subprocess.run(
        ["git", "show", "HEAD:src/waivphaet/train/contrastive.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    head = _load_module_from_source("_contrastive_at_head", head_src)
    if hasattr(head, "relational_kl"):
        # Once the retention commit lands, HEAD is no longer the "before" side and this
        # comparison is vacuous. Re-point it at the pre-retention commit to re-run it.
        pytest.skip("HEAD already contains the retention term; nothing to compare against")

    import waivphaet.train.contrastive as new

    batches = _retention_batches()
    torch.manual_seed(0)
    weights = {k: v.clone() for k, v in _TinyEncoder().state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        old_params, old_hist = _run_one_training(
            head.train, head.TrainConfig, weights, batches, Path(tmp) / "old",
        )
        new_params, new_hist = _run_one_training(
            new.train, new.TrainConfig, weights, batches, Path(tmp) / "new",
            retention_kl_weight=0.0,
        )

    assert old_params.keys() == new_params.keys()
    for k in old_params:
        assert torch.equal(old_params[k], new_params[k]), f"parameter {k} diverged"
    # Drop the wall-clock fields, which are timings and not results.
    timing = {"elapsed_s", "tiles_per_s"}
    strip = lambda h: [{k: v for k, v in r.items() if k not in timing} for r in h]  # noqa: E731
    assert json.dumps(strip(old_hist)) == json.dumps(strip(new_hist)), "history diverged"
    # ... and the history really did record something to compare.
    assert len(new_hist) == len(batches) and new_hist[0]["loss"] > 0
    # No retention keys leak into the default run's logs.
    assert not any(k.startswith("loss_retention") for k in new_hist[0])
    assert "loss_total" not in new_hist[0]


def test_relational_kl_is_nonnegative_and_exactly_zero_when_student_equals_teacher():
    """(b) Gibbs' inequality, asserted rather than assumed."""
    from waivphaet.train.contrastive import relational_kl

    torch.manual_seed(0)
    g = torch.arange(24) // 8
    t = torch.randn(24, 16)

    # Identical student and teacher -> exactly 0.0, not merely small.
    loss, m = relational_kl(t, t.clone(), group_id=g, temperature=0.07)
    assert float(loss) == 0.0
    assert m["loss_retention_kl"] == 0.0
    assert m["retention_kl_neighbours"] == pytest.approx(7.0)  # group_size - 1, no self

    # A global rotation + rescale is free: relational geometry is what is preserved.
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    rot, _ = relational_kl(3.0 * (t @ q), t, group_id=g, temperature=0.07)
    assert float(rot) == pytest.approx(0.0, abs=1e-5)

    # Anything else is strictly positive.
    for scale in (0.1, 1.0, 5.0):
        val, _ = relational_kl(t + scale * torch.randn_like(t), t, group_id=g, temperature=0.07)
        assert float(val) > 0.0
    # Including with no group mask at all (whole-batch candidates).
    val, m = relational_kl(torch.randn(24, 16), t, temperature=0.07)
    assert float(val) > 0.0 and m["retention_kl_neighbours"] == pytest.approx(23.0)
    assert torch.isfinite(val)


def test_relational_kl_masks_self_similarity_and_masks_both_sides_identically():
    """Self-similarity is 1.0 for teacher and student alike; at tau=0.07 leaving it in
    would make both rows ~one-hot and the term silently inert."""
    from waivphaet.train.contrastive import relational_kl

    torch.manual_seed(0)
    s, t = torch.randn(12, 8), torch.randn(12, 8)
    val, _ = relational_kl(s, t, temperature=0.07)
    # If the diagonal were included, a peaked temperature would drive this to ~0.
    assert float(val) > 1e-3
    # Scaling a single row of the student does not change ITS OWN masked row target set:
    # the mask is a function of indices only, never of the values.
    s2 = s.clone()
    s2[3] *= 7.0  # cosine-invariant
    val2, _ = relational_kl(s2, t, temperature=0.07)
    assert float(val) == pytest.approx(float(val2), abs=1e-5)


def test_retention_teacher_is_gradient_free_and_rng_neutral():
    """The teacher must contribute no gradient and must not move the random stream.

    An extra forward pass that consumed RNG would desynchronise everything seeded from
    the global generator relative to a weight=0 run, and the "off is identical" claim
    would only hold until the first dropout call.
    """
    from waivphaet.train.contrastive import retention_teacher_embed

    torch.manual_seed(0)
    model = _TinyEncoder()
    model.train()
    x = torch.randn(8, 10)

    before = torch.get_rng_state()
    emb = retention_teacher_embed(model, x)
    after = torch.get_rng_state()

    assert torch.equal(before, after), "the teacher forward moved the global RNG stream"
    assert model.training, "teacher forward left the model in eval mode"
    assert not emb.requires_grad and emb.grad_fn is None

    # It really is the BASE model: adapters off, and different from the student's output.
    with torch.no_grad():
        student = model.embed(x)
    assert not torch.allclose(student, emb)
    with torch.no_grad(), model.backbone.disable_adapter():
        assert torch.equal(model.embed(x), emb)

    # No gradient reaches the trainable adapter through the teacher path: the teacher
    # output is not even a leaf of a graph, so there is nothing to back-propagate.
    model.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError):
        retention_teacher_embed(model, x).sum().backward()
    assert model.backbone.delta.weight.grad is None


def test_retention_with_full_ft_raises():
    """(c) full-FT + retention is a degenerate combination and must be an error.

    With no adapter to disable, teacher == student, the KL is identically 0, and the run
    reads as a retention-regularised fine-tune that regularised nothing.
    """
    from waivphaet.train.contrastive import TrainConfig, assert_retention_teacher_available, train

    assert_retention_teacher_available(_TinyEncoder(use_lora=True))  # the LoRA case is fine

    with pytest.raises(ValueError, match="requires LoRA"):
        assert_retention_teacher_available(_TinyEncoder(use_lora=False))

    # And it is caught by train() before any compute happens.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(out_dir=tmp, max_steps=1, retention_kl_weight=0.1)
        with pytest.raises(ValueError, match="requires LoRA"):
            train(_TinyEncoder(use_lora=False), _retention_batches(1), cfg, device="cpu")

    # train_lora.py refuses the flag combination up front, before the backbone is built.
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    text = src.read_text()
    assert "--retention-kl-weight" in text and "--retention-kl-temperature" in text
    assert "args.retention_kl_weight > 0 and args.full_ft" in text


def test_retention_defaults_are_off():
    """The one guarantee everything else rests on."""
    from waivphaet.train.contrastive import TrainConfig

    assert TrainConfig().retention_kl_weight == 0.0
    assert TrainConfig().retention_kl_temperature == 0.07


def test_retention_on_logs_both_terms_separately():
    """The trade-off has to be readable in history.json, not just the sum."""
    from waivphaet.train.contrastive import TrainConfig, train

    batches = _retention_batches()
    with tempfile.TemporaryDirectory() as tmp:
        _, hist = _run_one_training(
            train, TrainConfig, _TinyEncoder().state_dict(), batches, Path(tmp),
            retention_kl_weight=1.0, retention_kl_temperature=0.07,
        )
    rec = hist[0]
    for k in ("loss", "loss_infonce", "loss_retention_kl", "loss_total"):
        assert k in rec, f"missing {k} in history record"
    assert rec["loss_infonce"] == rec["loss"]  # "loss" keeps its pre-retention meaning
    assert rec["loss_retention_kl"] >= 0.0
    assert rec["loss_total"] == pytest.approx(rec["loss_infonce"] + rec["loss_retention_kl"])


# --------------------------------------------------------------------------------------
# GRID sampler (waivphaet.data.grid): one shared tile list across C condition groups, so
# every image is both an anchor and a query. It inherits the pair sampler's load-bearing
# invariant (candidates are condition-homogeneous) and ADDS one: every condition group
# must use the SAME tiles in the SAME ORDER, because the loss identifies the positive by
# POSITION. Break that and every "positive" is a mislabelled pair -- while the loss curve
# still falls perfectly plausibly. Hence the must-FAIL tests below, not just a happy path.


class _PixelFreeGridDataset:
    """GridTileDataset with the pixel gather stubbed out.

    Uses the REAL ``GridTileDataset.__getitem__`` (so the id tensors, the ``group_id`` /
    ``tile_pos`` bookkeeping and the row-major flatten are the shipped ones, not a
    re-implementation that could drift), but returns 2x2 "images" instead of 224x224x3 --
    the invariants under test are about indices, and a real gather would be ~360 MB.
    """

    def __init__(self, n_conditions: int):
        from waivphaet.data.grid import GridTileDataset

        self._real = object.__new__(GridTileDataset)
        self._real.conditions = list(range(n_conditions))
        self._real.transform = None
        self._real._slides = {}
        self._real._gather = lambda cond, tiles: np.zeros(
            (*cond.shape, 2, 2, 3), dtype=np.uint8
        )

    def __getitem__(self, plan):
        return type(self._real).__getitem__(self._real, plan)


def _grid_collated(n_cond=4, n_tiles=5, n_available=12, seed=0):
    """One real sampler plan, materialised and collated. Returns (plan, collated batch)."""
    from waivphaet.data.grid import GridBatchSampler, collate_grid_batch

    sampler = GridBatchSampler(
        list(range(n_available)), n_cond=n_cond, n_tiles=n_tiles,
        batches_per_epoch=1, tile_indices=np.arange(200), seed=seed,
    )
    plan = next(iter(sampler))
    item = _PixelFreeGridDataset(n_available)[plan]
    return plan, collate_grid_batch(item)


def test_grid_happy_path_reports_the_geometry_it_actually_built():
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=4, n_tiles=5)
    stats = assert_grid_batch(batch, allowed_conditions=set(range(12)))
    assert stats["n_cond"] == 4.0
    assert stats["n_tiles"] == 5.0
    assert stats["negatives_per_anchor"] == 4.0          # T - 1
    assert stats["n_rows"] == 4 * 3 * 5                  # C * (C-1) * T
    assert stats["distinct_conditions"] == 4.0
    assert batch["image"].shape[0] == 4 * 5              # C*T images, no positive tensor


def test_grid_batch_MUST_FAIL_when_tile_sets_differ_between_condition_groups():
    """THE new load-bearing invariant.

    If group a and group b are drawn over different tiles, position t is not the same
    tissue in both, so every cross-group "positive" grid_info_nce scores is a mislabelled
    pair -- and the loss would still fall. Nothing else in the pipeline notices.
    """
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=4, n_tiles=5)
    tiles = batch["tile_idx"].clone()
    # give group 2 a tile that no other group has
    tiles[2 * 5 + 3] = 999
    batch["tile_idx"] = tiles

    with pytest.raises(AssertionError, match="shared tile list"):
        assert_grid_batch(batch, allowed_conditions=set(range(12)))


def test_grid_batch_MUST_FAIL_when_the_shared_tiles_are_merely_reordered():
    """Same SET, different ORDER, is just as broken -- the match is positional.

    A set-equality check would pass this batch. It must not: swapping two tiles inside one
    group silently re-labels two positives per pair involving that group.
    """
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=4, n_tiles=5)
    tiles = batch["tile_idx"].clone()
    g1 = slice(1 * 5, 2 * 5)
    block = tiles[g1].clone()
    block[0], block[1] = block[1].clone(), block[0].clone()
    tiles[g1] = block
    batch["tile_idx"] = tiles
    # the SET is identical -- prove it, so the test is really about order
    assert set(batch["tile_idx"][g1].tolist()) == set(batch["tile_idx"][0:5].tolist())

    with pytest.raises(AssertionError, match="shared order"):
        assert_grid_batch(batch, allowed_conditions=set(range(12)))


def test_grid_batch_MUST_FAIL_on_a_duplicated_condition():
    """Two groups on the same acquisition: their cross-group 'positive' is one image twice,
    so that row is solvable at similarity 1 without learning anything."""
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=4, n_tiles=5)
    cond = batch["cond_idx"].clone()
    cond[3 * 5:4 * 5] = cond[0]  # group 3 becomes a copy of group 0's condition
    batch["cond_idx"] = cond

    with pytest.raises(AssertionError, match="duplicate condition"):
        assert_grid_batch(batch, allowed_conditions=set(range(12)))


def test_grid_batch_MUST_FAIL_on_a_duplicated_tile_in_the_shared_list():
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=3, n_tiles=5)
    tiles = batch["tile_idx"].clone().view(3, 5)
    tiles[:, 4] = tiles[:, 0]  # same duplication in EVERY group -> invariant 2 still holds
    batch["tile_idx"] = tiles.reshape(-1)

    with pytest.raises(AssertionError, match="repeats a tile index"):
        assert_grid_batch(batch, allowed_conditions=set(range(12)))


def test_grid_batch_MUST_FAIL_on_a_heldout_condition_leak():
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=3, n_tiles=5, n_available=12)
    allowed = set(range(12)) - {int(batch["cond_idx"][0])}
    with pytest.raises(AssertionError, match="outside the loader's condition list"):
        assert_grid_batch(batch, allowed_conditions=allowed)


def test_grid_batch_MUST_FAIL_when_a_candidate_block_is_not_condition_homogeneous():
    """The original PLAN.md 2 constraint, carried over: a mixed candidate row lets
    'different acquisition' stand in for 'different tile'."""
    from waivphaet.data.grid import assert_grid_batch

    _, batch = _grid_collated(n_cond=3, n_tiles=5)
    cond = batch["cond_idx"].clone()
    cond[2] = cond[-1]  # one cell of group 0 defects to another condition
    batch["cond_idx"] = cond

    with pytest.raises(AssertionError, match="mixes conditions"):
        assert_grid_batch(batch, allowed_conditions=set(range(12)))


def test_grid_sampler_refuses_more_conditions_than_exist():
    """Conditions are drawn WITHOUT replacement; asking for more must be an error, not a
    silent fallback to sampling with replacement (which is the duplicate-condition bug)."""
    from waivphaet.data.grid import GridBatchSampler

    with pytest.raises(ValueError, match="exceeds the"):
        GridBatchSampler(list(range(10)), n_cond=11, n_tiles=4)


@pytest.mark.parametrize(
    "n_cond,n_tiles,expect_neg,expect_rows",
    [(24, 100, 99, 55_200), (49, 49, 48, 115_248)],
)
def test_grid_negative_and_row_counts_at_the_launch_geometries(
    n_cond, n_tiles, expect_neg, expect_rows
):
    """G4: the two arms' arithmetic, measured on real batches rather than asserted on paper."""
    from waivphaet.data.grid import assert_grid_batch
    from waivphaet.train.contrastive import grid_info_nce

    _, batch = _grid_collated(n_cond=n_cond, n_tiles=n_tiles, n_available=50)
    stats = assert_grid_batch(batch, allowed_conditions=set(range(50)))
    assert stats["negatives_per_anchor"] == float(expect_neg)
    assert stats["n_rows"] == float(expect_rows)
    assert batch["image"].shape[0] == n_cond * n_tiles

    # ...and the loss agrees, independently, from the tensor it actually reduces
    z = torch.randn(n_cond * n_tiles, 32)
    _, m = grid_info_nce(z, n_cond, n_tiles, temperature=1.0)
    assert m["negatives_per_anchor"] == float(expect_neg)
    assert m["n_rows"] == float(expect_rows)


@pytest.mark.parametrize("n_cond,n_tiles", [(24, 100), (49, 49)])
def test_grid_loss_on_random_embeddings_sits_at_the_random_guess_value(n_cond, n_tiles):
    """G5: random embeddings must score log(T), i.e. chance over T candidates per row.

    Two readings, because they fail differently:

    * At tau=1 with a high embedding dimension the logits are ~0 and the cross-entropy
      must land on log(T) to a couple of decimals. A loss meaningfully BELOW log(T) on
      random inputs would mean the rows are not really T-way -- e.g. an a==b pair leaking
      in, whose positive is the image itself at similarity 1.
    * top-1 must be ~1/T at ANY temperature. Temperature rescales the logits and so moves
      the cross-entropy, but it cannot move chance accuracy, which makes this the
      scale-free version of the same statement -- and it is checked at the tau=0.07 the
      arms actually train at.
    """
    import math

    from waivphaet.train.contrastive import grid_info_nce

    g = torch.Generator().manual_seed(0)
    z = torch.randn(n_cond * n_tiles, 2048, generator=g)

    _, m1 = grid_info_nce(z, n_cond, n_tiles, temperature=1.0)
    assert m1["loss"] == pytest.approx(math.log(n_tiles), abs=0.02), (
        f"C={n_cond} T={n_tiles}: random-embedding loss {m1['loss']:.4f} vs "
        f"log(T)={math.log(n_tiles):.4f}"
    )

    _, m2 = grid_info_nce(z, n_cond, n_tiles, temperature=0.07)
    assert m2["top1"] == pytest.approx(1.0 / n_tiles, abs=3.0 / n_tiles**0.5 / n_tiles**0.5)
    # cross-entropy is bounded below by the uniform value in expectation, never above chance
    assert m2["loss"] >= math.log(n_tiles) - 0.02


def test_grid_loss_excludes_the_self_pair_and_matches_a_reference_implementation():
    """Orientation and exclusion, checked against a literal double loop.

    For pair (a,b) the query is z[a,t] and the candidates are the whole of z[b,:] -- one
    condition, so acquisition carries no information down the row. Transposing it would
    put cross-condition candidates back in the row. A hand-written loop is the cheapest
    way to pin the orientation rather than trust the einsum's index letters.
    """
    import torch.nn.functional as F

    from waivphaet.train.contrastive import grid_info_nce

    c, t, d = 4, 6, 16
    g = torch.Generator().manual_seed(3)
    z = torch.randn(c * t, d, generator=g)
    loss, _ = grid_info_nce(z, c, t, temperature=0.07)

    zn = F.normalize(z.float(), dim=-1).view(c, t, d)
    terms = []
    for a in range(c):
        for b in range(c):
            if a == b:
                continue  # its "positive" is the image itself, at similarity 1
            logits = (zn[a] @ zn[b].t()) / 0.07  # query z[a,t], candidates z[b,:]
            terms.append(F.cross_entropy(logits, torch.arange(t)))
    assert float(loss) == pytest.approx(float(torch.stack(terms).mean()), rel=1e-6)


def test_grid_loss_rejects_a_geometry_that_does_not_match_the_tensor():
    from waivphaet.train.contrastive import grid_info_nce

    z = torch.randn(4 * 5, 8)
    with pytest.raises(ValueError, match="does not match the declared geometry"):
        grid_info_nce(z, 4, 6, temperature=0.07)
    with pytest.raises(ValueError, match="n_cond >= 2"):
        grid_info_nce(torch.randn(5, 8), 1, 5, temperature=0.07)


def test_grid_and_pair_batching_flags_are_mutually_exclusive_in_the_cli():
    """--grid with --n-groups must be a hard error: two runs batched differently under one
    label is exactly how a comparison gets silently confounded."""
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_lora.py"),
         "--out-dir", "/tmp/_nope", "--grid", "--n-groups", "6"],
        capture_output=True, text=True, cwd=repo,
    )
    assert r.returncode != 0
    assert "mutually exclusive" in (r.stderr + r.stdout)


def test_train_config_records_which_sampler_produced_the_run():
    """config.json is the only durable record of the batching; default must stay OFF."""
    from waivphaet.train.contrastive import TrainConfig

    assert TrainConfig().grid is False
    assert TrainConfig().grid_conditions == 0 and TrainConfig().grid_tiles == 0
    cfg = TrainConfig(grid=True, grid_conditions=49, grid_tiles=49)
    from dataclasses import asdict
    d = asdict(cfg)
    assert d["grid"] is True and d["grid_conditions"] == 49 and d["grid_tiles"] == 49


def test_grid_forward_chunking_is_a_memory_device_not_a_maths_change():
    """--grid-forward-chunk must not move the numbers, forward OR backward.

    It exists purely to bound gradient checkpointing's per-block recompute buffer (2400
    images in one forward OOMs an 80 GiB H100). If it perturbed the objective it would
    silently make the grid arms incomparable to CTRL, so both the concatenated output and
    the resulting gradients are checked against a single unchunked forward.
    """
    from waivphaet.train.contrastive import _chunked_forward

    torch.manual_seed(0)
    model = _TinyEncoder()
    # The REAL ProjectionHead contains nn.BatchNorm1d, which couples the batch together.
    # _TinyEncoder's plain Linear projector does not, so without this the test cannot see
    # the bug that chunking through the projector would introduce -- and that bug is not
    # hypothetical: it shipped, and only surfaced because 2401 images at chunk 600 leaves
    # a trailing chunk of 1 and BatchNorm refuses a batch of one in train mode.
    model.projector = torch.nn.Sequential(
        torch.nn.Linear(8, 6), torch.nn.BatchNorm1d(6), torch.nn.Dropout(0.5)
    )
    images = torch.randn(20, 10, generator=torch.Generator().manual_seed(5))

    model.eval()  # the projector carries Dropout(0.5); chunking must not re-draw it
    with torch.no_grad():
        whole_e, whole_z = _chunked_forward(model, images, 0)
        for chunk in (1, 3, 7, 20, 999):
            e, z = _chunked_forward(model, images, chunk)
            # NOT torch.equal: a different chunk size is a different GEMM shape, so the
            # library picks a different kernel and the reductions associate differently.
            # The maths is the same (a ViT is per-image, nothing normalises across the
            # batch); the last mantissa bits are not. Chunk size is therefore a FIXED
            # per-run setting recorded in config.json, not something to vary mid-run.
            assert torch.allclose(e, whole_e, atol=1e-6), f"pooled embedding moved at chunk={chunk}"
            assert torch.allclose(z, whole_z, atol=1e-6), f"projection moved at chunk={chunk}"
        # ...and at a FIXED chunk size it is exactly reproducible, which is what a run
        # replaying from a seed actually depends on.
        again_e, again_z = _chunked_forward(model, images, 6)
        once_e, once_z = _chunked_forward(model, images, 6)
        assert torch.equal(again_e, once_e) and torch.equal(again_z, once_z)

    def grads(chunk):
        model.zero_grad(set_to_none=True)
        _, z = _chunked_forward(model, images, chunk)
        z.square().sum().backward()
        return {k: p.grad.detach().clone() for k, p in model.named_parameters()
                if p.grad is not None}

    ref, chunked = grads(0), grads(6)
    assert ref.keys() == chunked.keys() and ref
    for k in ref:
        assert torch.allclose(ref[k], chunked[k], atol=1e-6), f"gradient moved for {k}"

    # --- TRAIN mode is where BatchNorm actually couples the batch -----------------------
    # In eval() BatchNorm uses running statistics and is per-image, so the checks above
    # cannot see a projector-chunking bug at all. Train mode can.
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0  # keep the comparison deterministic; BN is the thing under test
    with torch.no_grad():
        whole_e, whole_z = _chunked_forward(model, images, 0)
        for chunk in (3, 6, 20):
            e, z = _chunked_forward(model, images, chunk)
            assert torch.allclose(e, whole_e, atol=1e-6), f"train-mode embedding moved at {chunk}"
            assert torch.allclose(z, whole_z, atol=1e-6), (
                f"train-mode PROJECTION moved at chunk={chunk}: BatchNorm statistics are "
                "being computed per chunk instead of over the whole batch"
            )

        # A trailing chunk of exactly ONE image is the real launch geometry (2401 images
        # at chunk 600). BatchNorm raises on a batch of 1 in train mode, so if the
        # projector were inside the chunk loop this would throw rather than merely drift.
        odd = torch.randn(21, 10, generator=torch.Generator().manual_seed(9))
        e21, z21 = _chunked_forward(model, odd, 20)
        assert e21.shape[0] == 21 and z21.shape[0] == 21


def test_grid_heldout_eval_runs_at_its_own_narrower_geometry():
    """The held-out grid has FEWER conditions than the training grid (41 vs 50 here), so
    evaluate_heldout must read C and T off the batch, not off cfg.

    The gate smoke runs with eval disabled, so nothing else would catch this until step
    250 of a 1500-step run -- i.e. 25 minutes into a job, after the GPUs are committed.
    """
    from waivphaet.train.contrastive import TrainConfig, evaluate_heldout
    from waivphaet.data.grid import GridBatchSampler, GridTileDataset, collate_grid_batch

    d_in = 10
    ds = object.__new__(GridTileDataset)
    ds.conditions, ds.transform, ds._slides = list(range(6)), None, {}
    ds._gather = lambda cond, tiles: np.zeros((*cond.shape, d_in), dtype=np.float32)

    # training geometry is 5x4; the held-out loader below is a NARROWER 3x4
    sampler = GridBatchSampler(list(range(6)), n_cond=3, n_tiles=4, batches_per_epoch=2,
                               tile_indices=np.arange(40), seed=0)
    batches = []
    for plan in sampler:
        b = collate_grid_batch(ds[plan])
        b["image"] = b["image"].float()
        batches.append(b)

    model = _TinyEncoder()
    cfg = TrainConfig(grid=True, grid_conditions=5, grid_tiles=4, amp_dtype="none",
                      grid_forward_chunk=2)
    out = evaluate_heldout(model, batches, cfg, torch.device("cpu"), n_batches=2)
    assert set(out) == {"heldout_loss", "heldout_top1"}
    assert out["heldout_loss"] > 0 and 0.0 <= out["heldout_top1"] <= 1.0
    assert model.training, "evaluate_heldout must restore train mode"


# ======================================================================================
# SPLIT LOSS HEADS -- CLS and mean scored separately, one ProjectionHead each
# ======================================================================================
#
# The failure this whole block exists for: if a wiring bug fed BOTH heads the same pooled
# vector, `L_cls + L_mean` is just 2x one loss. The curve falls, the top1 rises, every
# shape checks out -- and the experiment measures nothing. No shape assertion can see it,
# so it gets numerical must-fail tests here AND a per-step assertion in the train loop.


class _SplitTinyEncoder(torch.nn.Module):
    """Minimal split-head model that runs the REAL ``WaivEncoder`` pooling/forward code.

    Only ``self`` is fake: ``_pool``, ``_pool_parts``, ``pool_from_parts``,
    ``embed_parts`` and ``forward_split`` are the shipped implementations, bound here so
    they can be exercised without downloading a 1.2 GB backbone. Same idiom as
    ``_seg_shim`` above.

    ``images`` is a ``(B, T, hidden)`` token tensor -- the backbone stand-in is per-token,
    which is exactly what makes cls and mean genuinely different vectors.
    """

    def __init__(self, heads=("cls", "mean"), hidden=8, n_tokens=7, num_prefix_tokens=1,
                 d_proj=512, use_lora=True, pool_head="mean"):
        import torch.nn as nn

        from waivphaet.models.encoder import ProjectionHead, WaivEncoder
        from waivphaet.models.pooling import build_pool_head

        super().__init__()
        self.cfg = types.SimpleNamespace(use_lora=use_lora, pooling="clsmean")
        self.hidden_size = hidden
        self.embed_dim = 2 * hidden
        self.n_tokens = n_tokens
        self.num_prefix_tokens = num_prefix_tokens
        self.split_heads = tuple(heads)
        self.pool_head_name = pool_head
        self.pool_head = None if pool_head == "mean" else build_pool_head(pool_head, hidden)
        self.backbone = _FakeAdapterBackbone(hidden, hidden)
        self.projector = None
        self.projectors = nn.ModuleDict(
            {h: ProjectionHead(hidden, 16, d_proj) for h in heads}
        )
        for name in ("_pool", "_pool_parts", "pool_from_parts", "embed_parts",
                     "forward_split", "pool_head_metrics"):
            setattr(self, name, types.MethodType(getattr(WaivEncoder, name), self))

    def tokens(self, images):
        return self.backbone(images)

    def embed(self, images):
        return self._pool(self.tokens(images))


def _split_batches(n_batches=3, n_groups=2, group_size=6, hidden=8, n_tokens=7, seed=11):
    """Constraint-satisfying collated batches whose pixels are ``(B, T, hidden)`` tokens."""
    out = []
    for i in range(n_batches):
        _, batch = _fake_collated(n_groups=n_groups, group_size=group_size, seed=seed + i)
        n = n_groups * group_size
        g = torch.Generator().manual_seed(2000 + i)
        batch["anchor"] = torch.randn(n, n_tokens, hidden, generator=g)
        batch["positive"] = torch.randn(n, n_tokens, hidden, generator=g)
        out.append(batch)
    return out


# --- G3(a) THE CRITICAL ONE: the two heads must get genuinely DIFFERENT inputs ---------


def test_split_heads_receive_genuinely_different_inputs():
    """cls_vec and mean_vec must differ by a real margin on a real batch, not merely in
    dtype/shape. If they did not, L_cls + L_mean would be 2x one loss."""
    from waivphaet.train.contrastive import assert_split_head_inputs

    torch.manual_seed(0)
    model = _SplitTinyEncoder()
    images = torch.randn(12, 7, 8)
    parts = model.embed_parts(images)

    assert set(parts) == {"cls", "mean"}
    assert parts["cls"].shape == parts["mean"].shape == (12, 8)
    # not the same object, not the same values, and not close
    assert parts["cls"].data_ptr() != parts["mean"].data_ptr()
    assert not torch.allclose(parts["cls"], parts["mean"])

    stats = assert_split_head_inputs(parts)
    # a REAL margin: the two pools are order-1 apart relative to their own norms
    assert stats["split_input_rel_distance"] > 0.5, stats
    assert abs(stats["split_input_cosine"]) < 0.5, stats

    # ... and the two heads then produce different projections, which is the thing the
    # loss actually sees.
    _, z = model.forward_split(images)
    assert not torch.allclose(z["cls"], z["mean"])


def test_split_head_input_assertion_MUST_FAIL_when_both_heads_get_the_same_vector():
    """The wiring bug, simulated: hand both heads the identical pooled vector.

    This is the test that has to RAISE. Without it the bug is invisible -- same shapes,
    same dtype, a perfectly plausible falling loss curve.
    """
    from waivphaet.train.contrastive import assert_split_head_inputs

    torch.manual_seed(0)
    v = torch.randn(12, 8)
    with pytest.raises(ValueError, match="SPLIT-HEAD WIRING BUG"):
        assert_split_head_inputs({"cls": v, "mean": v})
    # a *copy* is just as wrong as the same object, and must fail identically
    with pytest.raises(ValueError, match="SPLIT-HEAD WIRING BUG"):
        assert_split_head_inputs({"cls": v, "mean": v.clone()})
    # and near-identical (a bug that adds a whisker of noise) is caught too
    with pytest.raises(ValueError, match="SPLIT-HEAD WIRING BUG"):
        assert_split_head_inputs({"cls": v, "mean": v + 1e-9 * torch.randn_like(v)})


def test_split_head_input_assertion_MUST_FAIL_on_a_concat_width_input():
    """Feeding a head the 2048-d concat instead of one 1024-d pool is a width bug, and
    ``ProjectionHead`` would raise somewhere far away. Catch it at the source."""
    from waivphaet.train.contrastive import assert_split_head_inputs

    torch.manual_seed(0)
    cls = torch.randn(12, 8)
    with pytest.raises(ValueError, match="differ in shape"):
        assert_split_head_inputs({"cls": cls, "mean": torch.randn(12, 16)})


def test_split_head_loss_is_not_two_copies_of_one_loss():
    """End to end: the per-head losses must be genuinely different numbers."""
    from waivphaet.train.contrastive import split_head_info_nce

    torch.manual_seed(0)
    model = _SplitTinyEncoder()
    _, batch = _fake_collated(n_groups=2, group_size=6, seed=3)
    a = torch.randn(12, 7, 8)
    p = torch.randn(12, 7, 8)
    _, az = model.forward_split(a)
    _, pz = model.forward_split(p)
    loss, m = split_head_info_nce(
        az, pz, batch["group_id"], {"cls": 0.5, "mean": 0.5}, temperature=0.07
    )
    assert {"loss_cls", "loss_mean", "top1_cls", "top1_mean"} <= set(m)
    assert m["loss_cls"] != m["loss_mean"], "the two heads produced identical losses"
    assert abs(m["loss"] - (0.5 * m["loss_cls"] + 0.5 * m["loss_mean"])) < 1e-5


# --- G3(b) a zero-weight head is ABSENT, not zero-multiplied ---------------------------


def test_zero_weight_head_is_not_built_at_all():
    from waivphaet.train.contrastive import build_split_head_names

    assert build_split_head_names(0.5, 0.5) == ("cls", "mean")
    assert build_split_head_names(1.0, 0.0) == ("cls",)
    assert build_split_head_names(0.0, 1.0) == ("mean",)
    with pytest.raises(ValueError, match="both split-head weights are 0"):
        build_split_head_names(0.0, 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        build_split_head_names(-1.0, 1.0)


def test_mean_head_is_absent_and_gets_no_gradient_at_mean_weight_zero():
    """(G3b) ``--mean-weight 0`` must remove the head, not multiply it by zero.

    Absence is checked structurally (no module, no parameter, no state_dict entry) and
    dynamically (a full training run leaves gradient only on cls-head parameters).
    """
    import waivphaet.train.contrastive as C

    torch.manual_seed(0)
    model = _SplitTinyEncoder(heads=("cls",))

    # structural: the head does not exist anywhere in the module tree
    assert model.split_heads == ("cls",)
    assert "mean" not in model.projectors
    assert not any(k.startswith("projectors.mean.") for k in model.state_dict()), \
        sorted(model.state_dict())
    with pytest.raises(KeyError):
        # forward_split cannot run what was never built
        model.projectors["mean"]

    batches = _split_batches(n_batches=2)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = C.TrainConfig(
            out_dir=str(Path(tmp) / "run"), max_steps=len(batches), warmup_steps=1,
            log_every=1, eval_every=10**9, ckpt_every=10**9, amp_dtype="none",
            n_groups=2, group_size=6,
            split_heads=True, cls_weight=1.0, mean_weight=0.0,
            # no optimizer step on the last micro-batch, so .grad survives for inspection
            grad_accum=10**9,
        )
        summary = C.train(model, batches, cfg, device="cpu")

    got_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("projectors.cls.") for n in got_grad), got_grad
    assert not any(n.startswith("projectors.mean.") for n in got_grad), got_grad
    # the history records the single head and nothing about a phantom one
    rec = summary["history"][0]
    assert "loss_cls" in rec and "loss_mean" not in rec
    assert rec["n_heads"] == 1.0
    assert rec["loss"] == pytest.approx(rec["loss_cls"])
    assert rec["top1"] == pytest.approx(rec["top1_cls"])


def test_a_zero_weighted_head_would_still_move_its_batchnorm_stats():
    """WHY removal, not multiplication by zero -- with teeth.

    ``ProjectionHead`` contains ``nn.BatchNorm1d``. Its running mean/var update happens in
    the FORWARD pass and is not gated by the loss weight, so a head built at weight 0 keeps
    mutating state every step while contributing nothing. That is not a single-head arm.
    """
    from waivphaet.models.encoder import ProjectionHead

    torch.manual_seed(0)
    head = ProjectionHead(8, 16, 512).train()
    bn = head.net[1]
    before = bn.running_mean.clone()
    loss = 0.0 * head(torch.randn(12, 8)).sum()   # weight 0: contributes nothing
    loss.backward()
    assert not torch.equal(bn.running_mean, before), (
        "if this ever passes, a zero-weighted head is harmless and the removal could be "
        "dropped -- until then, do not build a head you are not training"
    )


def test_train_refuses_a_model_whose_heads_disagree_with_the_weights():
    """A config asking for two heads against a one-head model is the silent-arm bug."""
    import waivphaet.train.contrastive as C

    torch.manual_seed(0)
    model = _SplitTinyEncoder(heads=("cls",))
    batches = _split_batches(n_batches=1)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = C.TrainConfig(
            out_dir=str(Path(tmp) / "run"), max_steps=1, warmup_steps=1, log_every=1,
            eval_every=10**9, ckpt_every=10**9, amp_dtype="none", n_groups=2, group_size=6,
            split_heads=True, cls_weight=0.5, mean_weight=0.5,
        )
        with pytest.raises(ValueError, match="require the heads"):
            C.train(model, batches, cfg, device="cpu")


# --- G3(c) checkpoint round-trip preserves BOTH projectors -----------------------------


def test_checkpoint_roundtrip_preserves_both_projectors():
    import waivphaet.train.contrastive as C

    torch.manual_seed(0)
    model = _SplitTinyEncoder(heads=("cls", "mean"))
    # make the two heads genuinely different so a "saved one twice" bug cannot pass
    with torch.no_grad():
        for p in model.projectors["mean"].parameters():
            p.add_(torch.randn_like(p))
    saved = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("projectors.")}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = C.save_projectors(model, out)
        assert manifest["split_heads"] and manifest["heads"] == ["cls", "mean"]
        assert (out / "projector_cls.pt").exists() and (out / "projector_mean.pt").exists()
        # projector.pt must exist too: extract_pathorob_features.build_model loads it
        # UNCONDITIONALLY on the adapter path, so its absence is a crashed eval follower.
        assert (out / "projector.pt").exists()
        alias = json.loads((out / "projector_heads.json").read_text())["projector_pt_alias"]
        assert alias == "cls"
        # it is a real head's weights, and a one-pool head, so a clsmean eval will skip it
        proj_sd = torch.load(out / "projector.pt", map_location="cpu")
        assert proj_sd["net.0.weight"].shape[1] == model.hidden_size != model.embed_dim

        torch.manual_seed(999)
        fresh = _SplitTinyEncoder(heads=("cls", "mean"))
        assert not torch.equal(
            fresh.state_dict()["projectors.mean.net.0.weight"],
            saved["projectors.mean.net.0.weight"],
        ), "the fresh model already matches; the round-trip would be vacuous"
        C.load_projectors(fresh, out)

    for k, v in saved.items():
        assert torch.equal(fresh.state_dict()[k], v), f"{k} did not round-trip"


def test_single_head_checkpoint_artifact_is_unchanged():
    """The default path must still write exactly one ``projector.pt`` and no manifest."""
    import waivphaet.train.contrastive as C

    torch.manual_seed(0)
    model = _TinyEncoder()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = C.save_projectors(model, out)
        assert manifest["split_heads"] is False
        assert sorted(p.name for p in out.iterdir()) == ["projector.pt"]
        fresh = _TinyEncoder()
        C.load_projectors(fresh, out)
    for k, v in model.projector.state_dict().items():
        assert torch.equal(fresh.projector.state_dict()[k], v)


# --- G4 shapes: head inputs are `hidden`, embed() is untouched, prefix tokens are read --


def _fake_hf_encoder(monkeypatch, hidden=1024, n_tokens=197, split_heads=(),
                     pooling="clsmean", pool_head="mean", tokens=None):
    """A REAL ``WaivEncoder`` over a fake HF backbone -- no download, real construction.

    This is what makes the width assertions meaningful: the ProjectionHeads below are
    built by the shipped ``__init__``, at the real phikon-v2 hidden size.
    """
    import waivphaet.models.encoder as E

    # ``tokens`` is injectable so two encoders can be built over the SAME backbone output,
    # which is what makes "these two arms are byte-identical" a real comparison rather
    # than a comparison of two different random draws.
    tokens = torch.randn(3, n_tokens, hidden) if tokens is None else tokens
    backbone = _HFBackbone(tokens)
    backbone.config = types.SimpleNamespace(
        hidden_size=hidden, num_hidden_layers=2, patch_size=16, image_size=224,
        model_type="dinov2",
    )
    monkeypatch.setattr(E, "AutoModel",
                        types.SimpleNamespace(from_pretrained=lambda *_a, **_k: backbone))
    monkeypatch.setattr(E, "is_timm_backbone", lambda _b: False)
    monkeypatch.setattr(E, "normalization_for", lambda _b: (E.IMAGENET_MEAN, E.IMAGENET_STD))
    model = E.WaivEncoder(E.EncoderConfig(
        backbone="owkin/phikon-v2", use_lora=False, pooling=pooling,
        split_heads=tuple(split_heads), pool_head=pool_head,
    ))
    return model, tokens


def test_split_head_widths_are_hidden_and_embed_stays_2048(monkeypatch):
    """(G4) proj_cls in == proj_mean in == hidden (1024); ``embed()`` still 2048-d."""
    model, _ = _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"))

    assert model.hidden_size == 1024
    assert model.embed_dim == 2048, "eval pooling width must not follow the training split"
    assert model.projector is None
    for name in ("cls", "mean"):
        w = model.projectors[name].net[0].weight
        assert w.shape[1] == 1024, f"{name} head takes {w.shape[1]}-d, expected hidden=1024"

    images = torch.randn(3, 3, 224, 224)
    assert model.embed(images).shape == (3, 2048)
    parts = model.embed_parts(images)
    assert parts["cls"].shape == (3, 1024) and parts["mean"].shape == (3, 1024)
    # and the split really is the two halves of what embed() exports
    assert torch.equal(model.embed(images), model.pool_from_parts(parts))
    assert torch.equal(model.embed(images)[:, :1024], parts["cls"])

    _, z = model.forward_split(images)
    assert z["cls"].shape == z["mean"].shape == (3, 512)
    with pytest.raises(RuntimeError, match="no single `projector`"):
        model(images)


def test_single_head_encoder_is_structurally_unchanged(monkeypatch):
    """Default construction: one 2048-d concat projector, no ModuleDict, no forward_split."""
    model, _ = _fake_hf_encoder(monkeypatch)
    assert model.split_heads == ()
    assert model.projectors is None
    assert model.projector.net[0].weight.shape[1] == 2048 == model.embed_dim
    emb, z = model(torch.randn(3, 3, 224, 224))
    assert emb.shape == (3, 2048) and z.shape == (3, 512)
    with pytest.raises(RuntimeError, match="requires an encoder built with cfg.split_heads"):
        model.forward_split(torch.randn(3, 3, 224, 224))


def test_split_encoder_rejects_an_unknown_head_name(monkeypatch):
    import waivphaet.models.encoder as E

    with pytest.raises(ValueError, match="unknown split head"):
        _fake_hf_encoder(monkeypatch, split_heads=("cls", "patchmax"))
    with pytest.raises(ValueError, match="duplicate split head"):
        _fake_hf_encoder(monkeypatch, split_heads=("cls", "cls"))
    assert E.POOL_PARTS == ("cls", "mean")


@pytest.mark.parametrize("num_prefix_tokens,n_tokens", [(1, 197), (5, 261)])
def test_pool_parts_reads_num_prefix_tokens_from_the_encoder(num_prefix_tokens, n_tokens):
    """(G4) The mean head's slice must come from the ENCODER, never a hardcoded 1.

    phikon-v2 emits 1 prefix token, Virchow2 emits 5 ([CLS] + 4 registers). Averaging the
    registers in is right-shape, right-dtype, no-warning, just a worse number -- so the
    5-prefix case gets a numerical test against the model card's own expression.
    """
    from waivphaet.models.encoder import WaivEncoder

    torch.manual_seed(0)
    tokens = torch.randn(3, n_tokens, 32, dtype=torch.float64)
    shim = _PoolOnly("clsmean", num_prefix_tokens)
    parts = WaivEncoder._pool_parts(shim, tokens)

    assert torch.equal(parts["cls"], tokens[:, 0, :])
    assert torch.equal(parts["mean"], tokens[:, num_prefix_tokens:, :].mean(dim=1))
    # consistent with _pool, which is what every published number came through
    assert torch.equal(
        torch.cat([parts["cls"], parts["mean"]], dim=1), shim.pool(tokens)
    )
    if num_prefix_tokens == 5:
        naive = tokens[:, 1:, :].mean(dim=1)     # the hardcoded-1 bug
        assert not torch.allclose(parts["mean"], naive), "the test has no teeth"


# --- G5 scale neutrality: 0.5/0.5 on the SAME input reproduces the single-head loss ----


def test_split_weights_are_scale_neutral_not_a_hidden_lr_change():
    """(G5) Same vector into both heads, weights 0.5/0.5 -> exactly the single-head loss.

    This is the check that 0.5/0.5 is a convex combination rather than a doubled learning
    rate in disguise. It also fails loudly if someone "helpfully" changes the defaults to
    1.0/1.0, which would make the arm measure 'split PLUS 2x LR'.
    """
    import copy

    from waivphaet.train.contrastive import (
        TrainConfig, masked_info_nce, split_head_info_nce,
    )
    from waivphaet.models.encoder import ProjectionHead

    torch.manual_seed(0)
    head = ProjectionHead(8, 16, 512).eval()   # eval(): BN in inference mode, so the two
    twin = copy.deepcopy(head)                 # calls below cannot differ via BN stats
    _, batch = _fake_collated(n_groups=2, group_size=6, seed=5)
    gid = batch["group_id"]
    a, p = torch.randn(12, 8), torch.randn(12, 8)

    single, m_single = masked_info_nce(head(a), head(p), gid, 0.07)
    total, m_split = split_head_info_nce(
        {"cls": head(a), "mean": twin(a)}, {"cls": head(p), "mean": twin(p)},
        gid, {"cls": 0.5, "mean": 0.5}, temperature=0.07,
    )

    assert m_split["loss_cls"] == pytest.approx(m_split["loss_mean"], abs=0.0)
    err = abs(float(total.detach()) - float(single.detach()))
    assert err < 1e-6, f"0.5/0.5 is not scale-neutral: |split - single| = {err:e}"
    assert m_split["top1"] == pytest.approx(m_single["top1"])

    # 1.0/1.0 would NOT be -- that is precisely why it is not the default.
    doubled, _ = split_head_info_nce(
        {"cls": head(a), "mean": twin(a)}, {"cls": head(p), "mean": twin(p)},
        gid, {"cls": 1.0, "mean": 1.0}, temperature=0.07,
    )
    assert float(doubled.detach()) == pytest.approx(
        2.0 * float(single.detach()), rel=1e-6
    )

    # and the shipped defaults really are the scale-neutral ones
    cfg = TrainConfig()
    assert (cfg.cls_weight, cfg.mean_weight) == (0.5, 0.5)
    assert cfg.cls_weight + cfg.mean_weight == 1.0
    assert cfg.split_heads is False


# --- CLI / config plumbing -------------------------------------------------------------


def _train_lora_module():
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("_train_lora_split_test", str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_split_head_cli_flags_and_defaults():
    mod = _train_lora_module()
    old = sys.argv
    try:
        sys.argv = ["prog", "--out-dir", "/tmp/x"]
        args = mod.parse_args()
    finally:
        sys.argv = old
    assert args.split_heads is False
    # None until passed -- that is how "was it given without --split-heads?" stays
    # answerable, exactly as --n-groups/--group-size do for --grid.
    assert args.cls_weight is None and args.mean_weight is None


def test_split_head_help_states_the_scale_neutrality_rationale():
    """The 0.5/0.5 default is a controlled variable, and the help text has to say so."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py").read_text()
    lo = src.index("--cls-weight")
    hi = src.index("--grad-checkpointing")
    block = src[lo:hi].lower()
    assert "0.5" in block
    assert "scale" in block or "gradient magnitude" in block
    assert "learning rate" in block
    assert "batchnorm" in block, "the zero-weight-head removal rationale must be stated"


def test_split_head_weights_without_the_flag_are_an_error(tmp_path):
    """A silently-ignored objective flag is how two runs get labelled the same."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_lora.py"),
         "--out-dir", str(tmp_path / "x"), "--cls-weight", "1.0"],
        capture_output=True, text=True, cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
    )
    assert r.returncode != 0
    assert "requires --split-heads" in (r.stderr + r.stdout)


def test_split_heads_and_grid_are_mutually_exclusive(tmp_path):
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_lora.py"),
         "--out-dir", str(tmp_path / "x"), "--split-heads", "--grid"],
        capture_output=True, text=True, cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
    )
    assert r.returncode != 0
    assert "not implemented for --grid" in (r.stderr + r.stdout)


def test_train_config_records_the_split_head_objective():
    """config.json is the only place a later reader can tell the arms apart."""
    import waivphaet.train.contrastive as C

    cfg = C.TrainConfig(split_heads=True, cls_weight=1.0, mean_weight=0.0)
    d = json.loads(json.dumps(dataclasses.asdict(cfg)))
    assert d["split_heads"] is True
    assert d["cls_weight"] == 1.0 and d["mean_weight"] == 0.0


def test_split_head_history_logs_both_terms_separately():
    """Without per-head loss and top1 in history.json the three arms are uninterpretable."""
    import waivphaet.train.contrastive as C

    torch.manual_seed(0)
    model = _SplitTinyEncoder(heads=("cls", "mean"))
    batches = _split_batches(n_batches=3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        cfg = C.TrainConfig(
            out_dir=str(out), max_steps=len(batches), warmup_steps=1, log_every=1,
            eval_every=10**9, ckpt_every=10**9, amp_dtype="none", n_groups=2, group_size=6,
            split_heads=True, cls_weight=0.5, mean_weight=0.5,
        )
        C.train(model, batches, cfg, device="cpu")
        hist = json.loads((out / "history.json").read_text())
    assert len(hist) == 3
    for rec in hist:
        for k in ("loss", "loss_cls", "loss_mean", "top1", "top1_cls", "top1_mean",
                  "split_input_rel_distance", "split_input_cosine"):
            assert k in rec, f"{k} missing from history.json"
        assert rec["loss"] == pytest.approx(0.5 * rec["loss_cls"] + 0.5 * rec["loss_mean"])
        assert rec["loss_cls"] != rec["loss_mean"]
        assert rec["split_input_rel_distance"] > 1e-3


def test_both_probe_readers_skip_a_width_mismatched_projector():
    """A split-head checkpoint's projector.pt is 1024-d while the clsmean eval is 2048-d.

    Both readers that load it -- ``embed_probe.load_adapter`` and
    ``extract_pathorob_features.build_model`` -- must SKIP it and say so, not raise. They
    score ``model.embed()``; the projector is training-only. Unguarded, the LoRA branch of
    ``embed_probe`` hard-crashed the RI-curve follower on every checkpoint of a split-head
    run (size mismatch for net.0.weight, 1024 vs 2048) and lost the whole curve for a
    tensor it never reads.
    """
    repo = Path(__file__).resolve().parents[1]
    for rel, fn in (("scripts/embed_probe.py", "load_adapter"),
                    ("scripts/extract_pathorob_features.py", "build_model")):
        src = (repo / rel).read_text()
        i = src.index(f"def {fn}(")
        body = src[i:i + 6000]
        j = body.index('projector.pt')
        window = body[j:j + 1200]
        assert "model.embed_dim" in window, f"{rel}:{fn} loads projector.pt unguarded"
        assert "skipping projector" in window, f"{rel}:{fn} skips silently"


# ======================================================================================
# ALTERNATIVE TOKEN POOLINGS -- gem / attn / lse for the non-CLS loss head
# ======================================================================================
#
# RESULTS 9's defect, restated: `mean` is LINEAR, so d(mean)/d(t_i) = (1/N) I and the direct
# gradient reaching every patch token is the IDENTICAL vector. The loss can translate the
# token cloud but never expresses a preference about the tokens' relative arrangement --
# and a uniform translation is exactly what THUNDER's biased proj_dec absorbs into its
# bias. G3 below is the entire point of the feature: it MEASURES the per-token gradient
# spread and asserts the partition (mean == 0, the others >> 0).


def _layernormed_tokens(b=4, n=196, d=64, seed=7):
    """Tokens with the property that actually drives the sign problem: zero-mean channels.

    The last op before a ViT's token sequence leaves the backbone is a LayerNorm, so every
    token vector is EXACTLY zero-mean across its channels and ~half its entries are
    negative. Random gaussians are only approximately that; applying the LayerNorm makes
    the test's premise exact rather than statistical, which is what lets G4 assert a
    ~50% clamp-zeroed fraction rather than "some".
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, d, generator=g) * 2.0 + 0.3
    return torch.nn.functional.layer_norm(x, (d,))


# --- G3 THE CRITICAL GATE: d(pool)/d(t_i) must depend on i ------------------------------


def test_G3_mean_pooling_gives_every_token_the_IDENTICAL_gradient():
    """The defect, documented as a number. This test is SUPPOSED to show zero spread."""
    from waivphaet.models.pooling import MeanPool, token_gradient_spread

    x = _layernormed_tokens()
    stats = token_gradient_spread(MeanPool(), x)
    # ~1e-7 rather than a hard 0.0: the only difference between g_i and mean_i(g) is the
    # rounding of the float32 mean reduction itself. That number IS the noise floor, and
    # G3's 1e-2 margin for the alternatives sits five orders of magnitude above it.
    assert stats["spread_max"] < 1e-6, stats
    # and the underlying claim, directly: g_i == g_j for every pair
    xg = x.clone().requires_grad_(True)
    MeanPool()(xg).sum().backward()
    g = xg.grad
    assert torch.equal(g[:, 0, :], g[:, -1, :])
    assert torch.allclose(g, torch.full_like(g, 1.0 / x.shape[1]))


@pytest.mark.parametrize("name", ["gem", "gem_clamp", "attn", "lse"])
def test_G3_alternative_poolings_have_token_DEPENDENT_gradients(name):
    """(G3) The gate the whole change lives or dies on.

    A variant whose per-token gradients are (near-)identical is just the mean wearing a
    hat: it can still only translate the token cloud, so it cannot address the
    segmentation null-space argument and is not worth a GPU. The margin is 1e-2, four
    orders of magnitude above the float32 noise floor that `mean` measures at (exactly 0).
    """
    from waivphaet.models.pooling import build_pool_head, token_gradient_spread

    torch.manual_seed(0)
    x = _layernormed_tokens()
    pool = build_pool_head(name, x.shape[-1])
    stats = token_gradient_spread(pool, x)
    assert stats["spread_min"] > 1e-2, (name, stats)
    assert stats["max_abs_grad"] > 0.0, (name, stats)


def test_G3_pool_head_registry_and_partition_are_consistent():
    """The names the CLI offers, the ones the encoder accepts and the ones G3 claims are
    token-dependent must be ONE list, not three that can drift apart."""
    import waivphaet.models.pooling as P
    from waivphaet.models.encoder import EncoderConfig

    assert P.POOL_HEAD_NAMES == ("mean", "gem", "gem_clamp", "attn", "lse")
    assert set(P.TOKEN_DEPENDENT_POOLS) == set(P.POOL_HEAD_NAMES) - {"mean"}
    assert EncoderConfig().pool_head == "mean", "the default must remain the incumbent"
    for n in P.POOL_HEAD_NAMES:
        assert isinstance(P.build_pool_head(n, 8), torch.nn.Module)
    with pytest.raises(ValueError, match="unknown pool head"):
        P.build_pool_head("softmax", 8)


# --- G4 SIGN HANDLING: what the clamp destroys, measured --------------------------------


def test_G4_clamp_gem_zeroes_about_half_of_every_token_and_says_so():
    """(G4) THE reason plain clamp-GeM is not the default.

    GeM assumes post-ReLU x >= 0. LayerNorm'd ViT tokens are exactly zero-mean, so the
    textbook clamp deletes ~half of every token vector -- silently, with the right shape.
    The number has to be reported, so it is asserted here and logged as
    ``pool_zero_fraction`` in history.json.
    """
    from waivphaet.models.pooling import GeMPool

    x = _layernormed_tokens()
    assert (x < 0).float().mean() > 0.4, "the fixture is not actually signed"

    clamp = GeMPool(mode="clamp")
    clamp(x)
    frac = clamp.extra_metrics()["pool_zero_fraction"]
    assert 0.4 < frac < 0.6, f"expected ~half the entries clamped away, measured {frac}"

    # ... and those entries get EXACTLY zero gradient, which is the part that matters:
    # the pooling cannot start discriminating between tokens in coordinates it cannot see.
    xg = x.clone().requires_grad_(True)
    GeMPool(mode="clamp")(xg).sum().backward()
    dead = (xg.grad == 0).float().mean()
    assert dead > 0.4, f"clamped entries should be gradient-dead, measured {dead}"


@pytest.mark.parametrize("name", ["gem", "attn", "lse"])
def test_G4_softplus_attn_and_lse_zero_nothing(name):
    """(G4) The shipped variants must not destroy a single entry, nor kill a gradient."""
    from waivphaet.models.pooling import build_pool_head

    torch.manual_seed(0)
    x = _layernormed_tokens()
    pool = build_pool_head(name, x.shape[-1])
    assert pool.extra_metrics().get("pool_zero_fraction", 0.0) == 0.0

    xg = x.clone().requires_grad_(True)
    out = pool(xg)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(xg.grad).all()
    assert (xg.grad == 0).float().mean() == 0.0, (
        f"{name} produced dead gradients; only the clamp variant may"
    )
    if name == "gem":
        pool(x)
        assert pool.extra_metrics()["pool_zero_fraction"] == 0.0


def test_G4_softplus_gem_keeps_the_ordering_the_clamp_destroys():
    """Monotonicity is the whole reason softplus replaces the clamp: two tokens that
    differ ONLY in their negative entries are indistinguishable to clamp-GeM and
    distinguishable to softplus-GeM."""
    from waivphaet.models.pooling import GeMPool

    a = torch.tensor([[[-1.0, 2.0], [-3.0, 2.0]]])   # (1, 2, 2)
    b = torch.tensor([[[-9.0, 2.0], [-0.5, 2.0]]])   # same positives, different negatives
    assert torch.allclose(GeMPool(mode="clamp")(a), GeMPool(mode="clamp")(b)), (
        "the clamp is supposed to be blind here -- if it is not, this test has no teeth"
    )
    assert not torch.allclose(GeMPool(mode="softplus")(a), GeMPool(mode="softplus")(b))


def test_pool_head_learnable_state_is_reported_and_starts_where_documented():
    from waivphaet.models.pooling import build_pool_head

    torch.manual_seed(0)
    x = _layernormed_tokens()
    gem = build_pool_head("gem", 64)
    assert gem.extra_metrics()["pool_gem_p"] == pytest.approx(3.0)
    lse = build_pool_head("lse", 64)
    assert lse.extra_metrics()["pool_lse_tau"] == pytest.approx(1.0)
    attn = build_pool_head("attn", 64)
    attn(x)
    m = attn.extra_metrics()
    # Non-degenerate but not peaked: the query is unit-VARIANCE per component (NOT unit
    # norm), which puts the logits at std ~1 and keeps the attention from collapsing to
    # the uniform weighting that would make this head literally the mean pooling. A
    # unit-norm query measured entropy 0.99999 / G3 spread 6e-3 on real phikon-v2 tokens.
    assert 0.5 < m["pool_attn_entropy"] < 0.99999, m
    assert m["pool_attn_max"] > 1.5 / 196
    # every variant's parameters must be TRAINABLE, or the "learnable p/tau" claim is false
    for p in list(gem.parameters()) + list(lse.parameters()) + list(attn.parameters()):
        assert p.requires_grad
    assert [p.numel() for p in gem.parameters()] == [1]
    assert [p.numel() for p in lse.parameters()] == [1]


def test_lse_pooling_interpolates_mean_to_max():
    """tau -> 0 is the mean, large tau is the max. The interpolation is the claim."""
    from waivphaet.models.pooling import LSEPool

    x = _layernormed_tokens(b=2, n=32, d=8)
    tiny = LSEPool(tau_init=1e-6)
    assert torch.allclose(tiny(x), x.mean(dim=1), atol=1e-4)
    # tau is clamped at e^4 ~ 54.6 (see LSEPool.log_tau_max), so "max" here means
    # decisively nearer the max than the mean, not the max to machine precision.
    big = LSEPool(tau_init=50.0)
    out, mx, mn = big(x), x.max(dim=1).values, x.mean(dim=1)
    assert (out - mx).abs().max() < 0.05
    assert (out - mx).abs().mean() * 20 < (mn - mx).abs().mean()


# --- G5 SHAPES + EVAL ISOLATION ---------------------------------------------------------


@pytest.mark.parametrize("pool_head", ["gem", "gem_clamp", "attn", "lse"])
def test_G5_pool_head_widths_are_hidden_and_embed_is_untouched(monkeypatch, pool_head):
    """(G5) Each head's projector input is `hidden` (1024), and ``embed()`` at pooling
    ``clsmean`` still returns 2048-d and is byte-identical to a non-pool-head run's.

    Eval pooling is a PROTOCOL CONSTANT (PathoROB's reference row is ``phikonv2_clsmean``);
    the alternative poolings are training-time loss heads and must not leak into it.
    """
    shared = torch.randn(3, 197, 1024)
    plain, _ = _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"), tokens=shared)
    model, _ = _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"),
                                pool_head=pool_head, tokens=shared)

    assert model.pool_head_name == pool_head and model.pool_head is not None
    assert model.embed_dim == 2048
    for name in ("cls", "mean"):
        assert model.projectors[name].net[0].weight.shape[1] == 1024

    images = torch.randn(3, 3, 224, 224)
    emb = model.embed(images)
    assert emb.shape == (3, 2048)
    # THE eval-isolation assertion: byte-identical to the run without a pool head
    assert torch.equal(emb, plain.embed(images))
    assert torch.equal(model.embed_parts(images)["mean"], plain.embed_parts(images)["mean"])

    parts, z = model.forward_split(images)
    assert z["cls"].shape == z["mean"].shape == (3, 512)
    # `parts` still carries the TRUE clsmean halves -- the retention term reassembles them
    assert torch.equal(model.pool_from_parts(parts), emb)
    # ... and the head's actual input is published separately and really is different
    assert parts["pool"].shape == (3, 1024)
    assert not torch.allclose(parts["pool"], parts["mean"])


def test_G5_pool_head_mean_is_the_SAME_CODE_as_no_pool_head(monkeypatch):
    """(G1/G5) ``--pool-head mean`` must not perturb the arm already running.

    Guaranteed structurally, not numerically: on 'mean' the encoder builds NO pooling
    module at all and keeps the literal ``patches.mean(dim=1)`` inside ``_pool_parts``. So
    there is no new parameter, no new state_dict key and no RNG draw to shift the
    projector init.
    """
    shared = torch.randn(3, 197, 1024)
    plain, _ = _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"), tokens=shared)
    explicit, _ = _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"),
                                   pool_head="mean", tokens=shared)
    assert plain.pool_head is None and explicit.pool_head is None
    assert plain.state_dict().keys() == explicit.state_dict().keys()
    assert not any("pool_head" in k for k in plain.state_dict())

    images = torch.randn(3, 3, 224, 224)
    p_parts, _ = plain.forward_split(images)
    e_parts, _ = explicit.forward_split(images)
    assert set(p_parts) == set(e_parts) == {"cls", "mean"}, "no 'pool' key on the mean arm"
    for k in p_parts:
        assert torch.equal(p_parts[k], e_parts[k])


def test_pool_head_reads_num_prefix_tokens_from_the_encoder(monkeypatch):
    """(G5) 1 on phikon-v2, 5 on Virchow2. The pooling modules never see the prefix slice
    at all -- the ENCODER does it -- so this asserts the encoder passes the right window."""
    import waivphaet.models.encoder as E

    for num_prefix, n_tokens in ((1, 197), (5, 261)):
        model, tokens = _fake_hf_encoder(monkeypatch, hidden=32, n_tokens=n_tokens,
                                         split_heads=("cls", "mean"), pool_head="lse")
        model.num_prefix_tokens = num_prefix
        parts, _ = model.forward_split(torch.randn(3, 3, 224, 224))
        expected = model.pool_head(tokens[:, num_prefix:, :])
        assert torch.equal(parts["pool"], expected)
        if num_prefix == 5:
            naive = model.pool_head(tokens[:, 1:, :])     # the hardcoded-1 bug
            assert not torch.allclose(parts["pool"], naive), "the test has no teeth"
    assert E.POOL_PARTS == ("cls", "mean")


def test_pool_head_requires_a_mean_split_head(monkeypatch):
    with pytest.raises(ValueError, match="requires split_heads to include 'mean'"):
        _fake_hf_encoder(monkeypatch, split_heads=(), pool_head="gem")
    with pytest.raises(ValueError, match="requires split_heads to include 'mean'"):
        _fake_hf_encoder(monkeypatch, split_heads=("cls",), pool_head="gem")
    with pytest.raises(ValueError, match="unknown pool_head"):
        _fake_hf_encoder(monkeypatch, split_heads=("cls", "mean"), pool_head="max")


# --- the per-step wiring assertion must watch what the head ACTUALLY got -----------------


def test_split_head_assertion_watches_the_POOLED_input_not_the_unused_mean():
    """With a gem/attn/lse head, ``parts['mean']`` is still the true arithmetic mean (the
    retention term needs it) but the head is fed ``parts['pool']``. If the assertion kept
    reading 'mean' it would be comparing CLS against a vector nothing consumed, and a gem
    head accidentally wired to CLS would sail straight past it."""
    from waivphaet.train.contrastive import assert_split_head_inputs

    cls = torch.randn(6, 8)
    parts = {"cls": cls, "mean": torch.randn(6, 8), "pool": cls.clone()}
    with pytest.raises(ValueError, match="SPLIT-HEAD WIRING BUG"):
        assert_split_head_inputs(parts)
    # and the legacy two-key form is untouched
    stats = assert_split_head_inputs({"cls": cls, "mean": torch.randn(6, 8)})
    assert stats["split_input_rel_distance"] > 0.5


@pytest.mark.parametrize("pool_head", ["gem", "attn", "lse"])
def test_pool_head_end_to_end_step_trains_the_pooling_parameters(pool_head):
    """One real optimiser step through the shipped split-head loss: the pooling's own
    parameters must receive gradient, or the "learnable p / tau / query" claim is empty."""
    from waivphaet.train.contrastive import assert_split_head_inputs, split_head_info_nce

    torch.manual_seed(0)
    model = _SplitTinyEncoder(hidden=8, n_tokens=7, pool_head=pool_head)
    batch = _split_batches(1)[0]
    parts, az = model.forward_split(batch["anchor"])
    _, pz = model.forward_split(batch["positive"])
    stats = assert_split_head_inputs(parts)
    assert stats["split_input_rel_distance"] > 1e-3
    loss, metrics = split_head_info_nce(
        az, pz, batch["group_id"], {"cls": 0.5, "mean": 0.5}, 0.07, False
    )
    loss.backward()
    grads = {n: p.grad for n, p in model.pool_head.named_parameters()}
    assert grads, "the pooling has no parameters to train"
    for n, g in grads.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
    assert set(metrics) >= {"loss_cls", "loss_mean", "top1_cls", "top1_mean"}
    assert set(model.pool_head_metrics()), "pooling state must be logged"


def test_pool_head_checkpoint_roundtrips(tmp_path):
    """GeM's p, LSE's tau and the attention query are TRAINED and are not reconstructible
    from anything else in the directory, so they have to be in the checkpoint."""
    from waivphaet.train.contrastive import load_projectors, save_projectors

    torch.manual_seed(0)
    model = _SplitTinyEncoder(hidden=8, pool_head="gem")
    with torch.no_grad():
        model.pool_head.p.fill_(4.25)
    man = save_projectors(model, tmp_path)
    assert man["pool_head"] == "gem" and man["pool_head_pt"] == "pool_head.pt"
    assert (tmp_path / "pool_head.pt").exists()

    fresh = _SplitTinyEncoder(hidden=8, pool_head="gem")
    assert float(fresh.pool_head.p.detach()) == pytest.approx(3.0)
    load_projectors(fresh, tmp_path)
    assert float(fresh.pool_head.p.detach()) == pytest.approx(4.25)


def test_mean_pool_head_checkpoint_artifact_is_unchanged(tmp_path):
    """The arm already running must keep writing exactly the files it writes today."""
    from waivphaet.train.contrastive import save_projectors

    model = _SplitTinyEncoder(hidden=8, pool_head="mean")
    man = save_projectors(model, tmp_path)
    assert not (tmp_path / "pool_head.pt").exists()
    assert man["pool_head"] == "mean" and "pool_head_pt" not in man
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "projector.pt", "projector_cls.pt", "projector_heads.json", "projector_mean.pt",
    ]


def test_train_refuses_a_config_pool_head_the_encoder_does_not_have():
    """A run whose name says gem and whose encoder pools with the mean is a result nothing
    downstream could ever catch. Fail before any compute."""
    from waivphaet.train.contrastive import TrainConfig, train

    model = _SplitTinyEncoder(hidden=8, pool_head="mean")
    cfg = TrainConfig(split_heads=True, pool_head="gem", max_steps=1)
    with pytest.raises(ValueError, match="the encoder pools with 'mean'"):
        train(model, [], cfg, device="cpu")
    assert TrainConfig().pool_head == "mean"


# --- CLI plumbing ------------------------------------------------------------------------


def test_pool_head_cli_flag_defaults_and_choices():
    mod = _train_lora_module()
    old = sys.argv
    try:
        sys.argv = ["prog", "--out-dir", "/tmp/x"]
        args = mod.parse_args()
    finally:
        sys.argv = old
    # None until passed -- that is how "was it given without --split-heads?" stays
    # answerable, exactly as --cls-weight / --n-groups do.
    assert args.pool_head is None

    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py").read_text()
    i = src.index('"--pool-head"')
    block = src[i:i + 2000]
    assert 'choices=list(POOL_HEAD_NAMES)' in block
    for phrase in ("d(mean)/d(t_i)", "TOKEN-DEPENDENT", "SOFTPLUS", "DIAGNOSTIC ONLY",
                   "LayerNorm", "Eval pooling is UNAFFECTED"):
        assert phrase in block, f"--pool-head help must state {phrase!r}"


def test_pool_head_without_split_heads_is_an_error(tmp_path):
    """A silently-ignored objective flag is how two runs get labelled the same."""
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_lora.py"),
         "--out-dir", str(tmp_path / "x"), "--pool-head", "gem"],
        capture_output=True, text=True, cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
    )
    assert r.returncode != 0
    out = r.stderr + r.stdout
    assert "requires --split-heads" in out and "protocol constant" in out


def test_pool_head_gem_clamp_warns_that_it_is_a_diagnostic():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py").read_text()
    i = src.index('pool_head == "gem_clamp"')
    window = src[i:i + 900]
    assert "WARNING" in window and "pool_zero_fraction" in window
    assert "--pool-head gem" in window, "the warning must name the variant to use instead"


def test_train_config_and_encoder_config_both_record_the_pooling():
    """config.json is the only place a later reader can tell a gem run from a mean one."""
    from waivphaet.models.encoder import EncoderConfig
    from waivphaet.train.contrastive import TrainConfig

    assert "pool_head" in dataclasses.asdict(TrainConfig())
    assert EncoderConfig(split_heads=("cls", "mean"), pool_head="attn").pool_head == "attn"
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py").read_text()
    assert "pool_head=pool_head" in src
    assert '"pool_head": pool_head' in src, "the encoder block must record it too"


def test_probe_readers_ignore_pool_head_pt():
    """(G6) ``embed_probe.load_adapter`` and ``extract_pathorob_features.build_model`` must
    tolerate the new checkpoints. Both load by NAME, never by globbing the directory, so an
    extra pool_head.pt is inert -- and neither may start reading it, because the pooling is
    a training-time loss head and every eval reads model.embed()."""
    repo = Path(__file__).resolve().parents[1]
    for rel, fn in (("scripts/embed_probe.py", "def load_adapter"),
                    ("scripts/extract_pathorob_features.py", "def build_model")):
        src = (repo / rel).read_text()
        assert "pool_head" not in src, f"{rel} must not read the training-time pooling"
        # The checkpoint reader must load BY NAME. A directory enumeration there would
        # make every new artifact a potential crash for the whole RI follower -- which is
        # exactly the class of launch blocker RESULTS 9 records for projector.pt.
        body = src[src.index(fn):][:6000]
        assert ".iterdir()" not in body and ".glob(" not in body, (
            f"{rel}:{fn} enumerates the checkpoint dir; an extra artifact could break it"
        )
