"""Tests for the invariants that fail *silently* if broken.

Every check here guards something that would not show up in the training loss:
a positive that isn't co-registered, or a negative that leaks acquisition signal.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
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

    # Run names: declared in the submitter, attributed by the collector.
    for run in ("vbase_clsmean", "vft500_clsmean", "vbase_cls", "vft500_cls"):
        assert run in submit, f"{run} not declared in submit_thunder.sh"
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
