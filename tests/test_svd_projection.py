"""Unit tests for the INFERENCE-TIME nuisance projection inside ``WaivEncoder`` (CPU).

``tests/test_svd_nuisance.py`` covers the cached-feature route (fit + apply on PathoROB
npzs). This file covers the other application point: the same rank-k subtraction done
inside the encoder, which is the only way HEST and THUNDER can see it -- both embed
their own tiles and never hand back a vector we could post-process.

Tests:
  (a) the encoder's output equals the reference numpy formula f - (f @ Vk.T) @ Vk
  (b) k=0 is a bit-exact no-op
  (c) every output row is orthogonal to every row of Vk
  (d) the projection is applied EXACTLY once, and to the whole pooled vector rather
      than to the clsmean halves separately
  (e) a D-mismatched fit is a hard error; a backbone-mismatched one warns and runs

No GPU, no HF download, no real fit: the backbone is a stub and the npz is synthetic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from waivphaet.models.encoder import EncoderConfig, WaivEncoder  # noqa: E402

HIDDEN = 8
NUM_TOKENS = 5
BATCH = 6
FIT_BACKBONE = "paige-ai/Virchow2"


class _StubBackbone(nn.Module):
    """Returns a fixed token sequence, so ``embed`` is deterministic without weights."""

    def __init__(self, tokens):
        super().__init__()
        self.register_buffer("tokens", tokens)

    def forward(self, pixel_values=None):
        # WaivEncoder's HF branch reads ``.last_hidden_state``.
        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = self.tokens
        return out


def _make_encoder(pooling="clsmean", seed=0):
    """A WaivEncoder with every attribute ``_pool``/``tokens``/``embed`` touch, no download.

    ``WaivEncoder.__init__`` builds a real backbone off the hub, which no unit test can
    do. The pooling and projection code under test reads only the attributes set here.
    """
    enc = WaivEncoder.__new__(WaivEncoder)
    nn.Module.__init__(enc)
    enc.cfg = EncoderConfig(backbone=FIT_BACKBONE, pooling=pooling, use_lora=False)
    enc.is_timm = False
    enc.hidden_size = HIDDEN
    enc.num_blocks = 2
    enc.patch_size = 16
    enc.config_image_size = 224
    enc.model_type = "stub"
    enc.num_prefix_tokens = 1
    enc.embed_dim = HIDDEN * (2 if pooling == "clsmean" else 1)
    enc.norm_mean, enc.norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    enc.split_heads = ()
    enc.pool_head = None
    enc.pool_head_name = "mean"
    enc.infer_pool_head = False
    g = torch.Generator().manual_seed(seed)
    enc.backbone = _StubBackbone(torch.randn(BATCH, NUM_TOKENS, HIDDEN, generator=g))
    enc.load_svd_nuisance(None, 0)
    return enc


def _images():
    # Float NCHW, so ``normalize_uint8`` is skipped; the stub ignores the pixels anyway.
    return torch.zeros(BATCH, 3, 224, 224)


def _write_fit(tmp_path, D, k_available=None, backbone=FIT_BACKBONE, pooling="clsmean",
               seed=0, name="fit.npz"):
    """A synthetic fit npz with the same keys ``fit_svd_nuisance.py`` writes."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    V = np.ascontiguousarray(q.T.astype(np.float32))
    path = tmp_path / name
    np.savez(
        path,
        V=V,
        eigvals=np.arange(D, 0, -1).astype(np.float32),
        backbone=np.array(backbone),
        pooling=np.array(pooling),
        n_tiles=np.array(100),
        seed=np.array(0),
        D=np.array(D),
    )
    return path, V


# ----- (a) the arithmetic ------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 8])
def test_encoder_matches_reference_formula(tmp_path, k):
    """embed() under rank-k equals numpy's f - (f @ Vk.T) @ Vk on the unprojected f."""
    enc = _make_encoder()
    base = enc.embed(_images()).detach().numpy()

    fit, V = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), k)
    got = enc.embed(_images()).detach().numpy()

    Vk = V[:k]
    want = base - (base @ Vk.T) @ Vk
    assert np.allclose(got, want, atol=1e-5)


def test_k_zero_is_a_bit_exact_noop(tmp_path):
    """The control arm must reproduce the unprojected number to the last bit."""
    enc = _make_encoder()
    base = enc.embed(_images())

    fit, _ = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 0)
    assert enc.svd_basis is None
    assert torch.equal(enc.embed(_images()), base)


def test_output_is_orthogonal_to_every_basis_row(tmp_path):
    """Nothing of the removed subspace survives in the exported embedding."""
    enc = _make_encoder()
    fit, V = _write_fit(tmp_path, enc.embed_dim)
    k = 5
    enc.load_svd_nuisance(str(fit), k)

    out = enc.embed(_images()).detach().numpy()
    assert np.abs(out @ V[:k].T).max() < 1e-5


def test_projection_survives_autocast_dtype(tmp_path):
    """A half-precision pooled vector still gets a float32 subtraction, cast back."""
    enc = _make_encoder()
    fit, V = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 4)

    half = enc._pool(enc.backbone.tokens.half())
    assert half.dtype == torch.float16
    assert np.abs(half.float().numpy() @ V[:4].T).max() < 1e-2


# ----- (b) applied exactly once ------------------------------------------------------


def test_applied_exactly_once_on_the_embed_path(tmp_path, monkeypatch):
    """One call to _project_nuisance per embed(), not zero and not two."""
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 4)

    calls = []
    real = enc._project_nuisance

    def counting(emb):
        calls.append(emb.shape)
        return real(emb)

    monkeypatch.setattr(enc, "_project_nuisance", counting)
    enc.embed(_images())
    assert len(calls) == 1
    assert calls[0][-1] == enc.embed_dim


def test_parts_path_agrees_with_embed_and_is_not_double_applied(tmp_path):
    """pool_from_parts(embed_parts()) == embed(), with the parts themselves UNPROJECTED.

    Projection is idempotent, so applying it twice cannot be caught by comparing values.
    What CAN go wrong is applying it at the wrong granularity: projecting the cls and
    mean halves separately and concatenating is not the same operation as projecting the
    2*hidden concat, and it is what a basis pushed down into _pool_parts would do.
    """
    enc = _make_encoder()
    fit, V = _write_fit(tmp_path, enc.embed_dim)
    k = 4
    enc.load_svd_nuisance(str(fit), k)

    parts = enc.embed_parts(_images())
    pooled = enc.pool_from_parts(parts)
    emb = enc.embed(_images())
    assert torch.allclose(pooled, emb, atol=1e-6)
    assert np.abs(pooled.detach().numpy() @ V[:k].T).max() < 1e-5

    # The halves are the raw pools: they carry the full 2*hidden basis' worth of nothing.
    raw_cls = enc.backbone.tokens[:, 0, :]
    raw_mean = enc.backbone.tokens[:, enc.num_prefix_tokens:, :].mean(dim=1)
    assert torch.allclose(parts["cls"], raw_cls, atol=1e-6)
    assert torch.allclose(parts["mean"], raw_mean, atol=1e-6)


def test_cls_pooling_projects_the_cls_vector(tmp_path):
    """The cls arm is a different width, so it needs its own fit -- and gets projected."""
    enc = _make_encoder(pooling="cls")
    assert enc.embed_dim == HIDDEN
    fit, V = _write_fit(tmp_path, HIDDEN, pooling="cls")
    enc.load_svd_nuisance(str(fit), 2)

    out = enc.embed(_images()).detach().numpy()
    assert np.abs(out @ V[:2].T).max() < 1e-5


def test_reload_replaces_rather_than_composes(tmp_path):
    """Calling load_svd_nuisance twice leaves one rank-k subtraction, not two stacked."""
    enc = _make_encoder()
    fit, V = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 6)
    enc.load_svd_nuisance(str(fit), 2)

    assert enc.svd_basis.shape[0] == 2
    out = enc.embed(_images()).detach().numpy()
    # Components 2..5 were removed by the first load and must be back.
    assert np.abs(out @ V[2:6].T).max() > 1e-3


# ----- (c) refusals ------------------------------------------------------------------


def test_dimension_mismatch_is_a_hard_error(tmp_path):
    """A basis fitted at another width is a different intervention, not a weaker one."""
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim * 2)
    with pytest.raises(ValueError, match="but this encoder exports"):
        enc.load_svd_nuisance(str(fit), 4)


def test_k_above_available_components_is_a_hard_error(tmp_path):
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim)
    with pytest.raises(ValueError, match="exceeds"):
        enc.load_svd_nuisance(str(fit), enc.embed_dim + 1)


def test_k_without_a_fit_is_a_hard_error():
    enc = _make_encoder()
    with pytest.raises(ValueError, match="no svd_fit"):
        enc.load_svd_nuisance(None, 4)


def test_missing_fit_file_is_a_hard_error(tmp_path):
    enc = _make_encoder()
    with pytest.raises(FileNotFoundError):
        enc.load_svd_nuisance(str(tmp_path / "nope.npz"), 4)


def test_backbone_mismatch_warns_and_continues(tmp_path, capsys):
    """Cross-backbone transfer is a legitimate experiment at equal width -- say so, run it."""
    enc = _make_encoder()
    fit, V = _write_fit(tmp_path, enc.embed_dim, backbone="kaiko-ai/midnight")
    enc.load_svd_nuisance(str(fit), 3)

    assert "WARNING" in capsys.readouterr().out
    out = enc.embed(_images()).detach().numpy()
    assert np.abs(out @ V[:3].T).max() < 1e-5


def test_pooling_mismatch_warns_and_continues(tmp_path, capsys):
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim, pooling="mean")
    enc.load_svd_nuisance(str(fit), 3)
    assert "WARNING" in capsys.readouterr().out


def test_config_carries_the_projection(tmp_path):
    """cfg is the record of what was applied -- collect_* scripts read it back."""
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 7)
    assert enc.cfg.svd_k == 7 and enc.cfg.svd_fit == str(fit)


def test_basis_is_not_persistent(tmp_path):
    """A post-hoc intervention must not leak into checkpoints written afterwards."""
    enc = _make_encoder()
    fit, _ = _write_fit(tmp_path, enc.embed_dim)
    enc.load_svd_nuisance(str(fit), 4)
    assert "svd_basis" not in enc.state_dict()
