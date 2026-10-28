"""CPU tests for spectra.evals: key derivation, ckpt resolution, spec identity stability,
protocol tables, result writer, EvalModel pooling. No weights are downloaded (fake checkpoint
trees in tmp_path; the base backbone identity is monkeypatched). Tests that need real backbone
weights run only when SPECTRA_EVAL_TEST_WEIGHTS=1."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from spectra.evals import keys as K
from spectra.evals import protocol as P
from spectra.evals import results as R
from spectra.evals import spec as S

FAKE_BB = {"source": "test", "revision": "rev0", "files": {"config.json": "0" * 64, "pytorch_model.bin": "1" * 64}}


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "backbone_identity", lambda repo: dict(FAKE_BB, files=dict(FAKE_BB["files"], repo=repo)))
    monkeypatch.setattr(S, "SHA_CACHE", tmp_path / "shacache")
    monkeypatch.setenv("SPECTRA_EVAL_OUT", str(tmp_path / "evals"))


def _sweep_run(d: Path, mode="lora", backbone="uni2", steps=(25, 50), ema=False) -> Path:
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"backbone": {"name": backbone}, "adapt": {"mode": mode, "lora_rank": 16}}))
    for s in steps:
        (d / "ckpt" / f"step_{s:07d}.safetensors").write_bytes(f"w{s}".encode())
        if ema:
            (d / "ckpt" / f"ema_step_{s:07d}.safetensors").write_bytes(f"e{s}".encode())
    return d


# ---------------------------------------------------------------------------- keys ----

def test_key_forms():
    assert K.derive_key("bioptimus/H-optimus-0", "base") == "hopt0-base"
    assert K.derive_key("MahmoodLab/UNI2-h", "sweep_lora", "/x/ckpt/step_0000400.safetensors", 400,
                        run_name="s1-9aa01da7") == "uni2-s1-9aa01da7-s400"
    assert K.derive_key("MahmoodLab/UNI2-h", "sweep_lora", "/x", 20, ema=True, run_name="s1-x") == "uni2-s1-x-s20-ema"


def test_key_truncation_deterministic_and_keeps_suffix():
    run = "genMASK-c50-lr1e-4-kl0-ms500-uni2-s0-t900-395390"
    a = K.derive_key("MahmoodLab/UNI2-h", "sweep_lora", "/r/ckpt/step_0000100.safetensors", 100, run_name=run)
    b = K.derive_key("MahmoodLab/UNI2-h", "sweep_lora", "/r/ckpt/step_0000100.safetensors", 100, run_name=run)
    assert a == b and len(a) <= K.MAX_KEY and a.endswith("-s100") and "uni2-genMASK" in a
    c = K.derive_key("MahmoodLab/UNI2-h", "sweep_lora", "/r/ckpt/step_0000100.safetensors", 100, run_name=run + "x")
    assert c != a                                   # hash suffix separates shared prefixes
    for k in (a, c, K.thunder_key(a)):
        assert re.fullmatch(r"[A-Za-z0-9._-]+", k)
    assert len(K.thunder_key(a)) <= 64 and K.thunder_key(a).endswith("_optimized")


def test_validate_key_rejects():
    for bad in ("", "a" * 53, "a/b", "a b", ".."):
        with pytest.raises(ValueError):
            K.validate_key(bad)


def test_aliases():
    for a in ("hoptimus", "hoptimus0", "hopt0", "bioptimus/H-optimus-0"):
        assert K.resolve_repo(a) == "bioptimus/H-optimus-0"
    assert K.resolve_repo("uni2") == "MahmoodLab/UNI2-h"
    assert K.resolve_repo("org/unknown") == "org/unknown"


# ---------------------------------------------------------------------- ckpt kinds ----

def test_kinds(tmp_path):
    assert S.resolve_ckpt(None)["kind"] == "base"
    sw = _sweep_run(tmp_path / "sweeps" / "s1-aaaa", ema=True)
    r = S.resolve_ckpt(str(sw))
    assert r["kind"] == "sweep_lora" and r["step"] == 50 and not r["ema"]      # last non-EMA ckpt
    r = S.resolve_ckpt(str(sw), 25, ema=True)
    assert r["ema"] and r["ckpt_path"].endswith("ema_step_0000025.safetensors")
    assert S.resolve_ckpt(str(sw / "ckpt" / "step_0000025.safetensors"))["step"] == 25
    with pytest.raises(ValueError, match="disagrees"):
        S.resolve_ckpt(str(sw / "ckpt" / "step_0000025.safetensors"), 50)
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt(str(sw), 99)
    full = _sweep_run(tmp_path / "sweeps" / "s1-full", mode="full")
    with pytest.raises(ValueError, match="only LoRA"):
        S.resolve_ckpt(str(full))
    loose = tmp_path / "soup.safetensors"
    loose.write_bytes(b"x")
    with pytest.raises(ValueError):
        S.resolve_ckpt(str(loose))


# ------------------------------------------------------------------- spec identity ----

def test_spec_identity_stable_and_sensitive(tmp_path):
    sw = _sweep_run(tmp_path / "s1-aaaa")
    a = S.resolve_spec("uni2", str(sw), 50)
    b = S.resolve_spec("MahmoodLab/UNI2-h", str(sw / "ckpt" / "step_0000050.safetensors"))
    assert a["identity_sha256"] == b["identity_sha256"] and a["model_key"] == b["model_key"] == "uni2-s1-aaaa-s50"
    json.dumps(a)                                    # JSON-serialisable
    assert S.resolve_spec("uni2")["identity_sha256"] != a["identity_sha256"]
    assert S.resolve_spec("uni2")["model_key"] == "uni2-base"
    (sw / "ckpt" / "step_0000050.safetensors").write_bytes(b"different weights!")
    assert S.resolve_spec("uni2", str(sw), 50)["identity_sha256"] != a["identity_sha256"]


def test_backbone_mismatch_refused(tmp_path):
    sw = _sweep_run(tmp_path / "s1-x", backbone="uni2")
    with pytest.raises(ValueError, match="base model"):
        S.resolve_spec("hoptimus", str(sw))
    assert S.resolve_spec(None, str(sw))["backbone_repo"] == "MahmoodLab/UNI2-h"   # inferred


def test_write_spec_refuses_other_identity(tmp_path):
    sw = _sweep_run(tmp_path / "s1-x")
    a = S.resolve_spec("uni2", str(sw), 50)
    out = tmp_path / "spec.json"
    assert S.write_spec(a, out)[1] == "written"
    assert S.write_spec(S.resolve_spec("uni2", str(sw), 50), out)[1] == "exists"
    other = S.resolve_spec("uni2", str(sw), 50)
    other["identity_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        S.write_spec(other, out)
    assert S.write_spec(other, out, force=True)[1] == "written"


# --------------------------------------------------------------- protocol / results ----

def test_pooling_tables():
    assert P.pooling_for("hest", "bioptimus/H-optimus-0") == "cls"
    assert P.pooling_for("hest", "paige-ai/Virchow2") == "clsmean"
    assert P.pooling_for("thunder", "kaiko-ai/midnight") == "clsmean"
    assert P.pooling_for("hest", "kaiko-ai/midnight") == "cls"
    assert P.pooling_for("pathorob", "bioptimus/H-optimus-0") == "clsmean"
    assert P.pooling_for("thunder", "hoptimus") == "cls"
    assert P.pooling_for("cptac", "kaiko-ai/midnight") == "clsmean"
    assert P.pooling_for("cptac", "MahmoodLab/UNI2-h") == "cls"
    assert P.pooling_for("thunder", "SophontAI/OpenMidnight") == "cls"
    assert P.pooling_for("thunder", "paige-ai/Virchow") == "clsmean"
    finalists = ["bioptimus/H-optimus-0", "MahmoodLab/UNI2-h", "owkin/phikon-v2", "kaiko-ai/midnight",
                 "paige-ai/Virchow2", "paige-ai/Virchow", "SophontAI/OpenMidnight"]
    for b in ("hest", "thunder", "cptac", "pathorob"):
        for r in finalists:
            assert P.pooling_for(b, r) in ("cls", "clsmean")
    with pytest.raises(KeyError):
        P.pooling_for("thunder", "org/unknown")
    assert P.pooling_for("thunder", "org/unknown", override="cls") == "cls"


def test_write_result(tmp_path):
    sw = _sweep_run(tmp_path / "s1-x")
    spec = S.resolve_spec("uni2", str(sw), 50)
    S.write_spec(spec, R.out_dir(spec["model_key"]) / "spec.json")
    p = R.write_result(spec["model_key"], "hest", {"metrics": {"mean_r": 0.4}, "status": "complete"})
    d = json.loads(p.read_text())
    assert d["schema"] == "spectra-evals/1" and d["spec"]["identity_sha256"] == spec["identity_sha256"]
    assert d["metrics"]["mean_r"] == 0.4 and d["bench"] == "hest" and "git_commit" in d
    bad = dict(spec, identity_sha256="0" * 64)
    with pytest.raises(ValueError):
        R.write_result(spec["model_key"], "hest", {}, spec=bad)


def test_eval_model_pooling_cpu():
    """EvalModel pooling on a tiny fake timm-style backbone (no weights, no sweep import)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from spectra.evals.member import Member
    from spectra.evals.model import EvalModel

    class Tiny(nn.Module):
        num_prefix_tokens = 2

        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.patch_size = (16, 16)
            self.proj = nn.Linear(3 * 16 * 16, 8)

        def forward_features(self, x):
            p = x.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2)
            t = self.proj(p)
            return torch.cat([t[:, :2], t], 1)       # 2 prefix tokens + N patch tokens

    info = dict(is_timm=True, hidden=8, num_blocks=1, prefix=2, norm_mean=(0.5,) * 3, norm_std=(0.25,) * 3,
                repo="org/tiny")
    spec = {"model_key": "tiny-base", "backbone_repo": "org/tiny", "kind": "base", "identity_sha256": "0" * 64}
    em = EvalModel(spec, Member(bb=Tiny(), info=info, kind="base")).eval()
    x = torch.randn(2, 3, 32, 32)
    assert em.patch_size == 16 and em.num_patches(32) == 4
    assert em.features(x, "cls").shape == (2, 8) and em.features(x, "clsmean").shape == (2, 16)
    assert em.patch_tokens(x).shape == (2, 4, 8)
    f = em.all_features(x)
    assert torch.allclose(f["clsmean"], em.clsmean_features(x))
    assert not any(p.requires_grad for p in em.parameters())
    with pytest.raises(ValueError):
        em.features(x, "bogus")


@pytest.mark.skipif(os.environ.get("SPECTRA_EVAL_TEST_WEIGHTS") != "1", reason="needs real backbone weights")
def test_load_base_backbone():
    from spectra.evals.model import load_eval_model

    spec = S.resolve_spec("uni2")
    m = load_eval_model(spec, device="cpu", precision="fp32", verify=False)
    import torch

    assert m.features(torch.randn(1, 3, 224, 224), "clsmean").shape == (1, 2 * m.embed_dim)
