"""Base-vs-finetuned embedding-delta diagnostic (shared-shift fraction).

Mirrors the phikon-v2 measurement on an arbitrary backbone/checkpoint.

Base embeddings are produced by the SAME module with the LoRA adapter DISABLED
(peft `disable_adapter()`), which is bit-exact to the frozen base (contrastive.py
uses the same idiom for its retention teacher).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/admin/home/ryan.kim/waiv/src")
from waivphaet.models.encoder import EncoderConfig, WaivEncoder  # noqa: E402

REPACK = Path("/data/plism/repacked")


def load_tiles(fname: str, idx: np.ndarray) -> torch.Tensor:
    arr = np.load(REPACK / fname, mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(arr[idx]))


@torch.no_grad()
def embed_all(model, tiles: torch.Tensor, bs: int, adapter: bool) -> np.ndarray:
    model.eval()
    outs = []
    import contextlib
    ctx = contextlib.nullcontext() if adapter else model.backbone.disable_adapter()
    with ctx:
        for i in range(0, tiles.shape[0], bs):
            outs.append(model.embed(tiles[i : i + bs]).float().cpu().numpy())
            print(f"    {'ft ' if adapter else 'base'} {i + bs}/{tiles.shape[0]}", flush=True)
    return np.concatenate(outs, 0)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xty / (xx * yy))


def offdiag_cos(A: np.ndarray) -> float:
    """Mean cosine between DIFFERENT tiles in one space (calibration baseline)."""
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    S = a @ a.T
    n = S.shape[0]
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def rowcos(A: np.ndarray, B: np.ndarray) -> float:
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float((a * b).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ref", default="GMH_S60_to_GMH_S60.npy")
    ap.add_argument("--other", default="GIVH_AT2_to_GMH_S60.npy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rc = json.loads(Path(a.run_config).read_text())
    enc = rc["encoder"]
    acfg = json.loads(Path(a.ckpt, "adapter", "adapter_config.json").read_text())
    # timm-loaded backbones (Virchow2) save no base_model_name_or_path -- peft only
    # records it for HF `from_pretrained` models. Check it only when present.
    _saved_base = acfg.get("base_model_name_or_path")
    if _saved_base:
        assert _saved_base == enc["backbone"], (_saved_base, enc["backbone"])

    cfg = EncoderConfig(
        backbone=enc["backbone"],
        pooling=enc.get("pooling", "clsmean"),
        use_lora=True,
        lora_rank=int(acfg["r"]),
        lora_alpha=int(acfg["lora_alpha"]),
        proj_out_dim=int(enc.get("proj_out_dim", 512)),
        # embed() is the eval-time export: unaffected by split heads / pool head
        # (infer_pool_head is False in every final5 config).
        split_heads=(),
        pool_head="mean",
        infer_pool_head=False,
        dtype="float32",
    )
    assert not enc.get("infer_pool_head", False), "infer_pool_head on -> embed() differs"
    t0 = time.time()
    model = WaivEncoder(cfg)
    print(json.dumps(model.trainable_parameter_summary(), indent=1), flush=True)

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(Path(a.ckpt, "adapter", "adapter_model.safetensors")))
    res = set_peft_model_state_dict(model.backbone, sd)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    assert not unexpected, unexpected[:5]
    print(f"[load] {len(sd)} adapter tensors, {time.time()-t0:.1f}s", flush=True)

    N_TILES = 16278
    idx = np.linspace(0, N_TILES - 1, a.n).astype(np.int64)
    ref = load_tiles(a.ref, idx)
    oth = load_tiles(a.other, idx)

    # sanity: adapter really changes the output
    with torch.no_grad():
        p_ft = model.embed(ref[:2]).float().numpy()
        with model.backbone.disable_adapter():
            p_b = model.embed(ref[:2]).float().numpy()
    print(f"[sanity] max|ft-base| on 2 tiles = {np.abs(p_ft-p_b).max():.6f}", flush=True)

    Zb = embed_all(model, ref, a.bs, adapter=False)
    Zf = embed_all(model, ref, a.bs, adapter=True)
    Ob = embed_all(model, oth, a.bs, adapter=False)
    Of = embed_all(model, oth, a.bs, adapter=True)

    D = Zf - Zb
    mu = D.mean(0)
    fro2 = float((D**2).sum())
    shared = float(D.shape[0] * (mu**2).sum())

    out = {
        "backbone": enc["backbone"],
        "ckpt": a.ckpt,
        "n_tiles": int(a.n),
        "embed_dim": int(Zb.shape[1]),
        "cosine_base_vs_ft": rowcos(Zb, Zf),
        "rel_l2_change": float(
            (np.linalg.norm(D, axis=1) / np.linalg.norm(Zb, axis=1)).mean()
        ),
        "rel_l2_change_global_fro": float(np.sqrt(fro2) / np.linalg.norm(Zb)),
        "inter_tile_cos_base": offdiag_cos(Zb),
        "inter_tile_cos_ft": offdiag_cos(Zf),
        "cka_base_vs_ft": linear_cka(Zb, Zf),
        "frac_change_shared_shift": shared / fro2,
        "mean_delta_norm": float(np.linalg.norm(mu)),
        "mean_base_norm": float(np.linalg.norm(Zb, axis=1).mean()),
        "mean_delta_row_norm": float(np.linalg.norm(D, axis=1).mean()),
        "cross_scanner_cos_base": rowcos(Zb, Ob),
        "cross_scanner_cos_ft": rowcos(Zf, Of),
        "ref_condition": a.ref,
        "other_condition": a.other,
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(out, indent=2), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    np.savez_compressed(a.out.replace(".json", "_emb.npz"), Zb=Zb, Zf=Zf, Ob=Ob, Of=Of, idx=idx)


if __name__ == "__main__":
    main()
