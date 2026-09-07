#!/usr/bin/env python
"""The one figure a line chart cannot make: WHICH PATCH gets retrieved, base vs tuned.

The RI/HEST/THUNDER curves say cross-condition retrieval improves on average. This
figure shows the mechanism on individual tiles: for a query tile under one acquisition
condition, does its nearest neighbour under a *different, held-out* condition land on
the same registered tissue location, or on a merely similar-looking one?

    query tile x_{c_a,i}  -- one held-out condition, one registered location i
    candidate pool        -- all 256 probe locations, SAME held-out condition c_b
    retrieval             -- cosine on the embedding view (CLS+mean concat, L2-normalised,
                             NO projection head -- the head is training-only, see
                             embed_probe.py's pair_stats / docs/... "GeM pool head is
                             vacuous at inference")

Base retrieves visually-similar-but-wrong locations; fine-tuning (LoRA step 200 of
genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165) is supposed to fix that. This
script renders the query, base top-5, and tuned top-5, for 3 query locations chosen by
a DETERMINISTIC rule (see ``pick_examples``) -- never hand-picked to exaggerate.

Condition pair
---------------
embed_probe.py's own pair-generation order (nested loop over held-out conditions,
``all_conditions()`` order: stain-major, scanner-minor) produces 171 cross-stain
held-out pairs for the default split -- exactly the ``n_pairs=171`` in every
``probe_step_*.json['groups']['heldout']['cross_stain.embedding']``. Its literal first
pair is (GIV_GT450, GIVH_GT450), but GT450 is ITSELF a held-out scanner, so that pair
would confound the cross-stain story with an unseen-scanner story. This script instead
takes the first pair in that same order whose shared scanner is NOT held out (15 of
171 qualify) -- isolating the stain axis, the axis embed_probe reports separately from
cross-scanner. That pair is (HRH_AT2, KR_AT2): both conditions use scanner AT2, which
is in the training split; HRH and KR are both held-out stains.

Numbers reproduced from this exact run: base cross_stain.embedding heldout top1 =
0.6958 (finalgem-phikon-384585/probe_before.json, model-agnostic given seed=1234,
n_tiles=256 -- see plism_base_probe_4bb.sbatch's header note), tuned step-200 top1 =
0.8960 (probe_step_0000200.json) -- this is the "0.696 -> 0.897" the figure is meant to
make legible tile-by-tile.

    ./.venv-hest/bin/python scripts/retrieval_qualitative.py
      -> waiv-asci/figures/retrieval_examples.{pdf,png}
      -> caches embeddings at --cache (default: alongside the run) so a re-render
         after a style tweak does not need the GPU again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import PLISM_PACKED, RUNS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

RUN = RUNS / "genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165"
STEP = 200
BACKBONE = "owkin/phikon-v2"
PACKED_DIR = PLISM_PACKED
OUT = REPO.parent / "waiv-asci" / "figures"
HELDOUT_SCANNERS = ["GT450", "S210"]
HELDOUT_STAINS = ["HRH", "KR", "MY"]
N_TILES = 256
SEED = 1234
N_EXAMPLES = 3
TOPK = 5

INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"
GOOD = "#16a34a"   # correct registered match
BAD = "#dc2626"    # this cell is what got retrieved instead, and it's wrong


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--packed-dir", type=Path, default=PACKED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--cache", type=Path, default=None,
                     help="npz cache of the 4 embedding tensors (base/tuned x cond a/b); "
                          "default <run>/retrieval_qualitative_cache.npz")
    ap.add_argument("--n-tiles", type=int, default=N_TILES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def pick_condition_pair(packed_dir: Path):
    """First cross-stain held-out pair, embed_probe's own order, on a non-held-out scanner.

    Returns (cond_a, cond_b, n_total_cross_stain, n_clean). See module docstring.
    """
    from spectra.data.conditions import make_split, available_conditions
    from spectra.data.repack import present_filenames

    split = make_split(HELDOUT_SCANNERS, HELDOUT_STAINS)
    present = present_filenames(packed_dir)
    conds = available_conditions(split.heldout, present)
    pairs = [
        (a, b) for i, a in enumerate(conds) for b in conds[i + 1:]
        if a.scanner == b.scanner and a.stain != b.stain
    ]
    clean = [(a, b) for a, b in pairs if a.scanner not in split.heldout_scanners]
    if not clean:
        raise SystemExit("no cross-stain heldout pair on a non-held-out scanner")
    return clean[0][0], clean[0][1], len(pairs), len(clean)


def embed_both(args, cond_a, cond_b, tiles):
    """-> dict with base_a, base_b, tuned_a, tuned_b, each (N, D) L2-normalised embedding."""
    if args.cache and args.cache.exists():
        z = np.load(args.cache)
        return {k: z[k] for k in ("base_a", "base_b", "tuned_a", "tuned_b")}

    import torch
    import embed_probe as ep
    from spectra.data.repack import open_slide
    from spectra.models.encoder import build_encoder

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("[retrieval_qualitative] WARNING: no CUDA device -- this will be slow", flush=True)

    slide_a = open_slide(args.packed_dir, cond_a.slide_id.replace(".tif", ""))
    slide_b = open_slide(args.packed_dir, cond_b.slide_id.replace(".tif", ""))

    torch.manual_seed(args.seed)
    base = build_encoder(backbone=args.backbone, lora_rank=32, lora_alpha=64,
                          proj_out_dim=512, pooling="clsmean").to(device).eval()
    torch.manual_seed(args.seed)
    tuned = build_encoder(backbone=args.backbone, lora_rank=32, lora_alpha=64,
                           proj_out_dim=512, pooling="clsmean").to(device).eval()
    adapter = args.run / f"step_{args.step:07d}"
    ep.load_adapter(tuned, adapter)
    print(f"[retrieval_qualitative] loaded adapter from {adapter}", flush=True)

    out = {}
    for tag, model in (("base", base), ("tuned", tuned)):
        for suf, slide in (("a", slide_a), ("b", slide_b)):
            e, _p = ep.embed_condition(model, slide, tiles, device, args.batch_size)
            out[f"{tag}_{suf}"] = e.numpy()
        print(f"[retrieval_qualitative] embedded {tag}", flush=True)

    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **out)
    return out


def retrieval_ranks(emb_a: np.ndarray, emb_b: np.ndarray, topk: int):
    """-> (topk_idx (N,topk), rank_of_match (N,)) for query rows of emb_a against pool emb_b."""
    sim = emb_a @ emb_b.T
    order = np.argsort(-sim, axis=1)
    n = sim.shape[0]
    rank_of_match = np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)])  # 0-indexed
    return order[:, :topk], rank_of_match


def pick_examples(rank_base: np.ndarray, rank_tuned: np.ndarray, n_examples: int):
    """First ``n_examples`` locations (ascending pool index) where base's top-1 is wrong
    (rank_base != 0) and tuned's top-1 is correct (rank_tuned == 0). Also returns the
    total count meeting that rule and the reverse count (base right, tuned wrong), so
    the report states how selective the rule is out of the 256-location pool.
    """
    n = rank_base.shape[0]
    fixed = [i for i in range(n) if rank_base[i] != 0 and rank_tuned[i] == 0]
    broken = [i for i in range(n) if rank_base[i] == 0 and rank_tuned[i] != 0]
    return fixed[:n_examples], len(fixed), len(broken)


def render(args, cond_a, cond_b, tiles, examples, rank_base, rank_tuned,
           topk_base, topk_tuned):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spectra.data.repack import open_slide

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                         "axes.edgecolor": INK, "pdf.fonttype": 42})

    slide_a = open_slide(args.packed_dir, cond_a.slide_id.replace(".tif", ""))
    slide_b = open_slide(args.packed_dir, cond_b.slide_id.replace(".tif", ""))

    n_ex = len(examples)
    k = args.topk
    ncols = k + 1
    fig, axes = plt.subplots(2 * n_ex, ncols, figsize=(1.15 * ncols, 1.15 * 2 * n_ex + 0.4 * n_ex),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.10})
    if n_ex == 1:
        axes = axes.reshape(2, ncols)

    def style(ax, color=None, lw=1.0):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(color is not None)
            if color is not None:
                sp.set_edgecolor(color); sp.set_linewidth(lw)

    for row_block, i in enumerate(examples):
        r0, r1 = 2 * row_block, 2 * row_block + 1
        q_img = np.asarray(slide_a[int(tiles[i])])
        for r in (r0, r1):
            axes[r, 0].imshow(q_img)
            style(axes[r, 0], ACCENT, 2.2)
        axes[r0, 0].set_ylabel("base", fontsize=7.5, rotation=0, ha="right", va="center", labelpad=6, color=INK)
        axes[r1, 0].set_ylabel("tuned", fontsize=7.5, rotation=0, ha="right", va="center", labelpad=6, color=INK)
        if row_block == 0:
            axes[r0, 0].set_title("query\n($c_a$)", fontsize=7.5)

        for tag, r, topk_idx, rk in ((0, r0, topk_base, rank_base), (1, r1, topk_tuned, rank_tuned)):
            for c in range(k):
                ax = axes[r, c + 1]
                j = int(topk_idx[i, c])
                ax.imshow(np.asarray(slide_b[int(tiles[j])]))
                is_match = (j == i)
                style(ax, GOOD if is_match else MUTED, 2.2 if is_match else 0.8)
                if row_block == 0 and tag == 0:
                    ax.set_title(f"rank {c + 1}", fontsize=7.5)
                if is_match:
                    ax.text(0.5, -0.10, "match", transform=ax.transAxes, ha="center", va="top",
                            fontsize=6.5, color=GOOD, weight="bold")
            if rk[i] >= k:  # true match never appears in the top-k row
                axes[r, k].text(1.06, 0.5, f"true match\nis rank {rk[i] + 1}", transform=axes[r, k].transAxes,
                                ha="left", va="center", fontsize=6.2, color=BAD)

        fig.text(0.015, 1 - (row_block + 0.5) / n_ex, f"loc. {int(tiles[i])}", rotation=90,
                 ha="left", va="center", fontsize=7, color=MUTED)

    fig.suptitle(
        f"query condition $c_a$={cond_a.key}   candidate pool $c_b$={cond_b.key}"
        f"  (256 held-out registered locations, cosine on CLS+mean, no head)",
        fontsize=8, y=1.0 + 0.012 * n_ex,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "retrieval_examples.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "retrieval_examples.png", dpi=200, bbox_inches="tight")


def main() -> int:
    args = parse_args()
    if args.cache is None:
        args.cache = args.run / "retrieval_qualitative_cache.npz"

    from spectra.data.conditions import NUM_TILES

    cond_a, cond_b, n_pairs_total, n_pairs_clean = pick_condition_pair(args.packed_dir)
    rng = np.random.default_rng(args.seed)
    tiles = np.sort(rng.choice(NUM_TILES, size=args.n_tiles, replace=False))

    emb = embed_both(args, cond_a, cond_b, tiles)
    topk_base, rank_base = retrieval_ranks(emb["base_a"], emb["base_b"], args.topk)
    topk_tuned, rank_tuned = retrieval_ranks(emb["tuned_a"], emb["tuned_b"], args.topk)

    examples, n_fixed, n_broken = pick_examples(rank_base, rank_tuned, args.n_examples)
    if len(examples) < args.n_examples:
        print(f"[retrieval_qualitative] WARNING: only {len(examples)} locations meet the "
              f"selection rule (< requested {args.n_examples})", flush=True)

    render(args, cond_a, cond_b, tiles, examples, rank_base, rank_tuned, topk_base, topk_tuned)

    report = {
        "cond_a": cond_a.key, "cond_b": cond_b.key,
        "n_cross_stain_pairs_total": n_pairs_total,
        "n_cross_stain_pairs_clean_scanner": n_pairs_clean,
        "n_pool_locations": int(args.n_tiles),
        "base_top1_acc": float(np.mean(rank_base == 0)),
        "tuned_top1_acc": float(np.mean(rank_tuned == 0)),
        "n_base_wrong_tuned_right": n_fixed,
        "n_base_right_tuned_wrong": n_broken,
        "examples_plotted": [{"pool_index": int(i), "tile_id": int(tiles[i]),
                              "rank_base": int(rank_base[i]) + 1, "rank_tuned": int(rank_tuned[i]) + 1}
                             for i in examples],
    }
    print(json.dumps(report, indent=2))
    (args.out_dir / "retrieval_examples_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
