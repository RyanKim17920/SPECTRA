#!/usr/bin/env python
"""The mechanism figure: acquisition structure REPLACED BY tissue structure.

Waiv/PathoROB's own argument for why cross-condition contrastive fine-tuning works is a
representational claim, not just a retrieval-accuracy number: in the BASE embedding
space, nearest neighbours are dominated by acquisition condition (scanner/stain) --
scanner and stain leave a stronger fingerprint than tissue identity does. Fine-tuning is
supposed to erase that fingerprint and leave tissue identity as the dominant axis. The
RI/HEST/THUNDER curves show this indirectly (average retrieval goes up); this figure
shows it directly, on the same points, coloured two ways.

Layout (2 rows x 2 cols, per backbone):

    row 1: points coloured by ACQUISITION CONDITION (scanner_stain key)
    row 2: SAME points coloured by PSEUDO-TISSUE group
    col 1: BASE model (zero-init LoRA, i.e. literally the frozen backbone)
    col 2: FINE-TUNED model (LoRA step 200 of the same genMASK-c50 recipe used
           throughout the paper: c50-lr1e-4-kl0-ms500-<backbone>-s0-t900)

Expected story: base col shows tight, separated clusters in row 1 (acquisition
dominates) and a comparatively mixed row 2; tuned col collapses row 1 into one blob
(acquisition no longer linearly separable) while row 2 becomes structured.

Run across 3 backbones with the largest published retrieval gains: phikon-v2, midnight
(kaiko-ai/midnight, i.e. the "midnight" arm -- NOT the locally-bound OpenMidnight
teacher checkpoint, which was never fine-tuned in this genMASK-c50 family), Virchow2.

PSEUDO-TISSUE LABELS -- READ THIS BEFORE TRUSTING THE COLOURS
---------------------------------------------------------------
There are no ground-truth tissue labels for PLISM. The labels used here for row 2 are
INFERRED pseudo-tissue groups from scripts/derive_core_map_v2.py:

  1. TMA cores are found by 8-connectivity on the tile lattice (coordinate-only, no
     model) -- this recovers 49 physically-separate cores exactly.
  2. Cores are pooled into ``n_tissues=46`` pseudo-tissue groups by average-linkage
     agglomerative clustering on cosine distance between per-core MEAN embeddings from
     a *base, untuned* Virchow2 (paige-ai/Virchow2, clsmean pooling) -- chosen because
     the slide's TMA design has 46 real tissue types with ~1.07 replicate cores each,
     so 46 clusters approximately recovers "same tissue, different core".

The saved array is ``runs/.plism_core_labels.npy`` (16278 int32 labels, one per
registered tile location, meta at ``runs/.plism_core_labels_meta.json``: spatial
contiguity 1.0 vs a random-label null of 0.0245, 40.8x lift -- i.e. the clustering
recovers spatially compact groups, consistent with "real histologic regions" but NOT
independently verified against a pathologist's tissue-type call. Report this plainly:
row 2's colouring is INFERRED PSEUDO-TISSUE GROUPS, not ground-truth tissue labels.

Because the label array is indexed by REGISTERED LOCATION (same physical spot under
every acquisition condition), one tissue id applies to a given location under every
condition, which is exactly what row 2 needs: the same point set as row 1, recoloured.

Sampling
--------
10 acquisition conditions, chosen deterministically to cover as many distinct
(scanner, stain) values as possible out of the 91 available (7 scanners, 13 stains) --
see ``pick_conditions``. 400 registered tile locations, sampled once with
``np.random.default_rng(SEED).choice(NUM_TILES, 400, replace=False)`` and reused
identically for every condition and every backbone, so a point's identity (which
location, which condition) is comparable across all panels. Each panel therefore holds
10 x 400 = 4000 points.

Projection
----------
PCA to 50 dims, then UMAP(n_components=2) to 2 dims if umap-learn is importable
(checked at runtime; used here), else scikit-learn TSNE(2) on the PCA-50 features as a
documented fallback. The reducer is fit SEPARATELY for every (backbone, base/tuned)
panel -- i.e. once per COLUMN, shared by both rows in that column, since the embedding
spaces of the base and tuned model are different and not directly comparable; do not
compare absolute coordinates across columns or backbones, only the qualitative
structure (clustered vs mixed).

Quantitative backing
---------------------
Under each panel: silhouette score (cosine metric) of the SAME labelling shown in that
row, computed on the 50-d PCA features (NOT the 2-D UMAP layout, whose distances are not
faithful) -- this is the number the visual "collapses into one blob" / "becomes
structured" claim is graded against, not eyeballing.

    ./.venv-hest/bin/python scripts/embedding_shift.py
      -> waiv-asci/figures/embedding_shift.{pdf,png}
      -> waiv-asci/figures/embedding_shift_report.json
      -> caches embeddings per backbone at <run>/embedding_shift_cache.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CELLS, PLISM_PACKED  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

PACKED_DIR = PLISM_PACKED
OUT = REPO.parent / "waiv-asci" / "figures"
CORE_LABELS_PATH = REPO / "runs" / ".plism_core_labels.npy"

HELDOUT_SCANNERS = ["GT450", "S210"]
HELDOUT_STAINS = ["HRH", "KR", "MY"]

# Same genMASK-c50 recipe family used throughout the paper (lr1e-4, kl0, ms500, seed 0),
# one run per backbone, evaluated at step 200 -- the exact checkpoint
# retrieval_qualitative.py uses for phikon-v2 (0.696 -> 0.897 top-1).
BACKBONES = {
    "phikon2": dict(backbone="owkin/phikon-v2",
                    run=REPO / "runs" / "genMASK-c50-lr1e-4-kl0-ms500-phikon-s0-t900-399165",
                    label="Phikon-v2"),
    "midnight": dict(backbone="kaiko-ai/midnight",
                      run=REPO / "runs" / "genMASK-c50-lr1e-4-kl0-ms500-midnight-s0-t900-399166",
                      label="Midnight-12k"),
    "virchow2": dict(backbone="paige-ai/Virchow2",
                      run=REPO / "runs" / "genMASK-c50-lr1e-4-kl0-ms500-virchow2-s0-t900-399167",
                      label="Virchow2"),
}
# Each backbone's seed-0 checkpoint is its OWN 1-SE selected step, resolved from the same
# cells the paper's tables use: phikon2 200, midnight 150, virchow2 100. A single global
# STEP=200 was showing a checkpoint we do NOT release for midnight and virchow2.
def _selected_step(slug: str) -> int:
    import re as _re
    import pathorob_submetrics as _pm
    _b, seeds = _pm.cells_for(slug)
    s0 = [c for c in seeds if "-s0-" in c] or sorted(seeds)
    txt = (CELLS / s0[0] / "model.py").read_text()
    return int(_re.search(r'^STEP = "(.*)"$', txt, _re.M).group(1).replace("step_", ""))

N_TILES = 400
N_CONDITIONS = 10
SEED = 1234

INK, MUTED, ACCENT = "#1f2937", "#9ca3af", "#2563eb"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed-dir", type=Path, default=PACKED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--core-labels", type=Path, default=CORE_LABELS_PATH)
    ap.add_argument("--step", type=int, default=None,
                    help="override; default = each backbone's own selected step")
    ap.add_argument("--n-tiles", type=int, default=N_TILES)
    ap.add_argument("--n-conditions", type=int, default=N_CONDITIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES.keys()),
                     help="subset of {phikon2, midnight, virchow2} to run")
    ap.add_argument("--force-tsne", action="store_true",
                     help="use TSNE fallback even if umap-learn is importable (debugging)")
    return ap.parse_args()


def pick_conditions(all_conds, n: int):
    """Deterministically greedy-pick <=n conditions to cover as much of BOTH the
    scanner axis and the stain axis as possible (set-cover greedy, not a single OR
    check -- an OR check saturates on whichever axis is smaller (7 scanners) and then
    keeps re-using it, starving stain coverage). At each step, pick the not-yet-chosen
    condition whose (scanner, stain) pair introduces the most NEW axis values (2, 1, or
    0), breaking ties by sorted key for determinism (no RNG). Stops adding for new
    coverage once every remaining condition would add 0; any leftover slots (if
    n exceeds the number of conditions needed to exhaust both axes) are filled in
    sorted-key order.
    """
    conds_sorted = sorted(all_conds, key=lambda c: c.key)
    seen_scan, seen_stain = set(), set()
    chosen, remaining = [], list(conds_sorted)
    while len(chosen) < n and remaining:
        def gain(c):
            return (c.scanner not in seen_scan) + (c.stain not in seen_stain)
        remaining.sort(key=lambda c: (-gain(c), c.key))
        best = remaining[0]
        if gain(best) == 0:
            break
        chosen.append(best)
        seen_scan.add(best.scanner)
        seen_stain.add(best.stain)
        remaining.remove(best)
    for c in conds_sorted:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:n]


def embed_backbone(args, name, cfg, tiles: np.ndarray, conds):
    """-> dict(base=(N,D), tuned=(N,D)) stacked in (condition, tile) order, L2-normalised."""
    cache = cfg["run"] / "embedding_shift_cache.npz"
    if cache.exists():
        z = np.load(cache)
        if list(z["cond_keys"]) == [c.key for c in conds] and int(z["n_tiles"]) == tiles.size:
            print(f"[embedding_shift] {name}: using cache {cache}", flush=True)
            return {"base": z["base"], "tuned": z["tuned"]}
        print(f"[embedding_shift] {name}: cache at {cache} is stale (conds/tiles changed), recomputing", flush=True)

    import torch
    import embed_probe as ep
    from spectra.data.repack import open_slide
    from spectra.models.encoder import build_encoder

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print(f"[embedding_shift] WARNING: no CUDA device -- {name} will be slow", flush=True)

    step = args.step if args.step is not None else _selected_step(name)
    cfg["step"] = step
    adapter = cfg["run"] / f"step_{step:07d}"
    out = {}
    for tag in ("base", "tuned"):
        torch.manual_seed(args.seed)
        model = build_encoder(backbone=cfg["backbone"], lora_rank=32, lora_alpha=64,
                               proj_out_dim=512, pooling="clsmean").to(device).eval()
        if tag == "tuned":
            ep.load_adapter(model, adapter)
            print(f"[embedding_shift] {name}: loaded adapter from {adapter}", flush=True)
        embs = []
        for c in conds:
            slide = open_slide(args.packed_dir, c.slide_id.replace(".tif", ""))
            e, _p = ep.embed_condition(model, slide, tiles, device, args.batch_size)
            embs.append(e.numpy())
        out[tag] = np.concatenate(embs, axis=0)
        print(f"[embedding_shift] {name}: embedded {tag} ({out[tag].shape})", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, base=out["base"], tuned=out["tuned"],
              cond_keys=[c.key for c in conds], n_tiles=tiles.size)
    return out


def project(emb: np.ndarray, seed: int, use_umap: bool):
    """PCA(50) then UMAP(2) if use_umap else TSNE(2) on the PCA-50 features.

    -> (xy (N,2), pca50 (N,50), method_str)
    """
    from sklearn.decomposition import PCA

    n_comp = min(50, emb.shape[0] - 1, emb.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit_transform(emb)

    if use_umap:
        try:
            import umap
            xy = umap.UMAP(n_components=2, random_state=seed).fit_transform(pca)
            return xy, pca, "PCA(50)->UMAP(2)"
        except Exception as e:
            print(f"[embedding_shift] UMAP failed ({e}); falling back to TSNE", flush=True)

    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(pca)
    return xy, pca, "PCA(50)->TSNE(2)"


def safe_silhouette(pca50: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import silhouette_score
    if len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= len(labels):
        return None
    return float(silhouette_score(pca50, labels, metric="cosine"))


def render(args, results, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "stix",
                         "axes.edgecolor": INK, "pdf.fonttype": 42})

    names = list(results.keys())
    nb = len(names)
    fig, axes = plt.subplots(2, 2 * nb, figsize=(1.55 * 2 * nb, 3.5),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.16})
    if nb == 1:
        axes = axes.reshape(2, 2)

    cond_cmap = plt.get_cmap("tab10")
    n_cond = report["n_conditions"]
    cond_colors = [cond_cmap(i % 10) for i in range(n_cond)]
    tissue_cmap = plt.get_cmap("gist_ncar")

    def style(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(MUTED); sp.set_linewidth(0.6)

    for bi, name in enumerate(names):
        r = results[name]
        cfg = BACKBONES[name]
        for ci, tag in enumerate(("base", "tuned")):
            col = 2 * bi + ci
            xy = r[f"{tag}_xy"]
            cond_lab = r["cond_labels"]
            tis_lab = r["tissue_labels"]

            # Points arrive grouped by condition, so a single scatter draws condition 10 on
            # top of conditions 1-9. Where the conditions collapse onto each other that makes
            # the LAST condition's colour cover everything, which reads as a cleaner collapse
            # than the data shows. Draw in a fixed shuffled order instead.
            order = np.random.default_rng(0).permutation(len(xy))
            ax0 = axes[0, col]
            ax0.scatter(xy[order, 0], xy[order, 1],
                        c=[cond_colors[cond_lab[i]] for i in order],
                        s=1.6, linewidths=0, alpha=0.45)
            style(ax0)
            sil_c = r[f"{tag}_sil_cond"]
            ax0.set_xlabel(f"sil={sil_c:.2f}" if sil_c is not None else "sil=n/a",
                            fontsize=6.3, color=INK, labelpad=2)
            ax0.xaxis.set_label_position("bottom")
            if bi == 0 and ci == 0:
                ax0.set_ylabel("by acquisition\ncondition", fontsize=6.8, color=INK)
            ax0.set_title("BASE" if ci == 0 else f"TUNED (step {cfg.get('step', '?')})",
                          fontsize=6.8, color=INK)

            ax1 = axes[1, col]
            ax1.scatter(xy[order, 0], xy[order, 1], c=np.asarray(tis_lab)[order],
                        cmap=tissue_cmap, vmin=0, vmax=45, s=1.6, linewidths=0, alpha=0.45)
            style(ax1)
            sil_t = r[f"{tag}_sil_tissue"]
            ax1.set_xlabel(f"sil={sil_t:.2f}" if sil_t is not None else "sil=n/a",
                            fontsize=6.3, color=INK, labelpad=2)
            if bi == 0 and ci == 0:
                ax1.set_ylabel("by pseudo-tissue\ngroup", fontsize=6.8, color=INK)

        # backbone label spanning its 2 columns
        mid = axes[0, 2 * bi].get_position().x0 * 0.5 + axes[0, 2 * bi + 1].get_position().x1 * 0.5
        fig.text(mid, 0.995, cfg["label"], ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
        if bi > 0:
            x0 = axes[0, 2 * bi].get_position().x0 - 0.006
            fig.add_artist(plt.Line2D([x0, x0], [0.04, 0.9], transform=fig.transFigure,
                                       color=MUTED, linewidth=0.6))

    handles = [Line2D([0], [0], marker="o", linestyle="", color=cond_colors[i], markersize=4)
               for i in range(n_cond)]
    fig.legend(handles, report["conditions"], loc="lower center", ncol=min(n_cond, 10),
               fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.05))

    # Title states only what the silhouettes support. Tissue structure strengthens on all
    # three backbones; acquisition structure is only strongly present to begin with on
    # Phikon-v2 (sil 0.23), so "acquisition is removed" is NOT a claim the other two panels
    # can carry -- their base condition silhouette is already ~0.02-0.04.
    fig.suptitle("Embedding structure before and after fine-tuning", fontsize=11, y=1.10)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "embedding_shift.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "embedding_shift.png", dpi=220, bbox_inches="tight")


def main() -> int:
    args = parse_args()
    if not args.core_labels.exists():
        raise SystemExit(
            f"pseudo-tissue label file not found at {args.core_labels}; STOPPING rather than "
            "inventing tissue labels. Regenerate with scripts/derive_core_map_v2.py."
        )
    core_labels = np.load(args.core_labels)

    from spectra.data.conditions import NUM_TILES, make_split, available_conditions
    from spectra.data.repack import present_filenames

    if core_labels.shape[0] != NUM_TILES:
        raise SystemExit(f"core_labels has {core_labels.shape[0]} entries, expected NUM_TILES={NUM_TILES}")

    split = make_split(HELDOUT_SCANNERS, HELDOUT_STAINS)
    present = present_filenames(args.packed_dir)
    all_conds = available_conditions(split.train, present) + available_conditions(split.heldout, present)
    conds = pick_conditions(all_conds, args.n_conditions)
    print(f"[embedding_shift] {len(conds)} conditions: {[c.key for c in conds]} "
          f"(scanners={sorted({c.scanner for c in conds})}, stains={sorted({c.stain for c in conds})})",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tiles = np.sort(rng.choice(NUM_TILES, size=args.n_tiles, replace=False))
    tissue_labels_1d = core_labels[tiles]  # tissue id per location, condition-independent
    cond_labels_1d = np.repeat(np.arange(len(conds)), tiles.size)
    tissue_labels = np.tile(tissue_labels_1d, len(conds))

    try:
        import umap  # noqa: F401
        use_umap = not args.force_tsne
    except ImportError:
        use_umap = False
        print("[embedding_shift] umap-learn not importable; falling back to TSNE", flush=True)

    results = {}
    report = {
        "conditions": [c.key for c in conds],
        "n_conditions": len(conds),
        "n_tiles": int(args.n_tiles),
        "seed": args.seed,
        "step": args.step,
        "sampling": (f"{len(conds)} acquisition conditions x {args.n_tiles} registered tile "
                     "locations, same locations reused across every condition and backbone "
                     "(np.random.default_rng(seed).choice(NUM_TILES, n_tiles, replace=False))"),
        "pseudo_tissue_source": str(args.core_labels),
        "backbones": {},
    }

    for name in args.backbones:
        cfg = BACKBONES[name]
        emb = embed_backbone(args, name, cfg, tiles, conds)
        r = {"cond_labels": cond_labels_1d, "tissue_labels": tissue_labels}
        method = None
        for tag in ("base", "tuned"):
            xy, pca50, method = project(emb[tag], args.seed, use_umap)
            r[f"{tag}_xy"] = xy
            r[f"{tag}_sil_cond"] = safe_silhouette(pca50, cond_labels_1d)
            r[f"{tag}_sil_tissue"] = safe_silhouette(pca50, tissue_labels)
            print(f"[embedding_shift] {name}/{tag}: sil(condition)={r[f'{tag}_sil_cond']:.4f}  "
                  f"sil(tissue)={r[f'{tag}_sil_tissue']:.4f}", flush=True)
        results[name] = r
        report["projection_method"] = method
        report["backbones"][name] = {
            "run": str(cfg["run"]), "backbone": cfg["backbone"],
            "silhouette_condition": {"base": r["base_sil_cond"], "tuned": r["tuned_sil_cond"]},
            "silhouette_tissue": {"base": r["base_sil_tissue"], "tuned": r["tuned_sil_tissue"]},
        }

    render(args, results, report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "embedding_shift_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
