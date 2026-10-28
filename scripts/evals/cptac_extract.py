"""CPTAC GPU stage: stream pre-extracted 224px tiles -> per-slide fp64 mean embeddings.

    python scripts/evals/cptac_extract.py --key K --shard I --shards 8 [--precision fp16]
        [--batch 1024] [--workers 16] [--pooling cls|clsmean|mean]

Reads <evals>/<K>/spec.json, writes <evals>/<K>/cptac_work/meanpool_embeddings/<slide>.h5
(dataset ``features`` [1, D] fp32 = unweighted mean of patch features summed in fp64) and
<evals>/<K>/cptac_work/extract_<I>_of_<N>.done (+ .json with timing). Slides are flushed
as soon as all their tiles are embedded, so a preempted/requeued shard resumes per slide.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptac_common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pooling", default=None)
    ap.add_argument("--limit-slides", type=int, default=0, help="smoke: only the first N slides of the shard")
    ap.add_argument("--work", default=None, help="override work dir (smoke)")
    a = ap.parse_args(argv)
    assert 0 <= a.shard < a.shards

    import h5py
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info

    from spectra.evals.model import load_eval_model
    from spectra.evals.results import out_dir
    from spectra.evals.spec import load_spec

    spec = load_spec(out_dir(a.key) / "spec.json")
    assert spec["model_key"] == a.key
    work = Path(a.work) if a.work else C.work_dir(a.key)
    emb_dir = work / "meanpool_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    counts = C.patch_counts()
    groups = C.shard_slides(counts, a.shards)
    slide_ids = groups[a.shard]
    if a.limit_slides:
        slide_ids = sorted(slide_ids, key=lambda s: counts[s])[: a.limit_slides]
    totals = [sum(counts[s] for s in g) for g in groups]
    print(f"[cptac-extract] {a.key} shard {a.shard}/{a.shards}: {len(slide_ids)} slides, "
          f"{sum(counts[s] for s in slide_ids):,} patches; shard totals {totals}", flush=True)

    t_load = time.time()
    model = load_eval_model(spec, device="cuda", precision=a.precision)
    pooling = a.pooling or C.CPTAC_READOUT[model.backbone_repo]
    dim = model.feat_dim(pooling)
    transform = model.transform(resize=False)
    print(f"[cptac-extract] loaded {model.name} kind={model.kind} pooling={pooling} dim={dim} "
          f"precision={a.precision} in {time.time() - t_load:.0f}s", flush=True)

    def done_ok(sid):
        p = emb_dir / f"{sid}.h5"
        if not p.exists():
            return False
        with h5py.File(p) as h:
            f = h["features"][:]
        assert f.shape == (1, dim), (sid, f.shape, dim)
        assert torch.isfinite(torch.from_numpy(f)).all(), sid
        return True

    pending = [s for s in slide_ids if not done_ok(s)]
    print(f"[cptac-extract] {len(slide_ids) - len(pending)} slides already done, {len(pending)} pending", flush=True)

    class PatchStream(IterableDataset):
        def __init__(self, ids):
            self.ids = ids

        def __iter__(self):
            info = get_worker_info()
            wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
            for i in range(wid, len(self.ids), nw):
                d = os.path.join(C.PATCH_DIR, self.ids[i])
                for name in sorted(os.listdir(d)):
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        with Image.open(os.path.join(d, name)) as handle:
                            image = handle.convert("RGB")
                        assert image.size == (224, 224)
                        yield transform(image), i

    t0 = time.time()
    n_done_patches = 0
    if pending:
        expected = torch.tensor([counts[s] for s in pending])
        n_pending = int(expected.sum())
        loader = DataLoader(PatchStream(pending), batch_size=a.batch, num_workers=a.workers,
                            pin_memory=True, prefetch_factor=2 if a.workers else None,
                            persistent_workers=False)
        sums = torch.zeros(len(pending), dim, dtype=torch.float64, device="cuda")
        cnt = torch.zeros(len(pending), dtype=torch.long, device="cuda")
        written = torch.zeros(len(pending), dtype=torch.bool)

        def flush(final=False):
            c = cnt.cpu()
            ready = ((c == expected) & ~written).nonzero().flatten().tolist()
            if ready:
                vecs = (sums[ready] / cnt[ready].unsqueeze(1).double()).float().cpu()
                assert torch.isfinite(vecs).all()
                for j, i in enumerate(ready):
                    out = emb_dir / f"{pending[i]}.h5"
                    tmp = out.with_suffix(f".h5.{os.getpid()}.tmp")
                    with h5py.File(tmp, "w") as h:
                        h.create_dataset("features", data=vecs[j].numpy()[None])
                    os.replace(tmp, out)
                    written[i] = True
            if final:
                assert torch.equal(c, expected), "patch count mismatch"
                assert bool(written.all())

        with torch.inference_mode():
            for bi, (images, idx) in enumerate(loader):
                idx = idx.cuda(non_blocking=True)
                feats = model.features(images.cuda(non_blocking=True), pooling)
                sums.index_add_(0, idx, feats.double())
                cnt.index_add_(0, idx, torch.ones_like(idx))
                n_done_patches += len(idx)
                if bi % 100 == 0:
                    flush()
                    el = time.time() - t0
                    print(f"[shard {a.shard}] {n_done_patches:,}/{n_pending:,} ({n_done_patches / n_pending:.1%}) "
                          f"{n_done_patches / max(el, 1e-9):.0f} img/s, {int(written.sum())}/{len(pending)} slides",
                          flush=True)
        flush(final=True)
    assert all(done_ok(s) for s in slide_ids)
    el = time.time() - t0
    stats = {"shard": a.shard, "shards": a.shards, "slides": len(slide_ids), "pending_slides": len(pending),
             "patches_this_run": n_done_patches, "extract_seconds": el,
             "img_per_s": n_done_patches / el if el > 0 else None, "host": socket.gethostname(),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
             "pooling": pooling, "precision": a.precision, "batch": a.batch, "dim": dim}
    if not a.limit_slides:
        (work / f"extract_{a.shard}_of_{a.shards}.json").write_text(json.dumps(stats, indent=2) + "\n")
        (work / f"extract_{a.shard}_of_{a.shards}.done").touch()
    print(f"[cptac-extract] DONE {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
