#!/usr/bin/env python3
"""Freeze the data behind figures/unused/plism_traj.pdf into paper/data/plism_traj.json.

Data dependency (what the original make_plism_traj.py read, 2026-09-05)
-----------------------------------------------------------------------
* Trajectory: ``$SPECTRA_RUNS/genMASK-b00-ms500-<bb>-s0-t900-*/probe_step_*.json`` for
  bb in phikon, midnight, virchow2, hoptimus, uni2 (resume dirs whose name contains ".r"
  skipped), field ``groups.heldout.{cross_scanner,cross_stain}.embedding.top1``.
* Base (step 0): ``$SPECTRA_RUNS/finalgem-{phikon-384585,midnight-384586,virchow2-384587}/
  probe_before.json`` (same fields) -- only these three backbones have one.

As of 2026-10 all of those directories are gone from $SPECTRA_RUNS (only empty ``.r``
resume dirs remain), so by default each backbone is taken from disk when its files exist
and otherwise from ``--recover-from-pdf <plism_traj.pdf>``: the published figure's per-series
marker offsets are read from the PDF content stream (see _pdfvec.py) and mapped to data
with a per-panel transform calibrated on the gridlines/tick labels.  Every top-1 value is
a count over ``n_pairs * n_tiles`` queries (n_tiles = 256; n_pairs = 73 cross-scanner,
171 cross-stain under the pinned conditions_used.json), so recovered values are snapped to
the nearest k / (n_pairs * 256); the snap residual (in units of one query) is printed and
must be ~0.  Without a PDF and without disk data the existing JSON entry is kept.

    python3 paper/scripts/extract_plism_traj.py [--recover-from-pdf PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import RUNS  # noqa: E402

DATA = REPO / "paper/data/plism_traj.json"
NAME = {"phikon": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus": "H-optimus-0", "uni2": "UNI2-h"}
BASE_RUN = {"phikon": "finalgem-phikon-384585", "midnight": "finalgem-midnight-384586",
            "virchow2": "finalgem-virchow2-384587"}
SEL = {"phikon": 150, "midnight": 100, "virchow2": 150, "hoptimus": 100, "uni2": 150}
N_TILES = 256
N_PAIRS = (73, 171)          # (cross-scanner, cross-stain) held-out pairs
SRC_DISK = "probe_step_*.json / probe_before.json"
SRC_PDF = "recovered from published plism_traj.pdf vector paths"


def heldout(path):
    d = json.load(open(path))["groups"]["heldout"]
    return (d["cross_scanner.embedding"]["top1"], d["cross_stain.embedding"]["top1"])


def disk_series(bb):
    """Verbatim from make_plism_traj.py, with RUNS from _config."""
    out = {}
    for f in glob.glob(f"{RUNS}/genMASK-b00-ms500-{bb}-s0-t900-*/probe_step_*.json"):
        if ".r" in os.path.basename(os.path.dirname(f)):
            continue                                   # skip resumed duplicates
        s = int(re.search(r"probe_step_(\d+)", f).group(1))
        try: out[s] = heldout(f)
        except Exception: pass
    if bb in BASE_RUN:
        try: out[0] = heldout(f"{RUNS}/{BASE_RUN[bb]}/probe_before.json")
        except Exception: pass
    return dict(sorted(out.items()))


def recover_pdf(pdf):
    """-> {bb: {step: (scanner, stain)}}, plus the worst snap residual in query units."""
    import _pdfvec as pv
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10.colors
    pg = pv.read(pdf)
    clips = sorted({it.clip for it in pg.items if isinstance(it, pv.Path_) and it.clip
                    and pv.close(it.stroke, (0.690, 0.690, 0.690))}, key=lambda c: c[0])
    assert len(clips) == 2, clips
    grid = (176 / 255,) * 3                              # matplotlib default grid colour
    frames = {c: f for c, f in pv.axes_frames(pg, grid).items() if c in clips}
    # sharey: only the left panel carries y tick labels; the right panel's horizontal
    # gridlines sit at the same heights, so its y map is the left one's.
    fx0, fy0, r0 = pv.calibrate(pg, clips[0], frames[clips[0]])
    assert sorted(frames[clips[0]]["hy"]) == sorted(frames[clips[1]]["hy"])
    out, worst = {}, 0.0
    for k, clip in enumerate(clips):
        x_lab = pv.tick_labels(pg, clip, "x")
        fr = frames[clip]
        prs = [(v, min(fr["vx"], key=lambda q: abs(q - c))) for v, c in x_lab]
        ax_, bx, rx = pv.fit_line([p[0] for p in prs], [p[1] for p in prs])
        assert max(r0, rx) < 1e-3
        fx = lambda X: (X - bx) / ax_  # noqa: E731
        denom = N_PAIRS[k] * N_TILES
        items = [it for it in pg.items if getattr(it, "clip", None) == clip]
        for i, bb in enumerate(NAME):
            col = colors[i]
            line = next(it for it in items if isinstance(it, pv.Path_) and it.op == "S"
                        and abs(it.width - 1.5) < 1e-6 and pv.close(it.stroke, col))
            dots = [(m.x, m.y) for m in items if isinstance(m, pv.Mark) and pv.close(m.fill, col)]
            assert len(dots) == len(line.pts) and max(
                max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(dots, line.pts)) < 1e-4
            for x, y in dots:
                s, v = fx(x), fy0(y)
                assert abs(s - round(s)) < 1e-3, s
                q = v * denom
                worst = max(worst, abs(q - round(q)))
                out.setdefault(bb, {}).setdefault(int(round(s)), [None, None])[k] = round(q) / denom
    for bb, s in out.items():
        assert all(None not in v for v in s.values()), (bb, s)   # same steps in both panels
        out[bb] = dict(sorted((st, tuple(v)) for st, v in s.items()))
    return out, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recover-from-pdf", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()

    prev = json.loads(args.out.read_text())["series"] if args.out.exists() else {}
    rec, worst = recover_pdf(args.recover_from_pdf) if args.recover_from_pdf else ({}, None)
    if worst is not None:
        print(f"recovered from {args.recover_from_pdf}: worst snap residual = {worst:.2e} queries")
    series = {}
    for bb in NAME:
        s, src = disk_series(bb), SRC_DISK
        if s and rec.get(bb):
            d = max(abs(a - b) for st in s if st in rec[bb] for a, b in zip(s[st], rec[bb][st]))
            print(f"  validate {bb}: max |recovered - disk| = {d:.2e}")
        if not s and rec.get(bb):
            s, src = rec[bb], SRC_PDF
        if not s:
            if bb in prev:
                series[bb] = prev[bb]
                print(f"  {bb}: kept existing entry ({prev[bb]['source']})")
            else:
                print(f"  {bb}: no data")
            continue
        series[bb] = {"name": NAME[bb], "selected_step": SEL[bb], "source": src,
                      "points": [[st, sc, sn] for st, (sc, sn) in s.items()]}
        print(f"  {bb:9s} steps={list(s)} <- {src}")
        if 0 in s:
            print(f"            base scanner {s[0][0]:.4f} stain {s[0][1]:.4f}")
    out = {"description": "PLISM held-out top-1 retrieval (scanner / stain shift, `.embedding` "
                          "view, 256 tiles) vs training step for the 500-step genMASK-b00 runs, "
                          "seed 0; step 0 = base probe. Rendered by paper/scripts/plism_traj.py; "
                          "regenerated by paper/scripts/extract_plism_traj.py.",
           "series": series}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
