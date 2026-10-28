#!/usr/bin/env python3
"""Freeze the 18 PLISM tiles Figure 1 (grid_batch) shows into paper/data/grid_batch_tiles.webp.

Data dependency: the repacked PLISM tiles, <SPECTRA_PLISM_PACKED>/<condition>.npy (uint8
N x 224 x 224 x 3, written by scripts/acquire_plism.py). The figure uses three conditions x
six registered locations (scripts/paper_figures.py CONDITIONS / LOCATIONS); they are stored (lossless WebP)
losslessly as one 3 x 6 mosaic so the figure renders without the 30+ GB corpus.

    SPECTRA_PLISM_PACKED=/path/to/repacked python3 paper/scripts/extract_grid_batch.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from _config import PLISM_PACKED  # noqa: E402
import paper_figures as pf  # noqa: E402

OUT = REPO / "paper/data/grid_batch_tiles.webp"


def main():
    rows = []
    for code, _ in pf.CONDITIONS:
        arr = np.load(PLISM_PACKED / f"{code}.npy", mmap_mode="r")
        rows.append(np.concatenate([np.asarray(arr[loc]) for loc in pf.LOCATIONS], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT, lossless=True, quality=100, method=6)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
