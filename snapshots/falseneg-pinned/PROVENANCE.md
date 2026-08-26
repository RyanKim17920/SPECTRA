# falseneg-pinned — the code that produced GEN-3

Vendored 2026-08-26 from `/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned`
(an uncommitted snapshot dir on volatile storage).

**Why this exists:** every GEN-3 result (`runs/genMASK-c3s-*`, the current candidate
recipe) was trained by THIS code, and it was not in version control. The repo's own
`scripts/train_lora.py` and `src/waivphaet/train/contrastive.py` do NOT implement
`mask_same_core` — the central mechanism of the recipe — even though sbatch scripts in
HEAD pass `--mask-same-core`. Without this directory the GEN-3 numbers are
unreproducible.

Divergence from the repo working tree at time of vendoring:
  scripts/train_lora.py                 59 diff lines
  src/waivphaet/train/contrastive.py   147 diff lines
  src/waivphaet/models/encoder.py       62 diff lines
  src/waivphaet/data/grid.py            35 diff lines

`mask_same_core` implementation lives at:
  scripts/train_lora.py:234,252,258,262,269,619
  src/waivphaet/train/contrastive.py:890,895-897,1632-1643,1729,1779

Excluded from the copy: `.venv*`, `third_party/`, `runs/`, `__pycache__`, `*.pyc`.
This is a verbatim code copy, NOT a merge. Do not treat it as the maintained source.
