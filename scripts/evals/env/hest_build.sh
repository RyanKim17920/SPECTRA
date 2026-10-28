#!/bin/bash
# Rebuild third_party/HEST (pinned + hest.patch) and a HEST venv exactly as used for hest.json.
#   SPECTRA_REPO=/path/to/checkout PYTHON311=/path/to/python3.11 [SPECTRA_VENV_HEST=venv_dir] \
#     bash scripts/evals/env/hest_build.sh
# hest_requirements.in.txt = `uv pip freeze` of an upstream HEST environment.
# hest_lock.txt = the resulting freeze (+ peft 0.20.0 / accelerate). The checkout's src/ is put
# on sys.path with a .pth file because pyproject.toml requires python>=3.12 and HEST's env is
# 3.11 (spectra.evals itself imports fine under 3.11).
# Afterwards export SPECTRA_VENV_HEST=<venv_dir> for scripts/evals/run_hest.sbatch.
set -euo pipefail
WT="${SPECTRA_REPO:?set SPECTRA_REPO}"
PYTHON311="${PYTHON311:?set PYTHON311 to a python 3.11 interpreter}"
VENV="${SPECTRA_VENV_HEST:-$WT/.venv-hest}"
cd "$WT"
if [ ! -d third_party/HEST/.git ]; then
  git clone -q https://github.com/mahmoodlab/HEST.git third_party/HEST
  git -C third_party/HEST checkout -q 3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3
  git -C third_party/HEST apply "$WT/scripts/evals/env/hest.patch"
fi
# expected: git -C third_party/HEST diff | sha256sum == b5836af5a03a1ab112d7993973542ea1bc07815b03cd43be300fd6f2bd865ce0
# (hest.patch = I/O-only changes + prefetch_factor=None when num_workers==0, which torch 2.11 requires)
uv venv -q --python "$PYTHON311" "$VENV"
uv pip install -q --python "$VENV/bin/python" -r scripts/evals/env/hest_lock.txt \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$WT/src" > "$SP/_spectra_evals_worktree.pth"
"$VENV/bin/python" -c "import spectra.evals.model, sys; sys.path.insert(0, 'third_party/HEST/src'); import hest.bench.benchmark; print('ok')"
