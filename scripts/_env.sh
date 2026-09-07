# shellcheck shell=bash
# ======================================================================================
# Path roots for every shell/sbatch script here -- the shell half of waivphaet.paths.
# ======================================================================================
#
# Source it near the top of a job script:
#
#     REPO="${SPECTRA_REPO:-${WAIV_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}}"
#     . "$REPO/scripts/_env.sh"
#
# SLURM copies a batch script into its spool directory, so $0 and BASH_SOURCE do NOT
# point at the checkout inside a job -- $SLURM_SUBMIT_DIR does, and that is also why the
# #SBATCH --output paths here are relative. Submit from the repo root, or set
# SPECTRA_REPO, and both work.
#
# Every root is <VAR>=${VAR:-<default>}, so anything already exported by the caller wins,
# then .env fills the rest, then repo-relative defaults. See src/waivphaet/paths.py for
# what each root means; the two files must agree on names and defaults.

SPECTRA_REPO="${SPECTRA_REPO:-${WAIV_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}}"

# .env is optional and must not clobber the real environment, so each line is only
# applied when its variable is unset. `export -n` after the fact would be too late.
if [ -f "$SPECTRA_REPO/.env" ]; then
  while IFS= read -r _spectra_line || [ -n "$_spectra_line" ]; do
    case "$_spectra_line" in
      ''|'#'*) continue ;;
    esac
    _spectra_line="${_spectra_line#export }"
    case "$_spectra_line" in
      *=*) ;;
      *) continue ;;
    esac
    _spectra_key="${_spectra_line%%=*}"
    _spectra_val="${_spectra_line#*=}"
    _spectra_val="${_spectra_val%\"}"; _spectra_val="${_spectra_val#\"}"
    _spectra_val="${_spectra_val%\'}"; _spectra_val="${_spectra_val#\'}"
    if [ -z "$(eval "printf '%s' \"\${$_spectra_key:-}\"")" ]; then
      eval "$_spectra_key=\$_spectra_val"
      eval "export $_spectra_key"
    fi
  done < "$SPECTRA_REPO/.env"
  unset _spectra_line _spectra_key _spectra_val
fi

SPECTRA_RUNS="${SPECTRA_RUNS:-$SPECTRA_REPO/runs}"
SPECTRA_DATA="${SPECTRA_DATA:-$SPECTRA_REPO/data_root}"
SPECTRA_PLISM="${SPECTRA_PLISM:-$SPECTRA_DATA/plism}"
SPECTRA_PLISM_PACKED="${SPECTRA_PLISM_PACKED:-$SPECTRA_PLISM/repacked}"
SPECTRA_THUNDER="${SPECTRA_THUNDER:-${THUNDER_BASE_DATA_FOLDER:-$SPECTRA_DATA/thunder}}"
SPECTRA_HEST_WORK="${SPECTRA_HEST_WORK:-$SPECTRA_DATA/hest_work}"
SPECTRA_HEST_BENCH="${SPECTRA_HEST_BENCH:-$SPECTRA_DATA/hest_bench}"
SPECTRA_EVALS="${SPECTRA_EVALS:-$SPECTRA_DATA/full-evals}"
SPECTRA_HF_HOME="${SPECTRA_HF_HOME:-$SPECTRA_DATA/huggingface}"
SPECTRA_INPUTS="${SPECTRA_INPUTS:-$SPECTRA_DATA/inputs}"
SPECTRA_CELLS="${SPECTRA_CELLS:-$SPECTRA_REPO/cells}"
SPECTRA_PAPER="${SPECTRA_PAPER:-$SPECTRA_REPO/paper}"
SPECTRA_SNAPSHOTS="${SPECTRA_SNAPSHOTS:-$SPECTRA_REPO/snapshots}"
SPECTRA_BACKUPS="${SPECTRA_BACKUPS:-$SPECTRA_REPO/result_backups}"

export SPECTRA_REPO SPECTRA_RUNS SPECTRA_DATA SPECTRA_PLISM SPECTRA_PLISM_PACKED \
       SPECTRA_THUNDER SPECTRA_HEST_WORK SPECTRA_HEST_BENCH SPECTRA_EVALS \
       SPECTRA_HF_HOME SPECTRA_INPUTS SPECTRA_CELLS SPECTRA_PAPER \
       SPECTRA_SNAPSHOTS SPECTRA_BACKUPS

# Names the third-party harnesses read. Set only if the caller has not already.
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE_DATA_FOLDER:-$SPECTRA_THUNDER}"
