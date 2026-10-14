#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
# Downstream robustness for ANY SPECTRA-CONCH checkpoint(s), vs the cached base-CONCH features:
#   sbatch scripts/conch_downstream.sbatch fixA_s100=/path/run/step_0000100 [name=<step dir or adapter dir> ...]
# Stages (each skips work already cached; base CONCH features are reused, never recomputed):
#   1. PLISM held-out flip rates      -> results/spectra_conch/plism_flip/flip_results.json
#   2. TCGA 300-slide embed (true 20x, 448px, <=500 tiles) + site/type kNN + site-grouped LP
#                                     -> results/spectra_conch/tcga/slide_level.json
#   3. 8-slide clustering coherence   -> results/spectra_conch/cluster8/slides_cluster.json
set -euo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch; RES=/data/ryan.kim/ashby/results/spectra_conch
source ~/.cache_env.sh 2>/dev/null || true
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local} PYTHONPATH="$REPO/src"
PY=/data/ryan.kim/ashby/envs/conch/bin/python; cd "$REPO"
[ $# -ge 1 ] || { echo "usage: name=<ckpt> ..."; exit 2; }
$PY scripts/conch_plism_flip.py --models "$@"
$PY scripts/conch_slides_embed.py --slides $RES/tcga/selection.json --out /data/ryan.kim/ashby/scratch/spectra_conch_tcga --max-tiles 500 --models "$@"
$PY scripts/conch_tcga_slide_level.py
$PY scripts/conch_slides_embed.py --slides $RES/cluster8/selection.json --out /data/ryan.kim/ashby/scratch/spectra_conch_cluster8 --max-tiles 0 --chunk-slides 1 --models "$@"
$PY scripts/conch_slides_cluster.py
echo "DOWNSTREAM DONE $(date)"
