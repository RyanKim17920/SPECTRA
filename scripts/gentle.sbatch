#!/bin/bash
#SBATCH --job-name=spectra-ret
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# ======================================================================================
# SPECTRA training: ONE recipe for every backbone; only the arm and the seed vary.
# ======================================================================================
#
# Paper recipe (set explicitly by the invocation below; the library defaults differ):
#   LoRA rank 32 / alpha 64 on every attention and MLP layer, C=2 conditions x T=900
#   locations per step, 500 steps, checkpoint (and RI eval) every 50 steps, lr 1e-4,
#   warmup 200, temperature 0.07, same-pseudo-tissue negatives up-weighted by e^3 in the
#   CLS loss (SPECTRA_BCLS=3.0) and dropped from the GeM loss (SPECTRA_BMEAN=-inf).
#
#   SPECTRA_ARM=virchow2 SPECTRA_SEED=0 SPECTRA_T=900 SPECTRA_MASK=1 SPECTRA_BCLS=3.0 \
#   SPECTRA_MAX_STEPS=500 SPECTRA_CKPT_EVERY=50 sbatch scripts/gentle.sbatch
#
# Arms: phikon | midnight | virchow2 | hoptimus | uni2 | virchow | openmidnight.
#
# GRID_CHUNK=0 is not tunable: a learned GeM pool head pools over the raw patch-token
# sequence, so the whole C*T forward is one graph. --activation-offload parks the saved
# activations in pinned host RAM (an exact save/restore hook, no approximation), which is
# why --mem=700G is load-bearing: pinned pages are not swappable.
#
# Training runs out of a frozen code snapshot under $SPECTRA_SNAPSHOTS (scripts/_env.sh,
# default <repo>/snapshots), chosen by arm so each result is reproduced by the code that
# produced it: snapshots/falseneg-pinned for phikon / midnight / virchow2, and
# snapshots/falseneg-gated (the same code plus local-directory backbone loading in
# models/encoder.py) for hoptimus / uni2 / virchow / openmidnight, whose hub repos are
# gated or need a local conversion. SPECTRA_PIN overrides the snapshot with any path.
# Variables are read as SPECTRA_*; scripts/_env.sh mirrors them onto the WAIV_* spellings
# that the frozen snapshots still read.
set -euo pipefail

REPO="${SPECTRA_REPO:-${WAIV_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
PACKED_DIR="${WAIV_PACKED_DIR:-$SPECTRA_PLISM_PACKED}"
ARM="${WAIV_ARM:?set WAIV_ARM=phikon|midnight|virchow|virchow2|hoptimus|uni2|openmidnight}"
SEED="${WAIV_SEED:?set WAIV_SEED}"
GRID_TILES="${WAIV_T:?set WAIV_T (common across backbones -- that is the point)}"

case "$ARM" in
  phikon)   BACKBONE="" ;;                        # unset => train_lora.py default owkin/phikon-v2
  midnight) BACKBONE="kaiko-ai/midnight" ;;
  virchow2) BACKBONE="paige-ai/Virchow2" ;;
  # Both hub repos are GATED (403).  encoder.BACKBONE_LOCAL_DIRS binds these ids to local
  # checkpoints under $SPECTRA_INPUTS; override with WAIV_BACKBONE_LOCAL_DIRS.
  hoptimus) BACKBONE="bioptimus/H-optimus-0" ;;
  uni2)     BACKBONE="MahmoodLab/UNI2-h" ;;
  # OpenMidnight is NOT gated; it is bound locally because the source is a raw DINOv2
  # training checkpoint, converted once into $SPECTRA_INPUTS/OpenMidnight by
  # scripts/convert_openmidnight.py.  Same DINOv2 ViT-g/14 shape as H-Optimus-0.
  openmidnight) BACKBONE="SophontAI/OpenMidnight" ;;
  # Virchow **v1** -- a DIFFERENT model from the virchow2 arm above, not a variant of it:
  # timm ViT-H/14, 32 blocks, 1280-d, packed-SwiGLU FFN, but NO register tokens (1 prefix
  # token / 257 tokens, against Virchow2's 5 / 261).  The repo is gated, so
  # encoder.BACKBONE_LOCAL_DIRS binds the id to the pinned snapshot under $SPECTRA_INPUTS/Virchow.
  #
  # Exact arm tokens, not globs, so `virchow` and `virchow2` cannot shadow each other. The
  # eval scripts glob on the RUN NAME with `*virchow2*`, which a `-virchow-s<seed>` name
  # does not match; do not add a `*virchow*` case, it would also capture Virchow2 runs.
  virchow)  BACKBONE="paige-ai/Virchow" ;;
  *) echo "bad WAIV_ARM=$ARM" >&2; exit 2 ;;
esac
BACKBONE_ARGS=()
[ -n "$BACKBONE" ] && BACKBONE_ARGS=(--backbone "$BACKBONE")

# --- pin resolution -------------------------------------------------------------------
# See the PIN note in the header for why the default differs by arm. $SPECTRA_SNAPSHOTS is
# repo-relative (scripts/_env.sh), so these defaults resolve inside whatever checkout this
# job was submitted from -- no absolute path, no machine assumption.
case "$ARM" in
  hoptimus|uni2|openmidnight|virchow) DEFAULT_PIN="$SPECTRA_SNAPSHOTS/falseneg-gated" ;;
  *)                                  DEFAULT_PIN="$SPECTRA_SNAPSHOTS/falseneg-pinned" ;;
esac
PIN="${SPECTRA_PIN:-${WAIV_PIN:-$DEFAULT_PIN}}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

# The package directory is DISCOVERED rather than named, so the checks below keep working
# across a package rename in the working tree (the snapshots are frozen and still ship the
# old name; nothing here should encode which name that is).
PIN_PKG=""
for _cand in "$PIN"/src/*/; do
  [ -f "${_cand}train/contrastive.py" ] || continue
  PIN_PKG="${_cand%/}"
  break
done

if [ -z "$PIN_PKG" ] || [ ! -f "$PIN/scripts/train_lora.py" ]; then
  {
    echo "pin does not resolve to a runnable snapshot: $PIN"
    echo "  wanted: \$PIN/src/<package>/train/contrastive.py and \$PIN/scripts/train_lora.py"
    echo "  arm     $ARM  ->  default pin $DEFAULT_PIN"
    echo "  roots   SPECTRA_SNAPSHOTS=$SPECTRA_SNAPSHOTS   (from scripts/_env.sh, repo-relative)"
    echo "          SPECTRA_REPO=$REPO"
    echo "  This repo vendors snapshots/falseneg-pinned and snapshots/falseneg-gated, so on"
    echo "  a clean checkout the defaults exist. If they do not, the checkout is partial"
    echo "  (git lfs / sparse-checkout) or SPECTRA_SNAPSHOTS points somewhere else."
    echo "  Override with SPECTRA_PIN=/path/to/snapshot."
  } >&2
  exit 3
fi

# The pin MUST carry the pool-head delegation, or GEM+grid dies at step 0.
grep -q 'delegate rather than refuse' "$PIN_PKG/train/contrastive.py" \
  || { echo "PIN LACKS pool-head delegation -- GEM+grid cannot run: $PIN" >&2; exit 3; }

# WAIV_TAG: free-form discriminator folded into RUN_NAME. REQUIRED whenever an arm differs
# from another only by a parameter that RUN_NAME does not otherwise encode -- above all
# WAIV_BCLS / WAIV_BMEAN, which are invisible here. HEST and PathoROB caches are keyed on
# run name / exp_code ALONE, so two arms that collide in this string silently share results.
RUN_NAME="gen${WAIV_MASK:+MASK}${WAIV_TAG:+-${WAIV_TAG}}${WAIV_MINTF:+-mtf${WAIV_MINTF}}-lr${WAIV_LR:-1e-4}-r${WAIV_RANK:-32}-kl${WAIV_KL:-0}-t${WAIV_TEMP:-0.07}-wd${WAIV_WD:-0.05}-ms${WAIV_MAX_STEPS:-1500}-pd${WAIV_PROJDIM:-512}-${ARM}-s${SEED}-t${GRID_TILES}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
# WAIV_RUN_NAME: replace the generated name wholesale (the job id is still appended, so
# HEST/PathoROB cache keys stay unique per job). This exists because THUNDER's pydantic
# run_tags hard-caps a model key at 64 chars and the THUNDER key is f5_<RUN_NAME>_s<step:07d>
# (collect_final5.py:559) = len + 12. The generated name is ~71 chars, so every THUNDER job
# for a gentle.sbatch run is REJECTED AT VALIDATION -- silently, as a killed sweep, not an error
# here. If you override, keep the tokens scoreboard2.py:_parse_run_meta greps back out --
# MASK, -lr, -kl, -ms, -s<seed>, -t<T>, a trailing job id, and a name in scoreboard2._BACKBONES --
# or that run's scoreboard metadata degrades to unknown/None.
if [ -n "${WAIV_RUN_NAME:-}" ]; then
  RUN_NAME="${WAIV_RUN_NAME}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
fi

THUNDER_KEY_LEN=$(( ${#RUN_NAME} + 12 ))
if [ "$THUNDER_KEY_LEN" -gt 64 ]; then
  echo "RUN_NAME is too long to ever be THUNDER-evaluated:" >&2
  echo "  f5_<run>_s<step> = $THUNDER_KEY_LEN chars > 64 (THUNDER run_tags cap)" >&2
  echo "  run  = $RUN_NAME (${#RUN_NAME} chars; budget is 52)" >&2
  echo "  fix  = set WAIV_RUN_NAME to something shorter" >&2
  echo "  or   = WAIV_ALLOW_LONG_RUN_NAME=1 to proceed anyway (RI/HEST only, NO THUNDER)" >&2
  if [ -z "${WAIV_ALLOW_LONG_RUN_NAME:-}" ]; then exit 4; fi
  echo "  WAIV_ALLOW_LONG_RUN_NAME=1 -- proceeding; THUNDER eval will be impossible for this run" >&2
fi

OUT_DIR="$REPO/runs/$RUN_NAME"

mkdir -p "$OUT_DIR"

MAX_STEPS="${WAIV_MAX_STEPS:-1500}"
CKPT_EVERY="${WAIV_CKPT_EVERY:-250}"
WORKERS="${WAIV_WORKERS:-10}"
LORA_RANK="${WAIV_RANK:-32}"
LORA_ALPHA="${WAIV_ALPHA:-$((LORA_RANK * 2))}"
LR="${WAIV_LR:-1e-4}"
TEMPERATURE="${WAIV_TEMP:-0.07}"
GRID_COND=2
GRID_CHUNK=0          # NOT tunable: GEM requires unchunked. See header.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-72000}"
HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem --retention-kl-weight "${WAIV_KL:-0}")

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} ==="
echo "run       $RUN_NAME"
echo "arm       $ARM  backbone=${BACKBONE:-<default owkin/phikon-v2>}  seed=$SEED"
echo "geometry  C=$GRID_COND T=$GRID_TILES B=$((GRID_COND * GRID_TILES)) chunk=$GRID_CHUNK neg/row=$((GRID_TILES - 1))"
echo "head      ${HEAD_ARGS[*]}"
echo "steps     $MAX_STEPS  ckpt/eval $CKPT_EVERY"
echo "pin       $PIN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
free -g | head -2 || true

"$PY" - "$PIN" "$PIN_PKG" <<'EOF'
import importlib
import sys
from pathlib import Path

pin, pkg_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(pin / "src"))
pkg = importlib.import_module(pkg_dir.name)
# The point of the pin is that NOTHING resolves to the working tree. An installed copy of
# the same package would shadow it silently and the run would be off-provenance with no
# symptom, so this is fatal rather than a warning.
assert Path(pkg.__file__).is_relative_to(pin), f"PIN LEAK: {pkg.__file__}"
print(f"[pin] {pkg_dir.name}={pkg.__file__}", flush=True)
EOF

"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, len(ok)
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} conditions, {len(stains)} stains, {len(scanners)} scanners")
assert len(stains) == 13 and len(scanners) == 7
EOF
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

rm -f "$OUT_DIR/TRAIN_DONE"
# WAIV_NO_FOLLOWER=1 skips the RI eval follower. ONLY for memory/geometry smokes, which
# write no checkpoints and so give the follower nothing to read. Never set it for a real
# run: the follower is where ri_curve.json comes from, and RI is the only readout in this
# project with the dynamic range to rank arms.
FOLLOWER=""
if [ "${WAIV_NO_FOLLOWER:-0}" = "1" ]; then
  echo "=== eval follower DISABLED (WAIV_NO_FOLLOWER=1) -- smoke only, no ri_curve.json ==="
else
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" "${BACKBONE_ARGS[@]}" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="
fi

set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean \
  --proj-out-dim "${WAIV_PROJDIM:-512}" \
  "${BACKBONE_ARGS[@]}" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --weight-decay "${WAIV_WD:-0.05}" \
  ${WAIV_MINTF:+--min-tissue-frac ${WAIV_MINTF}} \
  ${WAIV_MASK:+--mask-same-core} \
  ${WAIV_MASK:+--same-core-logit-bias-cls=${WAIV_BCLS:--inf}} \
  ${WAIV_MASK:+--same-core-logit-bias-mean=${WAIV_BMEAN:--inf}} \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "${WAIV_WARMUP:-200}" \
  --grid --grid-conditions "$GRID_COND" --grid-tiles "$GRID_TILES" \
  --grid-forward-chunk "$GRID_CHUNK" \
  "${HEAD_ARGS[@]}" \
  --grad-checkpointing \
  --activation-offload \
  --workers "$WORKERS" --amp bfloat16 --log-every 20 \
  --ckpt-every "$CKPT_EVERY" --eval-every "$CKPT_EVERY" \
  --seed "$SEED"
TRAIN_RC=$?
set -e

echo "rc=$TRAIN_RC" > "$OUT_DIR/TRAIN_DONE"
[ -n "$FOLLOWER" ] && wait $FOLLOWER 2>/dev/null || true
kill $SAMPLER 2>/dev/null || true

echo "=== RI curve ==="
"$PY" - "$OUT_DIR" <<'EOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json")
EOF

echo "=== done $(date) | $OUT_DIR | arm=$ARM seed=$SEED T=$GRID_TILES rc=$TRAIN_RC ==="
exit $TRAIN_RC
