#!/bin/bash
#SBATCH --job-name=waiv-fullft
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
# 600 steps, not 4000. Full FT moves every one of the 303M backbone parameters instead of
# a rank-32 adapter, so the interesting part of the curve is at the FRONT: if this works it
# shows by step 200, and if it collapses it collapses in tens of steps. 4 h is training
# (~35-45 min at an expected 3-4 s/step -- slower than the LoRA run's 1.91 s because the
# backward now touches the whole backbone) plus the Tier-2 follower draining 10 checkpoints
# of PathoROB behind it, which is the actual long pole.
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# The full fine-tuning arm of the LoRA-vs-full-FT A/B (PLAN.md 3 phase 7/8). The control is
# runs/waiv-real-369043 (rank-32 LoRA, 4000 steps). Everything that could confound the
# comparison is pinned to that run's value:
#
#   batch      2 x 192 (191 same-condition negatives per anchor)   same
#   temperature 0.07                                               same
#   split      heldout scanners GT450 S210 / stains HRH KR MY      same
#   pooling    clsmean                                             same (script default)
#   proj-dim   512                                                 same (script default)
#   seed       0                                                   same (script default)
#   backbone   owkin/phikon-v2                                     same (script default)
#
# What deliberately differs, and why:
#
#   --full-ft            THE variable under test.
#   --lr 1e-5            10x below the LoRA run's 1e-4. A LoRA adapter starts at zero and
#                        cannot destroy the base weights; full FT at 1e-4 destroys a
#                        pathology representation in tens of steps. train_lora.py warns
#                        above 5e-5. Waiv's Phaet is ~1e-5 class.
#   --weight-decay 0.02  Below the 0.05 default. Decay applied to every backbone weight is
#                        not the same operator as decay applied to a rank-32 adapter: at
#                        0.05 it is itself a pull toward zero on 303M pretrained params.
#   --warmup-steps 100   vs 200, because the run is 600 steps not 4000 -- same fraction.
#   --max-steps 600
#   --ckpt-schedule      Dense at the front: 25,50,75,100 then thinning out. Uniform
#                        --ckpt-every 500 would give this run ONE checkpoint before the
#                        step at which a full-FT collapse would already be complete.
#
# 34 GiB of checkpoints (11 x ~3.4 GiB: backbone.safetensors 1.13 GiB + optim.pt 2.26 GiB
# of AdamW moments + projector), so OUT_DIR is on /data, NOT /admin. /admin is 87% used
# with 838G free and is not where 34 GiB of throwaway optimizer state belongs.
#
# GPU 1 runs TWO followers:
#   * probe_follow.py  -- Tier 1, ~30 s/checkpoint, the collapse tripwire. Its probe json
#                         lands at the path eval_checkpoints.py skips on, so it is free.
#   * eval_checkpoints.py -- Tier 2, ~15-20 min/checkpoint, the PathoROB RI curve we quote.
# Gating is on embedding-space separation + top-1, never on training loss or matched
# cosine (matched cosine RISES under collapse).
#
#   sbatch scripts/train_full_ft.sbatch
#   sbatch scripts/train_full_ft.sbatch --lr 5e-6 --max-steps 1000   # extra args pass through

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/ryan.kim/hf_home}"
export HF_HUB_OFFLINE=0               # the Tier-2 follower streams the PathoROB parquet
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PACKED_DIR="${WAIV_PACKED_DIR:-/data/ryan.kim/plism/repacked}"
RUN_NAME="${SLURM_JOB_NAME:-waiv-fullft}-${SLURM_JOB_ID:-local}"
# /data, not $REPO/runs: 11 checkpoints x ~3.4 GiB = ~34 GiB.
OUT_DIR="${WAIV_RUNS_DIR:-/data/ryan.kim/waiv_runs}/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

N_GROUPS=2
GROUP_SIZE=192
MAX_STEPS=600
CKPT_SCHEDULE="25,50,75,100,150,200,300,400,500,600"
EVAL_EVERY=25
WARMUP_STEPS=100
PROJ_OUT_DIM=512

# Backbone knob. UNSET => nothing is passed and train_lora.py's own default
# (owkin/phikon-v2) applies, i.e. byte-identical to the A/B arm this file was written for
# (control runs/waiv-real-369043) and to every run submitted before this knob existed.
# Set WAIV_BACKBONE=kaiko-ai/midnight (or any HF id) to full-FT something else; the same
# value is threaded into the BEFORE probe and the Tier-2 eval follower below.
#
# SWITCHING THE BACKBONE ALMOST CERTAINLY MEANS RE-SIZING THE BATCH. N_GROUPS/GROUP_SIZE
# above are sized for phikon-v2 (ViT-L/16, 24 blocks, 1024-d) at 2 x 192 with gradient
# checkpointing, and full FT already holds AdamW moments for all 303M params on top of
# that. midnight is ViT-g/14, 40 blocks, 1536-d -- ~3x the parameters -- and train_lora.py
# warns that a phikon-v2-sized batch will OOM on a wider backbone. Measure the new one
# first with
#     sbatch scripts/sizing_probe.sbatch kaiko-ai/midnight
# and edit N_GROUPS/GROUP_SIZE (or pass --n-groups/--group-size through "$@") from what it
# reports. WAIV_BACKBONE deliberately does NOT change the batch defaults, and it also does
# NOT change --lr 1e-5 / --weight-decay 0.02, which were chosen for a 303M ViT-L.
#
# NOTE: probe_follow.py (Tier 1, below) takes no --backbone. It does not need one -- it
# always passes --checkpoint/--adapter, and embed_probe.py derives the backbone from the
# checkpoint's own run config.json. Only the BEFORE probe, which has no checkpoint to read,
# needs to be told.
BACKBONE="${WAIV_BACKBONE:-}"
BACKBONE_ARGS=()
if [ -n "$BACKBONE" ]; then BACKBONE_ARGS=(--backbone "$BACKBONE"); fi

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN_NAME ==="
echo "=== backbone ${BACKBONE:-<script default: owkin/phikon-v2>} ==="
echo "=== out_dir $OUT_DIR ==="
df -h /data | tail -1
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- preflight: all 91, verified, and pin the probe's condition set --------------------
"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
# PLAN.md 3 phase 7 wants the full 13 x 7 grid, and the A/B control (369043) was scored on
# exactly this set -- a different condition set here would make the comparison meaningless.
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)} (repack incomplete)"
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} verified slides | {len(stains)} stains {stains} "
      f"| {len(scanners)} scanners {scanners}")
assert len(stains) == 13 and len(scanners) == 7
EOF

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base phikon-v2, pinned condition set) --------------------
# No --lora-rank/--lora-alpha here: this arm has no adapter, and the BEFORE probe is the
# bare backbone either way. This is the row the collapse thresholds are defined against.
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  "${BACKBONE_ARGS[@]}" \
  --out "$OUT_DIR/probe_before.json"

rm -f "$OUT_DIR/TRAIN_DONE"

# --- Tier-1 collapse tripwire on GPU 1 ------------------------------------------------
# Polls fast (20 s) and costs ~30 s per checkpoint. Writes probe_step_NNNNNNN.json to the
# path eval_checkpoints.py checks-and-skips on, so the Tier-2 follower below inherits the
# probe for free instead of recomputing it.
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/probe_follow.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --warmup-steps "$WARMUP_STEPS" \
  --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 20 \
  > "$OUT_DIR/probe_follow.log" 2>&1 &
TRIPWIRE=$!
trap 'kill $SAMPLER $TRIPWIRE 2>/dev/null || true' EXIT
echo "=== tier-1 tripwire pid $TRIPWIRE -> $OUT_DIR/probe_follow.log ==="

# --- Tier-2 eval follower on GPU 1 ----------------------------------------------------
# Starts now and idles on the poll until the first checkpoint lands. It exits once
# TRAIN_DONE exists and no unevaluated checkpoint remains.
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  "${BACKBONE_ARGS[@]}" \
  --num-workers 8 \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER $TRIPWIRE $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="

# --- training on GPU 0 ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --full-ft \
  "${BACKBONE_ARGS[@]}" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --temperature 0.07 \
  --lr 1e-5 \
  --weight-decay 0.02 \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --n-groups "$N_GROUPS" \
  --group-size "$GROUP_SIZE" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 10 \
  --ckpt-schedule "$CKPT_SCHEDULE" \
  --eval-every "$EVAL_EVERY" \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="
touch "$OUT_DIR/TRAIN_DONE"

# --- let both followers drain the remaining checkpoints -------------------------------
wait $TRIPWIRE || true
wait $FOLLOWER || true
kill $SAMPLER 2>/dev/null || true

echo "=== tier-1 collapse signals ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "collapse_watch.json"
if not p.exists():
    print("no collapse_watch.json written")
else:
    d = json.loads(p.read_text())
    for pt in d["points"]:
        flag = "TRIPPED" if pt["signals"] else "clean"
        print(f"  step {pt['step']:>6} wcr={pt['within_condition_random']:.4f} "
              f"scan_sep={pt['cross_scanner.separation']:.4f} "
              f"stain_top1={pt['cross_stain.top1']:.4f}  {flag}")
        for s in pt["signals"]:
            print(f"    *** {s}")
EOF

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
