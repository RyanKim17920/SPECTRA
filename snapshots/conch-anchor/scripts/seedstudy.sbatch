#!/bin/bash
#SBATCH --job-name=waiv-seedstudy
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# SEED STUDY: settle split-vs-GEM on the ONLY metric with demonstrated dynamic range
# (PathoROB avg_robustness_index), and price the multi-seed noise floor at the same time.
#
# WHY THIS EXISTS. Both external readouts are blind at this effect size:
#   - HEST spans ~0.5 noise units across all arms.
#   - THUNDER linear probe spans 1.0-1.6 noise units; ~2 nu is the minimum to rank.
#   - `fast5_ctrlseed_clsmean` -- a PURE SEED REPLICATE of the control with no config
#     change at all -- outscored all nine real arms. Arm ordering on those instruments is
#     a seed draw, not a result.
# RI is the only metric with range (~0.023 span against a ~0.007 seed floor), and the
# split-vs-GEM question currently FLIPS SIGN between the cls and clsmean THUNDER
# protocols. So: more seeds, ranked on RI.
#
#   WAIV_ARM=split  --split-heads --cls-weight 0.5 --mean-weight 0.5
#   WAIV_ARM=gem    --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
#   WAIV_ARM=ctrl   --n-groups 6 --group-size 200          (no head args at all)
#
# EXISTING SEED-0 POINTS -- do NOT re-run:
#   split seed 0   runs/headcmp-split-380856   (job 380856, 2026-08-14 05:24)
#   gem   seed 0   runs/poolcmp-gem-381014     (job 381014, 2026-08-14 06:00)
#   ctrl  seed 0   runs/gridcmp-ctrl-380777    (job 380777, 2026-08-14 00:31)
#   ctrl  seed 1   runs/gridcmp2-ctrlseed-380889 (job 380889, 2026-08-14 05:24)
#
# THE ONLY THING THIS SCRIPT MOVES IS --seed. Every other flag below was read back out of
# those three runs' config.json, not retyped from memory. A seed replicate whose config
# differs anywhere else is not a seed replicate.
#
# --- WHY THE PIN, AND WHY THIS PIN ----------------------------------------------------
# None of headcmp.sbatch / poolcmp.sbatch / gridcmp.sbatch pin their python source: they
# read scripts/ and src/ from the live working tree at job START time. That was survivable
# then and is NOT survivable now -- the tree currently carries uncommitted work on
# scripts/train_lora.py, src/waivphaet/models/encoder.py and
# src/waivphaet/train/contrastive.py (blocked grid InfoNCE, activation offload). A seed
# replicate that silently picked that up would be measuring the code change, not the seed.
#
# gridcmp2-pinned is the RIGHT snapshot, verified by md5 rather than assumed: for
# encoder.py, contrastive.py, train_lora.py, eval_checkpoints.py and embed_probe.py it is
# byte-identical to commits 42990ba AND 6bfa52a -- which are the two commits the seed-0
# split / gem / ctrlseed arms actually ran at (42990ba..6bfa52a touches only
# poolcmp_gates.sbatch, so the two trees are the same source). It is also the snapshot
# ctrlseed-380889 itself ran under. So these runs execute the same bytes their seed-0
# partners did.
#
# The pin is read-only and shared with completed runs. Do NOT edit it.
#
#   WAIV_ARM=split WAIV_SEED=1 sbatch --account=max --qos=high \
#     --job-name=waiv-seedstudy-split-s1 scripts/seedstudy.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0               # the follower streams the PathoROB parquet
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gridcmp2-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PIN/src/waivphaet" ] || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
ARM="${WAIV_ARM:?set WAIV_ARM to split | gem | ctrl}"
SEED="${WAIV_SEED:?set WAIV_SEED (seed 0 already exists for all three arms)}"
# SLURM_RESTART_COUNT keeps a requeued attempt out of the previous attempt's run dir, so a
# preemption cannot silently mix two training trajectories (commit 42990ba).
RUN_NAME="seedstudy-${ARM}-s${SEED}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64

# How long the eval follower tolerates before giving up. THE PINNED eval_checkpoints.py
# PREDATES the idle-clock fix that is still uncommitted in the working tree: in the pin,
# t0 is set once at follower startup and never reset, so --max-wait-s is a cap on TOTAL
# follower runtime, and its 8 h default is a live truncation hazard (this is exactly what
# stranded arm F's step_0000750 unscored). The pin is read-only and must not be edited, so
# the fix here is to pass a value that cannot be reached inside the 10 h wall: 20 h. Under
# EITHER semantics -- total cap or idle timeout -- 20 h means the follower only ever exits
# via --stop-file (TRAIN_DONE) or a genuinely wedged trainer.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-$((20 * 3600))}"

# --- the ONE axis that differs between arms (seed is the other, and it is the point) ---
# Read verbatim from the seed-0 configs. split and gem share the loss-head block; gem adds
# --pool-head gem. ctrl passes NO head args -- its config has split_heads:false.
case "$ARM" in
  split) HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5) ;;
  gem)   HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem) ;;
  ctrl)  HEAD_ARGS=() ;;
  *) echo "unknown WAIV_ARM=$ARM (want split | gem | ctrl)" >&2; exit 2 ;;
esac

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | restart ${SLURM_RESTART_COUNT:-0} | run $RUN_NAME ==="
echo "=== arm $ARM | seed $SEED | heads ${HEAD_ARGS[*]:-<none>} | pin $PIN ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Fail at launch, not at the first checkpoint, if PYTHONPATH lost the pin to .venv.
"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import waivphaet
assert waivphaet.__file__.startswith(sys.argv[1]), f"PYTHONPATH lost to .venv: {waivphaet.__file__}"
print(f"[pin] waivphaet={waivphaet.__file__}", flush=True)
EOF

# --- preflight: all 91, verified, and pin the probe's condition set --------------------
"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)} (repack incomplete)"
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} verified slides | {len(stains)} stains | {len(scanners)} scanners")
assert len(stains) == 13 and len(scanners) == 7
EOF
# The seed-0 arms all scored this exact condition set; if the manifest ever moved, the RI
# numbers stop being comparable. Print the digest so a later reader can check.
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base phikon-v2, pinned condition set) --------------------
# No --proj-out-dim / --n-tiles: the seed-0 arms passed neither, so the defaults are the
# protocol here.
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1 -----------------------------------------------------------
# NO --pooling. eval_checkpoints.py defaults to clsmean, which is the protocol constant
# (PathoROB's reference row is phikonv2_clsmean); passing the training head selection
# through would silently change what the arms are compared against. NO --model-prefix
# either: the default names the feature/results dirs the way every comparable arm's were.
rm -f "$OUT_DIR/TRAIN_DONE"
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER_PID $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="

# --- training on GPU 0 ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 1500 \
  --warmup-steps 200 \
  --n-groups 6 --group-size 200 \
  ${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"} \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 20 \
  --ckpt-every 250 \
  --eval-every 250 \
  --seed "$SEED" \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="
touch "$OUT_DIR/TRAIN_DONE"

wait $FOLLOWER || true
kill $SAMPLER_PID 2>/dev/null || true

# --- the comparability check, printed rather than trusted ------------------------------
# Diffs this run's config.json against its seed-0 partner. The ONLY key that may differ
# besides out_dir is `seed`. Anything else printed here means this is not a seed replicate
# and the run must not enter the seed table.
echo "=== config diff vs seed-0 partner (expect: seed, out_dir ONLY) ==="
"$PY" - "$OUT_DIR" "$ARM" <<'EOF'
import json, sys
from pathlib import Path
REF = {"split": "runs/headcmp-split-380856",
       "gem":   "runs/poolcmp-gem-381014",
       "ctrl":  "runs/gridcmp-ctrl-380777"}
out, arm = Path(sys.argv[1]), sys.argv[2]
repo = Path("/admin/home/ryan.kim/waiv")
mine = json.loads((out / "config.json").read_text())
ref = json.loads((repo / REF[arm] / "config.json").read_text())

def flat(d, p=""):
    f = {}
    for k, v in d.items():
        f.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else f.setdefault(p + k, v)
    return f

a, b = flat(mine), flat(ref)
allowed = {"seed", "out_dir"}
bad = []
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        tag = "ok " if k in allowed else "!! "
        print(f"{tag}{k}: {b.get(k, '<absent>')} -> {a.get(k, '<absent>')}")
        if k not in allowed:
            bad.append(k)
print("[comparability] CLEAN -- differs in seed/out_dir only" if not bad
      else f"[comparability] BROKEN -- unexpected keys moved: {bad}")
EOF

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (arm $ARM, seed $SEED, train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
