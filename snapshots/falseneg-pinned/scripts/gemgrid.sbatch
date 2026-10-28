#!/bin/bash
#SBATCH --job-name=waiv-gemgrid
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# ======================================================================================
# DOES HEAD-SPLIT COMPOSE WITH HIGH NEGATIVES?
# ======================================================================================
#
# THE QUESTION. On the common 3-seed ctrl baseline, RI at step 500:
#
#   ctrl                                             .8063
#   split  (cls/mean split heads, pair path)         .8208   +0.0145
#   gem    (GEM pool head, pair path)                .8227   +0.0163
#   d      (C=2 x T=1800 grid, SINGLE head)          .8285   +0.0222
#
# The d-vs-gem gap is +0.0059 against a 3-seed 2SE floor of 0.0076, i.e. NOT significant.
# So we have TWO roughly equal-sized wins over ctrl -- a head change and a negatives
# change -- and no evidence about whether they ADD. This run is the crossing term: arm d's
# geometry with the SPLIT head bolted on. If the split's ~+0.0145 survives on top of the
# grid, the two mechanisms are separable and stack; if it vanishes, they were competing
# routes to the same gain and the leaderboard is a ceiling, not a sum.
#
# --- WHY SPLIT x GRID AND NOT GEM x GRID ----------------------------------------------
# GEM x grid is the arm we would rather have run, and it is BLOCKED in this source tree.
# The grid path calls _chunked_forward_split unconditionally (contrastive.py:1606), and
# that function places its `pool_head is not None` guard AFTER the chunk/no-chunk branch
# (contrastive.py:911-924), so it raises even unchunked. Fixing that guard is not in this
# run's scope -- the budget buys an experiment, not a refactor, and a source change would
# also break comparability with the pinned split/gem/ctrl arms.
#
# Split heads with ARITHMETIC MEAN pooling do NOT hit that guard, and this is a source
# fact rather than a hope: WaivEncoder sets self.pool_head = None when pool_head_name is
# "mean" (encoder.py:706-712, the `if self.pool_head_name != "mean" else None` branch), so
# the guard's `is not None` test is False and the grid path runs. Verified live by the
# 20-step smoke (job 384752) before this was submitted.
#
# --- WHAT MOVES vs THE SPLIT ARM ------------------------------------------------------
# Base config is runs/headcmp-split-380856/config.json, read back out of the file rather
# than retyped. The ONLY thing changed is the sampler geometry, taken from arm d
# (runs/gridcmp3-d-381559.r2/config.json):
#
#   grid            false            -> true
#   grid_conditions 0                -> 2
#   grid_tiles      0                -> 1800      (1799 negatives/anchor, B=3600)
#   grid_forward_chunk 0             -> 300
#   n_groups        6 x group_size 200  ->  4 x 64   (arm d's values; both are the pair
#                                                     path's knobs and are INERT under
#                                                     --grid, carried over so the config
#                                                     diff against arm d is geometry-clean)
#
# Held identical to the split arm: --split-heads --cls-weight 0.5 --mean-weight 0.5 and
# NO --pool-head, LoRA r32/alpha64, temperature 0.07, lr 1e-4, warmup 200, seed 0,
# pooling clsmean, proj_out_dim 512, heldout scanners GT450 S210 / stains HRH KR MY,
# owkin/phikon-v2, 1500 steps, ckpt/eval every 250 -- the same checkpoint grid the
# split/gem/ctrl arms were scored on, so the step-500 comparison is like-for-like.
#
# --- SOURCE PIN -----------------------------------------------------------------------
# Reads the REPAIRED pin at waiv-snapshots/finalgem-pinned (commit 386283a), which is a
# descendant of 106b437 and so contains the split-heads x grid unblock. The repair that
# matters: this pin HAS third_party/ and .venv/ symlinks. A pin built without them is why
# three runs came back unscored -- every eval follower died instantly inside
# PathoRobPaths.check(). Both symlinks and a live PathoRobPaths.check() through the pin
# were verified before submission, and the same three assertions are re-run below at job
# start, because a pin that was fine at submit time is not evidence about run time.
#
#   sbatch --account=max --qos=high scripts/splitgrid.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/finalgem-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

# --- pin preflight: the three things whose absence killed three prior runs -------------
[ -d "$PIN/src/waivphaet" ]      || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }
[ -d "$PIN/third_party/PathoROB" ] || { echo "pin missing third_party/PathoROB: $PIN" >&2; exit 3; }
[ -x "$PIN/.venv/bin/python" ]   || { echo "pin missing .venv/bin/python: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"

# SLURM_RESTART_COUNT keeps a requeued attempt out of the previous attempt's run dir, so
# preemption cannot silently mix two training trajectories (commit af243c2).
RUN_NAME="gemgrid-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

REF_RUN="runs/headcmp-split-380856"

MAX_STEPS="${WAIV_MAX_STEPS:-1500}"
CKPT_EVERY="${WAIV_CKPT_EVERY:-250}"
WORKERS="${WAIV_WORKERS:-20}"
LORA_RANK="${WAIV_LORA_RANK:-32}"
LORA_ALPHA="${WAIV_LORA_ALPHA:-64}"
LR="${WAIV_LR:-1e-4}"
TEMPERATURE="${WAIV_TEMPERATURE:-0.07}"
SEED="${WAIV_SEED:-0}"

# arm d's geometry, verbatim.
GRID_COND="${WAIV_GRID_CONDITIONS:-2}"
GRID_TILES="${WAIV_GRID_TILES:-900}"
GRID_CHUNK="${WAIV_GRID_CHUNK:-0}"
# Inert under --grid; carried from arm d so the config diff stays geometry-clean.
N_GROUPS="${WAIV_N_GROUPS:-4}"
GROUP_SIZE="${WAIV_GROUP_SIZE:-64}"

# --max-wait-s is an IDLE clock, not a cap on total follower runtime (commit fa1083c).
# Passed explicitly regardless of pin, so the follower cannot inherit a short default and
# quietly give up between two widely-spaced checkpoints.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-72000}"

HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem)   # GEM composition

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | restart ${SLURM_RESTART_COUNT:-0} ==="
echo "run       $RUN_NAME"
echo "geometry  C=$GRID_COND T=$GRID_TILES B=$((GRID_COND * GRID_TILES)) chunk=$GRID_CHUNK neg/row=$((GRID_TILES - 1))"
echo "head      ${HEAD_ARGS[*]}"
echo "steps     $MAX_STEPS  ckpt/eval $CKPT_EVERY  seed $SEED"
echo "ref       $REF_RUN"
echo "pin       $PIN  commit $(cat "$PIN/PINNED_COMMIT" 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Import THROUGH the pin, the way the job will -- proves PYTHONPATH wins over .venv AND
# that PathoROB resolves. This exact call is what the three dead followers failed.
"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import waivphaet
from waivphaet.eval.pathorob_adapter import PathoRobPaths
assert waivphaet.__file__.startswith(sys.argv[1]), f"pin LEAK, not using pin: {waivphaet.__file__}"
PathoRobPaths(root=Path(sys.argv[1]) / "third_party" / "PathoROB").check()
print(f"[pin] waivphaet={waivphaet.__file__}", flush=True)
print("[pin] PathoRobPaths.check() OK", flush=True)
EOF

# --- corpus preflight: all 91 verified, and pin the probe's condition set --------------
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
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE training (frozen backbone, pinned condition set) ----------
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1, trailing the trainer's checkpoints ------------------------
# No --pool-head / --infer-pool-head: this arm has no learned pool head, and the RI is
# scored through the standard clsmean eval protocol, same as split/gem/ctrl.
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
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log (idle max-wait ${EVAL_MAX_WAIT_S}s) ==="

# --- training on GPU 0 -----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean \
  --proj-out-dim 512 \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 200 \
  --grid --grid-conditions "$GRID_COND" --grid-tiles "$GRID_TILES" \
  --grid-forward-chunk "$GRID_CHUNK" \
  "${HEAD_ARGS[@]}" \
  --grad-checkpointing \
  --workers "$WORKERS" \
  --amp bfloat16 \
  --log-every 20 \
  --ckpt-every "$CKPT_EVERY" \
  --eval-every "$CKPT_EVERY" \
  --seed "$SEED" \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="

touch "$OUT_DIR/TRAIN_DONE"
wait $FOLLOWER || true
kill $SAMPLER 2>/dev/null || true

# --- comparability check ---------------------------------------------------------------
# Diffs this run's config.json against the SPLIT arm it is meant to extend. The only keys
# that may differ are out_dir and the sampler geometry. Anything else prints !! and means
# this is not a clean single-axis comparison.
echo "=== diff vs $REF_RUN (expect: out_dir + grid geometry ONLY) ==="
"$PY" - "$OUT_DIR" "$REPO/$REF_RUN" <<'EOF'
import json, sys
from pathlib import Path
mine = json.loads((Path(sys.argv[1]) / "config.json").read_text())
ref = json.loads((Path(sys.argv[2]) / "config.json").read_text())
def flat(d, p=""):
    f = {}
    for k, v in d.items():
        f.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else f.setdefault(p + k, v)
    return f
a, b = flat(mine), flat(ref)
allowed = {"out_dir", "grid", "grid_conditions", "grid_tiles", "grid_forward_chunk",
           "n_groups", "group_size"}
bad = []
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        tag = "ok  " if k in allowed else "!!  "
        print(f"{tag}{k}: {b.get(k, '<absent>')} -> {a.get(k, '<absent>')}")
        if k not in allowed:
            bad.append(k)
print("CLEAN: only geometry moved" if not bad else f"!! UNEXPECTED DIFFS: {bad}")

# The split has to be REAL, and the geometry has to be arm d's.
assert mine["split_heads"] is True, "split_heads did not stick"
assert mine["encoder"]["split_heads"] == ["cls", "mean"], mine["encoder"]["split_heads"]
assert mine["encoder"].get("pool_head", "mean") == "mean", "a learned pool head leaked in"
assert mine["grid"] is True and mine["grid_conditions"] == 2 and mine["grid_tiles"] == 1800
print("[config] split x grid geometry confirmed")
EOF

echo "=== history (loss, per-head loss, geometry witnesses) ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
h = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
for r in h[::10]:
    parts = " ".join(f"{k}={r[k]:.4f}" for k in ("loss_cls", "loss_mean") if k in r)
    print(f"step {r['step']:5d} loss={r['loss']:.4f} top1={r['top1']:.3f} {parts} "
          f"neg/row={r.get('negatives_per_anchor')} C={r.get('batch_distinct_conditions')}")
if h:
    assert {r["negatives_per_anchor"] for r in h if "negatives_per_anchor" in r} == {1799.0}, \
        "negatives_per_anchor drifted off 1799 -- geometry not what was intended"
    print("[history] negatives_per_anchor == 1799 for every logged step")
EOF

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF

echo "=== done $(date) -> $OUT_DIR (train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
