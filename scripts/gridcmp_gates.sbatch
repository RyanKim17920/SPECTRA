#!/bin/bash
#SBATCH --job-name=waiv-gridgates
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Validation gates for the GRID sampler, all on ONE GPU in ONE job so the comparisons are
# hardware-identical:
#
#   G1  bit-identical fallback -- 20 steps of the EXISTING pair path, run twice: once
#       against a pristine snapshot of the source taken before any grid edits, once
#       against the live tree. The per-step loss sequences must match EXACTLY. Same node,
#       same GPU, back to back, because "bit-identical" across two different H100s is a
#       weaker claim than the one this gate is supposed to make.
#   G6  20-step smoke on BOTH grid arms: no OOM, plus peak memory and img/s.
#
# The snapshot path is passed in as $1 (PYTHONPATH + its own copy of train_lora.py).

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
SNAP="$1"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# The grid step keeps ~70 GiB live and then asks for one more large recompute
# buffer; the default caching allocator stranded 5.56 GiB in reserved-but-unused
# segments and OOMed. expandable_segments hands that back.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"
GATE_DIR="$REPO/runs/gridgates-${SLURM_JOB_ID:-local}"
mkdir -p "$GATE_DIR"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- preflight: 91 verified slides, or stop -------------------------------------------
"$PY" - "$PACKED_DIR" <<'EOF'
import json, sys
from pathlib import Path
man = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)}"
print(f"[preflight] {len(ok)} verified slides OK")
EOF

# --- G1: the SAME 20-step pair-sampler command, before vs after ------------------------
# Deliberately small (2 x 32) -- this gate is about whether the numbers moved, not about
# throughput, and a small batch makes it cheap to run twice.
G1_ARGS=(
  --packed-dir "$PACKED_DIR"
  --heldout-scanners GT450 S210
  --heldout-stains HRH KR MY
  --lora-rank 32 --lora-alpha 64
  --temperature 0.07 --lr 1e-4
  --max-steps 20 --warmup-steps 200
  --n-groups 2 --group-size 32
  --grad-checkpointing --amp bfloat16
  --workers 8 --log-every 1
  --ckpt-every 1000000000 --eval-every 1000000000
  --seed 0
)

echo "=== G1 BEFORE: pristine snapshot at $SNAP ==="
# The venv installs waivphaet via a .pth pointing at the LIVE tree, so prove PYTHONPATH
# actually wins before trusting the "before" arm -- otherwise this gate silently compares
# the new code against itself and passes for the wrong reason.
PYTHONPATH="$SNAP/src" "$PY" - "$SNAP" <<'EOF'
import sys
import waivphaet, waivphaet.data.pairs
snap = sys.argv[1]
assert waivphaet.__file__.startswith(snap), (
    f"G1 BEFORE is importing {waivphaet.__file__}, not the snapshot under {snap}: "
    "PYTHONPATH lost to the editable-install .pth and this gate would be vacuous"
)
try:
    import waivphaet.data.grid  # noqa: F401
except ModuleNotFoundError:
    print(f"[G1] before-arm imports {waivphaet.__file__} and has NO grid module: pristine OK")
else:
    raise AssertionError("the snapshot contains waivphaet.data.grid -- it is not pristine")
EOF
PYTHONPATH="$SNAP/src" "$PY" "$SNAP/scripts/train_lora.py" \
  --out-dir "$GATE_DIR/g1_before" "${G1_ARGS[@]}"

echo "=== G1 AFTER: live tree at $REPO ==="
"$PY" "$REPO/scripts/train_lora.py" \
  --out-dir "$GATE_DIR/g1_after" "${G1_ARGS[@]}"

echo "=== G1 COMPARISON ==="
"$PY" - "$GATE_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
before = [r["loss"] for r in json.loads((d / "g1_before" / "history.json").read_text())]
after = [r["loss"] for r in json.loads((d / "g1_after" / "history.json").read_text())]
(d / "g1_losses.json").write_text(json.dumps({"before": before, "after": after}, indent=2))
print("before:", json.dumps(before))
print("after: ", json.dumps(after))
assert len(before) == 20, f"expected 20 logged steps, got {len(before)}"
assert before == after, (
    "G1 FAILED: the non-grid path was perturbed.\n"
    + "\n".join(
        f"  step {i+1}: before={b!r} after={a!r}"
        for i, (b, a) in enumerate(zip(before, after)) if b != a
    )
)
print(f"G1 PASS: {len(before)}/{len(before)} per-step losses bit-identical")
EOF

# --- G6: 20-step smoke on both grid arms ----------------------------------------------
for ARM in "24 100" "49 49"; do
  set -- $ARM
  C=$1; T=$2
  echo "=== G6 GRID${C}: C=$C T=$T ($((C*T)) images/step) ==="
  "$PY" "$REPO/scripts/train_lora.py" \
    --out-dir "$GATE_DIR/g6_grid${C}" \
    --packed-dir "$PACKED_DIR" \
    --heldout-scanners GT450 S210 \
    --heldout-stains HRH KR MY \
    --lora-rank 32 --lora-alpha 64 \
    --temperature 0.07 --lr 1e-4 \
    --max-steps 20 --warmup-steps 200 \
    --grid --grid-conditions "$C" --grid-tiles "$T" \
    --grid-forward-chunk 600 \
    --grad-checkpointing --amp bfloat16 \
    --workers 20 --log-every 1 \
    --ckpt-every 1000000000 --eval-every 1000000000 \
    --seed 0
done

echo "=== G6 SUMMARY ==="
"$PY" - "$GATE_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for name in ("g6_grid24", "g6_grid49"):
    s = json.loads((d / name / "summary.json").read_text())
    h = json.loads((d / name / "history.json").read_text())
    cfg = json.loads((d / name / "config.json").read_text())
    # step 1 carries dataloader warm-up; quote the steady-state window instead
    tail = h[5:]
    tps = sum(r["tiles_per_s"] for r in tail) / len(tail)
    print(
        f"{name}: grid={cfg['grid']} C={cfg['grid_conditions']} T={cfg['grid_tiles']} "
        f"| peak_alloc={s['peak_alloc_gib']:.2f} GiB peak_reserved={s['peak_reserved_gib']:.2f} "
        f"of {s['device_total_gib']:.1f} GiB | {tps:.0f} img/s "
        f"| {s['elapsed_s']/s['step']:.2f} s/step "
        f"| neg/row={h[-1]['batch_negatives_per_anchor']:.0f} rows={h[-1]['batch_n_rows']:.0f} "
        f"| loss {h[0]['loss']:.4f} -> {h[-1]['loss']:.4f} top1 {h[-1]['top1']:.3f}"
    )
EOF
echo "=== ALL GATES DONE $(date) -> $GATE_DIR ==="
