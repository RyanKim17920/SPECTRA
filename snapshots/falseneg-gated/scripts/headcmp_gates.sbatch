#!/bin/bash
#SBATCH --job-name=waiv-headgates
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Validation gates for SPLIT LOSS HEADS, all on ONE GPU in ONE job so the comparisons are
# hardware-identical (same rule as gridcmp_gates.sbatch, which this is modelled on):
#
#   G1  bit-identical fallback -- 20 steps of the EXISTING single-concat-head path, run
#       twice: once against a pristine snapshot of the source taken BEFORE any split-head
#       edits, once against the live tree. The per-step loss sequences must match EXACTLY.
#       Same node, same GPU, back to back: "bit-identical across two different H100s" is a
#       weaker claim than the one this gate is supposed to make.
#   G6  20-step smoke on ALL THREE arms at the LAUNCH geometry (6 x 200 pair path,
#       2400 images/step): no OOM, plus peak memory and img/s. Reference for the
#       single-head control on this hardware is 65.0 GiB of 79.2 and ~409 img/s.
#
# G2 (pytest), G3 (must-fail tests), G4 (shapes) and G5 (scale neutrality) are hermetic
# CPU checks and run under `pytest tests/ -q`; they do not need a GPU and are not here.
#
# The snapshot path is passed in as $1 (PYTHONPATH + its own copy of train_lora.py).
#
#   sbatch scripts/headcmp_gates.sbatch /admin/home/ryan.kim/waiv_snapshots/headcmp_pristine_XXXX

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"
GATE_DIR="$REPO/runs/headgates-${SLURM_JOB_ID:-local}"
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

# --- G1: the SAME 20-step single-head command, before vs after -------------------------
# Deliberately small (2 x 32): this gate is about whether the numbers moved, not about
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
import inspect, sys
import waivphaet, waivphaet.models.encoder, waivphaet.train.contrastive
snap = sys.argv[1]
assert waivphaet.__file__.startswith(snap), (
    f"G1 BEFORE is importing {waivphaet.__file__}, not the snapshot under {snap}: "
    "PYTHONPATH lost to the editable-install .pth and this gate would be vacuous"
)
assert not hasattr(waivphaet.train.contrastive, "split_head_info_nce"), (
    "the snapshot already has split_head_info_nce -- it is not pristine"
)
assert "split_heads" not in inspect.getsource(waivphaet.models.encoder.WaivEncoder), (
    "the snapshot encoder already knows about split_heads -- it is not pristine"
)
print(f"[G1] before-arm imports {waivphaet.__file__} and has NO split-head code: pristine OK")
EOF
if grep -q -- "--split-heads" "$SNAP/scripts/train_lora.py"; then
  echo "G1 FAILED: snapshot train_lora.py already carries --split-heads" >&2
  exit 1
fi
echo "[G1] snapshot scripts/train_lora.py has no --split-heads: pristine OK"

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
    "G1 FAILED: the single-head path was perturbed.\n"
    + "\n".join(
        f"  step {i+1}: before={b!r} after={a!r}"
        for i, (b, a) in enumerate(zip(before, after)) if b != a
    )
)
# and no split-head key leaked into a run that did not ask for one
after_rec = json.loads((d / "g1_after" / "history.json").read_text())[0]
assert not any(k.startswith(("loss_cls", "loss_mean", "split_input")) for k in after_rec), \
    f"split-head keys leaked into the default path: {sorted(after_rec)}"
cfg = json.loads((d / "g1_after" / "config.json").read_text())
assert cfg["split_heads"] is False and cfg["encoder"]["split_heads"] == []
print(f"G1 PASS: {len(before)}/{len(before)} per-step losses bit-identical")
EOF

# --- G6: 20-step smoke on all three arms, AT THE LAUNCH GEOMETRY -----------------------
# 6 x 200 pair path = 1200 anchors + 1200 positives = 2400 images/step, identical to the
# CTRL arm of the running gridcmp experiment (65.0 GiB of 79.2, ~409 img/s).
for ARM in "split 0.5 0.5" "clsonly 1.0 0.0" "meanonly 0.0 1.0"; do
  set -- $ARM
  NAME=$1; WCLS=$2; WMEAN=$3
  echo "=== G6 ${NAME}: --cls-weight $WCLS --mean-weight $WMEAN ==="
  "$PY" "$REPO/scripts/train_lora.py" \
    --out-dir "$GATE_DIR/g6_${NAME}" \
    --packed-dir "$PACKED_DIR" \
    --heldout-scanners GT450 S210 \
    --heldout-stains HRH KR MY \
    --lora-rank 32 --lora-alpha 64 \
    --temperature 0.07 --lr 1e-4 \
    --max-steps 20 --warmup-steps 200 \
    --n-groups 6 --group-size 200 \
    --split-heads --cls-weight "$WCLS" --mean-weight "$WMEAN" \
    --grad-checkpointing --amp bfloat16 \
    --workers 20 --log-every 1 \
    --ckpt-every 20 --eval-every 1000000000 \
    --seed 0
done

echo "=== G6 SUMMARY ==="
"$PY" - "$GATE_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for name in ("g6_split", "g6_clsonly", "g6_meanonly"):
    s = json.loads((d / name / "summary.json").read_text())
    h = json.loads((d / name / "history.json").read_text())
    cfg = json.loads((d / name / "config.json").read_text())
    tail = h[5:]                      # step 1 carries dataloader warm-up
    tps = sum(r["tiles_per_s"] for r in tail) / len(tail)
    heads = cfg["encoder"]["split_heads"]
    per_head = " ".join(
        f"{k}={h[-1][f'loss_{k}']:.4f}/top1={h[-1][f'top1_{k}']:.3f}" for k in heads
    )
    print(
        f"{name}: heads={heads} w=({cfg['cls_weight']},{cfg['mean_weight']}) "
        f"| peak_alloc={s['peak_alloc_gib']:.2f} GiB "
        f"peak_reserved={s['peak_reserved_gib']:.2f} of {s['device_total_gib']:.1f} GiB "
        f"| {tps:.0f} img/s | {s['elapsed_s']/s['step']:.2f} s/step "
        f"| loss {h[0]['loss']:.4f} -> {h[-1]['loss']:.4f} | {per_head}"
    )
    # the checkpoint artifact must be faithful AND round-trip
    ck = d / name / "step_0000020"
    man = json.loads((ck / "projector_heads.json").read_text())
    assert man["heads"] == heads, (man, heads)
    for k in heads:
        assert (ck / f"projector_{k}.pt").exists(), f"{name}: missing projector_{k}.pt"
    # projector.pt must exist (extract_pathorob_features.build_model loads it
    # unconditionally on the adapter path) and must be a one-pool, hidden-d head
    import torch
    sd = torch.load(ck / "projector.pt", map_location="cpu")
    assert sd["net.0.weight"].shape[1] == 1024, sd["net.0.weight"].shape
    print(f"  artifacts OK: {sorted(p.name for p in ck.iterdir())}")
    # the split arm must have run on genuinely different inputs, every logged step
    if len(heads) == 2:
        rel = [r["split_input_rel_distance"] for r in h]
        assert min(rel) > 1e-3, rel
        print(f"  split_input_rel_distance: min={min(rel):.4f} max={max(rel):.4f} "
              f"cos={h[-1]['split_input_cosine']:.4f}")
EOF
echo "=== ALL GPU GATES DONE $(date) -> $GATE_DIR ==="
