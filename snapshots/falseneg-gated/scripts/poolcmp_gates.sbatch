#!/bin/bash
#SBATCH --job-name=waiv-poolgates
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Validation gates for ALTERNATIVE TOKEN POOLINGS (--pool-head), all on ONE GPU in ONE job
# so every comparison is hardware-identical. Modelled on headcmp_gates.sbatch.
#
#   G1  bit-identical fallback -- 20 steps of the DEFAULT path (no --split-heads, no
#       --pool-head) run against a pristine pre-change snapshot and against the live tree;
#       the per-step loss sequences must match EXACTLY. Then the same for
#       `--split-heads` vs `--split-heads --pool-head mean`, which must ALSO be exact:
#       the refactor must not perturb the headcmp arm already running.
#   G3  THE CRITICAL GATE, on REAL phikon-v2 tokens from REAL PLISM tiles: the per-token
#       gradient spread of d(pool)/d(t_i) for each variant. mean must be at the float
#       noise floor (its gradient is the same vector for every token -- that is the
#       defect); gem/attn/lse must be orders of magnitude above it.
#   G4  fraction of token entries the clamp-GeM destroys, measured on those same real
#       tokens. The softplus/attn/lse variants must destroy nothing.
#   G6  20-step smoke per variant at the launch geometry (6 x 200 pair path, 2400
#       images/step): peak memory, img/s, per-head loss/top1, and the checkpoint artifacts
#       (including that embed_probe / extract_pathorob_features still load them).
#
# G2 and the hermetic halves of G3/G4/G5 run under `pytest tests/ -q` and are not here.
#
#   sbatch scripts/poolcmp_gates.sbatch /admin/home/ryan.kim/waiv_snapshots/poolcmp_pristine_XXXX

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
GATE_DIR="$REPO/runs/poolgates-${SLURM_JOB_ID:-local}"
mkdir -p "$GATE_DIR"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

"$PY" - "$PACKED_DIR" <<'EOF'
import json, sys
from pathlib import Path
man = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)}"
print(f"[preflight] {len(ok)} verified slides OK")
EOF

COMMON=(
  --packed-dir "$PACKED_DIR"
  --heldout-scanners GT450 S210
  --heldout-stains HRH KR MY
  --lora-rank 32 --lora-alpha 64
  --temperature 0.07 --lr 1e-4
  --max-steps 20 --warmup-steps 200
  --grad-checkpointing --amp bfloat16
  --log-every 1
  --eval-every 1000000000
  --seed 0
)
G1_ARGS=("${COMMON[@]}" --n-groups 2 --group-size 32 --workers 8 --ckpt-every 1000000000)

# --- G1a: the DEFAULT path, before vs after -------------------------------------------
echo "=== G1a BEFORE: pristine snapshot at $SNAP ==="
PYTHONPATH="$SNAP/src" "$PY" - "$SNAP" <<'EOF'
import inspect, sys
import waivphaet, waivphaet.models.encoder
snap = sys.argv[1]
assert waivphaet.__file__.startswith(snap), (
    f"G1 BEFORE is importing {waivphaet.__file__}, not the snapshot under {snap}: "
    "PYTHONPATH lost to the editable-install .pth and this gate would be vacuous"
)
try:
    import waivphaet.models.pooling  # noqa: F401
    raise AssertionError("the snapshot already has waivphaet.models.pooling -- not pristine")
except ImportError:
    pass
assert "pool_head" not in inspect.getsource(waivphaet.models.encoder), (
    "the snapshot encoder already knows about pool_head -- it is not pristine"
)
print(f"[G1] before-arm imports {waivphaet.__file__} and has NO pool-head code: pristine OK")
EOF
if grep -q -- "--pool-head" "$SNAP/scripts/train_lora.py"; then
  echo "G1 FAILED: snapshot train_lora.py already carries --pool-head" >&2; exit 1
fi

PYTHONPATH="$SNAP/src" "$PY" "$SNAP/scripts/train_lora.py" \
  --out-dir "$GATE_DIR/g1a_before" "${G1_ARGS[@]}"
echo "=== G1a AFTER: live tree ==="
"$PY" "$REPO/scripts/train_lora.py" --out-dir "$GATE_DIR/g1a_after" "${G1_ARGS[@]}"

# --- G1b: the SPLIT-HEADS path (the arm already running), before vs after --------------
# The refactor moved forward_split's body; if that perturbed the running headcmp arm by
# one ULP the whole comparison it is producing becomes non-reproducible.
echo "=== G1b BEFORE: --split-heads on the snapshot ==="
PYTHONPATH="$SNAP/src" "$PY" "$SNAP/scripts/train_lora.py" \
  --out-dir "$GATE_DIR/g1b_before" "${G1_ARGS[@]}" --split-heads
echo "=== G1b AFTER: --split-heads --pool-head mean on the live tree ==="
"$PY" "$REPO/scripts/train_lora.py" \
  --out-dir "$GATE_DIR/g1b_after" "${G1_ARGS[@]}" --split-heads --pool-head mean

echo "=== G1 COMPARISON ==="
"$PY" - "$GATE_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for tag, keys in (("g1a", ("loss",)), ("g1b", ("loss", "loss_cls", "loss_mean"))):
    hb = json.loads((d / f"{tag}_before" / "history.json").read_text())
    ha = json.loads((d / f"{tag}_after" / "history.json").read_text())
    assert len(hb) == 20, f"{tag}: expected 20 logged steps, got {len(hb)}"
    for k in keys:
        b, a = [r[k] for r in hb], [r[k] for r in ha]
        assert b == a, f"{tag} FAILED on {k}: " + "; ".join(
            f"step {i+1} {x!r} != {y!r}" for i, (x, y) in enumerate(zip(b, a)) if x != y
        )
    print(f"{tag} PASS: 20/20 per-step {list(keys)} bit-identical")
rec = json.loads((d / "g1a_after" / "history.json").read_text())[0]
assert not any(k.startswith(("loss_cls", "loss_mean", "split_input", "pool_")) for k in rec), \
    f"pool/split keys leaked into the default path: {sorted(rec)}"
cfg = json.loads((d / "g1a_after" / "config.json").read_text())
assert cfg["split_heads"] is False and cfg["pool_head"] == "mean"
# the mean arm must not have written a pooling artifact either
assert json.loads((d / "g1b_after" / "config.json").read_text())["pool_head"] == "mean"
print("G1 PASS")
EOF

# --- G3 + G4 on REAL phikon-v2 tokens from REAL PLISM tiles ----------------------------
echo "=== G3/G4: token-gradient spread + clamp zeroing on a real batch ==="
"$PY" - "$PACKED_DIR" "$GATE_DIR" <<'EOF'
import json, sys, torch
from pathlib import Path
from waivphaet.data.conditions import available_conditions, make_split
from waivphaet.data.pairs import build_pair_loader
from waivphaet.data.repack import present_filenames
from waivphaet.models.encoder import build_encoder
from waivphaet.models.pooling import (
    POOL_HEAD_NAMES, TOKEN_DEPENDENT_POOLS, build_pool_head, token_gradient_spread,
)

packed, out = Path(sys.argv[1]), Path(sys.argv[2])
split = make_split(["GT450", "S210"], ["HRH", "KR", "MY"])
conds = available_conditions(split.train, present_filenames(packed))
loader = build_pair_loader(packed, conditions=conds, n_groups=2, group_size=16,
                           batches_per_epoch=1, num_workers=4, seed=0)
batch = next(iter(loader))

model = build_encoder(backbone="owkin/phikon-v2", use_lora=False).eval().cuda()
with torch.no_grad():
    tokens = model.tokens(batch["anchor"].cuda()).float()
patches = tokens[:, model.num_prefix_tokens:, :].cpu()
print(f"[G3] real phikon-v2 tokens: {tuple(tokens.shape)}, num_prefix_tokens="
      f"{model.num_prefix_tokens} -> patches {tuple(patches.shape)}")
print(f"[G3] fraction of token entries that are NEGATIVE: "
      f"{(patches < 0).float().mean():.4f}  (this is the sign problem)")

torch.manual_seed(0)
rows = {}
for name in POOL_HEAD_NAMES:
    pool = build_pool_head(name, patches.shape[-1])
    st = token_gradient_spread(pool, patches)
    with torch.no_grad():
        pool(patches)
    zf = pool.extra_metrics().get("pool_zero_fraction", 0.0)
    rows[name] = {**st, "zero_fraction": zf, **pool.extra_metrics()}
    print(f"  {name:10s} spread mean={st['spread_mean']:.4e} "
          f"[{st['spread_min']:.4e}, {st['spread_max']:.4e}]  zeroed={zf:.4f}")

(out / "g3_spread.json").write_text(json.dumps(rows, indent=2))

floor = rows["mean"]["spread_max"]
assert floor < 1e-5, f"G3: mean's spread {floor} is not at the float noise floor"
for name in TOKEN_DEPENDENT_POOLS:
    s = rows[name]["spread_min"]
    assert s > 1e-2, f"G3 FAILED: {name} spread {s:.3e} is not token-dependent"
    print(f"  G3 {name}: {s / max(floor, 1e-12):.1f}x above mean's noise floor")
assert rows["gem_clamp"]["zero_fraction"] > 0.3, (
    f"G4: clamp-GeM zeroed only {rows['gem_clamp']['zero_fraction']:.4f} -- if the real "
    "tokens are not ~half negative, the sign argument needs restating, not asserting"
)
for name in ("gem", "attn", "lse"):
    assert rows[name]["zero_fraction"] == 0.0, (name, rows[name])
print("G3/G4 PASS")
EOF

# --- G6: 20-step smoke per variant at the LAUNCH geometry ------------------------------
for POOL in mean gem attn lse gem_clamp; do
  echo "=== G6 pool_head=$POOL ==="
  "$PY" "$REPO/scripts/train_lora.py" \
    --out-dir "$GATE_DIR/g6_${POOL}" "${COMMON[@]}" \
    --n-groups 6 --group-size 200 --workers 20 --ckpt-every 20 \
    --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head "$POOL"
done

echo "=== G6 SUMMARY ==="
"$PY" - "$GATE_DIR" <<'EOF'
import json, sys, torch
from pathlib import Path
d = Path(sys.argv[1])
for pool in ("mean", "gem", "attn", "lse", "gem_clamp"):
    run = d / f"g6_{pool}"
    s = json.loads((run / "summary.json").read_text())
    h = json.loads((run / "history.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["pool_head"] == pool and cfg["encoder"]["pool_head"] == pool
    tail = h[5:]                       # step 1 carries dataloader warm-up
    tps = sum(r["tiles_per_s"] for r in tail) / len(tail)
    extra = " ".join(
        f"{k}={h[-1][k]:.4f}" for k in sorted(h[-1]) if k.startswith("pool_")
    )
    print(
        f"{pool:10s} peak_alloc={s['peak_alloc_gib']:.2f} GiB "
        f"reserved={s['peak_reserved_gib']:.2f}/{s['device_total_gib']:.1f} "
        f"| {tps:.0f} img/s | {s['elapsed_s']/s['step']:.2f} s/step "
        f"| cls {h[-1]['loss_cls']:.4f}/{h[-1]['top1_cls']:.3f} "
        f"| mean {h[-1]['loss_mean']:.4f}/{h[-1]['top1_mean']:.3f} "
        f"| relD {h[-1]['split_input_rel_distance']:.4f} | {extra}"
    )
    ck = run / "step_0000020"
    man = json.loads((ck / "projector_heads.json").read_text())
    assert man["pool_head"] == pool
    if pool == "mean":
        assert not (ck / "pool_head.pt").exists(), "the mean arm must write no new artifact"
    else:
        assert (ck / "pool_head.pt").exists(), f"{pool}: pool_head.pt missing"
    sd = torch.load(ck / "projector.pt", map_location="cpu")
    assert sd["net.0.weight"].shape[1] == 1024, sd["net.0.weight"].shape
    print(f"  artifacts: {sorted(p.name for p in ck.iterdir())}")
EOF

# --- G6b: the RI followers must tolerate the new checkpoints ---------------------------
# A projector-width mismatch here previously killed every RI follower (RESULTS 9), so the
# readers are exercised against a real pool-head checkpoint rather than assumed to cope.
echo "=== G6b: embed_probe.load_adapter / extract_pathorob_features.build_model ==="
"$PY" - "$GATE_DIR" <<'EOF'
import importlib.util, sys, torch
from pathlib import Path
d = Path(sys.argv[1])
ck = d / "g6_gem" / "step_0000020"
assert (ck / "pool_head.pt").exists()
from waivphaet.models.encoder import build_encoder
for rel in ("scripts/embed_probe.py", "scripts/extract_pathorob_features.py"):
    spec = importlib.util.spec_from_file_location("_m", rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if rel.endswith("embed_probe.py"):
        # load_adapter(model, ckpt) mutates in place -- it is the RI follower's entry point
        # rank must match the checkpoint (the gate arms run --lora-rank 32), exactly as the
        # real follower resolves it from its own CLI
        model = build_encoder(backbone="owkin/phikon-v2", pooling="clsmean",
                              lora_rank=32, lora_alpha=64)
        m.load_adapter(model, ck)
    else:
        model = m.build_model(None, "clsmean", adapter=ck, lora_rank=32, lora_alpha=64)
    model = model.eval()
    with torch.inference_mode():
        emb = model.embed(torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8))
    assert emb.shape == (2, 2048), (rel, emb.shape)
    assert getattr(model, "pool_head", None) is None, (
        f"{rel} built an encoder carrying the TRAINING-TIME pooling; eval pooling is a "
        "protocol constant"
    )
    print(f"  {rel}: loaded a --pool-head gem checkpoint, embed() -> {tuple(emb.shape)}")
EOF
echo "=== ALL GPU GATES DONE $(date) -> $GATE_DIR ==="
