#!/bin/bash
#SBATCH --job-name=waiv-finalgem
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
# ======================================================================================
# THE FINAL SYSTEM: the GEM pooling head on all three backbones.
# ======================================================================================
#
# WHAT THIS IS. The 3-seed study settled the head/pooling question on the only metric with
# demonstrated dynamic range (PathoROB avg_robustness_index): paired by seed, GEM beat the
# cls/mean SPLIT arm at EVERY checkpoint, 15/15 seed-paired deltas positive (step 250
# +0.00229 t=31.8 ... step 1500 +0.00132 t=32.7), and both beat the single-concat CTRL
# decisively (GEM +0.0163 at step 500, t=6.9). Mean GEM RI peaks at step 500
# (250 .8163 | 500 .8226 | 750 .8212 | 1000 .8205 | 1500 .8181).
#
#   WAIV_ARM=phikon    owkin/phikon-v2   4000 steps, ckpt/eval every 500, 20 workers
#   WAIV_ARM=midnight  kaiko-ai/midnight 1500 steps, ckpt/eval every 250, 16 workers
#   WAIV_ARM=virchow2  paige-ai/Virchow2 1500 steps, ckpt/eval every 250, 20 workers
#
# --- WHAT IT REPRODUCES, AND WHAT IT DELIBERATELY MOVES -------------------------------
# The baseline is the PUBLISHED RECONSTRUCTION run for each backbone, NOT the mechanism
# study's geometry arms. Every knob below was read back out of those three config.json
# files rather than retyped from memory or lifted from gridcmp3:
#
#   phikon-v2  runs/waiv-real-369043      (published FT checkpoint step_0001000)
#   midnight   runs/waiv-midnight-369159  (published FT checkpoint step_0000500)
#   virchow2   runs/waiv-virchow2-375367  (published FT checkpoint step_0000250)
#
# Shared by all three published runs, and reproduced here EXACTLY:
#   packed_dir /data/plism/repacked | lr 1e-4 | weight_decay 0.05 (default) |
#   warmup 200 | temperature 0.07 | symmetric false | grad_accum 1 | grad_clip 1.0 |
#   n_groups 2 x group_size 192 (191 negatives, 768 images/step) | seed 0 |
#   amp bfloat16 | log_every 20 | eval_heldout true | retention_kl_weight 0.0 |
#   encoder: lora_rank 32 / lora_alpha 64 / pooling clsmean / proj_out_dim 512 /
#            grad_checkpointing true
#   heldout: --heldout-scanners GT450 S210 --heldout-stains HRH KR MY
#            (not recorded in config.json; taken from scripts/train_real.sbatch, which is
#            the script those three runs were submitted through)
#
# NOTE THE BATCH GEOMETRY: 2 x 192, the RECONSTRUCTION geometry. It is NOT gridcmp3's
# 6 x 200 pair path and NOT arm F's wide-batch grid -- arm F's 9600 img/step would need
# ~778 GiB of activations on midnight and is simply not reachable. Measured cost at
# 2 x 192 with gradient checkpointing on one 80 GiB H100:
#   phikon-v2 0.0283 GiB/img, 1.91 s/step   -> 4000 steps ~ 2.1 h
#   midnight  0.055  GiB/img, 4.57 s/step   -> 1500 steps ~ 1.9 h
#   virchow2  0.081  GiB/img                -> 1500 steps ~ 2.2 h
# The 10 h wall is deliberately generous: a prior arm lost its tail to a self-imposed
# 10 h limit, and here the eval follower still has to drain after training exits.
#
# THE ONE THING THAT MOVES, on all three arms identically:
#   --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
#
# WHY THE SPLIT-HEAD FLAGS COME ALONG. They are not a second change smuggled in; they are
# what --pool-head gem MEANS. The alternative poolings are TRAINING-TIME loss heads:
# train_lora.py hard-rejects --pool-head without --split-heads, and WaivEncoder raises
# ValueError unless split_heads includes 'mean', because with the single concat projector
# the pooled vector IS _pool()'s clsmean, i.e. the eval-time protocol constant. The seed
# study's winning arm was exactly this flag triple, so this is the winner as measured, not
# a reconstruction of it. 0.5/0.5 is scale neutrality: at 1.0/1.0 the total gradient
# magnitude would be ~2x the concat baseline's, i.e. a silent LR change.
#
# --- --infer-pool-head: NOT PASSED, AND IT CANNOT BE ----------------------------------
# train_lora.py exposes NO --infer-pool-head flag and never passes infer_pool_head into
# build_encoder (checked at commit 386283a). EncoderConfig.infer_pool_head exists and
# defaults False; it is an EVAL-side switch, consumed by extract_pathorob_features.py and
# run_hest.py, which rebuild the encoder with infer_pool_head=True and restore the trained
# head from the checkpoint's pool_head.pt. So at TRAIN time it is not a choice this script
# can make, and the exported embedding stays plain clsmean.
#
# That is also the right answer scientifically. Every RI number in the seed study -- the
# entire basis for picking GEM -- was scored through the standard clsmean eval head, with
# the learned pooling reaching the readout only indirectly, via the LoRA weights it shaped.
# Turning infer_pool_head on here would change WHAT THE RUN EXPORTS, so these three runs
# would no longer be on the same measurement axis as the decision that selected them, nor
# as the published reconstruction runs they are meant to be compared against. It also
# hard-conflicts with the retention term (contrastive.py raises when
# infer_pool_head=True and retention_kl_weight>0); that conflict is inert here only
# because retention_kl_weight is 0.0, which the published configs also have.
# Scoring these checkpoints WITH the learned pool at inference is a separate, additive
# experiment: re-run extract_pathorob_features.py / run_hest.py with --pool-head gem
# against the SAME checkpoints. It costs no training budget and does not perturb this run.
#
# --- SOURCE PIN -----------------------------------------------------------------------
# train_real.sbatch / headcmp.sbatch / poolcmp.sbatch pin NOTHING: they read scripts/ and
# src/ from the live working tree at job START time. With three jobs that will start at
# three different times in a busy queue, that is not acceptable for the final runs -- an
# edit landing between job 1 and job 3 would make them incomparable with nothing in the log
# saying so. So all three read a pin taken from commit 386283a and verified by md5 against
# the live tree, which INCLUDES the pool-head inference fixes (106b437, 6d92ebf) and the
# --max-wait-s idle-clock fix (fa1083c). The pin is read-only; do NOT edit it.
#
#   WAIV_ARM=phikon sbatch --account=max --qos=high \
#     --job-name=waiv-finalgem-phikon scripts/finalgem.sbatch

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

PY="$REPO/.venv/bin/python"
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/finalgem-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PIN/src/waivphaet" ] || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }

# The corpus moved from /data/ryan.kim/plism to /data/plism on 2026-08-09.
PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
ARM="${WAIV_ARM:?set WAIV_ARM to phikon | midnight | virchow2}"
# SLURM_RESTART_COUNT keeps a requeued attempt out of the previous attempt's run dir, so a
# preemption cannot silently mix two training trajectories (commit af243c2).
RUN_NAME="finalgem-${ARM}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

# --- per-backbone knobs, read out of the published reconstruction configs -------------
# BACKBONE_ARGS is threaded into the BEFORE probe and the eval follower as well as the
# trainer: a probe run on a different backbone than the training run measures nothing.
case "$ARM" in
  phikon)
    BACKBONE=""                 # unset => train_lora.py's default owkin/phikon-v2
    MAX_STEPS=4000; CKPT_EVERY=500; WORKERS=20; REF_RUN="runs/waiv-real-369043" ;;
  midnight)
    BACKBONE="kaiko-ai/midnight"
    MAX_STEPS=1500; CKPT_EVERY=250; WORKERS=16; REF_RUN="runs/waiv-midnight-369159" ;;
  virchow2)
    BACKBONE="paige-ai/Virchow2"
    MAX_STEPS=1500; CKPT_EVERY=250; WORKERS=20; REF_RUN="runs/waiv-virchow2-375367" ;;
  *) echo "unknown WAIV_ARM=$ARM (want phikon | midnight | virchow2)" >&2; exit 2 ;;
esac
BACKBONE_ARGS=()
if [ -n "$BACKBONE" ]; then BACKBONE_ARGS=(--backbone "$BACKBONE"); fi

LORA_RANK="${WAIV_LORA_RANK:-32}"
LORA_ALPHA="${WAIV_LORA_ALPHA:-64}"
N_GROUPS="${WAIV_N_GROUPS:-2}"
GROUP_SIZE="${WAIV_GROUP_SIZE:-192}"
LR="${WAIV_LR:-1e-4}"
TEMPERATURE="${WAIV_TEMPERATURE:-0.07}"
SEED="${WAIV_SEED:-0}"

# The pin POSTDATES fa1083c, so --max-wait-s here is a true IDLE timeout (the deadline is
# reset by progress), not a cap on total follower runtime. Passed explicitly anyway: the
# 8 h default is what stranded arm F's step_0000750 unscored under the old semantics, and
# an explicit 20 h is unreachable inside a 10 h wall under EITHER reading -- the follower
# then only ever exits via --stop-file (TRAIN_DONE) or a genuinely wedged trainer.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-72000}"

# --- the ONE axis that differs from the published reconstruction runs ------------------
HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem)

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | restart ${SLURM_RESTART_COUNT:-0} | run $RUN_NAME ==="
echo "=== arm $ARM | backbone ${BACKBONE:-<default owkin/phikon-v2>} | steps $MAX_STEPS | ckpt/eval every $CKPT_EVERY ==="
echo "=== heads ${HEAD_ARGS[*]} | batch ${N_GROUPS} x ${GROUP_SIZE} | seed $SEED | ref $REF_RUN ==="
echo "=== pin $PIN ($(cat "$PIN/PINNED_COMMIT" 2>/dev/null || echo unknown)) ==="
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
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base backbone, pinned condition set) --------------------
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  "${BACKBONE_ARGS[@]}" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1 -----------------------------------------------------------
# NO --pooling. eval_checkpoints.py defaults to clsmean, which is the protocol constant
# (PathoROB's reference row is phikonv2_clsmean), and NO --pool-head: see the
# --infer-pool-head note in the header. On a split-head run the follower will print
#   [build_model] skipping projector: trained with a 1024-d input, evaluating at 2048-d
# which is EXPECTED and benign -- every eval path reads model.embed(), the projector is
# training-only, and the LoRA backbone weights are what carry the fine-tune.
rm -f "$OUT_DIR/TRAIN_DONE"
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  "${BACKBONE_ARGS[@]}" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log (max-wait ${EVAL_MAX_WAIT_S}s) ==="

# --- training on GPU 0 ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  "${BACKBONE_ARGS[@]}" \
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 200 \
  --n-groups "$N_GROUPS" \
  --group-size "$GROUP_SIZE" \
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

# --- comparability check, printed rather than trusted ----------------------------------
# Diffs this run's config.json against the published reconstruction run it reproduces. The
# ONLY keys that may differ are out_dir and the pooling-head block. Anything else printed
# with !! means this is not a clean single-axis comparison.
echo "=== config diff vs $REF_RUN (expect: out_dir + the pool-head block ONLY) ==="
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
allowed = {"out_dir", "split_heads", "cls_weight", "mean_weight", "pool_head",
           "encoder.split_heads", "encoder.pool_head"}
bad = []
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        tag = "ok " if k in allowed else "!! "
        print(f"{tag}{k}: {b.get(k, '<absent>')} -> {a.get(k, '<absent>')}")
        if k not in allowed:
            bad.append(k)
print("[comparability] CLEAN -- differs in the pool head (and out_dir) only" if not bad
      else f"[comparability] CHECK -- other keys moved: {bad}")
EOF

# --- collapse check + RI curve ---------------------------------------------------------
# pool_gem_p is the COLLAPSE CHECK: still at its init (p=3.000) at the last step means the
# GEM head pooled like the plain mean all along and any RI delta is noise, not mechanism.
echo "=== per-head loss curve + learned pooling state ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
h = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
heads = cfg.get("encoder", {}).get("split_heads", [])
POOL = ("pool_gem_p", "pool_zero_fraction")
present = [k for k in POOL if h and any(k in r for r in h)]
for r in h[::10]:
    parts = " ".join(f"{k}={r.get('loss_' + k, float('nan')):.4f}" for k in heads)
    pool = " ".join(f"{k}={r[k]:.4f}" for k in present if k in r)
    print(f"step {r['step']:5d} loss={r['loss']:.4f} top1={r['top1']:.3f} {parts} {pool}")
if not present:
    print("!! NO pool_* key in history.json -- the collapse check is BLIND")
elif h:
    first, last = h[0], h[-1]
    for k in present:
        print(f"[pool] {k}: {first.get(k)} (step {first['step']}) -> "
              f"{last.get(k)} (step {last['step']})")
EOF

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (arm $ARM, train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
