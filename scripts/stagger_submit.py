#!/usr/bin/env python
"""Submit remaining final5 eval work into dependency-chained lanes, split fast/slow.

WHY TWO TRACKS. Measured mean runtimes vary ~10x across THUNDER datasets (bach 8.8 min vs
crc 88.2 min). In a single-track lane a short job can sit behind a 90-minute one, so the
cheap results that make the table readable arrive last. Fast datasets therefore get their
own lanes and never queue behind a slow one.

WHY LANES AT ALL. Submitting ~150 independent jobs floods the scheduler (this got the whole
batch cancelled once). Each lane is a chain: job k declares
--dependency=afterany:<job k-1 in the same lane>, so at most LANES jobs are ever eligible.
afterany, not afterok, so one failure does not strand the rest of its lane.

Within a track jobs are sorted longest-first (LPT) before round-robin assignment, which
lowers makespan versus arbitrary order.
"""
import json, os, subprocess, sys

REPO = "/admin/home/ryan.kim/waiv"
TOTAL_LANES = int(os.environ.get("WAIV_LANES", "16"))
FAST_LANES  = int(os.environ.get("WAIV_FAST_LANES", "4"))
SLOW_LANES  = TOTAL_LANES - FAST_LANES
DRY = os.environ.get("WAIV_DRY", "0") == "1"

BACKBONE = {"phikon": "owkin/phikon-v2", "midnight": "kaiko-ai/midnight",
            "virchow2": "paige-ai/Virchow2"}
# Mean minutes, measured from completed jobs this session; the ones without local data are
# taken from the previous round's sacct history. Used only for ordering/track assignment.
DUR = {"patch_camelyon": 2, "bach": 9, "mhist": 9, "break_his": 11, "esca": 12,
       "bracs": 23, "ccrcc": 33, "tcga_uniform": 41, "wilds": 41,
       "tcga_crc_msi": 53, "tcga_tils": 82, "crc": 88, "ocelot": 30, "pannuke": 30}
FAST_MAX_MIN = 15

def arm_of(run):
    for a in BACKBONE:
        if a in run: return a
    raise SystemExit(f"cannot infer arm from {run}")

miss = json.load(open("/tmp/missing_evals.json"))
fast, slow = [], []

for run in miss["hest"]:
    fast.append((5, f"hest-f5_{run}_s0000500",
                 ["sbatch", "--parsable", "--account=max", "--qos=high",
                  f"--job-name=hest-f5_{run}_s0000500",
                  f"--export=ALL,WAIV_RUN={run},WAIV_STEP=0000500",
                  "--cpus-per-task=4", f"{REPO}/scripts/hest_final5.sbatch"]))

for run, ds, kind in miss["thunder"]:
    tag = f"f5_{run}_s0000500"
    tasks = "segmentation" if kind == "seg" else "knn linear_probing simple_shot"
    argv = ["sbatch", "--parsable", "--account=max", "--qos=high",
            f"--job-name=thd-{tag}-{ds}",
            f"--export=ALL,WAIV_BACKBONE={BACKBONE[arm_of(run)]}",
            "--cpus-per-task=4", f"{REPO}/scripts/run_thunder.sbatch",
            ds, tasks, "auto", tag, "", f"{REPO}/runs/{run}/step_0000500"]
    d = DUR.get(ds, 30)
    (fast if d <= FAST_MAX_MIN else slow).append((d, f"thd-{tag}-{ds}", argv))

# Longest-processing-time first within each track lowers makespan.
fast.sort(key=lambda x: -x[0])
slow.sort(key=lambda x: -x[0])
print(f"fast track: {len(fast)} jobs -> {FAST_LANES} lanes   "
      f"slow track: {len(slow)} jobs -> {SLOW_LANES} lanes   (max {TOTAL_LANES} concurrent)")
if DRY:
    print("  fast sample:", [n for _, n, _ in fast[:3]])
    print("  slow sample:", [n for _, n, _ in slow[:3]])
    sys.exit(0)

def submit(track, nlanes, label):
    tail = [None] * nlanes
    n = 0
    for i, (_, name, argv) in enumerate(track):
        lane = i % nlanes
        cmd = list(argv)
        if tail[lane]:
            cmd.insert(1, f"--dependency=afterany:{tail[lane]}")
        try:
            jid = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        except subprocess.CalledProcessError as e:
            print(f"  FAILED {name}: {e}"); continue
        tail[lane] = jid; n += 1
        if n <= nlanes:
            print(f"  {label} lane{lane:2d} {jid} {name}")
    return n

a = submit(fast, FAST_LANES, "FAST")
b = submit(slow, SLOW_LANES, "SLOW")
print(f"submitted fast={a}/{len(fast)} slow={b}/{len(slow)}")
