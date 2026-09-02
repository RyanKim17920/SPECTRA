# RUNBOOK — how to run this project (single source of truth)

Written 2026-08-31 on branch `final5-and-ablations`, from the scripts themselves (not from
memory or older docs). Where an older doc (`docs/FINAL_RECIPE.md`, dated 2026-08-25)
disagrees with what the scripts currently do, that is called out explicitly below — do not
silently prefer one over the other.

---

## 1. The recipe: `scripts/gentle.sbatch`

One script, one recipe, only `--backbone` and `--seed` (via `WAIV_ARM` / `WAIV_SEED`) vary
across the 5 supported backbones (`phikon`, `midnight`, `virchow2`, `hoptimus`, `uni2`).

### 1.1 Environment variables it reads (all optional except where marked required)

| var | default | meaning |
|---|---|---|
| `WAIV_ARM` | **required** | `phikon\|midnight\|virchow2\|hoptimus\|uni2`. Maps to `--backbone` (phikon leaves it unset → train_lora.py's own default `owkin/phikon-v2`). `hoptimus`→`bioptimus/H-optimus-0`, `uni2`→`MahmoodLab/UNI2-h` — both **gated** HF repos (see §2.1). |
| `WAIV_SEED` | **required** | `--seed` |
| `WAIV_T` | **required** | grid tiles (`GRID_TILES`); `GRID_COND=2` is hardcoded, so batch = `2*T` |
| `WAIV_LR` | `1e-4` | `--lr` |
| `WAIV_MAX_STEPS` | `1500` | `--max-steps` |
| `WAIV_CKPT_EVERY` | `250` | `--ckpt-every` and `--eval-every` (same value) |
| `WAIV_BCLS` | `-inf` (only applied when `WAIV_MASK` set) | `--same-core-logit-bias-cls` |
| `WAIV_BMEAN` | `-inf` | `--same-core-logit-bias-mean` |
| `WAIV_RANK` | `32` | `--lora-rank` |
| `WAIV_ALPHA` | `2*WAIV_RANK` | `--lora-alpha` |
| `WAIV_PROJDIM` | `512` | `--proj-out-dim` |
| `WAIV_WARMUP` | `200` | `--warmup-steps` |
| `WAIV_WD` | `0.05` | `--weight-decay` |
| `WAIV_TEMP` | `0.07` | `--temperature` |
| `WAIV_KL` | `0` | `--retention-kl-weight` |
| `WAIV_MASK` | unset | if set (any value), turns on `--mask-same-core` and applies `WAIV_BCLS`/`WAIV_BMEAN`; also folds `MASK` into `RUN_NAME` and changes the default pin (see §2.1) |
| `WAIV_MINTF` | unset | `--min-tissue-frac`, only passed if set |
| `WAIV_PIN` | see §2.1 | which code snapshot's `src/` and `scripts/train_lora.py` to run |
| `WAIV_TAG` | unset | free-form string folded into `RUN_NAME`; **required** whenever two runs would otherwise only differ by something `RUN_NAME` doesn't encode (chiefly `WAIV_BCLS`/`WAIV_BMEAN` — those are invisible in the auto-generated name) |
| `WAIV_RUN_NAME` | unset | replaces the auto-generated `RUN_NAME` wholesale (job id still appended). See §2.2 for why you'll usually need this. |
| `WAIV_PACKED_DIR` | `/data/plism/repacked` | tile source |
| `WAIV_REPO` | `/admin/home/ryan.kim/waiv` | repo root |
| `WAIV_WORKERS` | `10` | dataloader workers |
| `WAIV_ALLOW_LONG_RUN_NAME` | unset | override the 64-char THUNDER key check (§2.2) — proceeds RI/HEST-only, THUNDER becomes impossible for that run |
| `WAIV_EVAL_MAX_WAIT_S` | `72000` | RI-eval follower max wait |
| `WAIV_NO_FOLLOWER` | unset | `1` disables the RI-eval follower — smokes only, never for a real run (no `ri_curve.json`, and RI is the only readout with dynamic range to rank arms) |
| `WAIV_BACKBONE_LOCAL_DIRS` | unset | overrides the gated-backbone local-path table (`repo_id=/path,...`) if `/data` is swept again |

Hardcoded, not tunable: `GRID_COND=2`, `GRID_CHUNK=0` (GEM/pool-head forces unchunked —
`contrastive.py` raises if you try to chunk a non-default pool head), pooling at train time
is `clsmean`, head is `--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem`,
`--grad-checkpointing --activation-offload` always on, `#SBATCH --mem=700G` (load-bearing:
pinned host RAM for offloaded activations is not swappable — do not shrink this).

### 1.2 The FINAL choice actually in production on this branch

```bash
WAIV_PIN=/admin/home/ryan.kim/waiv-snapshots/falseneg-gated \
WAIV_ARM=<phikon|midnight|virchow2|hoptimus|uni2> \
WAIV_SEED=<n> \
WAIV_T=900 \
WAIV_MASK=1 \
WAIV_BCLS=3.0 \
WAIV_MAX_STEPS=500 \
WAIV_CKPT_EVERY=50 \
WAIV_RUN_NAME=genMASK-c50-ms500-<arm>-s<n>-t900 \
sbatch --account=idle --qos=low scripts/gentle.sbatch
```

i.e. `WAIV_BMEAN` is left at its default (`-inf`) — the bias is **asymmetric**
(cls=+3.0, mean=-inf) — `WAIV_LR=1e-4`, `WAIV_WARMUP=200`, `WAIV_RANK=32`,
`WAIV_PROJDIM=512` are all left at their sbatch defaults. This is exactly what
`watch/saturate.py`'s priority-3 auto-launcher submits (`saturate.py:307-317`), producing
the `runs/genMASK-c50-*` grid: 10 checkpoints at `WAIV_CKPT_EVERY=50` from step 50 to 500,
2 seeds wanted per arm, 5 arms.

**Note this supersedes `docs/FINAL_RECIPE.md`** (2026-08-25), which documents an earlier
2-seed, 3-backbone (`phikon`/`midnight`/`virchow2` only) pilot at `WAIV_CKPT_EVERY=125` and
relied on `WAIV_MASK=1`'s *default* pin resolution (`falseneg-pinned`, no gated-backbone
support — it never needed it, since that pilot didn't cover `hoptimus`/`uni2`). The c50 grid
above is the current, actively-running production sweep and is the one to cite for the
5-backbone claim; `FINAL_RECIPE.md`'s numbers are still valid for the 3 ungated backbones
at their own (125-step) checkpoint cadence, just don't average the two together.

Verify what pin/eval protocol a given checkpoint was made under by reading the run's own
`runs/<run>/gpu.csv`-adjacent stdout log (`echo` block at sbatch top: prints `run`, `arm`,
`geometry`, `head`, `steps`, `pin`) rather than assuming from the run name alone.

---

## 2. Two traps that have cost real job failures

### 2.1 `WAIV_MASK=1` alone silently selects the wrong pin for gated backbones

`gentle.sbatch:78`:
```bash
PIN="${WAIV_PIN:-${WAIV_MASK:+/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned}}"
PIN="${PIN:-/admin/home/ryan.kim/waiv-snapshots/gemgrid-pinned}"
```
So setting only `WAIV_MASK=1` (without `WAIV_PIN`) resolves to `falseneg-pinned`. I checked
that snapshot directly: **`falseneg-pinned/src/waivphaet/models/encoder.py` has no
`BACKBONE_LOCAL_DIRS`** (grep for it returns nothing), while
`falseneg-gated/src/waivphaet/models/encoder.py` does (it defines the table and the
`WAIV_BACKBONE_LOCAL_DIRS` override mechanism, lines ~135-183). `BACKBONE_LOCAL_DIRS` is
what redirects `bioptimus/H-optimus-0` / `MahmoodLab/UNI2-h` (both gated on HF, 403 without
it) to their local `/data` checkpoints. So `WAIV_MASK=1` with no explicit `WAIV_PIN` trains
phikon/midnight/virchow2 fine but **403s at backbone load for hoptimus/uni2**.

**Always pass explicitly for any run touching a gated backbone:**
```bash
WAIV_PIN=/admin/home/ryan.kim/waiv-snapshots/falseneg-gated
```
This is exactly what `saturate.py`'s auto-launcher does (`saturate.py:314`) and what every
real gated-backbone run in git history used.

### 2.2 Run-name length: THUNDER's 64-char model-key cap

THUNDER's model key is `f5_<RUN_NAME>_s<step:07d>` = `len(RUN_NAME) + 12` chars, and its
pydantic `run_tags` hard-caps that at 64 — so `RUN_NAME` has a **52-char budget**.
`gentle.sbatch`'s auto-generated name (`gen${MASK}${TAG}-lr...-r...-kl...-t...-wd...-ms...
-pd...-<arm>-s<seed>-t<T>-<jobid>`) runs ~71 chars and **always exceeds this** — every
THUNDER job for an un-overridden `gentle.sbatch` run is rejected at validation, silently, as
a killed sweep rather than a visible error. The sbatch itself checks this at launch time and
`exit 4`s unless you either shorten the name or pass `WAIV_ALLOW_LONG_RUN_NAME=1` (which
gives up THUNDER entirely for that run).

**Fix: always set `WAIV_RUN_NAME` to something short**, e.g.
`genMASK-<tag>-ms500-<arm>-s<seed>-t900` (the c50 grid literally uses
`genMASK-c50-ms500-<arm>-s<seed>-t900`, ~35-40 chars). If you override it, the sbatch's own
comment (and `scoreboard2.py:_parse_run_meta`) says you must keep the tokens the scoreboard
greps back out: `MASK`, `-lr`, `-kl`, `-ms`, `-s<seed>`, `-t<T>`, a trailing job id, and an
arm name present in `scoreboard2._BACKBONES = ("phikon","midnight","virchow2","hoptimus","hopt","uni2")`
— otherwise that run's scoreboard metadata degrades to unknown/None. Note the current c50
`WAIV_RUN_NAME` scheme (`genMASK-c50-ms500-<arm>-s<seed>-t900`, no explicit `-lr`/`-kl`
tokens) means `scoreboard2._parse_run_meta`'s regex lookups for those fields fall back to
`None`/default rather than erroring — check `scripts/scoreboard2.py` output for a given run
if you need those fields reported, don't assume they parsed.

---

## 3. Copy-pasteable launch commands

All SLURM submissions in this project default to `--account=idle --qos=low` unless a stage
is specifically long-running and GPU-light, in which case it defaults to `max`/`high`
(THUNDER's `online` stage — see §5).

### 3.1 Training (`scripts/gentle.sbatch`)
```bash
cd /admin/home/ryan.kim/waiv
WAIV_PIN=/admin/home/ryan.kim/waiv-snapshots/falseneg-gated \
WAIV_ARM=uni2 WAIV_SEED=2 WAIV_T=900 WAIV_MASK=1 WAIV_BCLS=3.0 \
WAIV_MAX_STEPS=500 WAIV_CKPT_EVERY=50 \
WAIV_RUN_NAME=genMASK-c50-ms500-uni2-s2-t900 \
sbatch --account=idle --qos=low scripts/gentle.sbatch
```

### 3.2 RI (PathoROB-style co-registration metric) and THUNDER, 6-column suite, plus CPTAC
New harness, **not the old HEST script** — lives outside this repo at
`/admin/home/ryan.kim/pathfm-full-evals` (the runner code: `run_gpu.sbatch`,
`run_cpu.sbatch`, `run_manifest.py`, `thunder_eval.py`, `pathorob_eval.py`,
`cptac_eval.py`, `hest_eval.py`, `submit_suite.sh`, `submit_all.sh`), evaluated per-cell
under `/admin/home/ryan.kim/pathfm-cells/<cell>/` (one directory per checkpoint/model,
containing its own `model.py` + a copy of `submit_partial.sh` / `submit_online.sh`).

```bash
# Partial THUNDER (preflight -> precompute -> cached probes -> cleanup): KNN, linear
# probing, 16-shot SimpleShot, calibration/ECE. Skips segmentation + PGD.
cd /admin/home/ryan.kim/pathfm-cells/<cell>
./submit_partial.sh . idle low

# THUNDER completion: segmentation (4 tasks) + PGD + summary. Run ONLY after
# submit_partial.sh's cleanup stage has COMPLETED. Defaults to max/high, not idle/low,
# because it's the single biggest GPU-hour item (4-4.5h) and should not sit preemptible.
./submit_online.sh . max high

# Or drive one full suite end-to-end from the harness root:
cd /admin/home/ryan.kim/pathfm-full-evals
./submit_suite.sh thunder            # or hest | cptac | pathorob
./submit_suite.sh thunder --embeddings=retain   # keep intermediate embeddings
./submit_all.sh                       # everything, chained; --no-pathorob to skip PathoROB
```

### 3.3 HEST — the OLD per-checkpoint script, still the one in use for HEST specifically
```bash
cd /admin/home/ryan.kim/waiv
WAIV_RUN=genMASK-c50-ms500-virchow2-s0-t900-396382 WAIV_STEP=0000500 \
sbatch --account=idle --qos=low scripts/hest_final5.sbatch
```
Pooling is derived from the run name (never pass it by hand):
`*phikon*`/`*midnight*` → `cls`; `*virchow2*` → `clsmean`; `*hoptimus*`/`*hopt*` → `cls`;
`*uni2*` → `cls`. This must agree with `collect_final5.HEST_POOLING`;
`tests/test_invariants.py` asserts they match. Results land at
`/data/ryan.kim/hest_work/results/f5_<run>_s<step>_<pooling>_summary.json`.
Note `EXP_CODE` (and therefore the output filename) must be unique per `(run, step)` or
results silently collide on disk. `--num-workers 0` is required (HEST's dataloader crashes
in shared-memory teardown at higher worker counts).

**Why HEST is a separate old script and not folded into `pathfm-full-evals`**: it predates
that harness and its pooling-protocol logic (`src/waivphaet/eval/thunder_protocol.py` for
THUNDER; `HEST_POOLING`/pooling case statement in `hest_final5.sbatch` for HEST) is specific
per-backbone and already validated against Waiv's published numbers — see the header
comment in `hest_final5.sbatch`: phikon-v2 base 0.37470 reproduces published 0.3747 exactly
under `cls`; midnight base 0.39521 vs published 0.3952 under `cls`; Virchow2 base 0.40327
vs published 0.4034 under `clsmean` (under `cls` it's 0.39791, off by 0.0055). Separately:
HEST uses `Resize(224, bilinear)` while THUNDER's leaderboard protocol uses
`Resize(256, bicubic)` — a real, confirmed transform mismatch between the two eval paths
(each matches its own benchmark's published protocol; RI/HEST outcomes are unaffected by
it). Don't expect HEST and THUNDER numbers for the same checkpoint to have been produced
under identical preprocessing — that is expected, not a discrepancy to chase.

THUNDER's per-backbone pooling protocol is defined once, dependency-free, in
`src/waivphaet/eval/thunder_protocol.py`: `THUNDER_CLSMEAN_BACKBONES = {"kaiko-ai/midnight",
"paige-ai/Virchow2"}`, `THUNDER_CLS_BACKBONES = {"owkin/phikon-v2", "bioptimus/H-optimus-0",
"MahmoodLab/UNI2-h"}`, everything else raises rather than silently defaulting. Note this
differs from HEST's own pooling table above only in that HEST's case statement is
independently maintained in the sbatch and cross-checked by `tests/test_invariants.py` —
they encode the same facts but are two separate pieces of code, so a change to one without
the other is exactly the kind of drift the test exists to catch.

**Result paths to know** (depth has bitten people before — don't glob one level too
shallow):
- HEST: `/data/ryan.kim/hest_work/results/f5_<run>_s<step>_<pooling>_summary.json` (old
  script) and `/data/ryan.kim/pathfm-full-evals/hest/results/<cell>/aggregate.json` (new
  harness)
- PathoROB (new harness): `/data/ryan.kim/pathfm-full-evals/pathorob/results/<metric>/<cell>_clsmean/`
  — one directory **per metric** (`apd`, `clustering_score`, `robustness_index`, ...), each
  containing per-cell subdirectories suffixed `_clsmean`. Globbing
  `pathorob/results/<cell>*` (one level, no metric) finds nothing.
- CPTAC: `/data/ryan.kim/pathfm-full-evals/cptac/<cell>/aggregate.json`
- THUNDER (new harness): `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/<dataset>/<cell>_optimized/{knn,linear_probing,segmentation,...}/frozen/outputs.json`

---

## 4. Which readout to trust

Per-backbone HEST/THUNDER pooling as above; RI (PathoROB-style) and HEST are prioritized
over THUNDER for ranking arms per project memory (THUNDER has a documented noise floor
issue — 3 noise units of total range on some cells — so a flat THUNDER column can mean the
instrument is blind, not that the arm is null). Grade by **pooled average** (sum
numerator/denominator once across cells), not by averaging per-cell percentages, and always
report a per-cell breakdown alongside the pooled number so a >50%-share cell doesn't hide
behind the aggregate.

---

## 5. The automation: `watch/saturate.py`

Run as a persistent SLURM job via `watch/saturate.sbatch` (`--partition=c`, 8GB, 7-day
walltime, `--requeue`; the sbatch wraps it in a `while true` restart loop so a transient
exception in the controller doesn't end the daemon). It replaced an earlier `setsid nohup`
approach launched from an interactive agent shell, which died whenever the launching
session's process tree was torn down.

**Restart it**: `sbatch /admin/home/ryan.kim/waiv_status/watch/saturate.sbatch` from
`/admin/home/ryan.kim/waiv_status`. Check it's alive with `squeue -u ryan.kim -n
waiv-saturate` (job name from the sbatch's `#SBATCH --job-name`) or by tailing
`/admin/home/ryan.kim/waiv_status/saturate.out`. Only one instance should run — it self-loops
already; launching a second on top of a running one will double-submit.

**Three priorities, checked in order every tick** (`TARGET_INFLIGHT = 130`, polled every
`POLL_S = 300`s, capped at `MAX_ITERS = 2000` ticks per process lifetime — the restart loop
in the sbatch is what makes that not a real limit):

1. **Cell × suite** — for each cell under `/admin/home/ryan.kim/pathfm-cells/` not already
   busy (has a job in `squeue` whose working dir is that cell) and missing results for one
   of `SUITES = ["thunder", "online", "pathorob", "cptac"]` (checked via `has_results()`,
   the per-suite result-path logic in §3.3), submits the next missing suite via that cell's
   own `submit_partial.sh` / `submit_online.sh`, or the harness's `submit_suite.sh`.
2. **Unmeasured HEST `(run, step)` points** — scans `runs/genMASK-c50-*` /
   `-bm3-*` / `-b00-*` / `-c3s-*` / `-c3m-*` for checkpoints on disk lacking a matching
   `f5_<run>_s<step>_*_summary.json`, and submits `scripts/hest_final5.sbatch` for each,
   gated by `MAX_HEST_INFLIGHT = 4` — going above 4 concurrent HEST jobs causes
   `BlockingIOError: unable to lock file (errno 11)` because HEST jobs share an h5
   dataset-cache file lock; this is self-inflicted contention, not a real resource limit.
3. **Spare capacity → finish incomplete 50-step training grids.** Only fires once priorities
   1-2 haven't filled the queue to target. Picks the arm (of `TRAIN_ARMS = ("hoptimus",
   "midnight", "virchow2", "phikon", "uni2")`) with the fewest complete grids
   (`FULL_GRID = 10` checkpoints at `WAIV_CKPT_EVERY=50`, `SEEDS_WANTED = 2` seeds per arm
   before it stops adding more), and launches `gentle.sbatch` for it with exactly the §1.2
   final-choice invocation — including `WAIV_PIN=falseneg-gated` explicitly, which is why
   this auto-launch path does NOT hit trap §2.1.

Backfill logic: once in-flight drops more than 5 below `TARGET_INFLIGHT`, the effective
per-tick submission cap becomes unbounded (`float('inf')`) instead of `TARGET_INFLIGHT`, so
the queue actually climbs back to 130 in one tick rather than trickling up 1 job at a time.

Also on every tick, before submitting anything: cancels any job whose scheduler reason is
`DependencyNeverSatisfied` (a stranded job whose upstream dependency died), and calls
`clear_orphans()` to `rm -rf` any per-cell output directory under the harness's `pathorob/
cptac/hest/thunder` trees that has no corresponding `manifests/<cell>.json` — see §6 on why
that manifest check exists.

---

## 6. Operational traps

- **`scancel` on an array job silently no-ops if you pass the array-task id form.**
  `scancel "1234_[0-8%8]"` is rejected (`"Invalid range: 0-8%8"`) and the job keeps running
  — you must cancel the **base** job id (`scancel 1234`). `saturate.py`'s own stranded-job
  cleanup takes care to split off `l.split()[0].split("_")[0]` before calling `scancel` for
  exactly this reason.
- **One manifest per cell, keyed by `MODEL_NAME`.** `run_manifest.py:write_or_verify_manifest`
  writes/checks `manifests/<MODEL_NAME>.json` and refuses to run
  (`AssertionError: MODEL_NAME=... is already bound to a different model or evaluation
  protocol`) if that file exists with different content, or
  (`AssertionError: unverifiable outputs exist without a manifest`) if output directories
  for that `MODEL_NAME` already exist but no manifest does. **Never run two suites against
  the same cell concurrently** — their preflight stages race to write/verify the same
  manifest file and the loser dies on `verify_manifest()`. This is also why `saturate.py`
  tracks `busy_cells()` from `squeue` working directories before picking the next
  cell×suite to submit.
- **Orphan outputs block re-runs, not just races.** If a cell's output directories exist
  but its manifest was deleted or never written (e.g. a killed job, or a renamed cell), the
  next `write_or_verify_manifest` call for that `MODEL_NAME` refuses outright rather than
  overwriting. `saturate.py`'s `clear_orphans()` is the automated fix — it deletes any
  cell's output dirs under `pathorob/features`, `pathorob/results/*/*`, `cptac/*`,
  `hest/results/*`, `thunder/embeddings/*/*`, `thunder/outputs/res/*/*` that have no
  matching `manifests/<cell>.json`, every tick, before the priority queue runs.

---

## 7. Files referenced in this runbook (for follow-up reading)

- `/admin/home/ryan.kim/waiv/scripts/gentle.sbatch` — training launcher
- `/admin/home/ryan.kim/waiv/scripts/hest_final5.sbatch` — HEST launcher (old harness)
- `/admin/home/ryan.kim/waiv/scripts/scoreboard2.py` — run-name parsing / scoreboard
- `/admin/home/ryan.kim/waiv/src/waivphaet/eval/thunder_protocol.py` — THUNDER pooling table
- `/admin/home/ryan.kim/waiv/tests/test_invariants.py` — cross-checks HEST/THUNDER pooling agreement, sampler/loss invariants
- `/admin/home/ryan.kim/waiv/docs/FINAL_RECIPE.md` — earlier 3-backbone/125-step pilot (superseded for the 5-backbone claim, see §1.2)
- `/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned/` vs `falseneg-gated/` — the two pins in trap §2.1
- `/admin/home/ryan.kim/pathfm-full-evals/` — new RI/THUNDER/PathoROB/CPTAC harness (`run_manifest.py`, `submit_suite.sh`, `submit_all.sh`, per-suite `*_eval.py`)
- `/admin/home/ryan.kim/pathfm-cells/<cell>/` — per-checkpoint eval cells (`submit_partial.sh`, `submit_online.sh`)
- `/admin/home/ryan.kim/waiv_status/watch/saturate.py` + `saturate.sbatch` — the queue-saturation daemon (§5)
- `/admin/home/ryan.kim/waiv_status/watch/reaper.py` — saturate.py's predecessor, superseded but left in place
