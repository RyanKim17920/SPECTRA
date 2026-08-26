# Round: temperature dose-response + partial masking (PRE-REGISTERED)

Written 2026-08-24, **before any result from this round existed**. Do not edit the
hypothesis, floors, or decision rule after results land — append an outcome section instead.

## Priority inversion driving this round

HEST and THUNDER now OUTRANK PathoROB RI. We are explicitly permitted to LOSE up to
0.01 RI if HEST/THUNDER improve. No arm in this round is aimed at raising RI.

## Hypothesis

Temperature is a strong knob on the exact RI-vs-HEST axis, and every move we have made on
it so far went in the HEST-hurting direction.

- `--temperature 0.15` (vs default 0.07) moved midnight RI UP to 0.9239 while pushing HEST
  DOWN to 0.39582 — the worst HEST of any arm, -0.0126 vs base-relative expectation.
- Sub-metric decomposition: it inflated `confounder_insensitivity` (0.8349) at
  `prediction_performance`'s expense (0.8937). That is RI being gamed, not earned.

**H1.** Moving temperature BELOW 0.07 trades RI back for HEST/THUNDER — the direction we
now want. A monotone HEST response across 0.07 → 0.05 → 0.04 → 0.03 is the claim; a single
point is not, which is why this round is a dose-response and not one arm.

**H2.** `--mask-same-core` with both per-head biases at `-inf` is TOTAL masking and buys RI.
A FINITE bias is PARTIAL masking = weaker invariance, and should shift along the same axis
in the same (HEST-favouring) direction.

## Shared config (all arms)

`scripts/gentle.sbatch`, seed 0 unless stated, `WAIV_T=900`, `WAIV_MAX_STEPS=250`.
Step 250 is the **pre-registered scoring checkpoint** for these backbones. Both backbones
(`kaiko-ai/midnight`, `paige-ai/Virchow2`) unless stated.

| Arm | Priority | Config delta vs default | Backbones |
|---|---|---|---|
| temp-down + mask | P1 | `WAIV_TEMP=0.04`, `WAIV_MASK=1`, biases `-inf` / `-inf` | midnight, Virchow2 |
| temp-down harder + mask | P2 | `WAIV_TEMP=0.03`, `WAIV_MASK=1`, biases `-inf` / `-inf` | midnight, Virchow2 |
| partial mask | P3 | `WAIV_MASK=1`, `WAIV_BCLS=2.0`, `WAIV_BMEAN=2.0`, temp 0.07 | midnight, Virchow2 |
| best-HEST replication | P4 | `WAIV_LR=3e-5`, no mask, temp 0.07, seeds 1 and 2 | midnight only |

`WAIV_TEMP=0.05` is the middle dose point. It is the first thing to add if a slot frees
before 0.03 completes; the goal is a CURVE, not a point.

P4 exists because the current best-HEST midnight arm (lr 3e-5, no mask) scored **0.41326 at
n=1 and unreplicated**. Until seeds 1 and 2 land, that number is not a result.

## RI floors (hard gate)

An arm that falls below its floor is rejected regardless of HEST.

| Backbone | Current best RI | Floor (best - 0.01) |
|---|---|---|
| Virchow2 | 0.9234 | **0.9134** |
| midnight | 0.9240 | **0.9140** |

## Seed SDs at step 250

| Backbone | RI SD | HEST SD | 2 x HEST SD (the bar) |
|---|---|---|---|
| midnight | 0.00482 | 0.00227 | **0.00454** |
| Virchow2 | 0.00475 | 0.00153 | **0.00306** |

**Step-250 RI noise is ~2.3x the step-500 noise.** A step-250 RI gap that would be
significant at step 500 is NOT significant here. Do not carry step-500 intuitions across.

## Decision rule

An arm **WINS** iff BOTH hold:

1. **HEST improves by more than 2 seed-SD** for that backbone at step 250
   (midnight: > +0.00454; Virchow2: > +0.00306), AND
2. **RI stays above the floor** (Virchow2 >= 0.9134, midnight >= 0.9140).

An arm that improves HEST by less than 2 SD is NULL, not a small win — HEST has repeatedly
been shown to lack dynamic range, and a sub-2-SD move is indistinguishable from seed noise.

A drop in RI that stays above the floor is **accepted, not penalised**. That is the trade
this round is explicitly buying.

Never mix best-RI and best-HEST from different runs — a win must be one checkpoint clearing
both conditions simultaneously.

## Run-name discipline

HEST and PathoROB caches are keyed on run name / exp_code ALONE. Two arms that collide in
that string silently share results — this has produced fake results before.

`RUN_NAME` in `gentle.sbatch` encodes temp (`t0.04`), mask (`genMASK`), lr, seed, steps,
arm, and the SLURM job id — but **NOT** `WAIV_BCLS` / `WAIV_BMEAN`. A `WAIV_TAG` hook was
added to `gentle.sbatch` for this round; P3 MUST set `WAIV_TAG=b2.0` or it is
name-indistinguishable from a total-mask run except by job id.

---

## Launch log (appended after submission; hypothesis/floors/rule above are UNCHANGED)

| Job | Arm | Temp | Run name |
|---|---|---|---|
| 391888 | midnight | 0.04 | `genMASK-lr1e-4-r32-kl0-t0.04-wd0.05-ms250-pd512-midnight-s0-t900-391888` |
| 391889 | Virchow2 | 0.04 | `genMASK-lr1e-4-r32-kl0-t0.04-wd0.05-ms250-pd512-virchow2-s0-t900-391889` |
| 391921 | midnight | 0.03 | `genMASK-lr1e-4-r32-kl0-t0.03-wd0.05-ms250-pd512-midnight-s0-t900-391921` |
| 391922 | Virchow2 | 0.03 | `genMASK-lr1e-4-r32-kl0-t0.03-wd0.05-ms250-pd512-virchow2-s0-t900-391922` |

Still deferred: P3 partial mask (MUST set `WAIV_TAG=b2.0`), P4 midnight lr3e-5 seeds 1/2,
and the temp 0.05 mid-dose point.

### Note on early loss (do NOT read as a result)

At temp 0.04/0.03 the InfoNCE loss collapses fast (midnight step 60: loss 0.046, top1 0.994;
Virchow2 step 69: loss 0.020, top1 0.997). Lower temperature sharpens the softmax, so a low
loss here is a mechanical consequence of the knob, not evidence the arm is good.
**InfoNCE loss anti-correlates with RI in this project** — the `meanonly` arm had the best
loss and top1 and the worst robustness. Rank these arms on HEST/THUNDER and the RI floor
per the decision rule above. Ignore loss entirely.

---

## SUPERSESSION (coordinator, after launch) — relative floors replace absolute floors

The absolute RI floors above (Virchow2 0.9134 / midnight 0.9140) are **SUPERSEDED**. They are
left in place unaltered for audit; do not score against them.

New success bar: achieve **80-100% of Waiv's GAIN over base**, i.e. `(ours-base)/(Waiv-base)`,
averaged across datasets. HEST and THUNDER are the priority metrics; RI only needs >= 80% of
Waiv's gain.

Derived relative RI floor = `base + 0.80*(Waiv-base)`:

| Backbone | RI base | Relative floor | Old absolute floor | Change |
|---|---|---|---|---|
| midnight | 0.7589 | **0.89098** | 0.9140 | far looser (-0.023) |
| Virchow2 | 0.8582 | **0.90604** | 0.9134 | far looser (-0.007) |

This gives the temp 0.04 / 0.03 arms substantially more RI headroom than this doc originally
claimed. HEST base values for the gain denominator: midnight 0.39521, Virchow2 0.40324.

### Why replication now outranks new levers

HEST's seed noise is too large to resolve the bar at n=1. 2*seed_SD spans 22.4% of Waiv's
HEST gain on Virchow2 at step 500 (29.8% at step 250) and 21.1% on midnight at step 250. So
on HEST the difference between "80% of Waiv" and "100% of Waiv" — 20 percentage points of
gain — sits INSIDE the noise of a single seed. **An unreplicated arm cannot clear the bar
even if it is genuinely good.** Revised order: P4 replication first, then P3, temp 0.05 last.

### 0.41326 VERIFIED, and a schedule trap it exposed

Provenance confirmed, not circular: `hest_perf_per_encoder = 0.41326296` in
`/data/ryan.kim/hest_work/results/f5_gen-lr3e-5-r32-kl0-midnight-s0-t900-391082_s0000250_cls_summary.json`
(midnight, pooling `cls`, per-dataset avg 0.41325556). Source run
`runs/gen-lr3e-5-r32-kl0-midnight-s0-t900-391082`: lr 3e-5, temp 0.07, mask OFF, kl 0.

**TRAP: that run had `max_steps: 1500`, not 250.** It was scored at its step-250 checkpoint
but trained on a 1500-step LR schedule, so at step 250 the LR is still near peak. A 250-step
run anneals to ~0 by step 250 and is a DIFFERENT arm. `CKPT_EVERY` is 250 regardless of
`MAX_STEPS`, so `ms1500` still writes `step_0000250` at ~90 min and the step-250 HEST can be
scored long before training ends. **The seed replication therefore runs at
`WAIV_MAX_STEPS=1500` and is scored at step 250.** Scoring an ms250 run against 0.41326
would be an invalid comparison.

### Virchow2 replication NOT launched — arm could not be verified

There is no `ret0.01` arm in this repo, and no Virchow2 HEST score exists yet to identify a
"best-HEST Virchow2 arm". The two candidates are still being scored right now:
`genMASK-lr3e-5-...-virchow2-s0-t900-391769` (mask ON) and
`gen-lr3e-5-...-virchow2-s0-t900-391770` (mask OFF); both have results dirs under
`/data/ryan.kim/hest_work/results/` with no `_summary.json` yet. Spending 2 slots replicating
an arm whose score does not exist would be premature. Revisit once those summaries land.

### INCIDENT: a monitor manufactured 8 fake HEST scores (caught and retracted)

A monitoring script I armed reported "HEST SCORE READY" for all eight arms of this round,
every one showing `avg = 0.3886`. **All eight were fake.** No HEST score for this round
existed at that time; the first scoring job (391970) had only just been submitted.

Cause: the glob was written `f5_*-$J_s0000250_*_summary.json`. Bash parsed `$J_s0000250` as
a single (unset) variable, so the pattern collapsed to `f5_*-_*_summary.json`, which matches
essentially every summary in the results dir. `head -1` then returned the same alphabetically
first unrelated file — `f5_all91-phikon-s0-t900-387667_s0000500_cls` — for all eight jobs.
Note it is a **phikon** run at **step 500**: wrong backbone, wrong step, wrong arm.

This is the same failure mode this doc already warns about under run-name discipline — a
result keyed by a name pattern silently attributed to the wrong run — reproduced in the
monitoring layer rather than the cache layer. Missing `${}` braces were enough.

Fix: glob corrected to `f5_*-${J}_s0000250_*_summary.json`, plus a hard identity assertion
that the resolved `exp_code` contains `-<jobid>_s0000250_` before any number is reported;
on mismatch it prints IDENTITY MISMATCH instead of a score. Verified: the corrected pattern
resolves job 391082 to exactly its own summary and matches nothing for 391888.

**Lesson: a monitor that reports numbers is a results path and needs the same identity
checks as the cache.** Any HEST value quoted for this round before this fix must be discarded.

### INCIDENT 2: three consecutive false alarms from the monitoring layer

Three alerts fired on job 391889; all three were bugs in my monitors, not job failures. The
follower was verified alive and correct throughout (`eval_checkpoints` pid present on-node,
correct `falseneg-pinned` pin, probe for step 250 completed).

1. **"eval_follow.log absent/empty"** — tested for a NON-EMPTY file. Python block-buffers
   stdout, and the follower has nothing to print until the first checkpoint. File emptiness
   is not process death.
2. **"TRAIN_DONE but no ri_curve.json"** — a race, not a failure. Training writes TRAIN_DONE
   BEFORE the follower finishes scoring the final checkpoint, so this state is normal for
   several minutes on every run.
3. **"follower log idle >15min"** — the predicate itself was broken. `find` on this box is
   **`bfs`**, which does NOT accept relative timestamps in `-newermt`: both
   `-newermt '-15 minutes'` and `-newermt '15 minutes ago'` exit 1 with "Invalid timestamp"
   (only ISO-8601 is supported). stderr was sent to /dev/null, so the error was read as an
   empty result, i.e. "stalled", and fired every cycle regardless of actual mtime. The log
   was 2 minutes old when it claimed 15. **Use `-mmin -15` on this machine, never
   `-newermt <relative>`.**

Common root cause across all three (and the fake-HEST incident): a FILE STATE was used as a
proxy for PROCESS HEALTH, and the predicate was never tested against a known-good case
before being trusted. Monitors are results infrastructure. Test the predicate on a positive
AND a negative control before arming, exactly as one would for a metric.
