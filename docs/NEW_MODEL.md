# Running this recipe on a new backbone

The recipe is backbone-agnostic in principle: masked InfoNCE over PLISM co-registered pairs,
LoRA on all transformer blocks, PathoROB as the primary metric. Two backbones have been taken
end to end (`owkin/phikon-v2`, `kaiko-ai/midnight`). This file is the checklist for a third.

De-hardening of the scripts is in flight. Where an exact invocation is not yet settled it is
marked **TBD** below rather than guessed — an authoritative-looking wrong flag costs more than
a marked gap. Library-level symbols named here (`src/waivphaet/models/encoder.py`,
`src/waivphaet/eval/*`) are current; CLI surfaces are the part in motion.

---

## 1. Register the backbone

- **Normalization.** `BACKBONE_NORMALIZATION` in `src/waivphaet/models/encoder.py` is
  table-driven; `normalization_for()` falls back to the HF preprocessor config and raises on
  a backbone it cannot resolve. Add an explicit entry keyed by the HF id. Do not skip this:
  wrong stats change no shape and throw no error, they only cost base accuracy — the exact
  silent failure recorded in [`REPRODUCING.md`](REPRODUCING.md) §2.
- **LoRA targets.** `LORA_CANDIDATE_MODULES` / `LORA_TARGET_MODULES` are discovered by leaf
  name, and the per-block count is asserted non-empty and uniform. A new architecture (SwiGLU,
  fused QKV, GQA) may expose leaf names in neither list. Add them, then confirm the per-block
  assertion still passes rather than trusting "LoRA covers all blocks".
- **Block indexing.** Block numbers are parsed from parameter names by the regex in
  `encoder.py` covering `layer|layers|blocks|block|h`. A backbone with a different container
  name needs that pattern extended, or every module lands in block `None`.
- **Pooling and feature width.** Record which pooling the new backbone is evaluated under
  (`cls` vs `clsmean`) and its feature dimension. This is not a free choice — it follows the
  benchmark's published protocol for that backbone, and THUNDER's is per-backbone
  (see [`RESULTS.md`](RESULTS.md) §3).

## 2. Confirm the base reproduces a published reference

Do this **before** any training. The entire value of the deltas rests on it, and it is also
the only thing that catches a mis-registered backbone.

- Run the base backbone through PathoROB and check the Avg RI against a published row
  (Waiv Table 1, or PathoROB's own committed reference where one exists).
- Run the base through THUNDER and cross-check against THUNDER's published leaderboard;
  `scripts/collect_thunder.py` prints these as `# cross-check` lines.
- If no published counterpart exists for the backbone × pooling combination, say so
  explicitly in the summary — the harness already emits such a note for Midnight `clsmean` on
  HEST — and report base→FT deltas only, never absolute levels against another study.

A base that does not reproduce is a harness bug, not a result.

## 3. Sizing and smoke

- `scripts/smoke_test.py --steps 4 --device cpu` for wiring.
- `scripts/sizing_probe.py` for batch size / memory on the new width.
- Selecting the backbone for these: **TBD** (flag vs env var vs sbatch variable is part of the
  in-flight de-hardening; check the script's own `--help` at the time of use).

## 4. Train

- LoRA reference configuration is rank 32 / alpha 64, as in `scripts/train_real.sbatch`.
  `scripts/train_full_ft.sbatch` runs the full-FT arm (`--full-ft`).
- The PLISM split is *named*, not sampled — held-out scanners `GT450`, `S210` and stains
  `HRH`, `KR`, `MY`, giving 50 train / 41 held-out conditions — so it reproduces without a
  seed and needs no change per backbone.
- Checkpoint cadence: choose it from where the previous backbone's curve moved, not from a
  default. PathoROB curves plateau early on both backbones seen so far
  ([`CAVEATS.md`](CAVEATS.md)), so dense early checkpoints are worth more than a long tail.
- Passing a non-default backbone to the sbatch entrypoints: **TBD**.

## 5. Evaluate — all three benchmarks, base and fine-tuned

Run base and fine-tuned through identical code so the deltas carry no cross-study assumption.

- **PathoROB** (primary, never seen in training): extraction via SLURM, then the gate script.
- **THUNDER**: 16 datasets, 4 tasks. `segpath_epithelial` runs at 9 epochs and
  `segpath_lymphocytes` at 21 per THUNDER's own `docs/guidelines.md`.
- **HEST** (retention): pass `--base` and `--runs` explicitly. The default
  `--base base_<pooling>` will silently diff against another backbone's baseline.
- Exact invocations are in [`REPRODUCING.md`](REPRODUCING.md) §1 for the two backbones
  already wired; the new-backbone equivalents of the model/run naming convention are **TBD**.

## 6. Verify before reporting

- **Adapter applied.** The extractor compares against `disable_adapter()` and exits non-zero
  below 1e-4; observed `rel_l2_delta` on the existing runs is 0.73–0.93, recorded per point in
  `ri_curve.json`. A new backbone with near-zero delta means the LoRA targets did not attach.
- **No refactor drift.** `scripts/regression_bitcheck.py` compares raw feature arrays across
  worktrees. Any change made to accommodate a new backbone must leave the existing backbones
  bit-identical.
- **Tripwires are diagnostics.** `scripts/probe_follow.py` does not predict PathoROB RI in
  either direction; do not early-stop on it. Never report matched cosine alone. Both traps are
  documented with their evidence in [`CAVEATS.md`](CAVEATS.md).

## 7. Report

Follow the reporting discipline in [`CAVEATS.md`](CAVEATS.md) — in particular: one checkpoint
per backbone, selected by a blind model-agnostic rule ("best PathoROB checkpoint"), used on
*every* benchmark. Picking the per-benchmark best checkpoint inflates retention and breaks the
single-model-per-backbone comparison. Where the rule costs something, record the cost rather
than optimising it away.

Add the new backbone as its own subsection of [`RESULTS.md`](RESULTS.md) §1–§3 with the same
column layout, so base-published / base-ours / fine-tuned-ours stay side by side.
