# Caveats and reporting discipline

Every claim in [`RESULTS.md`](RESULTS.md) is conditioned on this file. These are not
formalities: each entry below records a trap that was actually hit or actually misled during
this reproduction.

---

## 1. Caveats

- **n=1 seed** throughout, no error bars — but this is **protocol parity with Waiv, not a
  shortfall against them**. Per their §3.3 (arXiv:2607.22861), THUNDER is "evaluated on frozen
  features following the default protocol" with no seed repetition, HEST follows "the default
  protocol and implementation", and PathoROB states no repetition at all. Patho-Bench is the
  only benchmark where they repeat — "Each task is run three times, and we report the mean
  over all data folds and random seeds" — and that is precisely the benchmark we dropped. So
  n=1 matches their protocol on all three benchmarks we run. The real asymmetry is **model
  count, not seed count**: their significance claims (e.g. p=7.5e-3, one-sided Wilcoxon
  signed-rank on THUNDER rank sums) aggregate across 10 model pairs, which we cannot replicate
  at 2 backbones no matter how many seeds we ran. We make no significance claim.
- **fp32 vs their mixed precision.** Each Δ is internally precision-consistent, so Δ-vs-Δ
  is valid; **absolute levels are not comparable** across the two studies.
- **Checkpoint selection is not neutral.** THUNDER and HEST use step 1000 (phikon-v2) and
  step 500 (Midnight), chosen because they were the best *PathoROB* checkpoints. A full HEST
  sweep over every checkpoint was run on **both** backbones ([`RESULTS.md`](RESULTS.md) §2): phikon-v2's HEST optimum is
  step 3500 (+0.0098, still climbing at the end of the trained range), Midnight's is step 250
  (+0.0035, decaying thereafter and negative by 1250). Robustness and retention peak at
  different steps and in *opposite* orderings on the two backbones, so any single-checkpoint
  headline understates one axis — but even the best checkpoint stays far short of Waiv, so
  the gap is not a selection artefact.
- **Coverage.** All three benchmarks now cover both backbones. THUNDER: 16 of Waiv's 16
  datasets (12/12 classification, 4/4 segmentation), so the [`RESULTS.md`](RESULTS.md) §2 segmentation average is
  like-for-like against theirs. HEST: phikon-v2 (`cls`) and Midnight (`clsmean`), base and
  fine-tuned. PathoROB: both. The only benchmark dropped is Patho-Bench (~8 TB of WSIs, no
  traceable target number).
- **The Tier-1 probe tripwire is not an early-stopping signal.** `scripts/probe_follow.py`
  detects collapse; it does not predict PathoROB RI, in either direction. On the full-FT run
  it fired its scanner-regression signal from step 300 while RI was still climbing — acting on
  it would have discarded the best checkpoint (0.8007 @ 500). On the LoRA run the opposite
  held: heldout cross-scanner separation kept rising (base 0.376 → 0.394 @ 500 → 0.439 @ 2500)
  while RI fell after step 1000. Its thresholds are diagnostics, not actionable. Relatedly, **matched cosine and
  top-1 mislead under collapse**: full FT drove heldout cross-scanner matched cosine
  0.741 → 0.963, but the random-pair baseline climbed just as fast, 0.365 → 0.597. Report
  rank-based top-1 and separation, never matched cosine alone.
- **Regressions.** `tcga_uniform` regresses on both backbones and on every probe
  (phikon-v2 −8.2 kNN / −5.7 lin / −7.3 few-shot; Midnight −2.6 / −2.9). Consistency
  across independent probes rules out a local-neighbourhood artefact — information is
  genuinely lost. `bach` regresses on Midnight only (−2.1 / −5.6 / −8.6) while it *improves*
  on phikon-v2. Both are unexplained and should be understood before any publication claim.
- **PathoROB curves are plateaus, not climbs.** Both backbones reach their peak by the
  first or second checkpoint and then flatten (phikon-v2 0.803–0.808 across steps
  500–3000; Midnight 0.898–0.908 across 250–1500). Training longer buys nothing on the
  headline metric.
- **camelyon's base RI of 0.019** is pathologically low, so its headline jump is the least
  surprising of the three PathoROB datasets.


---

## 2. Reporting discipline (`PLAN.md` §6)

- Cross-stain and cross-scanner **separately** — the composite hides the hard axis.
- Never cosine similarity alone (PLIP: 0.878 cosine at 0.054 top-10).
- Any PLISM number is a **training diagnostic**. Label it; never print it next to a
  leaderboard number.
- Retention (HEST / THUNDER) alongside **every** robustness claim, as a pair. Forgetting is
  the default outcome here, not a tail risk.
