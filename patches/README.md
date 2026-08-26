# Local patches to gitignored third-party trees

`third_party/` is excluded by `.gitignore:26`, so changes there cannot be committed
directly. Any local fix we depend on MUST be captured here as a patch, or it exists only
as an uncommitted working-tree edit and is lost on a fresh checkout.

## pathorob-enable-bootstrap-ri.patch

Applies to `third_party/PathoROB` at upstream commit **6583cf0**
("Merge pull request #10 from bifold-pathomics/feature-genbio_results").

    cd third_party/PathoROB && git apply ../../patches/pathorob-enable-bootstrap-ri.patch

**Why:** PathoROB's `bootstrapped_robustness_index()` gives a WITHIN-RUN standard error for
`robustness_index` by resampling queries — no seed replicates needed. Our checkpoint-selection
rule ("earliest checkpoint within 1 SE of best-so-far") requires it. The code path had
**never been executed by anyone**, upstream or here, and carried three defects:

1. `robustness_index_utils.py:765` re-indexed `robustness_index-mean` *after* the previous
   line had collapsed it to a scalar -> `IndexError: invalid index to scalar variable`.
2. Same site assigned the **mean** into the **std** field.
3. `robustness_index.py:388` then called `len()` on that scalar -> `TypeError`, killing the
   multi-dataset loop after the per-dataset JSON was already written.

All three are on the print/aggregate path. **`robustness_index` itself is untouched** —
verified byte-identical before and after on a real checkpoint
(camelyon 0.7828883387852533, tolkach_esca 0.9385187529672417, plus k_opt,
balanced_accuracy, ID/OOD_performance, generalization_index, bio_vs_confounding and
confounder_insensitivity all matching to the last digit). This matters because PathoROB is
invoked unmodified so our RI stays comparable to their published leaderboard.

**WARNING — the "pinned" snapshots are hardlinks, not copies.**
`waiv-snapshots/*-pinned/third_party/PathoROB` shares inodes with this tree
(verified: robustness_index_utils.py 72057594059451952, robustness_index.py
144115188097378941, data/features 25048248). Editing either side changes both. Do not
assume a `*-pinned` directory isolates you from a repo-side edit — check `stat -c %i`.
