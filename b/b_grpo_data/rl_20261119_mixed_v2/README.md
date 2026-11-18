# B mixed ECD + MAmmoTH CLEVR exploratory input

This file prepares a later 250/250 mixed-source RL comparison. **No mixed model has been trained or evaluated.** The manifest is a deterministic subset of the two quality-filtered B inputs; it does not repair or re-label any source answer.

## Inputs and selection

| Source | Pinned input SHA256 | Selected train | Selected heldout |
| --- | --- | ---: | ---: |
| ECD qc-v2, 500 train + 97 heldout | `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed` | 250 | 50 |
| MAmmoTH CLEVR v3, 500 train + 100 heldout | `b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d` | 250 | 50 |

`build_mixed_v2.mjs` ranks each source and split by SHA256 of the UTF-8 fields `mmdl-b-mixed-v2`, `3407`, source label, split and sample ID separated by NUL bytes. It takes the requested count, then puts selected train rows first and heldout rows second in a second fixed hash order. It copies each original JSONL row unchanged. This selection does not inspect answer correctness or model scores.

The output [mixed_manifest.jsonl](mixed_manifest.jsonl) has **500 train + 100 heldout** and SHA256 `68deeafa94f9f8fdb0f37107635c212520a0ff24dac86a8581fbfbbccbd1e465`. There are 600 unique sample IDs, original IDs, image SHA256 values and image paths; no image or ID crosses the split. The exact source hashes, selected counts and overlap counts are in [selection_report.json](selection_report.json).

## Interpretation limits

- C's audits addressed MMMU/MMMU-Pro leakage within their documented scopes. Neither source has C approval for training or an item-by-item gold answer audit. B's ECD spot check found incorrect/ambiguous source answers and filtered the known cases; undetected label noise may remain. B's CLEVR spot check and full image file/hash/decode validation do not establish that all CLEVR answers are correct.
- Of the 250 selected CLEVR training rows, **176 overlap A's DPO pool/pairs** and **59 were used in A's SFT1**. These are not new to A's experiments. The selected CLEVR heldout 50 all have `base_k_correct=4`, inherited from the source v3 split, so the mixed heldout set is not a representative ranking metric.
- This manifest keeps the same base model and v1.2 reward policy as the single-source runs. A mixed run should use a **new output root**, and method comparisons should use the same mixed manifest and training settings. Candidate ranking should rely on paired MMMU v1.2 screening and final 4090 evaluation, not the mixed heldout score alone.

## Rebuild

From the B repository root on Linux, with Node.js installed and the pinned source files present, choose a new empty output directory:

```bash
OUT_DIR=$(mktemp -d)
node b/b_grpo_data/rl_20261119_mixed_v2/build_mixed_v2.mjs \
  --output "$OUT_DIR/mixed_manifest.jsonl"
```

The builder refuses to overwrite a manifest or selection report. Compare the new file's SHA256 with the value above. Override source paths using `--ecd` and `--clevr` only when those files have the pinned hashes.
