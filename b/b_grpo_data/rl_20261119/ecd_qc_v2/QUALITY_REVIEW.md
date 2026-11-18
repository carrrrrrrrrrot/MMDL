# ECD exploratory gold-quality recovery (2026-11-19)

This is a B-side exploratory filter, **not** C training approval or a full gold review. The old 2026-11-19 manifest and failed GRPO run remain untouched.

## Inputs and filter

- C-audited ECD priority800: SHA256 `e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f`; C's scope was MMMU leakage only.
- ECD image archive: SHA256 `8ae3d9fb5b8345a17f4df420d50b662d5102cc54d46539870b0becad92c09cc1`. All included extracted PNGs were decoded and hashed. One question per chart.
- Remove questions containing whole-word `average`, `mean`, or `median` (200 rows, including chart 008560). Remove confirmed bad gold charts 000259 and 003223; also remove ambiguous chart 004222. Previously rejected charts 002406 and 000838 are absent from priority800.
- Result: 597 rows, deterministic hash split 500 train and 97 heldout. Compared with old 600-row ECD manifest: 444 retained, 153 added, 156 removed. Selected question IDs have no exact SFT1-question or DPO-pool overlap in the priority800 metadata.
- Manifest SHA256: `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed`.

## Visual evidence

A deterministic sample was drawn from the 600 non-average candidates by sorting SHA256(`ecd-qc-v2|3407|<id>`) and taking the first 30. Each PNG was visually inspected against the question, source rationale and final answer. Full item observations and image hashes are in `visual_sample30_review.jsonl`.

- `ecd_003223_229666`: **wrong gold**. The 4x4 annotated matrix sums to 500, while source rationale and final gold say 465. Removed.
- `ecd_004222_242239`: **unresolved exact value**. The Sem3 cumulative area chart does not establish all source component values precisely enough to affirm 848. Removed conservatively.
- 28 others: no clear final-answer contradiction found in this spot check. This does not validate their gold labels exhaustively. For example, `ecd_007081_278328` has plausible final answer 3 but the source rationale attributes the bars to an incorrect field.
- Earlier unfiltered 10-image sample found `ecd_000259_192242` gold 1.0 versus about 0.1 in the chart, and `ecd_008560_297572` gold 8 from a misread highlighted range. Both charts are excluded here.

The 30-item sample includes one definite wrong gold (and one ambiguous) even after removing average questions. The remaining 597 rows are **not** a verified clean training set. Any training on this manifest is exploratory, and possible label noise must be recorded when interpreting its results.
