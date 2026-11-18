# ECD QC v2 exploratory GRPO input

This dataset is for **exploratory training-signal checks**. C's 1,700-item audit cleared MMMU/MMMU-Pro leakage within its documented scope; it did **not** approve item-level gold answers. The 11/19 initial ECD run and manifest are preserved separately.

## Location and hashes

| Artifact | Location | SHA256 |
|---|---|---|
| New JSONL manifest | `/data/chaewon/mmdl_grpo_data/rl_ablation_20261119_ecd_qc_v2/ecd_manifest.jsonl` | `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed` |
| C-audited priority800 | `/data/chaewon/mmdl_grpo_data/ecd_10k/b_selection_20261118/priority_800.jsonl` | `e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f` |
| ECD image archive | `/data/chaewon/mmdl_grpo_data/ecd_10k/source/images.tar` | `8ae3d9fb5b8345a17f4df420d50b662d5102cc54d46539870b0becad92c09cc1` |
| C item audit | `/data/chaewon/mmdl_grpo_data/rl_ablation_inputs_20261119/item_screening_status.jsonl` | `008d9e3a2bba1ed58d9bd9693d0af6b589a390634a0d35b42fd7406e27189016` |
| Prior ECD manifest | `/data/chaewon/mmdl_grpo_data/rl_ablation_20261119/ecd_manifest.jsonl` | `f189d58b8b0e6019387b4dd0ae61b5c6930e74a646f99b36ec9c5b8c2a70f1f8` |

The 597 included PNGs under `/data/chaewon/mmdl_grpo_data/ecd_10k/images/` were individually decoded and hashed. Each has one selected question. The audit archive hash was checked against C's input provenance.

## Selection

From 800 audited ECD candidates, remove questions containing whole-word `average`, `mean`, or `median` (200, including image `008560`). Remove charts `000259` (confirmed wrong Region E maximum), `003223` (matrix sum actually 500, gold 465), and `004222` (exact Sem3 cumulative-area values unresolved). The resulting deterministic hash split has **500 train + 97 heldout**. Two previously rejected charts (`002406`, `000838`) were already absent from the priority800 input.

Relative to the prior 600-item ECD manifest, 444 IDs remain, 153 are new, and 156 are removed. The full changes are in `added_vs_old_ids.txt`, `removed_vs_old_ids.txt`, and `excluded.jsonl`. The priority800 metadata records no exact SFT1-question or DPO-pool use for its 800 IDs.

## Gold-quality limit

A fixed SHA256 sample of 30 non-average charts was visually compared with source question, rationale and final answer. One had confirmed wrong gold (`003223`), one exact answer was unresolved (`004222`), and 28 had no clear final-answer contradiction. Both flagged charts are excluded. Earlier 10-image sampling of the old selection found wrong gold in `000259` and `008560`; both charts are also excluded. See `QUALITY_REVIEW.md` and `visual_sample30_review.jsonl` for evidence and image hashes.

This sampling **does not establish that the remaining 597 gold answers are correct**. Training results must be described as exploratory and may be affected by label noise. `visual_gold_quality_verified=false` and `c_training_approval=false` are deliberate manifest fields.

## Rebuild

Use `../build_ecd_qc_v2.py` with the paths above and an unused `--output-dir`. The builder verifies the pinned candidate and archive SHA256, C per-item leakage status, unique charts, image decode/hash, split counts, and writes a selection report.
