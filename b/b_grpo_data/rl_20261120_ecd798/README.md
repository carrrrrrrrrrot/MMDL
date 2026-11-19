# ECD 798-item exploratory data-size extension

This manifest retains the prior ECD QC-v2 **500 train / 97 heldout** rows and
adds **298 train** rows from A's SFT1 ECD list. The 298 have different chart
IDs from all 597 prior rows. The final split is **798 train / 97 heldout**.

The 298 new IDs are exact members of C's audited SFT1 ECD 5,000-item list.
C's final audit reported no MMMU/MMMU-Pro leakage within its documented
scope. This is a subset of that audited list, not a new C approval of B's
GRPO training set. Each new image's SHA256 matches C's SFT1 image hash table.
All 298 PNGs decoded fully with Pillow. In the frozen v1.2 reward code,
`Answer: <gold>` received reward 1 on all 298 rows.

**Gold quality remains unverified.** ECD source answers include known errors;
the new 298 have not received item-by-item visual gold review. Both
`c_training_approval` and `visual_gold_quality_verified` remain false.
The new rows exclude bare-number format failures, average/mean/median and
approximation questions, known bad charts, and A's DPO pool. Reuse of SFT1
questions is explicit. B's compared models have the original base as parent.

## Files

- `build_ecd798.mjs`: deterministic builder with pinned source, C audit, and
  previous QC-v2 hashes. It refuses an existing output directory.
- `ecd_manifest.jsonl`: complete 798 train / 97 heldout manifest. Image paths
  refer to `/data/chaewon/mmdl_grpo_data/ecd_10k/images` on the 6000 server.
- `added_sft1_ecd.jsonl`: IDs, image paths and hashes for the 298 new rows.
- `selection_report.json`: counts, audit provenance, and manifest SHA256.

Remote manifest: `/data/chaewon/mmdl_grpo_data/rl_ablation_20261120_ecd798/manifest/ecd_manifest.jsonl`.
The combined manifest SHA256 is
`f1570689440a7c5efb173be201096ec0e770caf9c85a18b2b5efda1ef3f11a21`.

The 11/20 full-pass training queue uses 1,596 optimizer steps and two
iterations per reward group, targeting all 798 training questions once.
The earlier ECD500 run used 1,000 steps for 500 reward groups. Both runs use
the same base revision, reward, rollout count, learning rate, and fixed-300
evaluation IDs. Data count **and total training compute** therefore change;
the result cannot isolate the effect of data count alone. A 1,000-step ECD798
attempt was stopped after 38 steps when this coverage issue was identified;
it produced no model used for comparison.

Training root: `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261120_ecd798_fullpass`.
The follow-on fixed-300 evaluator waits for both GRPO and DAPO training runs
and does not launch a 900-item evaluation.
