# B RL ablation ledger (2026-11-19)

This ledger tracks actual runs. Server `plan.json`, `run_manifest.json`, adapter SHA256, `events.jsonl`, status and trainer logs are authoritative. **An unfinished run has no valid model score.** All runs use the pinned Qwen3-VL-4B-Instruct base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, v1.2 0/1 reward, seed 3407, BF16 language LoRA r16/alpha32, four rollouts, two iterations, 512 completion tokens, learning rate 1e-5 and up to 1,000 optimizer steps unless a later run manifest says otherwise.

| Dataset/version | Method | Status | Training output | Input SHA256 | Evaluation |
| --- | --- | --- | --- | --- | --- |
| Initial ECD v1 | GRPO | **Stopped by B at 140/1,000** after image/gold errors; no validated adapter | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119/runs/ecd_grpo` | `f189d58b8b0e6019387b4dd0ae61b5c6930e74a646f99b36ec9c5b8c2a70f1f8` | None |
| ECD quality filter v2, 500 train + 97 heldout | GRPO | **Running**, started 2026-11-19 01:51:46 KST | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_grpo` | `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed` | Pending |
| ECD quality filter v2 | DAPO | Queued after GRPO signal check | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_dapo` | Same ECD v2 | Pending |
| ECD quality filter v2 | SAPO | Queued after GRPO signal check | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_sapo` | Same ECD v2 | Pending |
| MAmmoTH CLEVR v3, 500 train + 100 heldout | GRPO | Prepared; waiting for GPU | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_clevr_v3/runs/mammoth_grpo` | `b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d` | Pending |
| MAmmoTH CLEVR v3 | DAPO | Queued after GRPO signal check | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_clevr_v3/runs/mammoth_dapo` | Same CLEVR v3 | Pending |
| MAmmoTH CLEVR v3 | SAPO | Queued after GRPO signal check | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_clevr_v3/runs/mammoth_sapo` | Same CLEVR v3 | Pending |

ECD v2 and CLEVR v3 are **exploratory** datasets. C checked leakage within its documented scope; neither dataset has full item-level gold validation or C training approval. See the two dataset READMEs and [QC_ABORT_20261119.md](QC_ABORT_20261119.md). The easy-question-biased CLEVR heldout set is a regression check only. Candidate ranking will use fixed 300 MMMU v1.2 questions; top candidates will be checked on full 900 on the 6000, and official three-seed results belong to the 4090 run.

## ECD GRPO progress snapshots (not evaluation scores)

| Optimizer steps | Unique questions | Generated responses | Correct reward | Wrong answer | Missing final `Answer:` | Artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 106 | 53 | 424 | 136 | 228 | 60 | Trainer completion parquet snapshot |
| 250 | 125 | 1,000 | 316 | 534 | 150 | `runs/ecd_grpo/trainer/checkpoint-250`; 250 completion parquet files |
| 500 | 250 | 2,000 | 622 | 1,014 | 364 | `runs/ecd_grpo/trainer/checkpoint-500`; 500 completion parquet files |
| 750 | 375 | 3,000 | 918 | 1,534 | 548 | `runs/ecd_grpo/trainer/checkpoint-750`; 750 completion parquet files |

All 150 missing-answer responses through step 250 reached the 512-token generation cap. Through step 500, 362/364 missing-answer responses reached that cap; two did not, and four of the 366 512-token responses contained an `Answer:` marker. This is a dominant truncation pattern with exceptions. These checkpoints are intermediate, not validated final adapters, and no MMMU evaluation has been run. See [ROLL_OUT_DIAG_20261119.md](ROLL_OUT_DIAG_20261119.md) and its read-only parser.

Through step 750, 546/548 missing-answer responses reached the 512-token cap; six of 552 cap-length responses contained an `Answer:` marker. The step-750 checkpoint is also intermediate and has no MMMU score.

At step 250, 15/125 distinct four-rollout question groups had mixed 0/1 rewards; their two optimizer iterations give 30/250 step records with mixed rewards and nonzero gradient norm. Loss was nonzero in 15/250 records. No non-finite reward/loss/gradient/entropy values were observed in this checkpoint snapshot. These are training-signal checks, not evidence of an MMMU score gain.
