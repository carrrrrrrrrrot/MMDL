# B RL ablation ledger (2026-11-19)

This ledger tracks actual runs. Server `plan.json`, `run_manifest.json`, adapter SHA256, `events.jsonl`, status and trainer logs are authoritative. **An unfinished run has no valid model score.** All runs use the pinned Qwen3-VL-4B-Instruct base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, v1.2 0/1 reward, seed 3407, BF16 language LoRA r16/alpha32, four rollouts, two iterations, 512 completion tokens, learning rate 1e-5 and up to 1,000 optimizer steps unless a later run manifest says otherwise.

## Method scope and implicit trainer settings

This is a comparison of three loss choices in the same TRL 1.14.2 GRPO trainer: `loss_type=grpo`, `loss_type=dapo`, and `loss_type=sapo`. The DAPO arm changes to TRL's token-level loss normalization; it does **not** reproduce the full DAPO training recipe (for example, asymmetric upper clipping, truncated-completion masking, or dynamic sampling). The SAPO arm uses TRL's advantage-sign-dependent soft gate with positive/negative temperatures 1.0/1.05. It tests the core SAPO loss under this small-run setup, not the paper's full training scale or a temperature sweep.

The current `train.py` explicitly passes the loss type and SAPO temperatures. It leaves these installed TRL 1.14.2 defaults implicit: `epsilon=0.2`, `epsilon_high=None` (so the effective upper clipping bound uses the same 0.2), `mask_truncated_completions=False`, and `scale_rewards='group'`. These values are **not explicitly recorded in the current run manifests**; use this paragraph together with the pinned trainer version and code when interpreting results. Truncation handling matters because many ECD and CLEVR responses reached the 512-token cap. Future changes to these settings should be separate, versioned experiments.

| Dataset/version | Method | Status | Training output | Input SHA256 | Evaluation |
| --- | --- | --- | --- | --- | --- |
| Initial ECD v1 | GRPO | **Stopped by B at 140/1,000** after image/gold errors; no validated adapter | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119/runs/ecd_grpo` | `f189d58b8b0e6019387b4dd0ae61b5c6930e74a646f99b36ec9c5b8c2a70f1f8` | None |
| ECD quality filter v2, 500 train + 97 heldout | GRPO | **Complete**, 1,000/1,000 steps; saved and reloaded adapter | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_grpo` | `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed` | Pending |
| ECD quality filter v2 | DAPO | **Complete**, 1,000/1,000 steps; saved and reloaded adapter at 2026-11-19 05:56 KST | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_dapo` | Same ECD v2 | Pending |
| ECD quality filter v2 | SAPO | **Running**; started automatically after DAPO at 2026-11-19 05:56 KST | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_sapo` | Same ECD v2 | Pending |
| MAmmoTH CLEVR v3, 500 train + 100 heldout | GRPO | **Complete**, 1,000/1,000 steps; saved and reloaded adapter at 2026-11-19 04:30 KST | `/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_clevr_v3/runs/mammoth_grpo` | `b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d` | Pending |
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
| 1,000 | 500 | 4,000 | 1,178 | 2,092 | 730 | Final run and saved PEFT adapter; 1,000 completion parquet files |

All 150 missing-answer responses through step 250 reached the 512-token generation cap. Through step 500, 362/364 missing-answer responses reached that cap; two did not, and four of the 366 512-token responses contained an `Answer:` marker. This is a dominant truncation pattern with exceptions. These checkpoints are intermediate, not validated final adapters, and no MMMU evaluation has been run. See [ROLL_OUT_DIAG_20261119.md](ROLL_OUT_DIAG_20261119.md) and its read-only parser.

Through step 750, 546/548 missing-answer responses reached the 512-token cap; six of 552 cap-length responses contained an `Answer:` marker. The step-750 checkpoint is also intermediate and has no MMMU score.

The **completed** ECD GRPO run has 75/500 distinct four-rollout question groups with mixed 0/1 rewards; mean rollout reward is 0.2945. Its training loss was finite, LoRA L1 weight change was 2367.97, and the saved adapter reloaded successfully. Adapter SHA256: `0513533927091e00ea979ec3dc63d9cfcc97cefc89beb17cca847405ed673b8f`. Of 730 final-answer-missing completions, 726 reached the 512-token cap; six of 732 cap-length completions contained an `Answer:` marker. The run has **no MMMU score yet**, so no performance improvement is claimed.

## ECD DAPO progress snapshot (not evaluation)

At optimizer step 100, 50 unique questions yielded 400 responses across 100 readable completion parquet files: 130 correct rewards, 224 wrong answers, and 46 v1.2 `missing_final_answer` reward reasons. All 46 missing-answer responses reached the 512-token generation cap. This is an intermediate diagnostic only: the DAPO run has no validated final adapter or MMMU score yet.

At optimizer step 250, 125 unique questions yielded 1,000 responses across 250 readable completion parquet files: 330 correct rewards, 542 wrong answers, and 128 v1.2 `missing_final_answer` reward reasons. All 128 missing-answer responses reached the 512-token cap, and `runs/ecd_dapo/trainer/checkpoint-250` exists. At the same step count on the same ECD v2 manifest, the GRPO arm had 316 correct rewards, 534 wrong answers, and 150 missing final answers (all 150 at the cap). These are stochastic **training rollout distributions**, not paired MMMU scores or evidence that DAPO improves generalization. Both runs still await the common v1.2 evaluation.

At optimizer step 500, 250 unique questions yielded 2,000 responses across 500 readable completion parquet files: 624 correct rewards, 1,088 wrong answers, and 288 v1.2 `missing_final_answer` reward reasons. All 288 missing-answer responses reached the 512-token cap, and `runs/ecd_dapo/trainer/checkpoint-500` exists. At step 500 on the same ECD v2 manifest, GRPO had 622 correct rewards, 1,014 wrong answers, and 364 missing final answers (362/364 at the cap). DAPO had fewer missing final answers in these stochastic training rollouts, while correct rewards were nearly the same. This is **not** a paired evaluation or a generalization result; both arms await MMMU v1.2 evaluation.

At optimizer step 750, 375 unique questions yielded 3,000 responses across 750 readable completion parquet files: 918 correct rewards, 1,640 wrong answers, and 442 v1.2 `missing_final_answer` reward reasons. All 442 missing-answer responses reached the 512-token cap, and `runs/ecd_dapo/trainer/checkpoint-750` exists. At the same step count on the same ECD v2 manifest, GRPO had 918 correct rewards, 1,534 wrong answers, and 548 missing final answers (546/548 at the cap). Correct rewards were identical; DAPO had fewer missing answers and more wrong answers in these stochastic training rollouts. These are intermediate training diagnostics, **not** MMMU evaluation scores or evidence of a generalization gain.

The **completed** ECD DAPO run covered 500 unique training questions and 4,000 responses over 1,000 optimizer steps: 1,190 correct rewards, 2,194 wrong answers and 616 v1.2 `missing_final_answer` reward reasons. Of those 616 missing answers, 614 reached the 512-token cap. The same-manifest GRPO run recorded 1,178 correct, 2,092 wrong and 730 missing (726/730 at the cap). The correct-reward difference is small and these are stochastic training rollouts, not paired MMMU results. DAPO had mixed 0/1 rewards in 54/500 distinct four-rollout question groups and mean rollout reward 0.2975. Training loss was finite, LoRA L1 weight change was 2221.7934, peak CUDA memory was 11979 MiB, and the saved adapter reloaded successfully. Adapter SHA256: `8dffc2be7011092b839ca031659b6130ecd064682d908080be7ccf3ed08f2b73`. The run has **no MMMU score yet**, so no performance improvement is claimed. ECD SAPO started automatically after DAPO at approximately 2026-11-19 05:56 KST.

## ECD SAPO progress snapshot (not evaluation)

At optimizer step 100, 50 unique questions yielded 400 responses across 100 readable completion parquet files: 126 correct rewards, 224 wrong answers, and 50 v1.2 `missing_final_answer` reward reasons. All 50 missing-answer responses reached the 512-token generation cap. This is an intermediate training diagnostic, not an MMMU score or a validated final adapter.

At optimizer step 250, 125 unique questions yielded 1,000 responses across 250 readable completion parquet files: 324 correct rewards, 536 wrong answers, and 140 v1.2 `missing_final_answer` reward reasons. All 140 missing-answer responses reached the 512-token cap, and `runs/ecd_sapo/trainer/checkpoint-250` exists. On the same ECD v2 manifest at step 250, DAPO had 330 correct / 542 wrong / 128 missing and GRPO had 316 correct / 534 wrong / 150 missing. These small differences are stochastic **training rollout distributions**, not paired MMMU scores or evidence of a generalization gain.

At optimizer step 500, 250 unique questions yielded 2,000 responses across 500 readable completion parquet files: 612 correct rewards, 1,068 wrong answers, and 320 v1.2 `missing_final_answer` reward reasons. All 320 missing-answer responses reached the 512-token cap, and `runs/ecd_sapo/trainer/checkpoint-500` exists. On the same ECD v2 manifest at step 500, DAPO had 624 correct / 1,088 wrong / 288 missing and GRPO had 622 correct / 1,014 wrong / 364 missing (362/364 at the cap). Correct-reward counts remain similar. These are stochastic **training rollout distributions**, not paired MMMU scores or evidence of a generalization gain.

At optimizer step 750, 375 unique questions yielded 3,000 responses across 750 readable completion parquet files: 918 correct rewards, 1,624 wrong answers, and 458 v1.2 `missing_final_answer` reward reasons. All 458 missing-answer responses reached the 512-token cap, and `runs/ecd_sapo/trainer/checkpoint-750` exists. At the same step count on the same ECD v2 manifest, DAPO had 918 correct / 1,640 wrong / 442 missing, while GRPO had 918 correct / 1,534 wrong / 548 missing. All three correct-reward totals are numerically identical here; the paired-output audit below shows substantial response differences. These are intermediate training rollout distributions, **not** paired MMMU scores or evidence of a generalization gain.

### ECD step-750 paired rollout audit

The three runs have the same `sample_id`, prompt and gold answer in the same order for all 3,000 completion rows. Their equal correct-reward totals are due to offsetting item-level changes, not copied responses:

| Pair | Identical response text / 3,000 | Identical text in last 100 steps / 400 | Both correct | First only correct | Second only correct | Same extracted answer / 3,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO / DAPO | 86 | 4 | 816 | 102 | 102 | 1,808 |
| GRPO / SAPO | 78 | 2 | 826 | 92 | 92 | 1,848 |
| DAPO / SAPO | 88 | 0 | 830 | 88 | 88 | 1,858 |

This is a paired **training rollout** audit. It demonstrates different sampled outputs while the aggregate correct counts tie; it is not an MMMU evaluation or evidence that the methods generalize equally.

## CLEVR GRPO progress snapshot (not evaluation)

At optimizer step 100, 50 unique questions yielded 400 responses: 270 correct rewards, 102 wrong answers and 28 missing final answers. All 28 missing answers reached the 512-token cap. Thirteen of the 50 distinct four-rollout question groups had mixed 0/1 rewards (26/100 repeated optimizer-step records); gradient norm was nonzero in 26/100 records and loss in 12/100. No non-finite loss or gradient was observed. This is a partial training-signal check, not an MMMU result.

At optimizer step 250, 125 unique questions yielded 1,000 responses: 672 correct rewards, 234 wrong answers and 94 missing final answers. Of the missing answers, 92 reached the 512-token cap; two did not. `runs/mammoth_grpo/trainer/checkpoint-250` exists. This checkpoint is intermediate and has no MMMU score.

At optimizer step 500, 250 unique questions yielded 2,000 responses: 1,382 correct rewards, 424 wrong answers and 194 `missing_final_answer` reward reasons. Of those 194, 192 reached the 512-token cap. `runs/mammoth_grpo/trainer/checkpoint-500` exists. A stricter line-start `Answer:` regex counted 196 missing markers: two repeated completions ended with `Final Answer: 1`, which the v1.2 scorer accepted as correct because it extracts the last `Answer:` substring. Thus the v1.2 missing-answer count is **194**; the 196 strict-format count is a diagnostic, not the reward outcome. This is an intermediate checkpoint without an MMMU score.

At optimizer step 750, 375 unique questions yielded 3,000 responses across 750 readable completion parquet files: 2,104 correct rewards, 624 wrong answers and 272 v1.2 `missing_final_answer` reward reasons. Of those 272 missing answers, 268 reached the 512-token cap. The stricter line-start `Answer:` check counted 274 missing markers; this diagnostic differs from the v1.2 reward because the latter accepts `Final Answer:` via its last `Answer:` substring. `runs/mammoth_grpo/trainer/checkpoint-750` exists. This is an intermediate checkpoint without an MMMU score.

At step 250, 15/125 distinct four-rollout question groups had mixed 0/1 rewards; their two optimizer iterations give 30/250 step records with mixed rewards and nonzero gradient norm. Loss was nonzero in 15/250 records. No non-finite reward/loss/gradient/entropy values were observed in this checkpoint snapshot. These are training-signal checks, not evidence of an MMMU score gain.

The **completed** CLEVR GRPO run covered 500 unique training questions and 4,000 responses over 1,000 optimizer steps: 2,846 correct rewards, 792 wrong answers and 362 v1.2 `missing_final_answer` reward reasons. Of those 362 missing answers, 358 reached the 512-token cap. The stricter line-start marker check counted 364 missing responses. Mixed 0/1 rewards occurred in 148/500 distinct four-rollout question groups; mean rollout reward was 0.7115. Training loss was finite, LoRA L1 weight change was 2874.2023, peak CUDA memory was 9474 MiB, and the saved adapter reloaded successfully. Adapter SHA256: `a17954622f2e651e3b893c0ae5c977ec55299af13e30991c0febd2280e00f49d`. The automatic supervisor then advanced to ECD DAPO/SAPO at 2026-11-19 04:29:57 KST. CLEVR GRPO has **no MMMU score yet**, so no performance improvement is claimed.
