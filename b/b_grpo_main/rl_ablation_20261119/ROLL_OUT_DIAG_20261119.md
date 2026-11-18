# ECD GRPO completion snapshot (2026-11-19)

The 6000 server's `rl_ablation_20261119_ecd_qc_v2` run was read while training
continued. This is a **step 1–106 snapshot**, not the final run result. All 106
`trainer/completions/completions_*.parquet` files in the snapshot were readable.
Each file contained four rollouts, giving 424 completion rows from 53 unique
training questions (two optimizer iterations can repeat a question).

| Check | Snapshot result |
| --- | ---: |
| Correct (`reward_reason`) | 136 / 424 |
| Wrong answer | 228 / 424 |
| Missing final `Answer:` | 60 / 424 (14.15%) |
| Qwen tokenizer re-encoded completion length = 512 | 60 / 424 |
| Both missing `Answer:` and length = 512 | 60 / 424 |

The 60 missing-answer completions all reached the configured 512-token
generation cap; all 512-token completions lacked an `Answer:` line. Their
responses typically end mid-explanation. The missing-answer rows come from 13
distinct question IDs. This establishes a truncation pattern for this snapshot;
it does not establish whether a longer generation would have reached a correct
answer. The counts are rollout rows, not independent questions.

Reproduce on the 6000 server with the GRPO Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/chaewon/mmdl_envs/grpo_pilot/bin/python \
  /data/chaewon/mmdl_b_main_code/rl_ablation_20261119/rollout_diag.py \
  --through-step 106
```

The script reads completion parquet files and the local Qwen tokenizer, then
prints JSON. Omitting `--through-step` takes a new snapshot of the ongoing run.

The `answer_marker_missing`/`strict_line_start_answer_missing` count requires a
line beginning with `Answer:`. The actual v1.2 `missing_final_answer` reward
count can be lower: v1.2 also accepts endings such as `Final Answer: 1` by
extracting the last `Answer:` substring. Use `v12_missing_final_answer` or the
`reward_reason` counts for training-outcome reporting.
