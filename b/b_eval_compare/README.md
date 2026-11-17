# Paired MMMU run comparison

The 6000 server has a clean v1.2 evaluation worktree at
`/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2`, a separate Python
environment at `/data/chaewon/mmdl_envs/vllm011_repro/bin/python`, and this
comparison script at `/data/chaewon/mmdl_b_compare/compare_runs.py`. The script
imports the repository's v1.2 `is_correct()` function to rescore both M4_final
and the 6000 run. A self-comparison of M4_final passed for all three seeds.

The 6000 v1.2 three-seed baseline is complete (558/555/559 correct; mean
61.93%). Its paired 4090 comparison and operational reproduction verdict are
in [the baseline report](b6000_baseline_20261117/REPORT.md).

The C-approved 104-row ECD **base→GRPO** pilot and its 6000 full 900-item,
single-seed screening result are in [the real ECD pilot report](base_ecd_pilot_001_screen/REPORT.md).
This is separate from the earlier [synthetic base insurance result](base_insurance_001_screen/REPORT.md).
The real ECD base-to-GRPO pilot took about 132 seconds end-to-end for 20 optimizer steps (Trainer 119.68 seconds); the 900-item seed 3407 screening evaluation took 46 minutes 37 seconds. A run from the selected SFT parent may take a different amount of time.

```bash
/data/chaewon/mmdl_envs/vllm011_repro/bin/python /data/chaewon/mmdl_b_compare/compare_runs.py \
  --repo-root /home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2 \
  --baseline-run /home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2/results/runs/M4_final \
  --candidate-run /data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117 \
  --output-dir /data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117/comparison
```

For a `--limit 20` smoke run, add `--allow-partial`. Partial comparisons are
diagnostic; they cannot establish 900-question score agreement. Both output
files join on the **same seed and question ID**, and include score changes,
parsed-answer equality, missing final answers, and any explicit length stop
metadata. Saved lmms-eval JSONL may omit `finish_reason`; the script reports
that field as unavailable instead of inferring truncation.

The controlled 512-token/64-step tuning run, its one-seed 6000 screen, and both paired comparison files are in [the B tuning report](base_ecd_cap512_steps64_001_screen/REPORT.md). This is a base→GRPO insurance experiment; the 4090 official three-seed result and same-SFT-parent GRPO are still separate.
