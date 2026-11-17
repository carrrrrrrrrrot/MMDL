# Paired MMMU run comparison

The 6000 server has a clean v1.2 evaluation worktree at
`/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2`, a separate Python
environment at `/data/chaewon/mmdl_envs/vllm011_repro/bin/python`, and this
comparison script at `/data/chaewon/mmdl_b_compare/compare_runs.py`. The script
imports the repository's v1.2 `is_correct()` function to rescore both M4_final
and the 6000 run. A self-comparison of M4_final passed for all three seeds.

As of 2026-11-17, the 20-question 6000 smoke run passed. Full seed 3407 is
running; `continue_baseline.sh` waits for its successful completion, then runs
seeds 1 and 2 and produces the paired comparison. **The 3-seed baseline has not
yet been declared reproduced.** The continuation log is
`/data/chaewon/mmdl_b_compare/continue_baseline.log`.

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
