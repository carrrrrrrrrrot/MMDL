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

The [2026-11-19 RL ablation fixed-300 screens](rl_ablation_20261119_qcv2/REPORT.md)
compare six ECD or MAmmoTH CLEVR candidates on the same 300 questions and
seed. They are diagnostic triage runs. The user released the top-two full-900
review gate on 2026-11-20, then stopped the first ECD GRPO evaluation before
completion to test a larger ECD training set. No full-900 score resulted;
the ECD DAPO full-900 evaluation never started.

## Completed 6000 screens

Each result below is an MMMU val **900-item evaluation**, compared with the
untrained base on the same GPU and seed. A selected the untrained base as the
common parent on 2026-11-18, after rejecting all SFT candidates. These
base→GRPO runs are exploratory; the synthetic run uses different training data
and reward from the C-approved ECD runs.

| Candidate | Training data and setup | Seed 3407 | Seed 1 |
|---|---|---:|---:|
| [Synthetic insurance](base_insurance_001_screen/REPORT.md) | 1,344 generated images, 128 steps, synthetic strict reward | 566 vs base 558 (+8) | Not run |
| [ECD pilot](base_ecd_pilot_001_screen/REPORT.md) | C-approved 104 items, 256-token cap, 20 steps | 564 vs 558 (+6) | Not run |
| [ECD 512/64](base_ecd_cap512_steps64_001_screen/REPORT.md) | Same 104 items, 512-token cap, 64 steps | 570 vs 558 (+12) | [550 vs 555 (−5)](base_ecd_cap512_steps64_seed1_20261118_screen/REPORT.md) |
| [ECD 1024/64](base_ecd_cap1024_steps64_20261118_screen/REPORT.md) | Same 104 items, 1024-token cap, 64 steps | 564 vs 558 (+6) | Not run |

The ECD 512/64 seed-1 result misses A's **564/900** gate for a 4090 official
candidate. Across seeds 3407 and 1, it scores 1,120/1,800 versus the 6000
base's 1,113/1,800, a gain of 7 items; this also falls short of A's proposed
two-seed +9 gate (1,122/1,800). Its seed-3407 gain is not a stable improvement
claim. The [two-seed error analysis](base_ecd_512_two_seed_error_20261118/REPORT.md)
separates recovered missing answers from losses on previously answered items.
The synthetic run used different data and a reward policy that was not
identical to v1.2;
keep it separate from the C-approved ECD runs. The ECD pilot took about 132
seconds end to end for 20 training steps, and its seed-3407 screen took 46
minutes 37 seconds. A run from the selected SFT parent may take longer.

The [512/1024-token and learning-rate sweep](../b_grpo_main/base_ecd_ablation_20261118/REPORT.md)
contains training-only 5e-6 and 2e-5 candidates with **no 900-item score**.
The [4-versus-8 rollout diagnostic](../b_grpo_main/base_ecd_numgen8_20261118/REPORT.md)
also has no 900-item score.

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

The final official comparison remains a 4090 v1.2 three-seed evaluation of
models starting from the selected base parent. The SFT-adapter watcher is
inactive after A's base-parent decision; further GRPO training uses the base
path directly.
