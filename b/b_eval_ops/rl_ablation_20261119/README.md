# B 6000 v1.2 screening queue (2026-11-19)

This queue starts **only after all listed training runs finish** and the 6000
GPU is idle. It never changes the frozen `MMDL_eval_v1_2` worktree. It uses the
same `run_mmmu_eval.sh`, image pipeline, generation settings, and v1.2 scorer.
The diagnostic task filters IDs; it is not an official MMMU result.

## Fixed 300 questions

`make_fixed300_ids.py` reads the saved 6000 base seed-3407 MMMU validation
results (900 IDs, 30 subjects × 30). It chooses 10 IDs per subject using
SHA256 of `3407\0<id>`, without looking at correctness. The committed
`fixed300_ids_seed3407.txt` has SHA256
`1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d`.
The exact selection and source sample hash are in its sidecar manifest.

`run_fixed300.py` runs the existing `mmmu_val_mmdl_diag` task with
`MMDL_DIAG_IDS_FILE` and `MMDL_DIAG_MIN_PIXELS=4096`. The task includes the
frozen v1.2 task, so the prompt, generation kwargs, and scorer are shared.
The queue first evaluates the base on these 300 IDs and pairs it with the
existing full-900 base run; all 300 input strings and document hashes must
match. Each candidate is then paired with this same base-300 run. No `--limit`
or first-300 ordering is used.

## GPU queue

Inputs are completed `train.py` run manifests. The queue validates base
revision, seed, 1,000 optimizer steps, 4 rollouts, two iterations, 512-token
training cap, reward-policy SHA, finite loss, mixed rewards, adapter SHA, and
reload. Each candidate then runs sequentially:

1. PEFT LoRA merge to the pinned base with `merge_lora.py`; complete merged
   model manifest and adapter SHA are checked.
2. v1.2 `--limit 20` smoke (functionality only, not ranking).
3. Fixed 300-ID diagnostic screen on seed 3407; save paired CSV and summary.
4. Select top two by 300-ID correct count (tie: fewer missing final answers)
   **for screening only**. Run full 900 seed 3407 for those two.
5. Run full 900 seed 1 for the seed-3407 leader. Also run the runner-up on seed
   1 when the 3407 margin is at most 9 questions, the leader has nonpositive
   3407 delta, or its seed-1 delta reverses. Final official 4090 three-seed
   evaluation remains a separate team step.

All stages have distinct run names, subprocess timeouts, persistent
`events.jsonl`, `progress.json`, `logs/*.log`, evaluator `stdout.log`, run
manifest, sample JSONL, per-item CSV, and SHA hashes. On restart, complete
matching stages are verified and skipped. An incomplete directory is preserved
and stops the queue for inspection; it is never silently overwritten.

Example after training, with the six actual paths substituted:

```bash
TRAIN_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
EVAL_PY=/data/chaewon/mmdl_envs/vllm011_repro/bin/python
EVAL_REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
MMMU=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
OPS=/data/chaewon/mmdl_b_eval_ops
"$TRAIN_PY" "$OPS/rl_ablation_20261119/run_eval_queue.py" \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_ecd_grpo/run_manifest.json \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_ecd_dapo/run_manifest.json \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_ecd_sapo/run_manifest.json \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_mammoth_grpo/run_manifest.json \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_mammoth_dapo/run_manifest.json \
  --train-manifest /data/chaewon/mmdl_grpo_runs/EXAMPLE_mammoth_sapo/run_manifest.json \
  --base-model "$BASE" --repo-root "$EVAL_REPO" --data-root "$MMMU" \
  --train-python "$TRAIN_PY" --eval-python "$EVAL_PY" \
  --output-root /data/chaewon/mmdl_eval_runs \
  --merged-root /data/chaewon/mmdl_merged_models \
  --baseline-900 /data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117 \
  --compare-script /data/chaewon/mmdl_b_compare/compare_runs.py \
  --plan-dir /data/chaewon/mmdl_eval_runs/rl_ablation_20261119_plan \
  --run-prefix rlab19
```

Use `--dry-run` first after all training manifests are available. The log
monitor is `tail -f /data/chaewon/mmdl_eval_runs/rl_ablation_20261119_plan/events.jsonl`.
No queue was launched when these scripts were prepared.
