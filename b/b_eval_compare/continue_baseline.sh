#!/usr/bin/env bash
# Continue the fixed 6000 v1.2 base-model baseline after seed 3407 finishes.
set -euo pipefail

repo=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
python=/data/chaewon/mmdl_envs/vllm011_repro/bin/python
model=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
data=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
output=/data/chaewon/mmdl_eval_runs
name=b6000_baseline_v12_20261117
first_meta="$output/$name/seed_3407/run_meta.txt"
first_dir="$output/$name/seed_3407"
export PYTHON="$python"
export HF_HOME=/home/chaewon/.cache/huggingface

missing_checks=0
for ((attempt=0; attempt<960; attempt++)); do
  if grep -q '^exit_status: ' "$first_meta" 2>/dev/null; then
    break
  fi
  if pgrep -f '^bash /home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2/scripts/run_mmmu_eval.sh .*--seeds 3407' >/dev/null; then
    missing_checks=0
  else
    ((missing_checks+=1))
    if ((missing_checks >= 4)); then
      if compgen -G "$first_dir/*/*_results.json" >/dev/null; then
        echo "seed 3407 process ended without exit_status but has a result; inspect $first_dir" >&2
        exit 1
      fi
      echo "seed 3407 process ended before producing a result; restarting detached"
      bash "$repo/scripts/run_mmmu_eval.sh" \
        --model_path "$model" \
        --data_root "$data" \
        --seeds 3407 \
        --output_path "$output" \
        --run_name "$name"
      break
    fi
  fi
  sleep 30
done
if ! grep -qx 'exit_status: 0' "$first_meta"; then
  echo "seed 3407 did not complete successfully; see $first_meta" >&2
  exit 1
fi

for seed in 1 2; do
  meta="$output/$name/seed_$seed/run_meta.txt"
  if grep -qx 'exit_status: 0' "$meta" 2>/dev/null; then
    echo "seed $seed already completed; skipping"
    continue
  fi
  bash "$repo/scripts/run_mmmu_eval.sh" \
    --model_path "$model" \
    --data_root "$data" \
    --seeds "$seed" \
    --output_path "$output" \
    --run_name "$name"
done

"$python" /data/chaewon/mmdl_b_compare/compare_runs.py \
  --repo-root "$repo" \
  --baseline-run "$repo/results/runs/M4_final" \
  --candidate-run "$output/$name" \
  --output-dir "$output/$name/comparison"
echo "6000 baseline and paired comparison completed"
