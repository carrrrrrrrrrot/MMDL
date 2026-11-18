#!/usr/bin/env bash
# Paired real-ECD pilot. The 104 examples are C's already approved manifest.
# No synthetic data or MMMU training examples enter this run.
set -euo pipefail

python_bin=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
code_dir=/data/chaewon/mmdl_grpo_data/ecd_10k/b_selection_20261118/compare_code
dataset=/data/chaewon/mmdl_grpo_data/ecd_10k/base_ecd_pilot_001/grpo_manifest.jsonl
parent=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
root=/data/chaewon/mmdl_grpo_runs/ecd104_grpo_sapo_compare_20261118
mkdir -p "$root"

for loss in grpo sapo; do
  output="$root/$loss"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] starting $loss on C-approved real ECD 104" | tee -a "$root/progress.log"
  "$python_bin" "$code_dir/train.py" \
    --dataset-manifest "$dataset" \
    --reward-policy "$code_dir/reward_policy.v12.json" \
    --parent-kind base \
    --parent-model "$parent" \
    --output-dir "$output" \
    --loss-type "$loss" \
    --num-iterations 2 \
    --max-steps 64 \
    --learning-rate 1e-5 \
    --max-completion-length 512 \
    --num-generations 4 \
    --save-steps 64 \
    --seed 3407 2>&1 | tee "$root/$loss.log"
  echo "[$(date --iso-8601=seconds)] completed $loss" | tee -a "$root/progress.log"
done
