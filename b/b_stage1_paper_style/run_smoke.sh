#!/usr/bin/env bash
# One optimizer step per run. This launches GPU work only when invoked.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 grpo|dapo_core 512|1024|2048 TRAIN_JSONL RUN_DIR" >&2
  exit 2
fi

method=$1
cap=$2
train_file=$3
run_dir=$4

case "$cap" in 512|1024|2048) ;; *) echo "Unsupported cap: $cap" >&2; exit 2;; esac
case "$method" in
  grpo)
    online_filtering=false
    loss_avg_mode=seq
    clip_high=0.2
    mini_rollout_batch=null
    ;;
  dapo_core)
    online_filtering=true
    loss_avg_mode=token
    clip_high=0.28
    mini_rollout_batch=5
    ;;
  *) echo "Unsupported method: $method" >&2; exit 2;;
esac

easy_root=${MMDL_EASYR1_ROOT:-/tmp/mmdl_easyr1_stage1_20261120}
stage1_root=${MMDL_STAGE1_ROOT:-/tmp/mmdl_stage1_scripts_20261120}
train_python=${MMDL_TRAIN_PYTHON:-/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python}
parent_model=${MMDL_PARENT_MODEL:-/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
val_file=${MMDL_VAL_JSONL:-$stage1_root/ecd5_holdout_easyr1.jsonl}

if [[ ! -f "$train_file" || ! -f "$val_file" || ! -f "$stage1_root/config_smoke.yaml" ]]; then
  echo "Stage1 data or config missing" >&2
  exit 2
fi
if [[ -e "$run_dir/console.log" ]]; then
  echo "Run log already exists: $run_dir/console.log" >&2
  exit 2
fi
mkdir -p "$run_dir"
export RAY_TMPDIR="${MMDL_RAY_TMPDIR:-/data/chaewon/ray_s1}"
mkdir -p "$RAY_TMPDIR"
export MMDL_V12_IMAGE_PARITY=1
export MMDL_VISIT_LOG_PATH="$run_dir/visits.jsonl"
export MMDL_PARAM_AUDIT_PATH="$run_dir/trainable_params.json"
export PYTHONPATH="$easy_root${PYTHONPATH:+:$PYTHONPATH}"

"$train_python" "$stage1_root/write_run_manifest.py" \
  --run-dir "$run_dir" --engine "$easy_root" --stage1-root "$stage1_root" \
  --train "$train_file" --val "$val_file" --parent-model "$parent_model" \
  --method "$method" --cap "$cap"

cd "$easy_root"
"$train_python" -m verl.trainer.main \
  "config=$stage1_root/config_smoke.yaml" \
  "data.train_files=$train_file" \
  "data.val_files=$val_file" \
  "data.max_response_length=$cap" \
  "worker.rollout.max_num_batched_tokens=16384" \
  "data.rollout_batch_size=5" \
  "data.mini_rollout_batch_size=$mini_rollout_batch" \
  "worker.actor.global_batch_size=5" \
  "worker.actor.loss_avg_mode=$loss_avg_mode" \
  "worker.actor.clip_ratio_low=0.2" \
  "worker.actor.clip_ratio_high=$clip_high" \
  "worker.actor.model.model_path=$parent_model" \
  "worker.reward.reward_function=$stage1_root/reward_v12.py:compute_score" \
  "worker.reward.reward_function_kwargs.max_response_length=$cap" \
  "worker.reward.reward_function_kwargs.overlong_buffer_length=$((cap / 4))" \
  "algorithm.online_filtering=$online_filtering" \
  "algorithm.filter_key=accuracy" \
  "trainer.experiment_name=ecd20_${method}_cap${cap}" \
  "trainer.save_checkpoint_path=$run_dir/checkpoints" \
  2>&1 | tee "$run_dir/console.log"
