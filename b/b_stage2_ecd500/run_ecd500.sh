#!/usr/bin/env bash
# Stage2 ECD500 single-GPU launch. CPU preflight runs before any GPU work.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash run_ecd500.sh grpo|dapo_core probe|full RUN_DIR" >&2
  exit 2
fi
method=$1
mode=$2
run_dir=$3
case "$method" in grpo|dapo_core) ;; *) echo "Bad method: $method" >&2; exit 2;; esac
case "$mode" in probe|full) ;; *) echo "Bad mode: $mode" >&2; exit 2;; esac

stage1_root=${MMDL_STAGE1_PACKAGE_ROOT:-/data/chaewon/mmdl_stage1_paper_style_20261120}
engine="$stage1_root/easyr1"
stage2_root=${MMDL_STAGE2_ROOT:-/data/chaewon/mmdl_stage2_ecd500_20261120}
train_python=${MMDL_TRAIN_PYTHON:-/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python}
base_model=${MMDL_BASE_MODEL:-/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
export PYTHONPATH="$engine${PYTHONPATH:+:$PYTHONPATH}"
export RAY_TMPDIR="${MMDL_RAY_TMPDIR:-/data/chaewon/ray_s1}"
mkdir -p "$RAY_TMPDIR"

"$train_python" "$stage2_root/preflight_ecd500.py" "$method" "$mode" "$run_dir" \
  --source-config "$stage2_root/config_ecd500.yaml" \
  --engine "$engine" --stage1 "$stage1_root" --base "$base_model"

export MMDL_V12_IMAGE_PARITY=1
export MMDL_VISIT_LOG_PATH="$run_dir/visits.jsonl"
export MMDL_PARAM_AUDIT_PATH="$run_dir/trainable_params.json"
printf 'timestamp_utc,gpu_index,used_mib,total_mib,utilization_pct\n' > "$run_dir/gpu_telemetry.csv"
"$train_python" "$stage2_root/update_run_manifest.py" "$run_dir" running

gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
if [[ ! "$gpu_used_mib" =~ ^[0-9]+$ ]] || (( gpu_used_mib > 4096 )); then
  echo "GPU is not idle: memory.used=${gpu_used_mib:-unknown} MiB" >&2
  "$train_python" "$stage2_root/update_run_manifest.py" "$run_dir" finished \
    --exit-code 75 --wall-seconds 0
  exit 75
fi

telemetry() {
  while true; do
    local stamp
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=, read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$stamp" "${index// /}" "${used// /}" "${total// /}" "${util// /}"
      done >> "$run_dir/gpu_telemetry.csv" || true
    sleep 5
  done
}
telemetry &
telemetry_pid=$!
start_seconds=$SECONDS

on_exit() {
  local code=$?
  trap - EXIT
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  local elapsed=$((SECONDS - start_seconds))
  "$train_python" "$stage2_root/update_run_manifest.py" "$run_dir" finished \
    --exit-code "$code" --wall-seconds "$elapsed" || true
  if (( code == 0 )); then
    "$train_python" "$stage2_root/finalize_ecd500.py" "$run_dir" --stage1 "$stage1_root" || code=78
    if (( code != 0 )); then
      "$train_python" "$stage2_root/update_run_manifest.py" "$run_dir" finished \
        --exit-code "$code" --wall-seconds "$elapsed" || true
    fi
  fi
  exit "$code"
}
trap on_exit EXIT

cd "$engine"
"$train_python" -m verl.trainer.main "config=$run_dir/resolved_config.yaml" \
  2>&1 | tee "$run_dir/console.log"
