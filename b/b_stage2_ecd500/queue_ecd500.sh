#!/usr/bin/env bash
# One-shot queue: completed GRPO -> DAPO full -> sequential 300-item screens.
# This does not launch anything in --dry-run mode. The live mode must be started once.
set -euo pipefail

stage2=/data/chaewon/mmdl_stage2_ecd500_20261120
py=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
grpo=/data/chaewon/mmdl_stage2_runs/ecd500_grpo_full_20261121
dapo=/data/chaewon/mmdl_stage2_runs/ecd500_dapo_core_full_20261121
queue_dir=/data/chaewon/mmdl_stage2_runs/ecd500_queue_20261121
eval_root=/data/chaewon/mmdl_eval_runs
grpo_tag=stage2_ecd500_grpo_20261121
dapo_tag=stage2_ecd500_dapo_core_20261121
max_wait_seconds=${MMDL_QUEUE_MAX_WAIT_SECONDS:-21600}

if [[ $# -eq 1 && "$1" == --dry-run ]]; then
  bash -n "$stage2/run_ecd500.sh"
  bash -n "$stage2/evaluate_full_candidate.sh"
  "$py" -m py_compile "$stage2/queue_gate.py" "$stage2/queue_status.py"
  printf 'GRPO gate: %s\n' "$("$py" "$stage2/queue_gate.py" training "$grpo" grpo)"
  printf 'DAPO gate: %s\n' "$("$py" "$stage2/queue_gate.py" training "$dapo" dapo_core)"
  printf 'GPU used MiB: %s\n' "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash queue_ecd500.sh [--dry-run]' >&2
  exit 2
fi
[[ "$max_wait_seconds" =~ ^[0-9]+$ ]] || { echo 'Bad wait limit' >&2; exit 2; }
[[ -f "$stage2/run_ecd500.sh" && -f "$stage2/evaluate_full_candidate.sh" ]] || {
  echo 'Stage2 launcher or evaluation script is missing' >&2; exit 1;
}
for tag in "$grpo_tag" "$dapo_tag"; do
  [[ ! -e "/data/chaewon/mmdl_merged_models/$tag" &&
     ! -e "$eval_root/${tag}_smoke20" && ! -e "$eval_root/${tag}_fixed300" ]] || {
    echo "Evaluation output for $tag already exists" >&2; exit 1;
  }
done
mkdir -p "$queue_dir"
mkdir "$queue_dir/lock" || { echo "Queue is already claimed: $queue_dir/lock" >&2; exit 75; }
phase=starting
record() {
  if [[ $# -eq 3 ]]; then
    "$py" "$stage2/queue_status.py" "$queue_dir" "$1" "$2" --exit-code "$3"
  else
    "$py" "$stage2/queue_status.py" "$queue_dir" "$1" "$2"
  fi
}
on_exit() {
  local code=$?
  trap - EXIT
  if (( code != 0 )); then record "$phase" failed "$code" || true; fi
  exit "$code"
}
trap on_exit EXIT

wait_training() {
  local method=$1 dir=$2 state deadline=$((SECONDS + max_wait_seconds))
  while true; do
    state=$("$py" "$stage2/queue_gate.py" training "$dir" "$method")
    case "$state" in
      complete) return 0 ;;
      failed) echo "Training completion gate failed: $dir" >&2; return 1 ;;
      pending) ;;
      *) echo "Unexpected training gate: $state" >&2; return 1 ;;
    esac
    (( SECONDS < deadline )) || { echo "Training gate timed out: $dir" >&2; return 1; }
    sleep 30
  done
}

wait_gpu_idle() {
  local used consecutive=0 deadline=$((SECONDS + 600))
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
    [[ "$used" =~ ^[0-9]+$ ]] || { echo "Invalid GPU memory sample: $used" >&2; return 1; }
    if (( used <= 4096 )); then ((consecutive+=1)); else consecutive=0; fi
    (( consecutive >= 2 )) && return 0
    (( SECONDS < deadline )) || { echo "GPU did not become idle: $used MiB" >&2; return 1; }
    sleep 10
  done
}

phase=wait_grpo_full
record "$phase" running
wait_training grpo "$grpo"
record "$phase" complete

phase=wait_gpu_for_dapo
record "$phase" running
wait_gpu_idle
record "$phase" complete

phase=dapo_full
record "$phase" running
[[ ! -e "$dapo" ]] || { echo "Refusing to reuse DAPO run directory: $dapo" >&2; exit 1; }
bash "$stage2/run_ecd500.sh" dapo_core full "$dapo" > "$queue_dir/dapo_full.log" 2>&1
[[ "$("$py" "$stage2/queue_gate.py" training "$dapo" dapo_core)" == complete ]] || {
  echo "DAPO launch returned without a valid completion report: $dapo" >&2; exit 1;
}
record "$phase" complete

phase=wait_gpu_for_grpo_eval
record "$phase" running
wait_gpu_idle
record "$phase" complete

phase=grpo_eval300
record "$phase" running
[[ ! -e "$eval_root/${grpo_tag}_fixed300" ]] || { echo 'GRPO eval output already exists' >&2; exit 1; }
bash "$stage2/evaluate_full_candidate.sh" "$grpo" "$grpo_tag" > "$queue_dir/grpo_eval300.log" 2>&1
[[ "$("$py" "$stage2/queue_gate.py" evaluation "$eval_root/${grpo_tag}_fixed300/comparison/comparison_summary.json")" == complete ]]
record "$phase" complete

phase=wait_gpu_for_dapo_eval
record "$phase" running
wait_gpu_idle
record "$phase" complete

phase=dapo_eval300
record "$phase" running
[[ ! -e "$eval_root/${dapo_tag}_fixed300" ]] || { echo 'DAPO eval output already exists' >&2; exit 1; }
bash "$stage2/evaluate_full_candidate.sh" "$dapo" "$dapo_tag" > "$queue_dir/dapo_eval300.log" 2>&1
[[ "$("$py" "$stage2/queue_gate.py" evaluation "$eval_root/${dapo_tag}_fixed300/comparison/comparison_summary.json")" == complete ]]
record "$phase" complete

phase=finished
record "$phase" complete
