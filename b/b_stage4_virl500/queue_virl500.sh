#!/usr/bin/env bash
# One-shot Stage4 ViRL500 queue: probes, full training, smoke20, fixed300.
# --dry-run performs CPU-only gates; live execution is sequential on one GPU.
set -euo pipefail

stage4=/data/chaewon/mmdl_stage4_virl500_20261121
stage2=/data/chaewon/mmdl_stage2_ecd500_20261120
stage1=/data/chaewon/mmdl_stage1_paper_style_20261120
py=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
base=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
run_root=/data/chaewon/mmdl_stage4_runs
queue_dir="$run_root/virl500_queue_20261121"
eval_root=/data/chaewon/mmdl_eval_runs
merge_root=/data/chaewon/mmdl_merged_models
grpo_probe="$run_root/virl500_grpo_probe_20261121"
grpo_full="$run_root/virl500_grpo_full_20261121"
dapo_probe="$run_root/virl500_dapo_core_probe_20261121"
dapo_full="$run_root/virl500_dapo_core_full_20261121"
grpo_tag=stage4_virl500_grpo_20261121
dapo_tag=stage4_virl500_dapo_core_20261121
skip_dapo_probe=${MMDL_SKIP_DAPO_PROBE:-0}
accept_salvaged_grpo_probe=${MMDL_ACCEPT_SALVAGED_GRPO_PROBE:-0}
max_wait_seconds=${MMDL_QUEUE_MAX_WAIT_SECONDS:-21600}
full_data_root=${MMDL_STAGE4_FULL_DATA_ROOT:-}
[[ -n "$full_data_root" ]] || {
  echo 'Set MMDL_STAGE4_FULL_DATA_ROOT to the frozen clean-v2 ViRL package' >&2; exit 2;
}

cpu_gate() {
  local method=$1 mode=$2
  local data_root=$stage4
  if [[ "$mode" == full || "$method" == dapo_core ]]; then data_root=$full_data_root; fi
  env PYTHONPATH="$stage1/easyr1${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" "$stage4/preflight_virl500.py" "$method" "$mode" \
    "$run_root/_cpu_check_only" \
    --source-config "$stage4/config_virl500.yaml" \
    --stage2-config "$stage2/config_ecd500.yaml" \
    --engine "$stage1/easyr1" --stage1 "$stage1" --stage4 "$stage4" \
    --data-root "$data_root" \
    --base "$base" --check-only
}

check_paths() {
  local method mode path tag state
  for spec in "grpo probe $grpo_probe" "grpo full $grpo_full" \
              "dapo_core probe $dapo_probe" "dapo_core full $dapo_full"; do
    read -r method mode path <<< "$spec"
    if [[ -e "$path" ]]; then
      state=$("$py" "$stage4/queue_gate.py" training "$path" "$method" "$mode") || return
      if [[ "$state" == complete ]]; then continue; fi
      if [[ "$method" == grpo && "$mode" == probe && "$accept_salvaged_grpo_probe" == 1 &&
            "$("$py" "$stage4/queue_gate.py" salvage_probe "$path")" == complete ]]; then
        continue
      fi
      if [[ "$state" == pending &&
            "$("$py" "$stage4/queue_gate.py" running_training "$path" "$method" "$mode")" == running ]]; then
        continue
      fi
      echo "Existing training path is neither complete nor audited/running: $path ($state)" >&2
      return 1
    fi
  done
  for tag in "$grpo_tag" "$dapo_tag"; do
    [[ ! -e "$merge_root/$tag" && ! -e "$eval_root/${tag}_smoke20" &&
       ! -e "$eval_root/${tag}_fixed300" ]] || {
      echo "Evaluation/merge output already exists for $tag" >&2; return 1;
    }
  done
  [[ ! -e "$queue_dir" ]] || { echo "Queue directory already exists: $queue_dir" >&2; return 1; }
}

syntax_gate() {
  for script in "$stage4/run_virl500.sh" "$stage4/evaluate_full_candidate.sh" "$stage4/queue_virl500.sh"; do
    bash -n "$script" || return
  done
  "$py" -m py_compile "$stage4/preflight_virl500.py" "$stage4/finalize_virl500.py" \
    "$stage4/audit_virl500_visits.py" "$stage4/queue_gate.py" \
    "$stage4/build_fixed300_lineage.py" || return
  if grep -En 'mmdl_stage2_runs|stage2_ecd500_(grpo|dapo)|scripts/ecd(500|97|5)_easyr1' \
      "$stage4/run_virl500.sh" "$stage4/evaluate_full_candidate.sh" \
      "$stage4/config_virl500.yaml" "$stage4/preflight_virl500.py"; then
    echo "Stale Stage2 output or ECD data path in Stage4 scripts" >&2
    return 1
  fi
}

if [[ $# -eq 1 && $1 == --dry-run ]]; then
  syntax_gate
  cpu_gate grpo probe
  cpu_gate grpo full
  cpu_gate dapo_core probe
  cpu_gate dapo_core full
  check_paths
  echo "STAGE4_DRY_RUN_PASS: no GPU process started"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash queue_virl500.sh [--dry-run]' >&2
  exit 2
fi
[[ "$skip_dapo_probe" == 0 || "$skip_dapo_probe" == 1 ]] || {
  echo 'MMDL_SKIP_DAPO_PROBE must be 0 or 1' >&2; exit 2;
}
[[ "$accept_salvaged_grpo_probe" == 0 || "$accept_salvaged_grpo_probe" == 1 ]] || {
  echo 'MMDL_ACCEPT_SALVAGED_GRPO_PROBE must be 0 or 1' >&2; exit 2;
}
[[ "$max_wait_seconds" =~ ^[0-9]+$ ]] || { echo 'Bad wait limit' >&2; exit 2; }
preflight_checks() {
  syntax_gate || return
  cpu_gate grpo probe || return
  cpu_gate grpo full || return
  cpu_gate dapo_core full || return
  if [[ "$skip_dapo_probe" == 0 ]]; then cpu_gate dapo_core probe || return; fi
  check_paths || return
}
if ! preflight_checks; then
  if [[ ! -e "$queue_dir" ]]; then
    mkdir -p "$queue_dir"
    "$py" "$stage4/queue_status.py" "$queue_dir" preflight failed --exit-code 79 || true
  fi
  echo 'Stage4 CPU/path preflight failed before any GPU work' >&2
  exit 79
fi
mkdir -p "$run_root"
mkdir "$queue_dir"
mkdir "$queue_dir/lock" || { echo 'Queue already claimed' >&2; exit 75; }

phase=starting
record() {
  if [[ $# -eq 3 ]]; then
    "$py" "$stage4/queue_status.py" "$queue_dir" "$1" "$2" --exit-code "$3"
  else
    "$py" "$stage4/queue_status.py" "$queue_dir" "$1" "$2"
  fi
}
on_exit() {
  local code=$?
  trap - EXIT
  if (( code != 0 )); then record "$phase" failed "$code" || true; fi
  exit "$code"
}
trap on_exit EXIT

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

wait_training() {
  local dir=$1 method=$2 mode=$3 state deadline=$((SECONDS + max_wait_seconds))
  while true; do
    state=$("$py" "$stage4/queue_gate.py" training "$dir" "$method" "$mode")
    case "$state" in
      complete) return 0 ;;
      failed) echo "Training completion gate failed: $dir" >&2; return 1 ;;
      pending)
        [[ "$("$py" "$stage4/queue_gate.py" running_training "$dir" "$method" "$mode")" == running ]] || {
          echo "Training stopped without completion: $dir" >&2; return 1;
        }
        ;;
      *) echo "Unexpected training gate: $state" >&2; return 1 ;;
    esac
    (( SECONDS < deadline )) || { echo "Training watchdog timed out: $dir" >&2; return 1; }
    sleep 30
  done
}

run_training() {
  local method=$1 mode=$2 dir=$3 label=$4
  phase=$label
  if [[ -e "$dir" ]]; then
    local existing_state
    existing_state=$("$py" "$stage4/queue_gate.py" training "$dir" "$method" "$mode")
    if [[ "$existing_state" == complete ]]; then
      record "$phase" complete
      echo "Using validated existing training run: $dir"
      return 0
    fi
    if [[ "$method" == grpo && "$mode" == probe && "$accept_salvaged_grpo_probe" == 1 &&
          "$("$py" "$stage4/queue_gate.py" salvage_probe "$dir")" == complete ]]; then
      record grpo_probe_salvaged complete
      echo "Probe process failed after saved checkpoint; using audited pipeline-only salvage: $dir"
      return 0
    fi
    [[ "$("$py" "$stage4/queue_gate.py" running_training "$dir" "$method" "$mode")" == running ]] || {
      echo "Existing training run is not valid or running: $dir" >&2; return 1;
    }
    record "$phase" running
    wait_training "$dir" "$method" "$mode" || return
    record "$phase" complete
    return 0
  fi
  record "$phase" running
  wait_gpu_idle
  if [[ "$mode" == full || "$method" == dapo_core ]]; then
    MMDL_STAGE4_DATA_ROOT="$full_data_root" \
      bash "$stage4/run_virl500.sh" "$method" "$mode" "$dir" > "$queue_dir/${label}.log" 2>&1
  else
    bash "$stage4/run_virl500.sh" "$method" "$mode" "$dir" > "$queue_dir/${label}.log" 2>&1
  fi
  [[ "$("$py" "$stage4/queue_gate.py" training "$dir" "$method" "$mode")" == complete ]] || {
    echo "Training completion gate failed: $dir" >&2; return 1;
  }
  record "$phase" complete
}

run_eval() {
  local dir=$1 tag=$2 label=$3
  phase=$label
  record "$phase" running
  wait_gpu_idle
  bash "$stage4/evaluate_full_candidate.sh" "$dir" "$tag" > "$queue_dir/${label}.log" 2>&1
  [[ "$("$py" "$stage4/queue_gate.py" evaluation \
      "$eval_root/${tag}_fixed300/comparison/comparison_summary.json")" == complete ]] || {
    echo "Fixed300 completion gate failed: $tag" >&2; return 1;
  }
  record "$phase" complete
}

run_training grpo probe "$grpo_probe" grpo_probe
run_training grpo full "$grpo_full" grpo_full
if [[ "$skip_dapo_probe" == 0 ]]; then
  run_training dapo_core probe "$dapo_probe" dapo_probe
fi
run_training dapo_core full "$dapo_full" dapo_full
run_eval "$grpo_full" "$grpo_tag" grpo_eval300
run_eval "$dapo_full" "$dapo_tag" dapo_eval300
phase=finished
record "$phase" complete
