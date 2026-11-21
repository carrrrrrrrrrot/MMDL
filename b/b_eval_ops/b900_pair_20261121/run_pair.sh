#!/usr/bin/env bash
# Sequential, fail-closed RTX PRO 6000 MMMU v1.2 full-900 screen for two B candidates.
set -euo pipefail

PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
EVAL_PY=/data/chaewon/mmdl_envs/vllm011_repro/bin/python
OPS=/data/chaewon/mmdl_b_eval_ops
REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
DATA=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
OUT=/data/chaewon/mmdl_eval_runs
BASELINE="$OUT/b6000_baseline_v12_20261117"
COMPARE=/data/chaewon/mmdl_b_compare/compare_runs.py
QUEUE="$OUT/b900_pair_20261121_queue"
OLD_MODEL=/data/chaewon/mmdl_merged_models/rlab19qcv2_ecd_grpo/model_manifest.json
OLD_TRAIN=/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/runs/ecd_grpo/run_manifest.json
NEW_MODEL=/data/chaewon/mmdl_merged_models/stage2_ecd500_dapo_core_20261121/model_manifest.json
NEW_TRAIN=/data/chaewon/mmdl_stage2_runs/ecd500_dapo_core_full_20261121/fixed300_training_manifest.json
OLD_NAME=b900_ecd500_grpo_trl_20261121_s3407
NEW_NAME=b900_ecd500_dapo_core_20261121_s3407

mkdir -p "$QUEUE"
exec 9>"$QUEUE/queue.lock"
flock -n 9 || { echo 'Another 900-question queue already holds the lock' >&2; exit 1; }

phase=preflight
record() {
  local state=$1 code=$2
  local line
  line=$(printf '{"timestamp_kst":"%s","phase":"%s","state":"%s","exit_code":%s}' \
    "$(TZ=Asia/Seoul date +%Y-%m-%dT%H:%M:%S%z)" "$phase" "$state" "$code")
  printf '%s\n' "$line" >> "$QUEUE/history.jsonl"
  printf '%s\n' "$line" > "$QUEUE/status.json.tmp"
  mv "$QUEUE/status.json.tmp" "$QUEUE/status.json"
}
on_exit() {
  local code=$?
  if (( code != 0 )); then record failed "$code"; fi
}
trap on_exit EXIT
record running null

test -f "$OLD_MODEL"
test -f "$NEW_MODEL"
test -f "$OLD_TRAIN"
test -f "$NEW_TRAIN"
test -f "$BASELINE/seed_3407/run_meta.txt"
test ! -e "$OUT/$OLD_NAME"
test ! -e "$OUT/$NEW_NAME"

run_one() {
  local label=$1 name=$2 model=$3 train=$4
  phase=$label
  record running null
  "$PY" "$OPS/candidate_eval.py" \
    --model-manifest "$model" --train-run-manifest "$train" \
    --repo-root "$REPO" --data-root "$DATA" --eval-python "$EVAL_PY" \
    --output-root "$OUT" --run-name "$name" --mode screen \
    --baseline-run "$BASELINE" --compare-script "$COMPARE" \
    --verify-model-hashes > "$QUEUE/$label.log" 2>&1
  "$PY" - "$OUT/$name/comparison/comparison_summary.json" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
summary = json.loads(path.read_text(encoding="utf-8"))
row = summary["per_seed"]["3407"]
if row.get("full_900_coverage") is not True:
    raise ValueError(f"{label}: full 900-question coverage is not confirmed")
if not (path.parent.parent / "handoff.json").is_file():
    raise ValueError(f"{label}: completed handoff.json is missing")
required = {
    "items_compared": 900,
    "baseline_correct": 558,
    "input_exact_agreement": 900,
    "doc_hash_agreement": 900,
}
for key, expected in required.items():
    if row.get(key) != expected:
        raise ValueError(f"{label}: {key}={row.get(key)!r}, expected {expected}")
if not (0 <= row["candidate_correct"] <= 900):
    raise ValueError(f"{label}: invalid candidate_correct")
print(f"{label}: {row['candidate_correct']}/900 vs base {row['baseline_correct']}/900")
PY
  record complete 0
}

run_one old_grpo "$OLD_NAME" "$OLD_MODEL" "$OLD_TRAIN"
run_one new_dapo "$NEW_NAME" "$NEW_MODEL" "$NEW_TRAIN"

phase=finished
record complete 0
