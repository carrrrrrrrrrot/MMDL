#!/usr/bin/env bash
# Evaluate one completed EasyR1 Stage2 full run on the RTX PRO 6000.
# Usage: bash evaluate_full_candidate.sh /data/chaewon/mmdl_stage2_runs/RUN UNIQUE_TAG
# This is intentionally sequential: CPU merge, v1.2 20-item smoke, fixed-ID 300 screen.
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 FULL_RUN_DIR UNIQUE_TAG" >&2
  exit 2
fi
RUN="${1%/}"
TAG="$2"
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'TAG must be alphanumeric/underscore' >&2; exit 2; }

PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
EVAL_PY=/data/chaewon/mmdl_envs/vllm011_repro/bin/python
STAGE1=/data/chaewon/mmdl_stage1_paper_style_20261120
STAGE2=/data/chaewon/mmdl_stage2_ecd500_20261120
OPS=/data/chaewon/mmdl_b_eval_ops
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
DATA=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
IDS=/data/chaewon/mmdl_eval_runs/rl_ablation_20261119_qcv2_plan/fixed300_ids_seed3407.txt
BASELINE=/data/chaewon/mmdl_eval_runs/rlab19qcv2_base_fixed300
OUT=/data/chaewon/mmdl_eval_runs
MERGED="/data/chaewon/mmdl_merged_models/${TAG}"
TRAIN_MANIFEST="$RUN/fixed300_training_manifest.json"

[[ -f "$RUN/completion_report.json" && -f "$RUN/run_manifest.json" ]] || {
  echo "Full run has no completed manifests: $RUN" >&2; exit 1;
}
ADAPTER="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_dir"])' "$RUN/completion_report.json")"
SHA="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_sha256"])' "$RUN/completion_report.json")"

lineage=(
  "$PY" "$STAGE2/build_fixed300_lineage.py"
  --run-manifest "$RUN/run_manifest.json"
  --completion-report "$RUN/completion_report.json"
  --train-file "$STAGE1/scripts/ecd500_easyr1.jsonl"
  --validation-file "$STAGE1/scripts/ecd97_holdout_easyr1.jsonl"
  --config-file "$RUN/resolved_config.yaml"
  --source-config-file "$STAGE2/config_ecd500.yaml"
  --reward-file "$STAGE1/scripts/reward_v12.py"
  --adapter-dir "$ADAPTER"
  --output "$TRAIN_MANIFEST"
)

echo "[1/4] Validate completed 15-step run and adapter lineage"
"${lineage[@]}" --dry-run
echo "[2/4] Merge pinned base and verified PEFT adapter: $MERGED"
"$PY" "$OPS/merge_lora.py" \
  --base-model "$BASE" --adapter "$ADAPTER" --adapter-role grpo \
  --adapter-sha256 "$SHA" --output-dir "$MERGED"
"${lineage[@]}" --model-manifest "$MERGED/model_manifest.json" --dry-run
"${lineage[@]}" --model-manifest "$MERGED/model_manifest.json"

echo "[3/4] Run unchanged v1.2 20-item smoke"
"$PY" "$OPS/candidate_eval.py" \
  --model-manifest "$MERGED/model_manifest.json" \
  --train-run-manifest "$TRAIN_MANIFEST" \
  --repo-root "$REPO" --data-root "$DATA" --eval-python "$EVAL_PY" \
  --output-root "$OUT" --run-name "${TAG}_smoke20" --mode smoke

echo "[4/4] Screen against the same-GPU, same-ID base 300 baseline"
"$PY" "$OPS/rl_ablation_20261119/run_fixed300.py" \
  --model-manifest "$MERGED/model_manifest.json" \
  --train-manifest "$TRAIN_MANIFEST" \
  --repo-root "$REPO" --data-root "$DATA" --eval-python "$EVAL_PY" \
  --output-root "$OUT" --run-name "${TAG}_fixed300" \
  --ids-file "$IDS" --baseline-run "$BASELINE" \
  --compare-script /data/chaewon/mmdl_b_compare/compare_runs.py

echo "COMPLETE: $OUT/${TAG}_fixed300/comparison/comparison_summary.json"
