#!/bin/bash
# MMMU test 제출용 답안 생성 실행기. scripts/run_mmmu_eval.sh (v1.2) 의 vllm 분기와 같은 lmms-eval 명령·model_args 를 쓰고,
# task 만 local/eval_mmmu_test 의 mmmu_test_mmdl 로 바꾼다 (v1.2 스크립트는 MMMU validation 30 parquet 를 강제하므로 별도 실행기).
# 사용: bash run_test.sh --model_path M --run_name R [--seed 3407] [--limit N] [--vllm_batch 2048]
set -euo pipefail
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT=/home/dekim/workspace/MMDL
PYTHON=/home/dekim/anaconda3/envs/vllm011/bin/python; DATA=/mnt/SSD-2TB/dekim/data/MMMU_test; OUT=/mnt/SSD-2TB/dekim/eval_runs
SEED=3407; LIMIT=""; VLLM_BATCH=2048; MODEL_PATH=""; RUN_NAME=""
while [[ $# -gt 0 ]]; do case "$1" in
  --model_path) MODEL_PATH="$2"; shift 2;; --run_name) RUN_NAME="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --limit) LIMIT="$2"; shift 2;; --vllm_batch) VLLM_BATCH="$2"; shift 2;; *) echo "bad arg $1" >&2; exit 1;; esac; done
[[ -n "$MODEL_PATH" && -n "$RUN_NAME" ]] || { echo "need --model_path --run_name" >&2; exit 1; }
RUN_DIR="$OUT/$RUN_NAME/seed_$SEED"; [[ -e "$RUN_DIR" ]] && { echo "exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR/task"
sed "s#__MMMU_DATA_ROOT__#${DATA}#g" "$E/mmmu_test_mmdl.yaml.tmpl" > "$RUN_DIR/task/mmmu_test_mmdl.yaml"
cp "$E"/mmdl_utils.py "$RUN_DIR/task/"; cp "$0" "$RUN_DIR/run_test.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_ARGS=(--model vllm --model_args "model=${MODEL_PATH},is_qwen3_vl=True,gpu_memory_utilization=0.85,max_model_len=40960,limit_mm_per_prompt={\"image\":7},mm_processor_kwargs={\"min_pixels\":4096,\"max_pixels\":16777216}")
CMD=("$PYTHON" -m accelerate.commands.launch --num_processes=1 --main_process_port="${MAIN_PROCESS_PORT:-12346}" -m lmms_eval
  "${MODEL_ARGS[@]}" --gen_kwargs "seed=${SEED}" --tasks mmmu_test_mmdl --include_path "$RUN_DIR/task"
  --batch_size "$VLLM_BATCH" --seed "$SEED" --output_path "$RUN_DIR" --log_samples)
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
{ echo "date: $(date -Iseconds)"; echo "repo_commit: $(git -C $REPO_ROOT rev-parse HEAD)"; echo "model_path: $MODEL_PATH"
  echo "data: $DATA revision $(cat $DATA/REVISION.txt)"; echo "limit: ${LIMIT:-none}"; echo "vllm_batch: $VLLM_BATCH"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"; echo "command: ${CMD[*]}"; } > "$RUN_DIR/run_meta.txt"
nvidia-smi --id=0 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$RUN_DIR/vram.csv" 2>/dev/null & VP=$!
trap 'kill $VP 2>/dev/null || true' EXIT
T0=$(date +%s); set +e; "${CMD[@]}" 2>&1 | tee "$RUN_DIR/stdout.log"; ST=${PIPESTATUS[0]}; set -e
[[ $ST -eq 0 ]] && ! ls "$RUN_DIR"/*/*_results.json >/dev/null 2>&1 && ST=1
echo "elapsed_s: $(( $(date +%s) - T0 ))" >> "$RUN_DIR/run_meta.txt"; echo "exit_status: $ST" >> "$RUN_DIR/run_meta.txt"; exit $ST
