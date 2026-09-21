#!/bin/bash

# Environment setup
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Local model & dataset configuration
MODEL_PATH="모델위치"
OUTPUT_PATH="로그위치"
NUM_GPUS="${NUM_GPUS:-1}"
PORT="${MAIN_PROCESS_PORT:-12346}"

echo "================================================================="
echo "Running Qwen3-VL-4B evaluation on local MMMU dataset"
echo "Model Path  : ${MODEL_PATH}"
echo "Tasks       : mmmu_val_local (Local dataset from /mnt/SSD-2TB/models/MMMU)"
echo "GPUs        : ${NUM_GPUS}"
echo "Output Path : ${OUTPUT_PATH}"
echo "================================================================="

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${PORT}" -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained="${MODEL_PATH}",max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks mmmu_val_local \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    "$@"