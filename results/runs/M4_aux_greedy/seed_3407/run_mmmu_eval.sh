#!/bin/bash
# Qwen3-VL-4B-Instruct × MMMU validation 평가 진입점.
#
# 평가 기준(프롬프트, 생성 설정, 채점)은 모두 eval_tasks/ 의 task 파일에 고정되어 있고,
# 이 스크립트는 경로·seed·attention 구현만 인자로 받습니다.
# fine-tune 이후에는 --model_path 만 바꿔서 같은 기준으로 다시 평가합니다.
#
# 사용 예:
#   bash scripts/run_mmmu_eval.sh \
#       --model_path /path/to/Qwen3-VL-4B-Instruct \
#       --data_root  /path/to/MMMU
#
# 필수 인자
#   --model_path  모델 checkpoint 경로 또는 HF repo id
#   --data_root   MMMU 데이터 루트 (그 아래에 <과목>/validation-*.parquet 가 있어야 함)
# 선택 인자
#   --task        평가 task 이름 (기본: mmmu_val_mmdl, 회귀 테스트: mmmu_val_mmdl_regress)
#   --output_path 결과 저장 루트 (기본: <repo>/results/runs)
#   --run_name    결과 폴더 이름 접두어 (기본: <task>_<시각>)
#   --seeds       lmms-eval --seed 값 목록, 공백 구분. seed마다 한 번씩 실행 (기본: "3407")
#                 예: "3407 1 2", 회귀 테스트는 "0,1234,1234,1234"
#   --backend     추론 백엔드: hf | vllm (기본: hf)
#                 hf   = transformers generate() (lmms-eval 모델 qwen3_vl), 배치 1
#                 vllm = vLLM (lmms-eval 모델 vllm). vLLM 이 설치된 env 의 python 을 PYTHON 으로 지정해야 함
#   --attn        attention 구현 (hf 전용): sdpa | flash_attention_2 | eager (기본: sdpa)
#   --vllm_batch  vLLM 에 한 번에 넘기는 요청 수 (vllm 전용, 기본: 1024 = 900문항 전체를 한 번에).
#                 나눠서 넘기면 묶음마다 가장 긴 응답을 기다리게 되어 느려집니다.
#   --limit       앞에서부터 N문항만 실행 (스모크 테스트 전용, 본 평가에는 쓰지 말 것)
#   --subjects    일부 과목만 실행, 공백 구분 (pilot 전용, 본 평가에는 쓰지 말 것). 예: "Math Finance"
#                 평가 기준은 그대로 두고 task 파일의 data_files 만 해당 과목으로 바꿉니다.
#   --gen_override  생성 설정 일부를 이 run 에서만 덮어씀 (pilot 전용). 예: "max_new_tokens=8192"
#                 본 평가에는 쓰지 말고, 확정된 값은 task 파일에 반영할 것
#   --            이후 인자는 lmms_eval 에 그대로 전달
#
# 환경 변수
#   PYTHON  사용할 python 실행 파일 (기본: PATH 의 python). 의존성은 requirements.txt 참고.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO_ROOT}/eval_tasks/mmmu_mmdl"
PYTHON="${PYTHON:-python}"

MODEL_PATH=""
DATA_ROOT=""
TASK="mmmu_val_mmdl"
OUTPUT_ROOT="${REPO_ROOT}/results/runs"
RUN_NAME=""
SEEDS="3407"
ATTN="sdpa"
LIMIT=""
SUBJECTS=""
GEN_OVERRIDE=""
BACKEND="hf"
VLLM_BATCH="1024"
EXTRA_ARGS=()

usage() { sed -n '2,37p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2"; shift 2 ;;
        --data_root)   DATA_ROOT="$2"; shift 2 ;;
        --task)        TASK="$2"; shift 2 ;;
        --output_path) OUTPUT_ROOT="$2"; shift 2 ;;
        --run_name)    RUN_NAME="$2"; shift 2 ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        --attn)        ATTN="$2"; shift 2 ;;
        --backend)     BACKEND="$2"; shift 2 ;;
        --vllm_batch)  VLLM_BATCH="$2"; shift 2 ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --subjects)    SUBJECTS="$2"; shift 2 ;;
        --gen_override) GEN_OVERRIDE="$2"; shift 2 ;;
        --)            shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)     usage ;;
        *) echo "알 수 없는 인자: $1" >&2; usage ;;
    esac
done

[[ -n "${MODEL_PATH}" ]] || { echo "--model_path 가 필요합니다" >&2; usage; }
[[ -n "${DATA_ROOT}" ]]  || { echo "--data_root 가 필요합니다" >&2; usage; }

TASK_TMPL="${TASK_DIR}/${TASK}.yaml.tmpl"
[[ -f "${TASK_TMPL}" ]] || { echo "task 템플릿이 없습니다: ${TASK_TMPL}" >&2; exit 1; }

DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
N_PARQUET=$(ls "${DATA_ROOT}"/*/validation-*.parquet 2>/dev/null | wc -l)
[[ "${N_PARQUET}" -eq 30 ]] || { echo "validation parquet 가 30개가 아닙니다 (${N_PARQUET}개): ${DATA_ROOT}" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-${TASK}_$(date +%Y%m%d_%H%M%S)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

for SEED in ${SEEDS}; do
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}/seed_${SEED//,/_}"
    mkdir -p "${RUN_DIR}/task"

    # 이 run 에 실제로 쓰인 task 파일 사본을 결과 폴더에 남깁니다.
    sed "s#__MMMU_DATA_ROOT__#${DATA_ROOT}#g" "${TASK_TMPL}" > "${RUN_DIR}/task/${TASK}.yaml"
    if [[ -n "${SUBJECTS}" ]]; then
        FILES=""
        for S in ${SUBJECTS}; do
            ls "${DATA_ROOT}/${S}"/validation-*.parquet > /dev/null || { echo "과목 데이터가 없습니다: ${S}" >&2; exit 1; }
            FILES+="${FILES:+, }\"${DATA_ROOT}/${S}/validation-*.parquet\""
        done
        sed -i "s#^\(    validation: \).*#\1[${FILES}]#" "${RUN_DIR}/task/${TASK}.yaml"
    fi
    cp "${TASK_DIR}"/*.py "${RUN_DIR}/task/" 2>/dev/null || true
    cp "$0" "${RUN_DIR}/run_mmmu_eval.sh"

    case "${BACKEND}" in
        hf)
            BATCH_SIZE=1
            GEN_KWARGS="${GEN_OVERRIDE}"
            MODEL_ARGS=(--model qwen3_vl --model_args "pretrained=${MODEL_PATH},attn_implementation=${ATTN}") ;;
        vllm)
            # 이미지 크기는 task 에서 미리 맞추므로(mmdl_utils.py), vLLM 쪽 processor 의 최소 픽셀을
            # HF 경로(qwen_vl_utils 기본값 4,096)와 같게 낮춰 다시 확대되지 않게 합니다.
            # vLLM 은 lmms-eval --seed 를 받지 않으므로 요청별 sampling seed 를 gen_kwargs 로 넘깁니다.
            # max_model_len=40960: 가장 긴 입력(5,649 토큰, Music_21) + 공식 out_seq_length 32,768 을 담을 수 있는 길이.
            # lmms-eval 은 --batch_size 를 모델 생성자에 넘기므로 vLLM 의 요청 묶음 크기는 --batch_size 로 지정합니다.
            BATCH_SIZE="${VLLM_BATCH}"
            MODEL_ARGS=(--model vllm --model_args "model=${MODEL_PATH},is_qwen3_vl=True,gpu_memory_utilization=0.85,max_model_len=40960,limit_mm_per_prompt={\"image\":7},mm_processor_kwargs={\"min_pixels\":4096,\"max_pixels\":16777216}")
            GEN_KWARGS="seed=${SEED%%,*}${GEN_OVERRIDE:+,${GEN_OVERRIDE}}" ;;
        *) echo "알 수 없는 backend: ${BACKEND}" >&2; exit 1 ;;
    esac

    CMD=("${PYTHON}" -m accelerate.commands.launch --num_processes=1 --main_process_port="${MAIN_PROCESS_PORT:-12346}" -m lmms_eval
        "${MODEL_ARGS[@]}"
        ${GEN_KWARGS:+--gen_kwargs "${GEN_KWARGS}"}
        --tasks "${TASK}"
        --include_path "${RUN_DIR}/task"
        --batch_size "${BATCH_SIZE}"
        --seed "${SEED}"
        --output_path "${RUN_DIR}"
        --log_samples)
    [[ -n "${LIMIT}" ]] && CMD+=(--limit "${LIMIT}")
    CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

    {
        echo "date: $(date -Iseconds)"
        echo "repo_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "repo_dirty_files: $(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | grep -v '^?? results/' | wc -l)"
        echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
        echo "model_path: ${MODEL_PATH}"
        echo "data_root: ${DATA_ROOT}"
        echo "backend: ${BACKEND}"
        echo "subjects: ${SUBJECTS:-all}"
        echo "limit: ${LIMIT:-none}"
        echo "gen_override: ${GEN_OVERRIDE:-none}"
        echo "command: ${CMD[*]}"
    } > "${RUN_DIR}/run_meta.txt"
    "${PYTHON}" -m pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>/dev/null || true

    # peak VRAM: nvidia-smi 로 500ms 간격 기록 (GPU 0 전체 사용량, CUDA context 포함)
    nvidia-smi --id=0 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "${RUN_DIR}/vram.csv" 2>/dev/null &
    VRAM_PID=$!
    trap 'kill ${VRAM_PID} 2>/dev/null || true' EXIT

    echo "=== ${TASK} | seed ${SEED} | ${RUN_DIR}"
    set +e
    "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/stdout.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    # lmms-eval 은 평가 중 오류가 나도 종료 코드 0 으로 끝날 수 있으므로, 결과 파일이 없으면 실패로 봅니다.
    if [[ "${STATUS}" -eq 0 ]] && ! ls "${RUN_DIR}"/*/*_results.json > /dev/null 2>&1; then
        STATUS=1
    fi

    kill "${VRAM_PID}" 2>/dev/null || true
    wait "${VRAM_PID}" 2>/dev/null || true
    PEAK=$(awk -F', *' 'BEGIN{m=0} {if ($2+0>m) m=$2+0} END{print m}' "${RUN_DIR}/vram.csv")
    echo "peak_vram_mib: ${PEAK}" >> "${RUN_DIR}/run_meta.txt"
    echo "exit_status: ${STATUS}" >> "${RUN_DIR}/run_meta.txt"
    echo "=== 종료 status=${STATUS}, peak VRAM ${PEAK} MiB"
    [[ "${STATUS}" -eq 0 ]] || exit "${STATUS}"
done
