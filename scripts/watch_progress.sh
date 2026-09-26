#!/bin/bash
# run_mmmu_eval.sh 로 실행 중인 평가의 진행 상황을 주기적으로 출력합니다.
# 평가 프로세스에는 손대지 않고, 결과 폴더의 stdout.log / vram.csv / run_meta.txt 만 읽습니다.
#
# 사용 예:
#   bash scripts/watch_progress.sh                      # results/runs 에서 가장 최근 run 을 따라감
#   bash scripts/watch_progress.sh M3_pilot_sdpa        # run 이름 지정 (seed 가 여러 개면 가장 최근 seed)
#   bash scripts/watch_progress.sh results/runs/M3_pilot_sdpa/seed_3407 -i 10
#
# 옵션
#   -i SEC   출력 간격 (기본 30초)
#   --once   한 번만 출력하고 종료
#
# 출력 예:
#   [23:14:24] M3_pilot_fa2_math/seed_3407 | 실행 중 | 19/30 (63%) | 경과 12:38, 남은 시간 ~05:10 | 28.21s/it | VRAM 9843 MiB (peak 9843)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${REPO_ROOT}/results/runs"
TARGET=""
INTERVAL=30
ONCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) INTERVAL="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) TARGET="$1"; shift ;;
    esac
done

# 따라갈 seed 폴더를 고릅니다 (매번 다시 골라서, seed 가 넘어가도 따라감).
resolve_run_dir() {
    local base
    if [[ -z "${TARGET}" ]]; then
        base="${RUNS_ROOT}"
    elif [[ -d "${TARGET}" ]]; then
        base="${TARGET}"
    else
        base="${RUNS_ROOT}/${TARGET}"
    fi
    [[ -f "${base}/run_meta.txt" ]] && { echo "${base}"; return; }
    # base 아래에서 run_meta.txt 가 가장 최근에 만들어진 seed 폴더
    find "${base}" -maxdepth 3 -name run_meta.txt -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

print_status() {
    local dir="$1"
    local name="${dir#${RUNS_ROOT}/}"
    local now; now="$(date +%H:%M:%S)"

    # 진행률: tqdm 의 마지막 표시 (\r 로 덮어쓰므로 줄 단위로 풀어서 읽음)
    #   vLLM 백엔드는 요청을 한 번에 넘겨서 lmms-eval 의 "Model Responding" 이 끝날 때까지 0 이므로
    #   vLLM 의 "Processed prompts" 표시가 있으면 그것을 씁니다.
    local bar="" log_lines=""
    if [[ -f "${dir}/stdout.log" ]]; then
        log_lines="$(tr '\r' '\n' < "${dir}/stdout.log")"
        bar="$(grep -a "Processed prompts" <<< "${log_lines}" | tail -1)"
        [[ -z "${bar}" ]] && bar="$(grep -a "Model Responding" <<< "${log_lines}" | tail -1)"
    fi
    local progress="모델 로딩/준비 중"
    local re='([0-9]+)%\|.*\| *([0-9]+)/([0-9]+) \[([0-9:]+)<([0-9:?]+), *([0-9.?]+ ?(s/it|it/s))[],]'
    if [[ "${bar}" =~ ${re} ]]; then
        progress="${BASH_REMATCH[2]}/${BASH_REMATCH[3]} (${BASH_REMATCH[1]}%) | 경과 ${BASH_REMATCH[4]}, 남은 시간 ~${BASH_REMATCH[5]} | ${BASH_REMATCH[6]}"
        local out_re='output: ([0-9.]+) toks/s'
        [[ "${bar}" =~ ${out_re} ]] && progress+=" | 생성 ${BASH_REMATCH[1]} tok/s"
    fi

    # VRAM: vram.csv 의 마지막 값과 지금까지의 최댓값
    local vram="-"
    if [[ -s "${dir}/vram.csv" ]]; then
        vram="$(awk -F', *' '{cur=$2+0; if (cur>m) m=cur} END{printf "%d MiB (peak %d)", cur, m}' "${dir}/vram.csv")"
    fi

    # 상태: run_meta.txt 에 exit_status 가 기록되면 종료된 것
    local state="실행 중"
    local exit_status; exit_status="$(grep -a '^exit_status:' "${dir}/run_meta.txt" 2>/dev/null | awk '{print $2}')"
    if [[ -n "${exit_status}" ]]; then
        state="종료(status=${exit_status})"
        # lmms-eval 결과 표의 행: |task|filter|n-shot|mmmu_acc|↑|값|±|stderr|
        local acc; acc="$(awk -F'|' '$5 ~ /mmmu_acc/ && NF > 7 {v = $7} END {gsub(/ /, "", v); print v}' "${dir}/stdout.log" 2>/dev/null)"
        [[ -n "${acc}" ]] && state+=" acc=${acc}"
    elif ! pgrep -f -- "--output_path ${dir}" > /dev/null 2>&1; then
        state="프로세스 없음(중단되었을 수 있음)"
    fi

    echo "[${now}] ${name} | ${state} | ${progress} | VRAM ${vram}"
    [[ -n "${exit_status}" ]]
}

while true; do
    RUN_DIR="$(resolve_run_dir)"
    if [[ -z "${RUN_DIR}" ]]; then
        echo "[$(date +%H:%M:%S)] 따라갈 run 을 찾지 못했습니다: ${TARGET:-${RUNS_ROOT}}"
    else
        print_status "${RUN_DIR}"
        FINISHED=$?
        # 지정한 run 의 마지막 seed 까지 끝났으면 종료 (다음 seed 가 시작되면 계속 따라감)
        if [[ ${FINISHED} -eq 0 && ${ONCE} -eq 0 ]]; then
            sleep 5
            [[ "$(resolve_run_dir)" == "${RUN_DIR}" ]] && { echo "완료: ${RUN_DIR}"; exit 0; }
            continue
        fi
    fi
    [[ ${ONCE} -eq 1 ]] && exit 0
    sleep "${INTERVAL}"
done
