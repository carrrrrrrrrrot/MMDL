# MMDL — Qwen3-VL-4B-Instruct × MMMU-val 평가 파이프라인

평가 파이프라인의 전체 명세(처음 상태와의 비교, 프롬프트 전문, 생성·채점 설정과 근거, 검증 내역, 한계)는
**[`docs/eval_pipeline.md`](docs/eval_pipeline.md)**에 있습니다. 이 README는 재현 방법만 다룹니다.

- 평가 기준: task `mmmu_val_mmdl`, `metadata.version: 1.2`
- 최종 결과: **61.70 ± 0.17** (MMMU val 900문항, 3 seed, 엄격 채점) → [`results/runs/M4_final/summary.md`](results/runs/M4_final/summary.md)

## 1. 환경

```bash
conda create -n vllm011 python=3.10 -y
conda activate vllm011
pip install -r requirements.txt               # vllm 0.11.0, torch 2.8.0 (CUDA 12.8), transformers 4.57.1 등
pip install --no-deps -e code/lmms-eval       # 이 repo에 포함된 lmms-eval 0.7.3 (수정 없음)
```

- 확인한 환경: RTX 4090 24GB 1장, NVIDIA 드라이버 550.78
- 900문항 × seed 1개에 약 2시간, peak VRAM 약 23GB

## 2. 모델·데이터 (revision 고정)

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir <MODEL_DIR>
huggingface-cli download MMMU/MMMU --repo-type dataset --revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 --local-dir <DATA_ROOT>
```

- `<DATA_ROOT>/<과목>/validation-*.parquet`이 30개 있어야 합니다. 스크립트가 개수를 확인합니다.
- 모델은 revision을 고정하기 위해 **로컬 snapshot 경로**로 넘깁니다.

## 3. 평가 실행 (한 커맨드)

```bash
PYTHON=$(which python) bash scripts/run_mmmu_eval.sh \
    --model_path <MODEL_DIR> \
    --data_root  <DATA_ROOT> \
    --seeds "3407 1 2" \
    --run_name <이름>
```

- 결과는 `results/runs/<이름>/seed_<seed>/`에 저장됩니다: lmms-eval 결과, 사용한 task 사본, `run_meta.txt`, `pip_freeze.txt`
- 진행 상황: `bash scripts/watch_progress.sh <이름>`
- **fine-tune 후 재평가:** `--model_path`만 바꿉니다. task 파일과 스크립트는 수정하지 않습니다.

## 4. 결과표

```bash
python scripts/summarize_results.py results/runs/<이름> --out results/runs/<이름>/summary.md
```

- 출력 내용: 과목별 정확도(seed별, 평균 ± 표준편차), 종합 점수(= 30과목 macro 평균 = 정답 수 / 900), 보조 지표(무응답 수, 공식 관례 점수)
- run 당시 lmms-eval이 기록한 점수와 일치하는지 assert로 먼저 확인한 뒤, 현재 채점 기준(v1.2)으로 계산합니다.

## 5. 구성

| 경로 | 내용 |
|---|---|
| `eval_tasks/mmmu_mmdl/` | 평가 기준: task yaml(프롬프트 문구, 생성 설정)과 `mmdl_utils.py`(메시지 구성, 채점 전처리) |
| `scripts/` | 실행(`run_mmmu_eval.sh`), 결과표, 모니터, HF·vLLM 입력 동일성 검증(`parity/`), 진단(`diagnosis/`), LLM judge 보조 분석(`judge/`) |
| `code/lmms-eval/` | lmms-eval 0.7.3 원본 |
| `results/` | 최종 결과(`runs/M4_final`), 보조·진단·pilot run, 처음 상태 baseline, 검증 자료 |
| `docs/eval_pipeline.md` | 파이프라인 명세 |
| `reports/mmmu_baseline.md` | 과제 제출 보고서 |
