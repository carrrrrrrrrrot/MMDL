# MMMU test 평가 키트 (A → B, 기준선 실행 부탁용)

MMMU test 정답은 2026-02-12 운영진이 공개함. revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`(2026-07-10)의 test parquet `answer` 열이 공식 정답이라 로컬에서 v1.2와 같은 공식 파서로 채점한다.

A는 4090에서 원 (e)(`dpo_e_trunc`)를 11/24 13:01부터 돌리는 중이다(완료 예상 11/25 19–21시). **B는 6000에서 원본 Qwen3-VL-4B-Instruct(revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`)로 같은 설정을 실행해 주면 된다.**

## 구성
- `mmmu_test_mmdl.yaml.tmpl`: v1.2 `eval_tasks/mmmu_mmdl/mmmu_val_mmdl.yaml.tmpl` 사본. 데이터 경로, split(test), task 이름, metric(submission 추가)만 바꿈. 프롬프트·생성 설정은 v1.2와 같음.
- `mmdl_utils.py`: v1.2와 바이트 동일(SHA256 앞 16자리 `289c4e30b278d1eb`).
- `run_test.sh`: v1.2 `scripts/run_mmmu_eval.sh`의 vllm 분기와 같은 lmms-eval 명령·model_args. v1.2 스크립트는 validation parquet 30개를 강제하므로 별도 실행기.
- `test_summary.py`: 결과 요약.
- `data_SHA256SUMS.txt`: test parquet 32개의 SHA256 (10,500문항 = 객관식 9,873 + 주관식 627).

## B 실행 방법
1. 데이터: HF `MMMU/MMMU` revision `98e6ac0c…`의 30과목 `test-*.parquet`를 `<DATA>/<과목>/`에 받고 `sha256sum -c data_SHA256SUMS.txt` 전부 OK 확인. `<DATA>/REVISION.txt`에 revision을 적는다.
2. `run_test.sh` 맨 위의 `REPO_ROOT`, `PYTHON`(vllm 0.11 평가 env), `DATA`, `OUT`을 6000 서버 경로로 바꾼다. 그 외는 바꾸지 않는다.
3. `bash run_test.sh --model_path <원본 Qwen3-VL-4B-Instruct 경로> --run_name test_base --seed 3407` (vllm_batch 기본 2048 유지).
4. 끝나면 `<OUT>/test_base/seed_3407/` 아래 `*_results.json`, `*_samples_*.jsonl`, `run_meta.txt`, 제출 JSON을 Git(또는 A가 받을 수 있는 곳)에 올려 준다. A가 (e)와 같은 문항끼리 비교한다.

## 주의
- 4090 기준 약 5.5문항/분 → 10,500문항 약 30–32시간. 중간 저장이 없으므로 끝까지 돌아야 결과가 나온다.
- GPU가 달라 같은 모델도 점수가 조금 흔들릴 수 있다(MMMU val 기준선 4090 555 vs 6000 558). 문항 수가 많아 영향은 작지만 결과에 GPU를 함께 적는다.
