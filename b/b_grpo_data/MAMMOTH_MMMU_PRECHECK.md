# MAmmoTH 비SFT1 900건과 MMMU validation 1차 중복 점검

2026-11-18, B 6000 서버에서 CPU로 실행했다. 입력은 A의 비SFT1 [900문항 목록](mammoth_c_leakage_review_900_20261118.jsonl) (SHA256 `e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8`)과 v1.2 평가에 쓰는 고정 MMMU validation 데이터셋 스냅샷 `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`이다. 후보 이미지 900장은 개별 크기·SHA256을 다시 확인했고 모두 디코딩됐다. MMMU Parquet 30개에서 900문항, 이미지 982장을 디코딩했다(오류 0).

| 검사 | 발견한 후보 쌍 |
| --- | ---: |
| `<image N>` 제거 후 유니코드 NFKC·대소문자·공백 정규화 질문 완전 일치 | 0 |
| 위 정규화에 더해 구두점까지 제거한 질문 일치 | 0 |
| 이미지 64비트 dHash 거리 4 이하 | 0 |

재현: [CPU 스크립트](precheck_mammoth_mmmu_validation.py), [원자료 JSON](mammoth_900.mmmu_validation_precheck.json). 실행 명령은 다음과 같다.

```bash
ROOT=/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118
MMMU=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
/data/chaewon/mmdl_envs/grpo_pilot/bin/python "$ROOT/precheck_mammoth_mmmu_validation.py" \
  --candidates "$ROOT/mammoth_not_sft1_900.jsonl" \
  --image-root "$ROOT/mammoth_grpo_for_b" \
  --mmmu-root "$MMMU" \
  --output "$ROOT/mammoth_900.mmmu_validation_precheck.json"
```

**C의 누수 승인 결과는 아니다.** 이 검사는 MMMU validation의 짧은 문자열 정규화와 전체 이미지 dHash에 한정된다. MMMU-Pro, MMMU dev/test, 이미지의 OCR 텍스트·부분 크롭·의미 유사 질문은 확인하지 않았다. C는 특히 A DPO와 겹치지 않는 `used_in=[]` 369건부터 원본 이미지·정답 품질과 나머지 누수 범위를 검수해야 한다. C 승인 전 새 900건은 B 학습에 포함하지 않는다.
