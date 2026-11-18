# ECD 800건과 MMMU validation 1차 중복 점검

2026-11-18, B 6000 서버에서 CPU로 실행했다. 입력은 `ecd_c_review_priority_800_seed3407.jsonl`(SHA256 `e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f`)과 고정 MMMU validation Parquet 30개다. MMMU는 900문항, 디코딩된 이미지 982장이다.

| 검사 | 후보 쌍 |
| --- | ---: |
| 대소문자·공백 정규화 질문 완전 일치 | 0 |
| 구두점까지 제거한 질문 일치 | 0 |
| 이미지 dHash 거리 4 이하 | 1 |

이미지 유사 후보는 `ecd_003242_229913`과 `validation_Art_6`(이미지 1), dHash 거리 3이다. **원본 이미지를 직접 열어 확인한 결과 ECD는 물리학 열지도 차트, MMMU는 양산을 든 인물의 회화로 서로 다른 이미지다.** dHash의 오탐으로 판정했다. 세부 쌍은 [원자료 JSON](ecd_c_review_priority_800.mmmu_validation_precheck.json)에 남겼다.

이 검사는 **누수 없음 확정이 아니다.** MMMU-Pro, MMMU dev/test, OCR 텍스트 유사도, 의미 유사도는 확인하지 않았다. C가 신규 후보를 최종 승인하기 전에 해당 범위를 검수해야 한다. 재현 스크립트는 [precheck_ecd_mmmu_validation.py](precheck_ecd_mmmu_validation.py)다.
