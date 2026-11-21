# A → B: (e) dpo_e_trunc 원 실행 어댑터 + 학습 문항 목록 (2026-11-21)

B의 강화학습을 (e) 위에 얹기 위한 전달물. 병합 모델은 포함하지 않음.

## 어댑터 (`adapter/`)
- `adapter_model.safetensors` SHA256 `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa` (47,228,944 B)
- `adapter_config.json`: LoRA r16 / α32 / dropout 0.05, target `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)` (비전 타워 제외), PEFT 0.21.0
- 부모: `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (config의 `base_model_name_or_path`는 A 로컬 경로 → B 경로로 바꿔서 로드)
- 학습: TRL 1.14.2 DPO, β 0.1, lr 5e-5, GA 8, 2 epoch 78 step, bf16 가중치. 4090 v1.2 공식 3시드 65.15±0.72% (기준선 61.70±0.17)
- 4090 평가는 bf16 병합(`merge_and_unload`) 후 `--model_path`로 실행했음

## 학습 문항 (`train_ids_169.jsonl`, `ecd_8.jsonl`)
- 학습 파일 `pairs_trunc.jsonl` (SHA256 `ab332200b264b6ee3e2883b3e1ff9e9e702bd4784b4f3366215e25044fc262e6`): 305쌍, 고유 169문항 = MAmmoTH 161 + ECD 8. 모든 쌍은 chosen = 원본 모델의 정답 응답, rejected = 4,096토큰에서 잘린 원본 응답
- 필드: `pool_id`(A 내부 ID), `dataset`, `source`, `idx`, `n_pairs_used`(이 문항이 쓰인 쌍 수, 합 305), `question_type`, `gold`, `image_files`, `image_sha256`, `question`
- ECD `idx` = `ECD_QAs_All.json`(321,544행)의 행 번호. **`b_original_id` = `ecd_<image_id>_<idx>`로 B manifest의 `original_id` 형식과 같음**(B의 `ecd_004168_241553` 행으로 형식 확인). 8개 모두 B의 ECD manifest(1,080 ID)에는 없음 → B의 ECD500·ECD798 학습 문항과 겹치지 않음
- MAmmoTH `idx` = A 풀 생성 시 MAmmoTH-VL 행 번호(`source`에 하위 출처)

## 누수 점검 (11/19)
169문항 전체를 MMMU validation 900문항(이미지 982장)과 C의 image_hash.py(sha256·pixel·dHash≤4) + 질문 문자열로 대조 → 누수 후보 0. 의미 유사도·OCR·MMMU-Pro는 미검사.
