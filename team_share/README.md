# team_share — A가 B·C와 공유하는 파일

main과 평가 기준(v1.2: `scripts/`, `eval_tasks/`)은 건드리지 않는 공유용 브랜치(`a/shared`)입니다.
데이터, 체크포인트, 어댑터, 병합 모델은 올리지 않습니다 (어댑터는 scp·드라이브로 따로 전달).

| 파일 | 대상 | 내용 |
|---|---|---|
| **`TEAM_LOG.md`** | B, C | **팀 실험 기록: 기준선, 결정 사항, 실험 결과(공통 양식), 구현 메모, 관찰, 미결 사항. 여기부터 보세요** |
| `c_format_example.md` | B, C | 학습 프롬프트 형식 (v1.2 평가 프롬프트와 같은 구조), Answer 없는 원본 처리 규칙, 실제 변환 예시 5개. 문서 안의 `local/data_prep/convert_v12.py`는 이 폴더의 `convert_v12.py`와 같은 파일 |
| `convert_v12.py` | C | MAmmoTH-VL 레코드 → v1.2 학습 형식 변환 코드 |
| `per_item_v1.2.csv` | B, C | 4090 기준선 M4_final(v1.2 재채점) 문항 × 시드 정오표와 최종 답. 시드별 555/557/554 → 61.70 ± 0.17 |
| `reward_cases.jsonl` | B, C | 보상 테스트 20개 (R01–R18 + 추가 2개). GRPO 보상은 v1.2 채점과 같은 판정으로 확정 → `v1_2_score` 열이 기대값 |
| `c_list_sft1.jsonl` | C | **SFT1 본학습에 실제로 쓴 12,278문항** (MAmmoTH-VL 7,278 + ECD 5,000). 누수 검사용. 필드: idx, id, source, image(데이터셋 안 상대경로), question·answer(원본 첫 질문·답), status, question_type, flags, dataset. MAmmoTH 이미지는 `MAmmoTH-VL/MAmmoTH-VL-Instruct-12M`의 `single_image_data/shard_1.tar.gz`, ECD 이미지는 `ChartFoundation/ECD-10k-Images`의 `images.tar` 안 경로 |
| `ecd_candidates_for_c.jsonl` | B, C | ECD(@284489f) 학습 후보 62,604문항(추론 있음, 최종 답 40자 이하). 필드: idx, id, image_id, image(images/<id>.png), question, final_answer, answer_raw, used_in_sft1(그 차트가 SFT1에 쓰였는지), used_in_sft1_this_question(그 문항이 SFT1에 쓰였는지), in_dpo_pool(A의 DPO 풀 포함), excluded_reason(not_applicable 등). GRPO·DPO 문항 선별용 |
| `sft1_image_hashes.jsonl` | C | SFT1 이미지 해시 12,278행 (dataset, idx, id, hash{sha256, pixel_sha256, dhash}). C의 `image_hash.py`(c/shared f5c7c28)를 수정 없이 `c_list_sft1.jsonl`에 실행, 실패 0, PIL 13.0.0. 고유 sha256 9,064 / pixel 9,061 / dhash 9,043 (경로만 다르고 내용이 같은 이미지 39개) |
| `leak_review_train_images.zip` | C | C의 누수 후보(c/shared `mmdl_overlap.zip`, f3baf8e) 388쌍에 해당하는 **SFT1 학습 원본 이미지 86장**(22 MB). 안의 `manifest.jsonl`이 후보 쌍마다 train_dataset, train_idx, train_id, train_image, zip_path, sha256, reference_repo/file/id, evidence를 연결. 이미지 경로는 `<dataset>/<데이터셋 안 상대경로>` |
| `convert_ecd.py` | C | ECD QA → v1.2 학습 형식 변환 코드 |
| `merge_lora.py` | B | LoRA 어댑터를 base에 병합 (양쪽 모두 이 스크립트 사용). `--adapter --out [--base]`, base 기본값은 `$MMDL_BASE_MODEL` 또는 HF repo id + 고정 revision. 결과 폴더에 MERGE_INFO.json(어댑터 SHA256, revision 등) 기록. 학습 전·후 어댑터 모두 병합 → vLLM v1.2 스모크까지 확인 (2026-11-17) |

## 확정 사항 (2026-11-17)
- 학습 스택: TRL 1.14.2 + PEFT 0.21.0 + transformers 4.57.1, base `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- LoRA: BF16, r16, alpha 32, 언어 모델 q/k/v/o_proj, vision·연결 모듈 고정
- 평가: 어댑터를 base에 병합한 모델을 `run_mmmu_eval.sh --model_path`로
- GRPO 보상: v1.2 채점 판정 그대로 (정답 1, 그 외 0)
- 데이터: MAmmoTH-VL shard_1로 시작. 추론 없이 짧은 답만 있는 문항 제외, 여러 턴 대화는 첫 턴만. 학습량이 부족하면 묶음을 더 받음
- 6000 재현 판정: 같은 시드 정오 일치율 ≥ 90%(810/900)면 완전 재현, 3시드 평균 61.70 ± 0.5%p + 문항 일치율 ≥ 85%면 통계적 재현, 0.5~1.0%p는 보류·원인 조사, 1%p 이상은 불일치
