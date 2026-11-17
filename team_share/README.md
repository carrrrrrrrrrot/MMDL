# team_share — A가 B·C와 공유하는 파일

main과 평가 기준(v1.2: `scripts/`, `eval_tasks/`)은 건드리지 않는 공유용 브랜치(`a/shared`)입니다.
데이터, 체크포인트, 어댑터, 병합 모델은 올리지 않습니다 (어댑터는 scp·드라이브로 따로 전달).

| 파일 | 대상 | 내용 |
|---|---|---|
| `c_format_example.md` | B, C | 학습 프롬프트 형식 (v1.2 평가 프롬프트와 같은 구조), Answer 없는 원본 처리 규칙, 실제 변환 예시 5개. 문서 안의 `local/data_prep/convert_v12.py`는 이 폴더의 `convert_v12.py`와 같은 파일 |
| `convert_v12.py` | C | MAmmoTH-VL 레코드 → v1.2 학습 형식 변환 코드 |
| `per_item_v1.2.csv` | B, C | 4090 기준선 M4_final(v1.2 재채점) 문항 × 시드 정오표와 최종 답. 시드별 555/557/554 → 61.70 ± 0.17 |
| `reward_cases.jsonl` | B, C | 보상 테스트 20개 (R01–R18 + 추가 2개). GRPO 보상은 v1.2 채점과 같은 판정으로 확정 → `v1_2_score` 열이 기대값 |
| `merge_lora.py` | B | LoRA 어댑터를 base에 병합 (양쪽 모두 이 스크립트 사용). `--adapter --out [--base]`, base 기본값은 `$MMDL_BASE_MODEL` 또는 HF repo id + 고정 revision. 결과 폴더에 MERGE_INFO.json(어댑터 SHA256, revision 등) 기록. 학습 전·후 어댑터 모두 병합 → vLLM v1.2 스모크까지 확인 (2026-11-17) |

## 확정 사항 (2026-11-17)
- 학습 스택: TRL 1.14.2 + PEFT 0.21.0 + transformers 4.57.1, base `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- LoRA: BF16, r16, alpha 32, 언어 모델 q/k/v/o_proj, vision·연결 모듈 고정
- 평가: 어댑터를 base에 병합한 모델을 `run_mmmu_eval.sh --model_path`로
- GRPO 보상: v1.2 채점 판정 그대로 (정답 1, 그 외 0)
- 데이터: MAmmoTH-VL shard_1로 시작. 추론 없이 짧은 답만 있는 문항 제외, 여러 턴 대화는 첫 턴만. 학습량이 부족하면 묶음을 더 받음
- 6000 재현 판정: 같은 시드 정오 일치율 ≥ 90%(810/900)면 완전 재현, 3시드 평균 61.70 ± 0.5%p + 문항 일치율 ≥ 85%면 통계적 재현, 0.5~1.0%p는 보류·원인 조사, 1%p 이상은 불일치
