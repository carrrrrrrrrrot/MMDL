# 실제 ECD base→GRPO 파일럿과 6000 선별 평가 (2026-11-18)

이 실험은 **C가 승인한 ECD 104문항·이미지 104장**으로 원본 Qwen3-VL-4B-Instruct에서 GRPO를 시작한 보험 경로다. 부모는 `Qwen/Qwen3-VL-4B-Instruct` revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`이며, A의 공통 SFT 부모나 DPO를 사용하지 않았다. C 검수에서 SFT1 이미지와 가까운 3건은 보류하고 104건을 승인했다. A의 최신 `team_share/ecd_candidates_for_c.jsonl`(커밋 `15acf1a`)과 C 승인 ZIP의 ID를 대조한 결과 **104/104가 목록에 있고 `excluded_reason`은 모두 비어 있다**. `used_in_sft1`과 `in_dpo_pool` 표시도 모두 거짓이다.

| 단계 | 실제 설정·결과 | 소요 시간 |
|---|---|---:|
| GRPO 학습 | 20 optimizer steps, 문항당 rollout 4개, 80개 생성, lr 1e-5, v1.2 정답 보상(0/1), BF16 LoRA r16/α32 | Trainer 119.68초, 전체 약 132초 |
| 6000 v1.2 선별 평가 | MMMU val 900문항, 시드 3407, 평가 코드 `6102585`, 병합 모델 | 시작부터 비교 파일 생성까지 46분 37초 |

학습 매니페스트는 `complete`이고, 보상 양성과 음성이 섞인 그룹은 **2/20**이다. 평균 보상은 0.3125(25/80), 저장·재로딩 검증은 통과했고 LoRA 변화량 L1은 82.15다. 80개 생성 응답의 v1.2 보상 재계산 불일치는 0건이다. 어댑터 `adapter_model.safetensors`의 SHA256은 `8a89846bc98327edce4b9203ef91470d3c153cdf7e492ad53ed0ede2debdec4d`다.

| 동일 6000 GPU·시드 3407 | 정답 | 점수 |
|---|---:|---:|
| 원본 기준선 | 558/900 | 62.00% |
| 실제 ECD base→GRPO | 564/900 | 62.67% |

차이는 **+6문항, +0.67%p**다. 입력·문서 해시는 900/900 일치한다. 정오 일치는 770/900(85.56%), 추출 답 일치는 700/900(77.78%), 오답→정답 68건, 정답→오답 62건이다. 최종 `Answer:`가 없는 문항은 기준선 103건, 후보 93건이다. 평가 프로세스는 종료 상태 0으로 900문항을 완료했다.

학습 생성 80개 중 37개가 재토큰화 기준 256토큰에 도달했다. 이 수치는 **길이 제한의 대리 지표**이며, 로그에 생성 종료 사유와 정확한 생성 토큰 수가 없어 잘림으로 확정할 수 없다. 보상이 섞인 그룹도 2개뿐이어서 학습 신호가 약하다. 이 1시드 +0.67%p는 후보 선별용 관찰이며, 최종 성능 향상이나 공통 SFT 부모 GRPO의 결과로 해석하지 않는다. 최종 공식 비교에는 4090 v1.2의 3시드 평가가 필요하다.

원본 서버 자료: `/data/chaewon/mmdl_grpo_runs/base_ecd_pilot_002/run_manifest.json`, `/data/chaewon/mmdl_grpo_runs/base_ecd_pilot_002/rollout_summary.json`, `/data/chaewon/mmdl_eval_runs/base_ecd_grpo_001_screen/handoff.json`. [비교 요약](comparison_summary.json)과 [문항별 대응](paired_items.csv)을 함께 보존했다.
