# 혼합 1,000문항 GRPO·DAPO·SAPO 선별 결과

2026-11-19, RTX PRO 6000 Blackwell에서 같은 Qwen3-VL-4B-Instruct 원본 부모로 학습한 세 후보를 v1.2 평가의 시드 3407·고정 MMMU 300문항으로 비교했다. **이번 혼합 실험 내부 공동 1위는 GRPO와 DAPO(각 188/300)**다. 그러나 [앞선 단일 출처 실험](../rl_ablation_20261119_qcv2/REPORT.md)의 **ECD GRPO 195/300과 ECD DAPO 193/300이 전체 후보 상위 2개를 유지한다.** 혼합 후보가 상위 2개를 대체한다는 근거는 나오지 않았다.

| 데이터·방법 | 정답/300 | 같은 6000 원본 185/300 대비 | 오답→정답 | 정답→오답 | 최종 답 누락 |
|---|---:|---:|---:|---:|---:|
| ECD 500·GRPO (앞선 실험) | **195** | +10 (+3.33%p) | 23 | 13 | 29 |
| ECD 500·DAPO (앞선 실험) | **193** | +8 (+2.67%p) | 19 | 11 | 29 |
| MAmmoTH CLEVR 500·SAPO (앞선 실험) | 189 | +4 (+1.33%p) | 22 | 18 | 37 |
| **ECD 500 + CLEVR 500·GRPO (이번 실험)** | **188** | **+3 (+1.00%p)** | 15 | 12 | 31 |
| **ECD 500 + CLEVR 500·DAPO (이번 실험)** | **188** | **+3 (+1.00%p)** | 18 | 15 | 17 |
| **ECD 500 + CLEVR 500·SAPO (이번 실험)** | **186** | **+1 (+0.33%p)** | 21 | 20 | 22 |
| MAmmoTH CLEVR 500·GRPO (앞선 실험) | 186 | +1 (+0.33%p) | 18 | 17 | 28 |
| MAmmoTH CLEVR 500·DAPO (앞선 실험) | 186 | +1 (+0.33%p) | 18 | 17 | 31 |
| ECD 500·SAPO (앞선 실험) | 184 | −1 (−0.33%p) | 17 | 18 | 31 |

위 순위는 **6000 단일 시드의 300문항 진단 순위**다. 앞선 단일 출처 후보는 500개 학습 문항·1,000 optimizer steps였고, 이번 혼합 후보는 1,000개·2,000 steps였다. 따라서 점수 차이를 데이터 혼합 자체의 인과 효과로 해석할 수 없다. 표본이 작고 vLLM 실행 간 문항별 출력도 흔들릴 수 있다. 900문항 또는 4090 공식 3시드 성능을 이 표에서 추정하지 않는다.

## 학습 입력과 검증

- 부모: `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, `parent_kind=base`. A의 SFT나 DPO 어댑터를 이어서 학습한 실험이 아니다. A의 base→DPO 경로와 부모 종류는 같다.
- 입력: ECD 추론 문항 500개와 MAmmoTH-VL의 CLEVR 출처 500개를 학습에 사용했다. 별도 heldout 100개(출처별 50개)는 이번 순위 산정에 쓰지 않았다. [데이터 README](../../b_grpo_data/rl_20261119_mixed1000_qc_v2/README.md)와 [검증 결과](../../b_grpo_data/rl_20261119_mixed1000_qc_v2/validation_report.json)의 manifest SHA256은 `3c46073d60f3b75b5515678b8960290daa8bc55b1c5daf3dabb49ace1e45c636`이다.
- 각 방법은 같은 초기 LoRA SHA256 `27d2e5cab7bf6becc2fa6ff8928f2832d3ef3b1a478971402819d1129d10aa51`에서 별도 출발했다. 동일한 1,000개 문항, 시드 3407, 문항당 rollout 4개, `num_iterations=2`, 최대 2,000 optimizer steps, 학습 completion 상한 512토큰, 학습률 `1e-5`, v1.2 이진 보상을 적용했다. TRL의 GRPOTrainer에서 loss 유형을 바꾼 실험이며 DAPO·SAPO 논문의 전체 학습 조합을 재현한 것은 아니다.
- [학습 요약](training_summary.json)은 세 run 모두 `complete`, 2,000 steps, 고유 학습 ID 1,000개, rollout 8,000개로 기록한다. 어댑터 저장·재로딩과 비영 학습 신호를 확인했다. 어댑터 SHA256은 GRPO `db7aae95ea7ab0e02f1cc19c3137ddfff57cfc2c2197cabd901f5fa0bcd6af31`, DAPO `f1b9812487e013970bb6856432e7b91f35d3610ab036c20a32ef247ded0b5292`, SAPO `599d9b3dd50196c4bfa9ea6036d7af9d388cbf2e28abc3e3b0006429d22d83f7`이다. 학습 중 평균 보상은 각각 0.530, 0.515, 0.52875였으며 MMMU 점수와 혼동하지 않는다.
- QC v2는 시각적으로 의심된 CLEVR 정답 3개를 제외하고 A 자료와 이미지로 확인한 3개를 교체했다. 1,100개 이미지의 해시·디코드와 정답 문자열의 v1.2 보상 형식을 검사했다. **전수 정답의 시각 검수나 C의 학습 승인까지 끝난 것은 아니다.** CLEVR 학습 문항 중 124개는 A의 SFT1, 343개는 A의 DPO 풀과 겹친다. 이는 평가 문항 누수 판정과는 별개다.

## 평가 절차와 문항별 결과

평가 레포 commit `61025850563d0331e365bc974a63c0d3c4ed4c2c`, 기존 v1.2 프롬프트·생성·채점, GPU `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`을 사용했다. 30개 분야에서 각 10문항을 정답 정보와 무관한 해시로 고른 동일한 300개 ID의 SHA256은 `1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d`이다. 원본 기준선은 이 300문항을 같은 6000에서 실행한 **185/300**이다. 세 후보를 병합→20문항 스모크→300문항 평가 순서로 실행했다. 세 후보 모두 입력 문자열과 문서 해시가 원본과 **300/300 일치**하고 평가 종료 상태는 0이다.

| 이번 후보 | 정오 일치/300 | 추출 답 일치/300 | 최종 답 누락 | 병합·스모크·300 평가 시간(초) | 파일 |
|---|---:|---:|---:|---|---|
| GRPO | 273 | 248 | 31 | 15 · 295 · 936 | [요약](grpo/comparison_summary.json) · [문항별](grpo/paired_items.csv) · [인계](grpo/handoff.json) |
| DAPO | 267 | 232 | 17 | 15 · 286 · 751 | [요약](dapo/comparison_summary.json) · [문항별](dapo/paired_items.csv) · [인계](dapo/handoff.json) |
| SAPO | 259 | 231 | 22 | 15 · 315 · 821 | [요약](sapo/comparison_summary.json) · [문항별](sapo/paired_items.csv) · [인계](sapo/handoff.json) |

원본의 최종 답 누락은 36/300이다. GRPO와 DAPO는 총점이 같지만 서로 다른 14문항씩을 맞혔고, 서로의 정오 일치는 272/300, 추출 답 일치는 248/300이다. GRPO–SAPO의 일치는 정오 264/300·추출 답 243/300, DAPO–SAPO는 262/300·239/300이다. SAPO의 +1문항은 개선 21개와 악화 20개가 상쇄된 결과다. 생성 종료 사유가 저장되지 않았으므로 최종 답 누락을 잘림으로 단정하지 않는다.

실행 기록은 [학습 큐 요약](training_summary.json), [평가 큐 요약](evaluation_summary.json), 후보별 [GRPO](grpo/result.json)·[DAPO](dapo/result.json)·[SAPO](sapo/result.json) 결과 JSON에 보존했다. 복사한 결과 파일의 SHA256은 [무결성 목록](artifact_hashes.json)에 있다. 최초 평가 큐는 실행기의 `merge_timeout` 인자 누락으로 **병합 전에** 종료됐다. `run_mixed_loss_fixed300.py`에 기본값 3,600초를 추가하고 새 접두사 `rlab19mix1000qcv2r1`에서 사전 검증부터 재실행해 3개 평가를 완료했다. 최초 실패 기록은 보존했다.

## 다음 판단

전체 후보의 300문항 선별 상위 2개는 여전히 **ECD GRPO(195)**와 **ECD DAPO(193)**다. 전체 순위는 [기계 판독용 목록](overall_ranking.json)에도 기록했다. 이번 혼합 실험 내부의 GRPO·DAPO는 공동 1위이며, 혼합 후보를 추가 900문항에 올릴지는 전체 후보와 학습 조건을 함께 검토한 뒤 결정해야 한다. 현재 [기존 900문항 게이트](../rl_ablation_20261119_qcv2/review_gate.json)는 `awaiting_user_review`이고 [평가 큐 요약](evaluation_summary.json)의 `official_900_started`는 `false`다. 사용자 검토 전에는 900문항을 실행하지 않는다. 팀 최종 성능은 합의한 4090 공식 3시드 평가에서 확정한다.
