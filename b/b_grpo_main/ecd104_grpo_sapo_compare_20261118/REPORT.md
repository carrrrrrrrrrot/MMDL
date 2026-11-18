# B 실제 ECD 104문항: GRPO와 SAPO 손실 비교

2026-11-18 KST. **A의 공통 SFT 어댑터를 받기 전**, 고정 원본 모델에서 시작한 B의 탐색 실험이다. C가 기존에 승인한 실제 ECD 차트 104문항·이미지 104장을 사용했다. 새로 선별한 800문항이나 B가 예전에 만든 합성 이미지는 사용하지 않았다.

## 고정 조건

- 부모: `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- 데이터: C 승인 `/data/chaewon/mmdl_grpo_data/ecd_10k/base_ecd_pilot_001/grpo_manifest.jsonl`, SHA256 `d7dbefc8281e2b4073b15f67bb44c2b3e0bdfc6aeb15c226e7a8507b6bf96d16`.
- 보상: v1.2 채점과 같은 0/1, 20개 보상 회귀 사례 통과. 두 방법은 같은 코드와 보상 파일을 사용했다.
- 학습: TRL 1.14.2 `GRPOTrainer`, PEFT 0.21.0, transformers 4.57.1, BF16, 언어부 q/k/v/o LoRA r16/α32, seed 3407, LR 1e-5, 512토큰 상한, 문항당 생성 4개, 64 optimizer steps.
- 핵심 차이: `loss_type=grpo` 또는 `loss_type=sapo`만 변경. 둘 다 `num_iterations=2`, 즉 생성 응답 한 세트를 두 번 갱신에 사용했다. SAPO 온도는 positive 1.0, negative 1.05. `num_iterations=1`에서는 두 손실의 초기 미분이 사실상 같으므로 비교에 쓰지 않았다.

## 학습 결과

| 지표 | GRPO | SAPO |
| --- | ---: | ---: |
| optimizer steps | 64 | 64 |
| 실제 생성 그룹 / 응답 | 32 / 128 | 32 / 128 |
| 보상 0과 1이 섞인 그룹 | 4/32 | 4/32 |
| 보상 1 | 71/128 | 71/128 |
| 최종 `Answer:` 추출 실패 | 11/128 | 12/128 |
| LoRA 변화량 L1 | 404.09 | 428.01 |
| 최대 CUDA 할당 | 10,375 MiB | 10,375 MiB |
| 어댑터 저장·재로딩 | 통과 | 통과 |
| 학습 완료 시각 | 20:01:40 KST | 20:06:41 KST |

TRL이 같은 생성 응답을 두 optimizer step의 completion log에 기록하므로 `rollouts.jsonl`은 각 방법 256행이다. 이 중 짝수 step은 앞선 홀수 step과 응답·보상이 32쌍 모두 동일하다. 생성된 응답은 **각 방법 128개**로 세야 한다. 두 방법에서 같은 문항과 rollout 번호 128쌍을 대조하면 출력 전문은 26쌍, 0/1 보상은 120쌍이 같다. GRPO만 보상 1인 것은 4개, SAPO만 보상 1인 것도 4개다. 이 숫자는 학습 중 생성 응답에 대한 것으로 MMMU 성능을 뜻하지 않는다.

기존 B의 DAPO 64스텝 실행은 `num_iterations=1`이라 실제 생성 그룹이 64개였다. 이 표의 32개와 노출 문항 수가 달라 DAPO를 동일 조건의 세 번째 방법으로 취급하지 않는다.

## 평가 상태

두 어댑터를 각각 원본에 병합했다. v1.2 seed 3407, 동일한 `--limit 20` 문항 스모크가 모두 종료코드 0으로 끝났다. v1.2 채점 코드로 저장 응답을 문항 ID별 재채점한 결과는 다음과 같다.

| 모델 | 정답 | 원본 대비 정오 전환 | 마지막 답 추출 실패 |
| --- | ---: | --- | ---: |
| 6000 원본 | 14/20 | 기준 | 4/20 |
| GRPO | 15/20 | 오답→정답 1, 정답→오답 0 | 2/20 |
| SAPO | 14/20 | 오답→정답 1, 정답→오답 1 | 4/20 |

GRPO와 SAPO를 직접 짝지으면 정오 일치 17/20, 추출 답 일치 15/20이다. SAPO만 맞은 문항이 1개, GRPO만 맞은 문항이 2개다. [원본→GRPO](grpo_smoke_comparison/comparison_summary.json), [원본→SAPO](sapo_smoke_comparison/comparison_summary.json), [GRPO→SAPO](grpo_vs_sapo_smoke/comparison_summary.json)의 문항별 CSV도 같은 폴더에 있다. **20문항 스모크로 MMMU 성능 향상이나 두 방법의 우열을 주장하지 않는다.**

GRPO 900문항 선별 평가는 실제 처리 표시 623/900에서 중단했다. GPU KV 캐시가 100%에 이르고 남은 시간 추정이 수시간으로 늘어나, A의 새 MAmmoTH 자료 수신 및 C 검수 뒤의 우선 실험에 GPU 시간을 남기기 위해 B가 직접 종료했다. **900문항 결과 파일·점수는 없으며, 부분 처리 수를 점수로 해석하지 않는다.** SAPO 900문항 평가는 시작하지 않았다. 두 방법의 정식 우열과 4090 공식 결과는 미확인이다.

원본 학습 로그와 어댑터: `/data/chaewon/mmdl_grpo_runs/ecd104_grpo_sapo_compare_20261118/`. 이 폴더의 `grpo/`·`sapo/` 매니페스트 및 롤아웃 요약을 함께 보존했다. [롤아웃 중복·대응 검증](rollout_pair_audit.json)은 `analyze_rollouts.mjs`로 재현할 수 있다.
