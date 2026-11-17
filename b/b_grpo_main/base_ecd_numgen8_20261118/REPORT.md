# B 6000 base→GRPO rollout 4개·8개 진단

2026-11-18 KST. C 승인 ECD 104문항에서 원본 Qwen3-VL-4B-Instruct를 부모로 삼은 **보험 경로**의 학습 신호를 비교했다. 두 실행은 앞 20개 학습 문항의 ID와 순서가 같다. A의 공통 SFT 부모 GRPO나 MMMU 성능 평가는 아니다.

## 비교 조건

Base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, 데이터 매니페스트 SHA256 `d7dbefc8281e2b4073b15f67bb44c2b3e0bdfc6aeb15c226e7a8507b6bf96d16`, v1.2 0/1 보상 정책 SHA256 `2669783e67ff0bdaa8988e32ad4e755052283a56e261f8aa9df57b613575cd2c`를 고정했다. 두 실행 모두 seed 3407, 20 optimizer steps, 생성 상한 512토큰, 학습률 1e-5, BF16 언어부 LoRA r16/α32다. rollout 수를 4→8로 늘리면서 `gradient_accumulation_steps`도 4→8로 바뀌었다. 따라서 결과를 rollout 수만의 순수 효과로 해석할 수 없다.

| 문항당 rollout | 학습 응답 | 보상 1 | 보상이 섞인 그룹 | 최종 `Answer:` 행 | 상한 근접 프록시 | Trainer 시간 | LoRA 변화 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 (기존) | 80 | 39/80 | 2/20 | 72/80 | 8/80 | 151.87초 | 119.22 |
| 8 | 160 | 80/160 | 3/20 | 145/160 | 16/160 | 225.60초 | 113.76 |

두 실행 모두 20스텝 완료, 유한 손실, LoRA 변경, 저장한 어댑터 재로딩에 성공했다. 각각 80/80·160/160 응답의 v1.2 보상 재계산 불일치는 0건이다. 8-rollout 실행에서 보상 혼합 그룹이 2→3개로 늘었지만 **17/20 그룹은 여전히 보상이 모두 같아**, 학습 신호 개선은 제한적이다. Trainer 시간은 약 74초 증가했다. `finish_reason`과 정확한 생성 토큰 수가 없어 상한 도달 여부는 재토큰화 프록시로만 기록했다.

이 두 어댑터의 차이를 MMMU 900문항으로 평가하지 않았다. 학습 중 보상이나 답 형식 수치만으로 어느 어댑터가 더 좋은지 순위를 매길 수 없다. 기존 512토큰·64스텝 후보의 [후속 seed 1 평가](../../b_eval_compare/base_ecd_cap512_steps64_seed1_20261118_screen/REPORT.md)도 시드별 결과가 달랐으므로, 보험 경로의 단일 시드 수치를 안정적인 향상으로 해석하지 않는다.

## 원본 기록

| 조건 | 실행 경로 | 어댑터 SHA256 |
|---|---|---|
| rollout 4개 | `/data/chaewon/mmdl_grpo_runs/base_ecd_pilot_cap512_001` | `03221455fe450358f6b97434089a3a73c30516e37f2c48bf3b55fd33567bb228` |
| rollout 8개 | `/data/chaewon/mmdl_grpo_runs/base_ecd_numgen8_cap512_steps20_lr1e5_20261118` | `80947770ee2b1bb22b8b5e0d960a6acb78e16a91231a2d1e8cba1a1ec56b78fd` |

각 경로의 `run_manifest.json`, `rollout_summary.json`, `rollouts.csv`가 측정 원본이다. 어댑터·이미지·개별 생성 출력은 GitHub에 올리지 않는다.
