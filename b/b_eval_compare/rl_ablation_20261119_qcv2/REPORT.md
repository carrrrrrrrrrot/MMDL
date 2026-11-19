# 6000 GRPO 계열 고정 300문항 선별 평가 (진행 중)

2026-11-19, RTX PRO 6000에서 평가 코드 `61025850563d0331e365bc974a63c0d3c4ed4c2c`와 v1.2 채점을 사용한다. 모든 후보를 시드 3407의 동일한 MMMU val 300문항(`ids_sha256=1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d`)으로 평가한다. 원본 모델을 이 300문항에서 다시 실행한 기준선은 **185/300(61.67%)**이다.

| 학습 데이터 | 방법 | 정답/300 | 기준선 대비 | 오답→정답 | 정답→오답 | 최종 답안 누락 |
|---|---|---:|---:|---:|---:|---:|
| ECD | GRPO | **195** | **+10문항 (+3.33%p)** | 23 | 13 | 29 (기준선 36) |
| ECD | DAPO | **193** | **+8문항 (+2.67%p)** | 19 | 11 | 29 (기준선 36) |
| ECD | SAPO | **184** | **−1문항 (−0.33%p)** | 17 | 18 | 31 (기준선 36) |
| MAmmoTH CLEVR | GRPO | **186** | **+1문항 (+0.33%p)** | 18 | 17 | 28 (기준선 36) |
| MAmmoTH CLEVR | DAPO | 평가 중 |  |  |  |  |
| MAmmoTH CLEVR | SAPO | 대기 |  |  |  |  |

ECD GRPO의 [비교 요약](ecd_grpo/comparison_summary.json), [문항별 대응표](ecd_grpo/paired_items.csv), [평가 인계 정보](ecd_grpo/handoff.json)를 보존했다. 300개 입력과 문서 해시는 모두 기준선과 일치한다. 정오 일치는 264/300, 추출 답 일치는 242/300이다. 비교 요약과 CSV의 SHA256은 각각 `dfaf8349f50e09cec8fc8cecfaa066e7d5266c7cbf7f50fabfd795055e11b2d6`, `a239f55d214dd546347dbf696a589c8fcd08d5eaaeabf9f7b9d93d4ff1ab5eed`다.

ECD DAPO의 [비교 요약](ecd_dapo/comparison_summary.json), [문항별 대응표](ecd_dapo/paired_items.csv), [평가 인계 정보](ecd_dapo/handoff.json)도 보존했다. 300개 입력과 문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `e3aac38e451dcb07787ca366bc9a956907a8c60462c4e0960a71737376b541b7`, `b36f270fcdf5ac8f223b3fdb657afdcb321fa3b3d74b924f0bb2d7fbc701690f`다. ECD GRPO와 DAPO의 차이는 2/300문항이지만, GRPO만 정답 22건과 DAPO만 정답 20건이 상쇄됐다. 현재 순위로 방법 간 우열을 결론내리지 않는다.

ECD SAPO의 [비교 요약](ecd_sapo/comparison_summary.json), [문항별 대응표](ecd_sapo/paired_items.csv), [평가 인계 정보](ecd_sapo/handoff.json)도 보존했다. 300개 입력과 문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `2972fbed954c2ce1e1a575a449b67fb63e3026360248941bece31b0efebf8c7c`, `d9a28b23716f8a8e0110278be7c3d435c51807b6a869c345b58aa955e5d97c43`다. 세 ECD 후보의 결과는 모두 단일 시드 선별 수치다.

MAmmoTH GRPO의 [비교 요약](mammoth_grpo/comparison_summary.json), [문항별 대응표](mammoth_grpo/paired_items.csv), [평가 인계 정보](mammoth_grpo/handoff.json)를 보존했다. 300개 입력·문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `c8ed5f3ce83ddb8f7044c1b65a7f12d9b4185c7cce409df8d42b288567ba34a2`, `1e3485b909c2bcc1b9b228f3893a9c0544380357ec161baf2e6a6d881e6f01e4`다. 이 학습은 **MAmmoTH-VL 전체가 아니라 CLEVR 출처 500문항**으로 했다. 학습 중 보상률이 높았어도 이번 MMMU 300문항에서의 순증은 1문항이므로 일반화 개선의 근거로 쓰지 않는다.

과거 6000 원본 900문항 결과에서 같은 300문항을 뽑으면 총점은 185/300으로 같지만, 이번 원본 실행과 문항별 정오가 34건 다르다. 그러므로 위 차이는 **이번에 함께 실행한 기준선**과 대응 계산했다. 300문항·단일 시드는 후보 선별용 진단이며 공식 MMMU 점수나 효과 확정이 아니다. 생성 종료 사유가 기록되지 않아 최종 답안 누락을 잘림으로 단정하지 않는다. 6개 결과의 순위와 문항별 오류를 사용자에게 보여 주기 전에는 상위 2개의 900문항 평가를 시작하지 않도록 [검토 게이트](../../b_eval_ops/rl_ablation_20261119/REVIEW_GATE.md)가 설치돼 있다.
