# 6000 GRPO 계열 고정 300문항 선별 평가 (완료, 900문항 중단)

2026-11-19, RTX PRO 6000에서 평가 코드 `61025850563d0331e365bc974a63c0d3c4ed4c2c`와 v1.2 채점을 사용한다. 모든 후보를 시드 3407의 동일한 MMMU val 300문항(`ids_sha256=1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d`)으로 평가한다. 원본 모델을 이 300문항에서 다시 실행한 기준선은 **185/300(61.67%)**이다.

**실험 범위:** 여섯 후보의 학습 부모는 모두 `parent_kind=base`, Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`이다. [A의 최신 팀 로그](https://github.com/carrrrrrrrrrot/MMDL/blob/a/shared/team_share/TEAM_LOG.md)는 11/18에 공통 부모를 원본 base로 확정하고 A의 base→DPO와 B의 base→GRPO를 본 경로로 기록한다. 따라서 이번 후보는 A의 DPO와 **부모 조건이 같다**. 이 6000의 300문항·단일 시드 선별 결과는 최종 4090 공통 평가를 대신하지 않는다. DAPO와 SAPO는 동일한 TRL 1.14.2 GRPOTrainer에서 `loss_type`만 바꾼 실험이다. DAPO는 토큰 단위 손실 정규화, SAPO는 부호별 soft gate의 핵심 구현을 시험했으며 각 논문의 전체 학습 조합을 재현한 결과는 아니다.

| 순위 | 학습 데이터 | 방법 | 정답/300 | 기준선 대비 | 오답→정답 | 정답→오답 | 최종 답안 누락 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | ECD | GRPO | **195** | **+10문항 (+3.33%p)** | 23 | 13 | 29 (기준선 36) |
| 2 | ECD | DAPO | **193** | **+8문항 (+2.67%p)** | 19 | 11 | 29 (기준선 36) |
| 3 | MAmmoTH CLEVR | SAPO | **189** | **+4문항 (+1.33%p)** | 22 | 18 | 37 (기준선 36) |
| 4= | MAmmoTH CLEVR | GRPO | **186** | **+1문항 (+0.33%p)** | 18 | 17 | 28 (기준선 36) |
| 4= | MAmmoTH CLEVR | DAPO | **186** | **+1문항 (+0.33%p)** | 18 | 17 | 31 (기준선 36) |
| 6 | ECD | SAPO | **184** | **−1문항 (−0.33%p)** | 17 | 18 | 31 (기준선 36) |

[기계적 선별 순위](ranking.json)의 상위 2개는 ECD GRPO와 ECD DAPO다. 이 둘의 총점 차이는 2문항이고, 2위 ECD DAPO와 3위 CLEVR SAPO의 차이도 4문항이다. ECD DAPO만 정답인 문항 22개와 CLEVR SAPO만 정답인 문항 18개가 상쇄됐다. **300문항·단일 시드에서 이 작은 순위 차이를 확정적 방법 우열로 해석하지 않는다.** 사용자가 이 결과를 검토하고 900문항 평가를 승인해 [게이트 상태](review_gate.json)가 `released_after_user_review`로 바뀌었다. 순위 파일 SHA256은 `7ff125e2ceb1de2f600a9aa3a08af6a096c97b5f4450a9b3d9b32b4385455f9d`다.

ECD GRPO의 [비교 요약](ecd_grpo/comparison_summary.json), [문항별 대응표](ecd_grpo/paired_items.csv), [평가 인계 정보](ecd_grpo/handoff.json)를 보존했다. 300개 입력과 문서 해시는 모두 기준선과 일치한다. 정오 일치는 264/300, 추출 답 일치는 242/300이다. 비교 요약과 CSV의 SHA256은 각각 `dfaf8349f50e09cec8fc8cecfaa066e7d5266c7cbf7f50fabfd795055e11b2d6`, `a239f55d214dd546347dbf696a589c8fcd08d5eaaeabf9f7b9d93d4ff1ab5eed`다.

ECD DAPO의 [비교 요약](ecd_dapo/comparison_summary.json), [문항별 대응표](ecd_dapo/paired_items.csv), [평가 인계 정보](ecd_dapo/handoff.json)도 보존했다. 300개 입력과 문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `e3aac38e451dcb07787ca366bc9a956907a8c60462c4e0960a71737376b541b7`, `b36f270fcdf5ac8f223b3fdb657afdcb321fa3b3d74b924f0bb2d7fbc701690f`다. ECD GRPO와 DAPO의 차이는 2/300문항이지만, GRPO만 정답 22건과 DAPO만 정답 20건이 상쇄됐다. 현재 순위로 방법 간 우열을 결론내리지 않는다.

ECD SAPO의 [비교 요약](ecd_sapo/comparison_summary.json), [문항별 대응표](ecd_sapo/paired_items.csv), [평가 인계 정보](ecd_sapo/handoff.json)도 보존했다. 300개 입력과 문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `2972fbed954c2ce1e1a575a449b67fb63e3026360248941bece31b0efebf8c7c`, `d9a28b23716f8a8e0110278be7c3d435c51807b6a869c345b58aa955e5d97c43`다. 세 ECD 후보의 결과는 모두 단일 시드 선별 수치다.

MAmmoTH GRPO의 [비교 요약](mammoth_grpo/comparison_summary.json), [문항별 대응표](mammoth_grpo/paired_items.csv), [평가 인계 정보](mammoth_grpo/handoff.json)를 보존했다. 300개 입력·문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `c8ed5f3ce83ddb8f7044c1b65a7f12d9b4185c7cce409df8d42b288567ba34a2`, `1e3485b909c2bcc1b9b228f3893a9c0544380357ec161baf2e6a6d881e6f01e4`다. 이 학습은 **MAmmoTH-VL 전체가 아니라 CLEVR 출처 500문항**으로 했다. 학습 중 보상률이 높았어도 이번 MMMU 300문항에서의 순증은 1문항이므로 일반화 개선의 근거로 쓰지 않는다.

MAmmoTH DAPO의 [비교 요약](mammoth_dapo/comparison_summary.json), [문항별 대응표](mammoth_dapo/paired_items.csv), [평가 인계 정보](mammoth_dapo/handoff.json)를 보존했다. 300개 입력·문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `8cf68f753709a26f1c1a241937cc5b49b076b01d9fed8e8056d83fffb7f80052`, `3470be5ef4fc55767424946009d56b83cb833026e0eb002a2f3f0d83af928bbd`다. CLEVR GRPO와 DAPO는 둘 다 186/300이지만 서로 정오가 다른 문항이 36개이며, 각 방법에서만 맞은 문항이 18개씩이다. 같은 점수라도 같은 답변 동작은 아니다.

MAmmoTH SAPO의 [비교 요약](mammoth_sapo/comparison_summary.json), [문항별 대응표](mammoth_sapo/paired_items.csv), [평가 인계 정보](mammoth_sapo/handoff.json)를 보존했다. 300개 입력·문서 해시는 기준선과 일치했다. 비교 요약과 CSV의 SHA256은 각각 `39d1d110ce705d1b1a0c1e363ba75231434366411006f26e888332dc4cfb3388`, `cf7d729773a470e9e5094c0105bf3598db10633b5b980f18fd19c6fd00e06de5`다. 이 모델도 MAmmoTH-VL의 **CLEVR 출처 500문항**만 학습했다.

과거 6000 원본 900문항 결과에서 같은 300문항을 뽑으면 총점은 185/300으로 같지만, 이번 원본 실행과 문항별 정오가 34건 다르다. 그러므로 위 차이는 **이번에 함께 실행한 기준선**과 대응 계산했다. 300문항·단일 시드는 후보 선별용 진단이며 공식 MMMU 점수나 효과 확정이 아니다. 생성 종료 사유가 기록되지 않아 최종 답안 누락을 잘림으로 단정하지 않는다. [검토 게이트 절차](../../b_eval_ops/rl_ablation_20261119/REVIEW_GATE.md)에 따라 11/19 11:54 KST에 기존 큐가 첫 900문항 실행 직전의 마커에서 의도대로 정지했다. 이 때문에 기존 supervisor의 `failed` 표시는 **검토를 위한 제어된 정지**다.

11/20 00:20 KST에 사용자의 검토 후 게이트를 해제하고 원래 평가 큐 명령을 재실행했다. ECD GRPO 시드 3407의 900문항 생성이 진행 중일 때 사용자가 ECD 학습 문항 수 확대 실험을 먼저 요청하여 00:43 KST에 평가 큐와 GPU 자식 프로세스를 중단했다. [중단 기록](USER_STOP_20261120.json)과 부분 실행 로그는 보존했다. **900문항 평가를 완료하거나 채점한 모델은 0개**이며 ECD DAPO 900문항은 시작하지 않았다. 중단된 GRPO 실행에는 완료 인계 파일 `handoff.json`이 없으므로 점수로 사용하지 않는다. 확대 데이터 실험을 검토한 뒤 새 실행 경로에서 900문항 평가 여부를 다시 결정한다.
