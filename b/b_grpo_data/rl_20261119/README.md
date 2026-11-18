# 11/19 B 탐색 RL 데이터 고정

C의 [`c/shared` 4c9152f 누수 감사 ZIP](https://github.com/carrrrrrrrrrot/MMDL/blob/c/shared/Audit1700_Leakage_Only_GitHub.zip)의 문항별 1,700행을 B의 ECD 800·MAmmoTH 900 후보 ID와 전수 대조했다. ID 누락·중복은 0건이고 모든 행의 `leakage_result`가 `not_detected_within_scope`다. **C의 범위는 누수 검사뿐이다.** 아래 manifest의 `c_training_approval=false`, `visual_gold_quality_verified=false`는 의도적이며, B의 구조·이미지 무결성 점검을 통과한 탐색 실험으로만 해석해야 한다.

| 파일 | 학습 / 보류 | 선택 |
| --- | ---: | --- |
| `ecd_manifest.jsonl` | 500 / 100 | B 우선순위 800에서 앞 600, 차트 ID 중복 없음 |
| `mammoth_manifest.jsonl` | 500 / 100 | C 신규 감사 900 안에서 완전 미사용 369 + A DPO 선호쌍 100 + DPO 풀 131 |
| `mixed_manifest.jsonl` | 500 / 100 | 위 두 학습 집합에서 ECD 250 + MAmmoTH 250; 위 두 보류 집합에서 50 + 50 |

MAmmoTH 500 학습에는 A의 DPO 문항과 겹치는 문항이 일부 들어간다. 혼합 보상 가능성이 있는 문제를 확보하기 위해 DPO 선호쌍을 100개 포함했으며, 각 행에 `a_used_in_dpo_pair`와 `a_used_in_dpo_pool`을 기록했다. MAmmoTH 출처는 신규 감사 900의 CLEVR·Geo170K 두 종류에 한정된다. A의 SFT1 사용 이력 2,100건은 이 실험에 포함하지 않았다.

ECD 원본 `images.tar` SHA256 `8ae3d9fb5b8345a17f4df420d50b662d5102cc54d46539870b0becad92c09cc1`이 C의 입력 기록과 일치한다. C의 ECD 문항별 `image_sha256`은 모두 null이므로, B가 실제 이미지의 SHA256을 새로 계산하고 디코딩했다. MAmmoTH는 문항별 C 해시, A 목록 해시·크기, 실제 파일을 대조했다. 선택한 1,200개 원본 이미지 검사에서 실패 0건이다. 보류 집합은 학습 입력에서 제외된다.

세 manifest는 B `train.py --validate-only`에서 각각 학습 500건으로 로드됐고 v1.2 보상 20사례 사전검사를 통과했다. `Answer: <gold>`에 대한 v1.2 보상 재계산도 각 manifest 600행씩 전부 1점이었다. 실제 차트·도형을 보고 정답이 맞는지는 아직 확인되지 않았다. [선별 보고서](selection_report.json)에 입력·manifest SHA256과 구성 수를 적었다.

재현 코드는 [manifest 생성기](build_exploratory_manifests.py)와 [정답 보상 확인기](check_manifest_gold_reward.py)다. 이 폴더의 manifest에는 6000 서버의 절대 이미지 경로가 들어 있으며, 실제 학습 경로는 `/data/chaewon/mmdl_grpo_data/rl_ablation_20261119/`이다. C 원본 ZIP과 증빙 이미지는 C 브랜치에 유지하고 B Git에는 다시 올리지 않는다.
