# B ECD 후보 선별 및 C 검수 인계 (2026-11-18)

## 현재 상태

`ecd_c_review_priority_800_seed3407.jsonl`은 **검수 후보**이며 학습 승인을 뜻하지 않는다. B의 새 GRPO/SAPO 비교 학습에는 이 목록을 사용하지 않는다. 비교 학습에는 기존 C 승인 ECD 104문항 매니페스트만 사용한다.

| 단계 | 수량 | 확인 내용 |
| --- | ---: | --- |
| A 공유 ECD 후보 원본 | 62,604문항 | `team_share/ecd_candidates_for_c.jsonl`, SHA256 `7065fc36cf148741c85ecf06be7a61948081c9dc422ef11b1103262847451e5c` |
| 넓은 1차 목록 | 2,000문항, 차트 2,000장 | 정답 형식과 중복을 메타데이터로 점검. `ecd_previsual_2000_seed3407.jsonl` |
| C 우선 검수 목록 | 800문항, 차트 800장 | 단순 수치 정답, 근삿값 질문·SFT1 동일 문항·A DPO 후보·확인된 원본 정답 오류 제외. `ecd_c_review_priority_800_seed3407.jsonl`, SHA256 `e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f` |
| 실제 이미지 확인 | 800/800장 열림 | B 6000의 `/data/chaewon/mmdl_grpo_data/ecd_10k/images`에서 PIL로 전체 디코딩. 이는 정답 검수가 아니다. |

우선 검수 목록에서는 ID·차트·정규화 질문 중복이 0이다. SFT1과 **같은 문항**은 0이고, **같은 차트의 다른 문항**은 421개다. A DPO 후보와 겹치는 문항은 0이다. SFT1과의 차트 겹침은 일부러 기록하고 허용했다. 정답의 `%`·통화 기호·천 단위 쉼표·소수 셋째 자리 이상은 우선 목록에서 제외했다. 목록의 `selection_status`는 모두 `metadata_only_requires_image_gold_and_leakage_review`다.

B 서버에 있던 C의 기존 승인 104문항과 비교하면 새 800건 중 **동일 문항 6건, 동일 차트 33건**이다. 첫 검토표 100건에는 동일 문항이 0건이다. 기존 104건은 6000의 과거 파일럿에 이미 사용됐으므로, 새 검수에서는 나머지 문항부터 볼 수 있다.

[MMMU validation 1차 중복 점검](ECD_MMMU_PRECHECK.md)은 질문 일치 0건, dHash 유사 후보 1건(이미지 대조 후 오탐)이었다. MMMU-Pro·다른 MMMU split·OCR·의미 검사는 이 점검에 포함되지 않았다.

## 원본 정답 오류 확인

이미지 표본을 직접 보니 A가 공유한 ECD 원본 QA에도 잘못된 정답이 있다. 아래 두 문항은 우선 목록에서 제외했다.

- `ecd_002406_219293`: 질문은 바이올린 차트의 Concentration Levels 평균을 묻는다. 이미지에는 **Mean: 2.82**가 적혀 있으나 원본 정답은 **2.63**이다.
- `ecd_000838_199595`: 원본 풀이에는 Inflation Rate의 Q1/Q4 값이 **92.1/83.2**라고 적혀 있으나 차트의 Inflation Rate 계열은 해당 값이 아니다. 원본 정답 **8.9%**는 이 풀이로 검증되지 않는다.

이 두 사례가 전체 오류율을 뜻하지는 않는다. 추가 표본에서 여러 차트는 눈으로 해석 가능한 것으로 보였지만, 800문항 모두의 정답을 독립 검증하지는 않았다. `ecd_visual_rejects_20261118.jsonl`에 제외 근거를 기록했다.

## C에게 필요한 작업

1. 우선 800건 중 학습에 쓸 첫 묶음(최소 100건)을 질문·차트·정답과 대조하고, 원본 정답 오류나 애매한 질문은 제외한다. `ecd_c_review_first100_seed3407.csv`는 첫 검토 묶음이고, `ecd_c_review_priority_first24_seed3407.csv`는 빠른 시작용 표다.
2. 승인할 문항에 대해 MMMU와 MMMU-Pro의 이미지·질문·OCR·의미 유사도를 검사한다. 기존 SFT1 누수 검사만으로 신규 ECD 전체가 검증되지는 않는다.
3. 승인 JSONL은 기존 네 필드 `id`, `question`, `final_answer`, `image_relative_path`를 유지한다. `selection_status`를 `approved`로 바꾸고, 제외 ID와 이유를 별도 표로 남긴다. B 변환기는 `metadata_only_requires_image_gold_and_leakage_review` 상태를 거부한다.
4. B는 C 승인 목록만 A의 `convert_ecd.py` 형식으로 변환하고, 고정 base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`에서 동일 데이터·시드로 GRPO/SAPO를 비교한다. 실제 이미지 파일은 B 서버에 이미 있다.

선별 재현 명령은 다음과 같다. 원본 A 파일과 제외 목록을 입력으로 쓰며, 출력 경로는 실행 때 새 경로를 지정한다.

```bash
node b/b_grpo_data/select_ecd_candidates.mjs \
  --input team_share/ecd_candidates_for_c.jsonl \
  --output b/b_grpo_data/ecd_c_review_priority_800_seed3407.jsonl \
  --limit 800 --seed 3407 --mode priority_review \
  --exclude-ids b/b_grpo_data/ecd_visual_rejects_20261118.jsonl
```

스크립트는 기존 출력 덮어쓰기를 거부한다. 출력과 입력 SHA256 및 탈락 수는 동명 `.summary.json`에 기록된다.
