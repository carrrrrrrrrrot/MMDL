# B MAmmoTH GRPO 검토 후보 선별 (11/18)

입력은 A가 전달한 [3,000문항 목록](mammoth_a_handoff_3000_20261118.jsonl)이며 SHA256은 `686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55`다. [선별 스크립트](select_mammoth_candidates.py)는 seed 3407과 ID 해시로 문항 순서를 고정한다. MAVIS 등 **공개 MAmmoTH 생성 자료는 허용**하고, B가 직접 만든 합성 자료는 입력에 없다.

| 검토 목록 | 범위 | 문항 수 | 출처·형식 | 현재 상태 |
| --- | --- | ---: | --- | --- |
| [SFT1 사용 이력 후보](mammoth_grpo_priority_300_sft1_checked_seed3407.jsonl) | C가 과거 SFT1용 누수를 검사한 2,100건에서 텍스트만으로 풀릴 의심 412건과 질문 형식 이상 2건을 제외한 1,686건 | 300 | MAVIS 75, CLEVR 60, Geo170K 60, 차트·표 45, vision-flan 30, GeomVerse 15, 기타 15; MC 116, 숫자 154, 단어 30 | C의 문항별 정답·이미지·GRPO 적합성 승인 필요 |
| [완전 미사용 1차](mammoth_c_review_priority_100_unused_seed3407.jsonl), [C 검토 CSV](mammoth_c_review_priority_100_unused_seed3407.csv) | A의 SFT1·DPO 어디에도 쓰지 않은 369건 | 100 | CLEVR 33, Geo170K 67; MC 67, 숫자 31, 단어 2 | C의 MMMU/MMMU-Pro 누수와 문항별 정답·이미지 승인 필요 |

첫 목록은 MAVIS가 원본 3,000건의 40.5%인 것과 달리 25%로 제한해 출처 편중을 줄였다. **이 비율은 실험용 다양성 설정일 뿐, MMMU 최적 비율로 검증된 값은 아니다.** 두 번째 목록의 완전 미사용 풀에는 CLEVR가 33건, Geo170K가 336건뿐이어서 첫 100건에 CLEVR 33건이 모두 들어갔다. 두 목록 간 ID·이미지는 겹치지 않는다. 문항과 이미지 내용의 실제 정답성 및 GRPO 보상 분산은 C 검수와 B 파일럿에서 확인해야 한다.

SFT1 후보 300의 SHA256은 `a4c1e2174c5af4f71c227828492b8399e13833109d7d1ada8110bf4e233bb2e9`; 완전 미사용 첫 100의 SHA256은 `e6b19b858bee4fc6cc9bac18720e7f3e3a25a35fe5a003d7ae20f6e9df606e78`이다. [300건 통계](mammoth_grpo_priority_300_sft1_checked_seed3407.summary.json)와 [100건 통계](mammoth_c_review_priority_100_unused_seed3407.summary.json)에 선별 이력과 제외 수를 기록했다. 둘 다 `training_ready: false`다.

C에게는 첫 100건부터 검토를 부탁드린다. B의 변환기는 실제 C 검토 파일의 각 승인 행에 `source`, `id`, `approval: "approved"`, `mmmu_leakage: "pass"`, `quality: "pass"`, `split: "train"`, `reviewer: "C"`가 있어야 해당 문항을 통과시킨다. 거절 행에는 `approval: "rejected"`와 사유를 적고, 빠진 문항은 승인으로 간주하지 않는다. A의 이미지 패키지를 C가 받을 때까지 `a/data_for_b` 브랜치를 유지해야 한다.
