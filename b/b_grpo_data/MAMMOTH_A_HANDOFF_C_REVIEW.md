# A MAmmoTH 3,000문항 수신 및 C 검수 요청 (11/18)

B의 6000 서버에서 A의 `a/data_for_b` 커밋 `a61adb899d3c2274ba98cba7a360c5b4f333979f`을 받아 검증했다. 7개 조각을 결합한 tar의 SHA256 `f24172660a8abe33d027100416c4ad79fd4af9e61edf64fad5fd76d3507b80a4`가 A의 값과 일치했다. tar 안팎 목록 3개가 같고 `SHA256SUMS.txt`의 목록 3개와 이미지 3,000장의 해시, 목록의 이미지 경로·크기·개별 SHA256이 모두 일치했다. 서버 검증 기록은 `/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118/b_verify_report.json`이다.

- [B에 보존한 전체 3,000문항 목록](mammoth_a_handoff_3000_20261118.jsonl), SHA256 `686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55`.
- [C 검수 대상 SFT1 미사용 900문항](mammoth_c_leakage_review_900_20261118.jsonl), SHA256 `e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8`.
- [가장 먼저 볼 완전 미사용 100문항 CSV](mammoth_c_review_priority_100_unused_seed3407.csv), [동일 JSONL](mammoth_c_review_priority_100_unused_seed3407.jsonl). 이 100건은 A의 SFT1·DPO 모두에 쓰이지 않았고, B 학습에 아직 쓰지 않았다. 선별 기준과 별도 300건 후보는 [선별 보고서](MAMMOTH_SELECTION_20261118.md)에 적었다.
- 900문항은 CLEVR 482·Geo170K 418이며, `used_in`이 완전 미사용 369, A DPO 풀 399, DPO 풀과 선호쌍 132다. A의 SFT1에 사용한 2,100문항은 C의 기존 SFT1 누수 검사 범위였다고 [A 전달 설명](https://github.com/carrrrrrrrrrot/MMDL/blob/a/data_for_b/README.md)에 적혀 있다. 새 900문항은 아직 C 검사가 없다.
- 공개 MAmmoTH 안의 생성 이미지·문항은 사용 가능하다. B가 예전에 직접 만든 합성 데이터는 이 실험에서 사용하지 않는다. 원천별 비율과 학습 이력은 별도 기록한다.
- [B의 validation 1차 중복 점검](MAMMOTH_MMMU_PRECHECK.md)은 900건의 질문 문자열 일치와 이미지 dHash 가까운 쌍을 찾지 못했다. MMMU-Pro·다른 분할·OCR·의미 유사도는 미검사이므로 C 최종 판단이 필요하다.
- B 변환기는 C 검수 파일의 **`source`와 `id`가 모두 일치**하고 `approval="approved"`, `mmmu_leakage="pass"`, `quality="pass"`, `split="train"` 또는 `"heldout"`, `reviewer="C"`인 문항만 받는다. 제외 문항은 `approval="rejected"`와 이유를 적으면 된다. 예: `{"source":"clevr_700k_new.json","id":"mammoth_123","approval":"approved","mmmu_leakage":"pass","quality":"pass","split":"train","reviewer":"C"}`. 예시 ID는 실제 문항이 아니다.

## C에게 전달할 문안

> B가 A의 MAmmoTH 3,000문항과 이미지 3,000장을 수신해 SHA256까지 검증했습니다. **이번 누수 검수 범위는 SFT1 미사용 900건 전체**입니다. A의 DPO와도 겹치지 않는 [완전 미사용 우선 100문항](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/mammoth_c_review_priority_100_unused_seed3407.csv)을 먼저 전달해 주시면 B가 20스텝 보상 신호 파일럿을 시작하겠습니다. C는 이어서 나머지 완전 미사용 269건, DPO 풀 531건을 계속 검수해 주세요. 승인 문항이 가능하면 300건 이상 쌓였을 때 별도 본실험을 진행하겠습니다. [SFT1 미사용 전체 900문항 목록](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/mammoth_c_leakage_review_900_20261118.jsonl)의 SHA256은 `e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8`입니다. B의 [validation 1차 점검](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/MAMMOTH_MMMU_PRECHECK.md)은 질문·이미지 중복 0건이지만 다른 분할·의미/OCR 유사성은 확인하지 못했습니다. 승인 건은 `source,id,approval,mmmu_leakage,quality,split,reviewer` 필드가 있는 JSONL 또는 CSV로, 제외 건은 `approval="rejected"`와 이유로 돌려주세요. 필드 예시는 이 문서 위에 있습니다. 이미지는 [A의 전달 브랜치 README](https://github.com/carrrrrrrrrrot/MMDL/blob/a/data_for_b/README.md)대로 내려받아 해시를 확인할 수 있습니다. C가 자료를 받을 때까지 A의 브랜치는 유지해 주세요.

이 문서는 B의 수신과 검수 요청을 기록한다. 900문항의 C 승인을 뜻하지 않는다.
