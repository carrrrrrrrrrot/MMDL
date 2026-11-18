# A MAmmoTH 3,000문항 수신 및 C 검수 요청 (11/18)

B의 6000 서버에서 A의 `a/data_for_b` 커밋 `a61adb899d3c2274ba98cba7a360c5b4f333979f`을 받아 검증했다. 7개 조각을 결합한 tar의 SHA256 `f24172660a8abe33d027100416c4ad79fd4af9e61edf64fad5fd76d3507b80a4`가 A의 값과 일치했다. tar 안팎 목록 3개가 같고 `SHA256SUMS.txt`의 목록 3개와 이미지 3,000장의 해시, 목록의 이미지 경로·크기·개별 SHA256이 모두 일치했다. 서버 검증 기록은 `/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118/b_verify_report.json`이다.

- [B에 보존한 전체 3,000문항 목록](mammoth_a_handoff_3000_20261118.jsonl), SHA256 `686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55`.
- [C 검수 대상 SFT1 미사용 900문항](mammoth_c_leakage_review_900_20261118.jsonl), SHA256 `e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8`.
- 900문항은 CLEVR 482·Geo170K 418이며, `used_in`이 완전 미사용 369, A DPO 풀 399, DPO 풀과 선호쌍 132다. A의 SFT1에 사용한 2,100문항은 C의 기존 SFT1 누수 검사 범위였다고 [A 전달 설명](https://github.com/carrrrrrrrrrot/MMDL/blob/a/data_for_b/README.md)에 적혀 있다. 새 900문항은 아직 C 검사가 없다.
- 공개 MAmmoTH 안의 생성 이미지·문항은 사용 가능하다. B가 예전에 직접 만든 합성 데이터는 이 실험에서 사용하지 않는다. 원천별 비율과 학습 이력은 별도 기록한다.

## C에게 전달할 문안

> B가 A의 MAmmoTH 3,000문항과 이미지 3,000장을 수신해 SHA256까지 검증했습니다. C에게는 [SFT1 미사용 900문항 목록](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/mammoth_c_leakage_review_900_20261118.jsonl)의 **MMMU/MMMU-Pro 질문·이미지·정답 누수 및 정답 품질 검사**를 부탁드립니다. 목록 SHA256은 `e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8`입니다. 우선 A의 DPO와 겹치지 않는 `used_in=[]` 369문항부터 검사해 주시면 GRPO 첫 본실험에 바로 쓸 수 있습니다. 나머지 DPO 풀 531문항은 이어서 검사해 주세요. 각 ID의 승인 여부와 제외 이유가 담긴 JSONL 또는 CSV를 보내 주세요. 이미지는 [A의 전달 브랜치 README](https://github.com/carrrrrrrrrrot/MMDL/blob/a/data_for_b/README.md)대로 내려받아 해시를 확인할 수 있습니다. C가 자료를 받을 때까지 A의 브랜치는 유지해 주세요.

이 문서는 B의 수신과 검수 요청을 기록한다. 900문항의 C 승인을 뜻하지 않는다.
