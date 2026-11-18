# C에게 보낼 ECD 검수 요청

아래 문안을 그대로 공유할 수 있습니다.

> B가 실제 ECD에서 GRPO/SAPO 학습 후보 **800문항(차트 800장)**을 선별해 B 브랜치에 올렸습니다. 우선 [첫 100문항 검토표](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/ecd_c_review_first100_seed3407.csv)부터 **질문·이미지·정답 대조와 MMMU/MMMU-Pro 누수 검사**를 부탁드립니다. [전체 후보 JSONL](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/ecd_c_review_priority_800_seed3407.jsonl)의 SHA256은 `e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f`입니다.
>
> B 서버에서 후보 이미지 800/800장은 열리는 것을 확인했습니다. **정답 800개를 전부 검수한 상태는 아닙니다.** 표본 대조에서 원본 정답 오류 두 건(`ecd_002406_219293`, `ecd_000838_199595`)을 발견해 후보에서 제외했습니다. MMMU validation 대상 질문 일치 0건, 이미지 해시 후보 1건은 원본 대조로 오탐이었지만, [이 1차 점검](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/ECD_MMMU_PRECHECK.md)은 MMMU-Pro·dev/test·OCR·의미 유사도를 포함하지 않습니다.
>
> 첫 100건부터 검수 통과 문항을 `id`, `question`, `final_answer`, `image_relative_path`, `selection_status: "approved"`가 있는 JSONL로 돌려주시고, 제외한 ID와 이유도 함께 알려 주세요. 그다음 나머지 700건을 순차 검수하면 됩니다. [선별 기준과 겹침 현황](https://github.com/carrrrrrrrrrot/MMDL/blob/codex/b_grpo_impl_20261117/b/b_grpo_data/ECD_SELECTION_HANDOFF.md)에 적었습니다. 이미지 접근이 필요하면 B 서버의 ECD 이미지 경로(`/data/chaewon/mmdl_grpo_data/ecd_10k/images`)를 기준으로 전달 방법을 맞추겠습니다. C 승인 전까지 새 800건은 학습에 쓰지 않겠습니다.

검수 범위와 결과 형식을 명시하기 위한 문안이며, GitHub 링크의 자료는 C의 승인을 뜻하지 않습니다.
