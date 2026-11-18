# A → B: MAmmoTH GRPO 후보 문항 (11/18)

이 브랜치는 데이터 전달용입니다. a/shared·main과 이력을 공유하지 않는 별도 브랜치이며, B가 받은 뒤 삭제할 예정입니다.

## 받는 법

```bash
git fetch origin a/data_for_b
git worktree add ../data_for_b origin/a/data_for_b   # 또는 git archive
cd ../data_for_b/mammoth_grpo_for_b
cat mammoth_grpo_for_b.tar.part* > mammoth_grpo_for_b.tar
sha256sum -c tar.sha256          # f2417266…80a4
tar -xf mammoth_grpo_for_b.tar   # → mammoth_grpo_for_b/{list*.jsonl, images/, SHA256SUMS.txt, summary.json}
cd mammoth_grpo_for_b && sha256sum -c SHA256SUMS.txt --quiet   # 목록 3개 + 이미지 3,000장
```

`list*.jsonl`, `summary.json`은 tar 없이도 바로 볼 수 있게 밖에도 두었습니다(내용 동일).

## 내용

- 3,000문항 (`part` 필드: part1 1,000 / part2 2,000), 이미지 3,000장 616 MB, 원래 상대경로 유지
- 원천: MAmmoTH-VL shard_1에서 A가 처리한 ok 16,513문항
- 필드: `id, idx, source, question`(프롬프트 접두 없음)`, options, answer`(풀이에서 추출한 최종답)`, answer_type`(mc/numeric/word)`, image, image_sha256, image_bytes, used_in, base_k_correct, text_only_suspect, part`

## 선별 기준과 제외 수

1. 최종답 채점 가능: 보기 문자·숫자(단위/% 허용)·20자 이하 단어. 제외: yes/no 2,900, 여러 값 1,493, 여러 단어 540, 근삿값 서술 109, 괄호·hex 43, LaTeX 잔재 43, 범위 6 등 (A의 ECD gold 필터 규칙 재사용)
2. 이미지: PIL로 열기·로드 확인, 단색(흑백 표준편차 < 2) 9장 제외, 깨짐·누락 0
3. 이미지 없이 풀리는 문항: geo170k에서 시각 참조어 없고 숫자 조건이 문장에 다 있는 1,699건 제외. 다른 출처는 휴리스틱 오탐이 많아 제외하지 않고 `text_only_suspect=true`로만 표시(최종 3,000 중 412) → **판단은 B에게**
4. 중복: 같은 이미지 SHA256·같은 정규화 질문은 1문항만
5. 출처 상한: geo170k·clevr 각 25% 이하

## B가 알아야 할 점

- **출처 분포**: mavis 계열 1,214 (40.5%), clevr 664 (22.1%), geo170k 582 (19.4%), geomverse 150, ureader_chart 111, vflan 97, 기타. mavis(규칙 생성 기하)가 가장 큼 — mavis 상한이 필요하면 말해 주세요(약 2,500으로 줄어듦)
- **사용 이력 `used_in`**: sft1 2,100 (70%), dpo_pool 399, dpo_pool+dpo_pairs 132, 미사용 369. SFT1은 기각되어 공통 부모(base)와 무관하지만, A의 DPO 모델 위에 GRPO를 얹을 경우 dpo_pairs 132문항은 중복
- **`base_k_correct`**: DPO 풀에 들어갔던 531문항만 원본 4회 생성(T0.7, 4096토큰, v1.2 채점) 정답 수가 있음: 0개 71 / 1개 32 / 2개 32 / 3개 68 / 4개 328. 나머지는 null
- 누수 검사: SFT1 사용 2,100문항은 C의 SFT1 검사(누수 0) 범위 안. 나머지 900문항은 C 검수 필요
