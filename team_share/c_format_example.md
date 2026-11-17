# 학습 데이터 형식 예시 (MAmmoTH-VL → v1.2 형식)

작성: A, 2026-11-17. 대상: C(데이터 선별·검수), B(GRPO 프롬프트). 변환 코드: `local/data_prep/convert_v12.py`

## 1. 형식 요약 (v1.2 평가와 같음)

기준 파일: `eval_tasks/mmmu_mmdl/mmdl_utils.py`의 `mmdl_doc_to_text`/`mmdl_doc_to_messages`, `mmmu_val_mmdl.yaml.tmpl` (읽기만 함)

| 역할 | 내용 |
|---|---|
| system | `You are a helpful assistant.` (vLLM 평가에서 task가 넣는 문장과 같음) |
| user | `Question: ` + 질문 본문 [+ `\nOptions:\nA. …\nB. …`] + `\n` + 지시문 |
| 이미지 | 본문의 `<image 1>` 자리에 끼움. MAmmoTH 원본의 `<image>`는 `<image 1>`로 바꿈 (대부분 질문 맨 앞이라 `Question: ` 바로 뒤에 이미지가 옴) |
| 지시문 (객관식) | `Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.` |
| 지시문 (그 외) | `Think step by step, then give your final answer on the last line in the form "Answer: <answer>".` |
| assistant | 추론 + **마지막 줄 `Answer: X`** (객관식은 보기 글자, 그 외는 값) |

- 보기 표기는 평가와 같이 `A. 내용` (lmms-eval `parse_options`). 원본의 `A: …`, `(A) …`, `选项：` 블록은 이 형식으로 바꾼다.
- 원본 질문 안의 답 형식 지시문(`Answer concisely with one word or phrase.`, `First perform reasoning, then finally select … Answer: xxx.` 등)은 v1.2 지시문과 충돌하므로 지운다.
- 이미지 크기: 평가는 `qwen_vl_utils.fetch_image(image_patch_size=16)`로 맞추고 픽셀 수를 [4,096, 16,777,216]로 제한한다. 학습 쪽 max_pixels는 파일럿에서 VRAM을 보고 정한다 (미정).
- 다중 턴 레코드는 첫 질문-답 쌍만 쓰고 나머지 턴은 버린다 (평가가 단일 턴, 2026-11-17 확정).
- B(GRPO): 프롬프트는 위 system + user를 그대로 쓰고, 보상은 v1.2 채점(`mmdl_process_results`와 같은 판정, 정답 1·그 외 0)이다.

## 2. 마지막 줄에 Answer가 없는 원본의 처리

| status | 규칙 | 처리 (권장) |
|---|---|---|
| `ok` (explicit) | 원본 마지막 줄이 `Answer: X` / `Answer:X` / `**Answer:** X` / `- Answer: X` | `Answer: X`로 정규화. 학습에 씀 |
| `ok` (extracted) | 원본 끝 400자에서 `\boxed{X}`, `the answer is X`, `X is the correct answer`를 찾음 | 본문은 그대로 두고 마지막 줄에 `Answer: X`를 덧붙임. 학습에 씀 |
| `short_no_reasoning` | 답이 8단어 이하 한 줄이거나, `The answer is C`처럼 추론이 거의 없음 | **학습 제외 (2026-11-17 확정)** |
| `no_final_answer` | 긴 설명문(캡션·서술)인데 최종 답을 못 찾음 | **학습 제외** |
| `mc_answer_unmapped` | 객관식인데 답이 보기 글자로도, 보기 본문과도 맞지 않음 | 제외 (C 검수) |
| `unsupported` | 첫 턴이 human→gpt가 아님, `<image>`가 1개가 아님 | 제외 |

추가 flags: `long_final_answer`(40자 넘는 문장형 최종답: open 채점에 맞지 않을 수 있어 검수 대상), `multi_turn`(원본 턴 수), `removed_instructions`(지운 지시문), `mapped_option_text`(보기 본문을 글자로 바꿈).

## 3. 부분 데이터 집계 (참고, 확정 아님)

다운로드 중인 파일 앞부분으로 측정했다: shard_1 앞 2 GB(이미지 3,648장) × JSON 앞 약 580만 레코드 → 매칭 6,240건.

| status | 건수 | 비율 |
|---|---|---|
| no_final_answer | 4,507 | 72% |
| short_no_reasoning | 1,226 | 20% |
| ok | 507 | 8% (객관식 249, open 258) |

`ok`의 주 출처: geo170k(qa), clevr_700k, mavis_math_rule_geo, vflan, ureader_chart. 전체 shard 기준 수치는 다운로드가 끝난 뒤 다시 잰다.

## 4. 실제 레코드 변환 예시

`rendered`는 Qwen3-VL chat template을 적용한 실제 학습 문자열이다(이미지는 `<|vision_start|><|image_pad|><|vision_end|>` 자리).

### 예시 1. MC, 원본에 Answer 줄 있음 — status `ok`
- idx 18523, id `4738`, source `geo170k(qa)`, image `LLaVA-OneVision-Data/geo170k(qa)/4738.png`, n_turns 1
- flags: `{"removed_instructions": ["First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx."], "answer_source": "explicit"}`

원본 질문:
```text
<image>
First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.
Question: In triangle ABC, given that DE is parallel to BC, AQ is perpendicular to BC at Q and intersects DE at P, AD is 1.5 units long, BD is 1 unit long, what is the value of (AP/AQ)?
Choices:
A: \frac{3}{2}
B: \frac{3}{5}
C: \frac{4}{9}
D: \frac{9}{25}
```
원본 답:
```text
Since DE is parallel to BC, it follows that AP/AQ = AD/AB = AD/(AD+BD) = 1.5/(1.5+1) = 1.5/2.5 = 3/5. Therefore, B is the correct answer.
Answer:B
```
변환 결과 (rendered):
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: <|vision_start|><|image_pad|><|vision_end|>
In triangle ABC, given that DE is parallel to BC, AQ is perpendicular to BC at Q and intersects DE at P, AD is 1.5 units long, BD is 1 unit long, what is the value of (AP/AQ)?
Options:
A. \frac{3}{2}
B. \frac{3}{5}
C. \frac{4}{9}
D. \frac{9}{25}
Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.<|im_end|>
<|im_start|>assistant
Since DE is parallel to BC, it follows that AP/AQ = AD/AB = AD/(AD+BD) = 1.5/(1.5+1) = 1.5/2.5 = 3/5. Therefore, B is the correct answer.
Answer: B<|im_end|>

```

### 예시 2. open, 본문에서 최종답 추출 — status `ok`
- idx 74171, id `037449`, source `mavis_math_rule_geo_processed_new.json`, image `LLaVA-OneVision-Data/mavis_math_rule_geo/037449.png`, n_turns 1
- flags: `{"removed_instructions": ["According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx"], "answer_source": "extracted"}`

원본 질문:
```text
<image>
According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx
Question: AB measures 8.0. What is the length of FE in shape ADEF?
```
원본 답:
```text
AB is 8.0. Since ABCD is a square and all four sides are equal, AD is also AB = 8.0. AD equals 8.0. As ADEF is a square, all its sides are of equal length. Thus, FE = AD = 8.0.
The answer is 8.0
```
변환 결과 (rendered):
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: <|vision_start|><|image_pad|><|vision_end|>
AB measures 8.0. What is the length of FE in shape ADEF?
Think step by step, then give your final answer on the last line in the form "Answer: <answer>".<|im_end|>
<|im_start|>assistant
AB is 8.0. Since ABCD is a square and all four sides are equal, AD is also AB = 8.0. AD equals 8.0. As ADEF is a square, all its sides are of equal length. Thus, FE = AD = 8.0.
The answer is 8.0
Answer: 8.0<|im_end|>

```

### 예시 3. MC(중국어 보기), 답만 있고 추론 없음 — status `short_no_reasoning`
- idx 1124694, id `identity_24906`, source `GeoQA+(MathV360K)`, image `LLaVA-OneVision-Data/GeoQA+(MathV360K)/identity_24906.png`, n_turns 1
- flags: `{"removed_instructions": ["Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."], "answer_source": "extracted"}`

원본 질문:
```text
<image>
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: 如图,若AB是⊙O的直径,CD是⊙O的弦,∠ABD=50°,则∠C的度数为()
Choices:
(A) 60°
(B) 50°
(C) 40°
(D) 30°
```
원본 답:
```text
The answer is C
```
변환 결과 (rendered):
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: <|vision_start|><|image_pad|><|vision_end|>
如图,若AB是⊙O的直径,CD是⊙O的弦,∠ABD=50°,则∠C的度数为()
Options:
A. 60°
B. 50°
C. 40°
D. 30°
Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.<|im_end|>
<|im_start|>assistant
The answer is C
Answer: C<|im_end|>

```
→ `Answer:` 줄은 만들었지만 추론이 없어 학습에서 제외한다 (2장, 확정).

### 예시 4. 짧은 답 VQA — status `short_no_reasoning`
- idx 14388, id `cauldron/st_vqa/images/st_vqa_00010337.png`, source `st_vqa(cauldron,llava_format)_processed.json`, image `LLaVA-OneVision-Data/st_vqa(cauldron,llava_format)/st_vqa_00010337.png`, n_turns 1
- flags: `{"removed_instructions": ["Answer the question using a single word or phrase."], "answer_source": "short_answer"}`

원본 질문:
```text
<image>
What is on the red sign?
Answer the question using a single word or phrase.
```
원본 답:
```text
Stop.
```
변환 결과 (rendered):
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: <|vision_start|><|image_pad|><|vision_end|>
What is on the red sign?
Think step by step, then give your final answer on the last line in the form "Answer: <answer>".<|im_end|>
<|im_start|>assistant
Answer: Stop<|im_end|>

```
→ `Answer:` 줄은 만들었지만 추론이 없어 학습에서 제외한다 (2장, 확정).

### 예시 5. 긴 서술형, 최종답 없음 — status `no_final_answer`
- idx 1770, id `sharegpt4v(coco)_processed_new.json46876`, source `sharegpt4v(coco)_processed_new.json`, image `LLaVA-OneVision-Data/sharegpt4v(coco)/000000186738.png`, n_turns 1
- flags: `{}`

원본 질문:
```text
<image>

Write a poem that captures the essence of the beach scene, focusing on the contrast between the dynamic action of the frisbee thrower and the serene atmosphere of the setting sun. Consider the emotions and feelings evoked by this juxtaposition.
```
원본 답:
```text
'Sunset in Motion, a Dance Unfurled', 

A silhouette stands, poised in the west,
As frisbee soars, a moment possessed,
The sun's warm rays upon the shore,
A tranquil scene, yet one in store.

The mountains stand, a backdrop high,
As day yields to the evening's sigh,
The sky ablaze with colors bold,
Orange, yellow, young, and old.

Time pauses, in this fleeting glance,
The world united, in a serene dance,
The thrower's movement, wild and free,
A contrast to the evening's symmetry.

The onlookers bask, in the fading light,
Their faces aglow, their hearts alight,
Their worries lost, in the ebbing tide,
As the sun's warmth touches, side by side.

In this brief moment, they are free,
A glimpse of joy, for all humanity,
A dance of contrasts, wild and still,
A testament to life's simple thrills.
```
변환 결과: assistant를 만들지 못함 → **학습 제외**. (user 부분만 참고)
```text
Question: <image>

Write a poem that captures the essence of the beach scene, focusing on the contrast between the dynamic action of the frisbee thrower and the serene atmosphere of the setting sun. Consider the emotions and feelings evoked by this juxtaposition.
Think step by step, then give your final answer on the last line in the form "Answer: <answer>".
```
