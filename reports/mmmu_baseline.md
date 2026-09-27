# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입)_
- **팀원**: 김다은, 김어령, 서채원
- **작성일**: 2026.9.28
- **재현 커맨드**: `PYTHON=$(which python) bash scripts/run_mmmu_eval.sh --model_path <MODEL_DIR> --data_root <DATA_ROOT> --seeds "3407 1 2" --run_name baseline_reproduction`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.11.0 (Qwen 공식 생성 설정의 presence_penalty=1.5를 적용하고, 900문항을 연속 배치로 효율적으로 처리하기 위해 선택) |
| 사용 GPU | NVIDIA RTX 4090 24 GB × 1 |
| 실측 peak VRAM | 22.54 GiB (23,085 MiB, 3회 실행 중 최대) |
| 총 소요 시간 | 900문항당 seed별 2시간 5분 / 2시간 6분 / 2시간 2분 -> 3회 합계 약 6시간 13분 |
| 의존성 | [requirements.txt](https://github.com/carrrrrrrrrrot/MMDL/blob/main/requirements.txt)  |
| 실행 커맨드 | `PYTHON="$(command -v python)" bash scripts/run_mmmu_eval.sh --model_path "/path/to/Qwen3-VL-4B-Instruct" --data_root "/path/to/MMMU" --seeds "3407 1 2" --run_name baseline_reproduction` |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
• 객관식 프롬프트 설계:
  <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: {question}
Options:
A. {option_A}
B. {option_B}
{나머지 선택지가 있으면 C. ..., D. ...}
Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.<|im_end|>
<|im_start|>assistant

  • 주관식 프롬프트 설계:
  <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: {question}
Think step by step, then give your final answer on the last line in the form "Answer: <answer>".<|im_end|>
<|im_start|>assistant
```

- **출처**: Question:/Options: 틀은 lmms-eval의 Qwen3-VL용 MMMU 프롬프트를 수정했으며, 마지막 Answer: 지시문 및 이미지 interleave는 직접 설계했다.
- **선택 이유**: 기본 프롬프트는 주관식 문항에서 존재하지 않는 선택지를 고르라는 지시가 있었고, 이미지를 텍스트 앞에 모아 배치해 질문·보기에서 이미지가 참조되는 위치를 반영하지 못했다. 이를 수정해 이미지를 <image N>이 등장하는 자리에 삽입하고, 추론 후 마지막 줄에 Answer: 형식으로 최종 답을 쓰도록 했다. 추가로 HF와 vLLM 경로의 입력 토큰 ID가 900문항 모두에서 일치함을 확인했다.

<details>
<summary>작성 형식 예시 (내용은 예시일 뿐입니다. 본인이 실제 찾은/설계한 프롬프트로 교체)</summary>

```
Question: {question}
Choices:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Pick the single best choice from the list above.
```

- **출처**: (예시) 오픈소스 평가 툴킷 XYZ의 프롬프트 생성 함수에서 차용, 문구 일부만 수정
- **선택 이유**: (예시) 모델이 장황한 설명 없이 선택지 하나로 바로 답하도록 유도하기 위해 간결한 지시문 사용

</details>

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | Yes (vLLM sampling; greedy=false) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 요청별 3407, 1, 2로 900문항씩 독립 평가 (공식 seed 3407 + 변동성 확인용 2개) |

- **출처**: [Qwen3-VL 공식 README의 Instruct 모델 생성 설정](https://github.com/QwenLM/Qwen3-VL#generation-hyperparameters)에 제시된 sampling 설정과 `seed=3407`을 사용했다. (추가로 seed 1과 2에서도 평가해 결과 변동성을 확인했으며, seed가 vLLM의 `SamplingParams`에 실제로 전달되는지도 검증했다. 보조 greedy 실험의 정확도는 57.33%로, 최종 sampling 평가의 평균 정확도 61.70%보다 약 4.4%p 낮았다.)

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 16,384 |
| 이미지 해상도 처리 | min_pixels=4096, max_pixels=16777216 |

**선택 근거** <br>RTX 4090 24GB에서 생성 시간과 답변 완결성 사이의 trade-off를 고려해 max_new_tokens=16384로 설정했다. 150문항 pilot에서 공식 설정인 32,768토큰까지 늘렸을 때 추가로 최종 답에 도달한 문항은 1개였지만, 생성 시간은 약 3배 걸렸다. 다만 본 평가에서 약 9.9%가 토큰 제한 내에 Answer:를 출력하지 못했으므로 생성 길이의 영향을 완전히 배제할 수는 없다. 이미지는 fetch_image(image_patch_size=16)으로 크기를 32의 배수에 맞춰 처리했다. 작은 이미지 확대 실험에서는 정확도 향상이 뚜렷하지 않아(+1.34%p, 95% 신뢰구간 −0.89~+3.58%p) 대부분 원본에 가까운 해상도를 유지했다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: 자체 구현한 [`mmdl_process_results`](https://github.com/carrrrrrrrrrot/MMDL/blob/main/eval_tasks/mmmu_mmdl/mmdl_utils.py)로 응답을 전처리한 뒤, 수정하지 않은 lmms-eval의 MMMU 공식 파서 포트인 [`mmmu_process_results`와 `mmmu_aggregate_results`](https://github.com/carrrrrrrrrrot/MMDL/blob/main/code/lmms-eval/lmms_eval/tasks/mmmu/utils.py)를 사용한다.
- 동작 방식 요약: 응답의 마지막 answer:(대소문자 무시) 뒤 첫 비어 있지 않은 줄만 뽑고 \boxed{}, *, 백틱을 제거한다. open 답은 $, 단순 \frac{a}{b}, \text{...} 등의 LaTeX 표기를 일반 텍스트로 정규화하며, 문자열로 저장된 복수 정답은 목록으로 복원한다. 이어 공식 파서로 객관식 선택지 또는 open 정답을 판정한다. 객관식에서 답 글자 후보가 없거나 Answer:가 없으면 오답 처리하며 공식 파서의 random fallback 결과는 참고용으로만 기록한다. open 문항의 추출 실패도 오답이다. 공식 open 판정의 정답 문자열 포함 비교는 그대로 사용한다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 72.2 |
| 2 | Agriculture | 30 | 55.6 |
| 3 | Architecture_and_Engineering | 30 | 40.0 |
| 4 | Art | 30 | 61.1 |
| 5 | Art_Theory | 30 | 76.7|
| 6 | Basic_Medical_Science | 30 |71.1 |
| 7 | Biology | 30 |48.9 |
| 8 | Chemistry | 30 | 44.4|
| 9 | Clinical_Medicine | 30 | 65.6 |
| 10 | Computer_Science | 30 |62.2 |
| 11 | Design | 30 | 75.6|
| 12 | Diagnostics_and_Laboratory_Medicine | 30 |38.9 |
| 13 | Economics | 30 | 75.6|
| 14 | Electronics | 30 |34.4 |
| 15 | Energy_and_Power | 30 | 47.8|
| 16 | Finance | 30 |68.9 |
| 17 | Geography | 30 | 67.8|
| 18 | History | 30 |71.1 |
| 19 | Literature | 30 | 82.2|
| 20 | Manage | 30 |57.8 |
| 21 | Marketing | 30 | 84.4|
| 22 | Materials | 30 | 44.4|
| 23 | Math | 30 | 65.6|
| 24 | Mechanical_Engineering | 30 | 37.8|
| 25 | Music | 30 | 26.7|
| 26 | Pharmacy | 30 |74.4 |
| 27 | Physics | 30 | 76.7|
| 28 | Psychology | 30 |78.9 |
| 29 | Public_Health | 30 |83.3 |
| 30 | Sociology | 30 | 61.1|
| | **Overall (macro avg)** | **900** | 61.70|

계산식: `Overall = mean(30개 과목 accuracy)`

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 61.70 |
| 차이 (Δ) | −5.70pt |

## 7. 격차 분석

재현 점수 61.70%(±0.17% : seed간 표준편차)는 공식수치인 67.4%보다 5.70pt 낮다. 3회 평가에서 평균 약 89/900문항(9.9%)이 16,384토큰 내 Answer:에 도달하지 못했다. 그중 약 83%에서 반복 루프가 관찰됐고, 공학·음악 과목에 집중됐다. 답을 낸 응답의 정확도는 1,666/2,432=68.5%다. 무응답을 오답 처리하는 엄격 채점을 공식 파서의 random fallback으로 바꾸면 같은 재현 점수가 64.26%가 되어 격차가 3.14pt로 줄어들지만, 공식 평가의 무응답 처리 방식이 공개되지 않아 이것이 격차의 원인 중 하나인지 확신할 수 없다. 생성 토큰 예산 또한 공식 32,768보다 짧지만 150문항 pilot에서 추가 예산으로 결론에 도달한 문항은 1개였다. 같은 sampling 설정의 greedy 실험은 57.33%로 낮고 무응답도 많았으므로 sampling 선택 자체가 점수 하락의 근거는 약하다. 이미지 확대 효과는 유의하지 않았으며(+1.34pt, 95% CI −0.89~+3.58), 이미지 제거 시 점수는 45.44%였다. 남은 차이에는 비공개 공식 프롬프트·답 추출 규칙, 이미지 처리 및 백엔드 차이가 관여했다 분석된다. 


## 8. 기타 특이사항 / 한계 (Optional)

- `max_new_tokens=16384` 내에 최종 답을 내지 못한 응답이 약 9.9%였다. 이 문항들은 엄격 채점 기준에 따라 오답 처리되므로, 생성 길이와 반복 출력이 점수에 영향을 줄 수 있다.
- 3개 seed에서 종합 점수 표준편차는 0.17%p였지만, 문항별로는 약 21%의 정오가 seed에 따라 달라졌다. 향후 fine-tuning 전후 성능도 동일한 3개 seed의 평균으로 비교할 예정이다.
- 주관식은 MMMU 공식 파서의 정답 문자열 포함 판정을 사용하므로 일부 답을 관대하게 채점할 수 있다. LaTeX 정규화도 중첩 수식까지 완전히 처리하지는 못한다.
- 공식 평가의 프롬프트 전문과 미완료 응답 처리 방식은 공개되지 않아 공식 점수와의 차이를 완전히 분리해 설명할 수 없다. 이후 재평가에서는 현재 파이프라인의 프롬프트·생성 설정·채점 규칙을 고정하고 모델 checkpoint만 교체한다.
