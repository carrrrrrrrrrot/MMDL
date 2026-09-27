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
| 총 소요 시간 | 900문항당 seed별 2시간 5분 / 2시간 6분 / 2시간 2분 ; 3회 합계 약 6시간 13분 |
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
- **선택 이유**: 기본 프롬프트는 주관식 문항에서 존재하지 않는 선택지를 고르라는 지시가 있었, 이미지를 텍스트 앞에 모아 배치해 질문·보기에서 이미지가 참조되는 위치를 반영하지 못했다. 이를 수정해 이미지를 <image N>이 등장하는 자리에 삽입하고, 추론 후 마지막 줄에 Answer: 형식으로 최종 답을 쓰도록 했다. 추가로 HF와 vLLM 경로의 입력 토큰 ID가 900문항 모두에서 일치함을 확인했다.

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

**선택 근거** RTX 4090 24GB에서 생성 시간과 답변 완결성 사이의 trade-off를 고려해 max_new_tokens=16384로 설정했다. 150문항 pilot에서 공식 설정인 32,768토큰까지 늘렸을 때 추가로 최종 답에 도달한 문항은 1개였지만, 생성 시간은 약 3배 걸렸다. 다만 본 평가에서 약 9.9%가 토큰 제한 내에 Answer:를 출력하지 못했으므로 생성 길이의 영향을 완전히 배제할 수는 없다. 이미지는 fetch_image(image_patch_size=16)으로 크기를 32의 배수에 맞춰 처리했다. 작은 이미지 확대 실험에서는 정확도 향상이 뚜렷하지 않아(+1.34%p, 95% 신뢰구간 −0.89~+3.58%p) 대부분 원본에 가까운 해상도를 유지했다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: 자체 구현한 [`mmdl_process_results`](https://github.com/carrrrrrrrrrot/MMDL/blob/main/eval_tasks/mmmu_mmdl/mmdl_utils.py)로 응답을 전처리한 뒤, 수정하지 않은 lmms-eval의 MMMU 공식 파서 포트인 [`mmmu_process_results`와 `mmmu_aggregate_results`](https://github.com/carrrrrrrrrrot/MMDL/blob/main/code/lmms-eval/lmms_eval/tasks/mmmu/utils.py)를 사용한다.
- 동작 방식 요약: 응답의 마지막 answer:(대소문자 무시) 뒤 첫 비어 있지 않은 줄만 뽑고 \boxed{}, *, 백틱을 제거한다. open 답은 $, 단순 \frac{a}{b}, \text{...} 등의 LaTeX 표기를 일반 텍스트로 정규화하며, 문자열로 저장된 복수 정답은 목록으로 복원한다. 이어 공식 파서로 객관식 선택지 또는 open 정답을 판정한다. 객관식에서 답 글자 후보가 없거나 Answer:가 없으면 오답 처리하며 공식 파서의 random fallback 결과는 참고용으로만 기록한다. open 문항의 추출 실패도 오답이다. 공식 open 판정의 정답 문자열 포함 비교는 그대로 사용한다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | |
| 2 | Agriculture | 30 | |
| 3 | Architecture_and_Engineering | 30 | |
| 4 | Art | 30 | |
| 5 | Art_Theory | 30 | |
| 6 | Basic_Medical_Science | 30 | |
| 7 | Biology | 30 | |
| 8 | Chemistry | 30 | |
| 9 | Clinical_Medicine | 30 | |
| 10 | Computer_Science | 30 | |
| 11 | Design | 30 | |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | |
| 13 | Economics | 30 | |
| 14 | Electronics | 30 | |
| 15 | Energy_and_Power | 30 | |
| 16 | Finance | 30 | |
| 17 | Geography | 30 | |
| 18 | History | 30 | |
| 19 | Literature | 30 | |
| 20 | Manage | 30 | |
| 21 | Marketing | 30 | |
| 22 | Materials | 30 | |
| 23 | Math | 30 | |
| 24 | Mechanical_Engineering | 30 | |
| 25 | Music | 30 | |
| 26 | Pharmacy | 30 | |
| 27 | Physics | 30 | |
| 28 | Psychology | 30 | |
| 29 | Public_Health | 30 | |
| 30 | Sociology | 30 | |
| | **Overall (macro avg)** | **900** | |

계산식: `Overall = mean(30개 과목 accuracy)` _(다른 방식을 썼다면 명시)_

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | |
| 차이 (Δ) | |

## 7. 격차 분석

_(1000 char 이내로 작성 - Official 성능과 차이가 발생하는지, 그렇다면 그 이유를 서술. 길게 쓴다고 credit이 느는 게
아니라, 근거의 질이 핵심입니다. 레포트는 짧을수록 좋습니다.)_


## 8. 기타 특이사항 / 한계 (Optional)

_(재현 중 겪은 문제, 시간 관계상 못 해본 것, 다음에 시도해보고 싶은 것 등. 자유롭게)_
