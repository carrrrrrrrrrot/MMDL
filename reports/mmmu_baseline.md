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
| 실측 peak VRAM | 22.54 GiB |
| 총 소요 시간 | 900문항당 seed별 2시간 5분 / 2시간 6분 / 2시간 2분 -> 3회 총 약 6시간 13분 |
| 의존성 | [requirements.txt](https://github.com/carrrrrrrrrrot/MMDL/blob/main/requirements.txt)  |
| 실행 커맨드 | `PYTHON="$(command -v python)" bash scripts/run_mmmu_eval.sh --model_path "/path/to/Qwen3-VL-4B-Instruct" --data_root "/path/to/MMMU" --seeds "3407 1 2" --run_name baseline_reproduction` |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
• 객관식 프롬프트 설계:
  Question: {문제}\nOptions:\n{보기 A, B, C, D}\nAnswer with the option letter only.
  • 주관식 프롬프트 설계:
  Question: {문제}\nOptions:\n{보기}\nPlease select the correct answer from the options above.
```

- **출처**: _(직접 설계 / 차용한 도구·저장소명 + 링크)_
- **선택 이유**: _(왜 이 프롬프트를 골랐는지)_

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
| `do_sample` | |
| `temperature` | |
| `top_p` | |
| `top_k` | |
| `repetition_penalty` | |
| `presence_penalty` | |
| `seed` | |

- **출처**: _(모델 제공사의 공식 recipe를 찾았다면 그 출처/링크. 못 찾았거나 다른 값(예: greedy)을 쓰기로
  했다면 그 사실과 이유)_

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | |

**선택 근거** (본인이 사용한 인프라 제약과 어떻게 연결되는지 — 속도/VRAM/응답 잘림 등 trade-off): _(적절히)_

## 4. 채점(파싱) 방식

- 사용한 파서/로직: _(자체 구현 / 차용 도구명 + 링크)_
- 동작 방식 요약: _(예: 어떤 순서로 규칙을 적용하는지, 실패 시 fallback은 무엇인지)_

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
