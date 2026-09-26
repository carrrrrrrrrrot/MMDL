# Qwen3-VL-4B-Instruct × MMMU-val 평가 파이프라인 (v1.2)

이 문서는 **최종 확정한 평가 파이프라인만** 정리한 것입니다. 처음 상태(lmms-eval 기본 설정)와 무엇이 달라졌는지, 그리고 왜 바꿨는지를 함께 적었습니다.
결정 과정과 모든 실험 기록은 `/home/dekim/workspace/eval_pipeline_log.md`(작업 기록)에 있습니다.

- 기준일: 2026-09-27
- 파이프라인 버전: task `mmmu_val_mmdl`, `metadata.version: 1.2`

  | 버전 | 변경 | 생성 | 채점 |
  |---|---|---|---|
  | v1.0 | 최초 확정 (M4 본 실행) | — | — |
  | v1.1 | 정답이 여러 개인 open 문항의 정답 목록 파싱 수정 (채점 버그) | 같음 | 바뀜 |
  | v1.2 | open 최종 답의 LaTeX 표기 정규화 | 같음 | 바뀜 |

  - 생성 조건은 v1.0 이후 바뀌지 않았습니다. 그래서 v1.0으로 생성한 run은 **다시 생성하지 않고 저장된 로그로 재채점**했습니다.
- fine-tune 이후 재평가에서도 **이 문서의 설정을 그대로 사용**하고, 모델 경로만 바꿉니다.

---

## 1. 처음 상태 vs 최종 파이프라인

처음 상태는 run #4(`results/baseline_v0_lmms_default/20260921_153124_*`)입니다. lmms-eval 기본 MMMU task를 그대로 실행한 것입니다.

### 코드·환경

| 항목 | 처음 상태 | 최종 (v1.2) |
|---|---|---|
| 진입점 | `lmms-eval/examples/models/qwen3vl.sh` (커밋 안 된 수정본으로 실행, 경로 하드코딩) | `MMDL/scripts/run_mmmu_eval.sh` (모델·데이터 경로와 seed를 인자로 받음) |
| task 정의 | lmms-eval 내부의 `mmmu_val_local.yaml` (데이터 경로 하드코딩) | `MMDL/eval_tasks/mmmu_mmdl/` (lmms-eval 밖, `--include_path`로 연결) |
| lmms-eval 코드 | 원본 | **원본 그대로** (0.7.3, 한 줄도 수정하지 않음) |
| 추론 백엔드 | HF transformers `generate()`, 배치 1 | **vLLM 0.11.0** (900문항을 한 번에 넘기고 연속 배치로 처리) |
| attention | flash_attention_2 | vLLM 기본 |
| 실행 환경 | conda `vlm`: torch 2.6.0, transformers 5.17.0 | conda `vllm011`: torch 2.8.0+cu128, transformers 4.57.1, vllm 0.11.0, qwen-vl-utils 0.0.14, Python 3.10.21 |

### 프롬프트

| 항목 | 처음 상태 | 최종 (v1.2) |
|---|---|---|
| 객관식 지시문 | `Answer with the option letter only.` (추론 억제) | `Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.` |
| open 지시문 | 빈 `Options:` 줄 + `Please select the correct answer from the options above.` (**입력과 모순**) | `Options:` 줄 없음 + `Think step by step, then give your final answer on the last line in the form "Answer: <answer>".` |
| 이미지 배치 | 모든 이미지를 텍스트 앞에 붙이고, 본문의 `<image N>`은 글자로 남김 | 본문과 보기의 `<image N>` **자리에 이미지를 삽입** (interleave) |
| 이미지 해상도 | `max_pixels=12845056`을 인자로 줬지만 **적용되지 않음** (실제로는 원본 해상도) | `qwen_vl_utils.fetch_image(patch 16)`: 한 변을 32의 배수로 반올림, 픽셀 수 [4,096, 16,777,216]. 사실상 원본 해상도이며 이를 명시 |
| system prompt | `You are a helpful assistant.` | 같음 |

### 생성·채점·결과

| 항목 | 처음 상태 | 최종 (v1.2) |
|---|---|---|
| sampling | **greedy** (lmms-eval 기본값, 근거 없음) | **Qwen 공식 Instruct recipe**: T 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5 |
| max_new_tokens | 128 (79문항 잘림, open 53개 중 48개 잘림) | **16384** |
| seed | 0 / 1234 (greedy라 사실상 무의미) | 요청별 sampling seed **3407, 1, 2** (3회 실행) |
| 채점 | MMMU 공식 파서에 응답 **전체**를 넘김. 객관식 답 글자가 없으면 **random fallback**. 정답 목록형 문항과 LaTeX 답은 오채점 | 마지막 `Answer:` 뒤 한 줄만 공식 파서에 넘김. **객관식 답 글자가 없으면 오답** (엄격 채점). 정답 목록과 LaTeX 표기를 정규화 |
| 집계 | 문항 가중 평균 (= 30과목 macro 평균) | 같음. 3 seed 평균 ± 표준편차 |
| 점수 | **51.22%** (seed 1회) | **61.70 ± 0.17%** |

---

## 2. 최종 파이프라인 상세

### 2.1 파일 구성 (MMDL repo)

```
MMDL/
├─ scripts/
│   ├─ run_mmmu_eval.sh          # 평가 진입점 (본 평가·진단 공용)
│   ├─ summarize_results.py      # 과목별 표와 요약 지표. 기록 정합성 assert 후 현재 기준(v1.2)으로 재채점
│   ├─ watch_progress.sh         # 실행 중 진행 상황 모니터
│   ├─ parity/                   # HF와 vLLM의 입력 동일성 검증
│   ├─ diagnosis/                # 오답 원인 진단 (검토 자료 생성, 일치도 계산, 작은 이미지 목록)
│   └─ judge/                    # LLM judge 보조 분석 (본 평가에는 쓰지 않음)
├─ eval_tasks/mmmu_mmdl/
│   ├─ mmmu_val_mmdl.yaml.tmpl         # 본 평가 task (평가 기준이 모두 여기에 고정됨)
│   ├─ mmdl_utils.py                   # 프롬프트 구성, 이미지 배치·크기, 최종 답 추출, 엄격 채점, 정답·LaTeX 정규화
│   ├─ mmmu_val_mmdl_diag.yaml.tmpl    # 진단 전용 task (본 평가 task를 include하고 메시지 구성과 문항 필터만 교체)
│   └─ mmmu_val_mmdl_regress.yaml.tmpl # 회귀 테스트용 (lmms-eval 기본 기준 재현)
├─ code/lmms-eval/               # lmms-eval 0.7.3 원본 (수정 없음, editable 설치)
├─ docs/eval_pipeline.md         # 이 문서
└─ results/
    ├─ runs/M4_final/seed_{3407,1,2}/   # 최종 결과 (summary.md = v1.2 결과표)
    ├─ runs/M4_aux_greedy/              # 보조: greedy
    ├─ runs/diag_a_upscale/, diag_d_noimage/   # 진단: 작은 이미지 확대, 이미지 제거
    ├─ runs/M1_*, M2_*, M3_*, M3b_*, v1_1_*    # 회귀·스모크·pilot·채점 수정 확인 기록
    ├─ baseline_v0_lmms_default/        # 처음 상태 (run #2, #4)
    ├─ parity/                          # HF와 vLLM 입력 토큰 비교 결과
    ├─ diagnosis_eye_brain/, diagnosis_upscale/   # 진단 자료
    ├─ judge/                           # LLM judge 호출 결과 캐시
    └─ rescore_v1_1/                    # 채점 수정 확인용 문항 목록
```

### 2.2 실행 방법

```bash
# 1) 평가 (seed 3개를 차례로 실행)
PYTHON=<vllm011 env의 python> bash scripts/run_mmmu_eval.sh \
    --model_path <Qwen3-VL-4B-Instruct 경로 (revision ebb281ec…)> \
    --data_root  <MMMU 데이터 루트 (아래에 <과목>/validation-*.parquet)> \
    --seeds "3407 1 2" \
    --run_name <이름>

# 2) 결과표
python scripts/summarize_results.py results/runs/<이름> --out results/runs/<이름>/summary.md
```

- 스크립트 기본 backend는 `vllm`(본 평가 백엔드)입니다. `--backend hf`는 HF transformers 경로(env `vlm`, presence_penalty 미적용)로, 비교·검증용입니다.
- 다음은 pilot·진단 전용이므로 본 평가에는 쓰지 않습니다.
  - `--subjects`, `--limit`, `--gen_override`
  - `--task mmmu_val_mmdl_diag`와 `MMDL_DIAG_*` 환경 변수
- 스크립트는 task 폴더의 템플릿(`*.yaml.tmpl`)을 모두 run 폴더에 렌더링합니다. 진단 task가 본 평가 task를 include하기 때문입니다.
- 각 run 폴더에 남는 것
  - 실제로 쓴 task 파일, `mmdl_utils.py`, 스크립트의 사본
  - `run_meta.txt`: 명령, 커밋, GPU, peak VRAM, 진단용 환경 변수
  - `pip_freeze.txt`
- 결과표는 항상 **현재 채점 기준(v1.2)**으로 계산됩니다.
  - `summarize_results.py`는 먼저 run 당시 lmms-eval이 기록한 점수를 그대로 재현하는지 assert로 확인합니다.
  - 그다음 v1.2 기준으로 재채점하고, 두 값을 함께 출력합니다.

### 2.3 데이터·모델 고정

- **모델:** `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (로컬 snapshot 경로로 pin), bf16
- **데이터:** `MMMU/MMMU`, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, 30과목 validation parquet, 과목당 30문항, 총 900문항
  - 30과목 parquet를 과목 폴더 이름순으로 한 번에 읽습니다. 결과는 과목별 config로 읽은 것과 같은 900문항입니다.

### 2.4 모델에 들어가는 실제 입력 (토큰을 디코딩한 것, `<|image_pad|>`는 개수로 줄여 표시)

객관식 (`validation_Biology_24`, 보기 자체가 이미지)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: Which image of the human heart muscle was produced by fluorescent microscopy?
Options:
A. <|vision_start|><|image_pad|>×138<|vision_end|>
B. <|vision_start|><|image_pad|>×138<|vision_end|>
C. <|vision_start|><|image_pad|>×138<|vision_end|>
Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.<|im_end|>
<|im_start|>assistant
```

open (`validation_Architecture_and_Engineering_14`)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question: Using a finite summation, compute the  initial deflection at midspan for the beam in  Figure P8.42. Given: E = 3000 kips/in.2 .  Use 3-ft segments. Assume I = 0.5IG. <|vision_start|><|image_pad|>×168<|vision_end|>
Think step by step, then give your final answer on the last line in the form "Answer: <answer>".<|im_end|>
<|im_start|>assistant
```

- 템플릿: `{pre_prompt}{question}\nOptions:\n{A. …\nB. …}\n{지시문}` (open 문항은 `Options:` 부분이 없음)
- **이미지 배치**
  - 같은 `<image N>`이 여러 번 나오면 첫 위치에만 이미지를 넣고, 이후는 글자로 둡니다 (900문항 중 1개).
  - 본문에서 참조하지 않는 이미지는 넣지 않습니다 (4개 문항, MMMU 공식 기준과 같음).
- **출처:** 틀(`Question:`, `Options:`)은 lmms-eval의 Qwen3-VL용 MMMU 프롬프트(docstring: Qwen3-VL Technical Report)에서 가져왔습니다. 마지막 지시문은 직접 설계했습니다.

### 2.5 생성 설정

| 파라미터 | 값 | 근거 |
|---|---|---|
| temperature / top_p / top_k | 0.7 / 0.8 / 20 | Qwen3-VL GitHub README "Generation Hyperparameters" (Instruct). 같은 조건의 greedy보다 +4.4pt, greedy는 무응답이 약 2배 (3.2절) |
| repetition_penalty / presence_penalty | 1.0 / 1.5 | 위와 같음. presence_penalty를 적용하려고 vLLM을 사용 (HF `generate`는 미지원) |
| seed | 3407, 1, 2 (요청별) | 3407은 공식 값. 나머지는 편차 측정용 |
| max_new_tokens | 16384 | 공식 `out_seq_length`는 32768. 150문항 pilot에서 16384를 넘겨 결론에 도달한 응답이 1개뿐이었고(잘림 12 vs 11), 시간은 약 1/3 |
| vLLM | `gpu_memory_utilization=0.85`, `max_model_len=40960`, `limit_mm_per_prompt={"image":7}`, `mm_processor_kwargs={"min_pixels":4096,"max_pixels":16777216}`, 요청 묶음 1024 | 가장 긴 입력(5,649 토큰) + 32,768을 담을 수 있게. processor가 작은 이미지를 다시 확대하지 않도록 최소 픽셀을 HF 경로와 맞춤 |

- `results.json`에 기록된 실제 설정 (seed 3407): `{max_new_tokens: 16384, temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.0, presence_penalty: 1.5, seed: 3407}`
  - CLI로 넘긴 seed는 yaml 설정에 **병합**되고 yaml 값을 덮어쓰지 않습니다. 코드와 실제 SamplingParams로 확인했습니다.
- 이미지 해상도: 작은 이미지를 확대해도 점수가 오르지 않았습니다 (3.2절). 그래서 원본 해상도를 유지합니다.

### 2.6 채점

1. **최종 답 추출:** 응답에서 마지막 `answer:`(대소문자 무시) 뒤의 첫 번째 비어 있지 않은 줄을 꺼냅니다. `\boxed{}`, `*`, `` ` ``는 제거합니다.
2. **open 문항 정규화**
   - 최종 답의 LaTeX 표기를 일반 텍스트로 바꿉니다 (`normalize_latex`, v1.2): `$` 제거, `\frac`·`\dfrac`·`\tfrac{a}{b}` → `a/b`, `\text{…}` → 내용, `\left`·`\right` 제거
     - 지시문이 답의 표기 형식을 정하지 않았으므로, 수학적으로 같은 답이 표기 때문에 틀리지 않게 하기 위해서입니다.
   - 정답이 여러 개인 문항은 문자열로 저장된 정답(`"['24/7', '3.429']"`)을 **목록으로 바꿉니다** (`normalize_gold`, v1.1).
     - lmms-eval 포트는 이 문자열을 그대로 비교해서 맞는 답도 오답 처리합니다. 해당 문항은 Math_15, Geography_4, Chemistry_30입니다.
     - 목록은 "둘 중 하나면 정답"이라는 뜻입니다.
   - 결과 파일의 `final_answer`에는 정규화 전 원문을 남깁니다.
3. **공식 판정:** 위의 한 줄을 lmms-eval에 포팅된 **MMMU 공식 `mmmu_process_results`**(eval_utils, commit `51ce7f3e`)에 넘깁니다.
   - open 판정은 공식 `eval_open`을 그대로 씁니다. 정답 문자열이 예측에 **포함되면** 정답입니다. 예를 들어 "Saint Petersburg, Florida"는 정답 목록 `['Tampa', 'Florida']`의 'florida'와 맞아 정답이 됩니다 (공식 방식의 관대함).
4. **엄격 규칙**
   - 객관식에서 공식 파서가 답 글자 후보를 찾지 못하면 오답입니다. `Answer:`가 없는 경우와, 있어도 글자가 없는 경우가 모두 해당합니다.
   - 공식 파서 내부의 random 선택은 `random_fallback_pred`에 **기록만** 하고 점수에는 반영하지 않습니다.
   - open 문항은 추출에 실패하면 오답입니다.
5. **집계:** 공식 `mmmu_aggregate_results` → 문항 가중 평균입니다. 과목당 30문항으로 같으므로 30과목 macro 평균과 같습니다.

**공식 채점 코드 앞에 전처리를 둔 이유**
- 공식 파서는 짧은 답을 전제로 만들어졌습니다. 추론문 전체를 넘기면 두 가지 왜곡이 생깁니다.
  - open 문항: 본문의 모든 숫자가 정답 후보가 되어 과대 채점됩니다.
  - 객관식: 본문 중간의 `(B)` 같은 표기가 최종 답보다 우선됩니다.
- random fallback을 쓰지 않는 이유
  - 답하지 않은 문항에 찍기 점수(평균 26%)를 주지 않기 위해서입니다.
  - fine-tune 전후의 잘림률 차이가 점수에 섞이지 않게 하기 위해서입니다.
- 정답 목록과 LaTeX 정규화: **맞는 답을 틀리다고 판정하는 경우만** 바로잡습니다. 저장된 모든 run에서 새로 관대해진 판정이 없음을 확인했습니다 (4절).

---

## 3. 최종 결과

### 3.1 본 평가 (`results/runs/M4_final`, `summary.md`)

| 지표 | seed 3407 | seed 1 | seed 2 | 평균 ± 표준편차 |
|---|---|---|---|---|
| **엄격 점수 (주 점수)** | 61.67 | 61.89 | 61.56 | **61.70 ± 0.17** |
| 공식 관례 점수 (random fallback 적용, 보조) | 64.22 | 64.00 | 64.56 | 64.26 |
| `Answer:` 없음 (/900) | 90 | 93 | 85 | 약 9.9% |
| 생성 시간 (RTX 4090 24GB 1장) | 2시간 5분 | 2시간 6분 | 2시간 2분 | |
| peak VRAM (nvidia-smi, GPU 전체) | 23,013 MiB | 23,085 MiB | 22,889 MiB | |

- **공식 수치와의 비교:** 67.4 대비 −5.70pt (공식 관례 점수 기준으로는 −3.14pt)
- **답을 낸 응답만의 정확도:** 68.5% (1,666 / 2,432)
- 점수는 v1.2 채점 기준입니다. run 당시 lmms-eval이 기록한 v1.0 점수는 61.44 ± 0.22였고, 차이는 채점 수정분뿐입니다 (v1.1: seed당 +2, v1.2: seed 2 +1).
- 과목별 표: `results/runs/M4_final/summary.md`

### 3.2 보조 실험과 진단 (같은 task, 조건 하나만 바꿈)

| 실험 | 바꾼 조건 | 결과 | 의미 |
|---|---|---|---|
| greedy (`M4_aux_greedy`) | temperature 0 | 57.33% (무응답 168) | 공식 sampling이 +4.4pt. greedy는 긴 생성에서 반복 루프가 약 2배 |
| 작은 이미지 확대 (`diag_a_upscale`, 522문항 × 3 seed) | 최소 픽셀 262,144 | +1.34pt, 95% CI −0.89~+3.58 | 해상도는 병목이 아님 |
| 이미지 제거 (`diag_d_noimage`) | 이미지 없이 텍스트만 | 45.44% | 이미지 기여 약 +16pt. Math는 기여가 거의 없음(−1.1), Physics +30, Chemistry +10 |
| 오답 원인 분류 (`diagnosis_eye_brain`, 21문항) | 수작업 (Claude 판정, 사용자 교차 검증 2/21 일치) | 눈 12 / 뇌 8 / 채점 1 | Math는 도식 구조 오독, Physics는 규칙 적용 오류, Chemistry는 둘 다 |
| LLM judge (`results/judge/`, `gpt-4o-mini-2024-07-18`) | 규칙 실패 객관식만 judge로 매핑 (v1.1 기준 참고치) | +0.63pt | 매핑된 답의 정답률 34%로 추측에 가까움 → 본 평가에 쓰지 않음 |

---

## 4. 검증한 것

| 검증 | 결과 |
|---|---|
| 새 task 구조가 엔진에 제대로 연결되는가 (예전 기준 재현) | run #2와 응답이 **900/900 동일**, 점수 동일 (0.51111) |
| 이미지를 미리 맞춰도 HF 경로 결과가 그대로인가 | 같은 문항·같은 seed에서 응답이 글자 단위로 동일 |
| HF와 vLLM의 입력이 같은가 | 입력 토큰 id가 **900/900 완전히 일치** (이미지 토큰 수 포함) |
| CLI seed가 yaml 설정을 덮어쓰는가 | 병합됨 (results.json, 코드 경로, SamplingParams 재구성, 실제 생성 길이로 확인) |
| 결과 run과 현재 코드의 관계 | M4는 v1.0 파일로 생성되었습니다 (run 폴더 사본 == 당시 파일). 이후 진단 함수를 추가하고 채점을 두 번 수정한 뒤에도 **본 평가의 메시지 구성(텍스트와 이미지 픽셀)이 900/900 동일**함을 확인했습니다. 채점 차이는 재채점으로 반영했습니다 |
| 점수 산술 | 기록된 정답 그대로 다시 판정한 합계 == lmms-eval `mmmu_acc` (`summarize_results.py`의 assert) |
| 엄격 채점 판정 == 공식 파서의 random 발동 여부 | 경계 사례 8개 일치. 본 결과에서 random이 점수에 반영된 문항 0건 |
| 채점 수정 v1.1의 영향 범위 | 수정한 함수로 M4 응답 2,700개를 다시 채점: `parsed_pred`는 900/900 그대로이고 바뀐 판정은 해당 3문항뿐 (seed당 +2). lmms-eval 실행(3문항)에서도 정답이 목록으로 전달됨 |
| 채점 수정 v1.2의 영향 범위 | 저장된 모든 run(M4, greedy, pilot, 진단)의 open 응답을 v1.1과 v1.2로 비교: 바뀐 판정 4건이 전부 Math_15의 `$\frac{24}{7}$`가 정답이 된 경우. 새로 관대해진 오판정 0건 |

---

## 5. 알려진 한계

- **결론 미도달:** 약 10%의 응답이 16384 토큰 안에 결론을 내지 못합니다 (약 83%가 반복 루프, 공학·음악 과목에 집중). presence_penalty는 긴 주기의 반복을 막지 못합니다.
- **문항 단위 잡음:** 종합 점수는 안정적(±0.17)이지만, 문항의 약 21%는 seed에 따라 정오가 바뀝니다. 과목과 문항 단위 비교는 3 seed를 합쳐서 봅니다 (문항별 정답 횟수 k/3).
- **예산:** 공식 32768 대신 16384를 씁니다 (pilot 기준 영향은 150문항 중 1문항).
- **환경 차이:** HF 경로(transformers 5.17)와 vLLM 경로(4.57.1)는 입력 토큰은 같지만, 이미지 pixel 값이 완전히 같은지는 확인하지 않았습니다.
- **채점**
  - 공식 `eval_open`의 포함 비교는 관대합니다 (예: 'florida').
  - LaTeX 정규화는 중괄호가 중첩된 식(`\frac{5\sqrt{3}}{4\pi}`)을 바꾸지 않습니다.
  - submission 파일은 open 문항의 후보 순서가 실행마다 달라질 수 있습니다 (정확도에는 영향 없음).
- **비공개 정보:** 공식 평가의 프롬프트 수정 내용, 답 추출 방식, 끝나지 않는 응답의 처리 방식은 공개되지 않았습니다.
- **진단의 표본:** 오답 원인 분류는 21문항이고 교차 검증은 2문항만 했습니다. 이미지 제거 실험은 seed 1개입니다.

## 6. 환경·재현

- 의존성: `requirements.txt` (env `vllm011`의 pip freeze) + `pip install --no-deps -e code/lmms-eval`
- 재현 커맨드와 모델·데이터 다운로드 방법: repo 루트 `README.md`
