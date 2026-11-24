"""MMMU validation 평가용 프롬프트 함수 (task: mmmu_val_mmdl).

채점과 집계는 lmms-eval 에 포팅된 MMMU 공식 코드를 그대로 쓰고
(yaml 에서 lmms_eval.tasks.mmmu.utils.* 를 직접 지정), 이 파일은 프롬프트 구성만 담당합니다.

lmms-eval 의 mmmu_doc_to_text_qwen3vl / mmmu_doc_to_messages_qwen3vl
(lmms_eval/tasks/mmmu/utils.py) 에서 바꾼 점:
  1. open 문항에서는 비어 있는 "Options:" 줄을 넣지 않습니다.
  2. 문항 유형별 지시문을 yaml 의 lmms_eval_specific_kwargs 로 받습니다
     (multiple_choice_prompt / open_ended_prompt).
  3. 이미지를 모두 앞에 붙이지 않고, 본문과 보기의 <image N> 자리에 끼워 넣습니다 (interleave).
     - 같은 <image N> 이 여러 번 나오면 첫 위치에만 이미지를 넣고, 이후 위치는 글자로 둡니다.
     - 본문에서 참조하지 않는 이미지는 넣지 않습니다 (lmms-eval / MMMU 공식 mmmu_doc_to_visual 과 같은 기준).
  4. 이미지 크기를 여기서 미리 맞춥니다: qwen_vl_utils.fetch_image(image_patch_size=16) 로
     한 변을 32의 배수로 반올림하고 픽셀 수를 [4,096, 16,777,216] 로 제한합니다.
     HF 백엔드(qwen3_vl)가 내부에서 하는 처리와 같은 함수이고, 이미 맞춘 이미지는 다시 바뀌지 않으므로
     HF 경로에는 영향이 없습니다. vLLM 백엔드도 같은 크기의 이미지를 받게 하기 위한 처리입니다.
  5. lmms_eval_specific_kwargs 에 system_prompt 가 있으면 system 메시지를 앞에 넣습니다.
     HF 백엔드는 "You are a helpful assistant." 를 스스로 넣으므로 비워 두고,
     vLLM 백엔드(system prompt 를 넣지 않음)에서만 같은 문장을 지정합니다 (yaml 참고).

채점 전처리 (mmdl_process_results):
  MMMU 공식 파서는 짧은 응답을 전제로 해서, 추론문 전체를 넘기면 open 문항은 본문의 모든 숫자가
  후보가 되어 과대 채점되고, 객관식은 본문의 "(B)" 표기가 최종 답보다 우선됩니다.
  그래서 응답의 마지막 "Answer:" 뒤 한 줄만 공식 mmmu_process_results 에 넘깁니다.
  "Answer:" 가 없으면(주로 max_new_tokens 에서 잘린 응답) 빈 응답으로 넘깁니다 → open 은 오답.
  객관식에서 답 글자를 찾지 못하면(= "Answer:" 가 없거나, 있어도 공식 파서가 후보를 못 찾는 경우)
  공식 파서는 random fallback 으로 임의의 보기를 고르지만, 여기서는 오답으로 처리합니다(엄격 점수).
  공식 관례 점수도 계산할 수 있도록 그 random 선택은 mmmu_acc.random_fallback_pred 에 남깁니다.
  정답이 여러 개인 open 문항은 문자열로 저장된 정답 목록을 목록으로 바꿔서 넘깁니다 (normalize_gold, v1.1).
  open 문항의 최종 답은 LaTeX 표기를 일반 텍스트로 바꿔서 넘깁니다 (normalize_latex, v1.2).
"""

import ast
import os
import re

from qwen_vl_utils import fetch_image

from lmms_eval.tasks._task_utils.mmmu_mcq_utils import get_multi_choice_info
from lmms_eval.tasks.mmmu.utils import mmmu_process_results, parse_options

IMAGE_TOKEN = re.compile(r"<image (\d+)>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
LATEX_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
LATEX_TEXT = re.compile(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}")


def mmdl_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")

    if doc["question_type"] == "multiple-choice":
        options = parse_options(ast.literal_eval(doc["options"]))
        return f"{pre_prompt}{doc['question']}\nOptions:\n{options}\n{kwargs.get('multiple_choice_prompt', '')}"
    return f"{pre_prompt}{doc['question']}\n{kwargs.get('open_ended_prompt', '')}"


def mmdl_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    text = mmdl_doc_to_text(doc, lmms_eval_specific_kwargs)

    content = []
    inserted = set()
    pos = 0
    for match in IMAGE_TOKEN.finditer(text):
        n = int(match.group(1))
        if n in inserted or doc.get(f"image_{n}") is None:
            continue
        if match.start() > pos:
            content.append({"type": "text", "text": text[pos : match.start()]})
        image = fetch_image({"image": doc[f"image_{n}"].convert("RGB")}, image_patch_size=16)
        content.append({"type": "image", "url": image})
        inserted.add(n)
        pos = match.end()
    if pos < len(text):
        content.append({"type": "text", "text": text[pos:]})

    messages = [{"role": "user", "content": content}]
    system_prompt = (lmms_eval_specific_kwargs or {}).get("system_prompt")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    return messages


def extract_final_answer(response):
    """응답에서 마지막 "Answer:" 뒤의 첫 번째 비어 있지 않은 줄을 반환합니다. 없으면 None."""
    idx = response.lower().rfind("answer:")
    if idx < 0:
        return None
    for line in response[idx + len("answer:") :].splitlines():
        line = BOXED.sub(r"\1", line).replace("*", "").replace("`", "").strip()
        if line:
            return line
    return None


def mc_has_candidate(response, doc):
    """공식 parse_mmmu_multi_choice_response 가 random fallback 없이 후보를 찾는지 여부.

    lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py 의 후보 탐색 조건을 그대로 따릅니다.
    """
    index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    if any(f"({c})" in response or f"{c} " in response or f"{c}." in response for c in all_choices):
        return True
    return len(response.split()) > 5 and any(ans.lower() in response.lower() for ans in index2ans.values())


def normalize_gold(doc):
    """정답이 여러 개인 open 문항의 answer 를 목록으로 바꿉니다.

    MMMU 데이터셋은 이런 정답을 "['24/7', '3.429']" 같은 **문자열**로 저장합니다.
    lmms-eval 의 mmmu_process_results 는 이를 그대로 eval_open 에 넘기므로 문자열 전체가 하나의 정답이 되어
    맞는 답도 오답 처리됩니다 (900문항 중 Math_15, Geography_4, Chemistry_30).
    목록으로 바꾸면 eval_open 이 원래 설계대로 목록의 각 정답과 비교합니다.
    """
    answer = doc["answer"]
    if doc["question_type"] != "multiple-choice" and isinstance(answer, str) and answer.strip().startswith("["):
        try:
            parsed = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            return answer
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
    return answer


def normalize_latex(text):
    """open 문항 최종 답의 LaTeX 표기를 일반 텍스트로 바꿉니다 (v1.2).

    공식 parse_open_response 는 LaTeX 를 해석하지 못해 "$\\frac{24}{7}$" 를 24, 7 로 쪼개 읽고 오답 처리합니다.
    지시문은 답의 표기 형식을 정하지 않으므로, 수학적으로 같은 답이 표기 때문에 틀리지 않도록 바꿉니다.
      - \\text{…}, \\mathrm{…}, \\textbf{…} → 내용
      - \\frac, \\dfrac, \\tfrac {a}{b} → a/b (중첩되면 안쪽부터 반복)
      - $ 제거, \\left / \\right 제거, \\, \\; \\! 간격 기호 정리
    """
    prev = None
    while prev != text:
        prev = text
        text = LATEX_TEXT.sub(r"\1", text)
        text = LATEX_FRAC.sub(r"\1/\2", text)
    for token in ("\\left", "\\right", "\\!", "$"):
        text = text.replace(token, "")
    for token in ("\\,", "\\;"):
        text = text.replace(token, " ")
    return text.strip()


def mmdl_process_results(doc, results):
    final_answers = [extract_final_answer(pred) for pred in results]
    gold = normalize_gold(doc)
    if gold is not doc["answer"]:
        doc = {**doc, "answer": gold}
    scored = [a if a is not None else "" for a in final_answers]
    if doc["question_type"] != "multiple-choice":
        scored = [normalize_latex(a) for a in scored]
    out = mmmu_process_results(doc, scored)
    acc = out["mmmu_acc"]
    acc["final_answer"] = final_answers[0]
    acc["random_fallback_pred"] = None
    if doc["question_type"] == "multiple-choice" and not mc_has_candidate(final_answers[0] or "", doc):
        acc["random_fallback_pred"] = acc["parsed_pred"][0]
        acc["parsed_pred"] = [""]
    return out


# ---------------------------------------------------------------------------
# 진단 실험 전용 (task: mmmu_val_mmdl_diag). 본 평가 task(mmmu_val_mmdl)는 이 함수들을 쓰지 않습니다.
# 설정은 환경 변수로 받습니다 (scripts/run_mmmu_eval.sh 가 run_meta.txt 에 기록).
#   MMDL_DIAG_IDS_FILE    한 줄에 문항 id 하나. 지정하면 그 문항만 평가
#   MMDL_DIAG_MIN_PIXELS  이미지 최소 픽셀 수 (기본 4,096 = 본 평가와 같음). 작은 이미지를 이 크기까지 확대
#   MMDL_DIAG_DROP_IMAGES 1 이면 이미지를 넣지 않음 (본문의 <image N> 은 글자로 남김)
# ---------------------------------------------------------------------------


def mmdl_diag_process_docs(dataset):
    ids_file = os.environ.get("MMDL_DIAG_IDS_FILE")
    if not ids_file:
        return dataset
    with open(ids_file) as f:
        keep = {line.strip() for line in f if line.strip()}
    return dataset.filter(lambda doc: doc["id"] in keep)


def mmdl_diag_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    text = mmdl_doc_to_text(doc, lmms_eval_specific_kwargs)
    min_pixels = int(os.environ.get("MMDL_DIAG_MIN_PIXELS", "4096"))
    drop_images = os.environ.get("MMDL_DIAG_DROP_IMAGES") == "1"

    content = []
    inserted = set()
    pos = 0
    if not drop_images:
        for match in IMAGE_TOKEN.finditer(text):
            n = int(match.group(1))
            if n in inserted or doc.get(f"image_{n}") is None:
                continue
            if match.start() > pos:
                content.append({"type": "text", "text": text[pos : match.start()]})
            image = fetch_image({"image": doc[f"image_{n}"].convert("RGB"), "min_pixels": min_pixels}, image_patch_size=16)
            content.append({"type": "image", "url": image})
            inserted.add(n)
            pos = match.end()
    if pos < len(text):
        content.append({"type": "text", "text": text[pos:]})

    messages = [{"role": "user", "content": content}]
    system_prompt = (lmms_eval_specific_kwargs or {}).get("system_prompt")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    return messages
