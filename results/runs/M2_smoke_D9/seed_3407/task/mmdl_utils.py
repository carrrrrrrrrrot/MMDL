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

채점 전처리 (mmdl_process_results):
  MMMU 공식 파서는 짧은 응답을 전제로 해서, 추론문 전체를 넘기면 open 문항은 본문의 모든 숫자가
  후보가 되어 과대 채점되고, 객관식은 본문의 "(B)" 표기가 최종 답보다 우선됩니다.
  그래서 응답의 마지막 "Answer:" 뒤 한 줄만 공식 mmmu_process_results 에 넘깁니다.
  "Answer:" 가 없으면(주로 max_new_tokens 에서 잘린 응답) 빈 응답으로 넘깁니다
  → 객관식은 공식 파서의 random fallback, open 은 오답.
"""

import ast
import re

from lmms_eval.tasks.mmmu.utils import mmmu_process_results, parse_options

IMAGE_TOKEN = re.compile(r"<image (\d+)>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


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
        content.append({"type": "image", "url": doc[f"image_{n}"].convert("RGB")})
        inserted.add(n)
        pos = match.end()
    if pos < len(text):
        content.append({"type": "text", "text": text[pos:]})

    return [{"role": "user", "content": content}]


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


def mmdl_process_results(doc, results):
    final_answers = [extract_final_answer(pred) for pred in results]
    out = mmmu_process_results(doc, [a if a is not None else "" for a in final_answers])
    out["mmmu_acc"]["final_answer"] = final_answers[0]
    return out
