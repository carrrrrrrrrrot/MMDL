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
"""

import ast
import re

from lmms_eval.tasks.mmmu.utils import parse_options

IMAGE_TOKEN = re.compile(r"<image (\d+)>")


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
