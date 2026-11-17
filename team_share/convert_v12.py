#!/usr/bin/env python
"""MAmmoTH-VL 레코드(filter_mammoth.py 출력)를 v1.2 평가와 같은 학습 메시지 형식으로 바꾼다.

평가 쪽 기준: eval_tasks/mmmu_mmdl/mmdl_utils.py (mmdl_doc_to_text / mmdl_doc_to_messages) + mmmu_val_mmdl.yaml.tmpl
  system  : "You are a helpful assistant."   (vLLM 백엔드에서 task 가 넣는 문장)
  user    : "Question: " + 질문 [+ "\\nOptions:\\nA. ...\\nB. ..."] + "\\n" + 지시문
            이미지는 본문의 <image N> 자리에 끼워 넣음 (평가와 같은 interleave)
  지시문  : 객관식 MC_PROMPT / 그 외 OPEN_PROMPT (평가 yaml 과 글자 하나까지 같음)
  assistant: 추론 + 마지막 줄 "Answer: X" (객관식은 보기 글자)

확정 정책 (2026-11-17 사용자 결정): 학습에는 status=ok 만 쓴다.
  short_no_reasoning 은 제외, 다중 턴은 첫 human+gpt 쌍만 쓰고 나머지 턴은 버린다.

변환 규칙 (원본의 마지막 줄에 Answer 가 없을 때 등):
  1. 첫 질문-답 쌍만 쓴다 (평가가 단일 턴이므로). n_turns > 1 이면 flags 에 multi_turn 표시.
  2. 질문의 "<image>" → "<image 1>". 이미지가 하나인 레코드만 지원 (single_image 데이터).
  3. 보기 탐지: "Choices:/Options:" 다음 줄들이 "A: …", "A. …", "(A) …" 형태면 객관식으로 보고
     평가 형식 "Options:\\nA. …" 로 다시 쓴다. 원본의 보기 블록은 질문에서 뺀다.
  4. 원본 질문 안의 답 형식 지시문(예: "Answer concisely with one word or phrase.",
     "First perform reasoning, then finally select ... Answer: xxx.")은 v1.2 지시문과 충돌하므로 지운다.
     지운 문장은 flags 의 removed_instructions 에 남긴다.
  5. 최종 답:
     a. 마지막 줄이 "Answer: X" / "Answer:X" / "**Answer:** X" → "Answer: X" 로 정규화
     b. 아니면 본문 끝에서 "the answer is X", "correct answer is X", "\\boxed{X}", "X is the correct answer"
        를 찾아 "Answer: X" 를 마지막 줄로 덧붙임 (answer_source=extracted)
     c. 답 전체가 한 줄·짧은 답(단어 8개 이하)이면 "Answer: <답>" 만 붙일 수 있으나 추론이 없으므로
        status=short_no_reasoning 로 표시 (학습에 넣을지는 결정 사항, c_format_example.md 참고)
     d. 그 외(긴 설명문인데 최종 답을 못 찾음: 캡션·서술형) → status=no_final_answer, 학습 제외 권장
  6. 객관식에서 최종 답이 보기 글자가 아니고 보기 본문과 정확히 같으면 글자로 바꾼다. 못 바꾸면 status=mc_answer_unmapped.

status: ok | short_no_reasoning | no_final_answer | mc_answer_unmapped | unsupported
"""
import json
import re
import sys

SYSTEM_PROMPT = "You are a helpful assistant."
PRE_PROMPT = "Question: "
MC_PROMPT = 'Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.'
OPEN_PROMPT = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'

IMAGE_TOKEN = re.compile(r"<image (\d+)>")
CHOICE_HEADER = re.compile(r"(?im)^\s*(choices|options|选项)\s*[:：]\s*$")
CHOICE_LINE = re.compile(r"^\s*\(?([A-Z])\s*[\.\):：]\s*(.*)$")
FORMAT_INSTRUCTIONS = [
    r"First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx\.?",
    r"First perform reasoning, then finally answer the question[^\n]*?Answer: xxx\.?",
    r"Answer (?:the question )?(?:concisely |directly )?(?:with|using) (?:a |one )?single[- ]word or (?:a )?phrase\.?",
    r"Answer concisely with one word or phrase\.?",
    r"Provide a single-word or phrase answer\.?",
    r"Answer with the option'?s letter from the given choices directly\.?",
    r"Please answer the question and provide the correct option letter, e\.g\., A, B, C, D, at the end\.?",
    r"Please answer the question and provide the final answer at the end\.?",
    r"Please answer the question requiring an integer answer and provide the final value, e\.g\., 1, 2, 3, at the end\.?",
    r"Please answer the question requiring a floating-point number with (?:one|two) decimal places? and provide the final value, e\.g\., [0-9., ]+ at the end\.?",
    r"According to the question shown in the image, please (?:first )?(?:conduct|perform) reasoning, (?:and )?then (?:finally )?(?:answer the question and provide the final value|select the right answer from the choices), e\.g\., (?:The answer is xxx|Answer: xxx)\.?",
]
FORMAT_RE = [re.compile(p, re.I) for p in FORMAT_INSTRUCTIONS]
FINAL_LINE = re.compile(r"^\s*[-*#>]*\s*\**\s*(?:final |simple |short |revised |correct )?answer\s*\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$", re.I)
EXTRACT = [
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"(?:the )?(?:correct |final )?answer is[:\s]*\(?([^\n]+?)\)?\s*\.?\s*$", re.I),
    re.compile(r"(?:option |choice )?\(?([A-Z])\)? is the correct (?:answer|option|choice)", re.I),
]


def _split_choices(question):
    """질문에서 보기 블록을 떼어 (본문, [보기...]) 를 돌려준다. 보기가 없으면 (질문, None)."""
    lines = question.split("\n")
    for i, line in enumerate(lines):
        if CHOICE_HEADER.match(line):
            opts, j = [], i + 1
            while j < len(lines) and (m := CHOICE_LINE.match(lines[j])) and m.group(1) == chr(ord("A") + len(opts)):
                opts.append(m.group(2).strip()); j += 1
            if len(opts) >= 2:
                rest = lines[:i] + lines[j:]
                return "\n".join(rest).strip(), opts
    return question, None


def _final_answer(answer):
    """(본문, 최종답, 출처) . 출처: explicit | extracted | None"""
    text = answer.rstrip()
    lines = text.split("\n")
    m = FINAL_LINE.match(lines[-1])
    if m:
        return "\n".join(lines[:-1]).rstrip(), m.group(1).strip(), "explicit"
    tail = text[-400:]
    for rx in EXTRACT:
        hits = list(rx.finditer(tail))
        if hits:
            return text, hits[-1].group(1).strip().rstrip("."), "extracted"
    return text, None, None


ANSWER_PREFIX = re.compile(r"^(?:the )?(?:correct |final )?answer is[:\s]*", re.I)


def _clean_final(x):
    x = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", x).replace("**", "").strip()
    x = re.sub(r"^(?:(?:final|simple|short|revised|correct)\s+)?answer\s*[:：]\s*", "", x, flags=re.I).strip()  # "Simple Answer: X"
    x = ANSWER_PREFIX.sub("", x).strip()
    return x.rstrip(".").strip()


def convert(rec):
    """레코드 → {"messages": [...], "images": [경로], "status": ..., "flags": {...}}"""
    flags = {}
    conv = rec.get("conversations") or []
    if not conv or conv[0].get("from") != "human" or len(conv) < 2 or conv[1].get("from") != "gpt":
        return {"status": "unsupported", "flags": {"reason": "first turn is not human→gpt"}}
    if rec.get("n_turns", 1) > 1:
        flags["multi_turn"] = rec["n_turns"]
    q, a = conv[0]["value"], conv[1]["value"]
    if q.count("<image>") != 1:
        return {"status": "unsupported", "flags": {"reason": f"<image> count {q.count('<image>')}"}}

    removed = []
    for rx in FORMAT_RE:
        removed += rx.findall(q)
        q = rx.sub("", q)
    if removed:
        flags["removed_instructions"] = removed
    q = re.sub(r"(?im)^\s*Hint:\s*$", "", q)  # 지시문을 지우고 남은 빈 "Hint:" 줄
    q = re.sub(r"^\s*Question:\s*", "", q.replace("<image>", "<image 1>").strip(), flags=re.M)
    q = re.sub(r"\n{3,}", "\n\n", q).strip()
    body, options = _split_choices(q)

    reasoning, final, src = _final_answer(a)
    if final is not None:
        final = _clean_final(final)
    status = "ok"
    # 답만 있고 추론이 거의 없는 경우 ("The answer is C" 한 줄 등)
    if final is not None and len(ANSWER_PREFIX.sub("", reasoning).split()) < 15 and len(reasoning.split("\n")) <= 1:
        status = "short_no_reasoning"
    if final is None:
        if len(a.split()) <= 8 and "\n" not in a.strip():
            final, reasoning, status = a.strip().rstrip("."), "", "short_no_reasoning"
        else:
            status = "no_final_answer"
    if final is not None and not options and len(final) > 40:
        flags["long_final_answer"] = True  # 문장형 최종답: open 채점(숫자/문자열 매칭)에 맞지 않을 수 있음, 검수 대상
    if options and final is not None:
        f = final.strip("() .*")
        if len(f) == 1 and f.upper() in [chr(ord("A") + i) for i in range(len(options))]:
            final = f.upper()
        elif (hit := [chr(ord("A") + i) for i, o in enumerate(options) if o.strip().lower() == final.strip().lower()]):
            final = hit[0]; flags["mapped_option_text"] = True
        else:
            status = "mc_answer_unmapped" if status == "ok" else status

    if options:
        opt_text = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options))
        user_text = f"{PRE_PROMPT}{body}\nOptions:\n{opt_text}\n{MC_PROMPT}"
    else:
        user_text = f"{PRE_PROMPT}{body}\n{OPEN_PROMPT}"

    # 평가 mmdl_doc_to_messages 와 같은 방식으로 <image N> 자리에 이미지를 끼움
    content, pos = [], 0
    for m in IMAGE_TOKEN.finditer(user_text):
        if m.start() > pos:
            content.append({"type": "text", "text": user_text[pos:m.start()]})
        content.append({"type": "image", "image": rec["image"]})
        pos = m.end()
        break  # 이미지 1개
    if pos < len(user_text):
        content.append({"type": "text", "text": user_text[pos:]})

    assistant = None
    if final is not None:
        assistant = (reasoning + "\n" if reasoning else "") + f"Answer: {final}"
        flags["answer_source"] = src or "short_answer"
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
    return {"idx": rec.get("idx"), "id": rec.get("id"), "source": rec.get("source"),
            "question_type": "multiple-choice" if options else "open",
            "messages": msgs, "images": [rec["image"]], "status": status, "flags": flags}


if __name__ == "__main__":
    # 사용: convert_v12.py <filtered.jsonl> <out.jsonl>  → 마지막에 status 집계 출력
    import collections
    cnt = collections.Counter()
    with open(sys.argv[1]) as fi, open(sys.argv[2], "w") as fo:
        for line in fi:
            out = convert(json.loads(line))
            cnt[out["status"]] += 1
            if out["status"] == "ok":
                cnt["ok_" + out["question_type"]] += 1
                if out["flags"].get("multi_turn"):
                    cnt["ok_multi_turn_first_turn_used"] += 1
                if out["flags"].get("long_final_answer"):
                    cnt["ok_long_final_answer"] += 1
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps(dict(cnt), ensure_ascii=False))
    print("학습 사용 = status ok 만 (short_no_reasoning·no_final_answer·mc_answer_unmapped·unsupported 제외)")
