#!/usr/bin/env python
"""ECD-10k-Images (ChartFoundation/ECD-10k-Images @ 284489f) QA 를 v1.2 학습 형식으로 바꾼다.

  python convert_ecd.py <ECD_QAs_All.json> <image_root> <out.jsonl>
  → 한 줄 = convert_v12.convert() 와 같은 스키마 {idx, id, source, question_type, messages, images, status, flags}

ECD 원본: {"image_id", "question", "answer", "split": descriptive|reasoning}. 보기 없는 open 문항뿐이다.
  추론형 답은 "<추론> Answer: <최종답>" 처럼 Answer 가 같은 줄 끝에 붙어 있다.
변환:
  user      : "Question: <image 1>\\n<질문>\\n" + OPEN_PROMPT (convert_v12 와 같은 문구). 이미지는 질문 앞 1장.
  assistant : <추론>\\nAnswer: <최종답>   (마지막 "Answer:" 를 기준으로 나눔)
status (convert_v12 와 같은 이름):
  ok                 : 마지막 "Answer:" 앞에 추론이 15단어 이상
  short_no_reasoning : descriptive split, 또는 추론이 15단어 미만  → 학습 제외 (MAmmoTH 정책과 같게)
  no_final_answer    : "Answer:" 가 없음 → 제외
  long_final_answer  : 최종답 40자 초과 문장형 → 학습 제외 (2026-11-17 사용자 확정; open 채점의 숫자·문자열 매칭에 안 맞음)
학습 후보 = status ok (= reasoning, 추론 15단어 이상, 최종답 40자 이하). MAmmoTH 와의 혼합 비율은 미정.
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_v12 import SYSTEM_PROMPT, PRE_PROMPT, OPEN_PROMPT, _clean_final  # noqa: E402

ANS = re.compile(r"answer\s*:", re.I)


def convert(i, x, image_root):
    img = os.path.join(image_root, f"{x['image_id']}.png")
    flags = {"split": x["split"]}
    a = x["answer"].strip()
    hits = list(ANS.finditer(a))
    if not hits:
        status, reasoning, final = "no_final_answer", a, None
    else:
        h = hits[-1]
        reasoning, final = a[: h.start()].strip(), _clean_final(a[h.end():].strip())
        status = "ok" if (x["split"] == "reasoning" and len(reasoning.split()) >= 15) else "short_no_reasoning"
    if x["split"] == "descriptive":
        status = "short_no_reasoning" if final is not None else "no_final_answer"
    if final is not None and len(final) > 40:
        flags["long_final_answer"] = True
        if status == "ok":
            status = "long_final_answer"
    user = f"{PRE_PROMPT}<image 1>\n{x['question'].strip()}\n{OPEN_PROMPT}"
    pre, post = user.split("<image 1>")
    content = [{"type": "text", "text": pre}, {"type": "image", "image": img}, {"type": "text", "text": post}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]
    if final is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": (reasoning + "\n" if reasoning else "") + f"Answer: {final}"}]})
    return {"idx": i, "id": f"ecd_{x['image_id']}_{i}", "source": f"ECD-{x['split']}", "question_type": "open",
            "messages": msgs, "images": [img], "status": status, "flags": flags}


if __name__ == "__main__":
    src, image_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    data = json.load(open(src))  # 107 MB, 메모리에 올려도 됨
    cnt = collections.Counter()
    with open(out, "w") as fo:
        for i, x in enumerate(data):
            r = convert(i, x, image_root)
            cnt[(x["split"], r["status"])] += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(cnt.items()):
        print(k, v)
