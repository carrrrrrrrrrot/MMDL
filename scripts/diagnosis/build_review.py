"""오답 원인("눈" = 이미지 인식 vs "뇌" = 지식·추론) 수작업 분류용 검토 자료를 만듭니다.

표본 선정 규칙 (고정, 임의 선택 없음):
  과목 Math, Chemistry, Physics 에서
  - M4_final 3 seed 중 2개 이상에서 오답이고
  - seed 3407 에서 "Answer:" 로 답을 냈지만 틀린 문항
  을 문항 번호순으로 과목당 최대 8개.

출력 (out_dir):
  README.md        검토 방법과 분류 기준
  index.md         문항 목록 (각 문항 페이지 링크)
  items/<id>.md    이미지, 문제, 보기, 정답, 모델 응답(seed 3407 전문), 다른 seed 의 답
  images/<id>_<n>.png
  labels.csv       id, 과목, 정답, 모델 답, claude_label, claude_evidence, user_label, user_note

사용: python scripts/diagnosis/build_review.py <M4_final run dir> <data_root> <out_dir>
"""

import ast
import csv
import glob
import io
import json
import os
import re
import sys

import pyarrow.parquet as pq
from PIL import Image

from lmms_eval.tasks.mmmu.utils import eval_open

SUBJECTS = ["Math", "Chemistry", "Physics"]
PER_SUBJECT = 8
SEEDS = ["3407", "1", "2"]


def is_correct(acc):
    pred = acc["parsed_pred"][0]
    if acc["question_type"] == "multiple-choice":
        return pred == acc["answer"]
    return eval_open(acc["answer"], pred)


def main():
    run_dir, data_root, out_dir = sys.argv[1:4]
    runs = {}
    for s in SEEDS:
        path = glob.glob(os.path.join(run_dir, f"seed_{s}", "*", "*_samples_*.jsonl"))[0]
        runs[s] = {r["mmmu_acc"]["id"]: r for r in map(json.loads, open(path))}

    selected = []
    for subj in SUBJECTS:
        ids = sorted((i for i in runs["3407"] if runs["3407"][i]["mmmu_acc"]["subdomain"] == subj), key=lambda x: int(x.rsplit("_", 1)[1]))
        cand = [
            i
            for i in ids
            if sum(not is_correct(runs[s][i]["mmmu_acc"]) for s in SEEDS) >= 2
            and runs["3407"][i]["mmmu_acc"]["final_answer"] is not None
            and not is_correct(runs["3407"][i]["mmmu_acc"])
        ]
        selected += cand[:PER_SUBJECT]

    docs = {}
    for subj in SUBJECTS:
        for doc in pq.read_table(glob.glob(os.path.join(data_root, subj, "validation-*.parquet"))[0]).to_pylist():
            docs[doc["id"]] = doc

    os.makedirs(os.path.join(out_dir, "items"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    index = ["# 오답 원인 분류 검토 목록", "", f"표본 {len(selected)}문항 (선정 규칙은 README.md 참고)", "", "| # | 문항 | 유형 | 정답 | 모델 답 (3407 / 1 / 2) | 페이지 |", "|---|---|---|---|---|---|"]
    rows = []
    for k, i in enumerate(selected, 1):
        doc, r = docs[i], runs["3407"][i]
        acc = r["mmmu_acc"]
        preds = [runs[s][i]["mmmu_acc"]["parsed_pred"][0] if runs[s][i]["mmmu_acc"]["question_type"] == "multiple-choice" else runs[s][i]["mmmu_acc"]["final_answer"] for s in SEEDS]
        refs = sorted({int(n) for n in re.findall(r"<image (\d+)>", doc["question"] + doc["options"])})
        img_lines = []
        for n in refs:
            value = doc.get(f"image_{n}")
            if not value:
                continue
            img = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            name = f"{i}_{n}.png"
            img.save(os.path.join(out_dir, "images", name))
            img_lines.append(f"**image {n}** ({img.size[0]}×{img.size[1]} px)\n\n![image {n}](../images/{name})\n")
        # 응답 안에 ``` 가 들어 있어도 코드 블록이 끊기지 않도록, 응답 속 가장 긴 백틱 연속보다 긴 fence 를 씁니다.
        longest = max((len(m) for m in re.findall(r"`+", r["filtered_resps"])), default=0)
        fence = "`" * max(3, longest + 1)
        options = ""
        if acc["question_type"] == "multiple-choice":
            opts = ast.literal_eval(doc["options"])
            options = "\n".join(f"- **{chr(65 + j)}.** {o}" for j, o in enumerate(opts))
        page = [
            f"# {i}",
            "",
            f"- 과목: {acc['subdomain']} / 유형: {acc['question_type']}",
            f"- **정답: {acc['answer']}**",
            f"- 모델 최종 답 (seed 3407 / 1 / 2): {preds[0]} / {preds[1]} / {preds[2]}",
            f"- seed 3407 추출 줄: `{acc['final_answer']}`",
            "",
            "## 이미지",
            "",
            *img_lines,
            "## 문제",
            "",
            doc["question"],
            "",
            options,
            "",
            "## 모델 응답 (seed 3407 전문)",
            "",
            fence + "text",
            r["filtered_resps"],
            fence,
        ]
        with open(os.path.join(out_dir, "items", f"{i}.md"), "w") as f:
            f.write("\n".join(page) + "\n")
        index.append(f"| {k} | {i} | {acc['question_type']} | {acc['answer']} | {' / '.join(map(str, preds))} | [열기](items/{i}.md) |")
        rows.append({"id": i, "subject": acc["subdomain"], "gold": acc["answer"], "pred_3407": preds[0], "claude_label": "", "claude_evidence": "", "user_label": "", "user_note": ""})

    with open(os.path.join(out_dir, "index.md"), "w") as f:
        f.write("\n".join(index) + "\n")
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(selected)} items -> {out_dir}")


if __name__ == "__main__":
    main()
