"""hf_prompt_ids.py 와 vllm_prompt_ids.py 의 결과를 문항별로 비교합니다.

사용: python scripts/parity/compare.py <hf.jsonl> <vllm.jsonl>
"""

import json
import sys

IMAGE_PAD = 151655  # <|image_pad|>


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


hf, vl = load(sys.argv[1]), load(sys.argv[2])
assert hf.keys() == vl.keys(), "문항 목록이 다릅니다"

same, diffs = 0, []
for k in hf:
    a, b = hf[k]["input_ids"], vl[k]["input_ids"]
    if a == b:
        same += 1
    else:
        diffs.append((k, len(a), len(b), a.count(IMAGE_PAD), b.count(IMAGE_PAD)))

print(f"문항 {len(hf)}개 중 입력 토큰 id 완전 일치: {same}")
for k, la, lb, ia, ib in diffs[:20]:
    print(f"  불일치 {k}: 길이 hf={la} vllm={lb}, 이미지 토큰 hf={ia} vllm={ib}")
