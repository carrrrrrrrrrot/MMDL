"""참조 이미지 중 하나라도 THRESHOLD 픽셀 미만인 문항 id 를 출력합니다 (해상도 확대 진단 실험 (a) 의 대상).

THRESHOLD = 262,144 (= 256 × 32 × 32). Qwen 계열 processor 들이 흔히 쓰는 최소 픽셀 기준입니다.
사용: python scripts/diagnosis/small_image_ids.py <data_root> > ids.txt
"""

import glob
import io
import os
import re
import sys

import pyarrow.parquet as pq
from PIL import Image

THRESHOLD = 256 * 32 * 32

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "validation-*.parquet"))):
    for doc in pq.read_table(path).to_pylist():
        refs = {int(n) for n in re.findall(r"<image (\d+)>", doc["question"] + doc["options"])}
        sizes = [Image.open(io.BytesIO(doc[f"image_{n}"]["bytes"])).size for n in refs if doc.get(f"image_{n}")]
        if any(w * h < THRESHOLD for w, h in sizes):
            print(doc["id"])
