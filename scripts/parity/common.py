"""HF / vLLM 백엔드 입력 동일성 검증용 공통 함수.

두 백엔드가 같은 문항에 대해 모델에 넣는 토큰 id 가 같은지 확인하기 위해,
평가 task(mmmu_val_mmdl)와 같은 함수로 메시지를 만듭니다.
"""

import glob
import io
import os
import sys

import pyarrow.parquet as pq
import yaml
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TASK_DIR = os.path.join(REPO_ROOT, "eval_tasks", "mmmu_mmdl")
sys.path.insert(0, TASK_DIR)

import mmdl_utils  # noqa: E402


def load_task_kwargs():
    """task 템플릿의 lmms_eval_specific_kwargs 를 읽습니다 (!function 줄은 제외)."""
    with open(os.path.join(TASK_DIR, "mmmu_val_mmdl.yaml.tmpl")) as f:
        text = "\n".join(line for line in f.read().splitlines() if "!function" not in line)
    return yaml.safe_load(text)["lmms_eval_specific_kwargs"]


def load_docs(data_root):
    """lmms-eval 과 같은 순서(과목 폴더 이름순, 파일 안의 순서)로 900문항을 읽습니다."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_root, "*", "validation-*.parquet"))):
        for doc in pq.read_table(path).to_pylist():
            for i in range(1, 8):
                value = doc.get(f"image_{i}")
                doc[f"image_{i}"] = Image.open(io.BytesIO(value["bytes"])) if value else None
            docs.append(doc)
    return docs
