"""vLLM 백엔드(lmms-eval vllm chat) 경로로 900문항의 입력 토큰 id 를 만들어 저장합니다.

lmms_eval/models/chat/vllm.py 와 같은 순서로 처리합니다:
ChatMessages.to_qwen3_vl_openai_messages → LLM.chat.
LLM 설정은 scripts/run_mmmu_eval.sh --backend vllm 과 같습니다. 생성은 1 토큰만 합니다.

사용: python scripts/parity/vllm_prompt_ids.py <model_path> <data_root> <out.jsonl>
"""

import json
import sys

from vllm import LLM, SamplingParams

from common import load_docs, load_task_kwargs, mmdl_utils
from lmms_eval.protocol import ChatMessages

model_path, data_root, out_path = sys.argv[1:4]
kwargs = load_task_kwargs()["vllm"]
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.85,
    max_model_len=32768,
    limit_mm_per_prompt={"image": 7},
    mm_processor_kwargs={"min_pixels": 4096, "max_pixels": 16777216},
    trust_remote_code=True,
    seed=1,
)
# lmms-eval chat/vllm.py 기본값 (비디오 전용이라 이미지에는 영향 없음)
video_kwargs = {"max_pixels": 1605632, "min_pixels": 28, "max_frames": 768, "nframes": 32}

docs = load_docs(data_root)
conversations = [ChatMessages(messages=mmdl_utils.mmdl_doc_to_messages(doc, kwargs)).to_qwen3_vl_openai_messages(video_kwargs) for doc in docs]
outputs = llm.chat(conversations, SamplingParams(max_tokens=1, temperature=0))

with open(out_path, "w") as f:
    for doc, out in zip(docs, outputs):
        f.write(json.dumps({"id": doc["id"], "input_ids": list(out.prompt_token_ids)}) + "\n")
print("done", out_path)
