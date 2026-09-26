"""HF 백엔드(lmms-eval qwen3_vl chat) 경로로 900문항의 입력 토큰 id 를 만들어 저장합니다.

lmms_eval/models/chat/qwen3_vl.py 의 generate_until 과 같은 순서로 처리합니다:
system 메시지 삽입 → ChatMessages.to_hf_messages → processor.apply_chat_template
→ qwen_vl_utils.process_vision_info(image_patch_size=16) → processor(do_resize=False).
모델은 올리지 않습니다 (processor 만 사용).

사용: python scripts/parity/hf_prompt_ids.py <model_path> <data_root> <out.jsonl>
"""

import json
import sys

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from common import load_docs, load_task_kwargs, mmdl_utils
from lmms_eval.protocol import ChatMessages

model_path, data_root, out_path = sys.argv[1:4]
kwargs = load_task_kwargs()["default"]
# lmms-eval simple/qwen3_vl.py 와 같은 인자 (do_resize=False 라 결과에는 영향 없음)
processor = AutoProcessor.from_pretrained(model_path, max_pixels=2048 * 32 * 32, min_pixels=256 * 32 * 32)

with open(out_path, "w") as f:
    for doc in load_docs(data_root):
        messages = mmdl_utils.mmdl_doc_to_messages(doc, kwargs)
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]})
        hf_messages = ChatMessages(messages=messages).to_hf_messages()
        text = processor.apply_chat_template([hf_messages], tokenize=False, add_generation_prompt=True)
        images, _, _ = process_vision_info([hf_messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
        inputs = processor(text=text, images=images, do_resize=False, return_tensors="pt")
        grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        f.write(json.dumps({"id": doc["id"], "input_ids": inputs["input_ids"][0].tolist(), "image_grid_thw": grid}) + "\n")
print("done", out_path)
