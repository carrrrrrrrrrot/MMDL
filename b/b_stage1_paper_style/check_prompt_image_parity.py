"""CPU parity check: EasyR1 vs existing B v1.2 prompt and image tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest_tensor(value) -> str:
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--b-main", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MMDL_V12_IMAGE_PARITY") != "1":
        raise RuntimeError("Set MMDL_V12_IMAGE_PARITY=1")
    sys.path.insert(0, str(args.b_main))

    from PIL import Image
    from qwen_vl_utils import fetch_image
    from transformers import AutoProcessor
    from verl.utils.dataset import RLHFDataset, process_image
    from verl.utils.tokenizer import get_processor, get_tokenizer

    source = next(json.loads(line) for line in args.source_manifest.open(encoding="utf-8") if json.loads(line)["split"] == "train")
    converted = json.loads(args.converted.open(encoding="utf-8").readline())
    if converted["sample_id"] != source["sample_id"]:
        raise RuntimeError("Converted row order differs from source")
    if len(converted["images"]) != 1:
        raise RuntimeError("This parity check expects one image")

    original = Image.open(converted["images"][0]).convert("RGB")
    b_image = fetch_image({"image": original}, image_patch_size=16)
    e_image = process_image(converted["images"][0], 4096, 16777216)

    b_processor = AutoProcessor.from_pretrained(args.model)
    b_processor.image_processor.size = {"shortest_edge": 4096, "longest_edge": 16777216}
    # The existing B trainer calls TRL prepare_multimodal_messages. For this
    # one-image source row, the exact transformation is: wrap system text in a
    # text block and populate the sole image placeholder with its PIL image.
    b_messages = [
        {"role": "system", "content": [{"type": "text", "text": source["prompt"][0]["content"]}]},
        {
            "role": "user",
            "content": [
                ({**block, "image": b_image} if block["type"] == "image" else block)
                for block in source["prompt"][1]["content"]
            ],
        },
    ]
    b_rendered = b_processor.apply_chat_template(b_messages, add_generation_prompt=True, tokenize=False)
    b_encoded = b_processor.apply_chat_template(
        b_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )

    e_processor = get_processor(str(args.model))
    e_tokenizer = get_tokenizer(str(args.model))
    e_dataset = RLHFDataset(
        data_path=str(args.converted),
        tokenizer=e_tokenizer,
        processor=e_processor,
        prompt_key="prompt",
        answer_key="answer",
        image_key="images",
        max_prompt_length=2048,
        min_pixels=4096,
        max_pixels=16777216,
        filter_overlong_prompts=False,
    )
    e_messages = e_dataset._build_messages(converted)
    e_rendered = e_processor.apply_chat_template(e_messages, add_generation_prompt=True, tokenize=False)
    e_encoded = e_processor([e_image], [e_rendered], add_special_tokens=False, return_tensors="pt")
    result = {
        "sample_id": converted["sample_id"],
        "rendered_prompt_equal": b_rendered == e_rendered,
        "input_ids_equal": bool((b_encoded["input_ids"] == e_encoded["input_ids"]).all()),
        "pixel_values_equal": bool((b_encoded["pixel_values"] == e_encoded["pixel_values"]).all()),
        "b_pixel_sha256": digest_tensor(b_encoded["pixel_values"]),
        "e_pixel_sha256": digest_tensor(e_encoded["pixel_values"]),
    }
    print(json.dumps(result, ensure_ascii=False))
    if not all(result[key] for key in ("rendered_prompt_equal", "input_ids_equal", "pixel_values_equal")):
        raise SystemExit("EasyR1 v1.2 prompt/image parity failed")


if __name__ == "__main__":
    main()
