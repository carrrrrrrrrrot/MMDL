"""CPU diagnostic for Qwen image resizing used by B versus the v1.2 policy.

This does not modify training. It exposes the processor size and token/grid
differences so B can set min/max pixels and use the same fetch_image path as A.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=16777216)
    args = parser.parse_args()

    from PIL import Image
    from qwen_vl_utils import fetch_image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.processor.resolve(strict=True))
    original_size = dict(proc.image_processor.size)
    with Image.open(args.image.resolve(strict=True)) as opened:
        image = opened.convert("RGB")
    text = "Question: How many red squares?\nThink step by step, then give your final answer on the last line in the form \"Answer: X\", where X is the option letter."

    def encode(pic):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": pic}, {"type": "text", "text": text}]},
        ]
        result = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return {
            "image_size": list(pic.size),
            "input_token_count": int(result["input_ids"].shape[1]),
            "image_grid_thw": result["image_grid_thw"].tolist(),
            "pixel_values_shape": list(result["pixel_values"].shape),
        }

    default = encode(image)
    proc.image_processor.size = {"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels}
    fetched = fetch_image(
        {"image": image, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        image_patch_size=16,
    )
    aligned = encode(fetched)
    tiny = Image.new("RGB", (64, 64), "red")
    proc.image_processor.size = original_size
    tiny_default = encode(tiny)
    proc.image_processor.size = {"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels}
    tiny_fetched = fetch_image(
        {"image": tiny, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        image_patch_size=16,
    )
    tiny_aligned = encode(tiny_fetched)
    report = {
        "processor_default_size": original_size,
        "required_size": proc.image_processor.size,
        "default_raw_image": default,
        "v12_fetch_and_processor_size": aligned,
        "grid_equal": default["image_grid_thw"] == aligned["image_grid_thw"],
        "pixel_shape_equal": default["pixel_values_shape"] == aligned["pixel_values_shape"],
        "tiny_64_default": tiny_default,
        "tiny_64_v12_aligned": tiny_aligned,
        "note": "For training parity, set processor size to required_size and fetch each image with qwen_vl_utils.fetch_image(image_patch_size=16).",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
