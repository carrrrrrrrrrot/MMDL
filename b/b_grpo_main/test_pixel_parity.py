"""CPU check that training images use the same Qwen resize limits as v1.2."""
from __future__ import annotations

import argparse

from datasets import Dataset, Image as DatasetImage
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor
from trl.data_utils import prepare_multimodal_messages

from train import prepare_images_v12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.processor)
    original_size = processor.image_processor.size
    source = Image.new("RGB", (73, 61), "red")
    reference = fetch_image({"image": source}, image_patch_size=16)
    dataset = Dataset.from_list([{"image": source}]).cast_column("image", DatasetImage())
    dataset = prepare_images_v12(dataset, processor)
    processed = dataset[0]["image"]
    assert processed.size == reference.size
    assert list(processed.getdata()) == list(reference.getdata())
    assert processor.image_processor.size == {"shortest_edge": 4096, "longest_edge": 16777216}
    prompt = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": [{"type": "image"},
                                           {"type": "text", "text": "Question: value?\nAnswer: <answer>"}]}]
    messages = prepare_multimodal_messages(prompt, images=[processed])
    reference_messages = prepare_multimodal_messages(prompt, images=[reference])
    tokens = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    reference_tokens = processor.apply_chat_template(reference_messages, add_generation_prompt=True,
                                                     tokenize=True, return_dict=True, return_tensors="pt")
    assert tokens["pixel_values"].numel() > 0
    assert tokens["input_ids"].equal(reference_tokens["input_ids"])
    assert tokens["pixel_values"].equal(reference_tokens["pixel_values"])
    multi = Dataset.from_list([{"images": [source, source]}]).cast_column("images", [DatasetImage()])
    multi = prepare_images_v12(multi, processor)
    assert [image.size for image in multi[0]["images"]] == [reference.size, reference.size]
    print(f"PASS: raw {source.size}, v1.2 fetch_image {reference.size}, train {processed.size}, "
          f"processor_size {original_size} -> {processor.image_processor.size}, "
          f"tokens {tokens['input_ids'].shape[1]}, 1- and 2-image mapping")


if __name__ == "__main__":
    main()
