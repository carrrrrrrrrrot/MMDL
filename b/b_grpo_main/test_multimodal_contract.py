"""CPU contract check for ordered image placeholders in TRL 1.14.2."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from datasets import Dataset, Image as DatasetImage, Sequence
from transformers import AutoProcessor
from trl.data_utils import prepare_multimodal_messages

from data import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-a", required=True, type=Path)
    parser.add_argument("--image-b", required=True, type=Path)
    parser.add_argument("--processor", required=True, type=Path)
    args = parser.parse_args()
    question = "Compare the two synthetic grid images."
    row = {
        "sample_id": "synthetic_ordered_images_test",
        "source": "synthetic_generated",
        "dataset_revision": "contract-test-v1",
        "split": "train",
        "original_id": "ordered_images_test",
        "image_paths": [str(args.image_a.resolve(strict=True)), str(args.image_b.resolve(strict=True))],
        "question": question,
        "answer": "A",
        "answer_type": "mc",
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": question + " First image:"},
                {"type": "image"},
                {"type": "text", "text": "Second image:"},
                {"type": "image"},
                {"type": "text", "text": "A. same B. different\nEnd with Answer: X"},
            ]},
        ],
    }
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "test.jsonl"
        single = {
            **row,
            "sample_id": "synthetic_single_image_test",
            "original_id": "single_image_test",
            "image_path": str(args.image_a.resolve(strict=True)),
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question + " End with Answer: X"},
            ],
        }
        del single["image_paths"]
        path.write_text(json.dumps(row) + "\n" + json.dumps(single) + "\n", encoding="utf-8")
        rows, summary = load_manifest(path)
        assert summary["image_mode"] == "images" and summary["max_images_per_example"] == 2
        dataset = Dataset.from_list(rows).cast_column("images", Sequence(DatasetImage()))
        item = dataset[0]
        assert len(item["images"]) == 2
        assert len(dataset[1]["images"]) == 1
        messages = prepare_multimodal_messages(item["prompt"], images=item["images"])
        blocks = messages[1]["content"]
        assert [block["type"] for block in blocks] == ["text", "image", "text", "image", "text"]
        assert blocks[1]["image"].size == item["images"][0].size
        assert blocks[3]["image"].size == item["images"][1].size
        processor = AutoProcessor.from_pretrained(args.processor)
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        assert rendered.count("<|image_pad|>") == 2, rendered[:500]

        wrong = json.loads(json.dumps(row))
        wrong["prompt"][1]["content"].pop(3)
        path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
        try:
            load_manifest(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Image placeholder count mismatch must fail")

        wrong = json.loads(json.dumps(row))
        wrong["prompt"][1]["content"][0]["text"] += " <image 1>"
        path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
        try:
            load_manifest(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Literal image marker in text must fail")
    print("PASS: mixed one/two-image TRL/HF processor contract and mismatch guards")


if __name__ == "__main__":
    main()
