"""CPU checks for synthetic MC regression and A's open-answer prompt shape."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from data import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows, info = load_manifest(args.synthetic_manifest)
    assert len(rows) == 1344 and info["total_examples"] == 1536
    assert all(len(row["options"]) == 4 and row["answer"] in "ABCD" for row in rows)

    source_image = next(item["image"] for item in rows if "image" in item)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "open.jsonl"
        sample = {
            "sample_id": "open_1", "source": "synthetic_contract", "dataset_revision": "test-v1",
            "split": "train", "original_id": "open_1", "image_path": source_image,
            "question": "What is the value shown?", "answer": "15 cm", "answer_type": "open",
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Question: "},
                    {"type": "image"},
                    {"type": "text", "text": "What is the value shown?\nThink step by step, then give your final answer on the last line in the form \"Answer: <answer>\"."},
                ]},
            ],
        }
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        selected, open_info = load_manifest(path)
        assert len(selected) == 1 and selected[0]["answer_type"] == "open"
        assert selected[0]["options"] == [] and open_info["image_mode"] == "images"
    print("PASS: synthetic MC 1344 rows and v1.2 structured open prompt")


if __name__ == "__main__":
    main()
