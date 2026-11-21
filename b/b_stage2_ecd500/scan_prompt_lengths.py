"""Check all ECD500 multimodal prompts against the training 8192-token cap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = get_tokenizer(args.base, trust_remote_code=False, use_fast=True)
    processor = get_processor(args.base, trust_remote_code=False, use_fast=True)
    dataset = RLHFDataset(
        data_path=str(args.train), tokenizer=tokenizer, processor=processor,
        prompt_key="prompt", answer_key="answer", image_key="images",
        max_prompt_length=8192, truncation="error", min_pixels=4096,
        max_pixels=16777216, filter_overlong_prompts=False,
    )
    if len(dataset) != 500:
        raise ValueError(f"Expected 500 rows, got {len(dataset)}")
    lengths = []
    ids = []
    for index in range(len(dataset)):
        example = dataset[index]
        lengths.append(int(example["attention_mask"].sum().item()))
        ids.append(str(example["sample_id"]))
        if (index + 1) % 50 == 0:
            print(json.dumps({"scanned": index + 1, "max_prompt_tokens": max(lengths)}), flush=True)
    ordered = sorted(lengths)
    result = {
        "rows": len(lengths), "max_prompt_tokens": max(lengths),
        "mean_prompt_tokens": statistics.mean(lengths),
        "p95_prompt_tokens": ordered[int(0.95 * (len(ordered) - 1))],
        "cap": 8192, "over_cap": sum(length > 8192 for length in lengths),
        "longest_ids": [ids[index] for index in sorted(range(len(ids)), key=lambda i: lengths[i], reverse=True)[:10]],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
