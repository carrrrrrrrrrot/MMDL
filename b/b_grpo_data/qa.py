"""Validate the synthetic GRPO manifest, image files, and heldout isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from generate import HELDOUT_POSITIONS, LETTERS, SPLIT_BLOCK, SYSTEM, TASKS


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(dataset_dir: Path) -> dict:
    root = dataset_dir.resolve(strict=True)
    manifest = root / "manifest.jsonl"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    _require(_hash(manifest) == metadata["manifest_sha256"], "Manifest SHA256 changed")
    _require(metadata["source"] == "synthetic_generated", "Unexpected source")
    per_task = metadata["per_task"]
    _require(per_task > 0 and per_task % SPLIT_BLOCK == 0, "Invalid per_task count")
    _require(metadata["revision"] == f"synthetic-v1-seed{metadata['master_seed']}-n{per_task}", "Revision mismatch")
    seen_ids: set[str] = set()
    seen_images: dict[str, str] = {}
    counts: Counter[tuple[str, str, str]] = Counter()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, 1):
            row = json.loads(text)
            sample_id = row.get("sample_id")
            _require(isinstance(sample_id, str) and sample_id not in seen_ids, f"Duplicate/bad ID at line {line_number}")
            seen_ids.add(sample_id)
            _require(row.get("original_id") == sample_id, f"Original ID mismatch: {sample_id}")
            _require(row.get("source") == "synthetic_generated", f"Forbidden source: {sample_id}")
            _require(row.get("dataset_revision") == metadata["revision"], f"Revision mismatch: {sample_id}")
            task = row.get("task")
            split = row.get("split")
            _require(task in TASKS and split in {"train", "heldout"}, f"Bad task/split: {sample_id}")
            index = int(sample_id.rsplit("_", 1)[1])
            expected_split = "heldout" if index % SPLIT_BLOCK in HELDOUT_POSITIONS else "train"
            _require(split == expected_split, f"Split rule changed: {sample_id}")
            answer = row.get("answer")
            _require(answer in LETTERS and answer == LETTERS[index % 4], f"Bad gold: {sample_id}")
            _require(row.get("answer_type") == "mc", f"Bad answer type: {sample_id}")
            choices = row.get("choices")
            _require(isinstance(choices, dict) and set(choices) == set(LETTERS), f"Bad choices: {sample_id}")
            _require(len(set(choices.values())) == 4, f"Duplicate choices: {sample_id}")
            _require(isinstance(row.get("generator_seed"), int), f"Bad seed: {sample_id}")
            messages = row.get("prompt")
            _require(isinstance(messages, list) and len(messages) == 2, f"Bad messages: {sample_id}")
            _require(messages[0] == {"role": "system", "content": SYSTEM}, f"Bad system: {sample_id}")
            user = messages[1]
            _require(user.get("role") == "user" and isinstance(user.get("content"), str), f"Bad user: {sample_id}")
            content = user["content"]
            _require(content.startswith(row["question"] + "\n"), f"Question/prompt mismatch: {sample_id}")
            _require("<image" not in content, f"Duplicate image marker risk: {sample_id}")
            _require(content.endswith("Answer: X"), f"Answer instruction missing: {sample_id}")
            for letter in LETTERS:
                _require(f"\n{letter}. {choices[letter]}\n" in content + "\n", f"Option missing: {sample_id}/{letter}")
            image_rel = row.get("image_path")
            _require(isinstance(image_rel, str) and not Path(image_rel).is_absolute(), f"Absolute image path: {sample_id}")
            image_path = (root / image_rel).resolve(strict=True)
            _require(image_path.is_relative_to(root / "images"), f"Image escapes dataset: {sample_id}")
            image_hash = _hash(image_path)
            _require(image_hash == row.get("image_sha256"), f"Image SHA mismatch: {sample_id}")
            _require(image_hash not in seen_images, f"Image reused: {sample_id} and {seen_images.get(image_hash)}")
            seen_images[image_hash] = sample_id
            with Image.open(image_path) as image:
                _require(image.mode == "RGB", f"Non-RGB image: {sample_id}")
                _require(image.width >= 320 and image.height >= 320, f"Small image: {sample_id}")
                image.verify()
            counts[(task, split, answer)] += 1
    _require(len(seen_ids) == metadata["total_items"] == per_task * len(TASKS), "Total count mismatch")
    for task in TASKS:
        for answer in LETTERS:
            _require(counts[(task, "heldout", answer)] == per_task // 32, f"Heldout balance: {task}/{answer}")
            _require(counts[(task, "train", answer)] == per_task * 7 // 32, f"Train balance: {task}/{answer}")
    _require(
        metadata["counts_by_task_split_answer"] == {"/".join(key): value for key, value in sorted(counts.items())},
        "Metadata counts mismatch",
    )
    return {
        "status": "PASS",
        "total_items": len(seen_ids),
        "train_items": sum(count for (task, split, answer), count in counts.items() if split == "train"),
        "heldout_items": sum(count for (task, split, answer), count in counts.items() if split == "heldout"),
        "unique_image_sha256": len(seen_images),
        "manifest_sha256": metadata["manifest_sha256"],
        "counts_by_task_split_answer": metadata["counts_by_task_split_answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
