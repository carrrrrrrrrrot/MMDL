"""Validate a provenance-bearing JSONL and build raw TRL multimodal rows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reward import validate_gold


SYSTEM_PROMPT = "You are a helpful assistant."
_IMAGE_MARKER = re.compile(r"<image(?:\s+\d+)?>", re.IGNORECASE)
_REQUIRED = {
    "sample_id", "source", "dataset_revision", "split", "original_id",
    "question", "prompt", "answer", "answer_type",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string(row: dict[str, Any], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {key} must be a non-empty string")
    return value


def _prompt(row: dict[str, Any], line_number: int, image_count: int) -> list[dict[str, Any]]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"Line {line_number}: prompt must have system and user messages")
    if not all(isinstance(item, dict) and set(item) == {"role", "content"} for item in prompt):
        raise ValueError(f"Line {line_number}: prompt messages need role and content only")
    if prompt[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise ValueError(f"Line {line_number}: system prompt differs from the agreed text")
    if prompt[1]["role"] != "user":
        raise ValueError(f"Line {line_number}: second prompt message must be user")
    user_content = prompt[1]["content"]
    if isinstance(user_content, str):
        if image_count != 1:
            raise ValueError(
                f"Line {line_number}: multiple images require structured user content with image blocks"
            )
        user_text = user_content
    elif isinstance(user_content, list):
        image_blocks = 0
        text_blocks = []
        for block in user_content:
            if not isinstance(block, dict):
                raise ValueError(f"Line {line_number}: user content block must be an object")
            if block.get("type") == "image" and set(block) == {"type"}:
                image_blocks += 1
            elif block.get("type") == "text" and set(block) == {"type", "text"} and isinstance(block["text"], str):
                text_blocks.append(block["text"])
            else:
                raise ValueError(f"Line {line_number}: unsupported or pre-filled content block: {block}")
        if image_blocks != image_count:
            raise ValueError(
                f"Line {line_number}: {image_blocks} image blocks for {image_count} image paths"
            )
        user_text = "\n".join(text_blocks)
    else:
        raise ValueError(f"Line {line_number}: user content must be text or structured blocks")
    if not user_text.strip():
        raise ValueError(f"Line {line_number}: empty user prompt")
    if _IMAGE_MARKER.search(user_text):
        raise ValueError(
            f"Line {line_number}: omit literal <image N>; TRL inserts the image through its processor"
        )
    if row["question"].strip() not in user_text:
        raise ValueError(f"Line {line_number}: question is absent from user prompt")
    if "Answer: X" not in user_text:
        raise ValueError(f"Line {line_number}: user prompt must specify final Answer: X format")
    return prompt


def load_manifest(path: Path, selected_split: str = "train") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Dataset manifest is not a file: {path}")
    rows: list[dict[str, Any]] = []
    sources: set[tuple[str, str, str]] = set()
    ids: set[str] = set()
    available_splits: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON") from exc
            if not isinstance(raw, dict) or not _REQUIRED <= set(raw):
                missing = sorted(_REQUIRED - set(raw)) if isinstance(raw, dict) else sorted(_REQUIRED)
                raise ValueError(f"Line {line_number}: missing fields {missing}")
            sample_id = _string(raw, "sample_id", line_number)
            if sample_id in ids:
                raise ValueError(f"Line {line_number}: duplicate sample_id {sample_id}")
            ids.add(sample_id)
            source = _string(raw, "source", line_number)
            if "mmmu" in source.lower().replace("-", "_"):
                raise ValueError(f"Line {line_number}: MMMU/MMMU-Pro may not be GRPO training data")
            revision = _string(raw, "dataset_revision", line_number)
            split = _string(raw, "split", line_number)
            available_splits.add(split)
            _string(raw, "original_id", line_number)
            _string(raw, "question", line_number)
            answer = _string(raw, "answer", line_number)
            answer_type = _string(raw, "answer_type", line_number)
            validate_gold(answer, answer_type)
            if ("image_path" in raw) == ("image_paths" in raw):
                raise ValueError(f"Line {line_number}: provide exactly one of image_path or image_paths")
            if "image_path" in raw:
                image_texts = [_string(raw, "image_path", line_number)]
            else:
                image_texts = raw["image_paths"]
                if not isinstance(image_texts, list) or not image_texts or not all(
                    isinstance(item, str) and item.strip() for item in image_texts
                ):
                    raise ValueError(f"Line {line_number}: image_paths must be a non-empty string list")
            if len(image_texts) > 7:
                raise ValueError(f"Line {line_number}: v1.2 supports at most seven images")
            image_paths = []
            for image_text in image_texts:
                image_path = Path(image_text)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                image_path = image_path.resolve(strict=True)
                if not image_path.is_file():
                    raise ValueError(f"Line {line_number}: image is not a file: {image_path}")
                image_paths.append(str(image_path))
            prompt = _prompt(raw, line_number, len(image_paths))
            if split == selected_split:
                sources.add((source, revision, split))
                entry = {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "answer": answer,
                    "answer_type": answer_type,
                }
                if len(image_paths) == 1 and isinstance(prompt[1]["content"], str):
                    entry["image"] = image_paths[0]
                else:
                    entry["images"] = image_paths
                rows.append(entry)
    if not rows:
        raise ValueError(f"Dataset manifest has no examples in split {selected_split}: {path}")
    if any("images" in row for row in rows):
        for row in rows:
            if "image" in row:
                row["images"] = [row.pop("image")]
            # Arrow cannot store a string and a list in the same nested
            # `content` field. Keep the multimodal dataset homogeneous.
            system = row["prompt"][0]
            user = row["prompt"][1]
            row["prompt"] = [
                {"role": "system", "content": [{"type": "text", "text": system["content"]}]},
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"}, {"type": "text", "text": user["content"]}]
                        if isinstance(user["content"], str) else user["content"]
                    ),
                },
            ]
    image_mode = "images" if "images" in rows[0] else "image"
    all_images = [image for row in rows for image in (row["images"] if image_mode == "images" else [row["image"]])]
    summary = {
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "total_examples": len(ids),
        "selected_split": selected_split,
        "selected_examples": len(rows),
        "available_splits": sorted(available_splits),
        "sources": [
            {"source": source, "revision": revision, "split": split}
            for source, revision, split in sorted(sources)
        ],
        "image_mode": image_mode,
        "max_images_per_example": max(len(row["images"]) if image_mode == "images" else 1 for row in rows),
        "unique_images": len(set(all_images)),
        "answer_types": {
            kind: sum(row["answer_type"] == kind for row in rows)
            for kind in sorted({row["answer_type"] for row in rows})
        },
    }
    return rows, summary
