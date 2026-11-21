"""Convert the B ECD JSONL manifest to EasyR1's image dataset schema.

This preserves the user text, image order, and v1.2 system prompt. The pinned
EasyR1 fork must apply system_prompt.patch before using this output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = "You are a helpful assistant."


def convert_row(row: dict) -> dict:
    prompt = row["prompt"]
    if len(prompt) != 2 or prompt[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise ValueError(f"Unexpected system prompt for {row.get('sample_id')}")
    user = prompt[1]
    if user.get("role") != "user":
        raise ValueError(f"Missing user message for {row.get('sample_id')}")
    content = user["content"]
    if isinstance(content, str):
        user_text = content
    else:
        parts = []
        for block in content:
            if block.get("type") == "image":
                parts.append("<image>")
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                raise ValueError(f"Unsupported content block for {row.get('sample_id')}: {block}")
        user_text = "".join(parts)
    images = row.get("image_paths")
    if images is None:
        images = [row["image_path"]]
    if not images or user_text.count("<image>") != len(images):
        raise ValueError(f"Image marker count differs from paths for {row.get('sample_id')}")
    if row["answer_type"] == "mc":
        choices = row.get("choices") or {}
        options = list(choices.values()) if choices else row.get("options", [])
    else:
        options = []
    gold = {
        "answer": row["answer"],
        "answer_type": row["answer_type"],
        "options": options,
        "sample_id": row["sample_id"],
    }
    return {
        "sample_id": row["sample_id"],
        "system_prompt": SYSTEM_PROMPT,
        "prompt": user_text,
        "answer": json.dumps(gold, ensure_ascii=False, sort_keys=True),
        "images": images,
    }


def selected_rows(manifest: Path, split: str, limit: int | None, seed: int) -> list[dict]:
    rows = []
    seen = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != split:
                continue
            if row["sample_id"] in seen:
                raise ValueError(f"Duplicate sample_id at line {line_number}")
            seen.add(row["sample_id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in split {split}")
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError(f"Limit must be 1..{len(rows)}")
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}|{row['sample_id']}".encode()).digest())
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    converted = [convert_row(row) for row in selected_rows(args.manifest, args.split, args.limit, args.seed)]
    if args.check_images:
        for row in converted:
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for row in converted:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(converted), "output": str(args.output), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
