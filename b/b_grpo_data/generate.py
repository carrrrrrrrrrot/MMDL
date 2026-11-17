"""Generate a leakage-free synthetic visual-reasoning GRPO fallback set.

These examples are procedurally generated and contain no MMMU/MMMU-Pro source
material. They are an independent base->GRPO insurance experiment, not a
substitute for C's checked external-data manifest or the common-parent run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


SYSTEM = "You are a helpful assistant."
LETTERS = "ABCD"
TASKS = ("count", "bar", "spatial")
SPLIT_BLOCK = 32
HELDOUT_POSITIONS = {0, 9, 18, 27}  # one of each answer letter per block


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _options(gold: str, distractors: list[str], answer_slot: int, rng: random.Random) -> dict[str, str]:
    if len(set(distractors + [gold])) != 4:
        raise ValueError("Choices must have four distinct values")
    rng.shuffle(distractors)
    values = distractors[:]
    values.insert(answer_slot, gold)
    return dict(zip(LETTERS, values, strict=True))


def _count(rng: random.Random, answer_slot: int) -> tuple[Image.Image, str, dict[str, str]]:
    red_count = rng.randint(3, 13)
    red_cells = set(rng.sample(range(16), red_count))
    image = Image.new("RGB", (344, 344), "white")
    draw = ImageDraw.Draw(image)
    for cell in range(16):
        row, col = divmod(cell, 4)
        x, y = 12 + col * 80, 12 + row * 80
        fill = (211, 52, 52) if cell in red_cells else (48, 94, 187)
        draw.rectangle((x, y, x + 72, y + 72), fill=fill, outline="black", width=2)
    choices = _options(str(red_count), [str(red_count - 2), str(red_count - 1), str(red_count + 1)], answer_slot, rng)
    return image, "How many red squares are in the image?", choices


def _bar(rng: random.Random, answer_slot: int) -> tuple[Image.Image, str, dict[str, str]]:
    # Force the answer location to rotate evenly across A-D.
    heights = sorted(rng.sample(range(4, 15), 4))
    rng.shuffle(heights)
    highest = max(heights)
    highest_at = heights.index(highest)
    heights[highest_at], heights[answer_slot] = heights[answer_slot], heights[highest_at]
    image = Image.new("RGB", (380, 330), "white")
    draw = ImageDraw.Draw(image)
    draw.line((35, 275, 355, 275), fill="black", width=3)
    colors = ((76, 123, 205), (225, 126, 60), (80, 170, 95), (165, 96, 182))
    for index, value in enumerate(heights):
        x = 54 + index * 78
        top = 275 - value * 16
        draw.rectangle((x, top, x + 48, 274), fill=colors[index], outline="black", width=2)
        draw.text((x + 18, 283), LETTERS[index], fill="black")
    choices = {letter: f"Bar {letter}" for letter in LETTERS}
    return image, "Which labeled bar is the tallest?", choices


def _star_points(cx: float, cy: float, outer: float = 23, inner: float = 10) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _spatial(rng: random.Random, answer_slot: int) -> tuple[Image.Image, str, dict[str, str]]:
    star_cell = rng.randrange(16)
    distractor_cells = rng.sample([cell for cell in range(16) if cell != star_cell], 3)
    choices = _options(
        _cell_name(star_cell), [_cell_name(cell) for cell in distractor_cells], answer_slot, rng
    )
    circle_cells = rng.sample([cell for cell in range(16) if cell != star_cell], 6)
    image = Image.new("RGB", (344, 344), "white")
    draw = ImageDraw.Draw(image)
    for cell in range(16):
        row, col = divmod(cell, 4)
        left, top = 12 + col * 80, 12 + row * 80
        draw.rectangle((left, top, left + 72, top + 72), outline="black", width=2)
        cx, cy = left + 36, top + 36
        if cell == star_cell:
            draw.polygon(_star_points(cx, cy), fill=(245, 195, 39), outline="black")
        elif cell in circle_cells:
            draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=(53, 112, 190), outline="black", width=2)
    return image, "In which cell is the yellow star? Count rows from top and columns from left.", choices


def _cell_name(cell: int) -> str:
    row, col = divmod(cell, 4)
    return f"row {row + 1}, column {col + 1}"


def _render(task: str, rng: random.Random, answer_slot: int) -> tuple[Image.Image, str, dict[str, str]]:
    return {"count": _count, "bar": _bar, "spatial": _spatial}[task](rng, answer_slot)


def generate(output_dir: Path, per_task: int, seed: int) -> dict:
    if per_task <= 0 or per_task % SPLIT_BLOCK:
        raise ValueError(f"--per-task must be a positive multiple of {SPLIT_BLOCK}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = f"synthetic-v1-seed{seed}-n{per_task}"
    images_dir = output_dir / "images"
    images_dir.mkdir()
    manifest = output_dir / "manifest.jsonl"
    seen_images: set[str] = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    with manifest.open("w", encoding="utf-8") as stream:
        for task_index, task in enumerate(TASKS):
            for index in range(per_task):
                sample_id = f"synth_{task}_{index:04d}"
                answer_slot = index % 4
                split = "heldout" if index % SPLIT_BLOCK in HELDOUT_POSITIONS else "train"
                image_path = images_dir / f"{sample_id}.png"
                for salt in range(100):
                    item_seed = seed + task_index * 1_000_003 + index * 7919 + salt
                    image, question, choices = _render(task, random.Random(item_seed), answer_slot)
                    image.save(image_path, format="PNG")
                    image_hash = _hash(image_path)
                    if image_hash not in seen_images:
                        seen_images.add(image_hash)
                        break
                else:
                    raise RuntimeError(f"Could not generate a unique image for {sample_id}")
                answer = LETTERS[answer_slot]
                choices_text = "\n".join(f"{letter}. {choices[letter]}" for letter in LETTERS)
                user_text = (
                    f"{question}\n{choices_text}\n"
                    "Explain briefly. End with exactly one final line in this form: Answer: X"
                )
                row = {
                    "sample_id": sample_id,
                    "original_id": sample_id,
                    "source": "synthetic_generated",
                    "dataset_revision": revision,
                    "split": split,
                    "task": task,
                    "image_path": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                    "image_sha256": image_hash,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "answer_type": "mc",
                    "prompt": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_text},
                    ],
                    "generator_seed": item_seed,
                }
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[(task, split, answer)] += 1
    metadata = {
        "purpose": "base-to-GRPO insurance experiment; not common-parent final comparison",
        "source": "synthetic_generated",
        "revision": revision,
        "master_seed": seed,
        "per_task": per_task,
        "total_items": per_task * len(TASKS),
        "manifest_sha256": _hash(manifest),
        "counts_by_task_split_answer": {"/".join(key): value for key, value in sorted(counts.items())},
        "note": "No MMMU or MMMU-Pro questions, answers, screenshots, or images were used.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-task", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    metadata = generate(args.output_dir, args.per_task, args.seed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
