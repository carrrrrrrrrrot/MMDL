"""Check selected ECD image files and make contact sheets for manual review.

This is a read-only data check. It never approves a question for training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=24)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid = []
    invalid = []
    sizes: Counter[str] = Counter()
    for row in rows:
        relative = Path(row["image_relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            invalid.append({"id": row["id"], "reason": "invalid_relative_path"})
            continue
        path = args.image_root / relative
        try:
            with Image.open(path) as pic:
                pic.load()
                size = pic.size
                if size[0] < 64 or size[1] < 64:
                    raise ValueError(f"image_too_small:{size}")
            sizes[f"{size[0]}x{size[1]}"] += 1
            valid.append(row)
        except (OSError, ValueError) as exc:
            invalid.append({"id": row["id"], "reason": str(exc)[:200]})

    # Sample throughout the ranked list; include the first rows, which are easiest
    # for a teammate to locate when checking the source metadata.
    sample_count = min(max(args.sample, 0), len(valid))
    indices = sorted(set(range(min(8, sample_count))) | {
        round(i * (len(valid) - 1) / max(sample_count - 1, 1)) for i in range(sample_count)
    })
    sampled = [valid[i] for i in indices]
    sheet_paths = []
    font = ImageFont.load_default()
    tile_w, tile_h = 800, 620
    for start in range(0, len(sampled), 4):
        group = sampled[start : start + 4]
        sheet = Image.new("RGB", (tile_w * 2, tile_h * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(group):
            x, y = (offset % 2) * tile_w, (offset // 2) * tile_h
            path = args.image_root / row["image_relative_path"]
            with Image.open(path) as original:
                pic = ImageOps.contain(original.convert("RGB"), (tile_w - 16, 455))
            sheet.paste(pic, (x + (tile_w - pic.width) // 2, y + 6))
            draw.rectangle((x, y + 460, x + tile_w - 1, y + tile_h - 1), outline="black", width=1)
            label = f"{row['id']}  gold={row['final_answer']}  SFT1={row['used_in_sft1']}  DPO={row['in_dpo_pool']}"
            draw.text((x + 8, y + 468), label[:120], fill="black", font=font)
            question = row["question"]
            for line_number in range(5):
                chunk = question[line_number * 102 : (line_number + 1) * 102]
                if not chunk:
                    break
                draw.text((x + 8, y + 490 + line_number * 21), chunk, fill="black", font=font)
        sheet_path = args.output_dir / f"contact_{start // 4 + 1:02d}.jpg"
        sheet.save(sheet_path, quality=90)
        sheet_paths.append(str(sheet_path))

    report = {
        "status": "image_preflight_only_not_approved_for_training",
        "input_count": len(rows),
        "valid_image_count": len(valid),
        "invalid_image_count": len(invalid),
        "invalid": invalid,
        "image_sizes_top10": sizes.most_common(10),
        "sampled_ids": [row["id"] for row in sampled],
        "contact_sheets": sheet_paths,
    }
    (args.output_dir / "image_preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
