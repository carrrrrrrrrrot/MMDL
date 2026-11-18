"""CPU-only preliminary ECD vs MMMU validation exact/dHash overlap screen.

This does not establish benchmark non-leakage. C must review image candidates and
also inspect MMMU-Pro, OCR content, and semantic question overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def folded(text: str) -> str:
    return re.sub(r"[^\w]+", "", normalize(text), flags=re.UNICODE)


def dhash(image: Image.Image) -> int:
    small = ImageOps.grayscale(ImageOps.exif_transpose(image)).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--mmmu-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    ecd = []
    for row in selected:
        path = args.image_root / row["image_relative_path"]
        with Image.open(path) as image:
            image_hash = dhash(image)
        ecd.append({"id": row["id"], "question": row["question"], "image_relative_path": row["image_relative_path"],
                    "dhash": image_hash, "question_normalized": normalize(row["question"]), "question_folded": folded(row["question"])})

    parquet_files = sorted(args.mmmu_root.glob("*/validation-*.parquet"))
    if len(parquet_files) != 30:
        raise ValueError(f"Expected 30 MMMU validation files, got {len(parquet_files)}")
    benchmark = []
    decode_errors = []
    for parquet in parquet_files:
        table = pq.read_table(parquet, columns=["id", "question", *[f"image_{i}" for i in range(1, 8)]])
        for row in table.to_pylist():
            for index in range(1, 8):
                entry = row[f"image_{index}"]
                if not entry or not entry.get("bytes"):
                    continue
                try:
                    with Image.open(io.BytesIO(entry["bytes"])) as image:
                        image_hash = dhash(image)
                except (OSError, ValueError) as exc:
                    decode_errors.append({"id": row["id"], "image_index": index, "reason": str(exc)[:120]})
                    continue
                benchmark.append({"id": row["id"], "question": row["question"], "image_index": index,
                                  "dhash": image_hash, "question_normalized": normalize(row["question"]),
                                  "question_folded": folded(row["question"])})

    exact_question = []
    folded_question = []
    near_image = []
    for candidate in ecd:
        for reference in benchmark:
            if candidate["question_normalized"] == reference["question_normalized"]:
                exact_question.append({"ecd_id": candidate["id"], "mmmu_id": reference["id"], "mmmu_image_index": reference["image_index"]})
            elif candidate["question_folded"] == reference["question_folded"]:
                folded_question.append({"ecd_id": candidate["id"], "mmmu_id": reference["id"], "mmmu_image_index": reference["image_index"]})
            distance = (candidate["dhash"] ^ reference["dhash"]).bit_count()
            if distance <= 4:
                near_image.append({"ecd_id": candidate["id"], "mmmu_id": reference["id"],
                                   "mmmu_image_index": reference["image_index"], "distance": distance,
                                   "ecd_question": candidate["question"], "mmmu_question": reference["question"]})
    near_image.sort(key=lambda row: (row["distance"], row["ecd_id"], row["mmmu_id"]))
    report = {
        "status": "preliminary_overlap_screen_only_not_leakage_clearance",
        "ecd_candidates_sha256": sha256(args.candidates),
        "ecd_candidates": len(ecd),
        "mmmu_split": "validation",
        "mmmu_parquet_files": len(parquet_files),
        "mmmu_unique_questions": len({row["id"] for row in benchmark}),
        "mmmu_decoded_images": len(benchmark),
        "mmmu_image_decode_errors": decode_errors,
        "question_exact_pairs": exact_question,
        "question_punctuation_folded_pairs": folded_question,
        "dhash_distance_at_most_4_pairs": near_image,
        "not_checked": ["MMMU-Pro", "MMMU dev/test", "OCR text overlap", "semantic question overlap", "manual visual review of near-image pairs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: (len(value) if isinstance(value, list) and key not in {"not_checked"} else value)
                      for key, value in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
