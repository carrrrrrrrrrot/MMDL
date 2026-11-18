"""CPU-only preliminary MAmmoTH vs frozen MMMU validation overlap screen.

Checks exact/normalized question strings and 64-bit image dHash distance <= 4.
This is not C's leakage clearance and does not cover MMMU-Pro or other splits.
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


IMAGE_MARKER = re.compile(r"<image\s*\d*>\s*", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    clean = IMAGE_MARKER.sub("", text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", clean).casefold()).strip()


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


def candidate_rows(list_path: Path, image_root: Path) -> list[dict]:
    output = []
    seen = set()
    for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["id"])
        if question_id in seen:
            raise ValueError(f"Duplicate MAmmoTH ID {question_id} at line {line_number}")
        seen.add(question_id)
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError(f"Unsafe image path at line {line_number}: {relative}")
        image_path = (image_root / relative).resolve(strict=True)
        if not image_path.is_relative_to(image_root.resolve(strict=True)):
            raise ValueError(f"Image outside extracted root: {image_path}")
        if image_path.stat().st_size != row["image_bytes"] or sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image size or SHA256 mismatch for {question_id}")
        with Image.open(image_path) as image:
            image_hash = dhash(image)
        output.append({
            "id": question_id, "source": row["source"], "question": row["question"],
            "image": row["image"], "image_sha256": row["image_sha256"],
            "question_normalized": normalize(row["question"]),
            "question_folded": folded(row["question"]), "dhash": image_hash,
        })
    return output


def benchmark_rows(mmmu_root: Path) -> tuple[list[dict], list[dict], int]:
    parquet_files = sorted(mmmu_root.glob("*/validation-*.parquet"))
    if len(parquet_files) != 30:
        raise ValueError(f"Expected 30 MMMU validation Parquets, got {len(parquet_files)}")
    output = []
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
                    decode_errors.append({"id": row["id"], "image_index": index, "error": str(exc)[:120]})
                    continue
                output.append({
                    "id": row["id"], "question": row["question"], "image_index": index,
                    "question_normalized": normalize(row["question"]),
                    "question_folded": folded(row["question"]), "dhash": image_hash,
                })
    return output, decode_errors, len(parquet_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mmmu-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates = candidate_rows(args.candidates, args.image_root)
    benchmark, decode_errors, parquet_count = benchmark_rows(args.mmmu_root)
    exact = []
    punctuation_folded = []
    near_image = []
    for candidate in candidates:
        for reference in benchmark:
            pair = {"mammoth_id": candidate["id"], "mammoth_source": candidate["source"],
                    "mmmu_id": reference["id"], "mmmu_image_index": reference["image_index"]}
            if candidate["question_normalized"] == reference["question_normalized"]:
                exact.append(pair)
            elif candidate["question_folded"] == reference["question_folded"]:
                punctuation_folded.append(pair)
            distance = (candidate["dhash"] ^ reference["dhash"]).bit_count()
            if distance <= 4:
                near_image.append({**pair, "distance": distance,
                                   "mammoth_question": candidate["question"],
                                   "mmmu_question": reference["question"],
                                   "mammoth_image_sha256": candidate["image_sha256"]})
    near_image.sort(key=lambda row: (row["distance"], row["mammoth_id"], row["mmmu_id"]))
    report = {
        "status": "preliminary_overlap_screen_only_not_leakage_clearance",
        "candidate_list_sha256": sha256(args.candidates), "candidate_count": len(candidates),
        "candidate_images_size_and_sha256_verified": len(candidates),
        "mmmu_split": "validation", "mmmu_parquet_files": parquet_count,
        "mmmu_unique_questions": len({row["id"] for row in benchmark}),
        "mmmu_decoded_images": len(benchmark), "mmmu_image_decode_errors": decode_errors,
        "question_exact_pairs": exact, "question_punctuation_folded_pairs": punctuation_folded,
        "dhash_distance_at_most_4_pairs": near_image,
        "not_checked": ["MMMU-Pro", "MMMU dev/test", "OCR text overlap", "semantic question overlap",
                        "manual visual review of near-image pairs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: (len(value) if isinstance(value, list) and key != "not_checked" else value)
                      for key, value in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
