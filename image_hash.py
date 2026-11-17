import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps


def image_hash(data):
    with Image.open(io.BytesIO(data)) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        pixel = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
        gray = rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        values = list(gray.getdata())
        bits = sum((values[y*9+x] > values[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'pixel_sha256': pixel, 'dhash': f'{bits:016x}'}


def main():
    parser = argparse.ArgumentParser(description="Export SFT1 image hashes for the leakage audit notebook.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--ecd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sft1_image_hashes.jsonl"))
    args = parser.parse_args()
    roots = {"mammoth": args.mammoth_root, "ecd": args.ecd_root}
    errors = args.out.with_suffix(".errors.jsonl")
    if args.manifest.resolve() in {args.out.resolve(), errors.resolve()} or args.out == errors:
        parser.error("Use separate input, output and error paths.")
    cache = {}
    exported = failed = 0
    with args.manifest.open(encoding="utf-8-sig") as source, args.out.open(
        "w", encoding="utf-8"
    ) as out, errors.open("w", encoding="utf-8") as error_log:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = {field: row[field] for field in ("dataset", "idx", "id")}
            root = roots[row["dataset"]].resolve()
            path = (root / row["image"]).resolve()
            try:
                if not path.is_relative_to(root):
                    raise ValueError("Image path is outside its dataset root")
                if path not in cache:
                    cache[path] = image_hash(path.read_bytes())
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                error_log.write(json.dumps({**identity, "image": row["image"], "error": str(exc)}) + "\n")
                failed += 1
                continue
            out.write(json.dumps({**identity, "hash": cache[path]}) + "\n")
            exported += 1
    print(f"Exported {exported} rows to {args.out}. Failed: {failed}. Error log: {errors}")


if __name__ == "__main__":
    main()
