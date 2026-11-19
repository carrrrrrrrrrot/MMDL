#!/usr/bin/env python3
"""CPU-only source, image, and v1.2 gold-format check for mixed1000 v1."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


SOURCE_SHA = {
    "ecd": "c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed",
    "clevr": "b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "ecd", "clevr", "reward-policy", "code-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_dir.resolve(strict=True)))
    from reward import RewardPolicy, score_response

    policy = RewardPolicy.from_file(args.reward_policy)
    if policy.mode != "v1_2":
        raise ValueError("Expected frozen v1.2 reward policy")

    source_rows = {}
    for name in ("ecd", "clevr"):
        path = getattr(args, name)
        if sha256(path) != SOURCE_SHA[name]:
            raise ValueError(f"{name} input SHA256 differs from pinned manifest")
        for row in lines(path):
            key = row["sample_id"]
            if key in source_rows:
                raise ValueError(f"Source sample_id duplicate: {key}")
            source_rows[key] = row

    rows = lines(args.manifest)
    if len(rows) != 1100:
        raise ValueError(f"Expected 1100 rows; got {len(rows)}")
    counts = Counter((row["source"], row["split"]) for row in rows)
    expected = {
        ("ECD-reasoning", "train"): 500,
        ("ECD-reasoning", "heldout"): 50,
        ("clevr_700k_new.json", "train"): 500,
        ("clevr_700k_new.json", "heldout"): 50,
    }
    if dict(counts) != expected:
        raise ValueError(f"Unexpected source/split counts: {counts}")
    for field, values in {
        "sample_id": [row["sample_id"] for row in rows],
        "original_id": [row["original_id"] for row in rows],
        "question": [row["question"] for row in rows],
        "image_sha256": [row["image_sha256"] for row in rows],
        "image_path": [row["image_paths"][0] for row in rows],
    }.items():
        if len(set(values)) != len(rows):
            raise ValueError(f"Duplicate {field} in mixed manifest")

    for index, row in enumerate(rows, 1):
        if row != source_rows.get(row["sample_id"]):
            raise ValueError(f"Row {index} differs from pinned source")
        if row.get("c_training_approval") is not False or row.get("visual_gold_quality_verified") is not False:
            raise ValueError(f"Row {index} has unexpected C/gold quality approval")
        image = Path(row["image_paths"][0])
        if not image.is_file() or sha256(image) != row["image_sha256"]:
            raise ValueError(f"Row {index} image missing or SHA256 mismatch: {image}")
        if "image_bytes" in row and image.stat().st_size != row["image_bytes"]:
            raise ValueError(f"Row {index} image size mismatch: {image}")
        with Image.open(image) as decoded:
            decoded.load()
        outcome = score_response("Answer: " + row["answer"], row["answer"], row["answer_type"], policy,
                                 options=list(row.get("choices", {}).values()))
        if outcome.reward != 1.0:
            raise ValueError(f"Row {index} gold has no v1.2 self-reward: {outcome.reason}")

    report = {
        "status": "pass",
        "manifest_sha256": sha256(args.manifest),
        "source_sha256": SOURCE_SHA,
        "rows": len(rows),
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "heldout")},
        "source_split_counts": {f"{source}/{split}": value for (source, split), value in sorted(counts.items())},
        "unique_ids_questions_images": len(rows),
        "source_rows_identical": len(rows),
        "image_sha256_decode_checked": len(rows),
        "gold_v12_self_reward_one": len(rows),
        "c_training_approval": False,
        "visual_gold_quality_verified": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
