#!/usr/bin/env python3
"""Independently validate all QC v2 rows, images and v1.2 gold format (CPU)."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


PRIOR_SHA = "e71df0141dabc9d696cf67e5b47d35b05801633d145d855ca83b7efdbb95637e"
EXCLUDED = {"mammoth_5933982", "mammoth_8467823", "mammoth_7615917"}
ADDED = {"mammoth_9365435", "mammoth_9456721", "mammoth_5001946"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prior", "reward-policy", "code-dir", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.code_dir.resolve(strict=True)))
    from reward import RewardPolicy, score_response

    if sha(args.prior) != PRIOR_SHA:
        raise ValueError("Prior manifest SHA changed")
    old = {x["sample_id"]: x for x in rows(args.prior)}
    new = rows(args.manifest)
    if len(new) != 1100:
        raise ValueError("Expected 1,100 rows")
    counts = Counter((x["source"], x["split"]) for x in new)
    if counts != {("ECD-reasoning", "train"): 500,
                  ("ECD-reasoning", "heldout"): 50,
                  ("clevr_700k_new.json", "train"): 500,
                  ("clevr_700k_new.json", "heldout"): 50}:
        raise ValueError(f"Wrong source/split counts: {counts}")
    ids = {x["original_id"] for x in new}
    if ids & EXCLUDED or not ADDED <= ids:
        raise ValueError("Replacement selection differs")
    for field, key in (("sample_id", lambda x: x["sample_id"]),
                       ("original_id", lambda x: x["original_id"]),
                       ("image_sha256", lambda x: x["image_sha256"]),
                       ("image_path", lambda x: x["image_paths"][0]),
                       ("question", lambda x: x["question"])):
        values = [key(x) for x in new]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field}")
    policy = RewardPolicy.from_file(args.reward_policy)
    if policy.mode != "v1_2":
        raise ValueError("Not frozen v1.2 reward")
    unchanged = 0
    for i, row in enumerate(new, 1):
        if row["original_id"] not in ADDED:
            if row != old.get(row["sample_id"]):
                raise ValueError(f"Original row {i} changed")
            unchanged += 1
        if row.get("c_training_approval") is not False or row.get("visual_gold_quality_verified") is not False:
            raise ValueError(f"Row {i} has unexpected quality approval")
        path = Path(row["image_paths"][0])
        if not path.is_file() or sha(path) != row["image_sha256"]:
            raise ValueError(f"Image missing/hash mismatch at row {i}: {path}")
        if "image_bytes" in row and path.stat().st_size != row["image_bytes"]:
            raise ValueError(f"Image byte size mismatch at row {i}")
        with Image.open(path) as image:
            image.load()
        score = score_response("Answer: " + row["answer"], row["answer"], row["answer_type"], policy,
                               options=list(row.get("choices", {}).values()))
        if score.reward != 1.0:
            raise ValueError(f"Gold self-score failed at row {i}: {score.reason}")
    if unchanged != 1097:
        raise ValueError(f"Expected 1,097 unchanged rows, saw {unchanged}")
    report = {"status": "pass", "manifest_sha256": sha(args.manifest),
              "prior_sha256": PRIOR_SHA, "rows": len(new),
              "unchanged_prior_rows": unchanged, "replacements": sorted(ADDED),
              "image_sha256_decode_checked": len(new), "gold_v12_self_reward_one": len(new),
              "c_training_approval": False, "visual_gold_quality_verified": False}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
