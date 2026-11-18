#!/usr/bin/env python3
"""Build an exploratory ECD manifest after a visual-gold failure audit.

This filters risky question wording and known bad/ambiguous image IDs. It is
NOT an item-level gold approval. Every included image is decoded and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

ECD_800_SHA = "e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f"
ECD_TAR_SHA = "8ae3d9fb5b8345a17f4df420d50b662d5102cc54d46539870b0becad92c09cc1"
AVERAGE = re.compile(r"\b(?:average|mean|median)\b", re.IGNORECASE)
EXCLUDE_CHARTS = {
    "000259": "visual_wrong_gold_region_e_max",
    "008560": "visual_wrong_gold_highlight_boundary",
    "003223": "visual_wrong_gold_matrix_sum_465_vs_500",
    "004222": "visual_gold_unresolved_cumulative_area_values",
    "002406": "prior_visual_wrong_gold_violin_mean",
    "000838": "prior_visual_wrong_gold_inflation_values",
}
SYSTEM = "You are a helpful assistant."
OPEN = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def rank(identifier: str) -> str:
    return hashlib.sha256(f"3407|{identifier}".encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--priority800", type=Path, required=True)
    p.add_argument("--c-provenance", type=Path, required=True)
    p.add_argument("--c-item-status", type=Path, required=True)
    p.add_argument("--ecd-tar", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--old-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha(args.priority800) != ECD_800_SHA:
        raise ValueError("ECD priority800 input SHA mismatch")
    provenance = json.loads(args.c_provenance.read_text(encoding="utf-8"))
    if provenance.get("ecd800_list_sha256") != ECD_800_SHA or provenance.get("ecd_tar_sha256") != ECD_TAR_SHA:
        raise ValueError("C audit provenance mismatch")
    if provenance.get("scope") != "leakage_only_not_quality_or_training_approval":
        raise ValueError("Unexpected C audit scope")
    if sha(args.ecd_tar) != ECD_TAR_SHA:
        raise ValueError("ECD archive SHA mismatch")
    pool = jsonl(args.priority800)
    if len(pool) != 800 or len({x["id"] for x in pool}) != 800:
        raise ValueError("ECD input must have 800 unique IDs")
    audits = {(x["dataset"], x["id"]): x for x in jsonl(args.c_item_status)}
    if len(audits) != 1700:
        raise ValueError("Expected C 1700-item audit")
    old = jsonl(args.old_manifest)
    if len(old) != 600:
        raise ValueError("Expected old 600-item ECD manifest")
    rejects = []
    candidates = []
    for item in pool:
        reason = ("question_average_mean_median" if AVERAGE.search(item["question"]) else None)
        if item["image_id"] in EXCLUDE_CHARTS:
            reason = EXCLUDE_CHARTS[item["image_id"]]
        if reason:
            rejects.append({"id": item["id"], "image_id": item["image_id"], "reason": reason})
            continue
        audit = audits.get(("ecd", item["id"]))
        if audit is None or audit.get("leakage_result") != "not_detected_within_scope":
            raise ValueError(f"C leakage audit not clear: {item['id']}")
        candidates.append(item)
    if len(candidates) < 580:
        raise ValueError(f"Insufficient conservative candidates: {len(candidates)}")
    if len({x["image_id"] for x in candidates}) != len(candidates):
        raise ValueError("Duplicate chart")
    ordered = sorted(candidates, key=lambda x: (rank(x["id"]), x["id"]))
    heldout_count = len(ordered) - 500
    if heldout_count < 80:
        raise ValueError(f"Heldout too small: {heldout_count}")
    held_ids = {x["id"] for x in ordered[:heldout_count]}
    root = args.image_root.resolve(strict=True)
    rows = []
    for item in candidates:
        path = (root / f"{item['image_id']}.png").resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Image path traversal")
        image_sha = sha(path)
        with Image.open(path) as image:
            image.verify()
        q = item["question"].strip()
        rows.append({
            "sample_id": "b_" + hashlib.sha256(f"ECD\0{item['id']}".encode()).hexdigest()[:20],
            "source": "ECD-reasoning",
            "dataset_revision": "ECD-10k-Images-284489f",
            "split": "heldout" if item["id"] in held_ids else "train",
            "original_id": item["id"],
            "image_paths": [str(path)],
            "image_sha256": image_sha,
            "question": q,
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Question: "},
                    {"type": "image"},
                    {"type": "text", "text": "\n" + q + "\n" + OPEN},
                ]},
            ],
            "answer": item["final_answer"].strip(),
            "answer_type": "open", "choices": {},
            "c_leakage_result": "not_detected_within_scope",
            "c_audit_item_file_sha256": sha(args.c_item_status),
            "c_training_approval": False,
            "visual_gold_quality_verified": False,
            "b_filter_version": "ecd_qc_v2",
            "ecd_source_line": item["source_line"],
        })
    if len({x["sample_id"] for x in rows}) != len(rows):
        raise ValueError("Duplicate sample_id")
    old_ids = {x["original_id"] for x in old}
    new_ids = {x["original_id"] for x in rows}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "ecd_manifest.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    (args.output_dir / "excluded.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rejects), encoding="utf-8")
    (args.output_dir / "added_vs_old_ids.txt").write_text("\n".join(sorted(new_ids - old_ids)) + "\n", encoding="utf-8")
    (args.output_dir / "removed_vs_old_ids.txt").write_text("\n".join(sorted(old_ids - new_ids)) + "\n", encoding="utf-8")
    report = {
        "status": "b_exploratory_only_unverified_gold_no_c_training_approval",
        "pool_count": len(pool), "excluded_count": len(rejects),
        "excluded_reasons": dict(Counter(x["reason"] for x in rejects)),
        "train": 500, "heldout": heldout_count,
        "manifest_sha256": sha(manifest),
        "priority800_sha256": ECD_800_SHA,
        "ecd_archive_sha256": ECD_TAR_SHA,
        "c_item_status_sha256": sha(args.c_item_status),
        "c_provenance_sha256": sha(args.c_provenance),
        "old_manifest_sha256": sha(args.old_manifest),
        "added_vs_old": len(new_ids - old_ids), "removed_vs_old": len(old_ids - new_ids),
        "old_overlap": len(new_ids & old_ids),
        "visual_audit": "Hash-fixed 30 non-average sample: 1 confirmed wrong gold, 1 ambiguous; both charts excluded. The other 28 are spot-checked, not a full audit.",
        "gold_quality_claim": "Not gold-verified as a complete set; use only for exploratory signal checks with known label-noise risk.",
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
