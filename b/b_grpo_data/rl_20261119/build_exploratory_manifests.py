#!/usr/bin/env python3
"""Build B-provisional ECD/MAmmoTH manifests from C's leakage-only audit.

This never represents C's leakage result as image/gold quality or training
approval. Every selected image is decoded and checked against the audit SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ECD_SHA = "e05ae72d0e30bf79708cccd51b66b3a2be1ed48d4d69f77e5f0280ff00c7479f"
MAMMOTH_SHA = "e2ac13c827a125714fabf27c8f1dcef9bce35135438bff57991ad596ce8853c8"
A_COMMIT = "a61adb899d3c2274ba98cba7a360c5b4f333979f"
SYSTEM = "You are a helpful assistant."
OPEN = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'
BAD_QUESTION = re.compile(r"<image\s*\d*>|^\s*Answer\s*:", re.IGNORECASE | re.MULTILINE)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def rank(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}|{identifier}".encode()).hexdigest()


def valid_question(question: object) -> bool:
    return isinstance(question, str) and bool(question.strip()) and not BAD_QUESTION.search(question)


def decoded_image(path: Path, expected_sha: str | None, expected_bytes: int | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"Image missing: {path}")
    actual_sha = digest(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError(f"Image SHA mismatch: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Image byte-size mismatch: {path}")
    with Image.open(path) as image:
        image.verify()
    return actual_sha


def split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: (rank(seed, row["id"]), row["id"]))
    held = ordered[:100]
    train = ordered[100:]
    if len(train) != 500 or len(held) != 100:
        raise ValueError("Expected exactly 500 train + 100 heldout")
    return train, held


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ecd-list", required=True, type=Path)
    p.add_argument("--mammoth900-list", required=True, type=Path)
    p.add_argument("--c-item-status", required=True, type=Path)
    p.add_argument("--c-provenance", required=True, type=Path)
    p.add_argument("--ecd-image-root", required=True, type=Path)
    p.add_argument("--ecd-tar", required=True, type=Path)
    p.add_argument("--mammoth-root", required=True, type=Path)
    p.add_argument("--mammoth-converter-dir", required=True, type=Path)
    p.add_argument("--b-main", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    if digest(args.ecd_list) != ECD_SHA or digest(args.mammoth900_list) != MAMMOTH_SHA:
        raise ValueError("B candidate list SHA differs from C-audited input")
    provenance = json.loads(args.c_provenance.read_text(encoding="utf-8"))
    if provenance.get("ecd800_list_sha256") != ECD_SHA or provenance.get("mammoth900_list_sha256") != MAMMOTH_SHA:
        raise ValueError("C input provenance differs from B candidate lists")
    if provenance.get("scope") != "leakage_only_not_quality_or_training_approval":
        raise ValueError("Unexpected C audit scope")
    if digest(args.ecd_tar) != provenance.get("ecd_tar_sha256"):
        raise ValueError("B ECD image archive SHA differs from C input provenance")
    ecd = read_jsonl(args.ecd_list)
    mammoth = read_jsonl(args.mammoth900_list)
    audits = read_jsonl(args.c_item_status)
    expected = {("ecd", r["id"]) for r in ecd} | {("mammoth", r["id"]) for r in mammoth}
    observed = {(r["dataset"], r["id"]) for r in audits}
    if len(ecd) != 800 or len(mammoth) != 900 or len(audits) != 1700 or len(observed) != 1700 or observed != expected:
        raise ValueError("C audit IDs do not exactly match B's 800+900 candidate lists")
    audit = {(r["dataset"], r["id"]): r for r in audits}
    for row in audits:
        if row.get("leakage_audit_status") != "complete_within_documented_scope" or row.get("leakage_result") != "not_detected_within_scope":
            raise ValueError(f"C leakage audit not clear for {row['dataset']}:{row['id']}")

    sys.path.insert(0, str(args.mammoth_converter_dir))
    sys.path.insert(0, str(args.b_main))
    from convert_mammoth_handoff import Approval, make_row
    from reward import validate_gold

    ecd_candidates = []
    ecd_reject = Counter()
    seen_charts = set()
    for row in ecd:
        question, answer = row.get("question"), row.get("final_answer")
        if not valid_question(question):
            ecd_reject["question_format"] += 1
            continue
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 40:
            ecd_reject["gold_format"] += 1
            continue
        try:
            validate_gold(answer, "open")
        except ValueError:
            ecd_reject["gold_format"] += 1
            continue
        relative = Path(row["image_relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            ecd_reject["image_path"] += 1
            continue
        chart = relative.stem
        if chart in seen_charts:
            ecd_reject["duplicate_chart"] += 1
            continue
        seen_charts.add(chart)
        ecd_candidates.append(row)
    if len(ecd_candidates) < 600:
        raise ValueError(f"Only {len(ecd_candidates)} structurally eligible ECD rows")
    ecd_selected = ecd_candidates[:600]

    groups = {"unused": [], "dpo_pairs": [], "dpo_pool": []}
    mammoth_reject = Counter()
    for row in mammoth:
        if not valid_question(row.get("question")) or row.get("text_only_suspect"):
            mammoth_reject["question_or_text_only"] += 1
            continue
        if row["answer_type"] not in {"mc", "numeric", "word"}:
            mammoth_reject["answer_type"] += 1
            continue
        used = set(row["used_in"])
        if not used:
            groups["unused"].append(row)
        elif "dpo_pairs" in used:
            groups["dpo_pairs"].append(row)
        elif used == {"dpo_pool"}:
            groups["dpo_pool"].append(row)
        else:
            mammoth_reject["usage_outside_policy"] += 1
    quotas = {"unused": 369, "dpo_pairs": 100, "dpo_pool": 131}
    mammoth_selected = []
    for name, quota in quotas.items():
        ordered = sorted(groups[name], key=lambda row: (rank(args.seed, row["id"]), row["id"]))
        if len(ordered) < quota:
            raise ValueError(f"Only {len(ordered)} rows in {name}, need {quota}")
        mammoth_selected.extend(ordered[:quota])
    if len(mammoth_selected) != 600:
        raise AssertionError("MAmmoTH selection count")
    e_train, e_held = split_rows(ecd_selected, args.seed)
    m_train, m_held = split_rows(mammoth_selected, args.seed)

    c_hash = digest(args.c_item_status)
    ecd_output = []
    mammoth_output = []
    for split, rows in (("train", e_train), ("heldout", e_held)):
        for row in rows:
            info = audit[("ecd", row["id"])]
            relative = Path(row["image_relative_path"])
            path = (args.ecd_image_root.parent / relative).resolve(strict=True)
            if not path.is_relative_to(args.ecd_image_root.resolve(strict=True)):
                raise ValueError("ECD image path traversal")
            image_sha = decoded_image(path, info["image_sha256"])
            question = row["question"].strip()
            answer = row["final_answer"].strip()
            ecd_output.append({
                "sample_id": "b_" + hashlib.sha256(f"ECD\0{row['id']}".encode()).hexdigest()[:20],
                "source": "ECD-reasoning", "dataset_revision": "ECD-10k-Images-284489f",
                "split": split, "original_id": row["id"], "image_paths": [str(path)],
                "image_sha256": image_sha, "question": question,
                "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                    {"type": "text", "text": "Question: "}, {"type": "image"},
                    {"type": "text", "text": "\n" + question + "\n" + OPEN},
                ]}],
                "answer": answer, "answer_type": "open", "choices": {},
                "c_leakage_audit_status": info["leakage_audit_status"],
                "c_leakage_result": info["leakage_result"], "c_audit_item_file_sha256": c_hash,
                "c_training_approval": False, "b_exploratory_structural_image_qc": True,
                "ecd_archive_matches_c_provenance": True,
                "c_per_item_image_sha_available": False,
                "visual_gold_quality_verified": False,
                "ecd_source_line": row.get("source_line"),
            })
    for split, rows in (("train", m_train), ("heldout", m_held)):
        for row in rows:
            info = audit[("mammoth", row["id"])]
            if row["image_sha256"] != info["image_sha256"]:
                raise ValueError(f"MAmmoTH image SHA differs from C audit: {row['id']}")
            rendered = make_row(
                row, Approval(row["source"], row["id"], split, "B provisional"),
                args.mammoth_root.resolve(strict=True), "mammoth-vl-shard1-a61adb899d3c", A_COMMIT, c_hash,
            )
            image = Path(rendered["image_paths"][0])
            decoded_image(image, info["image_sha256"], row["image_bytes"])
            for key in ("c_approved", "c_reviewer", "c_mmmu_leakage", "c_quality", "c_approval_sha256"):
                rendered.pop(key, None)
            rendered.update({
                "c_leakage_audit_status": info["leakage_audit_status"],
                "c_leakage_result": info["leakage_result"], "c_audit_item_file_sha256": c_hash,
                "c_training_approval": False, "b_exploratory_structural_image_qc": True,
                "visual_gold_quality_verified": False,
                "a_used_in_dpo_pair": "dpo_pairs" in row["used_in"],
                "a_used_in_dpo_pool": "dpo_pool" in row["used_in"],
            })
            mammoth_output.append(rendered)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ecd_mixed_train = sorted(
        (row for row in ecd_output if row["split"] == "train"),
        key=lambda row: (rank(args.seed + 1, row["original_id"]), row["original_id"]),
    )[:250]
    mammoth_mixed_train = sorted(
        (row for row in mammoth_output if row["split"] == "train"),
        key=lambda row: (rank(args.seed + 1, row["original_id"]), row["original_id"]),
    )[:250]
    ecd_mixed_held = sorted(
        (row for row in ecd_output if row["split"] == "heldout"),
        key=lambda row: (rank(args.seed + 2, row["original_id"]), row["original_id"]),
    )[:50]
    mammoth_mixed_held = sorted(
        (row for row in mammoth_output if row["split"] == "heldout"),
        key=lambda row: (rank(args.seed + 2, row["original_id"]), row["original_id"]),
    )[:50]
    mixed_output = ecd_mixed_train + mammoth_mixed_train + ecd_mixed_held + mammoth_mixed_held
    if len({row["sample_id"] for row in mixed_output}) != 600:
        raise ValueError("Mixed manifest has duplicate sample IDs")
    for name, rows in (("ecd", ecd_output), ("mammoth", mammoth_output), ("mixed", mixed_output)):
        target = args.output_dir / f"{name}_manifest.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "status": "b_exploratory_qc_leakage_only_no_c_training_approval",
        "c_audit_item_sha256": c_hash, "c_provenance_sha256": digest(args.c_provenance),
        "ecd_tar_sha256": provenance["ecd_tar_sha256"],
        "ecd_c_per_item_image_sha_available": False,
        "ecd_input_sha256": ECD_SHA, "mammoth900_input_sha256": MAMMOTH_SHA,
        "seed": args.seed, "ecd_structural_rejects": dict(ecd_reject),
        "mammoth_structural_rejects": dict(mammoth_reject),
        "mammoth_selection_quota": quotas,
        "ecd_count": len(ecd_output), "ecd_train": len(e_train), "ecd_heldout": len(e_held),
        "mammoth_count": len(mammoth_output), "mammoth_train": len(m_train), "mammoth_heldout": len(m_held),
        "mammoth_source_counts": dict(Counter(row["source"] for row in mammoth_output)),
        "mammoth_used_in_counts": dict(Counter("+".join(row["a_used_in"]) or "unused" for row in mammoth_output)),
        "mammoth_answer_types": dict(Counter(row["answer_type"] for row in mammoth_output)),
        "mixed_train": 500, "mixed_heldout": 100,
        "mixed_train_sources": {"ecd": 250, "mammoth": 250},
        "mixed_heldout_sources": {"ecd": 50, "mammoth": 50},
        "ecd_manifest_sha256": digest(args.output_dir / "ecd_manifest.jsonl"),
        "mammoth_manifest_sha256": digest(args.output_dir / "mammoth_manifest.jsonl"),
        "mixed_manifest_sha256": digest(args.output_dir / "mixed_manifest.jsonl"),
        "visual_gold_quality_verified": False,
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
