#!/usr/bin/env python3
"""Convert C-approved rows from A's MAmmoTH handoff into a B GRPO manifest.

This consumes A's *selection list*, not A's convert_v12.py SFT output. It never
selects or approves training rows: an explicit C approval and leakage pass are
required for each emitted row. It does not run training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM = "You are a helpful assistant."
MC_INSTRUCTION = (
    'Think step by step, then give your final answer on the last line in the form '
    '"Answer: X", where X is the option letter.'
)
OPEN_INSTRUCTION = (
    'Think step by step, then give your final answer on the last line in the form '
    '"Answer: <answer>".'
)
IMAGE_MARKER = re.compile(r"<image\s*\d*>", re.IGNORECASE)
ANSWER_LINE = re.compile(r"(?im)^\s*Answer\s*:")
COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Approval:
    source: str
    original_id: str
    split: str
    reviewer: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def read_approvals(path: Path) -> dict[tuple[str, str], Approval]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
    else:
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not entries:
        raise ValueError("C approval file is empty")
    approved: dict[tuple[str, str], Approval] = {}
    seen: set[tuple[str, str]] = set()
    for number, item in enumerate(entries, 1):
        if not isinstance(item, dict):
            raise ValueError(f"C approval row {number} is not an object")
        source = _nonempty(item.get("source"), f"approval row {number} source")
        original_id = _nonempty(item.get("id", item.get("original_id")), f"approval row {number} id")
        key = source, original_id
        if key in seen:
            raise ValueError(f"Duplicate C review for {key}")
        seen.add(key)
        decision = item.get("approval", item.get("decision"))
        if decision not in {"approved", "rejected"}:
            raise ValueError(f"C review {key}: approval must be approved or rejected")
        if decision == "rejected":
            continue
        if item.get("mmmu_leakage") != "pass":
            raise ValueError(f"C review {key}: MMMU leakage audit must be pass")
        if item.get("quality") != "pass":
            raise ValueError(f"C review {key}: quality review must be pass")
        split = item.get("split")
        if split not in {"train", "heldout"}:
            raise ValueError(f"C review {key}: split must be train or heldout")
        reviewer = _nonempty(item.get("reviewer"), f"C review {key} reviewer")
        approved[key] = Approval(source, original_id, split, reviewer)
    if not approved:
        raise ValueError("C approval file contains no approved rows")
    return approved


def quality_issues(row: dict[str, Any]) -> list[str]:
    question = row.get("question")
    if not isinstance(question, str):
        return ["invalid_question"]
    issues = []
    if IMAGE_MARKER.search(question):
        issues.append("residual_image_marker")
    if ANSWER_LINE.search(question):
        issues.append("embedded_answer_line")
    return issues


def image_path(row: dict[str, Any], image_root: Path) -> Path:
    raw = _nonempty(row.get("image"), "image")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
        raise ValueError(f"Unsafe A image path: {raw}")
    path = (image_root / relative).resolve(strict=True)
    if not path.is_relative_to(image_root) or not path.is_file():
        raise ValueError(f"A image is outside the extracted root or missing: {raw}")
    expected = _nonempty(row.get("image_sha256"), "image_sha256").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(path) != expected:
        raise ValueError(f"A image SHA256 mismatch: {raw}")
    size = row.get("image_bytes")
    if not isinstance(size, int) or size != path.stat().st_size:
        raise ValueError(f"A image byte count mismatch: {raw}")
    return path


def make_row(
    raw: dict[str, Any], approval: Approval, image_root: Path,
    dataset_revision: str, handoff_commit: str, approval_sha256: str,
) -> dict[str, Any]:
    original_id = _nonempty(raw.get("id"), "id")
    source = _nonempty(raw.get("source"), "source")
    if (source, original_id) != (approval.source, approval.original_id):
        raise ValueError(f"Approval does not match A row: {(source, original_id)}")
    issues = quality_issues(raw)
    if issues:
        raise ValueError(f"A row {original_id} requires C source correction: {issues}")
    question = _nonempty(raw.get("question"), "question")
    answer = _nonempty(raw.get("answer"), "answer")
    kind = raw.get("answer_type")
    if kind not in {"mc", "numeric", "word"}:
        raise ValueError(f"A row {original_id}: unsupported answer_type {kind!r}")
    if len(answer) > 40:
        raise ValueError(f"A row {original_id}: final answer exceeds 40 characters")
    path = image_path(raw, image_root)
    options = raw.get("options")
    choices: dict[str, str] = {}
    if kind == "mc":
        if not isinstance(options, list) or not 2 <= len(options) <= 5:
            raise ValueError(f"A row {original_id}: MC requires 2–5 options")
        if not all(isinstance(item, str) and item.strip() for item in options):
            raise ValueError(f"A row {original_id}: MC option is empty")
        choices = {chr(ord("A") + index): item.strip() for index, item in enumerate(options)}
        if answer not in choices:
            raise ValueError(f"A row {original_id}: MC gold is outside its options")
        option_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
        text_after_image = f"\n{question}\nOptions:\n{option_text}\n{MC_INSTRUCTION}"
        answer_type = "mc"
    else:
        if options is not None:
            raise ValueError(f"A row {original_id}: open answer unexpectedly has options")
        text_after_image = f"\n{question}\n{OPEN_INSTRUCTION}"
        answer_type = "open"  # frozen v1.2 handles both numeric and short word gold
    used_in = raw.get("used_in")
    if not isinstance(used_in, list) or not all(isinstance(item, str) and item for item in used_in):
        raise ValueError(f"A row {original_id}: invalid used_in lineage")
    if not isinstance(raw.get("text_only_suspect"), bool):
        raise ValueError(f"A row {original_id}: invalid text_only_suspect flag")
    sample_id = "a_" + hashlib.sha256(f"{source}\0{original_id}".encode()).hexdigest()[:20]
    return {
        "sample_id": sample_id,
        "source": source,
        "dataset_revision": dataset_revision,
        "split": approval.split,
        "original_id": original_id,
        "image_paths": [str(path)],
        "image_sha256": raw["image_sha256"],
        "question": question,
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Question: "},
                {"type": "image"},
                {"type": "text", "text": text_after_image},
            ]},
        ],
        "answer": answer,
        "answer_type": answer_type,
        "choices": choices,
        "a_answer_type": kind,
        "a_idx": raw.get("idx"),
        "a_used_in": used_in,
        "a_base_k_correct": raw.get("base_k_correct"),
        "a_text_only_suspect": raw["text_only_suspect"],
        "a_part": raw.get("part"),
        "a_handoff_commit": handoff_commit,
        "c_approved": True,
        "c_reviewer": approval.reviewer,
        "c_mmmu_leakage": "pass",
        "c_quality": "pass",
        "c_approval_sha256": approval_sha256,
    }


def convert(
    source: Path, approved_path: Path, image_root: Path, output: Path,
    dataset_revision: str, handoff_commit: str,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    approved_path = approved_path.resolve(strict=True)
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError("Image root must be a directory")
    if output.exists() or output.with_suffix(output.suffix + ".report.json").exists():
        raise FileExistsError("Choose fresh manifest and report output paths")
    dataset_revision = _nonempty(dataset_revision, "dataset_revision")
    if not COMMIT.fullmatch(handoff_commit):
        raise ValueError("handoff_commit must be the full 40-character Git commit")
    approvals = read_approvals(approved_path)
    approval_hash = sha256(approved_path)
    rows = []
    flagged = []
    matched: set[tuple[str, str]] = set()
    seen_source_keys: set[tuple[str, str]] = set()
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"A list row {number} is not an object")
            key = (_nonempty(raw.get("source"), "source"), _nonempty(raw.get("id"), "id"))
            if key in seen_source_keys:
                raise ValueError(f"Duplicate A source/id: {key}")
            seen_source_keys.add(key)
            issues = quality_issues(raw)
            if issues:
                flagged.append({"source": key[0], "id": key[1], "issues": issues})
            if key in approvals:
                matched.add(key)
                rows.append(make_row(raw, approvals[key], image_root, dataset_revision, handoff_commit, approval_hash))
    missing = sorted(set(approvals) - matched)
    if missing:
        raise ValueError(f"{len(missing)} C-approved IDs are absent from A list: {missing[:5]}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Generated sample_id collision")
    if len({row["image_sha256"] for row in rows}) != len(rows):
        raise ValueError("Selected rows share an image SHA256")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_jsonl": str(source), "source_sha256": sha256(source),
        "approval_file": str(approved_path), "approval_sha256": approval_hash,
        "image_root": str(image_root), "handoff_commit": handoff_commit,
        "dataset_revision": dataset_revision, "output": str(output),
        "output_sha256": sha256(output), "source_count": len(seen_source_keys),
        "approved_count": len(approvals), "converted_count": len(rows),
        "source_distribution": dict(sorted(Counter(row["source"] for row in rows).items())),
        "used_in_distribution": dict(sorted(Counter("+".join(row["a_used_in"]) or "unused" for row in rows).items())),
        "answer_type_distribution": dict(sorted(Counter(row["a_answer_type"] for row in rows).items())),
        "split_distribution": dict(sorted(Counter(row["split"] for row in rows).items())),
        "quality_issues_in_source": flagged,
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-jsonl", required=True, type=Path)
    parser.add_argument("--approved-ids", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--handoff-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = convert(
        args.list_jsonl, args.approved_ids, args.image_root, args.output,
        args.dataset_revision, args.handoff_commit,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "quality_issues_in_source"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
