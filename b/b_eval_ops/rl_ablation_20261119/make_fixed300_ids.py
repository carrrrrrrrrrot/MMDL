"""Pick ten fixed MMMU validation IDs from each subject for B screening.

This is a diagnostic subset only. It uses the saved 6000 base run to identify
the exact 900 v1.2 validation IDs, and never reads model predictions for
selection. The frozen 900-item v1.2 evaluation remains the final metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing selection differs: {path}")
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="One MMMU ID per line")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    matches = sorted((args.baseline_run / "seed_3407").glob("*/*_samples_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one full base samples file, found {matches}")
    sample_path = matches[0]
    by_subject: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    with sample_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sample = json.loads(line)
            acc = sample.get("mmmu_acc") or {}
            question_id = acc.get("id")
            subject = acc.get("subdomain")
            if not isinstance(question_id, str) or not isinstance(subject, str):
                raise ValueError(f"Missing ID/subject at line {line_number}")
            if question_id in seen:
                raise ValueError(f"Duplicate MMMU ID: {question_id}")
            seen.add(question_id)
            by_subject[subject].append(question_id)
    if len(seen) != 900 or len(by_subject) != 30:
        raise ValueError(f"Expected 900 IDs across 30 subjects, got {len(seen)} / {len(by_subject)}")
    counts = {subject: len(ids) for subject, ids in by_subject.items()}
    if set(counts.values()) != {30}:
        raise ValueError(f"Expected 30 items in every subject, got {counts}")

    selected: list[str] = []
    for subject in sorted(by_subject):
        ranked = sorted(
            by_subject[subject],
            key=lambda qid: (hashlib.sha256(f"{args.seed}\0{qid}".encode()).hexdigest(), qid),
        )
        selected.extend(ranked[:10])
    selected = sorted(selected)
    ids_content = "\n".join(selected) + "\n"
    save_once(args.output, ids_content)
    metadata = {
        "purpose": "paired 300-item diagnostic screen; not official MMMU score",
        "selection_seed": args.seed,
        "baseline_samples": str(sample_path.resolve(strict=True)),
        "baseline_samples_sha256": sha256(sample_path),
        "source_count": len(seen),
        "source_subject_count": len(by_subject),
        "per_subject": 10,
        "selected_count": len(selected),
        "ids_sha256": sha256(args.output),
        "selected_ids": selected,
    }
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    save_once(manifest, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"ids": str(args.output), "ids_sha256": metadata["ids_sha256"], "count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
