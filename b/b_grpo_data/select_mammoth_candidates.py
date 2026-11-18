"""Select a reproducible, source-diverse MAmmoTH GRPO review queue.

The default queue uses only questions from A's SFT1 set, which A reports C
previously checked for MMMU leakage. It is still a *review queue*: this script
does not confer C approval for GRPO training or check the image pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_INPUT = Path(__file__).with_name("mammoth_a_handoff_3000_20261118.jsonl")
DEFAULT_SOURCE_SHA256 = "686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55"
DEFAULT_COUNT = 300
DEFAULT_SEED = 3407
FAMILY_WEIGHTS = {
    "mavis": 0.25,
    "clevr": 0.20,
    "geo170k": 0.20,
    "chart_table": 0.15,
    "vision_flan": 0.10,
    "geomverse": 0.05,
    "other": 0.05,
}
KNOWN_BAD_IDS = {"mammoth_8768873", "mammoth_6572627"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_for(source: str) -> str:
    name = source.lower()
    if "mavis" in name:
        return "mavis"
    if "clevr" in name:
        return "clevr"
    if "geo170k" in name:
        return "geo170k"
    if "geomverse" in name:
        return "geomverse"
    if any(token in name for token in ("chart", "plotqa", "mapqa", "tabmwp", "finqa")):
        return "chart_table"
    if "vflan" in name or "vision_flan" in name:
        return "vision_flan"
    return "other"


def normalized_question(question: str) -> str:
    return " ".join(question.casefold().split())


def reject_reason(row: dict, expected_used_in: list[str]) -> str | None:
    used_in = row.get("used_in")
    if used_in != expected_used_in:
        return "outside_requested_usage_scope"
    if row.get("text_only_suspect") is not False:
        return "text_only_suspect"
    if row.get("id") in KNOWN_BAD_IDS:
        return "known_malformed_question"
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        return "empty_question"
    if re.search(r"<image\s*\d*>|^\s*Answer\s*:", question, re.IGNORECASE | re.MULTILINE):
        return "embedded_image_or_answer_marker"
    path = row.get("image")
    if not isinstance(path, str) or not path.startswith("images/"):
        return "invalid_image_path"
    if ".." in PurePosixPath(path).parts:
        return "image_path_traversal"
    answer_type = row.get("answer_type")
    if answer_type not in {"mc", "numeric", "word"}:
        return "unsupported_answer_type"
    if answer_type == "mc":
        options = row.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 5:
            return "invalid_mc_options"
        if not isinstance(row.get("answer"), str) or row["answer"] not in "ABCDE"[: len(options)]:
            return "invalid_mc_answer"
    elif not isinstance(row.get("answer"), str) or not row["answer"].strip():
        return "empty_open_answer"
    return None


def load_pool(path: Path, expected_used_in: list[str]) -> tuple[list[dict], dict, int]:
    accepted: list[dict] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    seen_questions: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id in seen_ids:
                raise ValueError(f"Missing or duplicate id at line {line_number}: {row_id!r}")
            seen_ids.add(row_id)
            reason = reject_reason(row, expected_used_in)
            if reason:
                reasons[reason] += 1
                continue
            image_hash = row.get("image_sha256")
            question_key = normalized_question(row["question"])
            if not isinstance(image_hash, str) or len(image_hash) != 64:
                reasons["invalid_image_sha256"] += 1
                continue
            if image_hash in seen_images or question_key in seen_questions:
                reasons["duplicate_image_or_question_in_eligible_pool"] += 1
                continue
            seen_images.add(image_hash)
            seen_questions.add(question_key)
            accepted.append(row)
    return accepted, dict(sorted(reasons.items())), len(seen_ids)


def seeded_key(seed: int, row: dict) -> str:
    return hashlib.sha256(f"{seed}|{row['id']}".encode()).hexdigest()


def select(pool: list[dict], count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} eligible rows, requested {count}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        groups[family_for(row["source"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (seeded_key(seed, row), row["id"]))

    selected: list[dict] = []
    family_counts: Counter[str] = Counter()
    while len(selected) < count:
        step = len(selected) + 1
        available = [family for family, rows in groups.items() if rows]
        if not available:
            raise ValueError("No more eligible rows")
        # Weighted deficit keeps each prefix reasonably balanced for staged C review.
        family = min(
            available,
            key=lambda name: (family_counts[name] - step * FAMILY_WEIGHTS[name], name),
        )
        row = groups[family].pop(0)
        selected.append(row)
        family_counts[family] += 1
    return selected


def write_selection(
    *,
    selected: list[dict],
    output: Path,
    input_path: Path,
    source_hash: str,
    source_rows: int,
    pool_size: int,
    reasons: dict,
    seed: int,
    scope: str,
    required_next_step: str,
    review_csv: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "selection_scope": scope,
        "training_ready": False,
        "required_next_step": required_next_step,
        "source_file": str(input_path),
        "source_sha256": source_hash,
        "source_rows": source_rows,
        "eligible_rows": pool_size,
        "excluded_reasons": reasons,
        "seed": seed,
        "target_family_weights": FAMILY_WEIGHTS,
        "selected_rows": len(selected),
        "selected_jsonl_sha256": sha256_file(output),
        "family_counts": dict(sorted(Counter(family_for(r["source"]) for r in selected).items())),
        "source_counts": dict(sorted(Counter(r["source"] for r in selected).items())),
        "answer_type_counts": dict(sorted(Counter(r["answer_type"] for r in selected).items())),
        "first_100_family_counts": dict(sorted(Counter(family_for(r["source"]) for r in selected[:100]).items())),
        "sft1_used": sum("sft1" in r["used_in"] for r in selected),
        "dpo_overlap": sum(bool(set(r["used_in"]) & {"dpo_pool", "dpo_pairs"}) for r in selected),
        "text_only_suspect": sum(bool(r["text_only_suspect"]) for r in selected),
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(selected)} rows, SHA256 {report['selected_jsonl_sha256']})")
    print(f"Wrote {report_path}")
    if review_csv:
        csv_path = output.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("id", "source", "answer_type", "question", "options", "answer", "image", "used_in"),
            )
            writer.writeheader()
            for row in selected:
                writer.writerow({
                    key: json.dumps(row[key], ensure_ascii=False) if key in {"options", "used_in"} else row[key]
                    for key in writer.fieldnames
                })
        print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--expected-source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unused-count", type=int, default=100)
    parser.add_argument("--unused-output", type=Path)
    args = parser.parse_args()

    source_hash = sha256_file(args.input)
    if source_hash != args.expected_source_sha256.lower():
        raise ValueError(f"A handoff SHA256 mismatch: {source_hash}")
    pool, reasons, source_rows = load_pool(args.input, ["sft1"])
    selected = select(pool, args.count, args.seed)
    output = args.output or Path(__file__).with_name(
        f"mammoth_grpo_priority_{args.count}_sft1_checked_seed{args.seed}.jsonl"
    )
    write_selection(
        selected=selected,
        output=output,
        input_path=args.input,
        source_hash=source_hash,
        source_rows=source_rows,
        pool_size=len(pool),
        reasons=reasons,
        seed=args.seed,
        scope="A SFT1-used rows with prior C MMMU leakage screen only",
        required_next_step="C item-level quality and GRPO training approval; bridge approval file required",
        review_csv=False,
    )

    unused_pool, unused_reasons, _ = load_pool(args.input, [])
    unused_selected = select(unused_pool, args.unused_count, args.seed)
    unused_output = args.unused_output or Path(__file__).with_name(
        f"mammoth_c_review_priority_{args.unused_count}_unused_seed{args.seed}.jsonl"
    )
    write_selection(
        selected=unused_selected,
        output=unused_output,
        input_path=args.input,
        source_hash=source_hash,
        source_rows=source_rows,
        pool_size=len(unused_pool),
        reasons=unused_reasons,
        seed=args.seed,
        scope="A fully unused rows; no C MMMU leakage review yet",
        required_next_step="C MMMU/MMMU-Pro leakage, image, gold, and GRPO training review",
        review_csv=True,
    )


if __name__ == "__main__":
    main()
