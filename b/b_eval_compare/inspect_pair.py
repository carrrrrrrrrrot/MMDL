"""Inspect a single baseline/candidate question without changing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_runs import load_rows, raw_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seed", type=Path, required=True)
    parser.add_argument("--candidate-seed", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    for label, path in (("baseline", args.baseline_seed), ("candidate", args.candidate_seed)):
        row = load_rows(path)[args.id]
        acc = row["acc"]
        doc = row["sample"].get("doc")
        print(json.dumps({
            "label": label,
            "id": acc.get("id"),
            "answer": acc.get("answer"),
            "question_type": acc.get("question_type"),
            "parsed_pred": acc.get("parsed_pred"),
            "final_answer": acc.get("final_answer"),
            "raw": raw_output(row["sample"])[:500],
            "sample_keys": sorted(row["sample"].keys()),
            "input_type": type(row["sample"].get("input")).__name__,
            "doc_hash_type": type(row["sample"].get("doc_hash")).__name__,
            "doc_keys": sorted(doc.keys()) if isinstance(doc, dict) else None,
            "question": doc.get("question") if isinstance(doc, dict) else None,
            "options": doc.get("options") if isinstance(doc, dict) else None,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
