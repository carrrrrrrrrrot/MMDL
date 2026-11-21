"""Validate and compare two full MMMU evaluations on the same 900 IDs."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIXED300 = ROOT.parents[1] / "b_stage2_ecd500" / "results" / "dapo_fixed300" / "paired_items.csv"
OLD_FIXED300 = ROOT / "old_grpo" / "fixed300_paired_items.csv"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {(row["seed"], row["question_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate seed/question ID in {path}")
    return by_id


def count(rows, field):
    return sum(row[field] == "True" for row in rows)


def summarize(old, new, keys):
    old_rows = [old[key] for key in keys]
    new_rows = [new[key] for key in keys]
    return {
        "items": len(keys),
        "baseline_correct": count(old_rows, "baseline_correct"),
        "old_grpo_correct": count(old_rows, "candidate_correct"),
        "new_dapo_correct": count(new_rows, "candidate_correct"),
        "both_candidates_correct": sum(
            a["candidate_correct"] == b["candidate_correct"] == "True"
            for a, b in zip(old_rows, new_rows)
        ),
        "old_grpo_only_correct": sum(
            a["candidate_correct"] == "True" and b["candidate_correct"] == "False"
            for a, b in zip(old_rows, new_rows)
        ),
        "new_dapo_only_correct": sum(
            a["candidate_correct"] == "False" and b["candidate_correct"] == "True"
            for a, b in zip(old_rows, new_rows)
        ),
        "both_candidates_wrong": sum(
            a["candidate_correct"] == b["candidate_correct"] == "False"
            for a, b in zip(old_rows, new_rows)
        ),
        "old_grpo_missing_final_answer": count(old_rows, "candidate_final_answer_missing"),
        "new_dapo_missing_final_answer": count(new_rows, "candidate_final_answer_missing"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / "old_grpo" / "paired_items.csv")
    parser.add_argument("--new", type=Path, default=ROOT / "new_dapo" / "paired_items.csv")
    parser.add_argument("--fixed300", type=Path, default=FIXED300)
    parser.add_argument("--old-fixed300", type=Path, default=OLD_FIXED300)
    parser.add_argument("--output", type=Path, default=ROOT / "pair_analysis.json")
    args = parser.parse_args()
    old = read_rows(args.old)
    new = read_rows(args.new)
    fixed = read_rows(args.fixed300)
    old_fixed = read_rows(args.old_fixed300)
    if len(old) != 900 or old.keys() != new.keys():
        raise ValueError("The candidates do not have the same complete 900-item key set")
    if len(fixed) != 300 or fixed.keys() != old_fixed.keys() or not fixed.keys() <= old.keys():
        raise ValueError("The fixed300 IDs are incomplete or absent from the 900-item runs")
    baseline_fields = (
        "question_type", "gold_answer", "baseline_parsed_answer",
        "baseline_correct", "baseline_final_answer_missing", "input_exact_equal",
        "doc_hash_equal",
    )
    for key in old:
        if any(old[key][field] != new[key][field] for field in baseline_fields):
            raise ValueError(f"Baseline mismatch for {key}")
        if any(old[key][field] != "True" or new[key][field] != "True"
               for field in ("input_exact_equal", "doc_hash_equal")):
            raise ValueError(f"Evaluation input/hash mismatch for {key}")
    all_keys = sorted(old)
    subset = sorted(fixed)
    remaining = sorted(old.keys() - fixed.keys())
    report = {
        "all900": summarize(old, new, all_keys),
        "fixed300_ids_within_full900": summarize(old, new, subset),
        "remaining600": summarize(old, new, remaining),
        "old_grpo_fixed300_correctness_changed_vs_prior_run": sum(
            old[key]["candidate_correct"] != old_fixed[key]["candidate_correct"]
            for key in subset
        ),
        "new_dapo_fixed300_correctness_changed_vs_prior_run": sum(
            new[key]["candidate_correct"] != fixed[key]["candidate_correct"]
            for key in subset
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
