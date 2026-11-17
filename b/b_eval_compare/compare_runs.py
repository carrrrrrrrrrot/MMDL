"""Compare a candidate MMMU v1.2 run with M4_final by question ID and seed.

Run this with the evaluation environment in which the repository's bundled
``lmms_eval`` is installed. The repository's own v1.2 ``is_correct`` function
scores both saved runs, including the older M4_final responses.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any


def load_scorer(repo_root: Path):
    script = repo_root / "scripts" / "summarize_results.py"
    if not script.is_file():
        raise FileNotFoundError(f"Repository scorer not found: {script}")
    spec = importlib.util.spec_from_file_location("mmdl_v12_summarizer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repository scorer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_correct


def find_samples(seed_dir: Path) -> Path:
    matches = sorted(seed_dir.glob("*/*_samples_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one samples JSONL in {seed_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_rows(seed_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with find_samples(seed_dir).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sample = json.loads(line)
            acc = sample.get("mmmu_acc")
            if not isinstance(acc, dict) or not acc.get("id"):
                raise ValueError(f"Missing mmmu_acc.id: {seed_dir}, line {line_number}")
            question_id = str(acc["id"])
            if question_id in rows:
                raise ValueError(f"Duplicate question ID {question_id}: {seed_dir}")
            rows[question_id] = {"acc": acc, "sample": sample}
    return rows


def parsed_answer(acc: dict[str, Any]) -> str:
    predictions = acc.get("parsed_pred")
    value = predictions[0] if isinstance(predictions, list) and predictions else None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def canonical_gold(acc: dict[str, Any]) -> Any:
    """Match v1.2 normalize_gold when old/new MMMU stores open keys differently."""
    answer = acc.get("answer")
    if acc.get("question_type") != "multiple-choice" and isinstance(answer, str) and answer.strip().startswith("["):
        try:
            parsed = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            return answer
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return answer


def raw_output(sample: dict[str, Any]) -> str:
    value = sample.get("filtered_resps")
    while isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, str) else ""


def finish_reason(sample: Any, depth: int = 0) -> str | None:
    """Return explicit generation stop metadata when the JSONL contains it."""
    if depth > 4:
        return None
    if isinstance(sample, dict):
        for key in ("finish_reason", "stop_reason"):
            value = sample.get(key)
            if isinstance(value, str):
                return value
        for value in sample.values():
            if isinstance(value, (dict, list)):
                found = finish_reason(value, depth + 1)
                if found is not None:
                    return found
    elif isinstance(sample, list):
        for value in sample:
            found = finish_reason(value, depth + 1)
            if found is not None:
                return found
    return None


def seed_dirs(run_dir: Path) -> dict[str, Path]:
    found = {p.name.removeprefix("seed_"): p for p in run_dir.glob("seed_*") if p.is_dir()}
    if not found:
        raise ValueError(f"No seed_* directories in {run_dir}")
    return found


def read_run_meta(seed_dir: Path) -> dict[str, str]:
    path = seed_dir / "run_meta.txt"
    if not path.is_file():
        return {}
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(": ")
        if separator and key in {
            "repo_commit", "repo_dirty_files", "gpu", "model_path", "data_root",
            "backend", "subjects", "limit", "gen_override", "exit_status",
        }:
            meta[key] = value
    return meta


def compare_seed(
    seed: str,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    is_correct,
    allow_partial: bool,
    expected_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(baseline) != expected_items:
        raise ValueError(f"Baseline seed {seed} has {len(baseline)} items, expected {expected_items}")
    base_ids, candidate_ids = set(baseline), set(candidate)
    if not allow_partial and (len(candidate_ids) != expected_items or candidate_ids != base_ids):
        raise ValueError(
            f"Seed {seed} needs the same {expected_items} IDs: "
            f"candidate={len(candidate_ids)}, missing={len(base_ids-candidate_ids)}, "
            f"extra={len(candidate_ids-base_ids)}. Use --allow-partial for smoke runs."
        )
    if not candidate_ids <= base_ids:
        raise ValueError(f"Candidate seed {seed} has {len(candidate_ids-base_ids)} unknown IDs")

    output: list[dict[str, Any]] = []
    for question_id in sorted(candidate_ids):
        b = baseline[question_id]
        c = candidate[question_id]
        ba, ca = b["acc"], c["acc"]
        if (canonical_gold(ba), ba.get("question_type")) != (
            canonical_gold(ca), ca.get("question_type")
        ):
            raise ValueError(f"Gold answer/type changed for {question_id}, seed {seed}")
        b_ok, c_ok = bool(is_correct(ba)), bool(is_correct(ca))
        if b_ok and c_ok:
            change = "both_correct"
        elif not b_ok and not c_ok:
            change = "both_wrong"
        elif b_ok:
            change = "right_to_wrong"
        else:
            change = "wrong_to_right"
        b_raw, c_raw = raw_output(b["sample"]), raw_output(c["sample"])
        output.append({
            "seed": seed,
            "question_id": question_id,
            "question_type": ba.get("question_type"),
            "gold_answer": json.dumps(canonical_gold(ba), ensure_ascii=False),
            "baseline_parsed_answer": parsed_answer(ba),
            "candidate_parsed_answer": parsed_answer(ca),
            "baseline_correct": b_ok,
            "candidate_correct": c_ok,
            "change": change,
            "baseline_final_answer_missing": ba.get("final_answer") is None,
            "candidate_final_answer_missing": ca.get("final_answer") is None,
            "baseline_raw_empty": not b_raw.strip(),
            "candidate_raw_empty": not c_raw.strip(),
            "baseline_finish_reason": finish_reason(b["sample"]),
            "candidate_finish_reason": finish_reason(c["sample"]),
            "input_exact_equal": b["sample"].get("input") == c["sample"].get("input"),
            "doc_hash_equal": b["sample"].get("doc_hash") == c["sample"].get("doc_hash"),
        })

    count = len(output)
    n = lambda key: sum(bool(row[key]) for row in output)
    flips_up = sum(row["change"] == "wrong_to_right" for row in output)
    flips_down = sum(row["change"] == "right_to_wrong" for row in output)
    parsed_equal = sum(
        row["baseline_parsed_answer"] == row["candidate_parsed_answer"] for row in output
    )
    binary_equal = count - flips_up - flips_down
    both_missing = sum(
        row["baseline_final_answer_missing"] and row["candidate_final_answer_missing"]
        for row in output
    )
    baseline_finish_available = sum(row["baseline_finish_reason"] is not None for row in output)
    candidate_finish_available = sum(row["candidate_finish_reason"] is not None for row in output)
    stats = {
        "items_compared": count,
        "full_900_coverage": count == expected_items,
        "baseline_correct": n("baseline_correct"),
        "candidate_correct": n("candidate_correct"),
        "correct_difference": flips_up - flips_down,
        "score_difference_pp": round(100 * (flips_up - flips_down) / count, 6),
        "binary_correctness_agreement": binary_equal,
        "binary_correctness_agreement_pct": round(100 * binary_equal / count, 6),
        "exact_parsed_answer_agreement": parsed_equal,
        "exact_parsed_answer_agreement_pct": round(100 * parsed_equal / count, 6),
        "input_exact_agreement": n("input_exact_equal"),
        "doc_hash_agreement": n("doc_hash_equal"),
        "wrong_to_right": flips_up,
        "right_to_wrong": flips_down,
        "baseline_missing_final_answer": n("baseline_final_answer_missing"),
        "candidate_missing_final_answer": n("candidate_final_answer_missing"),
        "both_missing_final_answer": both_missing,
        "baseline_only_missing_final_answer": n("baseline_final_answer_missing") - both_missing,
        "candidate_only_missing_final_answer": n("candidate_final_answer_missing") - both_missing,
        "baseline_empty_raw": n("baseline_raw_empty"),
        "candidate_empty_raw": n("candidate_raw_empty"),
        "baseline_explicit_length_stop": sum(
            row["baseline_finish_reason"] in {"length", "max_tokens", "max_new_tokens"}
            for row in output
        ) if baseline_finish_available else None,
        "candidate_explicit_length_stop": sum(
            row["candidate_finish_reason"] in {"length", "max_tokens", "max_new_tokens"}
            for row in output
        ) if candidate_finish_available else None,
        "baseline_finish_reason_available": baseline_finish_available,
        "candidate_finish_reason_available": candidate_finish_available,
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=900)
    parser.add_argument("--allow-partial", action="store_true", help="Compare a smoke-run ID subset")
    parser.add_argument("--dry-run", action="store_true", help="Print paired statistics without writing files")
    args = parser.parse_args()
    scorer = load_scorer(args.repo_root.resolve())
    base_dirs = seed_dirs(args.baseline_run)
    candidate_dirs = seed_dirs(args.candidate_run)
    missing = set(candidate_dirs) - set(base_dirs)
    if missing:
        raise ValueError(f"Candidate seeds absent from baseline: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted(
        candidate_dirs,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        paired, stats = compare_seed(
            seed,
            load_rows(base_dirs[seed]),
            load_rows(candidate_dirs[seed]),
            scorer,
            args.allow_partial,
            args.expected_items,
        )
        rows.extend(paired)
        stats["baseline_run_meta"] = read_run_meta(base_dirs[seed])
        stats["candidate_run_meta"] = read_run_meta(candidate_dirs[seed])
        per_seed[seed] = stats

    if not rows:
        raise ValueError("No paired question IDs found")

    summary: dict[str, Any] = {
        "scoring": "repo scripts/summarize_results.py:is_correct(normalize=True), v1.2",
        "baseline_run": str(args.baseline_run.resolve()),
        "candidate_run": str(args.candidate_run.resolve()),
        "per_seed": per_seed,
        "note": (
            "M4_final is post-hoc rescored with v1.2. Agreement is paired by the SAME "
            "seed and question ID. Missing finish_reason "
            "metadata means truncation cannot be determined from this JSONL. "
            "Partial runs are diagnostic only; no full-score inference."
        ),
    }
    if len(per_seed) >= 2 and all(p["full_900_coverage"] for p in per_seed.values()):
        base_scores = [100 * p["baseline_correct"] / args.expected_items for p in per_seed.values()]
        candidate_scores = [100 * p["candidate_correct"] / args.expected_items for p in per_seed.values()]
        summary["overall"] = {
            "seed_count": len(per_seed),
            "baseline_mean_pct": statistics.mean(base_scores),
            "candidate_mean_pct": statistics.mean(candidate_scores),
            "mean_difference_pp": statistics.mean(candidate_scores) - statistics.mean(base_scores),
            "baseline_sample_sd_pp": statistics.stdev(base_scores),
            "candidate_sample_sd_pp": statistics.stdev(candidate_scores),
        }

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "paired_items.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        json_path = args.output_dir / "comparison_summary.json"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {json_path} and {csv_path}")
    for seed, stats in per_seed.items():
        print(
            f"seed {seed}: candidate {stats['candidate_correct']}/{stats['items_compared']}, "
            f"baseline {stats['baseline_correct']}/{stats['items_compared']}, "
            f"binary agreement {stats['binary_correctness_agreement_pct']:.2f}%, "
            f"parsed answer agreement {stats['exact_parsed_answer_agreement_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
