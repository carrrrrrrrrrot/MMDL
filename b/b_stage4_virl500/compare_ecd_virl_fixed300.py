"""Read-only paired ECD500 vs ViRL500 fixed-300 diagnostic comparison.

All four inputs must be completed comparisons of seed 3407 on the same MMMU
300 IDs and pinned base185. Stage 2 ECD hashes are frozen in this script;
Stage 4 hashes are computed, checked for stability while parsing, and may be
checked against a separately captured --expected-hashes JSON lock.

This never launches evaluation or GPU work. Output files are optional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEED = "3407"
ITEMS = 300
BASE_CORRECT = 185
BASELINE_RUN = "/data/chaewon/mmdl_eval_runs/rlab19qcv2_base_fixed300"
PINNED_STAGE2 = {
    "ecd_grpo": {
        "summary_sha256": "41267b10efb36fb36d0fc580150c00ba614140746a9e7bc326b55c64eb40594d",
        "paired_sha256": "45ffa7281aae74302edf1d6b1dc9c54198f92fa4b318d4f3c855a03b86166708",
        "run_name": "stage2_ecd500_grpo_20261121_fixed300",
    },
    "ecd_dapo": {
        "summary_sha256": "8ee976f6fb6d16661dc1a5785abcaba129f69db6acff6061f8b2debe44659d4c",
        "paired_sha256": "0fd20c84c6d97a128550b12b0c8378b2d9c12b087bbc0b98229822c406d2c631",
        "run_name": "stage2_ecd500_dapo_core_20261121_fixed300",
    },
}
EXPECTED_STAGE4_NAMES = {
    "virl_grpo": "stage4_virl500_grpo_20261121_fixed300",
    "virl_dapo": "stage4_virl500_dapo_core_20261121_fixed300",
}
KEYS = ("ecd_grpo", "ecd_dapo", "virl_grpo", "virl_dapo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_csv(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Unexpected CSV boolean {value!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_one(key: str, comparison: Path, expected: dict | None = None,
             self_test: bool = False) -> dict:
    summary_path = comparison / "comparison_summary.json"
    paired_path = comparison / "paired_items.csv"
    before = {
        "summary_sha256": sha256(summary_path),
        "paired_sha256": sha256(paired_path),
    }
    pin = PINNED_STAGE2.get(key)
    if pin is not None:
        require(all(before[field] == pin[field] for field in before),
                f"Pinned Stage2 SHA256 mismatch: {key}")
    if expected is not None:
        require(all(before[field] == expected[field] for field in before),
                f"Expected SHA256 mismatch: {key}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(set(summary["per_seed"]) == {SEED}, f"{key}: seed must be {SEED} only")
    require(summary["baseline_run"] == BASELINE_RUN, f"{key}: wrong baseline run")
    require("v1.2" in summary["scoring"], f"{key}: scorer is not v1.2")
    expected_name = (pin or {}).get("run_name") or EXPECTED_STAGE4_NAMES.get(key)
    if not self_test:
        require(Path(summary["candidate_run"]).name == expected_name,
                f"{key}: candidate run name mismatch")
    metric = summary["per_seed"][SEED]
    require(metric["items_compared"] == ITEMS and metric["baseline_correct"] == BASE_CORRECT,
            f"{key}: wrong item/baseline counts")
    require(metric["input_exact_agreement"] == ITEMS and metric["doc_hash_agreement"] == ITEMS,
            f"{key}: input/document parity failed")
    require(metric.get("full_900_coverage") is False,
            f"{key}: expected a partial fixed300 diagnostic")
    for run_meta in (metric["baseline_run_meta"], metric["candidate_run_meta"]):
        require(str(run_meta["exit_status"]) == "0", f"{key}: eval process failed")
    with paired_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "seed", "question_id", "question_type", "gold_answer",
            "baseline_parsed_answer", "candidate_parsed_answer",
            "baseline_correct", "candidate_correct", "input_exact_equal", "doc_hash_equal",
            "baseline_final_answer_missing", "candidate_final_answer_missing",
        }
        require(required.issubset(reader.fieldnames or ()), f"{key}: missing CSV fields")
        rows = list(reader)
    require(len(rows) == ITEMS, f"{key}: expected {ITEMS} CSV rows")
    by_id = {}
    for row in rows:
        question_id = row["question_id"]
        require(row["seed"] == SEED and question_id not in by_id,
                f"{key}: seed/duplicate ID error on {question_id}")
        require(bool_csv(row["input_exact_equal"]) and bool_csv(row["doc_hash_equal"]),
                f"{key}: input/document mismatch on {question_id}")
        row["base_ok"] = bool_csv(row["baseline_correct"])
        row["candidate_ok"] = bool_csv(row["candidate_correct"])
        row["base_missing"] = bool_csv(row["baseline_final_answer_missing"])
        row["candidate_missing"] = bool_csv(row["candidate_final_answer_missing"])
        by_id[question_id] = row

    require(sum(row["base_ok"] for row in rows) == BASE_CORRECT,
            f"{key}: CSV baseline count mismatch")
    correct = sum(row["candidate_ok"] for row in rows)
    correction = sum(not row["base_ok"] and row["candidate_ok"] for row in rows)
    regression = sum(row["base_ok"] and not row["candidate_ok"] for row in rows)
    missing = sum(row["candidate_missing"] for row in rows)
    require((metric["candidate_correct"] == correct and
             metric["wrong_to_right"] == correction and
             metric["right_to_wrong"] == regression and
             metric["correct_difference"] == correction - regression and
             metric["candidate_missing_final_answer"] == missing),
            f"{key}: summary/CSV counts differ")
    after = {
        "summary_sha256": sha256(summary_path),
        "paired_sha256": sha256(paired_path),
    }
    require(before == after, f"{key}: comparison files changed during read")
    return {
        "summary": summary,
        "metric": metric,
        "rows": by_id,
        "hashes": before,
        "comparison_dir": str(comparison),
        "correct": correct,
        "correction_vs_base": correction,
        "regression_vs_base": regression,
        "missing_final_answer": missing,
    }


def pair_counts(left: dict, right: dict) -> dict:
    corrected = []
    regressed = []
    both_correct = []
    both_wrong = []
    for question_id in sorted(left["rows"]):
        a = left["rows"][question_id]["candidate_ok"]
        b = right["rows"][question_id]["candidate_ok"]
        if not a and b:
            corrected.append(question_id)
        elif a and not b:
            regressed.append(question_id)
        elif a:
            both_correct.append(question_id)
        else:
            both_wrong.append(question_id)
    return {
        "left_wrong_right_correct": len(corrected),
        "left_correct_right_wrong": len(regressed),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "net_right_minus_left": len(corrected) - len(regressed),
        "left_wrong_right_correct_ids": corrected,
        "left_correct_right_wrong_ids": regressed,
    }


def compare(directories: dict[str, Path], expected_hashes: dict | None = None,
            self_test: bool = False) -> tuple[dict, list[dict]]:
    candidates = {
        key: load_one(key, directories[key],
                      (expected_hashes or {}).get(key), self_test)
        for key in KEYS
    }
    anchor = candidates["ecd_grpo"]
    anchor_ids = set(anchor["rows"])
    baseline_meta = anchor["metric"]["baseline_run_meta"]
    scoring = anchor["summary"]["scoring"]
    for key, candidate in candidates.items():
        require(set(candidate["rows"]) == anchor_ids, f"{key}: fixed300 ID set differs")
        require(candidate["summary"]["scoring"] == scoring, f"{key}: scorer differs")
        require(candidate["metric"]["baseline_run_meta"] == baseline_meta,
                f"{key}: base run metadata differs")
        for question_id in anchor_ids:
            reference, row = anchor["rows"][question_id], candidate["rows"][question_id]
            for field in ("question_type", "gold_answer", "baseline_parsed_answer",
                          "baseline_correct", "baseline_final_answer_missing"):
                require(reference[field] == row[field],
                        f"{key}: base/gold mismatch {question_id} field {field}")
    if not self_test:
        runs = [candidate["summary"]["candidate_run"] for candidate in candidates.values()]
        require(len(set(runs)) == 4, "Four inputs point to non-distinct candidate runs")
        parity = ("repo_commit", "repo_dirty_files", "gpu", "data_root", "backend",
                  "subjects", "limit", "gen_override")
        anchor_meta = anchor["metric"]["candidate_run_meta"]
        for key, candidate in candidates.items():
            meta = candidate["metric"]["candidate_run_meta"]
            for field in parity:
                require(meta.get(field) == anchor_meta.get(field),
                        f"{key}: candidate evaluation metadata differs: {field}")

    scores = {}
    for key, candidate in candidates.items():
        scores[key] = {
            "correct": candidate["correct"],
            "accuracy_pct": round(100 * candidate["correct"] / ITEMS, 4),
            "delta_vs_base_items": candidate["correct"] - BASE_CORRECT,
            "delta_vs_base_pp": round(100 * (candidate["correct"] - BASE_CORRECT) / ITEMS, 4),
            "correction_vs_base": candidate["correction_vs_base"],
            "regression_vs_base": candidate["regression_vs_base"],
            "missing_final_answer": candidate["missing_final_answer"],
            "comparison_dir": candidate["comparison_dir"],
            **candidate["hashes"],
        }
    per_algorithm = {
        "grpo": pair_counts(candidates["ecd_grpo"], candidates["virl_grpo"]),
        "dapo_core": pair_counts(candidates["ecd_dapo"], candidates["virl_dapo"]),
    }
    within_dataset = {
        "ecd_grpo_to_dapo": pair_counts(candidates["ecd_grpo"], candidates["ecd_dapo"]),
        "virl_grpo_to_dapo": pair_counts(candidates["virl_grpo"], candidates["virl_dapo"]),
    }
    paired_rows = []
    for question_id in sorted(anchor_ids):
        row = anchor["rows"][question_id]
        flags = {key: candidates[key]["rows"][question_id]["candidate_ok"] for key in KEYS}
        paired_rows.append({
            "seed": SEED,
            "question_id": question_id,
            "baseline_correct": row["base_ok"],
            **{key + "_correct": flags[key] for key in KEYS},
            "grpo_ecd_to_virl": "correction" if not flags["ecd_grpo"] and flags["virl_grpo"] else
                                "regression" if flags["ecd_grpo"] and not flags["virl_grpo"] else "same",
            "dapo_ecd_to_virl": "correction" if not flags["ecd_dapo"] and flags["virl_dapo"] else
                                "regression" if flags["ecd_dapo"] and not flags["virl_dapo"] else "same",
        })
    report = {
        "scope": "6000 fixed-ID MMMU 300, seed 3407; diagnostic only",
        "baseline_run": BASELINE_RUN,
        "baseline_correct": BASE_CORRECT,
        "baseline_accuracy_pct": round(100 * BASE_CORRECT / ITEMS, 4),
        "scoring": scoring,
        "checks": {
            "same_question_ids": ITEMS,
            "input_exact_agreement_per_candidate": ITEMS,
            "doc_hash_agreement_per_candidate": ITEMS,
            "baseline_meta_equal": True,
            "candidate_eval_meta_equal_except_model": not self_test,
            "stage2_hashes_pinned": True,
            "stage4_hashes_independently_pinned": bool(expected_hashes and
                                                    all(key in expected_hashes for key in ("virl_grpo", "virl_dapo"))),
            "all_eight_file_hashes_stable_during_read": True,
        },
        "scores": scores,
        "ecd_to_virl": per_algorithm,
        "within_dataset_grpo_to_dapo": within_dataset,
        "interpretation_limit": (
            "Same-ID/same-seed exploratory comparison of two training datasets. "
            "It does not isolate data causally from run variation and does not establish 900-item performance. "
            "ViRL remains pending C's OCR/semantic/manual clearance."
        ),
    }
    return report, paired_rows


def markdown(report: dict) -> str:
    scores = report["scores"]
    lines = [
        "# ECD500 vs ViRL500: fixed300 diagnostic", "",
        "RTX PRO 6000, seed 3407, same 300 MMMU IDs and same base 185/300. "
        "ViRL data awaits C's OCR/semantic/manual review.", "",
        "| Training data / algorithm | Correct / 300 | Accuracy | vs base | "
        "Corrections vs base | Regressions vs base | Missing final answer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Base | {BASE_CORRECT} | {report['baseline_accuracy_pct']:.2f}% | — | — | — | — |",
    ]
    labels = {
        "ecd_grpo": "ECD500 GRPO", "ecd_dapo": "ECD500 DAPO core",
        "virl_grpo": "ViRL500 GRPO", "virl_dapo": "ViRL500 DAPO core",
    }
    for key in KEYS:
        item = scores[key]
        lines.append(
            f"| {labels[key]} | {item['correct']} | {item['accuracy_pct']:.2f}% | "
            f"{item['delta_vs_base_pp']:+.2f}%p | {item['correction_vs_base']} | "
            f"{item['regression_vs_base']} | {item['missing_final_answer']} |"
        )
    lines.extend(["", "| Same algorithm, ECD → ViRL | New corrections | New regressions | Net |",
                  "| --- | ---: | ---: | ---: |"])
    for algorithm in ("grpo", "dapo_core"):
        pair = report["ecd_to_virl"][algorithm]
        lines.append(
            f"| {algorithm.upper()} | {pair['left_wrong_right_correct']} | "
            f"{pair['left_correct_right_wrong']} | {pair['net_right_minus_left']:+d} / 300 |"
        )
    lines.extend([
        "", "All four files passed 300-ID, seed, baseline, scorer, input/document, "
        "summary/CSV count, and SHA stability checks. Stage2 hashes were "
        "checked against pinned values. Stage4 hashes are listed in the JSON; "
        "an independent Stage4 hash lock was " +
        ("checked." if report["checks"]["stage4_hashes_independently_pinned"] else "not supplied."),
        "", "These are one-seed, 300-question candidate screens. Differences are "
        "paired observations; they do not prove which training data caused "
        "them or predict a 900-question result.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecd-grpo", type=Path, required=True)
    parser.add_argument("--ecd-dapo", type=Path, required=True)
    parser.add_argument("--virl-grpo", type=Path)
    parser.add_argument("--virl-dapo", type=Path)
    parser.add_argument("--expected-hashes", type=Path,
                        help="Optional JSON mapping four keys to summary_sha256 and paired_sha256")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional destination for report.json, report.md, paired_changes.csv")
    parser.add_argument("--self-test-stage2", action="store_true",
                        help="Use ECD inputs in all four slots to test arithmetic; no real ViRL claim")
    args = parser.parse_args()
    if not args.self_test_stage2 and (args.virl_grpo is None or args.virl_dapo is None):
        parser.error("Normal comparison requires both ViRL comparison directories")
    if args.self_test_stage2 and args.output_dir:
        parser.error("Self-test cannot write a comparative report")
    directories = {
        "ecd_grpo": args.ecd_grpo,
        "ecd_dapo": args.ecd_dapo,
        "virl_grpo": args.ecd_grpo if args.self_test_stage2 else args.virl_grpo,
        "virl_dapo": args.ecd_dapo if args.self_test_stage2 else args.virl_dapo,
    }
    expected = (json.loads(args.expected_hashes.read_text(encoding="utf-8"))
                if args.expected_hashes else None)
    report, paired = compare(directories, expected, args.self_test_stage2)
    if args.self_test_stage2:
        require(all(item["net_right_minus_left"] == 0
                    for item in report["ecd_to_virl"].values()),
                "Self-test ECD-vs-itself must have zero net difference")
        print(json.dumps({"self_test_stage2": "passed", "baseline": BASE_CORRECT,
                          "ecd_grpo": report["scores"]["ecd_grpo"]["correct"],
                          "ecd_dapo": report["scores"]["ecd_dapo"]["correct"],
                          "paired_items": len(paired)}, indent=2))
        return
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        hash_lock = {
            key: {
                "summary_sha256": report["scores"][key]["summary_sha256"],
                "paired_sha256": report["scores"][key]["paired_sha256"],
            }
            for key in KEYS
        }
        (args.output_dir / "hash_lock.json").write_text(
            json.dumps(hash_lock, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
        with (args.output_dir / "paired_changes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    print(json.dumps({
        "scope": report["scope"],
        "scores": {key: report["scores"][key]["correct"] for key in KEYS},
        "ecd_to_virl": {key: {
            "corrections": value["left_wrong_right_correct"],
            "regressions": value["left_correct_right_wrong"],
            "net": value["net_right_minus_left"],
        } for key, value in report["ecd_to_virl"].items()},
        "checks": report["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
