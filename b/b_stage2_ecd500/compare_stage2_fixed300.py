"""Read-only paired comparison of the two Stage2 fixed-ID MMMU 300 screens.

Use only after both candidate evaluation handoffs report status=complete.
The default prints a compact JSON result; optional output files are written
only when --output-json / --output-md is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEED = "3407"
EXPECTED_ITEMS = 300
EXPECTED_BASE_CORRECT = 185


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected Python CSV boolean, got {value!r}")


def read_candidate(comparison_dir: Path) -> dict:
    summary_path = comparison_dir / "comparison_summary.json"
    paired_path = comparison_dir / "paired_items.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_seed = summary["per_seed"]
    if set(per_seed) != {SEED}:
        raise ValueError(f"Expected seed {SEED} only: {summary_path}")
    metrics = per_seed[SEED]
    if (metrics["items_compared"] != EXPECTED_ITEMS or
            metrics["baseline_correct"] != EXPECTED_BASE_CORRECT or
            metrics["input_exact_agreement"] != EXPECTED_ITEMS or
            metrics["doc_hash_agreement"] != EXPECTED_ITEMS):
        raise ValueError(f"Fixed300/baseline/input gate failed: {summary_path}")
    if metrics.get("full_900_coverage") is not False:
        raise ValueError(f"Result was not labeled a partial diagnostic: {summary_path}")

    with paired_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"seed", "question_id", "gold_answer", "baseline_parsed_answer",
                    "baseline_correct", "candidate_correct", "input_exact_equal",
                    "doc_hash_equal"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Paired CSV lacks required columns: {paired_path}")
        rows = list(reader)
    if len(rows) != EXPECTED_ITEMS:
        raise ValueError(f"Expected {EXPECTED_ITEMS} paired rows, got {len(rows)}")
    by_id = {}
    for row in rows:
        if row["seed"] != SEED or row["question_id"] in by_id:
            raise ValueError(f"Bad seed/duplicate ID: {row['question_id']}")
        if not read_bool(row["input_exact_equal"]) or not read_bool(row["doc_hash_equal"]):
            raise ValueError(f"Input/document mismatch: {row['question_id']}")
        row["base_ok"] = read_bool(row["baseline_correct"])
        row["candidate_ok"] = read_bool(row["candidate_correct"])
        by_id[row["question_id"]] = row
    if sum(row["base_ok"] for row in rows) != EXPECTED_BASE_CORRECT:
        raise ValueError("Paired CSV baseline count differs from frozen 185/300")
    candidate_correct = sum(row["candidate_ok"] for row in rows)
    gain = sum(not row["base_ok"] and row["candidate_ok"] for row in rows)
    loss = sum(row["base_ok"] and not row["candidate_ok"] for row in rows)
    if (metrics["candidate_correct"] != candidate_correct or
            metrics["wrong_to_right"] != gain or metrics["right_to_wrong"] != loss or
            metrics["correct_difference"] != gain - loss):
        raise ValueError(f"Summary/paired CSV count mismatch: {comparison_dir}")
    return {"summary": summary, "metrics": metrics, "by_id": by_id,
            "summary_sha256": sha256(summary_path), "paired_sha256": sha256(paired_path),
            "comparison_dir": str(comparison_dir)}


def compare(grpo_dir: Path, dapo_dir: Path) -> dict:
    grpo = read_candidate(grpo_dir)
    dapo = read_candidate(dapo_dir)
    if grpo["summary"]["baseline_run"] != dapo["summary"]["baseline_run"]:
        raise ValueError("GRPO and DAPO use different baseline runs")
    if grpo["summary"]["scoring"] != dapo["summary"]["scoring"]:
        raise ValueError("GRPO and DAPO use different scorers")
    if grpo["summary"]["candidate_run"] == dapo["summary"]["candidate_run"]:
        raise ValueError("Both inputs point to the same candidate run")
    if set(grpo["by_id"]) != set(dapo["by_id"]):
        raise ValueError("GRPO and DAPO paired item ID sets differ")
    if (grpo["metrics"]["baseline_run_meta"] !=
            dapo["metrics"]["baseline_run_meta"]):
        raise ValueError("GRPO and DAPO baseline metadata differ")

    cross = {"both_correct": [], "grpo_only_correct": [],
             "dapo_only_correct": [], "both_wrong": []}
    for item_id in sorted(grpo["by_id"]):
        g = grpo["by_id"][item_id]
        d = dapo["by_id"][item_id]
        for field in ("gold_answer", "baseline_parsed_answer", "baseline_correct"):
            if g[field] != d[field]:
                raise ValueError(f"Baseline/gold mismatch on {item_id}: {field}")
        if g["candidate_ok"] and d["candidate_ok"]:
            cross["both_correct"].append(item_id)
        elif g["candidate_ok"]:
            cross["grpo_only_correct"].append(item_id)
        elif d["candidate_ok"]:
            cross["dapo_only_correct"].append(item_id)
        else:
            cross["both_wrong"].append(item_id)

    def candidate_report(candidate: dict) -> dict:
        m = candidate["metrics"]
        return {"correct": m["candidate_correct"],
                "accuracy_pct": round(m["candidate_correct"] / EXPECTED_ITEMS * 100, 4),
                "delta_vs_base_items": m["correct_difference"],
                "delta_vs_base_pp": round(m["correct_difference"] / EXPECTED_ITEMS * 100, 4),
                "wrong_to_right": m["wrong_to_right"],
                "right_to_wrong": m["right_to_wrong"],
                "missing_final_answer": m["candidate_missing_final_answer"],
                "comparison_dir": candidate["comparison_dir"],
                "summary_sha256": candidate["summary_sha256"],
                "paired_sha256": candidate["paired_sha256"]}

    return {"scope": "6000 fixed-ID MMMU 300, seed 3407; diagnostic only",
            "baseline_run": grpo["summary"]["baseline_run"],
            "scoring": grpo["summary"]["scoring"],
            "baseline": {"correct": EXPECTED_BASE_CORRECT,
                         "accuracy_pct": round(EXPECTED_BASE_CORRECT / EXPECTED_ITEMS * 100, 4)},
            "grpo": candidate_report(grpo), "dapo_core": candidate_report(dapo),
            "cross_candidate": {key: {"count": len(ids), "question_ids": ids}
                                for key, ids in cross.items()},
            "net_dapo_minus_grpo_items": (dapo["metrics"]["candidate_correct"] -
                                           grpo["metrics"]["candidate_correct"])}


def markdown(result: dict) -> str:
    base, grpo, dapo = result["baseline"], result["grpo"], result["dapo_core"]
    cross = result["cross_candidate"]
    return "\n".join([
        "# Stage2 ECD500 fixed300 paired comparison", "",
        "6000, seed 3407, 동일한 고정 300문항. 후보 선별용 진단 결과이며 공식 900문항 결과가 아닙니다.", "",
        "| 모델 | 정답/300 | 정확도 | base 대비 | 정답 전환 | 오답 전환 | 최종 답 누락 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Base | {base['correct']} | {base['accuracy_pct']:.2f}% | — | — | — | — |",
        f"| GRPO | {grpo['correct']} | {grpo['accuracy_pct']:.2f}% | "
        f"{grpo['delta_vs_base_pp']:+.2f}%p | {grpo['wrong_to_right']} | "
        f"{grpo['right_to_wrong']} | {grpo['missing_final_answer']} |",
        f"| DAPO core | {dapo['correct']} | {dapo['accuracy_pct']:.2f}% | "
        f"{dapo['delta_vs_base_pp']:+.2f}%p | {dapo['wrong_to_right']} | "
        f"{dapo['right_to_wrong']} | {dapo['missing_final_answer']} |", "",
        f"GRPO만 정답 {cross['grpo_only_correct']['count']}문항, "
        f"DAPO만 정답 {cross['dapo_only_correct']['count']}문항; "
        f"DAPO−GRPO {result['net_dapo_minus_grpo_items']:+d}문항.", "",
        "정답 수 차이는 동일 문항의 짝 비교입니다. 단일 시드 300문항의 후보 선별 결과를 "
        "통계적 유의성이나 공식 MMMU 성능으로 해석하지 않습니다.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-comparison", required=True, type=Path)
    parser.add_argument("--dapo-comparison", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    report = compare(args.grpo_comparison, args.dapo_comparison)
    if args.output_json:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(markdown(report), encoding="utf-8")
    compact = {"baseline": report["baseline"], "grpo": report["grpo"],
               "dapo_core": report["dapo_core"],
               "cross_candidate_counts": {k: v["count"] for k, v in report["cross_candidate"].items()},
               "net_dapo_minus_grpo_items": report["net_dapo_minus_grpo_items"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
