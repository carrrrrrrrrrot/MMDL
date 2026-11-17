"""Run B handoff R01–R18 plus ambiguity and length-stop regression cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reward import RewardPolicy, score_response


# Expected rewards are the B handoff's proposed policy, not a claim that the
# repository's v1.2 scorer uses identical rules for R09/R11/R13/R14.
CASES = [
    ("R01", "mc", "B", "Answer: B", 1, None),
    ("R02", "mc", "B", "Answer: C", 0, None),
    ("R03", "mc", "B", "B seems likely.\nAnswer: C", 0, None),
    ("R04", "mc", "B", "B is the answer.", 0, None),
    ("R05", "mc", "B", "Answer: C\nAnswer: B", 1, None),
    ("R06", "mc", "B", "Answer: B\nAnswer:", 0, None),
    ("R07", "mc", "B", "Answer: B or C", 0, None),
    ("R08", "mc", "B", "Answer: E", 0, None),
    ("R09", "mc", "B", "Answer: b", 1, None),
    ("R10", "number", "15", "Answer: 15", 1, None),
    ("R11", "number", "15", "Answer: 15.10", 1, None),
    ("R12", "number", "15", "Answer: 15.1001", 0, None),
    ("R13", "number", "15", "Answer: 15 cm", 0, None),
    ("R14", "quantity", "15 cm", "Answer: 0.15 m", 1, None),
    ("R15", "quantity", "15 cm", "Answer: 15 kg", 0, None),
    ("R16", "quantity", "15 cm", "Answer: 15", 0, None),
    ("R17", "number", "15", "풀이 중…\nAnswe", 0, None),
    ("R18", "number", "15", "Answer: NaN", 0, None),
    ("R07-C", "mc", "C", "Answer: B or C", 0, None),
    ("LENGTH-ANSWER", "number", "15", "Answer: 15", 1, "length"),
]

V12_EXPECTED = {
    "R01": 1, "R02": 0, "R03": 0, "R04": 0, "R05": 1, "R06": 0,
    "R07": 0, "R08": 0, "R09": 0, "R10": 1, "R11": 0, "R12": 0,
    "R13": 1, "R14": 0, "R15": 0, "R16": 0, "R17": 0, "R18": 0,
    "R07-C": 1, "LENGTH-ANSWER": 1,
}
SYNTHETIC_STRICT_OVERRIDES = {"R09": 0, "R11": 0, "R14": 0}


def run_cases(policy: RewardPolicy, expected_profile: str = "proposed") -> list[dict]:
    if expected_profile not in {"proposed", "synthetic_strict"}:
        raise ValueError(f"Unknown expected profile: {expected_profile}")
    rows = []
    for case_id, kind, answer, output, expected, finish_reason in CASES:
        if expected_profile == "synthetic_strict":
            expected = SYNTHETIC_STRICT_OVERRIDES.get(case_id, expected)
        outcome = score_response(output, answer, kind, policy, finish_reason)
        rows.append({
            "case_id": case_id,
            "answer_type": kind,
            "gold_answer": answer,
            "model_output": output,
            "finish_reason": finish_reason or "unavailable",
            "expected_reward": expected,
            "actual_reward": outcome.reward,
            "v12_reward": V12_EXPECTED[case_id],
            "differs_from_v12": outcome.reward != V12_EXPECTED[case_id],
            "parsed_answer": outcome.parsed_answer,
            "reason": outcome.reason,
            "reward_version": policy.version,
            "pass": outcome.reward == expected,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-profile", choices=("proposed", "synthetic_strict"), default="proposed")
    args = parser.parse_args()
    policy = RewardPolicy.from_file(args.policy)
    rows = run_cases(policy, args.expected_profile)
    try:
        score_response("no answer", None, "mc", policy)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing gold answer must fail, never receive reward 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reward_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "reward_cases.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failed = [row["case_id"] for row in rows if not row["pass"]]
    print(f"{len(rows) - len(failed)}/{len(rows)} reward cases passed; failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
