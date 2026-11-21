"""Audit exact ViRL500 GRPO coverage and bounded DAPO attempted/used visits."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def audit(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    method = run["method"]
    mode = run["mode"]
    with Path(run["train_file"]).open(encoding="utf-8") as stream:
        train_ids = [str(json.loads(line)["sample_id"]) for line in stream if line.strip()]
    if len(train_ids) != 500 or len(set(train_ids)) != 500:
        raise ValueError("Expected 500 unique ViRL train IDs")
    expected = set(train_ids)
    by_step: dict[int, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: {"attempt": collections.Counter(), "used": collections.Counter()}
    )
    with (run_dir / "visits.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("event")
            if kind not in ("attempt", "used"):
                raise ValueError(f"Bad visit event at line {line_number}: {kind}")
            step = int(row["step"])
            sample_id = str(row["sample_id"])
            if sample_id not in expected:
                raise ValueError(f"Unknown ID at line {line_number}: {sample_id}")
            by_step[step][kind][sample_id] += 1
    expected_steps = 1 if mode == "probe" else 15
    if set(by_step) != set(range(1, expected_steps + 1)):
        raise ValueError(f"Expected steps 1..{expected_steps}, got {sorted(by_step)}")
    attempts_total = collections.Counter()
    used_total = collections.Counter()
    steps = []
    for step in sorted(by_step):
        attempts = by_step[step]["attempt"]
        used = by_step[step]["used"]
        attempts_total.update(attempts)
        used_total.update(used)
        n_attempts = sum(attempts.values())
        n_used = sum(used.values())
        if n_used != 100 or n_attempts % 50 != 0:
            raise ValueError(f"Step {step}: attempted={n_attempts}, used={n_used}")
        if any(used[sample_id] > attempts[sample_id] for sample_id in used):
            raise ValueError(f"Step {step}: used more than attempted for an ID")
        draws = n_attempts // 50
        if method == "grpo" and (n_attempts != 100 or draws != 2 or used != attempts):
            raise ValueError(f"GRPO step {step} did not use both 50-question draws exactly")
        if method == "dapo_core" and draws > 24:
            raise ValueError(f"DAPO step {step} exceeded bounded 24 draws")
        steps.append({"step": step, "attempted_questions": n_attempts,
                      "used_questions": n_used, "draws": draws,
                      "used_unique_ids": len(used),
                      "used_duplicate_visits": n_used - len(used),
                      "generated_rollouts": n_attempts * 5,
                      "used_rollouts": n_used * 5,
                      "group_acceptance_rate": n_used / n_attempts})
    if method == "grpo" and mode == "full":
        if any(attempts_total[sample_id] != 3 or used_total[sample_id] != 3 for sample_id in expected):
            raise ValueError("GRPO did not visit/use all 500 IDs exactly 3 times")
    if method == "grpo" and mode == "probe":
        if len(attempts_total) != 100 or len(used_total) != 100:
            raise ValueError("GRPO probe did not use 100 distinct IDs")
    report = {
        "status": "visit_audit_passed",
        "method": method, "mode": mode,
        "expected_train_ids": 500,
        "outer_rollout_cycles": expected_steps,
        "attempted_question_visits": sum(attempts_total.values()),
        "used_question_visits": sum(used_total.values()),
        "attempted_unique_ids": len(attempts_total),
        "used_unique_ids": len(used_total),
        "not_attempted_ids": len(expected - attempts_total.keys()),
        "not_used_ids": len(expected - used_total.keys()),
        "attempt_visit_histogram": dict(sorted(collections.Counter(attempts_total.values()).items())),
        "used_visit_histogram": dict(sorted(collections.Counter(used_total.values()).items())),
        "generated_rollouts": sum(attempts_total.values()) * 5,
        "used_rollouts": sum(used_total.values()) * 5,
        "steps": steps,
        "epoch_interpretation": (
            "all 500 IDs exactly 3 fresh-generation visits" if method == "grpo" and mode == "full"
            else "nominal outer cycles; online filtering changes per-ID visits" if method == "dapo_core"
            else "one-step batch100 probe"
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    output = args.output or args.run_dir / "visit_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "outer_rollout_cycles": result["outer_rollout_cycles"],
                      "attempted_question_visits": result["attempted_question_visits"],
                      "used_question_visits": result["used_question_visits"]}, sort_keys=True))


if __name__ == "__main__":
    main()
