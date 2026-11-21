"""Audit attempted and optimizer-used EasyR1 prompt IDs after an RL run."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                sample_id = str(json.loads(line)["sample_id"])
                if sample_id in ids:
                    raise ValueError(f"Duplicate train ID: {sample_id}")
                ids.add(sample_id)
    return ids


def summarize(events_path: Path, train_path: Path, rollout_count: int) -> dict:
    expected = load_ids(train_path)
    counts = {"attempt": collections.Counter(), "used": collections.Counter()}
    steps = set()
    with events_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("event")
            if kind not in counts:
                raise ValueError(f"Unexpected event at line {line_number}: {kind}")
            sample_id = str(event["sample_id"])
            if sample_id not in expected:
                raise ValueError(f"Unexpected sample ID at line {line_number}: {sample_id}")
            counts[kind][sample_id] += 1
            steps.add(event["step"])
    used_exceeds_attempts = {
        sample_id: n for sample_id, n in counts["used"].items()
        if n > counts["attempt"][sample_id]
    }
    if used_exceeds_attempts:
        raise ValueError(f"Used count exceeds attempted count: {used_exceeds_attempts}")
    return {
        "train_unique_ids": len(expected),
        "logged_steps": sorted(steps),
        "rollouts_per_attempted_prompt": rollout_count,
        "attempted_prompt_visits": sum(counts["attempt"].values()),
        "used_prompt_visits": sum(counts["used"].values()),
        "attempted_unique_ids": len(counts["attempt"]),
        "used_unique_ids": len(counts["used"]),
        "attempted_rollouts_upper_bound": sum(counts["attempt"].values()) * rollout_count,
        "unused_ids": sorted(expected - counts["attempt"].keys()),
        "unused_for_optimizer_ids": sorted(expected - counts["used"].keys()),
        "attempt_visit_histogram": dict(sorted(collections.Counter(counts["attempt"].values()).items())),
        "used_visit_histogram": dict(sorted(collections.Counter(counts["used"].values()).items())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visits", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.visits, args.train, args.rollouts)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
