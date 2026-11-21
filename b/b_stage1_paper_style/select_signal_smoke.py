"""Select a deterministic ECD training smoke set with previously mixed rewards.

This is only to verify a nonzero learning signal, never an evaluation set.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--prior-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.converted.open(encoding="utf-8")]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("Duplicate converted sample IDs")
    frame = pd.read_csv(args.prior_rollouts, usecols=["step", "sample_id", "reward"])
    frame = frame[frame.sample_id.isin(by_id)]
    grouped = frame.groupby("sample_id").agg(
        first_step=("step", "min"),
        reward_min=("reward", "min"),
        reward_max=("reward", "max"),
    )
    mixed = grouped[(grouped.reward_min <= 0) & (grouped.reward_max >= 1)]
    mixed = mixed.sort_values(["first_step", "reward_max"], ascending=[True, False])
    chosen = mixed.index.tolist()[: args.limit]
    if len(chosen) != args.limit:
        raise RuntimeError(f"Only {len(chosen)} mixed-reward IDs, need {args.limit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for sample_id in chosen:
            stream.write(json.dumps(by_id[sample_id], ensure_ascii=False) + "\n")
    print(json.dumps({
        "converted_ids": len(rows),
        "prior_rollout_rows": len(frame),
        "mixed_ids": len(mixed),
        "selected_ids": chosen,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
