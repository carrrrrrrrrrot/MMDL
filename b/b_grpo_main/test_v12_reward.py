"""Check B's dependency-light v1.2 reward against A's exact 20-case fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reward import RewardPolicy, make_grpo_reward, score_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("a_shared_reward_cases.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("reward_policy.v12.json"))
    args = parser.parse_args()
    policy = RewardPolicy.from_file(args.policy)
    rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 20:
        raise AssertionError(f"Expected 20 A cases, got {len(rows)}")
    for row in rows:
        kind = "mc" if row["question_type"] == "multiple-choice" else "open"
        actual = score_response(
            row["model_output"], row["gold"], kind, policy,
            row.get("finish_reason"), row["options"],
        )
        expected = float(row["v1_2_score"])
        if actual.reward != expected:
            raise AssertionError((row["case_id"], expected, actual))
    try:
        score_response("no answer", None, "mc", policy, options=["A", "B", "C", "D"])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing gold must fail")
    # Exercise the exact callback shape TRL uses for a four-rollout group.
    sample = rows[0]
    reward = make_grpo_reward(policy)
    values = reward(
        completions=[[{"role": "assistant", "content": sample["model_output"]}]] * 4,
        answer=[sample["gold"]] * 4,
        answer_type=["mc"] * 4,
        options=[sample["options"]] * 4,
        sample_id=[sample["case_id"]] * 4,
    )
    if values != [1.0] * 4:
        raise AssertionError(values)
    print("PASS: A's 20/20 v1.2 cases, null-gold guard, and TRL four-rollout callback")


if __name__ == "__main__":
    main()
