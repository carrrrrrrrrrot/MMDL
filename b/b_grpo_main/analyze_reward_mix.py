"""Compare frozen v1.2 reward variance with saved synthetic pilot rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reward import RewardPolicy, score_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args()
    policy = RewardPolicy.from_file(args.policy)
    if policy.mode != "v1_2":
        raise ValueError("This analysis requires the frozen v1.2 policy")
    groups = defaultdict(list)
    rows = 0
    changed = 0
    with args.rollouts.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            options = list(row["choices"].values())
            reward = score_response(row["raw_output"], row["gold_answer"], "mc", policy, options=options).reward
            groups[row["group_id"]].append((row["reward"], reward))
            rows += 1
            changed += reward != row["reward"]
    if any(len(values) != 4 for values in groups.values()):
        raise ValueError("Expected four completions per saved GRPO group")
    strict_mixed = sum(len({value[0] for value in values}) > 1 for values in groups.values())
    v12_mixed = sum(len({value[1] for value in values}) > 1 for values in groups.values())
    print(json.dumps({"rollouts": rows, "groups": len(groups), "strict_mixed_groups": strict_mixed,
                      "v12_mixed_groups": v12_mixed, "reward_changed": changed}, indent=2))


if __name__ == "__main__":
    main()
