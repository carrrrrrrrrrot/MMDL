"""Resolve the frozen launcher overrides through EasyR1 config without Ray/GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.trainer.config import PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    root = args.package.resolve(strict=True)
    scripts = root / "scripts"
    config = OmegaConf.merge(
        OmegaConf.structured(PPOConfig()),
        OmegaConf.load(scripts / "config_smoke.yaml"),
        OmegaConf.from_cli([
            f"data.train_files={scripts / 'ecd20_signal_smoke.jsonl'}",
            f"data.val_files={scripts / 'ecd5_holdout_easyr1.jsonl'}",
            f"worker.reward.reward_function={scripts / 'reward_v12.py'}:compute_score",
            "data.max_response_length=1024",
        ]),
    )
    parsed: PPOConfig = OmegaConf.to_object(config)
    parsed.deep_post_init()
    reward_path = scripts / "reward_v12.py"
    if str(parsed.worker.reward.reward_function) != str(reward_path):
        raise ValueError(f"Resolved reward path differs: {parsed.worker.reward.reward_function!r}")
    if parsed.worker.reward.reward_function_name != "compute_score":
        raise ValueError(f"Resolved reward function differs: {parsed.worker.reward.reward_function_name!r}")
    assert Path(parsed.data.train_files).is_file()
    assert Path(parsed.data.val_files).is_file()
    assert reward_path.is_file()
    assert parsed.data.max_response_length == 1024
    print(json.dumps({"resolved_reward_function": str(reward_path) + ":compute_score",
                      "train": str(parsed.data.train_files),
                      "validation": str(parsed.data.val_files),
                      "cap": parsed.data.max_response_length}, sort_keys=True))


if __name__ == "__main__":
    main()
