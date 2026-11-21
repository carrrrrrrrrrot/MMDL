"""Parse Stage1 config with pinned EasyR1 and check single-GPU safety knobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.config import PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(args.config))
    parsed: PPOConfig = OmegaConf.to_object(config)
    parsed.deep_post_init()
    actor = parsed.worker.actor
    assert actor.fsdp.use_orig_params is False
    assert actor.model.freeze_vision_tower is False
    assert actor.model.lora.rank == 16
    assert actor.model.lora.target_modules == "q_proj,k_proj,v_proj,o_proj"
    assert actor.model.lora.exclude_modules == ".*visual.*"
    assert parsed.worker.rollout.n == 5
    assert parsed.worker.rollout.max_num_batched_tokens > (
        parsed.data.max_prompt_length + 2048
    )
    print(json.dumps({
        "use_orig_params": actor.fsdp.use_orig_params,
        "freeze_vision_tower_extra_flag": actor.model.freeze_vision_tower,
        "lora_target_modules": actor.model.lora.target_modules,
        "lora_exclude_modules": actor.model.lora.exclude_modules,
        "rollouts_per_question": parsed.worker.rollout.n,
        "max_num_batched_tokens": parsed.worker.rollout.max_num_batched_tokens,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
