"""EasyR1 reward adapter for the frozen MMDL v1.2 binary scorer.

The default optimized reward remains exactly 0/1. Optional soft-overlong
shaping is explicit and reports raw accuracy separately. Online DAPO group
filtering must use the ``accuracy`` key, never the shaped ``overall`` key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v12_scorer import final_answer, score_v12  # noqa: E402


REWARD_NAME = "mmdl_v12"
REWARD_TYPE = "batch"


def _length_penalty(length: int, cap: int, buffer: int) -> float:
    if cap < 1 or buffer < 1 or buffer > cap:
        raise ValueError("Invalid completion cap or overlong buffer")
    start = cap - buffer
    if length <= start:
        return 0.0
    return -min(1.0, (length - start) / buffer)


def compute_score(
    reward_inputs: list[dict[str, Any]],
    reward_mode: str = "v12_binary",
    max_response_length: int = 1024,
    overlong_buffer_length: int = 256,
    overlong_penalty_factor: float = 1.0,
) -> list[dict[str, float]]:
    if reward_mode not in {"v12_binary", "v12_soft_overlong"}:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")
    if overlong_penalty_factor < 0:
        raise ValueError("overlong_penalty_factor must be nonnegative")
    scores = []
    for item in reward_inputs:
        gold = json.loads(item["ground_truth"])
        if set(gold) != {"answer", "answer_type", "options", "sample_id"}:
            raise ValueError("Malformed versioned ground truth")
        response = item["response"]
        correct, _, _ = score_v12(
            response,
            gold["answer"],
            gold["answer_type"],
            gold["options"],
        )
        accuracy = float(correct)
        length = int(item["response_length"])
        overlong = (
            _length_penalty(length, max_response_length, overlong_buffer_length)
            if reward_mode == "v12_soft_overlong"
            else 0.0
        )
        overall = accuracy + overlong_penalty_factor * overlong
        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy,
                "overlong": overlong,
                "answer_present": float(final_answer(response) is not None),
                "at_cap_proxy": float(length >= max_response_length),
            }
        )
    return scores
