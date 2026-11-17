"""Small, explicit multiple-choice reward for the GRPO plumbing pilot.

This is a proposed strict policy, not the team's final numerical/units policy.
"""

from __future__ import annotations

import re
from typing import Any


_ANSWER_LINE = re.compile(r"Answer:[ \t]*([A-D])[ \t]*")


def parse_final_mc_answer(text: str) -> str | None:
    """Read the last Answer: line; reject missing, lowercase, or ambiguous answers."""
    answer_lines = [line for line in text.splitlines() if line.startswith("Answer:")]
    if not answer_lines:
        return None
    match = _ANSWER_LINE.fullmatch(answer_lines[-1])
    return match.group(1) if match else None


def _content(completion: Any) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        completion = completion[0]
    if isinstance(completion, dict):
        completion = completion.get("content", "")
    return completion if isinstance(completion, str) else ""


def strict_mc_reward(
    completions: list[Any],
    answer: list[str],
    sample_id: list[str] | None = None,
    log_extra: Any = None,
    **_: Any,
) -> list[float]:
    """Give one point only when the final unambiguous answer equals the key."""
    if len(completions) != len(answer):
        raise ValueError("One gold answer is required for each completion")
    if any(not isinstance(gold, str) or gold not in {"A", "B", "C", "D"} for gold in answer):
        raise ValueError("Gold answers must be A, B, C, or D")
    parsed = [parse_final_mc_answer(_content(item)) for item in completions]
    if log_extra is not None:
        log_extra("sample_id", sample_id or [None] * len(completions))
        log_extra("parsed_answer", parsed)
        log_extra("gold_answer", answer)
    return [float(got is not None and got == gold) for got, gold in zip(parsed, answer, strict=True)]


def diagnostic_position_reward(completions: list[Any], **_: Any) -> list[float]:
    """Force reward variance for a plumbing check; never use for a real run."""
    if len(completions) % 4:
        raise ValueError("Diagnostic reward expects four completions per prompt")
    return [1.0 if index % 4 == 0 else 0.0 for index in range(len(completions))]
