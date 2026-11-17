"""Explicit GRPO answer rewards, including frozen MMDL v1.2 parity mode."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from v12_scorer import normalize_gold, score_v12


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_VALUE_UNIT_RE = re.compile(rf"({_NUMBER})(?:[ \t]+([A-Za-z]+))?")
_UNIT_TO_METRE = {"m": Decimal("1"), "cm": Decimal("0.01")}


@dataclass(frozen=True)
class RewardPolicy:
    version: str
    lowercase_mc: bool
    numeric_abs_tolerance: Decimal
    unitless_allows_unit: bool
    allow_cm_m_conversion: bool
    mode: str = "custom"

    @classmethod
    def from_file(cls, path: Path) -> "RewardPolicy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and set(raw) == {"version", "mode", "eval_repo_commit"}:
            if raw["mode"] != "v1_2" or raw["eval_repo_commit"] != "61025850563d0331e365bc974a63c0d3c4ed4c2c":
                raise ValueError("v1.2 reward must pin MMDL evaluator commit 6102585")
            if not isinstance(raw["version"], str) or not raw["version"].strip():
                raise ValueError("Reward policy version must be a non-empty string")
            return cls(raw["version"], False, Decimal("0"), False, False, mode="v1_2")
        keys = {
            "version", "lowercase_mc", "numeric_abs_tolerance",
            "unitless_allows_unit", "allow_cm_m_conversion",
        }
        if not isinstance(raw, dict) or set(raw) != keys:
            raise ValueError(f"Reward policy must contain exactly {sorted(keys)}")
        if not isinstance(raw["version"], str) or not raw["version"].strip():
            raise ValueError("Reward policy version must be a non-empty string")
        for key in ("lowercase_mc", "unitless_allows_unit", "allow_cm_m_conversion"):
            if type(raw[key]) is not bool:
                raise ValueError(f"Reward policy {key} must be boolean")
        if not isinstance(raw["numeric_abs_tolerance"], str):
            raise ValueError("numeric_abs_tolerance must be a decimal string")
        try:
            tolerance = Decimal(raw["numeric_abs_tolerance"])
        except (InvalidOperation, TypeError) as exc:
            raise ValueError("numeric_abs_tolerance must be a decimal string") from exc
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError("numeric_abs_tolerance must be finite and non-negative")
        return cls(
            version=raw["version"],
            lowercase_mc=raw["lowercase_mc"],
            numeric_abs_tolerance=tolerance,
            unitless_allows_unit=raw["unitless_allows_unit"],
            allow_cm_m_conversion=raw["allow_cm_m_conversion"],
        )


@dataclass(frozen=True)
class RewardOutcome:
    parsed_answer: Any
    reward: float
    reason: str


def final_answer_payload(text: str) -> tuple[str | None, str]:
    """Use the last Answer: line; an empty or malformed last line blocks earlier ones."""
    if not isinstance(text, str):
        return None, "non_text_completion"
    lines = [line for line in text.splitlines() if line.startswith("Answer:")]
    if not lines:
        return None, "missing_answer_line"
    payload = lines[-1][len("Answer:"):].strip()
    return (payload, "parsed") if payload else (None, "empty_last_answer")


def _decimal(value: str) -> Decimal | None:
    if not _NUMBER_RE.fullmatch(value):
        return None
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _value_unit(value: str) -> tuple[Decimal, str | None] | None:
    match = _VALUE_UNIT_RE.fullmatch(value)
    if match is None:
        return None
    number = _decimal(match.group(1))
    if number is None:
        return None
    unit = match.group(2)
    return number, unit.lower() if unit else None


def validate_gold(answer: str, answer_type: str) -> None:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gold answer must be a non-empty string")
    if answer_type == "mc":
        if not re.fullmatch(r"[A-Z]", answer):
            raise ValueError(f"Invalid MC gold answer: {answer!r}")
    elif answer_type == "number":
        if _decimal(answer) is None:
            raise ValueError(f"Invalid number gold answer: {answer!r}")
    elif answer_type == "quantity":
        parsed = _value_unit(answer)
        if parsed is None or parsed[1] not in _UNIT_TO_METRE:
            raise ValueError(f"Invalid quantity gold answer: {answer!r}")
    elif answer_type == "open":
        normalized = normalize_gold(answer)
        if isinstance(normalized, list) and (not normalized or not all(item.strip() for item in normalized)):
            raise ValueError(f"Invalid open gold answer list: {answer!r}")
    else:
        raise ValueError(f"Unsupported answer_type: {answer_type!r}")


def score_response(
    text: str,
    answer: str,
    answer_type: str,
    policy: RewardPolicy,
    finish_reason: str | None = None,
    options: list[str] | None = None,
) -> RewardOutcome:
    """Score the visible answer, even on a length stop if the answer is complete."""
    validate_gold(answer, answer_type)
    if policy.mode == "v1_2":
        correct, parsed_answer, reason = score_v12(text, answer, answer_type, options)
        if correct and finish_reason == "length":
            reason = "correct_length_stop"
        return RewardOutcome(parsed_answer, float(correct), reason)
    if answer_type == "open":
        raise ValueError("Open text answers require the v1.2 reward mode")
    if answer_type == "mc" and answer not in "ABCD":
        raise ValueError("Custom MC reward supports A-D; use v1.2 mode for other choices")
    payload, status = final_answer_payload(text)
    if payload is None:
        return RewardOutcome(None, 0.0, status)

    if answer_type == "mc":
        pattern = r"[A-Da-d]" if policy.lowercase_mc else r"[A-D]"
        if re.fullmatch(pattern, payload) is None:
            return RewardOutcome(None, 0.0, "ambiguous_or_invalid_mc")
        parsed_answer = payload.upper() if policy.lowercase_mc else payload
        correct = parsed_answer == answer
    elif answer_type == "number":
        parsed = _value_unit(payload)
        if parsed is None:
            return RewardOutcome(None, 0.0, "invalid_number")
        value, unit = parsed
        if unit is not None and not policy.unitless_allows_unit:
            return RewardOutcome(payload, 0.0, "unexpected_unit")
        if unit is not None and unit not in _UNIT_TO_METRE:
            return RewardOutcome(payload, 0.0, "unknown_unit")
        parsed_answer = payload
        correct = abs(value - Decimal(answer)) <= policy.numeric_abs_tolerance
    else:
        parsed = _value_unit(payload)
        if parsed is None or parsed[1] is None:
            return RewardOutcome(None, 0.0, "missing_or_invalid_unit")
        value, unit = parsed
        gold_value, gold_unit = _value_unit(answer)  # validate_gold checked this.
        parsed_answer = payload
        if unit not in _UNIT_TO_METRE:
            return RewardOutcome(parsed_answer, 0.0, "unknown_unit")
        if unit != gold_unit and not policy.allow_cm_m_conversion:
            return RewardOutcome(parsed_answer, 0.0, "unit_conversion_disabled")
        correct = value * _UNIT_TO_METRE[unit] == gold_value * _UNIT_TO_METRE[gold_unit]

    if correct:
        reason = "correct_length_stop" if finish_reason == "length" else "correct"
        return RewardOutcome(parsed_answer, 1.0, reason)
    return RewardOutcome(parsed_answer, 0.0, "wrong_answer")


def completion_content(completion: Any) -> str:
    if isinstance(completion, list):
        completion = completion[0] if completion else None
    if isinstance(completion, dict):
        completion = completion.get("content")
    return completion if isinstance(completion, str) else ""


def make_grpo_reward(policy: RewardPolicy, observed: list[float] | None = None):
    """Return a TRL 1.14.2 reward callback with per-completion diagnostics."""

    def reward_func(completions, answer, answer_type, sample_id=None, options=None, log_extra=None, **kwargs):
        count = len(completions)
        if len(answer) != count or len(answer_type) != count:
            raise ValueError("TRL reward columns do not match completion count")
        if sample_id is None:
            sample_id = [None] * count
        if len(sample_id) != count:
            raise ValueError("sample_id count does not match completions")
        finish_reasons = kwargs.get("finish_reason")
        if finish_reasons is None:
            finish_reasons = [None] * count
        if len(finish_reasons) != count:
            raise ValueError("finish_reason count does not match completions")
        if options is None:
            options = [None] * count
        if len(options) != count:
            raise ValueError("option list count does not match completions")
        outcomes = [
            score_response(completion_content(item), gold, kind, policy, reason, item_options)
            for item, gold, kind, reason, item_options in zip(
                completions, answer, answer_type, finish_reasons, options, strict=True
            )
        ]
        if log_extra is not None:
            log_extra("sample_id", sample_id)
            # TRL writes this diagnostic column to Parquet. v1.2 open answers
            # can contain both numbers and strings in the parsed candidate list,
            # which PyArrow cannot infer as one column type.
            log_extra("parsed_answer", [json.dumps(item.parsed_answer, ensure_ascii=False) for item in outcomes])
            log_extra("reward_reason", [item.reason for item in outcomes])
            log_extra("gold_answer", answer)
        values = [item.reward for item in outcomes]
        if observed is not None:
            observed.extend(values)
        return values

    return reward_func
