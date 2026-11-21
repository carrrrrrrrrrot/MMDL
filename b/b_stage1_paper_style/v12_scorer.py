"""Dependency-light scoring parity with MMDL MMMU v1.2 (commit 6102585).

Adapted from eval_tasks/mmmu_mmdl/mmdl_utils.py and the repo-bundled MMMU
utilities at that commit. This module excludes the official random MC fallback:
MMDL's strict v1.2 score explicitly overwrites fallback predictions with an
empty answer. Tests compare this module to the frozen evaluator on A's cases.

Source SHA256:
  mmdl_utils.py          289c4e30b278d1ebc0adedebc188b999314a1253968373f64d5bbdd603b9abdc
  mmmu/utils.py          9f87d5b9b370210de4e532e50a0b18d7a0b0fa9f0924f6d66a77d8095f5b2a74
  mmmu_mcq_utils.py      32cfd79be959ea7cc21a64544bde57e64af2c7daf6669e80296a5b6b091301a6
"""

from __future__ import annotations

import ast
import re
from typing import Any


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_LATEX_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_TEXT = re.compile(r"\\(?:text|mathrm|textbf)\s*\{([^{}]*)\}")


def final_answer(response: str) -> str | None:
    """Take the first nonempty line after the last case-insensitive Answer:."""
    index = response.lower().rfind("answer:")
    if index < 0:
        return None
    for line in response[index + len("answer:") :].splitlines():
        line = _BOXED.sub(r"\1", line).replace("*", "").replace("`", "").strip()
        if line:
            return line
    return None


def normalize_latex(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_TEXT.sub(r"\1", text)
        text = _LATEX_FRAC.sub(r"\1/\2", text)
    for token in ("\\left", "\\right", "\\!", "$"):
        text = text.replace(token, "")
    for token in ("\\,", "\\;"):
        text = text.replace(token, " ")
    return text.strip()


def normalize_gold(gold: str | list[str]) -> str | list[str]:
    if isinstance(gold, str) and gold.strip().startswith("["):
        try:
            parsed = ast.literal_eval(gold)
        except (ValueError, SyntaxError):
            return gold
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return gold


def _normal_strings(value: str) -> list[str | float]:
    value = value.strip()
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        number = None
    if number is not None:
        return [round(number, 2)]
    lowered = value.lower()
    if len(lowered) == 1:
        return [" " + lowered, lowered + " "]
    return [lowered]


def _extract_numbers(value: str) -> list[str]:
    comma_numbers = re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", value)
    scientific = re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", value)
    simple = re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", value)
    return comma_numbers + scientific + simple


def parse_open(response: str) -> list[str | float]:
    response = response.strip().strip(".").lower()
    sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
    indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
    key_responses: list[str] = []
    for index, sub in enumerate(sub_responses):
        choices = indicators + (["="] if index == len(sub_responses) - 1 else [])
        shortest = None
        for marker in choices:
            if marker in sub:
                tail = sub.split(marker)[-1].strip()
                if not shortest or len(tail) < len(shortest):
                    shortest = tail
        if shortest and shortest.strip() not in {":", ",", ".", "!", "?", ";", "'"}:
            key_responses.append(shortest)
    if not key_responses:
        key_responses = [response]
    candidates = key_responses.copy()
    for part in key_responses:
        candidates.extend(_extract_numbers(part))
    normalized: list[str | float] = []
    for part in candidates:
        normalized.extend(_normal_strings(part))
    return list(set(normalized))


def eval_open(gold: str | list[str], parsed: list[str | float]) -> bool:
    if isinstance(gold, list):
        normal_answers = [item for answer in gold for item in _normal_strings(answer)]
    else:
        normal_answers = _normal_strings(gold)
    for prediction in parsed:
        if isinstance(prediction, str):
            if any(isinstance(answer, str) and answer in prediction for answer in normal_answers):
                return True
        elif prediction in normal_answers:
            return True
    return False


def parse_mc(response: str, options: list[str]) -> str | None:
    """Follow the official MMMU MC parser, returning None for strict fallback."""
    choices = [chr(ord("A") + i) for i in range(len(options))]
    option_text = dict(zip(choices, options, strict=True))
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_answer = True
    with_brackets = False
    candidates = [choice for choice in choices if f"({choice})" in response]
    if candidates:
        with_brackets = True
    if not candidates:
        candidates = [choice for choice in choices if f"{choice} " in response]
    if not candidates:
        candidates = [choice for choice in choices if f"{choice}." in response]
    if not candidates and len(response.split()) > 5:
        candidates = [choice for choice in choices if option_text[choice].lower() in response.lower()]
        index_answer = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if index_answer:
        if with_brackets:
            positions = [response.rfind(f"({choice})") for choice in candidates]
        else:
            positions = [response.rfind(f" {choice} ") for choice in candidates]
    else:
        positions = [response.lower().rfind(option_text[choice].lower()) for choice in candidates]
    return candidates[max(range(len(positions)), key=positions.__getitem__)]


def score_v12(
    response: str,
    gold: str | list[str],
    answer_type: str,
    options: list[str] | None = None,
) -> tuple[bool, Any, str]:
    """Return (correct, parsed answer, reason) under the frozen strict v1.2 rule."""
    answer = final_answer(response)
    if answer is None:
        return False, None, "missing_final_answer"
    if answer_type == "mc":
        if not options:
            raise ValueError("v1.2 multiple-choice reward requires option texts in A.. order")
        choice = parse_mc(answer, options)
        if choice is None:
            return False, None, "no_mc_candidate"
        return choice == gold, choice, "correct" if choice == gold else "wrong_answer"
    if answer_type not in {"open", "number", "quantity"}:
        raise ValueError(f"Unsupported v1.2 answer type: {answer_type}")
    parsed = parse_open(normalize_latex(answer))
    correct = eval_open(normalize_gold(gold), parsed)
    return correct, sorted(parsed, key=lambda item: (str(type(item)), str(item))), "correct" if correct else "wrong_answer"
