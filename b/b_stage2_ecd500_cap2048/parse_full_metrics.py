"""Extract per-update training metrics from a completed EasyR1 console log.

This reads the log only and emits JSON to stdout. It deliberately excludes
progress-bar fragments and validation metrics from the training series.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNNER = re.compile(r"\(Runner pid=\d+\)\s?")
STEP = re.compile(r"Step (\d+)")
FIELDS = {
    ("actor", "grad_norm"),
    ("actor", "lr"),
    ("actor", "pg_loss"),
    ("actor", "ppo_kl"),
    ("reward", "accuracy"),
    ("reward", "answer_present"),
    ("reward", "at_cap_proxy"),
    ("reward", "overlong"),
    ("response_length", "mean"),
    ("perf", "time_per_step"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("console_log", type=Path)
    parser.add_argument("--expected-steps", type=int, default=15,
                        help="Expected update count; use 1 for a completed probe")
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    raw = args.console_log.read_text(encoding="utf-8", errors="replace")
    steps: dict[int, dict[str, float]] = {}
    current: int | None = None
    section: str | None = None
    for raw_line in ANSI.sub("", raw).replace("\r", "\n").splitlines():
        runner = RUNNER.search(raw_line)
        if not runner:
            continue
        line = raw_line[runner.end():].rstrip()
        step = STEP.fullmatch(line)
        if step:
            current = int(step.group(1))
            if current in steps:
                # EasyR1 repeats the final step header to print heldout
                # validation. Keep the earlier training metric block.
                current = None
                section = None
                continue
            steps[current] = {}
            section = None
            continue
        if current is None:
            continue
        if line == "Start generating batch..." or line == "Final validation metrics:":
            current = None
            section = None
            continue
        if line.endswith(":") and not line.startswith(" "):
            section = line[:-1]
            continue
        if section is None:
            continue
        entry = re.fullmatch(r"\s+([^:]+):\s+(.+)", line)
        if entry and (section, entry.group(1)) in FIELDS:
            steps[current][f"{section}.{entry.group(1)}"] = float(entry.group(2))
    if sorted(steps) != list(range(1, args.expected_steps + 1)):
        raise ValueError(f"Expected Steps 1..{args.expected_steps}, found {sorted(steps)}")
    required = {f"{section}.{key}" for section, key in FIELDS}
    for number, metrics in steps.items():
        missing = required - metrics.keys()
        if missing:
            raise ValueError(f"Step {number} missing metrics: {sorted(missing)}")
    print(json.dumps({"steps": [{"step": number, **steps[number]} for number in sorted(steps)]}, indent=2))


if __name__ == "__main__":
    main()
