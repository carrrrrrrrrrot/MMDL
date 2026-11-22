"""Extract per-update training metrics from EasyR1 console or Runner stdout.

Auto mode first reads console.log. It only uses a completed run's copied
runner_stdout.log when console forwarding omitted a step or metric block,
and the completion report records the copy's path and SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(path: Path, expected_steps: int, source_kind: str) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    steps: dict[int, dict[str, float]] = {}
    current: int | None = None
    section: str | None = None
    lines: list[str] = []
    for raw_line in ANSI.sub("", raw).replace("\r", "\n").splitlines():
        if source_kind == "console":
            runner = RUNNER.search(raw_line)
            if not runner:
                continue
            line = raw_line[runner.end():].rstrip()
        elif source_kind == "ray_runner_stdout":
            line = raw_line.rstrip()
        else:
            raise ValueError(f"Unknown metric source kind: {source_kind}")
        lines.append(line)
    for index, line in enumerate(lines):
        step = STEP.fullmatch(line)
        if step:
            if index + 1 >= len(lines) or lines[index + 1] != "actor:":
                current = None
                section = None
                continue
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
            value = float(entry.group(2))
            if not math.isfinite(value):
                raise ValueError(f"Invalid numeric metric at step {current}: {section}.{entry.group(1)}={value}")
            steps[current][f"{section}.{entry.group(1)}"] = value
    if sorted(steps) != list(range(1, expected_steps + 1)):
        raise ValueError(f"Expected Steps 1..{expected_steps}, found {sorted(steps)}")
    required = {f"{section}.{key}" for section, key in FIELDS}
    for number, metrics in steps.items():
        missing = required - metrics.keys()
        if missing:
            raise ValueError(f"Step {number} missing metrics: {sorted(missing)}")
    return {"steps": [{"step": number, **steps[number]} for number in sorted(steps)]}


def auto_parse(path: Path, expected_steps: int) -> dict:
    if path.name == "runner_stdout.log":
        return parse(path, expected_steps, "ray_runner_stdout")
    try:
        return parse(path, expected_steps, "console")
    except ValueError as exc:
        missing_steps = str(exc).startswith("Expected Steps 1..")
        missing_fields = str(exc).startswith("Step ") and "missing metrics" in str(exc)
        if not (missing_steps or missing_fields):
            raise
        run_dir = path.resolve(strict=True).parent
        report = json.loads((run_dir / "completion_report.json").read_text(encoding="utf-8"))
        source = report.get("metric_source", {})
        if report.get("status") != "complete" or source.get("kind") != "ray_runner_stdout":
            raise ValueError("Console metrics are partial and no completed Runner stdout source is recorded") from exc
        raw = Path(source.get("path", "")).resolve(strict=True)
        if raw != (run_dir / "runner_stdout.log").resolve(strict=True):
            raise ValueError("Recorded Runner stdout is outside the run directory")
        if sha256(raw) != source.get("sha256"):
            raise ValueError("Recorded Runner stdout SHA256 differs")
        return parse(raw, expected_steps, "ray_runner_stdout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--expected-steps", type=int, default=15,
                        help="Expected update count; use 1 for a completed probe")
    parser.add_argument("--source-kind", choices=("auto", "console", "ray_runner_stdout"),
                        default="auto")
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    result = (auto_parse(args.log_path, args.expected_steps) if args.source_kind == "auto"
              else parse(args.log_path, args.expected_steps, args.source_kind))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
