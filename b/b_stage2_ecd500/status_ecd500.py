"""Read-only snapshot of an ECD500 training run and its saved checkpoints."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNNER = re.compile(r"\(Runner pid=\d+\)\s?")
STEP = re.compile(r"\bStep (\d+)\s*$")
FAILURE = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|Error executing job", re.I)


def snapshot(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    console_path = run_dir / "console.log"
    console = ANSI.sub("", console_path.read_text(encoding="utf-8", errors="replace")) if console_path.exists() else ""
    lines = console.replace("\r", "\n").splitlines()
    metric_lines = []
    for line in lines:
        match = RUNNER.search(line)
        if match:
            metric_lines.append(line[match.end():].rstrip())
    steps = set()
    failures = []
    for i, line in enumerate(metric_lines):
        match = STEP.search(line)
        if match and i + 1 < len(metric_lines) and metric_lines[i + 1] == "actor:":
            steps.add(int(match.group(1)))
    for line in lines:
        if FAILURE.search(line):
            failures.append(line[-500:])

    visits = collections.defaultdict(lambda: collections.Counter())
    visit_path = run_dir / "visits.jsonl"
    if visit_path.exists():
        with visit_path.open(encoding="utf-8") as stream:
            for raw in stream:
                if raw.strip():
                    row = json.loads(raw)
                    visits[int(row["step"])][row["event"]] += 1

    checkpoints = []
    for path in sorted((run_dir / "checkpoints").glob("global_step_*")):
        files = [p for p in path.rglob("*") if p.is_file()]
        checkpoints.append({
            "step": int(path.name.removeprefix("global_step_")),
            "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "adapter_present": (path / "actor/lora_adapter/adapter_model.safetensors").is_file(),
            "files": [str(p.relative_to(path)) for p in files],
        })
    return {
        "run_dir": str(run_dir),
        "status": manifest.get("status", "missing"),
        "method": manifest.get("method"),
        "mode": manifest.get("mode"),
        "start_utc": manifest.get("start_utc"),
        "end_utc": manifest.get("end_utc"),
        "process_exit_code": manifest.get("process_exit_code"),
        "completed_actor_steps": sorted(steps),
        "visits": {str(k): dict(v) for k, v in sorted(visits.items())},
        "checkpoints": checkpoints,
        "failure_lines": failures[-3:],
        "console_bytes": console_path.stat().st_size if console_path.exists() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
