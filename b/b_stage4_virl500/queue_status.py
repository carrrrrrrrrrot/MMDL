"""Atomically record sequential queue phase transitions in JSON and JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("phase")
    parser.add_argument("state", choices=("running", "complete", "failed"))
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": args.phase,
        "state": args.state,
        "exit_code": args.exit_code,
    }
    with (args.queue_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    target = args.queue_dir / "status.json"
    temp = args.queue_dir / "status.json.tmp"
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
