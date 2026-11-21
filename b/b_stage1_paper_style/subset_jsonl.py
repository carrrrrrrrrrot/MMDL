"""Take a deterministic SHA-ranked subset of JSONL rows by sample_id."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample ID")
    if not 1 <= args.count <= len(rows):
        raise ValueError("Invalid subset count")
    rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}|{row['sample_id']}".encode()).digest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows[:args.count]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": args.count, "ids": [row["sample_id"] for row in rows[:args.count]],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
