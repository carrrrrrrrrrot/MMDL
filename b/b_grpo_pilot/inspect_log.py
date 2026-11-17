"""Print a small, readable sample of TRL completion logs."""

from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    table = pq.read_table(args.parquet)
    print("columns:", table.column_names)
    for row in table.slice(0, args.rows).to_pylist():
        print(json.dumps(row, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
