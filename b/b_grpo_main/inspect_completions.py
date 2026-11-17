"""Print a compact schema/sample from a TRL completion Parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.parquet)
    print(json.dumps({"columns": list(frame.columns), "rows": len(frame)}, ensure_ascii=False))
    for index, row in frame.head(2).iterrows():
        print(json.dumps({str(key): str(value)[:300] for key, value in row.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
