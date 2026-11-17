"""Print JSONL field names/types only; no image/text payloads."""

import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as stream:
    row = json.loads(next(stream))
print({key: type(value).__name__ for key, value in row.items()})
for key, value in row.items():
    if isinstance(value, dict):
        print(key, {name: type(item).__name__ for name, item in value.items()})
