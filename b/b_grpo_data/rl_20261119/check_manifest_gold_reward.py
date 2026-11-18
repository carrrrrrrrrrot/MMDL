#!/usr/bin/env python3
"""Require every selected gold answer to earn v1.2 reward from its final line."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--b-main", required=True, type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.b_main))
    from v12_scorer import score_v12

    fails = []
    kinds = Counter()
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        options = list(row["choices"].values()) if row["answer_type"] == "mc" else None
        good, parsed, reason = score_v12("Answer: " + row["answer"], row["answer"], row["answer_type"], options)
        kinds[row["answer_type"]] += 1
        if not good:
            fails.append({"id": row["original_id"], "answer": row["answer"], "parsed": parsed, "reason": reason})
    print(json.dumps({"rows": sum(kinds.values()), "kinds": kinds, "failed": len(fails), "examples": fails[:10]}, ensure_ascii=False))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
