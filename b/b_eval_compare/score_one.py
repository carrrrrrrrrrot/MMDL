"""Read-only paired v1.2 comparison for one completed MMMU seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_runs import compare_seed, load_rows, load_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-seed", type=Path, required=True)
    parser.add_argument("--candidate-seed", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()
    _, stats = compare_seed(
        args.seed,
        load_rows(args.baseline_seed),
        load_rows(args.candidate_seed),
        load_scorer(args.repo_root),
        False,
        900,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
