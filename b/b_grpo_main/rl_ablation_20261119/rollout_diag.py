"""Read-only diagnostics for TRL completion parquet files.

Run on the 6000 host with the GRPO training environment. No training artifact is
changed; this script only reads existing parquet files and prints JSON.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_COMPLETIONS = Path(
    "/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2/"
    "runs/ecd_grpo/trainer/completions"
)
DEFAULT_MODEL = Path(
    "/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
ANSWER_LINE = re.compile(r"(?im)^\s*Answer\s*:\s*\S+")
FILE_STEP = re.compile(r"completions_(\d+)\.parquet$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions-dir", type=Path, default=DEFAULT_COMPLETIONS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--through-step", type=int, default=None)
    args = parser.parse_args()

    files = []
    for name in sorted(glob.glob(str(args.completions_dir / "completions_*.parquet"))):
        match = FILE_STEP.search(name)
        if match and (args.through_step is None or int(match.group(1)) <= args.through_step):
            files.append(name)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    rows = []
    unreadable = []
    for name in files:
        try:
            rows.extend(
                pq.read_table(
                    name,
                    columns=["step", "sample_id", "completion", "reward_reason"],
                ).to_pylist()
            )
        except Exception as exc:  # A live last file can be observed mid-write.
            unreadable.append({"file": name, "error": str(exc)})

    reasons = collections.Counter(str(row["reward_reason"]) for row in rows)
    by_step = collections.Counter(row["step"] for row in rows)
    no_answer = []
    clipped = []
    for row in rows:
        completion = row["completion"] or ""
        if not ANSWER_LINE.search(completion):
            no_answer.append(row)
        if len(tokenizer.encode(completion, add_special_tokens=False)) == 512:
            clipped.append(row)

    no_answer_ids = {id(row) for row in no_answer}
    clipped_ids = {id(row) for row in clipped}
    print(
        json.dumps(
            {
                "files_seen": len(files),
                "files_read": len(files) - len(unreadable),
                "unreadable": unreadable,
                "rows": len(rows),
                "step_min": min(by_step, default=None),
                "step_max": max(by_step, default=None),
                "rows_per_step": dict(collections.Counter(by_step.values())),
                "reward_reason": dict(reasons),
                "answer_marker_missing": len(no_answer),
                "reencoded_at_512_tokens": len(clipped),
                "missing_and_at_512": len(no_answer_ids & clipped_ids),
                "unique_sample_ids": len({row["sample_id"] for row in rows}),
                "unique_ids_with_missing_answer": len(
                    {row["sample_id"] for row in no_answer}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
