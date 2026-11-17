"""Export TRL completion Parquet files as C-facing per-candidate rollout records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from reward import RewardPolicy, score_response


def _provenance(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                result[item["sample_id"]] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--max-completion-length", type=int, default=256)
    args = parser.parse_args()
    policy = RewardPolicy.from_file(args.policy)
    tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "processor")
    provenance = _provenance(args.dataset_manifest)
    parquet_files = sorted((args.run_dir / "trainer" / "completions").glob("completions_*.parquet"))
    if not parquet_files:
        raise ValueError(f"No TRL completion Parquet files in {args.run_dir}")

    exported = []
    inconsistent_rewards = 0
    short_groups = []
    for file in parquet_files:
        frame = pd.read_parquet(file)
        required = {"step", "completion", "reward_func", "gold_answer", "sample_id"}
        if not required <= set(frame.columns):
            raise ValueError(f"Completion file missing columns {required - set(frame.columns)}: {file}")
        for (step, sample_id), group in frame.groupby(["step", "sample_id"], sort=False, dropna=False):
            if len(group) != 4:
                short_groups.append(f"{file.name}:{step}:{sample_id}:{len(group)}")
            source = provenance.get(str(sample_id), {})
            kind = source.get("answer_type", "mc")
            options = source.get("options")
            if options is None and isinstance(source.get("choices"), dict):
                options = list(source["choices"].values())
            for candidate_index, (_, item) in enumerate(group.iterrows()):
                raw = str(item["completion"])
                gold = str(item["gold_answer"])
                outcome = score_response(raw, gold, kind, policy, options=options)
                logged_reward = float(item["reward_func"])
                consistent = outcome.reward == logged_reward
                inconsistent_rewards += int(not consistent)
                token_proxy = len(tokenizer.encode(raw, add_special_tokens=False))
                exported.append({
                    "step": int(step),
                    "group_id": f"{step}:{sample_id}",
                    "sample_id": str(sample_id),
                    "candidate_index": candidate_index,
                    "source": source.get("source"),
                    "dataset_revision": source.get("dataset_revision"),
                    "split": source.get("split"),
                    "original_id": source.get("original_id", str(sample_id)),
                    "question": source.get("question"),
                    "choices": source.get("choices"),
                    "image_path": source.get("image_path"),
                    "image_paths": source.get("image_paths"),
                    "image_sha256": source.get("image_sha256"),
                    "raw_output": raw,
                    "parsed_answer": outcome.parsed_answer,
                    "gold_answer": gold,
                    "reward": logged_reward,
                    "reason": outcome.reason,
                    "reward_consistent_with_policy": consistent,
                    "retokenized_completion_tokens_proxy": token_proxy,
                    "max_completion_length": args.max_completion_length,
                    "clipped_proxy": token_proxy >= args.max_completion_length,
                    "finish_reason": "unavailable",
                    "exact_generated_token_count": "unavailable",
                    "reward_version": policy.version,
                })
    if short_groups or inconsistent_rewards:
        raise ValueError(
            f"Rollout export validation failed: groups_not_four={short_groups[:8]}, "
            f"reward_mismatches={inconsistent_rewards}"
        )

    jsonl = args.run_dir / "rollouts.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for row in exported:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = args.run_dir / "rollouts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    summary = {
        "rollouts": len(exported),
        "groups": len(exported) // 4,
        "groups_with_exactly_four": True,
        "reward_mismatches": inconsistent_rewards,
        "finish_reason_available": False,
        "exact_generated_token_count_available": False,
        "retokenized_tokens_are_proxy": True,
        "clipped_proxy_count": sum(row["clipped_proxy"] for row in exported),
        "reward_version": policy.version,
        "jsonl": str(jsonl),
        "csv": str(csv_path),
    }
    (args.run_dir / "rollout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
