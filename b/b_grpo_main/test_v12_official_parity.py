"""Compare the light GRPO scorer with the frozen v1.2 evaluator on A's cases."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

from v12_scorer import score_v12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("a_shared_reward_cases.jsonl"))
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo / "code" / "lmms-eval"))
    official = load_module(repo / "eval_tasks" / "mmmu_mmdl" / "mmdl_utils.py", "official_mmdl_utils")
    summarizer = load_module(repo / "scripts" / "summarize_results.py", "official_summary")
    rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows):
        kind = "mc" if row["question_type"] == "multiple-choice" else "open"
        doc = {
            "id": f"validation_Math_{index}",
            "question_type": row["question_type"],
            "answer": row["gold"],
            "options": repr(row["options"]),
        }
        actual = official.mmdl_process_results(doc, [row["model_output"]])["mmmu_acc"]
        official_correct = bool(summarizer.is_correct(actual))
        lightweight_correct, parsed, _ = score_v12(
            row["model_output"], row["gold"], kind, row["options"]
        )
        if lightweight_correct != official_correct:
            raise AssertionError((row["case_id"], lightweight_correct, official_correct, parsed, actual))
        if lightweight_correct != bool(row["v1_2_score"]):
            raise AssertionError((row["case_id"], lightweight_correct, row["v1_2_score"]))
    print(f"PASS: {len(rows)} cases match frozen v1.2 evaluator and A's expected scores")


if __name__ == "__main__":
    main()
