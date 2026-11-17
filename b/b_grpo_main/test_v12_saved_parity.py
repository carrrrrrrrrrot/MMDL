"""Differential check against all M4_final saved v1.2 responses (CPU only)."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

from v12_scorer import score_v12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options_from_prompt(prompt: str) -> list[str]:
    if "\nOptions:\n" not in prompt:
        raise ValueError("Saved MC prompt has no options section")
    section = prompt.split("\nOptions:\n", 1)[1].split("\nThink step by step", 1)[0]
    pairs = re.findall(r"(?m)^([A-Z])\. (.*)$", section)
    values = []
    for letter, value in pairs:
        if letter != chr(ord("A") + len(values)):
            raise ValueError("Saved MC options are not sequential")
        values.append(value.strip())
    if not 2 <= len(values) <= 26:
        raise ValueError(f"Expected 2-26 saved MC options; got {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo / "code" / "lmms-eval"))
    official = load_module(repo / "eval_tasks" / "mmmu_mmdl" / "mmdl_utils.py", "official_mmdl_utils")
    summarizer = load_module(repo / "scripts" / "summarize_results.py", "official_summary")
    files = sorted(glob.glob(str(repo / "results" / "runs" / "M4_final" / "seed_*" / "snapshots__*" / "*samples*jsonl")))
    if len(files) != 3:
        raise AssertionError(f"Expected 3 M4_final sample files; got {files}")
    counts = {"open": 0, "mc": 0}
    mismatches = []
    for file in files:
        seed = Path(file).parent.parent.name
        for line in Path(file).open(encoding="utf-8"):
            saved = json.loads(line)
            acc = saved["mmmu_acc"]
            kind = "mc" if acc["question_type"] == "multiple-choice" else "open"
            counts[kind] += 1
            options = options_from_prompt(saved["input"]) if kind == "mc" else []
            doc = {"id": acc["id"], "question_type": acc["question_type"],
                   "answer": acc["answer"], "options": repr(options)}
            official_acc = official.mmdl_process_results(doc, [saved["filtered_resps"]])["mmmu_acc"]
            official_correct = bool(summarizer.is_correct(official_acc))
            saved_correct = bool(summarizer.is_correct(acc))
            lite_correct, parsed, reason = score_v12(
                saved["filtered_resps"], acc["answer"], kind, options
            )
            if official_correct != saved_correct or lite_correct != official_correct:
                mismatches.append((seed, acc["id"], kind, saved_correct, official_correct,
                                   lite_correct, acc.get("final_answer"), parsed, reason))
    print(f"M4_final rows: {counts}; mismatches: {len(mismatches)}")
    for mismatch in mismatches[:20]:
        print(mismatch)
    if mismatches:
        raise AssertionError("Light scorer differs from frozen or saved v1.2 evaluation")


if __name__ == "__main__":
    main()
