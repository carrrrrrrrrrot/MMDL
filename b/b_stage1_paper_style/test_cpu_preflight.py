"""CPU checks for MMDL EasyR1 data and frozen v1.2 reward parity."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from convert_ecd_manifest import convert_row
from reward_v12 import compute_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("MMDL_B_SOURCE", ROOT / "b_publish_clone" / "b" / "b_grpo_main"))
MANIFEST = Path(os.environ.get("MMDL_ECD_MANIFEST", ROOT / "b_publish_clone" / "b" / "b_grpo_data" / "rl_20261119" / "ecd_qc_v2" / "ecd_manifest.jsonl"))


class CpuPreflight(unittest.TestCase):
    def test_ecd_prompt_and_images(self):
        with MANIFEST.open(encoding="utf-8") as stream:
            row = json.loads(stream.readline())
        converted = convert_row(row)
        self.assertEqual(converted["system_prompt"], "You are a helpful assistant.")
        self.assertEqual(converted["prompt"].count("<image>"), len(converted["images"]))
        self.assertEqual(json.loads(converted["answer"])["sample_id"], row["sample_id"])

    def test_all_shared_reward_cases_match_v12(self):
        cases = [json.loads(line) for line in (SOURCE / "a_shared_reward_cases.jsonl").read_text(encoding="utf-8").splitlines()]
        inputs = []
        for case in cases:
            gold = {
                "answer": case["gold"],
                "answer_type": "mc" if case["question_type"] == "multiple-choice" else "open",
                "options": case.get("options", []),
                "sample_id": case["case_id"],
            }
            inputs.append({"ground_truth": json.dumps(gold), "response": case["model_output"], "response_length": 100})
        scores = compute_score(inputs, reward_mode="v12_binary", max_response_length=1024)
        self.assertEqual([score["overall"] for score in scores], [float(case["v1_2_score"]) for case in cases])
        self.assertEqual([score["accuracy"] for score in scores], [score["overall"] for score in scores])

    def test_optional_length_mode_is_explicit(self):
        gold = {"answer": "B", "answer_type": "mc", "options": ["A", "B", "C", "D"], "sample_id": "test"}
        item = {"ground_truth": json.dumps(gold), "response": "Answer: B", "response_length": 1024}
        binary = compute_score([item], reward_mode="v12_binary", max_response_length=1024)[0]
        shaped = compute_score([item], reward_mode="v12_soft_overlong", max_response_length=1024)[0]
        self.assertEqual(binary["overall"], 1.0)
        self.assertEqual(shaped["accuracy"], 1.0)
        self.assertEqual(shaped["overall"], 0.0)


if __name__ == "__main__":
    unittest.main()
