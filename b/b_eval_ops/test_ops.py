"""Small CPU-only contract checks for B adapter lineage and evaluator guards."""

from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from candidate_eval import check_own_gpu_baseline, mode_settings, validate_training_lineage
from merge_lora import validate_roles
from orchestrate_insurance import proposed_baseline_gate
from ops_common import BASE_REVISION, ordered_lineage, safe_name, sha256_json


class ContractTests(unittest.TestCase):
    def test_one_seed_screen_keeps_default_and_selects_seed_1(self) -> None:
        self.assertEqual(mode_settings("screen", None)["seeds"], ("3407",))
        self.assertEqual(mode_settings("screen", "1")["seeds"], ("1",))
        self.assertEqual(mode_settings("full", None)["seeds"], ("3407", "1", "2"))
        with self.assertRaises(ValueError):
            mode_settings("full", "1")

    def test_role_order(self) -> None:
        for roles in (["sft"], ["grpo"], ["sft", "grpo"], ["sft", "dpo"]):
            validate_roles(roles)
        for roles in (["grpo", "sft"], ["sft", "sft"], []):
            with self.assertRaises(ValueError):
                validate_roles(roles)

    def test_run_name_guard(self) -> None:
        self.assertEqual(safe_name("b-grpo_001"), "b-grpo_001")
        for bad in ("../outside", "a b", "", ".", ".."):
            with self.assertRaises(ValueError):
                safe_name(bad)

    def test_lineage_is_order_sensitive(self) -> None:
        a = {"role": "sft", "weights_sha256": "a" * 64}
        b = {"role": "grpo", "weights_sha256": "b" * 64}
        self.assertNotEqual(ordered_lineage([a, b])["lineage_sha256"], ordered_lineage([b, a])["lineage_sha256"])
        self.assertEqual(len(sha256_json({"x": 1})), 64)

    def test_training_lineage_requires_exact_adapter_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_manifest.json"
            train = {
                "base_revision": BASE_REVISION,
                "parent_kind": "merged_sft",
                "parent_adapter_sha256": "a" * 64,
                "adapter_sha256": "b" * 64,
                "parent_model_path": str(Path(temp) / "missing_parent"),
            }
            path.write_text(json.dumps(train), encoding="utf-8")
            model = {"adapters": [
                {"role": "sft", "weights_sha256": "a" * 64},
                {"role": "grpo", "weights_sha256": "b" * 64},
            ]}
            self.assertEqual(validate_training_lineage(model, path), train)
            model["adapters"][0]["weights_sha256"] = "c" * 64
            with self.assertRaises(ValueError):
                validate_training_lineage(model, path)

    def test_base_branch_requires_one_grpo_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_manifest.json"
            train = {
                "base_revision": BASE_REVISION,
                "parent_kind": "base",
                "parent_adapter_sha256": None,
                "adapter_sha256": "b" * 64,
            }
            path.write_text(json.dumps(train), encoding="utf-8")
            self.assertEqual(
                validate_training_lineage({"adapters": [{"role": "grpo", "weights_sha256": "b" * 64}]}, path),
                train,
            )
            with self.assertRaises(ValueError):
                validate_training_lineage({"adapters": [{"role": "sft", "weights_sha256": "b" * 64}]}, path)

    def test_baseline_must_be_complete_and_same_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            seed = Path(temp) / "seed_3407"
            samples = seed / "timestamp" / "task_samples_test.jsonl"
            samples.parent.mkdir(parents=True)
            samples.write_text("{}\n", encoding="utf-8")
            meta = seed / "run_meta.txt"
            meta.write_text("gpu: NVIDIA RTX PRO 6000, 97887 MiB\nexit_status: 0\n", encoding="utf-8")
            with patch("candidate_eval.subprocess.check_output", return_value="NVIDIA RTX PRO 6000\n"):
                check_own_gpu_baseline(Path(temp), ("3407",))
                meta.write_text("gpu: NVIDIA GeForce RTX 4090, 24564 MiB\nexit_status: 0\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_own_gpu_baseline(Path(temp), ("3407",))

    def test_proposed_gate_checks_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "m4", root / "b6000"
            for run in (old, new):
                for seed in ("3407", "1", "2"):
                    target = run / f"seed_{seed}" / "timestamp" / "task_samples_test.jsonl"
                    target.parent.mkdir(parents=True)
                    target.write_text(
                        "".join(json.dumps({
                            "mmmu_acc": {"id": f"q{i}"}, "doc_hash": f"doc{i}", "input": f"prompt{i}"
                        }) + "\n" for i in range(900)), encoding="utf-8"
                    )
            compare = new / "comparison"
            compare.mkdir()
            (compare / "comparison_summary.json").write_text(json.dumps({
                "baseline_run": str(old), "candidate_run": str(new),
                "overall": {"mean_difference_pp": 0.3},
                "per_seed": {
                    seed: {"items_compared": 900, "full_900_coverage": True,
                           "binary_correctness_agreement_pct": 90.0}
                    for seed in ("3407", "1", "2")
                },
            }), encoding="utf-8")
            with (compare / "paired_items.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "seed", "question_id", "baseline_final_answer_missing",
                    "candidate_final_answer_missing", "baseline_raw_empty", "candidate_raw_empty",
                ])
                writer.writeheader()
                for seed in ("3407", "1", "2"):
                    for i in range(900):
                        writer.writerow({
                            "seed": seed, "question_id": f"q{i}",
                            "baseline_final_answer_missing": False,
                            "candidate_final_answer_missing": False,
                            "baseline_raw_empty": False, "candidate_raw_empty": False,
                        })
            self.assertTrue(proposed_baseline_gate(new)["screen_allowed"])
            target = new / "seed_3407" / "timestamp" / "task_samples_test.jsonl"
            lines = target.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["input"] = "changed prompt"
            lines[0] = json.dumps(first)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertFalse(proposed_baseline_gate(new)["screen_allowed"])


if __name__ == "__main__":
    unittest.main()
