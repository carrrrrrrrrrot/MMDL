"""CPU-only contract checks for the A/C handoff gate and idempotent resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import resume


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)

        def file(relative: str, content: str = "x") -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return path

        base = root / "base" / resume.BASE_REVISION
        base.mkdir(parents=True)
        adapter = root / "input" / "sft"
        adapter.mkdir(parents=True)
        weights = file("input/sft/adapter_model.safetensors", "fake weights")
        file("input/sft/adapter_config.json", "{}")
        approved = file("input/approved.csv", "source,id,split\nMAmmoTH-VL,1,train\n")
        file("input/converted.jsonl", '{"id":"1"}\n')
        file("input/policy.json", '{"mode":"v1_2"}')
        file("input/baseline/comparison/comparison_summary.json", "{}")
        file("input/baseline/comparison/paired_items.csv", "seed\n")
        script = file("repo/team_share/merge_lora.py", "# pinned fake merger")
        self.pinned = resume.SHARED_MERGE_SHA256
        resume.SHARED_MERGE_SHA256 = resume.sha256(script)
        self.addCleanup(lambda: setattr(resume, "SHARED_MERGE_SHA256", self.pinned))
        images = root / "input" / "images"
        images.mkdir()
        eval_repo = root / "eval_repo"
        eval_repo.mkdir()
        eval_data = root / "eval_data"
        eval_data.mkdir()
        eval_output = root / "eval_output"
        eval_output.mkdir()
        clearance = file("input/clearance.json", json.dumps({
            "status": "passed", "approved_ids_sha256": resume.sha256(approved),
            "dataset_revision": "shard_1@test", "checks": {"MMMU": "passed", "MMMU-Pro": "passed"},
        }))
        self.config = {
            "python": str(file("bin/python")),
            "base_model": str(base),
            "shared_merge_script": str(script),
            "merge_wrapper": str(file("repo/b/b_common_parent/merge_shared.py")),
            "sft_adapter": str(adapter),
            "sft_adapter_sha256": resume.sha256(weights),
            "a_converted_jsonl": str(root / "input/converted.jsonl"),
            "c_approved_ids": str(approved),
            "c_clearance_json": str(clearance),
            "image_root": str(images),
            "dataset_revision": "shard_1@test",
            "converter": str(file("repo/b/b_a_format_adapter/convert_approved.py")),
            "train_script": str(file("repo/b/b_grpo_main/train.py")),
            "reward_policy": str(root / "input/policy.json"),
            "candidate_eval": str(file("repo/b/b_eval_ops/candidate_eval.py")),
            "eval_python": str(file("bin/eval_python")),
            "eval_repo": str(eval_repo),
            "eval_data": str(eval_data),
            "eval_output_root": str(eval_output),
            "baseline_run": str(root / "input/baseline"),
            "compare_script": str(file("repo/b/b_compare/compare_runs.py")),
            "eval_run_prefix": "common_parent_test",
            "wait_for_queue_state": str(file("queue/state.json", '{"stage":"exploratory_screen_900_seed_3407"}')),
            "work_root": str(root / "output"),
            "max_steps": 128,
            "min_train_rows": 1,
        }
        file("repo/b/b_eval_ops/orchestrate_insurance.py")
        for name in ("data.py", "reward.py", "v12_scorer.py", "run_reward_cases.py", "export_rollouts.py"):
            file(f"repo/b/b_grpo_main/{name}")
        for name in ("ops_common.py", "merge_lora.py"):
            file(f"repo/b/b_eval_ops/{name}")

    def test_missing_handoff_waits(self) -> None:
        config = dict(self.config, sft_adapter=None, c_approved_ids=None)
        result = resume._status(config)
        self.assertEqual(result["status"], "waiting")
        self.assertIn("config.sft_adapter", result["missing"])

    def test_exact_hashes_and_audit_gate(self) -> None:
        self.assertEqual(resume._status(self.config)["status"], "ready")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            resume._status(dict(self.config, sft_adapter_sha256="0" * 64))
        clearance = Path(self.config["c_clearance_json"])
        document = json.loads(clearance.read_text())
        document["checks"]["MMMU-Pro"] = "pending"
        clearance.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "leakage checks"):
            resume._status(self.config)

    def test_prepare_train_and_repeat_are_idempotent(self) -> None:
        plan = resume._status(self.config)
        calls: list[list[str]] = []

        def fake_run(argv: list[str]) -> None:
            calls.append(argv)
            if "--output" in argv:
                converted = Path(plan["dataset_manifest"])
                converted.write_text("{}\n")
                converted.with_suffix(converted.suffix + ".report.json").write_text(json.dumps({
                    "source_sha256": plan["inputs"]["converted_sha256"],
                    "approved_sha256": plan["inputs"]["approved_sha256"],
                    "output_sha256": resume.sha256(converted),
                    "converted_count": 1,
                    "rendered_parity_count": 1,
                    "structural_parity_count": 1,
                }))
            if "--out" in argv and "--check-only" not in argv:
                out = Path(argv[argv.index("--out") + 1])
                out.mkdir()
                if "grpo" in argv:
                    trained = json.loads((Path(plan["train_dir"]) / "run_manifest.json").read_text())
                    adapters = [
                        {"role": "sft", "weights_sha256": self.config["sft_adapter_sha256"]},
                        {"role": "grpo", "weights_sha256": trained["adapter_sha256"]},
                    ]
                else:
                    adapters = [{"role": "sft", "weights_sha256": self.config["sft_adapter_sha256"]}]
                (out / "model_manifest.json").write_text(json.dumps({"status": "complete", "adapters": adapters}))
            if "--validate-only" not in argv and "--parent-kind" in argv:
                train = Path(plan["train_dir"])
                train.mkdir()
                adapter = train / "adapter"
                adapter.mkdir()
                weights = adapter / "adapter_model.safetensors"
                weights.write_text("fake GRPO weights")
                (train / "run_manifest.json").write_text(json.dumps({
                    "status": "complete", "parent_adapter_sha256": self.config["sft_adapter_sha256"],
                    "adapter_sha256": resume.sha256(weights), "adapter_dir": str(adapter),
                }))
            if "--model-manifest" in argv:
                mode = argv[argv.index("--mode") + 1]
                eval_dir = Path(plan[f"{mode}_dir"])
                eval_dir.mkdir()
                (eval_dir / "handoff.json").write_text(json.dumps({
                    "mode": mode, "model_path": plan["final_model"], "result_files": {"3407": {}},
                }))

        with (patch.object(resume, "_run", side_effect=fake_run),
              patch.object(resume, "gpu_gate", return_value={"status": "ready"}),
              patch.object(resume, "baseline_gate", return_value={"screen_allowed": True})):
            prepared = resume.execute(self.config, plan, train=False)
            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(len(calls), 4)
            self.assertEqual(resume.execute(self.config, plan, train=False)["status"], "prepared")
            self.assertEqual(len(calls), 4)
            completed = resume.execute(self.config, plan, train=True)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(len(calls), 10)
            self.assertEqual(resume.execute(self.config, plan, train=True)["status"], "complete")
            self.assertEqual(len(calls), 10)
        self.assertEqual(json.loads((Path(plan["work_root"]) / "state.json").read_text())["status"], "complete")

    def test_gpu_queue_blocks_training_but_allows_cpu_prepare(self) -> None:
        plan = resume._status(self.config)
        with patch.object(resume, "_run") as fake:
            # The CPU preparation path would need real outputs, so check the
            # blocker directly before any command is started.
            blocker = resume.gpu_gate(self.config)
            self.assertEqual(blocker["status"], "waiting")
            self.assertIn("earlier queue", blocker["reason"])
            fake.assert_not_called()
        Path(self.config["wait_for_queue_state"]).write_text('{"stage":"complete_exploratory_one_seed"}')
        self.config["wait_for_pilot_manifest"] = str(Path(self.temp.name) / "pilot" / "run_manifest.json")
        with patch.object(resume.subprocess, "check_output", return_value="0\n"):
            self.assertEqual(resume.gpu_gate(self.config)["status"], "waiting")
            pilot = Path(self.config["wait_for_pilot_manifest"])
            pilot.parent.mkdir(parents=True)
            pilot.write_text('{"status":"running"}')
            self.assertEqual(resume.gpu_gate(self.config)["status"], "waiting")
            pilot.write_text('{"status":"complete"}')
            self.assertEqual(resume.gpu_gate(self.config)["status"], "ready")

    def test_gpu_gate_precedes_sft_merge_and_resumes_cpu_conversion(self) -> None:
        plan = resume._status(self.config)
        converted = Path(plan["dataset_manifest"])
        parent = Path(plan["sft_parent"])
        calls: list[list[str]] = []

        def fake_run(argv: list[str]) -> None:
            calls.append(argv)
            if "--output" in argv:
                converted.write_text("{}\n", encoding="utf-8")

        with (patch.object(resume, "_run", side_effect=fake_run),
              patch.object(resume, "_verify_conversion") as verify,
              patch.object(resume, "gpu_gate", side_effect=[
                  {"status": "waiting", "reason": "GPU busy"},
                  {"status": "ready", "gpu_memory_used_mib": 0},
              ]) as gate):
            waiting = resume.execute(self.config, plan, train=False)
            self.assertEqual(waiting["status"], "waiting_gpu")
            self.assertEqual(len(calls), 1)  # CPU conversion only
            self.assertFalse(parent.exists())
            state_path = Path(plan["work_root"]) / "state.json"
            self.assertEqual(json.loads(state_path.read_text())["status"], "waiting_gpu_merge")

            prepared = resume.execute(self.config, plan, train=False)
            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(len(calls), 4)
            self.assertEqual([argv[argv.index("--stage") + 1] for argv in calls[1:3]],
                             ["sft", "sft"])
            self.assertEqual(gate.call_count, 2)
            self.assertEqual(verify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
