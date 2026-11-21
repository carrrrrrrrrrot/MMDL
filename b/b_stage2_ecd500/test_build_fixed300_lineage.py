"""Check the Stage2-to-existing-fixed300 lineage contract without GPU work."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from build_fixed300_lineage import BASE_REVISION, EASYR1_COMMIT, build, sha256


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "b_eval_ops"))
from candidate_eval import validate_training_lineage  # noqa: E402


def put_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def make_adapter(path: Path) -> None:
    path.mkdir()
    put_json(path / "adapter_config.json", {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "exclude_modules": ".*visual.*",
    })
    header = {}
    offset = 0
    for layer in range(36):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for side in ("A", "B"):
                key = (f"base_model.model.model.language_model.layers.{layer}."
                       f"self_attn.{module}.lora_{side}.weight")
                header[key] = {"dtype": "F32", "shape": [16, 2560],
                               "data_offsets": [offset, offset + 16 * 2560 * 4]}
                offset += 16 * 2560 * 4
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with (path / "adapter_model.safetensors").open("wb") as stream:
        stream.write(len(encoded).to_bytes(8, "little"))
        stream.write(encoded)
        stream.seek(8 + len(encoded) + offset - 1)
        stream.write(b"\0")


class LineageTest(unittest.TestCase):
    def test_legacy_fixed300_accepts_verified_easyr1_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / "adapter"
            make_adapter(adapter)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            config = root / "resolved.yaml"
            source_config = root / "source.yaml"
            reward = root / "reward.py"
            for path in (train, validation, config, source_config, reward):
                path.write_text("fixture\n", encoding="utf-8")
            source_path = root / "run_manifest.json"
            visit_audit = root / "visit_audit.json"
            put_json(visit_audit, {"status": "complete"})
            put_json(source_path, {
                "status": "completed", "process_exit_code": 0,
                "mode": "full", "method": "dapo_core",
                "base_model": str(root / BASE_REVISION), "base_revision": BASE_REVISION,
                "engine_git_head": EASYR1_COMMIT,
                "stage1_manifest_sha256": "f" * 64,
                "train_file": str(train), "validation_file": str(validation),
                "data": {"train_unique_ids": 500, "validation_rows": 97,
                         "train_sha256": sha256(train), "validation_sha256": sha256(validation)},
                "resolved_config_sha256": sha256(config),
                "source_config_sha256": sha256(source_config),
                "reward_v12_sha256": sha256(reward),
                "arithmetic": {"expected_optimizer_steps_full": 15},
            })
            completion_path = root / "completion_report.json"
            put_json(completion_path, {
                "status": "complete", "method": "dapo_core", "optimizer_steps": 15,
                "mode": "full", "validation_rows": 97, "positive_grad_steps": 15,
                "adapter_dir": str(adapter),
                "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                "run_manifest_sha256": sha256(source_path),
                "resolved_config_sha256": sha256(config), "train_sha256": sha256(train),
                "visit_audit_sha256": sha256(visit_audit),
            })
            model_manifest_path = root / "model_manifest.json"
            merged = {"status": "complete", "base_revision": BASE_REVISION,
                      "adapters": [{"role": "grpo", "weights_sha256":
                                    sha256(adapter / "adapter_model.safetensors")}]}
            put_json(model_manifest_path, merged)
            args = SimpleNamespace(
                run_manifest=source_path, completion_report=completion_path,
                train_file=train, validation_file=validation, config_file=config,
                source_config_file=source_config, reward_file=reward,
                adapter_dir=adapter, model_manifest=model_manifest_path,
            )
            output = build(args)
            self.assertEqual(output["status"], "complete")
            self.assertEqual(output["adapter_tensor_count"], 288)
            self.assertEqual(output["adapter_parameter_count"], 11_796_480)
            training_manifest_path = root / "fixed300_training_manifest.json"
            put_json(training_manifest_path, output)
            self.assertIsNotNone(validate_training_lineage(merged, training_manifest_path))

            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["adapter_sha256"] = "0" * 64
            put_json(completion_path, completion)
            with self.assertRaisesRegex(ValueError, "adapter SHA256 differ"):
                build(args)


if __name__ == "__main__":
    unittest.main()
