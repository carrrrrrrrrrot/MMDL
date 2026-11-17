"""CPU-only lineage checks for the A-script compatibility wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from merge_shared import check_merge_info, check_parent, sha256_file


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class SharedMergeContract(unittest.TestCase):
    def test_merge_info_matches_requested_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            adapter.mkdir()
            info = {
                "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "base_path": str(base),
                "adapter_path": str(adapter),
                "adapter_sha256": {"adapter_model.safetensors": "a" * 64},
                "dtype": "bfloat16",
            }
            check_merge_info(info, base.resolve(), adapter.resolve(), "a" * 64)
            info["adapter_sha256"]["adapter_model.safetensors"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "MERGE_INFO"):
                check_merge_info(info, base.resolve(), adapter.resolve(), "a" * 64)

    def test_grpo_parent_requires_exact_sft_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "sft_parent"
            base.mkdir()
            sft_sha, grpo_sha = "a" * 64, "b" * 64
            parent = {
                "status": "complete", "parent_kind": "merged_sft",
                "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "merged_model_path": str(base.resolve()),
                "adapter_sha256": sft_sha,
                "adapters": [{"role": "sft", "weights_sha256": sft_sha}],
            }
            parent_path = base / "model_manifest.json"
            write(parent_path, parent)
            write(base / "MERGE_INFO.json", {
                "adapter_sha256": {"adapter_model.safetensors": sft_sha}
            })
            train_path = Path(temp) / "train.json"
            train = {
                "status": "complete", "parent_kind": "merged_sft",
                "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "parent_adapter_sha256": sft_sha,
                "adapter_sha256": grpo_sha,
                "parent_model_path": str(base.resolve()),
                "parent_model_manifest_sha256": sha256_file(parent_path),
            }
            write(train_path, train)
            got_parent, got_train = check_parent(base.resolve(), parent_path, train_path, grpo_sha)
            self.assertEqual(got_parent["adapter_sha256"], sft_sha)
            self.assertEqual(got_train["adapter_sha256"], grpo_sha)
            with self.assertRaisesRegex(ValueError, "GRPO training manifest"):
                check_parent(base.resolve(), parent_path, train_path, "c" * 64)


if __name__ == "__main__":
    unittest.main()
