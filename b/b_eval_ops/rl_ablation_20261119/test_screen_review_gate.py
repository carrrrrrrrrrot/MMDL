"""Focused tests for the already-running queue's review gate."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("screen_review_gate.py")
NAMES = ["ecd_grpo", "ecd_dapo", "ecd_sapo", "mammoth_grpo", "mammoth_dapo", "mammoth_sapo"]


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / "plan"
        self.plan.mkdir()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        (self.plan / "plan.json").write_text(json.dumps({
            "training_runs": [f"/data/runs/{name}/run_manifest.json" for name in NAMES]
        }), encoding="utf-8")
        (self.plan / "progress.json").write_text(json.dumps({"stage": "base300", "status": "running"}), encoding="utf-8")

    def call(self, action, *extra, success=True):
        result = subprocess.run([
            sys.executable, str(SCRIPT), action,
            "--plan-dir", str(self.plan), "--output-root", str(self.outputs),
            "--run-prefix", "test", *extra,
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def test_arm_blocks_all_900_and_release_requires_reviewed_ranking(self):
        self.call("arm")
        targets = [self.outputs / f"test_{name}_screen900_s3407" for name in NAMES]
        self.assertTrue(all((target / "REVIEW_GATE.json").is_file() for target in targets))
        self.call("release", "--approved-ranking-sha256", "0" * 64, success=False)
        ranking = self.plan / "ranking.json"
        ranking.write_text(json.dumps({"selected_for_900": NAMES[:2]}), encoding="utf-8")
        (self.plan / "progress.json").write_text(json.dumps({
            "stage": "queue", "status": "failed",
            "error": f"FileExistsError('{targets[0]}')",
        }), encoding="utf-8")
        status = json.loads(self.call("status").stdout)
        self.assertEqual(status["status"], "awaiting_user_review")
        self.call("release", "--approved-ranking-sha256", "0" * 64, success=False)
        digest = hashlib.sha256(ranking.read_bytes()).hexdigest()
        released = json.loads(self.call("release", "--approved-ranking-sha256", digest).stdout)
        self.assertEqual(released["status"], "released_after_user_review")
        self.assertTrue(all(not target.exists() for target in targets))

    def test_arm_refuses_existing_900_result(self):
        (self.outputs / "test_ecd_grpo_screen900_s3407").mkdir()
        self.call("arm", success=False)
        self.assertFalse((self.plan / "review_gate.json").exists())


if __name__ == "__main__":
    unittest.main()
