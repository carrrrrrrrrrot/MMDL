"""CPU-only tests for console and run-specific Ray Runner metric sources."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from finalize_ecd500 import resolve_metrics
from parse_full_metrics import auto_parse, parse


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_lines() -> list[str]:
    return [
        "Step 1",
        "actor:",
        "  grad_norm: 0.031",
        "  lr: 1e-06",
        "  pg_loss: 0.001",
        "  ppo_kl: 0.0",
        "reward:",
        "  accuracy: 0.3",
        "  answer_present: 0.9",
        "  at_cap_proxy: 0.1",
        "  overlong: 0.0",
        "response_length:",
        "  mean: 420.0",
        "perf:",
        "  time_per_step: 212.0",
        "Final validation metrics:",
        "val_reward:",
        "  accuracy_reward: 0.2",
    ]


class MetricSourceFallbackTest(unittest.TestCase):
    def test_complete_console_and_explicit_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            console = root / "console.log"
            console.write_text(
                "\n".join(f"(Runner pid=123) {line}" for line in metrics_lines()) + "\n",
                encoding="utf-8",
            )
            raw = root / "runner_stdout.log"
            raw.write_text("\n".join(metrics_lines()) + "\n", encoding="utf-8")
            for parsed in (auto_parse(console, 1), parse(console, 1, "console"),
                           auto_parse(raw, 1), parse(raw, 1, "ray_runner_stdout")):
                self.assertEqual([step["step"] for step in parsed["steps"]], [1])
                self.assertEqual(parsed["steps"][0]["reward.accuracy"], 0.3)

    def test_grpo_partial_console_uses_verified_raw_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            console = run / "console.log"
            console.write_text("Ray forwarding omitted Step 1\n", encoding="utf-8")
            logs = root / "ray_tmp" / "ray" / "session_latest" / "logs"
            logs.mkdir(parents=True)
            worker = logs / "worker-test.out"
            worker.write_text(
                ":job_id:01000000\n:actor_name:Runner\n"
                + json.dumps({"experiment_name": "ecd500_grpo_probe_cap1024",
                              "save_checkpoint_path": str(run / "checkpoints")})
                + "\n" + "\n".join(metrics_lines()) + "\n",
                encoding="utf-8",
            )
            measured, source = resolve_metrics(
                run, {"method": "grpo", "mode": "probe"}, 1, root / "ray_tmp"
            )
            self.assertEqual(measured["positive_grad_steps"], 1)
            self.assertEqual(source["kind"], "ray_runner_stdout")
            self.assertEqual(source["sha256"], sha256(run / "runner_stdout.log"))
            (run / "completion_report.json").write_text(
                json.dumps({"status": "complete", "metric_source": source}), encoding="utf-8"
            )
            parsed = auto_parse(console, 1)
            self.assertEqual(parsed["steps"][0]["actor.grad_norm"], 0.031)
            self.assertEqual(parsed["steps"][0]["reward.answer_present"], 0.9)

    def test_no_fallback_for_invalid_gradient_or_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            raw = run / "runner_stdout.log"
            raw.write_text("\n".join(metrics_lines()) + "\n", encoding="utf-8")
            source = {"kind": "ray_runner_stdout", "path": str(raw), "sha256": sha256(raw)}
            (run / "completion_report.json").write_text(
                json.dumps({"status": "complete", "metric_source": source}), encoding="utf-8"
            )
            console = run / "console.log"
            invalid = [line.replace("grad_norm: 0.031", "grad_norm: nan")
                       for line in metrics_lines()]
            console.write_text(
                "\n".join(f"(Runner pid=123) {line}" for line in invalid) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid numeric metric"):
                auto_parse(console, 1)
            with self.assertRaisesRegex(ValueError, "Invalid actor grad_norm"):
                resolve_metrics(run, {"method": "grpo", "mode": "probe"}, 1,
                                run / "ray_tmp_does_not_exist")
            console.write_text("Partial forwarding\n", encoding="utf-8")
            source["sha256"] = "0" * 64
            (run / "completion_report.json").write_text(
                json.dumps({"status": "complete", "metric_source": source}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                auto_parse(console, 1)


if __name__ == "__main__":
    unittest.main()
