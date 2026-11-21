"""CPU-only gates for a completed Stage2 run and fixed-300 comparison."""

import json
import tempfile
from pathlib import Path

from queue_gate import eval_state, sha256, training_state


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        run = root / "run"
        adapter = run / "checkpoints/global_step_15/actor/lora_adapter"
        adapter.mkdir(parents=True)
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"x" * 1_000_001)
        manifest_path = run / "run_manifest.json"
        manifest = {"method": "grpo", "mode": "full", "status": "completed", "process_exit_code": 0}
        manifest_path.write_text(json.dumps(manifest))
        report_path = run / "completion_report.json"
        report = {"status": "complete", "method": "grpo", "mode": "full",
                  "optimizer_steps": 15, "positive_grad_steps": 1,
                  "run_manifest_sha256": sha256(manifest_path),
                  "adapter_dir": str(adapter), "adapter_sha256": sha256(weights)}
        report_path.write_text(json.dumps(report))
        assert training_state(run, "grpo") == "complete"
        assert training_state(run, "dapo_core") == "failed"
        weights.write_bytes(b"y" * 1_000_001)
        assert training_state(run, "grpo") == "failed"
        weights.write_bytes(b"x" * 1_000_001)
        report_path.unlink()
        assert training_state(run, "grpo") == "pending"
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest))
        assert training_state(run, "grpo") == "failed"
        assert training_state(root / "missing", "grpo") == "pending"

        summary = root / "comparison_summary.json"
        assert eval_state(summary) == "pending"
        summary.write_text(json.dumps({"count": 300}))
        assert eval_state(summary) == "complete"
        summary.write_text("{")
        assert eval_state(summary) == "failed"
    print("queue gates passed")


if __name__ == "__main__":
    main()
