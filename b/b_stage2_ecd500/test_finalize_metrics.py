"""Smoke-test Stage2 metric extraction on a known completed Stage1 signal run."""

from pathlib import Path

from finalize_ecd500 import check_metrics


if __name__ == "__main__":
    path = Path("/data/chaewon/mmdl_stage1_runs/ecd20_signal_grpo_cap1024_20261120/console.log")
    report = check_metrics(path, expected_steps=1)
    assert report["optimizer_steps_observed"] == 1
    assert report["positive_grad_steps"] == 1
    assert report["final_validation_accuracy"] == 0.6
    print(report)
