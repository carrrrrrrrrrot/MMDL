"""CPU-only checks for the Stage2 Ray Runner stdout metric fallback."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from finalize_ecd500_candidate import check_metrics, find_runner_stdout, resolve_metrics


def main() -> None:
    runs = Path("/data/chaewon/mmdl_stage2_runs")
    grpo = runs / "ecd500_grpo_full_20261121"
    assert check_metrics(grpo / "console.log", 15) == {
        "optimizer_steps_observed": 15,
        "positive_grad_steps": 15,
        "final_validation_accuracy": 0.34,
    }
    probe_logs = list(Path("/data/chaewon/ray_s1/ray").glob(
        "session_*/logs/worker-*1642974.out"))
    assert len(probe_logs) == 1, probe_logs
    assert check_metrics(probe_logs[0], 1, source_kind="ray_runner_stdout")[
        "final_validation_accuracy"] == 0.2

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        (run / "console.log").write_text("Ray progress without Runner metrics\n", encoding="utf-8")
        logs = root / "ray_tmp" / "ray" / "session_latest" / "logs"
        logs.mkdir(parents=True)
        worker = logs / "worker-synthetic.out"
        worker.write_text(
            ":job_id:01000000\n:actor_name:Runner\n"
            + json.dumps({"experiment_name": "ecd500_dapo_core_probe_cap1024",
                          "save_checkpoint_path": str(run / "checkpoints")})
            + "\nStep 1\nactor:\n  grad_norm: 0.031\n"
            + "Final validation metrics:\nval_reward:\n  accuracy_reward: 0.2\n",
            encoding="utf-8",
        )
        manifest = {"method": "dapo_core", "mode": "probe"}
        chosen, measured = find_runner_stdout(run, manifest, 1, root / "ray_tmp")
        assert chosen == worker and measured["positive_grad_steps"] == 1
        measured, source = resolve_metrics(run, manifest, 1, root / "ray_tmp")
        assert measured["final_validation_accuracy"] == 0.2
        assert source["kind"] == "ray_runner_stdout"
        assert (run / "runner_stdout.log").read_bytes() == worker.read_bytes()
        measured_again, source_again = resolve_metrics(run, manifest, 1, root / "ray_tmp")
        assert measured_again == measured and source_again["sha256"] == source["sha256"]
        (run / "runner_stdout.log").write_text("corrupted", encoding="utf-8")
        try:
            resolve_metrics(run, manifest, 1, root / "ray_tmp")
        except ValueError as exc:
            assert "SHA differs" in str(exc), str(exc)
        else:
            raise AssertionError("Different existing Runner stdout was accepted")

    current = runs / "ecd500_dapo_core_full_20261121"
    manifest = json.loads((current / "run_manifest.json").read_text(encoding="utf-8"))
    try:
        find_runner_stdout(current, manifest, 15)
    except ValueError as exc:
        assert "Incomplete matches" in str(exc), str(exc)
        assert "Expected actor metric steps 1..15" in str(exc), str(exc)
    else:
        raise AssertionError("Incomplete live DAPO run was accepted as complete")
    print("PASS: GRPO console, DAPO probe Runner stdout, idempotent strict "
          "fallback, SHA mismatch rejection, and incomplete live-run rejection")


if __name__ == "__main__":
    main()
