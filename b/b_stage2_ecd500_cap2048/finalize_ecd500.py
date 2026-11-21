"""Emit `status=complete` only after Stage2 visits, metrics, and adapter reload pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from audit_ecd500_visits import audit


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNNER = re.compile(r"\(Runner pid=\d+\)\s?")
STEP = re.compile(r"Step (\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_lines(path: Path) -> list[str]:
    lines = []
    for raw in ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n").splitlines():
        match = RUNNER.search(raw)
        if match:
            lines.append(raw[match.end():].rstrip())
    return lines


def check_metrics(path: Path, expected_steps: int, *, source_kind: str = "console") -> dict:
    if source_kind == "console":
        lines = runner_lines(path)
    elif source_kind == "ray_runner_stdout":
        lines = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n").splitlines()
    else:
        raise ValueError(f"Unknown metric source kind: {source_kind}")
    steps = {}
    for index, line in enumerate(lines):
        match = STEP.fullmatch(line)
        if not match or index + 1 >= len(lines) or lines[index + 1] != "actor:":
            continue
        number = int(match.group(1))
        block = lines[index + 2: index + 15]
        grad = next((float(line.split(":", 1)[1]) for line in block if line.startswith("  grad_norm:")), None)
        if grad is None or not math.isfinite(grad) or grad < 0:
            raise ValueError(f"Invalid actor grad_norm at step {number}: {grad}")
        steps[number] = grad
    if set(steps) != set(range(1, expected_steps + 1)):
        raise ValueError(f"Expected actor metric steps 1..{expected_steps}, got {sorted(steps)}")
    if not any(value > 0 for value in steps.values()):
        raise ValueError("No verified nonzero actor gradient")
    if "Final validation metrics:" not in lines:
        raise ValueError("Final heldout validation missing")
    validation_index = len(lines) - 1 - lines[::-1].index("Final validation metrics:")
    validation = lines[validation_index + 1:]
    score = next((float(line.split(":", 1)[1]) for line in validation
                  if line.startswith("  accuracy_reward:")), None)
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid final heldout v1.2 accuracy: {score}")
    return {"optimizer_steps_observed": len(steps),
            "positive_grad_steps": sum(value > 0 for value in steps.values()),
            "final_validation_accuracy": score}


def find_runner_stdout(run_dir: Path, manifest: dict, expected_steps: int,
                       ray_tmpdir: Path | None = None) -> tuple[Path, dict]:
    """Find one complete, run-specific Runner stdout in the current Ray session.

    Ray's log monitor can omit worker stdout from the console stream when the
    vLLM progress bars are noisy. The worker's own .out file is the primary
    source and contains the unmodified metric and final-validation prints.
    """
    root = ray_tmpdir or Path(os.environ.get("RAY_TMPDIR", "/data/chaewon/ray_s1"))
    logs = (root / "ray" / "session_latest" / "logs").resolve(strict=True)
    marker = f'"save_checkpoint_path": "{run_dir / "checkpoints"}"'
    experiment = f'"experiment_name": "ecd500_{manifest["method"]}_{manifest["mode"]}_cap2048"'
    matching = []
    incomplete = []
    for path in logs.glob("worker-*.out"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not raw.startswith(":job_id:") or ":actor_name:Runner" not in raw[:256]:
            continue
        if marker not in raw or experiment not in raw:
            continue
        try:
            metrics = check_metrics(path, expected_steps, source_kind="ray_runner_stdout")
        except ValueError as exc:
            incomplete.append(f"{path.name}: {exc}")
        else:
            matching.append((path, metrics))
    if len(matching) != 1:
        raise ValueError(f"Expected one complete Runner stdout for {run_dir}; "
                         f"found {len(matching)}. Incomplete matches: {incomplete}")
    return matching[0]


def resolve_metrics(run_dir: Path, manifest: dict, expected_steps: int,
                    ray_tmpdir: Path | None = None) -> tuple[dict, dict]:
    console_path = run_dir / "console.log"
    try:
        metrics = check_metrics(console_path, expected_steps)
    except ValueError as exc:
        # An invalid numeric gradient is a real validation failure. Only a
        # missing/partial forwarded stdout stream can use the raw Ray log.
        if manifest["method"] != "dapo_core" or not str(exc).startswith((
            "Expected actor metric steps", "Final heldout validation missing")):
            raise
        raw_path, metrics = find_runner_stdout(run_dir, manifest, expected_steps, ray_tmpdir)
        copy = run_dir / "runner_stdout.log"
        if not copy.exists():
            shutil.copyfile(raw_path, copy)
        if sha256(copy) != sha256(raw_path):
            raise ValueError("Copied Runner stdout SHA differs from source")
        source = {"kind": "ray_runner_stdout", "path": str(copy), "sha256": sha256(copy),
                  "ray_source_path": str(raw_path), "console_error": str(exc)}
    else:
        source = {"kind": "console", "path": str(console_path), "sha256": sha256(console_path)}
    return metrics, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("process_exit_code") != 0:
        raise ValueError("Training process has not completed successfully")
    if not manifest.get("gpu_samples") or manifest.get("gpu_peak_used_mib") is None:
        raise ValueError("GPU usage telemetry missing")
    expected_steps = 1 if manifest["mode"] == "probe" else 15
    visits = audit(args.run_dir)
    visit_path = args.run_dir / "visit_audit.json"
    visit_path.write_text(json.dumps(visits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics, metric_source = resolve_metrics(args.run_dir, manifest, expected_steps)
    adapter = args.run_dir / f"checkpoints/global_step_{expected_steps}/actor/lora_adapter"
    adapter_file = adapter / "adapter_model.safetensors"
    if not adapter_file.is_file() or adapter_file.stat().st_size < 1_000_000:
        raise ValueError("Final PEFT adapter missing or empty")
    reload_path = args.run_dir / "adapter_reload.json"
    subprocess.run([
        sys.executable, str(args.stage1 / "scripts/validate_adapter_reload.py"),
        "--base", manifest["base_model"], "--adapter", str(adapter), "--output", str(reload_path),
    ], check=True)
    reload_result = json.loads(reload_path.read_text(encoding="utf-8"))
    if reload_result.get("peft_reload") != "exact" or reload_result.get("adapter_sha256") != sha256(adapter_file):
        raise ValueError("PEFT reload result differs from final adapter")
    report = {
        "status": "complete", "method": manifest["method"], "mode": manifest["mode"],
        "run_manifest_sha256": sha256(manifest_path),
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "train_sha256": manifest["data"]["train_sha256"],
        "adapter_dir": str(adapter), "adapter_sha256": sha256(adapter_file),
        "optimizer_steps": metrics["optimizer_steps_observed"],
        "positive_grad_steps": metrics["positive_grad_steps"],
        "final_validation_accuracy": metrics["final_validation_accuracy"],
        "validation_rows": manifest["data"]["validation_rows"],
        "visit_audit_sha256": sha256(visit_path),
        "generated_rollouts": visits["generated_rollouts"],
        "used_rollouts": visits["used_rollouts"],
        "wall_seconds": manifest["wall_seconds"],
        "gpu_peak_used_mib": manifest["gpu_peak_used_mib"],
        "metric_source": metric_source,
    }
    path = args.run_dir / "completion_report.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completion_report": str(path), "status": "complete",
                      "adapter_sha256": report["adapter_sha256"],
                      "optimizer_steps": report["optimizer_steps"]}, sort_keys=True))


if __name__ == "__main__":
    main()
