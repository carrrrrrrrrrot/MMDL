"""Read-only completion gates for the sequential ECD500 training/evaluation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_state(run_dir: Path, method: str) -> str:
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "completion_report.json"
    if not manifest_path.is_file():
        return "pending"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("method") != method or manifest.get("mode") != "full":
            return "failed"
        if manifest.get("status") == "failed" or manifest.get("process_exit_code") not in (None, 0):
            return "failed"
        if not report_path.is_file():
            return "pending"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed" or manifest.get("process_exit_code") != 0
                or report.get("status") != "complete" or report.get("method") != method
                or report.get("mode") != "full" or report.get("optimizer_steps") != 15
                or report.get("positive_grad_steps", 0) < 1
                or report.get("run_manifest_sha256") != sha256(manifest_path)):
            return "failed"
        adapter = Path(report["adapter_dir"])
        if adapter != run_dir / "checkpoints/global_step_15/actor/lora_adapter":
            return "failed"
        adapter_file = adapter / "adapter_model.safetensors"
        if (not adapter_file.is_file() or adapter_file.stat().st_size < 1_000_000
                or report.get("adapter_sha256") != sha256(adapter_file)):
            return "failed"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return "failed"
    return "complete"


def eval_state(summary_path: Path) -> str:
    if not summary_path.is_file():
        return "pending"
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "failed"
    return "complete" if isinstance(data, dict) and data else "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("training", "evaluation"))
    parser.add_argument("path", type=Path)
    parser.add_argument("method", nargs="?", choices=("grpo", "dapo_core"))
    args = parser.parse_args()
    if args.kind == "training":
        if args.method is None:
            parser.error("training gate requires method")
        print(training_state(args.path, args.method))
    else:
        print(eval_state(args.path))


if __name__ == "__main__":
    main()
