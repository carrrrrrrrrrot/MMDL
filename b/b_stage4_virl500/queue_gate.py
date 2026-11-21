"""Read-only completion gates for the sequential ViRL500 training/evaluation queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_state(run_dir: Path, method: str, mode: str = "full") -> str:
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "completion_report.json"
    if not manifest_path.is_file():
        return "pending"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("method") != method or manifest.get("mode") != mode:
            return "failed"
        if manifest.get("status") == "failed" or manifest.get("process_exit_code") not in (None, 0):
            return "failed"
        if not report_path.is_file():
            return "pending"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed" or manifest.get("process_exit_code") != 0
                or report.get("status") != "complete" or report.get("method") != method
                or report.get("mode") != mode or report.get("optimizer_steps") != (1 if mode == "probe" else 15)
                or report.get("positive_grad_steps") != (1 if mode == "probe" else 15)
                or report.get("run_manifest_sha256") != sha256(manifest_path)):
            return "failed"
        adapter = Path(report["adapter_dir"])
        step = 1 if mode == "probe" else 15
        if adapter != run_dir / f"checkpoints/global_step_{step}/actor/lora_adapter":
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
        paired_path = summary_path.parent / "paired_items.csv"
        with paired_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        seed = data["per_seed"]["3407"]
        if (set(data["per_seed"]) != {"3407"} or
                "v1.2" not in data["scoring"] or
                data["baseline_run"] != "/data/chaewon/mmdl_eval_runs/rlab19qcv2_base_fixed300" or
                data["candidate_run"] != str(summary_path.parent.parent) or
                seed["items_compared"] != 300 or seed["baseline_correct"] != 185 or
                seed["input_exact_agreement"] != 300 or seed["doc_hash_agreement"] != 300 or
                str(seed["baseline_run_meta"]["exit_status"]) != "0" or
                str(seed["candidate_run_meta"]["exit_status"]) != "0" or
                len(rows) != 300 or len({row["question_id"] for row in rows}) != 300 or
                any(row["seed"] != "3407" or row["input_exact_equal"] != "True" or
                    row["doc_hash_equal"] != "True" for row in rows) or
                sum(row["baseline_correct"] == "True" for row in rows) != 185 or
                sum(row["candidate_correct"] == "True" for row in rows) != seed["candidate_correct"]):
            return "failed"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return "failed"
    return "complete"


def salvage_probe_state(run_dir: Path) -> str:
    """Accept only audited pipeline evidence; never turn a failed probe into a clean run."""
    evidence_path = run_dir / "probe_salvage_audit.json"
    if not evidence_path.is_file():
        return "failed"
    try:
        from audit_virl500_visits import audit
        from finalize_virl500 import check_metrics, runner_lines

        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        independent = json.loads((run_dir / "independent_salvage_evidence.json").read_text(encoding="utf-8"))
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        launch = run_dir.parent / f"{run_dir.name}.launch.log"
        visits = run_dir / "visits.jsonl"
        reload_path = run_dir / "salvage_adapter_reload.json"
        adapter = run_dir / "checkpoints/global_step_1/actor/lora_adapter/adapter_model.safetensors"
        if (evidence.get("status") != "salvaged_pipeline_only" or
                evidence.get("process_status") != "failed" or
                evidence.get("process_exit_code") != 127 or
                evidence.get("run_dir") != str(run_dir) or
                evidence.get("run_manifest_sha256") != sha256(manifest_path) or
                evidence.get("launch_log_sha256") != sha256(launch) or
                evidence.get("visits_sha256") != sha256(visits) or
                evidence.get("adapter_reload_report_sha256") != sha256(reload_path) or
                evidence.get("adapter_sha256") != sha256(adapter)):
            return "failed"
        if (independent.get("status") != "launcher_failed_training_step_completed" or
                independent.get("launcher_status") != "failed" or
                independent.get("process_exit_code") != 127 or
                independent.get("completed_optimizer_steps") != 1 or
                independent.get("grad_norm") != evidence.get("actor_grad_norm") or
                independent.get("unique_used_sample_ids") != 100 or
                independent.get("cpu_peft_reload_status") != "passed_cpu_peft_reload" or
                independent.get("cpu_exact_tensor_matches") != 288 or
                independent.get("adapter_weights_sha256") != evidence.get("adapter_sha256") or
                independent.get("data_manifest_sha256") != evidence.get("data_manifest_sha256")):
            return "failed"
        if (manifest.get("status") != "failed" or manifest.get("process_exit_code") != 127 or
                manifest.get("method") != "grpo" or manifest.get("mode") != "probe" or
                manifest.get("data_manifest_sha256") != evidence.get("data_manifest_sha256") or
                (run_dir / "completion_report.json").exists()):
            return "failed"
        reload = json.loads(reload_path.read_text(encoding="utf-8"))
        if (reload.get("peft_reload") != "exact" or
                reload.get("adapter_sha256") != evidence["adapter_sha256"] or
                reload.get("tensor_count") != 288 or
                reload.get("parameter_count") != 11_796_480 or
                reload.get("adapter_path") != str(adapter)):
            return "failed"
        metric = check_metrics(launch, 1)
        lines = runner_lines(launch)
        gradients = [float(value.split(":", 1)[1])
                     for i, line in enumerate(lines[:-2])
                     if line == "Step 1" and lines[i + 1] == "actor:"
                     for value in lines[i + 2:i + 15]
                     if value.startswith("  grad_norm:")]
        if (metric["optimizer_steps_observed"] != 1 or metric["positive_grad_steps"] != 1 or
                metric["final_validation_accuracy"] != evidence.get("validation_accuracy") or
                gradients != [evidence.get("actor_grad_norm")] or gradients[0] <= 0):
            return "failed"
        log_text = launch.read_text(encoding="utf-8", errors="replace")
        if ("run_virl500.sh: line 82:" not in log_text or
                "config=" not in log_text or "No such file or directory" not in log_text):
            return "failed"
        visit_report = audit(run_dir)
        if (visit_report["attempted_question_visits"] != 100 or
                visit_report["used_question_visits"] != 100 or
                visit_report["attempted_unique_ids"] != 100 or
                visit_report["used_unique_ids"] != 100 or
                visit_report["generated_rollouts"] != 500 or
                visit_report["used_rollouts"] != 500):
            return "failed"
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        return "failed"
    return "complete"


def running_training_state(run_dir: Path, method: str, mode: str) -> str:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "failed"
    if (data.get("method") == method and data.get("mode") == mode and
            data.get("status") == "running" and data.get("process_exit_code") is None):
        return "running"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("training", "evaluation", "salvage_probe", "running_training"))
    parser.add_argument("path", type=Path)
    parser.add_argument("method", nargs="?", choices=("grpo", "dapo_core"))
    parser.add_argument("mode", nargs="?", choices=("probe", "full"), default="full")
    args = parser.parse_args()
    if args.kind == "training":
        if args.method is None:
            parser.error("training gate requires method")
        print(training_state(args.path, args.method, args.mode))
    elif args.kind == "running_training":
        if args.method is None:
            parser.error("running_training gate requires method")
        print(running_training_state(args.path, args.method, args.mode))
    elif args.kind == "salvage_probe":
        print(salvage_probe_state(args.path))
    else:
        print(eval_state(args.path))


if __name__ == "__main__":
    main()
