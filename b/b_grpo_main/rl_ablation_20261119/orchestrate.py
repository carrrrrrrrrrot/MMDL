#!/usr/bin/env python3
"""Sequential, resumable six-run ECD/MAmmoTH loss ablation on one GPU.

This file never runs by import. Use --preflight for read-only validation, then
--execute explicitly. It trains from the same pinned base for every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EVAL_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
LOSSES = ("grpo", "dapo", "sapo")
DEFAULT_RUN_ORDER = (
    "ecd_grpo", "mammoth_grpo", "ecd_dapo", "ecd_sapo",
    "mammoth_dapo", "mammoth_sapo",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest_counts(path: Path, expected_train: int,
                         expected_heldout: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        split = row.get("split")
        sample_id = row.get("sample_id")
        if split not in {"train", "heldout"} or not isinstance(sample_id, str):
            raise ValueError(f"{path}:{line_no}: invalid split/sample_id")
        if sample_id in ids:
            raise ValueError(f"{path}:{line_no}: duplicate sample_id {sample_id}")
        ids.add(sample_id)
        counts[split] = counts.get(split, 0) + 1
    expected = {"train": expected_train, "heldout": expected_heldout}
    if counts != expected:
        raise ValueError(f"{path}: expected exactly {expected}, got {counts}")
    return {"split_counts": counts, "sha256": sha256(path), "rows": len(ids)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--ecd-manifest", type=Path)
    parser.add_argument("--mammoth-manifest", type=Path)
    parser.add_argument("--mixed-manifest", type=Path)
    parser.add_argument("--run-order", help="Comma-separated dataset_loss runs; default six-run order")
    parser.add_argument("--pause-after", type=int, default=0,
                        help="Pause after this many newly attempted runs; resume with same plan")
    parser.add_argument("--expected-train", type=int, default=500)
    parser.add_argument("--expected-heldout", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--reward-policy", required=True, type=Path)
    parser.add_argument("--train-script", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.expected_train, args.expected_heldout,
           args.max_steps, args.save_steps) < 1:
        raise ValueError("expected-train, expected-heldout, max-steps and save-steps must be positive")
    supplied = {
        name: getattr(args, f"{name}_manifest")
        for name in ("ecd", "mammoth", "mixed")
        if getattr(args, f"{name}_manifest") is not None
    }
    if not supplied:
        raise ValueError("Supply at least one dataset manifest")
    run_order = (
        [name.strip() for name in args.run_order.split(",")] if args.run_order
        else list(DEFAULT_RUN_ORDER)
    )
    if not run_order or len(run_order) != len(set(run_order)):
        raise ValueError("run-order must be nonempty with unique runs")
    for name in run_order:
        if "_" not in name:
            raise ValueError(f"Invalid run name {name}")
        dataset, loss = name.rsplit("_", 1)
        if dataset not in supplied or loss not in LOSSES:
            raise ValueError(f"Run {name} lacks a supplied dataset or supported loss")
    paths = {
        **{f"{name}_manifest": path.resolve(strict=True) for name, path in supplied.items()},
        "reward_policy": args.reward_policy.resolve(strict=True),
        "train_script": args.train_script.resolve(strict=True),
        "python": args.python.resolve(strict=True),
        "base_model": args.base_model.resolve(strict=True),
    }
    if BASE_REVISION not in str(paths["base_model"]):
        raise ValueError("Base model is not the pinned revision")
    if not (paths["base_model"] / "config.json").is_file():
        raise ValueError("Base model lacks config.json")
    policy = json.loads(paths["reward_policy"].read_text(encoding="utf-8"))
    if policy.get("mode") != "v1_2" or policy.get("eval_repo_commit") != EVAL_COMMIT:
        raise ValueError("Reward policy is not frozen v1.2")
    counts = {
        name: read_manifest_counts(paths[f"{name}_manifest"],
                                   args.expected_train, args.expected_heldout)
        for name in supplied
    }
    return {
        "schema": "b_rl_ablation_plan_v1",
        "parent_kind": "base",
        "base_revision": BASE_REVISION,
        "base_model": str(paths["base_model"]),
        "training_python": str(paths["python"]),
        "train_script": str(paths["train_script"]),
        "train_script_sha256": sha256(paths["train_script"]),
        "reward_policy": str(paths["reward_policy"]),
        "reward_policy_sha256": sha256(paths["reward_policy"]),
        "datasets": {
            name: {"manifest": str(paths[f"{name}_manifest"]), **counts[name]}
            for name in supplied
        },
        "run_order": run_order,
        "seed": 3407,
        "max_steps": args.max_steps,
        "num_iterations": 2,
        "learning_rate": 1e-5,
        "max_completion_length": 512,
        "num_generations": 4,
        "save_steps": args.save_steps,
    }


def train_command(plan: dict[str, Any], dataset: str, loss: str, run_dir: Path,
                  *, validate: bool = False) -> list[str]:
    cmd = [
        plan["training_python"], "-u", plan["train_script"],
        "--dataset-manifest", plan["datasets"][dataset]["manifest"],
        "--train-split", "train",
        "--reward-policy", plan["reward_policy"],
        "--parent-kind", "base",
        "--parent-model", plan["base_model"],
        "--output-dir", str(run_dir),
        "--max-steps", str(plan["max_steps"]),
        "--loss-type", loss,
        "--num-iterations", str(plan["num_iterations"]),
        "--learning-rate", str(plan["learning_rate"]),
        "--max-completion-length", str(plan["max_completion_length"]),
        "--num-generations", str(plan["num_generations"]),
        "--save-steps", str(plan["save_steps"]),
        "--seed", str(plan["seed"]),
    ]
    if validate:
        cmd.append("--validate-only")
    return cmd


def preflight(plan: dict[str, Any], output_root: Path) -> None:
    for dataset in plan["datasets"]:
        run_dir = output_root / "runs" / f"{dataset}_grpo"
        result = subprocess.run(
            train_command(plan, dataset, "grpo", run_dir, validate=True),
            capture_output=True, text=True, check=True,
        )
        metadata = json.loads(result.stdout)
        found = metadata["dataset"]
        if found["manifest_sha256"] != plan["datasets"][dataset]["sha256"]:
            raise RuntimeError(f"{dataset}: validate-only read a different manifest")
        expected = plan["datasets"][dataset]["split_counts"]
        if (found["selected_examples"] != expected["train"]
                or found["total_examples"] != expected["train"] + expected["heldout"]):
            raise RuntimeError(f"{dataset}: unexpected validated split counts")
        print(json.dumps({
            "dataset": dataset, "rows": found["total_examples"],
            "train": found["selected_examples"],
            "manifest_sha256": found["manifest_sha256"],
            "answer_types": found["answer_types"],
        }, ensure_ascii=False), flush=True)


def append_event(path: Path, event: str, **fields: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": utc_now(), "event": event, **fields}, ensure_ascii=False) + "\n")
        stream.flush()


def write_status(path: Path, **fields: Any) -> None:
    lines = [f"{key}: {value}" for key, value in fields.items()]
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, path)


def log_step(path: Path, total: int) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 2_000_000))
        tail = stream.read().decode("utf-8", errors="replace")
    matches = re.findall(r"(?<!\d)(\d+)/" + str(total) + r"\b", tail)
    return max((int(value) for value in matches), default=0)


def check_inputs_unchanged(plan: dict[str, Any]) -> None:
    if sha256(Path(plan["train_script"])) != plan["train_script_sha256"]:
        raise RuntimeError("train.py changed after plan creation")
    if sha256(Path(plan["reward_policy"])) != plan["reward_policy_sha256"]:
        raise RuntimeError("reward policy changed after plan creation")
    for dataset in plan["datasets"]:
        expected = plan["datasets"][dataset]
        if sha256(Path(expected["manifest"])) != expected["sha256"]:
            raise RuntimeError(f"{dataset} manifest changed after plan creation")


def verify_complete(plan: dict[str, Any], dataset: str, loss: str,
                    run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing run_manifest.json: {run_dir}")
    run = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = plan["datasets"][dataset]
    if run.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {run_dir} ({run.get('status')})")
    required = {
        "parent_kind": "base", "base_revision": BASE_REVISION,
        "loss_type": loss, "seed": plan["seed"],
        "max_steps": plan["max_steps"],
        "num_iterations": plan["num_iterations"],
        "max_completion_length": plan["max_completion_length"],
        "num_generations": plan["num_generations"],
        "learning_rate": plan["learning_rate"],
        "reward_policy_sha256": plan["reward_policy_sha256"],
    }
    for key, value in required.items():
        if run.get(key) != value:
            raise RuntimeError(f"{run_dir}: {key} differs from plan")
    if run.get("dataset", {}).get("manifest_sha256") != expected["sha256"]:
        raise RuntimeError(f"{run_dir}: dataset SHA differs from plan")
    if run.get("code_sha256", {}).get("train.py") != plan["train_script_sha256"]:
        raise RuntimeError(f"{run_dir}: train.py SHA differs from plan")
    adapter = run_dir / "adapter" / "adapter_model.safetensors"
    if not adapter.is_file() or sha256(adapter) != run.get("adapter_sha256"):
        raise RuntimeError(f"{run_dir}: saved adapter SHA mismatch")
    if not run.get("reload_success") or run.get("global_step") != plan["max_steps"]:
        raise RuntimeError(f"{run_dir}: reload or optimizer step count failed")
    initial_path = run_dir / "initial_lora_sha256.txt"
    initial = initial_path.read_text(encoding="utf-8").strip()
    if initial != run.get("initial_lora_sha256"):
        raise RuntimeError(f"{run_dir}: initial LoRA fingerprint mismatch")
    rollout_path = run_dir / "rollouts.jsonl"
    if not rollout_path.is_file() or sha256(rollout_path) != run.get("rollouts_sha256"):
        raise RuntimeError(f"{run_dir}: rollout SHA mismatch")
    sample_ids = set()
    rollout_rows = 0
    with rollout_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_ids.add(row["sample_id"])
            rollout_rows += 1
    return {
        "dataset": dataset, "loss": loss,
        "run_dir": str(run_dir),
        "manifest_sha256": sha256(manifest_path),
        "adapter_sha256": run["adapter_sha256"],
        "initial_lora_sha256": initial,
        "global_step": run["global_step"],
        "mixed_reward_groups": run["mixed_reward_groups"],
        "reward_groups": run["reward_groups"],
        "mean_reward": run["mean_reward"],
        "lora_delta_l1": run["lora_delta_l1"],
        "unique_training_sample_ids": len(sample_ids),
        "rollout_rows": rollout_rows,
    }


def run_queue(plan: dict[str, Any], output_root: Path, pause_after: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    plan_path = output_root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing output root belongs to a different plan")
    else:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    events = output_root / "events.jsonl"
    status = output_root / "status.txt"
    summary_path = output_root / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    blocked_datasets = set(previous.get("blocked_datasets", []))
    if not blocked_datasets <= set(plan["datasets"]):
        raise RuntimeError("Previous summary has unknown blocked datasets")
    append_event(events, "queue_started", plan_sha256=sha256(plan_path))
    results: list[dict[str, Any]] = []
    initial_by_dataset: dict[str, str] = {}
    attempted_new = 0
    try:
        for index, run_name in enumerate(plan["run_order"], 1):
            dataset, loss = run_name.rsplit("_", 1)
            run_dir = output_root / "runs" / run_name
            log = output_root / "logs" / f"{run_name}.log"
            if dataset in blocked_datasets:
                append_event(events, "run_skipped_no_signal", run=run_name, dataset=dataset)
                continue
            check_inputs_unchanged(plan)
            if run_dir.exists():
                result = verify_complete(plan, dataset, loss, run_dir)
                append_event(events, "run_skipped_complete", run=run_name, **result)
            else:
                attempted_new += 1
                cmd = train_command(plan, dataset, loss, run_dir)
                append_event(events, "run_started", run=run_name, index=index,
                             total=len(plan["run_order"]), command=cmd, log=str(log))
                write_status(status, queue="running", current_run=run_name,
                             run_index=f"{index}/{len(plan['run_order'])}",
                             step=f"0/{plan['max_steps']}", log=log, events=events,
                             updated_utc=utc_now())
                if log.exists():
                    raise RuntimeError(f"Refusing to overwrite existing log {log}")
                with log.open("xb") as stream:
                    process = subprocess.Popen(cmd, stdout=stream,
                                               stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    begun = time.monotonic()
                    last_event = 0.0
                    while process.poll() is None:
                        time.sleep(15)
                        step = log_step(log, plan["max_steps"])
                        initial_file = run_dir / "initial_lora_sha256.txt"
                        expected_initial = initial_by_dataset.get(dataset)
                        if initial_file.is_file() and expected_initial:
                            observed = initial_file.read_text(encoding="utf-8").strip()
                            if observed != expected_initial:
                                os.killpg(process.pid, signal.SIGTERM)
                                process.wait(timeout=30)
                                raise RuntimeError(
                                    f"{run_name}: initial LoRA SHA differs from first {dataset} run"
                                )
                        elapsed = round(time.monotonic() - begun)
                        write_status(status, queue="running", current_run=run_name,
                                     run_index=f"{index}/{len(plan['run_order'])}",
                                     step=f"{step}/{plan['max_steps']}",
                                     elapsed_seconds=elapsed, pid=process.pid,
                                     log=log, events=events, updated_utc=utc_now())
                        if time.monotonic() - last_event >= 30:
                            append_event(events, "heartbeat", run=run_name,
                                         step=step, max_steps=plan["max_steps"],
                                         elapsed_seconds=elapsed, pid=process.pid)
                            last_event = time.monotonic()
                    returncode = process.wait()
                if returncode != 0:
                    failed_manifest = run_dir / "run_manifest.json"
                    failure = json.loads(failed_manifest.read_text(encoding="utf-8")) if failed_manifest.is_file() else {}
                    reason = str(failure.get("error", ""))
                    if loss == "grpo" and "no verified learning signal" in reason:
                        blocked_datasets.add(dataset)
                        append_event(events, "dataset_blocked_no_signal", dataset=dataset,
                                     run=run_name, exit_code=returncode, reason=reason, log=str(log))
                        summary_path.write_text(json.dumps({
                            "status": "running", "completed": results,
                            "blocked_datasets": sorted(blocked_datasets),
                        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        if pause_after and attempted_new >= pause_after:
                            break
                        continue
                    raise RuntimeError(f"{run_name} exited {returncode}; see {log}; {reason}")
                result = verify_complete(plan, dataset, loss, run_dir)
                append_event(events, "run_completed", run=run_name, **result)
            expected_initial = initial_by_dataset.setdefault(dataset, result["initial_lora_sha256"])
            if result["initial_lora_sha256"] != expected_initial:
                raise RuntimeError(f"{run_name}: initial LoRA SHA differs within dataset")
            results.append(result)
            summary_path.write_text(json.dumps({"status": "running", "completed": results,
                                               "blocked_datasets": sorted(blocked_datasets)},
                                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if pause_after and attempted_new >= pause_after:
                break
        remaining = [name for name in plan["run_order"]
                     if name.rsplit("_", 1)[0] not in blocked_datasets
                     and not (output_root / "runs" / name / "run_manifest.json").is_file()]
        final_status = "paused" if remaining else ("complete_with_no_signal" if blocked_datasets else "complete")
        summary_path.write_text(json.dumps({"status": final_status, "completed": results,
                                           "blocked_datasets": sorted(blocked_datasets),
                                           "remaining": remaining},
                                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_status(status, queue=final_status,
                     completed=f"{len(results)}/{len(plan['run_order'])}",
                     blocked_datasets=','.join(sorted(blocked_datasets)) or 'none',
                     summary=summary_path, events=events, updated_utc=utc_now())
        append_event(events, "queue_paused" if final_status == "paused" else "queue_completed",
                     runs=len(results), blocked_datasets=sorted(blocked_datasets),
                     remaining=remaining, summary=str(summary_path))
    except Exception as exc:
        summary_path.write_text(json.dumps({
            "status": "failed", "completed": results,
            "blocked_datasets": sorted(blocked_datasets),
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_status(status, queue="failed", error=f"{type(exc).__name__}: {exc}",
                     completed=f"{len(results)}/{len(plan['run_order'])}",
                     events=events, updated_utc=utc_now())
        append_event(events, "queue_failed", error=f"{type(exc).__name__}: {exc}",
                     completed=len(results))
        raise


def main() -> None:
    args = parse_args()
    plan = make_plan(args)
    preflight(plan, args.output_root)
    if args.preflight:
        print(json.dumps({"status": "preflight_passed", "plan": plan},
                         ensure_ascii=False, indent=2), flush=True)
        return
    if args.pause_after < 0:
        raise ValueError("pause-after cannot be negative")
    run_queue(plan, args.output_root, args.pause_after)


if __name__ == "__main__":
    main()
