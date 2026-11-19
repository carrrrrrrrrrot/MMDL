#!/usr/bin/env python3
"""Wait for three mixed-data RL models, then screen each on fixed300 only.

The proven training orchestrator runs separately. This process never starts a
training job or a 900-item evaluation. ``--check`` is read-only. ``--execute``
creates a new queue and runs one GPU screen at a time after all three adapters
are complete and reloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path("/data/chaewon")
TRAIN_PY = ROOT / "mmdl_envs/grpo_pilot/bin/python3.10"
EVAL_PY = ROOT / "mmdl_envs/vllm011_repro/bin/python"
TRAIN = ROOT / "mmdl_b_main_code/train.py"
REWARD = ROOT / "mmdl_b_main_code/reward_policy.v12.json"
SCREEN = ROOT / "mmdl_b_eval_ops/rl_ablation_20261119/run_mixed_loss_fixed300.py"
BASE = Path("/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17")
EVAL_REPO = Path("/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2")
MMMU = Path("/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
COMPARE = ROOT / "mmdl_b_compare/compare_runs.py"
EVAL_ROOT = ROOT / "mmdl_eval_runs"
MERGED_ROOT = ROOT / "mmdl_merged_models"
BASE300 = EVAL_ROOT / "rlab19qcv2_base_fixed300"
IDS = EVAL_ROOT / "rl_ablation_20261119_qcv2_plan/fixed300_ids_seed3407.txt"
GATE = EVAL_ROOT / "rl_ablation_20261119_qcv2_plan/review_gate.json"

LOSSES = ("grpo", "dapo", "sapo")
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EVAL_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
TRAIN_SHA = "1ba870be51e1bf1cf5bc6bb47dc092fc76a3b52214e0b028bb351c8edce1595f"
REWARD_SHA = "dbaacb4a111d201fc30e3969434f93961c0a0573deb141e634785e8732b74f25"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def emit(queue: Path, stage: str, state: str, **fields: object) -> None:
    record = {"at_utc": now(), "stage": stage, "state": state, **fields}
    with (queue / "events.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False) + "\n")
        target.flush()
    write_json(queue / "status.json", record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read-only static validation")
    mode.add_argument("--execute", action="store_true", help="Wait, then screen GRPO/DAPO/SAPO")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--run-prefix", required=True,
                        help="New rlab19mix* prefix, loss suffix is added")
    parser.add_argument("--queue-root", type=Path, required=True)
    return parser.parse_args()


def check_static(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", args.dataset_sha256):
        raise ValueError("Dataset SHA256 must be 64 lowercase hex characters")
    if not re.fullmatch(r"rlab19mix[a-z0-9_]+", args.run_prefix) or "900" in args.run_prefix:
        raise ValueError("Use a new rlab19mix* prefix without 900")
    if args.train_root == args.queue_root or args.queue_root.exists():
        raise FileExistsError("New queue root must be distinct and absent")
    if list(EVAL_ROOT.glob(args.run_prefix + "*")) or list(MERGED_ROOT.glob(args.run_prefix + "*")):
        raise FileExistsError("Screen prefix already has evaluation or model outputs")
    if sha256(TRAIN) != TRAIN_SHA or sha256(REWARD) != REWARD_SHA:
        raise RuntimeError("Training code or v1.2 reward SHA changed")
    screen_sha = sha256(SCREEN)
    if read_json(GATE).get("status") != "awaiting_user_review":
        raise RuntimeError("Existing 900-item review gate changed")
    for path in (TRAIN_PY, EVAL_PY, BASE / "config.json", EVAL_REPO,
                 MMMU, COMPARE, IDS, BASE300):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "-C", str(EVAL_REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(EVAL_REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout
    if head != EVAL_COMMIT or dirty.strip():
        raise RuntimeError("Frozen evaluation worktree changed")
    return {"schema": "b_mixed_three_loss_screen_v1", "created_utc": now(),
            "training_root": str(args.train_root), "dataset_sha256": args.dataset_sha256,
            "screen_script_sha256": screen_sha, "train_script_sha256": TRAIN_SHA,
            "reward_policy_sha256": REWARD_SHA, "base_revision": BASE_REVISION,
            "evaluation_commit": EVAL_COMMIT, "run_prefix": args.run_prefix,
            "losses": list(LOSSES), "expected_examples": 1000,
            "expected_steps": 2000, "baseline_correct": 185,
            "fixed300_ids": str(IDS), "original_900_gate": str(GATE),
            "allows_900": False}


def check_unchanged(args: argparse.Namespace, plan: dict) -> None:
    if sha256(SCREEN) != plan["screen_script_sha256"]:
        raise RuntimeError("Fixed300 screen code changed since queue start")
    if sha256(TRAIN) != TRAIN_SHA or sha256(REWARD) != REWARD_SHA:
        raise RuntimeError("Training code or reward policy changed")
    if read_json(GATE).get("status") != "awaiting_user_review":
        raise RuntimeError("Existing 900 review gate changed")


def check_training(args: argparse.Namespace) -> None:
    plan = read_json(args.train_root / "plan.json")
    data = plan.get("datasets", {}).get("mixed", {})
    required = {
        "parent_kind": "base", "base_revision": BASE_REVISION,
        "train_script_sha256": TRAIN_SHA, "reward_policy_sha256": REWARD_SHA,
        "run_order": list("mixed_" + loss for loss in LOSSES),
        "seed": 3407, "max_steps": 2000, "num_iterations": 2,
        "learning_rate": 1e-5, "max_completion_length": 512,
        "num_generations": 4, "save_steps": 250,
    }
    for key, value in required.items():
        if plan.get(key) != value:
            raise RuntimeError(f"Training plan differs at {key}: {plan.get(key)}")
    if data.get("sha256") != args.dataset_sha256 or data.get("split_counts") != {"train": 1000, "heldout": 100}:
        raise RuntimeError("Training plan dataset hash or count differs")
    if sha256(Path(data["manifest"])) != args.dataset_sha256:
        raise RuntimeError("Dataset changed after training plan was created")
    summary = read_json(args.train_root / "summary.json")
    if summary.get("status") != "complete" or summary.get("blocked_datasets"):
        raise RuntimeError(f"Three-loss training did not complete: {summary.get('status')}")
    for loss in LOSSES:
        run = read_json(args.train_root / "runs" / f"mixed_{loss}" / "run_manifest.json")
        expected = {"status": "complete", "loss_type": loss, "parent_kind": "base",
                    "base_revision": BASE_REVISION, "reward_policy_sha256": REWARD_SHA,
                    "max_steps": 2000, "global_step": 2000, "num_iterations": 2,
                    "num_generations": 4, "max_completion_length": 512,
                    "learning_rate": 1e-5, "seed": 3407, "reload_success": True}
        for key, value in expected.items():
            if run.get(key) != value:
                raise RuntimeError(f"mixed_{loss} manifest differs at {key}")
        if run.get("dataset", {}).get("manifest_sha256") != args.dataset_sha256:
            raise RuntimeError(f"mixed_{loss} dataset differs")


def wait_training(queue: Path, args: argparse.Namespace, plan: dict) -> None:
    deadline = time.monotonic() + 14 * 3600
    while time.monotonic() < deadline:
        check_unchanged(args, plan)
        summary = args.train_root / "summary.json"
        if summary.is_file():
            state = read_json(summary).get("status")
            if state == "complete":
                check_training(args)
                emit(queue, "training", "complete")
                return
            if state in {"failed", "complete_with_no_signal"}:
                raise RuntimeError(f"Training queue ended {state}")
        status_path = args.train_root / "status.txt"
        status = status_path.read_text(encoding="utf-8") if status_path.is_file() else ""
        step = next((line.partition(": ")[2] for line in status.splitlines()
                     if line.startswith("step: ")), "unknown")
        current = next((line.partition(": ")[2] for line in status.splitlines()
                        if line.startswith("current_run: ")), "waiting")
        emit(queue, "training", "waiting", current_run=current, step=step)
        time.sleep(30)
    raise TimeoutError("Three-loss training did not finish within 14 hours")


def wait_gpu_idle(queue: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        pids = [line.strip() for line in result.stdout.splitlines()
                if line.strip().isdigit()]
        if not pids:
            return
        emit(queue, "gpu", "waiting", pids=pids)
        time.sleep(20)
    raise TimeoutError("6000 GPU remains occupied; no evaluation started")


def screen_command(args: argparse.Namespace, loss: str, mode: str) -> list[str]:
    return [str(TRAIN_PY), "-u", str(SCREEN),
            "--train-manifest", str(args.train_root / "runs" / f"mixed_{loss}" / "run_manifest.json"),
            "--dataset-sha256", args.dataset_sha256,
            "--expected-examples", "1000", "--expected-steps", "2000",
            "--expected-loss-type", loss,
            "--run-prefix", f"{args.run_prefix}{loss}",
            "--plan-dir", str(EVAL_ROOT / f"{args.run_prefix}{loss}_plan"),
            "--base-model", str(BASE), "--repo-root", str(EVAL_REPO),
            "--data-root", str(MMMU), "--train-python", str(TRAIN_PY),
            "--eval-python", str(EVAL_PY), "--output-root", str(EVAL_ROOT),
            "--merged-root", str(MERGED_ROOT), "--baseline-300", str(BASE300),
            "--ids-file", str(IDS), "--compare-script", str(COMPARE), f"--{mode}"]


def terminate_tree(child: subprocess.Popen) -> None:
    """Stop the wrapper and its own-session merge/evaluation descendants."""
    try:
        parent = psutil.Process(child.pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    own_group = os.getpgrp()
    groups = set()
    for process in processes:
        try:
            group = os.getpgid(process.pid)
            if group != own_group:
                groups.add(group)
        except ProcessLookupError:
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(processes, timeout=20)
    for process in alive:
        try:
            group = os.getpgid(process.pid)
            if group != own_group:
                os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_stage(queue: Path, args: argparse.Namespace, plan: dict,
              loss: str, mode: str) -> None:
    check_unchanged(args, plan)
    if mode == "execute":
        wait_gpu_idle(queue)
    stage = f"{loss}_{mode}"
    log = queue / "logs" / f"{stage}.log"
    command = screen_command(args, loss, mode)
    emit(queue, stage, "starting", command=command, log=str(log))
    start = time.monotonic()
    with log.open("xb") as target:
        child = subprocess.Popen(command, stdout=target, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while child.poll() is None:
            elapsed = round(time.monotonic() - start)
            if elapsed > (4 * 3600 if mode == "execute" else 600):
                terminate_tree(child)
                raise TimeoutError(f"{stage} exceeded timeout; child process tree terminated")
            emit(queue, stage, "running", pid=child.pid, elapsed_seconds=elapsed, log=str(log))
            time.sleep(30)
        code = child.wait()
    if code:
        raise RuntimeError(f"{stage} exited {code}; inspect {log}")
    emit(queue, stage, "complete", elapsed_seconds=round(time.monotonic() - start),
         log=str(log))


def screen_result(args: argparse.Namespace, loss: str) -> dict:
    result_path = EVAL_ROOT / f"{args.run_prefix}{loss}_plan/mixed_fixed300_result.json"
    result = read_json(result_path)
    if (result.get("status") != "complete" or result.get("base_correct") != 185
            or result.get("input_exact_agreement") != 300
            or result.get("doc_hash_agreement") != 300
            or result.get("dataset_sha256") != args.dataset_sha256):
        raise RuntimeError(f"{loss} fixed300 paired result failed checks")
    return result


def main() -> None:
    args = parse_args()
    plan = check_static(args)
    if args.check:
        print(json.dumps({"status": "check_passed", "plan": plan}, indent=2))
        return
    queue = args.queue_root
    queue.mkdir(parents=True, exist_ok=False)
    (queue / "logs").mkdir()
    write_json(queue / "plan.json", plan)
    try:
        wait_training(queue, args, plan)
        scores = {}
        for loss in LOSSES:
            run_stage(queue, args, plan, loss, "preflight")
            run_stage(queue, args, plan, loss, "execute")
            scores[loss] = screen_result(args, loss)["candidate_correct"]
        write_json(queue / "summary.json", {
            "status": "complete", "completed_utc": now(), "scores_300": scores,
            "base_correct": 185, "seed": 3407, "official_900_started": False})
        emit(queue, "supervisor", "complete", scores_300=scores)
    except Exception as error:
        write_json(queue / "summary.json", {"status": "failed", "failed_utc": now(),
                                             "error": repr(error), "official_900_started": False})
        emit(queue, "supervisor", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
