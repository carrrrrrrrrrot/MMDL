#!/usr/bin/env python3
"""Continue B's QC-v2 RL runs and 6000-only screening without unattended overwrites.

The first ECD GRPO run is launched separately with ``--pause-after 1``. This
supervisor waits for its verified pause, then runs one GPU stage at a time.
It never starts the 4090 official evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ECD_SHA = "c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed"
CLEVR_SHA = "b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d"
REWARD_SHA = "dbaacb4a111d201fc30e3969434f93961c0a0573deb141e634785e8732b74f25"
EVAL_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"

ROOT = Path("/data/chaewon")
MAIN = ROOT / "mmdl_b_main_code"
OPS = ROOT / "mmdl_b_eval_ops"
TRAIN_PY = ROOT / "mmdl_envs/grpo_pilot/bin/python"
EVAL_PY = ROOT / "mmdl_envs/vllm011_repro/bin/python"
TRAIN = MAIN / "train.py"
ORCHESTRATOR = MAIN / "rl_ablation_20261119/orchestrate.py"
REWARD = MAIN / "reward_policy.v12.json"
EVAL_QUEUE = OPS / "rl_ablation_20261119/run_eval_queue.py"
EVAL_SUPPORT = {
    "candidate_eval_sha256": OPS / "candidate_eval.py",
    "merge_lora_sha256": OPS / "merge_lora.py",
    "run_fixed300_sha256": OPS / "rl_ablation_20261119/run_fixed300.py",
    "make_fixed300_ids_sha256": OPS / "rl_ablation_20261119/make_fixed300_ids.py",
    "ops_common_sha256": OPS / "ops_common.py",
}
BASE = Path("/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots") / BASE_REV
ECD_DATA = ROOT / "mmdl_grpo_data/rl_ablation_20261119_ecd_qc_v2/ecd_manifest.jsonl"
CLEVR_DATA = ROOT / "mmdl_grpo_data/rl_ablation_20261119_clevr_v3/mammoth_clevr_manifest.jsonl"
ECD_RUN = ROOT / "mmdl_grpo_runs/rl_ablation_20261119_ecd_qc_v2"
CLEVR_RUN = ROOT / "mmdl_grpo_runs/rl_ablation_20261119_clevr_v3"
EVAL_REPO = Path("/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2")
MMMU = Path("/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
EVAL_ROOT = ROOT / "mmdl_eval_runs"
MERGED_ROOT = ROOT / "mmdl_merged_models"
BASELINE = EVAL_ROOT / "b6000_baseline_v12_20261117"
COMPARE = ROOT / "mmdl_b_compare/compare_runs.py"
EVAL_PLAN = EVAL_ROOT / "rl_ablation_20261119_qcv2_plan"
EVAL_PREFIX = "rlab19qcv2"

EXPECTED_RUNS = {
    "ecd_grpo": (ECD_RUN, "grpo", ECD_SHA),
    "ecd_dapo": (ECD_RUN, "dapo", ECD_SHA),
    "ecd_sapo": (ECD_RUN, "sapo", ECD_SHA),
    "mammoth_grpo": (CLEVR_RUN, "grpo", CLEVR_SHA),
    "mammoth_dapo": (CLEVR_RUN, "dapo", CLEVR_SHA),
    "mammoth_sapo": (CLEVR_RUN, "sapo", CLEVR_SHA),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def check_eval_worktree() -> None:
    head = subprocess.run(["git", "-C", str(EVAL_REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != EVAL_COMMIT:
        raise ValueError(f"Frozen v1.2 evaluation HEAD drifted: {head}")
    dirty = subprocess.run(["git", "-C", str(EVAL_REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("Frozen v1.2 evaluation worktree has local changes")


def check_files() -> dict:
    for path in (TRAIN_PY, EVAL_PY, TRAIN, ORCHESTRATOR, REWARD,
                 EVAL_QUEUE, *EVAL_SUPPORT.values(), ECD_DATA, CLEVR_DATA,
                 BASE / "config.json", EVAL_REPO, MMMU, BASELINE,
                 COMPARE, EVAL_ROOT, MERGED_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    for path, expected in ((ECD_DATA, ECD_SHA), (CLEVR_DATA, CLEVR_SHA),
                           (REWARD, REWARD_SHA)):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path} expected {expected}, got {actual}")
    reward = read_json(REWARD)
    if reward.get("mode") != "v1_2" or reward.get("eval_repo_commit") != EVAL_COMMIT:
        raise ValueError("Reward policy is not frozen v1.2")
    check_eval_worktree()
    if not (ECD_RUN / "plan.json").is_file():
        raise FileNotFoundError(f"First ECD run has no plan: {ECD_RUN}")
    first_plan = read_json(ECD_RUN / "plan.json")
    expected = {"ecd": (ECD_SHA, 500, 97)}
    for name, (digest, train_count, heldout_count) in expected.items():
        found = first_plan.get("datasets", {}).get(name, {})
        if (found.get("sha256") != digest or
                found.get("split_counts") != {"train": train_count, "heldout": heldout_count}):
            raise ValueError(f"First ECD plan has unexpected {name} input")
    if first_plan.get("run_order") != ["ecd_grpo", "ecd_dapo", "ecd_sapo"]:
        raise ValueError("First ECD run order changed")
    if (first_plan.get("base_revision") != BASE_REV or
            first_plan.get("reward_policy_sha256") != REWARD_SHA or
            first_plan.get("max_steps") != 1000 or
            first_plan.get("max_completion_length") != 512):
        raise ValueError("First ECD plan training config changed")
    if first_plan.get("train_script_sha256") != sha(TRAIN):
        raise ValueError("train.py changed since first ECD plan")
    return {
        "schema": "b_qc_v2_supervisor_v1",
        "created_at_utc": now(),
        "first_ecd_plan_sha256": sha(ECD_RUN / "plan.json"),
        "inputs": {
            "ecd_manifest_sha256": ECD_SHA,
            "clevr_manifest_sha256": CLEVR_SHA,
            "train_sha256": sha(TRAIN),
            "orchestrator_sha256": sha(ORCHESTRATOR),
            "eval_queue_sha256": sha(EVAL_QUEUE),
            "compare_sha256": sha(COMPARE),
            **{key: sha(path) for key, path in EVAL_SUPPORT.items()},
            "reward_sha256": REWARD_SHA,
        },
        "outputs": {
            "ecd": str(ECD_RUN), "clevr": str(CLEVR_RUN),
            "evaluation_plan": str(EVAL_PLAN), "evaluation_prefix": EVAL_PREFIX,
        },
    }


def check_unchanged(plan: dict) -> None:
    current = plan["inputs"]
    for path, key in ((ECD_DATA, "ecd_manifest_sha256"),
                      (CLEVR_DATA, "clevr_manifest_sha256"),
                      (TRAIN, "train_sha256"),
                      (ORCHESTRATOR, "orchestrator_sha256"),
                      (EVAL_QUEUE, "eval_queue_sha256"),
                      (COMPARE, "compare_sha256"),
                      *((path, key) for key, path in EVAL_SUPPORT.items()),
                      (REWARD, "reward_sha256")):
        if sha(path) != current[key]:
            raise RuntimeError(f"Input changed while supervisor ran: {path}")
    if sha(ECD_RUN / "plan.json") != plan["first_ecd_plan_sha256"]:
        raise RuntimeError("ECD first-run plan changed")
    check_eval_worktree()


def event(root: Path, stage: str, state: str, **extra) -> None:
    row = {"at_utc": now(), "stage": stage, "state": state, **extra}
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
    atomic_json(root / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify_run(name: str) -> Path:
    run_root, loss, dataset_sha = EXPECTED_RUNS[name]
    path = run_root / "runs" / name / "run_manifest.json"
    data = read_json(path)
    for key, expected in (
        ("status", "complete"), ("base_revision", BASE_REV),
        ("parent_kind", "base"), ("loss_type", loss),
        ("seed", 3407), ("max_steps", 1000),
        ("num_generations", 4), ("num_iterations", 2),
        ("max_completion_length", 512),
        ("reward_policy_sha256", REWARD_SHA),
    ):
        if data.get(key) != expected:
            raise ValueError(f"{path}: {key} differs from pinned run")
    if data.get("dataset", {}).get("manifest_sha256") != dataset_sha:
        raise ValueError(f"{path}: training dataset SHA differs")
    if not data.get("finite_loss") or not data.get("reload_success"):
        raise ValueError(f"{path}: finite loss or reload failed")
    if data.get("global_step") != 1000 or data.get("mixed_reward_groups", 0) < 1:
        raise ValueError(f"{path}: optimizer steps or mixed reward signal failed")
    if data.get("lora_delta_l1", 0) <= 0:
        raise ValueError(f"{path}: LoRA weights did not change")
    adapter = run_root / "runs" / name / "adapter" / "adapter_model.safetensors"
    if sha(adapter) != data.get("adapter_sha256"):
        raise ValueError(f"{path}: adapter SHA mismatch")
    return path


def wait_first_ecd(root: Path, plan: dict, timeout_seconds: int) -> None:
    event(root, "wait_ecd_grpo", "waiting", queue=str(ECD_RUN))
    start = time.monotonic()
    next_event = start
    while True:
        check_unchanged(plan)
        summary = ECD_RUN / "summary.json"
        status = read_json(summary) if summary.is_file() else {}
        state = status.get("status")
        if state in {"failed", "complete_with_no_signal"}:
            raise RuntimeError(f"ECD GRPO queue ended without validated model: {state}")
        if state in {"paused", "complete"}:
            verify_run("ecd_grpo")
            event(root, "wait_ecd_grpo", "verified", summary_state=state,
                  run_manifest=str(ECD_RUN / "runs/ecd_grpo/run_manifest.json"))
            return
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for ECD GRPO")
        if time.monotonic() >= next_event:
            queue_status = (ECD_RUN / "status.txt").read_text(encoding="utf-8") if (ECD_RUN / "status.txt").is_file() else ""
            step = next((line.split(": ", 1)[1] for line in queue_status.splitlines() if line.startswith("step: ")), "unknown")
            event(root, "wait_ecd_grpo", "heartbeat", step=step,
                  elapsed_seconds=round(time.monotonic() - start))
            next_event = time.monotonic() + 30
        time.sleep(10)


def train_command(dataset: str, *, pause_first: bool) -> list[str]:
    if dataset == "ecd":
        manifest, run_root, heldout, order = (
            ECD_DATA, ECD_RUN, 97, "ecd_grpo,ecd_dapo,ecd_sapo")
    elif dataset == "mammoth":
        manifest, run_root, heldout, order = (
            CLEVR_DATA, CLEVR_RUN, 100, "mammoth_grpo,mammoth_dapo,mammoth_sapo")
    else:
        raise ValueError(dataset)
    cmd = [str(TRAIN_PY), "-u", str(ORCHESTRATOR), "--execute",
           "--ecd-manifest" if dataset == "ecd" else "--mammoth-manifest", str(manifest),
           "--run-order", order, "--expected-train", "500",
           "--expected-heldout", str(heldout), "--max-steps", "1000",
           "--save-steps", "250", "--reward-policy", str(REWARD),
           "--train-script", str(TRAIN), "--python", str(TRAIN_PY),
           "--base-model", str(BASE), "--output-root", str(run_root)]
    if pause_first:
        cmd += ["--pause-after", "1"]
    return cmd


def assert_gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    active = [line.strip() for line in result.stdout.splitlines()
              if line.strip() and "No running" not in line]
    if active:
        raise RuntimeError(f"GPU is occupied by another compute process: {active}")


def run_stage(root: Path, plan: dict, stage: str, cmd: list[str], timeout_seconds: int,
              *, need_gpu: bool) -> None:
    check_unchanged(plan)
    if need_gpu:
        assert_gpu_idle()
    log = root / "logs" / f"{stage}.log"
    event(root, stage, "starting", command=cmd, log=str(log))
    start = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        event(root, stage, "running", pid=process.pid, log=str(log))
        next_event = start + 30
        while process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed > timeout_seconds:
                # Leave the independent process group and all output intact for
                # inspection; stop the supervisor instead of killing a GPU job.
                raise TimeoutError(f"{stage} exceeded {timeout_seconds}s; process {process.pid} remains running")
            if time.monotonic() >= next_event:
                child = {}
                if stage.startswith("mammoth"):
                    child = queue_progress(CLEVR_RUN)
                elif stage.startswith("ecd"):
                    child = queue_progress(ECD_RUN)
                elif stage == "eval_queue" and (EVAL_PLAN / "progress.json").is_file():
                    child = read_json(EVAL_PLAN / "progress.json")
                event(root, stage, "heartbeat", pid=process.pid,
                      elapsed_seconds=round(elapsed), log=str(log), child=child)
                next_event = time.monotonic() + 30
            time.sleep(10)
        code = process.wait()
    if code:
        event(root, stage, "failed", exit_code=code, log=str(log))
        raise RuntimeError(f"{stage} exited {code}; inspect {log}")
    event(root, stage, "process_complete", elapsed_seconds=round(time.monotonic() - start),
          log=str(log))


def queue_progress(run_root: Path) -> dict:
    status = run_root / "status.txt"
    if not status.is_file():
        return {}
    result = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in {"queue", "current_run", "run_index", "step", "elapsed_seconds"}:
                result[key] = value
    return result


def queue_state(root: Path, *, allow_blocked: bool = False) -> str:
    summary = read_json(root / "summary.json")
    status = summary.get("status")
    allowed = {"paused", "complete"}
    if allow_blocked:
        allowed.add("complete_with_no_signal")
    if status not in allowed:
        raise RuntimeError(f"Unexpected training queue state {status}: {root}")
    return status


def eval_command(manifests: list[Path], *, dry_run: bool) -> list[str]:
    cmd = [str(TRAIN_PY), "-u", str(EVAL_QUEUE)]
    for manifest in manifests:
        cmd += ["--train-manifest", str(manifest)]
    cmd += [
        "--base-model", str(BASE), "--repo-root", str(EVAL_REPO),
        "--data-root", str(MMMU), "--train-python", str(TRAIN_PY),
        "--eval-python", str(EVAL_PY), "--output-root", str(EVAL_ROOT),
        "--merged-root", str(MERGED_ROOT), "--baseline-900", str(BASELINE),
        "--compare-script", str(COMPARE), "--plan-dir", str(EVAL_PLAN),
        "--run-prefix", EVAL_PREFIX, "--expected-steps", "1000",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def check_eval_destinations() -> None:
    if EVAL_PLAN.exists():
        raise FileExistsError(f"Evaluation plan already exists: {EVAL_PLAN}")
    if list(EVAL_ROOT.glob(EVAL_PREFIX + "_*")) or list(MERGED_ROOT.glob(EVAL_PREFIX + "_*")):
        raise FileExistsError(f"Evaluation prefix {EVAL_PREFIX} already has outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read-only path/hash/plan validation")
    mode.add_argument("--execute", action="store_true", help="Start continuation and evaluation")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "mmdl_grpo_runs/rl_ablation_20261119_supervisor_qcv2")
    args = parser.parse_args()
    plan = check_files()
    if args.check:
        print(json.dumps({"status": "check_passed", "plan": plan}, ensure_ascii=False, indent=2))
        return
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    atomic_json(root / "plan.json", plan)
    try:
        wait_first_ecd(root, plan, 4 * 3600)
        assert_gpu_idle()
        if CLEVR_RUN.exists():
            raise FileExistsError(f"CLEVR training output already exists: {CLEVR_RUN}")
        run_stage(root, plan, "mammoth_grpo", train_command("mammoth", pause_first=True),
                  4 * 3600, need_gpu=True)
        mammoth_first_state = queue_state(CLEVR_RUN, allow_blocked=True)
        mammoth_signal = mammoth_first_state == "paused"
        if mammoth_signal:
            verify_run("mammoth_grpo")
        else:
            event(root, "mammoth_grpo", "blocked_no_learning_signal")
        run_stage(root, plan, "ecd_remaining", train_command("ecd", pause_first=False),
                  9 * 3600, need_gpu=True)
        if queue_state(ECD_RUN) != "complete":
            raise RuntimeError("ECD queue did not complete all three methods")
        manifests = [verify_run(name) for name in ("ecd_grpo", "ecd_dapo", "ecd_sapo")]
        if mammoth_signal:
            run_stage(root, plan, "mammoth_remaining", train_command("mammoth", pause_first=False),
                      9 * 3600, need_gpu=True)
            if queue_state(CLEVR_RUN) != "complete":
                raise RuntimeError("MAmmoTH queue did not complete all three methods")
            manifests += [verify_run(name) for name in ("mammoth_grpo", "mammoth_dapo", "mammoth_sapo")]
        if len(manifests) < 2:
            raise RuntimeError("Fewer than two completed candidates for screening")
        check_eval_destinations()
        run_stage(root, plan, "eval_dry_run", eval_command(manifests, dry_run=True),
                  600, need_gpu=False)
        check_eval_destinations()
        run_stage(root, plan, "eval_queue", eval_command(manifests, dry_run=False),
                  36 * 3600, need_gpu=True)
        summary = EVAL_PLAN / "screening_summary.json"
        if not summary.is_file():
            raise RuntimeError(f"Evaluation queue exited without screening summary: {summary}")
        atomic_json(root / "summary.json", {
            "status": "complete", "completed_at_utc": now(),
            "training_manifests": [str(path) for path in manifests],
            "screening_summary": str(summary),
            "official_4090_started": False,
        })
        event(root, "supervisor", "complete", summary=str(summary), candidates=len(manifests))
    except Exception as error:
        atomic_json(root / "summary.json", {"status": "failed", "error": repr(error),
                                             "failed_at_utc": now()})
        event(root, "supervisor", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
