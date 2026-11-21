"""Continue a passed DPO(e)→GRPO probe through a 6000 paired MMMU screen.

This is deliberately single-use. Every output directory must be absent, all
stage gates fail closed, and no 4090 or three-seed evaluation is launched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STAGE1 = Path("/data/chaewon/mmdl_stage1_paper_style_20261120")
STAGE2 = Path("/data/chaewon/mmdl_stage2_egrpo_20261122")
OPS = Path("/data/chaewon/mmdl_b_eval_ops_egrpo_20261122")
TRAIN_PY = Path("/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python")
MERGE_PY = Path("/data/chaewon/mmdl_envs/grpo_pilot/bin/python")
EVAL_PY = Path("/data/chaewon/mmdl_envs/vllm011_repro/bin/python3.10")
BASE = Path("/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17")
PARENT = Path("/data/chaewon/mmdl_merged_models/dpo_e_trunc_from_a_20261121")
PARENT_EVAL = Path("/data/chaewon/mmdl_eval_runs/a_dpo_e_parent_900_s3407_20261122")
A_ADAPTER = Path("/data/chaewon/mmdl_a_e_handoff_20261121/adapter")
A_SHA = "7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa"
PROBE = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_probe_20261122")
PROBE_QUEUE = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_probe_queue_20261122")
FULL = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_full_20261122")
MERGED = Path("/data/chaewon/mmdl_merged_models/dpo_e_grpo_ecd500_20261122")
EVAL_ROOT = Path("/data/chaewon/mmdl_eval_runs")
SMOKE_NAME = "dpo_e_grpo_ecd500_smoke20_20261122"
SCREEN_NAME = "dpo_e_grpo_ecd500_900_s3407_20261122"
SMOKE = EVAL_ROOT / SMOKE_NAME
SCREEN = EVAL_ROOT / SCREEN_NAME
QUEUE = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_full_queue_v2_20261122")
REPO = Path("/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2")
DATA = Path("/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
COMPARE = Path("/data/chaewon/mmdl_b_compare/compare_runs.py")
MAX_PROBE_WAIT_SECONDS = 3 * 60 * 60


def emit(event: str, **detail: object) -> None:
    payload = {"time": datetime.now().astimezone().isoformat(), "event": event, **detail}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    target = QUEUE / "status.json"
    temporary = QUEUE / "status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"event": "command_start", "argv": command}), flush=True)
    subprocess.run(command, env=env, check=True)


def wait_probe() -> None:
    deadline = time.monotonic() + MAX_PROBE_WAIT_SECONDS
    while True:
        status_path = PROBE_QUEUE / "status.json"
        if status_path.is_file():
            status = read_json(status_path)
            event = status.get("event")
            if event == "queue_failed":
                raise RuntimeError(f"Upstream probe queue failed: {status.get('error')}")
            if event == "probe_passed":
                manifest = read_json(PROBE / "run_manifest.json")
                report = read_json(PROBE / "completion_report.json")
                metrics = read_json(PROBE / "metric_summary.json")
                step = metrics.get("steps", [None])[0]
                if (manifest.get("status") != "completed"
                        or manifest.get("method") != "grpo"
                        or manifest.get("mode") != "probe"
                        or manifest.get("parent_kind") != "merged_dpo"
                        or manifest.get("base_model") != str(PARENT)
                        or manifest.get("parent_adapter_sha256") != A_SHA
                        or report.get("status") != "complete"
                        or report.get("optimizer_steps") != 1
                        or report.get("positive_grad_steps") != 1
                        or report.get("generated_rollouts") != 500
                        or report.get("used_rollouts") != 500
                        or not isinstance(step, dict)
                        or not 0.05 <= step.get("reward.accuracy", -1) <= 0.95
                        or step.get("reward.answer_present", -1) < 0.80
                        or step.get("reward.at_cap_proxy", 1) > 0.20):
                    raise ValueError("Probe status and saved learning-signal metrics differ")
                emit("probe_reverified", reward_accuracy=step["reward.accuracy"],
                     answer_present=step["reward.answer_present"],
                     at_cap_proxy=step["reward.at_cap_proxy"])
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("Probe did not pass within three hours")
        time.sleep(15)


def wait_gpu_idle(stage: str) -> None:
    deadline = time.monotonic() + 5 * 60
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = int(output.splitlines()[0].strip())
        if used <= 4096:
            emit("gpu_idle", stage=stage, used_mib=used)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU stayed busy before {stage}: {used} MiB")
        emit("waiting_gpu_release", stage=stage, used_mib=used)
        time.sleep(10)


def verify_parent_eval() -> None:
    parent_handoff = read_json(PARENT_EVAL / "handoff.json")
    if (parent_handoff.get("run_name") != PARENT_EVAL.name
            or parent_handoff.get("mode") != "screen"
            or parent_handoff.get("seeds") != ["3407"]
            or parent_handoff.get("model_path") != str(PARENT)
            or parent_handoff.get("model_manifest_sha256") != sha256(PARENT / "model_manifest.json")
            or not parent_handoff.get("completed_at_utc")):
        raise ValueError("DPO(e) parent 900-item handoff differs before training")
    comparison = read_json(PARENT_EVAL / "comparison" / "comparison_summary.json")
    paired = comparison.get("per_seed", {}).get("3407", {})
    if (paired.get("items_compared") != 900
            or paired.get("full_900_coverage") is not True
            or paired.get("input_exact_agreement") != 900
            or paired.get("doc_hash_agreement") != 900):
        raise ValueError("DPO(e) parent 900-item coverage/input/image parity differs")


def verify_full() -> tuple[Path, str]:
    manifest = read_json(FULL / "run_manifest.json")
    report = read_json(FULL / "completion_report.json")
    visits = read_json(FULL / "visit_audit.json")
    if (manifest.get("status") != "completed"
            or manifest.get("process_exit_code") != 0
            or manifest.get("method") != "grpo"
            or manifest.get("mode") != "full"
            or manifest.get("parent_kind") != "merged_dpo"
            or manifest.get("base_model") != str(PARENT)
            or manifest.get("parent_adapter_sha256") != A_SHA
            or report.get("status") != "complete"
            or report.get("optimizer_steps") != 15
            or report.get("positive_grad_steps") != 15
            or report.get("generated_rollouts") != 7500
            or report.get("used_rollouts") != 7500
            or visits.get("status") != "visit_audit_passed"
            or visits.get("used_unique_ids") != 500
            or visits.get("used_visit_histogram") != {"3": 500}
            or visits.get("generated_rollouts") != 7500
            or visits.get("used_rollouts") != 7500):
        raise ValueError("Full GRPO completion, parent, or three-pass visit gate failed")
    parsed = subprocess.check_output(
        [str(TRAIN_PY), str(STAGE2 / "parse_full_metrics.py"),
         str(FULL / "console.log"), "--expected-steps", "15"],
        text=True,
    )
    metrics = json.loads(parsed)
    metric_path = FULL / "metric_summary.json"
    if metric_path.exists():
        raise FileExistsError(metric_path)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    steps = metrics.get("steps")
    if not isinstance(steps, list) or len(steps) != 15:
        raise ValueError("Expected exactly 15 metric rows")
    for index, step in enumerate(steps, start=1):
        if (step.get("step") != index
                or step.get("actor.grad_norm", 0) <= 0
                or step.get("actor.lr") != 1e-6
                or step.get("actor.ppo_kl") != 0
                or step.get("reward.overlong") != 0):
            raise ValueError(f"Full GRPO metric gate failed at step {index}: {step}")
    reward_values = [step["reward.accuracy"] for step in steps]
    answer_values = [step["reward.answer_present"] for step in steps]
    cap_values = [step["reward.at_cap_proxy"] for step in steps]
    reward_mean = sum(reward_values) / 15
    answer_mean = sum(answer_values) / 15
    cap_mean = sum(cap_values) / 15
    if not (0.05 <= reward_mean <= 0.95 and answer_mean >= 0.80 and cap_mean <= 0.20):
        raise ValueError(f"Full GRPO aggregate reward/Answer/cap gate failed: "
                         f"reward={reward_mean:.4f}, Answer={answer_mean:.4f}, cap={cap_mean:.4f}")
    adapter = FULL / "checkpoints/global_step_15/actor/lora_adapter"
    adapter_sha = report.get("adapter_sha256")
    if not isinstance(adapter_sha, str) or sha256(adapter / "adapter_model.safetensors") != adapter_sha:
        raise ValueError("Full GRPO adapter SHA256 differs from completion report")
    lineage_args = [
        str(TRAIN_PY), str(STAGE2 / "build_fixed300_lineage.py"),
        "--run-manifest", str(FULL / "run_manifest.json"),
        "--completion-report", str(FULL / "completion_report.json"),
        "--train-file", str(STAGE1 / "scripts/ecd500_easyr1.jsonl"),
        "--validation-file", str(STAGE1 / "scripts/ecd97_holdout_easyr1.jsonl"),
        "--config-file", str(FULL / "resolved_config.yaml"),
        "--source-config-file", str(STAGE2 / "config_ecd500.yaml"),
        "--reward-file", str(STAGE1 / "scripts/reward_v12.py"),
        "--adapter-dir", str(adapter),
        "--output", str(FULL / "fixed300_training_manifest.json"),
    ]
    run(lineage_args + ["--dry-run"])
    emit("full_verified", adapter_sha256=adapter_sha,
         optimizer_steps=15, used_rollouts=7500,
         reward_mean=round(reward_mean, 4), reward_min=min(reward_values), reward_max=max(reward_values),
         answer_mean=round(answer_mean, 4), answer_min=min(answer_values), answer_max=max(answer_values),
         cap_mean=round(cap_mean, 4), cap_min=min(cap_values), cap_max=max(cap_values))
    return adapter, adapter_sha


def merge_and_lineage(adapter: Path, adapter_sha: str) -> Path:
    arguments = [
        str(MERGE_PY), str(OPS / "merge_lora.py"), "--base-model", str(BASE),
        "--adapter", str(A_ADAPTER), "--adapter-role", "dpo", "--adapter-sha256", A_SHA,
        "--adapter", str(adapter), "--adapter-role", "grpo", "--adapter-sha256", adapter_sha,
        "--output-dir", str(MERGED),
    ]
    run(arguments + ["--preflight-only"])
    emit("merge_start", model=str(MERGED))
    run(arguments)
    model = read_json(MERGED / "model_manifest.json")
    adapters = model.get("adapters")
    if (model.get("status") != "complete"
            or model.get("parent_kind") != "merged_dpo_grpo"
            or model.get("model_path") != str(MERGED)
            or not isinstance(adapters, list)
            or [(a.get("role"), a.get("weights_sha256")) for a in adapters] !=
               [("dpo", A_SHA), ("grpo", adapter_sha)]
            or (MERGED / "_INCOMPLETE").exists()):
        raise ValueError("Merged model is not the ordered base→DPO(e)→GRPO candidate")
    lineage = FULL / "fixed300_training_manifest.json"
    run([
        str(TRAIN_PY), str(STAGE2 / "build_fixed300_lineage.py"),
        "--run-manifest", str(FULL / "run_manifest.json"),
        "--completion-report", str(FULL / "completion_report.json"),
        "--train-file", str(STAGE1 / "scripts/ecd500_easyr1.jsonl"),
        "--validation-file", str(STAGE1 / "scripts/ecd97_holdout_easyr1.jsonl"),
        "--config-file", str(FULL / "resolved_config.yaml"),
        "--source-config-file", str(STAGE2 / "config_ecd500.yaml"),
        "--reward-file", str(STAGE1 / "scripts/reward_v12.py"),
        "--adapter-dir", str(adapter),
        "--model-manifest", str(MERGED / "model_manifest.json"),
        "--output", str(lineage),
    ])
    record = read_json(lineage)
    if (record.get("status") != "complete"
            or record.get("parent_kind") != "merged_dpo"
            or record.get("adapter_sha256") != adapter_sha
            or record.get("parent_adapter_sha256") != A_SHA):
        raise ValueError("GRPO training lineage differs from merged model")
    emit("merge_and_lineage_verified", model_manifest_sha256=sha256(MERGED / "model_manifest.json"),
         lineage_sha256=sha256(lineage))
    return lineage


def eval_command(name: str, mode: str, lineage: Path) -> list[str]:
    return [
        str(MERGE_PY), str(OPS / "candidate_eval.py"),
        "--model-manifest", str(MERGED / "model_manifest.json"),
        "--train-run-manifest", str(lineage),
        "--repo-root", str(REPO), "--data-root", str(DATA),
        "--eval-python", str(EVAL_PY), "--output-root", str(EVAL_ROOT),
        "--run-name", name, "--mode", mode, "--baseline-run", str(PARENT_EVAL),
        "--compare-script", str(COMPARE), "--verify-model-hashes",
    ]


def verify_eval(run_dir: Path, mode: str, expected: int) -> dict:
    handoff = read_json(run_dir / "handoff.json")
    if (handoff.get("mode") != mode
            or handoff.get("run_name") != run_dir.name
            or handoff.get("seeds") != ["3407"]
            or handoff.get("limit") != (20 if mode == "smoke" else None)
            or handoff.get("model_path") != str(MERGED)
            or handoff.get("model_manifest_sha256") != sha256(MERGED / "model_manifest.json")
            or handoff.get("baseline_run") != str(PARENT_EVAL)
            or not handoff.get("completed_at_utc")):
        raise ValueError(f"{mode} evaluation handoff differs")
    files = handoff.get("result_files", {}).get("3407", {})
    for key in ("samples", "results", "run_meta"):
        record = files.get(key, {})
        file = Path(record.get("path", ""))
        if not file.is_file() or run_dir.resolve() not in file.resolve().parents or sha256(file) != record.get("sha256"):
            raise ValueError(f"{mode} evaluation {key} file path/SHA differs")
    sample_file = Path(files["samples"]["path"])
    if sum(1 for line in sample_file.open(encoding="utf-8") if line.strip()) != expected:
        raise ValueError(f"{mode} evaluation sample count differs from {expected}")
    meta = Path(files["run_meta"]["path"]).read_text(encoding="utf-8")
    if "exit_status: 0" not in meta or "gpu: NVIDIA RTX PRO 6000" not in meta:
        raise ValueError(f"{mode} evaluation did not finish on the same 6000")
    comparison_path = Path(handoff.get("comparison_summary", ""))
    if not comparison_path.is_file() or run_dir.resolve() not in comparison_path.resolve().parents:
        raise ValueError(f"{mode} comparison missing or outside run directory")
    comparison = read_json(comparison_path).get("per_seed", {}).get("3407", {})
    if (comparison.get("items_compared") != expected
            or comparison.get("full_900_coverage") is not (expected == 900)
            or comparison.get("input_exact_agreement") != expected
            or comparison.get("doc_hash_agreement") != expected
            or not isinstance(comparison.get("candidate_correct"), int)
            or not isinstance(comparison.get("baseline_correct"), int)):
        raise ValueError(f"{mode} paired comparison coverage or input/image parity failed")
    return comparison


def main() -> None:
    for required in (STAGE1, STAGE2, OPS, TRAIN_PY, MERGE_PY, EVAL_PY,
                     BASE, PARENT, PARENT_EVAL, A_ADAPTER, REPO, DATA, COMPARE):
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256(A_ADAPTER / "adapter_model.safetensors") != A_SHA:
        raise ValueError("A's original DPO(e) adapter SHA256 differs before training")
    parent_model = read_json(PARENT / "model_manifest.json")
    if (parent_model.get("status") != "complete"
            or parent_model.get("parent_kind") != "merged_base_dpo"
            or [(a.get("role"), a.get("weights_sha256"))
                for a in parent_model.get("adapters", [])] != [("dpo", A_SHA)]):
        raise ValueError("DPO(e) parent manifest differs before training")
    for fresh in (QUEUE, FULL, MERGED, SMOKE, SCREEN):
        if fresh.exists():
            raise FileExistsError(f"Refusing to resume or overwrite: {fresh}")
    QUEUE.mkdir(parents=True, exist_ok=False)
    with (QUEUE / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        emit("waiting_probe", probe_queue=str(PROBE_QUEUE))
        wait_probe()
        verify_parent_eval()
        wait_gpu_idle("full training")
        env = os.environ.copy()
        env.update({
            "MMDL_STAGE1_PACKAGE_ROOT": str(STAGE1),
            "MMDL_STAGE2_ROOT": str(STAGE2),
            "MMDL_TRAIN_PYTHON": str(TRAIN_PY),
            "MMDL_BASE_MODEL": str(PARENT),
            "MMDL_DPO_PARENT_MANIFEST": str(PARENT / "model_manifest.json"),
            "MMDL_DPO_ADAPTER_SHA256": A_SHA,
        })
        emit("full_training_start", run_dir=str(FULL))
        run(["bash", str(STAGE2 / "run_ecd500.sh"), "grpo", "full", str(FULL)], env=env)
        adapter, adapter_sha = verify_full()
        lineage = merge_and_lineage(adapter, adapter_sha)
        wait_gpu_idle("20-item smoke")
        emit("smoke_start", run_dir=str(SMOKE))
        smoke_command = eval_command(SMOKE_NAME, "smoke", lineage)
        run(smoke_command + ["--dry-run"])
        run(smoke_command)
        smoke = verify_eval(SMOKE, "smoke", 20)
        emit("smoke_passed", candidate_correct=smoke["candidate_correct"],
             parent_correct=smoke["baseline_correct"])
        wait_gpu_idle("900-item screen")
        emit("screen_start", run_dir=str(SCREEN))
        screen_command = eval_command(SCREEN_NAME, "screen", lineage)
        run(screen_command + ["--dry-run"])
        run(screen_command)
        screen = verify_eval(SCREEN, "screen", 900)
        emit("screen_passed", candidate_correct=screen["candidate_correct"],
             parent_correct=screen["baseline_correct"],
             correct_difference=screen["correct_difference"],
             score_difference_pp=screen["score_difference_pp"],
             note="6000 one-seed paired screen; official 4090 three-seed result remains separate")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "queue_failed", "error": repr(exc)}), file=sys.stderr, flush=True)
        if QUEUE.is_dir():
            emit("queue_failed", error=repr(exc))
        raise
