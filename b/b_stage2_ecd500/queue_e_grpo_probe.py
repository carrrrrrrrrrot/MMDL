"""Wait for the DPO(e) 6000 screen, then run one fresh ECD500 GRPO probe.

This queue intentionally stops after the probe. Full GRPO, merge, and the
candidate MMMU screen require the probe metrics to be reviewed first.
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


PARENT = Path("/data/chaewon/mmdl_merged_models/dpo_e_trunc_from_a_20261121")
PARENT_EVAL = Path("/data/chaewon/mmdl_eval_runs/a_dpo_e_parent_900_s3407_20261122")
STAGE1 = Path("/data/chaewon/mmdl_stage1_paper_style_20261120")
STAGE2 = Path("/data/chaewon/mmdl_stage2_egrpo_20261122")
TRAIN_PYTHON = Path("/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python")
PROBE = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_probe_20261122")
QUEUE = Path("/data/chaewon/mmdl_stage2_runs/e_grpo_probe_queue_20261122")
DPO_SHA = "7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa"
PARENT_LINEAGE_SHA = "77b16bce101f14d895e821a4ed99e54243321ad3ee05c9b65a71a609440926a4"
EVAL_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
MAX_WAIT_SECONDS = 4 * 60 * 60


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


def within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=True)
    return resolved == root or root in resolved.parents


def parent_ready() -> bool:
    handoff_path = PARENT_EVAL / "handoff.json"
    if not handoff_path.is_file():
        meta = PARENT_EVAL / "seed_3407" / "run_meta.txt"
        if meta.is_file() and "exit_status:" in meta.read_text(encoding="utf-8"):
            if "exit_status: 0" not in meta.read_text(encoding="utf-8"):
                raise RuntimeError(f"Parent evaluation failed: {meta}")
        return False

    handoff = read_json(handoff_path)
    model_manifest = read_json(PARENT / "model_manifest.json")
    adapters = model_manifest.get("adapters")
    if (model_manifest.get("status") != "complete"
            or model_manifest.get("parent_kind") != "merged_base_dpo"
            or model_manifest.get("lineage", {}).get("lineage_sha256") != PARENT_LINEAGE_SHA
            or not isinstance(adapters, list) or len(adapters) != 1
            or (adapters[0].get("role"), adapters[0].get("weights_sha256")) != ("dpo", DPO_SHA)):
        raise ValueError("DPO(e) parent manifest/adapter lineage differs")
    if (handoff.get("run_name") != PARENT_EVAL.name
            or handoff.get("mode") != "screen"
            or handoff.get("seeds") != ["3407"]
            or handoff.get("limit") is not None
            or handoff.get("model_path") != str(PARENT)
            or handoff.get("evaluation_repo_commit") != EVAL_COMMIT
            or handoff.get("model_manifest_sha256") != sha256(PARENT / "model_manifest.json")
            or not handoff.get("completed_at_utc")):
        raise ValueError("Parent evaluation handoff differs from expected 900-item v1.2 screen")
    files = handoff.get("result_files", {}).get("3407", {})
    for key in ("samples", "results", "run_meta"):
        record = files.get(key, {})
        path = Path(record.get("path", ""))
        if not within(path, PARENT_EVAL.resolve(strict=True)) or sha256(path) != record.get("sha256"):
            raise ValueError(f"Parent {key} path or SHA256 differs")
    samples = Path(files["samples"]["path"])
    if sum(1 for line in samples.open(encoding="utf-8") if line.strip()) != 900:
        raise ValueError("Parent screen does not contain exactly 900 sample rows")
    meta_text = Path(files["run_meta"]["path"]).read_text(encoding="utf-8")
    if "exit_status: 0" not in meta_text or "gpu: NVIDIA RTX PRO 6000" not in meta_text:
        raise ValueError("Parent screen did not finish on the 6000 with exit status 0")
    comparison_path = Path(handoff.get("comparison_summary", ""))
    if not within(comparison_path, PARENT_EVAL.resolve(strict=True)):
        raise ValueError("Parent comparison path is outside its run directory")
    comparison = read_json(comparison_path).get("per_seed", {}).get("3407", {})
    if (comparison.get("items_compared") != 900
            or comparison.get("full_900_coverage") is not True
            or comparison.get("input_exact_agreement") != 900
            or comparison.get("doc_hash_agreement") != 900):
        raise ValueError("Parent/base 900-item paired comparison failed coverage/parity")
    emit("parent_screen_verified",
         parent_correct=comparison.get("candidate_correct"),
         base_correct=comparison.get("baseline_correct"))
    return True


def wait_gpu_idle() -> None:
    deadline = time.monotonic() + 5 * 60
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = int(output.splitlines()[0].strip())
        if used <= 4096:
            emit("gpu_idle", used_mib=used)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU remained busy for five minutes: {used} MiB")
        emit("waiting_gpu_release", used_mib=used)
        time.sleep(10)


def check_probe() -> None:
    manifest = read_json(PROBE / "run_manifest.json")
    report = read_json(PROBE / "completion_report.json")
    if (manifest.get("status") != "completed"
            or manifest.get("process_exit_code") != 0
            or manifest.get("method") != "grpo"
            or manifest.get("mode") != "probe"
            or manifest.get("parent_kind") != "merged_dpo"
            or manifest.get("base_model") != str(PARENT)
            or manifest.get("parent_adapter_sha256") != DPO_SHA
            or report.get("status") != "complete"
            or report.get("optimizer_steps") != 1
            or report.get("positive_grad_steps") != 1
            or report.get("generated_rollouts") != 500
            or report.get("used_rollouts") != 500):
        raise ValueError("GRPO probe completion/parent/learning-signal gate failed")
    parsed = subprocess.check_output(
        [str(TRAIN_PYTHON), str(STAGE2 / "parse_full_metrics.py"),
         str(PROBE / "console.log"), "--expected-steps", "1"],
        text=True,
    )
    metrics = json.loads(parsed)
    metric_path = PROBE / "metric_summary.json"
    if metric_path.exists():
        raise FileExistsError(metric_path)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    step = metrics["steps"][0]
    if (step["step"] != 1
            or not 0.05 <= step["reward.accuracy"] <= 0.95
            or step["reward.answer_present"] < 0.80
            or step["reward.at_cap_proxy"] > 0.20
            or step["actor.grad_norm"] <= 0
            or step["actor.lr"] != 1e-6
            or step["actor.ppo_kl"] != 0
            or step["reward.overlong"] != 0):
        raise ValueError(f"Probe reward/Answer/cap/optimizer gate failed: {step}")
    adapter = PROBE / "checkpoints/global_step_1/actor/lora_adapter"
    subprocess.run([
        str(TRAIN_PYTHON), str(STAGE2 / "build_fixed300_lineage.py"),
        "--run-manifest", str(PROBE / "run_manifest.json"),
        "--completion-report", str(PROBE / "completion_report.json"),
        "--train-file", str(STAGE1 / "scripts/ecd500_easyr1.jsonl"),
        "--validation-file", str(STAGE1 / "scripts/ecd5_holdout_easyr1.jsonl"),
        "--config-file", str(PROBE / "resolved_config.yaml"),
        "--source-config-file", str(STAGE2 / "config_ecd500.yaml"),
        "--reward-file", str(STAGE1 / "scripts/reward_v12.py"),
        "--adapter-dir", str(adapter),
        "--output", str(PROBE / "probe_lineage_unused.json"),
        "--probe-check", "--dry-run",
    ], check=True)
    emit("probe_passed",
         reward_accuracy=step["reward.accuracy"],
         answer_present=step["reward.answer_present"],
         at_cap_proxy=step["reward.at_cap_proxy"],
         adapter_sha256=report["adapter_sha256"],
         wall_seconds=report["wall_seconds"])


def main() -> None:
    if not PARENT_EVAL.is_dir() or not PARENT.is_dir():
        raise FileNotFoundError("DPO(e) parent model or its evaluation run is missing")
    if PROBE.exists() or QUEUE.exists():
        raise FileExistsError("Probe or queue path exists; refuse to resume or overwrite")
    if not STAGE2.is_dir() or not STAGE1.is_dir() or not TRAIN_PYTHON.is_file():
        raise FileNotFoundError("Pinned Stage1/Stage2 code or training Python is missing")
    QUEUE.mkdir(parents=True, exist_ok=False)
    with (QUEUE / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        emit("waiting_parent", parent_eval=str(PARENT_EVAL))
        deadline = time.monotonic() + MAX_WAIT_SECONDS
        while not parent_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError("Parent 900-item handoff did not appear within four hours")
            time.sleep(30)
        wait_gpu_idle()
        emit("starting_probe", probe=str(PROBE))
        env = os.environ.copy()
        env.update({
            "MMDL_STAGE1_PACKAGE_ROOT": str(STAGE1),
            "MMDL_STAGE2_ROOT": str(STAGE2),
            "MMDL_TRAIN_PYTHON": str(TRAIN_PYTHON),
            "MMDL_BASE_MODEL": str(PARENT),
            "MMDL_DPO_PARENT_MANIFEST": str(PARENT / "model_manifest.json"),
            "MMDL_DPO_ADAPTER_SHA256": DPO_SHA,
        })
        subprocess.run(["bash", str(STAGE2 / "run_ecd500.sh"),
                        "grpo", "probe", str(PROBE)], env=env, check=True)
        check_probe()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "queue_failed", "error": repr(exc)}), file=sys.stderr, flush=True)
        if QUEUE.is_dir():
            emit("queue_failed", error=repr(exc))
        raise
