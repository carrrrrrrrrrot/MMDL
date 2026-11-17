"""Resume B's real-data GRPO only after A/C handoffs pass explicit checks.

The default action is read-only. ``prepare`` performs only CPU conversion,
merge, and validation; ``train`` additionally starts GRPO on the GPU.
``wait`` polls the same gate and can prepare/train when inputs appear.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
SHARED_MERGE_SHA256 = "f0289466a0874154062498b6f988dcd6422c5253e6f31d8237a6da861543c245"
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED = (
    "python", "base_model", "shared_merge_script", "merge_wrapper",
    "sft_adapter", "sft_adapter_sha256", "a_converted_jsonl",
    "c_approved_ids", "c_clearance_json", "image_root", "dataset_revision",
    "converter", "train_script", "reward_policy", "work_root",
    "max_steps", "min_train_rows", "candidate_eval", "eval_python",
    "eval_repo", "eval_data", "eval_output_root", "baseline_run",
    "compare_script", "eval_run_prefix",
)
PATH_KEYS = (
    "python", "base_model", "shared_merge_script", "merge_wrapper",
    "sft_adapter", "a_converted_jsonl", "c_approved_ids",
    "c_clearance_json", "image_root", "converter", "train_script",
    "reward_policy", "candidate_eval", "eval_python", "eval_repo",
    "eval_data", "eval_output_root", "baseline_run", "compare_script",
)
DIR_KEYS = ("base_model", "sft_adapter", "image_root", "eval_repo", "eval_data", "eval_output_root", "baseline_run")
TERMINAL_QUEUE_STAGES = {"complete_exploratory_one_seed", "screen_blocked_baseline_mismatch"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _status(config: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED if key not in config or config[key] in (None, "")]
    if missing:
        return {"status": "waiting", "missing": [f"config.{key}" for key in missing]}
    if not isinstance(config["max_steps"], int) or config["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(config["min_train_rows"], int) or config["min_train_rows"] < 1:
        raise ValueError("min_train_rows must be a positive integer")
    if not isinstance(config["dataset_revision"], str) or not config["dataset_revision"].strip():
        raise ValueError("dataset_revision must be a non-empty string")
    if not isinstance(config["eval_run_prefix"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", config["eval_run_prefix"]):
        raise ValueError("eval_run_prefix must be a safe unique run name")
    if not HEX_SHA.fullmatch(str(config["sft_adapter_sha256"])):
        raise ValueError("sft_adapter_sha256 must be a lowercase 64-character SHA256")
    paths = {key: Path(config[key]).expanduser().resolve(strict=False) for key in PATH_KEYS}
    waiting = [key for key, value in paths.items() if not value.exists()]
    if waiting:
        return {"status": "waiting", "missing": [f"{key}: {paths[key]}" for key in waiting]}
    for key in DIR_KEYS:
        if not paths[key].is_dir():
            raise ValueError(f"{key} must be a directory: {paths[key]}")
    for key in PATH_KEYS:
        if key not in DIR_KEYS and not paths[key].is_file():
            raise ValueError(f"{key} must be a file: {paths[key]}")
    baseline_files = [paths["baseline_run"] / "comparison" / name
                      for name in ("comparison_summary.json", "paired_items.csv")]
    if any(not path.is_file() for path in baseline_files):
        return {"status": "waiting", "missing": [str(path) for path in baseline_files if not path.is_file()]}
    gate_script = paths["candidate_eval"].with_name("orchestrate_insurance.py")
    if not gate_script.is_file():
        return {"status": "waiting", "missing": [str(gate_script)]}
    runtime_code = [
        paths["train_script"].with_name(name)
        for name in ("data.py", "reward.py", "v12_scorer.py", "run_reward_cases.py", "export_rollouts.py")
    ] + [paths["candidate_eval"].with_name(name) for name in ("ops_common.py", "merge_lora.py")]
    if any(not path.is_file() for path in runtime_code):
        return {"status": "waiting", "missing": [str(path) for path in runtime_code if not path.is_file()]}
    if BASE_REVISION not in str(paths["base_model"]):
        raise ValueError("base_model path does not contain the fixed revision")
    if paths["shared_merge_script"].name != "merge_lora.py" or paths["shared_merge_script"].parent.name != "team_share":
        raise ValueError("shared_merge_script must be A's team_share/merge_lora.py")
    if sha256(paths["shared_merge_script"]) != SHARED_MERGE_SHA256:
        raise ValueError("A's merge_lora.py changed; review and pin its new hash first")
    weights = paths["sft_adapter"] / "adapter_model.safetensors"
    adapter_config = paths["sft_adapter"] / "adapter_config.json"
    if not weights.is_file() or not adapter_config.is_file():
        return {"status": "waiting", "missing": [f"PEFT adapter files in {paths['sft_adapter']}"]}
    actual_sha = sha256(weights)
    if actual_sha != config["sft_adapter_sha256"]:
        raise ValueError(f"SFT adapter SHA mismatch: expected {config['sft_adapter_sha256']}, got {actual_sha}")
    _json(adapter_config)
    policy = _json(paths["reward_policy"])
    if policy.get("mode") != "v1_2":
        raise ValueError("Real-data GRPO requires reward_policy mode v1_2")
    approved_sha = sha256(paths["c_approved_ids"])
    clearance = _json(paths["c_clearance_json"])
    if clearance.get("status") != "passed" or clearance.get("approved_ids_sha256") != approved_sha:
        raise ValueError("C clearance does not approve this exact ID list")
    checks = clearance.get("checks", {})
    if not isinstance(checks, dict) or checks.get("MMMU") != "passed" or checks.get("MMMU-Pro") != "passed":
        raise ValueError("C clearance must record passed MMMU and MMMU-Pro leakage checks")
    if clearance.get("dataset_revision") != config["dataset_revision"]:
        raise ValueError("C clearance dataset_revision differs from config")
    work_root = Path(config["work_root"]).expanduser().resolve(strict=False)
    if any(work_root == p or work_root in p.parents or p in work_root.parents for p in
           (paths["base_model"], paths["sft_adapter"], paths["image_root"])):
        raise ValueError("work_root must be separate from model, adapter and image inputs")
    fingerprint_material = {
        "base_revision": BASE_REVISION,
        "base_path": str(paths["base_model"]),
        "shared_merge_sha256": SHARED_MERGE_SHA256,
        "merge_wrapper_sha256": sha256(paths["merge_wrapper"]),
        "converter_sha256": sha256(paths["converter"]),
        "train_script_sha256": sha256(paths["train_script"]),
        "candidate_eval_sha256": sha256(paths["candidate_eval"]),
        "compare_script_sha256": sha256(paths["compare_script"]),
        "baseline_gate_script_sha256": sha256(gate_script),
        "runtime_code_sha256": {path.name: sha256(path) for path in runtime_code},
        "baseline_summary_sha256": sha256(paths["baseline_run"] / "comparison/comparison_summary.json"),
        "baseline_paired_sha256": sha256(paths["baseline_run"] / "comparison/paired_items.csv"),
        "sft_adapter_sha256": actual_sha,
        "converted_sha256": sha256(paths["a_converted_jsonl"]),
        "approved_sha256": approved_sha,
        "clearance_sha256": sha256(paths["c_clearance_json"]),
        "reward_policy_sha256": sha256(paths["reward_policy"]),
        "dataset_revision": config["dataset_revision"],
        "max_steps": config["max_steps"],
        "min_train_rows": config["min_train_rows"],
        "eval_run_prefix": config["eval_run_prefix"],
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_material, sort_keys=True).encode()).hexdigest()
    return {
        "status": "ready", "fingerprint": fingerprint, "inputs": fingerprint_material,
        "work_root": str(work_root), "sft_parent": str(work_root / "sft_parent"),
        "dataset_manifest": str(work_root / "dataset_manifest.jsonl"),
        "train_dir": str(work_root / "grpo_train"),
        "final_model": str(work_root / "sft_grpo_final"),
        "smoke_dir": str(paths["eval_output_root"] / (config["eval_run_prefix"] + "_smoke")),
        "screen_dir": str(paths["eval_output_root"] / (config["eval_run_prefix"] + "_screen")),
    }


def _run(command: list[str]) -> None:
    print("RUN", json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def _write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def gpu_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Check the earlier synthetic queue and current GPU use before any real GPU work."""
    queue_path = config.get("wait_for_queue_state")
    if queue_path:
        state_path = Path(queue_path).expanduser().resolve(strict=False)
        if not state_path.is_file():
            return {"status": "waiting", "reason": f"earlier queue state missing: {state_path}"}
        stage = _json(state_path).get("stage")
        if stage not in TERMINAL_QUEUE_STAGES:
            return {"status": "waiting", "reason": f"earlier queue still at {stage}"}
    pilot_path = config.get("wait_for_pilot_manifest")
    if pilot_path:
        manifest_path = Path(pilot_path).expanduser().resolve(strict=False)
        if not manifest_path.is_file():
            return {"status": "waiting", "reason": f"v1.2 GPU pilot has not completed: {manifest_path}"}
        if _json(manifest_path).get("status") != "complete":
            return {"status": "waiting", "reason": f"v1.2 GPU pilot is incomplete: {manifest_path}"}
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = int(output.splitlines()[0].strip())
    if used > 4096:
        return {"status": "waiting", "reason": f"GPU uses {used} MiB"}
    return {"status": "ready", "gpu_memory_used_mib": used}


def baseline_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Reuse the existing 6000 paired baseline gate rather than a looser copy."""
    ops_dir = Path(config["candidate_eval"]).resolve(strict=True).parent
    module_path = ops_dir / "orchestrate_insurance.py"
    sys.path.insert(0, str(ops_dir))
    spec = importlib.util.spec_from_file_location("b_baseline_gate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline gate: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.proposed_baseline_gate(Path(config["baseline_run"]).resolve(strict=True))


def _final_merge_command(config: dict[str, Any], plan: dict[str, Any], grpo_sha: str,
                         *, check_only: bool) -> list[str]:
    command = [
        str(config["python"]), str(config["merge_wrapper"]),
        "--stage", "grpo", "--shared-script", str(config["shared_merge_script"]),
        "--base-model", str(plan["sft_parent"]),
        "--parent-manifest", str(Path(plan["sft_parent"]) / "model_manifest.json"),
        "--train-run-manifest", str(Path(plan["train_dir"]) / "run_manifest.json"),
        "--adapter", str(Path(plan["train_dir"]) / "adapter"),
        "--adapter-sha256", grpo_sha,
        "--out", str(plan["final_model"]),
    ]
    if check_only:
        command.append("--check-only")
    return command


def _candidate_command(config: dict[str, Any], plan: dict[str, Any], *, mode: str) -> list[str]:
    command = [
        str(config["python"]), str(config["candidate_eval"]),
        "--model-manifest", str(Path(plan["final_model"]) / "model_manifest.json"),
        "--train-run-manifest", str(Path(plan["train_dir"]) / "run_manifest.json"),
        "--repo-root", str(config["eval_repo"]),
        "--data-root", str(config["eval_data"]),
        "--eval-python", str(config["eval_python"]),
        "--output-root", str(config["eval_output_root"]),
        "--run-name", Path(plan[f"{mode}_dir"]).name,
        "--mode", mode,
    ]
    if mode == "screen":
        command += [
            "--baseline-run", str(config["baseline_run"]),
            "--compare-script", str(config["compare_script"]),
        ]
    return command


def _verify_eval_handoff(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    path = Path(plan[f"{mode}_dir"]) / "handoff.json"
    handoff = _json(path.resolve(strict=True))
    if (handoff.get("mode") != mode
            or handoff.get("model_path") != str(Path(plan["final_model"]))
            or set(handoff.get("result_files", {})) != {"3407"}):
        raise ValueError(f"{mode} evaluation handoff has unexpected model, seed, or mode")
    return handoff


def _verify_conversion(plan: dict[str, Any], config: dict[str, Any], converted: Path) -> None:
    conversion = _json(converted.with_suffix(converted.suffix + ".report.json"))
    if (conversion.get("source_sha256") != plan["inputs"]["converted_sha256"]
            or conversion.get("approved_sha256") != plan["inputs"]["approved_sha256"]
            or conversion.get("output_sha256") != sha256(converted)
            or conversion.get("rendered_parity_count") != conversion.get("converted_count")
            or conversion.get("structural_parity_count") != conversion.get("converted_count")
            or conversion.get("converted_count", 0) < config["min_train_rows"]):
        raise ValueError("A/C conversion report failed hashes, parity, or minimum train-row count")


def execute(config: dict[str, Any], plan: dict[str, Any], *, train: bool) -> dict[str, Any]:
    root = Path(plan["work_root"])
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    previous = _json(state_path) if state_path.exists() else None
    if previous is not None and previous.get("fingerprint") != plan["fingerprint"]:
        raise ValueError("work_root belongs to different inputs; choose a new work_root")
    if previous is not None and previous.get("status") in {"complete", "screen_blocked_baseline_mismatch"}:
        return previous
    if previous is not None and previous.get("status") in {
        "preparing", "training", "merging_final", "evaluating_smoke", "evaluating_screen", "failed"
    }:
        raise RuntimeError("Interrupted run needs manual inspection; never overwrite partial outputs")
    if previous is not None and previous.get("status") == "prepared" and not train:
        return previous
    python = str(config["python"])
    converted = Path(plan["dataset_manifest"])
    parent = Path(plan["sft_parent"])
    train_dir = Path(plan["train_dir"])
    if previous is None:
        if converted.exists() or parent.exists() or train_dir.exists():
            raise FileExistsError("work_root has untracked outputs; choose a new work_root")
        state = {"status": "preparing", "fingerprint": plan["fingerprint"], "inputs": plan["inputs"]}
        _write_state(state_path, state)
        try:
            _run([
                python, str(config["converter"]),
                "--converted-jsonl", str(config["a_converted_jsonl"]),
                "--approved-ids", str(config["c_approved_ids"]),
                "--image-root", str(config["image_root"]),
                "--dataset-revision", str(config["dataset_revision"]),
                "--output", str(converted),
                "--processor", str(config["base_model"]),
                "--render-limit", "0",
            ])
            _verify_conversion(plan, config, converted)
            state = {"status": "waiting_gpu_merge", "fingerprint": plan["fingerprint"],
                     "inputs": plan["inputs"], "dataset_manifest": str(converted)}
            _write_state(state_path, state)
        except Exception as exc:
            _write_state(state_path, {"status": "failed", "fingerprint": plan["fingerprint"],
                                       "error": f"{type(exc).__name__}: {exc}"})
            raise
    elif previous.get("status") == "waiting_gpu_merge":
        if not converted.is_file() or parent.exists() or train_dir.exists():
            raise RuntimeError("Waiting merge has missing conversion or unexpected GPU outputs")
        _verify_conversion(plan, config, converted)
        state = previous
    else:
        state = previous
        _run(_train_command(config, plan, validate_only=True))
    if state["status"] == "waiting_gpu_merge":
        gpu = gpu_gate(config)
        if gpu["status"] != "ready":
            state = {**state, "gpu_gate": gpu}
            _write_state(state_path, state)
            return {**state, "status": "waiting_gpu"}
        state = {**state, "status": "preparing"}
        state.pop("gpu_gate", None)
        _write_state(state_path, state)
        try:
            _run([
                python, str(config["merge_wrapper"]), "--stage", "sft",
                "--shared-script", str(config["shared_merge_script"]),
                "--base-model", str(config["base_model"]),
                "--adapter", str(config["sft_adapter"]),
                "--adapter-sha256", str(config["sft_adapter_sha256"]),
                "--out", str(parent), "--check-only",
            ])
            _run([
                python, str(config["merge_wrapper"]), "--stage", "sft",
                "--shared-script", str(config["shared_merge_script"]),
                "--base-model", str(config["base_model"]),
                "--adapter", str(config["sft_adapter"]),
                "--adapter-sha256", str(config["sft_adapter_sha256"]),
                "--out", str(parent),
            ])
            _run(_train_command(config, plan, validate_only=True))
            state = {"status": "prepared", "fingerprint": plan["fingerprint"], "inputs": plan["inputs"],
                     "dataset_manifest": str(converted), "sft_parent": str(parent)}
            _write_state(state_path, state)
        except Exception as exc:
            _write_state(state_path, {"status": "failed", "fingerprint": plan["fingerprint"],
                                      "error": f"{type(exc).__name__}: {exc}"})
            raise
    if not train:
        return state
    gpu = gpu_gate(config)
    if gpu["status"] != "ready":
        return {**state, "status": "waiting_gpu", "gpu_gate": gpu}
    if train_dir.exists():
        raise FileExistsError(f"Training output already exists: {train_dir}")
    state = {"status": "training", "fingerprint": plan["fingerprint"], "inputs": plan["inputs"],
             "dataset_manifest": str(converted), "sft_parent": str(parent)}
    _write_state(state_path, state)
    try:
        _run(_train_command(config, plan, validate_only=False))
        train_manifest = _json(train_dir / "run_manifest.json")
        if train_manifest.get("status") != "complete" or train_manifest.get("parent_adapter_sha256") != config["sft_adapter_sha256"]:
            raise ValueError("GRPO training manifest failed completion/lineage verification")
        grpo_sha = train_manifest.get("adapter_sha256")
        if not isinstance(grpo_sha, str) or not HEX_SHA.fullmatch(grpo_sha):
            raise ValueError("GRPO adapter SHA256 missing from completed training manifest")
        adapter = Path(train_manifest.get("adapter_dir", "")).resolve(strict=True)
        if adapter != train_dir / "adapter" or sha256(adapter / "adapter_model.safetensors") != grpo_sha:
            raise ValueError("GRPO adapter output or SHA256 does not match training manifest")
        state = {**state, "status": "merging_final", "train_dir": str(train_dir),
                 "grpo_adapter_sha256": grpo_sha}
        _write_state(state_path, state)
        _run(_final_merge_command(config, plan, grpo_sha, check_only=True))
        _run(_final_merge_command(config, plan, grpo_sha, check_only=False))
        final_manifest = _json(Path(plan["final_model"]) / "model_manifest.json")
        adapters = final_manifest.get("adapters", [])
        if (final_manifest.get("status") != "complete"
                or [item.get("role") for item in adapters] != ["sft", "grpo"]
                or adapters[0].get("weights_sha256") != config["sft_adapter_sha256"]
                or adapters[1].get("weights_sha256") != grpo_sha):
            raise ValueError("Final model manifest does not preserve SFT→GRPO lineage")
        state = {**state, "status": "evaluating_smoke", "final_model": plan["final_model"]}
        _write_state(state_path, state)
        _run(_candidate_command(config, plan, mode="smoke"))
        _verify_eval_handoff(plan, "smoke")
        gate = baseline_gate(config)
        state = {**state, "baseline_gate": gate, "smoke_dir": plan["smoke_dir"]}
        if not gate.get("screen_allowed"):
            state["status"] = "screen_blocked_baseline_mismatch"
            _write_state(state_path, state)
            return state
        state["status"] = "evaluating_screen"
        _write_state(state_path, state)
        _run(_candidate_command(config, plan, mode="screen"))
        _verify_eval_handoff(plan, "screen")
        state = {**state, "status": "complete", "screen_dir": plan["screen_dir"]}
        _write_state(state_path, state)
    except Exception as exc:
        _write_state(state_path, {**state, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return state


def _train_command(config: dict[str, Any], plan: dict[str, Any], *, validate_only: bool) -> list[str]:
    command = [
        str(config["python"]), str(config["train_script"]),
        "--dataset-manifest", str(plan["dataset_manifest"]),
        "--reward-policy", str(config["reward_policy"]),
        "--parent-kind", "merged_sft", "--parent-model", str(plan["sft_parent"]),
        "--parent-manifest", str(Path(plan["sft_parent"]) / "model_manifest.json"),
        "--expected-sft-adapter-sha256", str(config["sft_adapter_sha256"]),
        "--output-dir", str(plan["train_dir"]),
        "--max-steps", str(config["max_steps"]),
    ]
    if "learning_rate" in config:
        command += ["--learning-rate", str(config["learning_rate"])]
    if "max_completion_length" in config:
        command += ["--max-completion-length", str(config["max_completion_length"])]
    if validate_only:
        command.append("--validate-only")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("check", "prepare", "train", "wait"), default="check")
    parser.add_argument("--auto-train", action="store_true", help="With wait, start GPU training once all gates pass")
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means wait indefinitely")
    args = parser.parse_args()
    if args.auto_train and args.mode != "wait":
        parser.error("--auto-train is only valid with --mode wait")
    if args.poll_seconds < 0 or args.max_polls < 0:
        parser.error("poll-seconds and max-polls must be non-negative")
    poll = 0
    while True:
        config = _json(args.config.resolve(strict=True))
        plan = _status(config)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if plan["status"] == "ready" and args.mode != "check":
            result = execute(config, plan, train=(args.mode == "train" or args.auto_train))
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] != "waiting_gpu":
                return 0
            if args.mode != "wait":
                return 2
        if args.mode != "wait":
            return 0 if plan["status"] == "ready" else 2
        poll += 1
        if args.max_polls and poll >= args.max_polls:
            return 0 if plan["status"] == "ready" else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
