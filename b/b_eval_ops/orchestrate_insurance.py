"""Queue B's exploratory synthetic base→GRPO branch after 6000 baseline.

This is an execution queue, not a claim of MMMU improvement. It waits for the
three-seed 6000 base run and paired 4090 comparison, trains on the fixed
synthetic non-MMMU set, CPU-merges its adapter, runs a 20-item smoke, and runs
one 900-item screen only if the proposed 6000 reproducibility gate passes.
It deliberately never launches a three-seed candidate evaluation.

Do not start this queue until A/C availability has been checked by the team.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ops_common import BASE_REVISION, read_json, safe_name, sha256_file, write_json


DEFAULT = {
    "train_python": "/data/chaewon/mmdl_envs/grpo_pilot/bin/python",
    "eval_python": "/data/chaewon/mmdl_envs/vllm011_repro/bin/python",
    "train_script": "/data/chaewon/mmdl_b_main_code/train.py",
    "ops_dir": "/data/chaewon/mmdl_b_eval_ops",
    "base_model": "/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "eval_repo": "/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2",
    "eval_data": "/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
    "baseline": "/data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117",
    "train_root": "/data/chaewon/mmdl_grpo_runs",
    "merged_root": "/data/chaewon/mmdl_merged_models",
    "eval_root": "/data/chaewon/mmdl_eval_runs",
    "queue_root": "/data/chaewon/mmdl_b_eval_ops/orchestrations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="Fresh unique base-insurance run name")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--reward-policy", required=True, type=Path)
    parser.add_argument("--reward-policy-sha256", required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--max-wait-hours", type=float, default=12.0)
    parser.add_argument("--check-only", action="store_true", help="Validate paths/hashes and print pending baseline status; no GPU or writes")
    for name, default in DEFAULT.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(default))
    return parser.parse_args()


def assert_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path.resolve(strict=True))
    if len(expected) != 64 or actual != expected.lower():
        raise ValueError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")
    return actual


def paths(args: argparse.Namespace) -> dict[str, Path]:
    name = safe_name(args.run_name)
    result = {
        "queue": args.queue_root / name,
        "train": args.train_root / name,
        "merged": args.merged_root / name,
        "smoke": args.eval_root / (name + "_smoke"),
        "screen": args.eval_root / (name + "_screen"),
    }
    for key, path in result.items():
        if path.exists():
            raise FileExistsError(f"Fresh {key} output required: {path}")
    return result


def baseline_ready(baseline: Path) -> tuple[bool, str]:
    for seed in ("3407", "1", "2"):
        meta = baseline / f"seed_{seed}" / "run_meta.txt"
        if not meta.is_file():
            return False, f"seed {seed} has no run_meta.txt"
        content = meta.read_text(encoding="utf-8")
        if "exit_status: " not in content:
            return False, f"seed {seed} is still running"
        if "exit_status: 0" not in content:
            raise RuntimeError(f"Baseline seed {seed} failed: {meta}")
    summary = baseline / "comparison" / "comparison_summary.json"
    paired = baseline / "comparison" / "paired_items.csv"
    if not summary.is_file() or not paired.is_file():
        return False, "paired 4090 comparison is pending"
    return True, "three seeds and paired comparison completed"


def proposed_baseline_gate(baseline: Path) -> dict[str, Any]:
    """Conservative implementation of the team's *proposed* operational rule."""
    summary = read_json(baseline / "comparison" / "comparison_summary.json")
    per_seed = summary.get("per_seed")
    overall = summary.get("overall")
    reasons: list[str] = []
    if not isinstance(per_seed, dict) or set(per_seed) != {"3407", "1", "2"}:
        reasons.append("three paired seeds are not present")
        per_seed = {}
    if not isinstance(overall, dict):
        reasons.append("three-seed overall comparison is missing")
        overall = {}
    mean_difference = overall.get("mean_difference_pp")
    if not isinstance(mean_difference, (int, float)) or abs(mean_difference) > 0.5:
        reasons.append(f"mean score difference exceeds ±0.5%p: {mean_difference}")
    for seed in ("3407", "1", "2"):
        row = per_seed.get(seed, {})
        if row.get("items_compared") != 900 or not row.get("full_900_coverage"):
            reasons.append(f"seed {seed} does not cover identical 900 items")
        agreement = row.get("binary_correctness_agreement_pct")
        if not isinstance(agreement, (int, float)) or agreement < 85.0:
            reasons.append(f"seed {seed} binary agreement below 85%: {agreement}")
    missing_by_seed: dict[str, dict[str, set[str]]] = {
        seed: {"baseline": set(), "candidate": set()} for seed in ("3407", "1", "2")
    }
    raw_empty_discordance = 0
    with (baseline / "comparison" / "paired_items.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            seed = row["seed"]
            if seed not in missing_by_seed:
                reasons.append(f"unexpected paired seed {seed}")
                continue
            if row["baseline_final_answer_missing"] == "True":
                missing_by_seed[seed]["baseline"].add(row["question_id"])
            if row["candidate_final_answer_missing"] == "True":
                missing_by_seed[seed]["candidate"].add(row["question_id"])
            if row["baseline_raw_empty"] != row["candidate_raw_empty"]:
                raw_empty_discordance += 1
    missing_stats: dict[str, dict[str, int]] = {}
    for seed, groups in missing_by_seed.items():
        base_missing = groups["baseline"]
        candidate_missing = groups["candidate"]
        count_difference = abs(len(base_missing) - len(candidate_missing))
        symmetric_difference = len(base_missing ^ candidate_missing)
        missing_stats[seed] = {
            "baseline_count": len(base_missing),
            "candidate_count": len(candidate_missing),
            "count_difference": count_difference,
            "symmetric_difference": symmetric_difference,
        }
        if count_difference > 20:
            reasons.append(f"seed {seed} missing-final count difference exceeds 20: {count_difference}")
        if symmetric_difference > 100:
            reasons.append(f"seed {seed} missing-final symmetric difference exceeds 100: {symmetric_difference}")

    def input_hashes(run: Path, seed: str) -> dict[str, tuple[str, str]]:
        matches = list((run / f"seed_{seed}").glob("*/*_samples_*.jsonl"))
        if len(matches) != 1:
            raise ValueError(f"Expected one samples JSONL for input hashes: {run}, seed {seed}")
        values = {}
        with matches[0].open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                question_id = item["mmmu_acc"]["id"]
                if question_id in values:
                    raise ValueError(f"Duplicate question ID in input hash check: {question_id}")
                doc_hash = item.get("doc_hash")
                prompt = item.get("input")
                if not isinstance(doc_hash, str) or not isinstance(prompt, str):
                    raise ValueError(f"Missing doc_hash/input for {question_id}")
                values[question_id] = (doc_hash, hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        return values

    base_run = Path(summary.get("baseline_run", "")).resolve(strict=True)
    candidate_run = Path(summary.get("candidate_run", "")).resolve(strict=True)
    if candidate_run != baseline.resolve(strict=True):
        reasons.append("paired comparison candidate path differs from the 6000 baseline")
    input_discordance: dict[str, int] = {}
    for seed in ("3407", "1", "2"):
        old = input_hashes(base_run, seed)
        new = input_hashes(candidate_run, seed)
        mismatches = sum(old.get(item) != value for item, value in new.items())
        mismatches += len(set(old) - set(new))
        input_discordance[seed] = mismatches
        if len(old) != 900 or len(new) != 900 or mismatches:
            reasons.append(f"seed {seed} input/doc hashes differ: {mismatches} mismatches")
    return {
        "screen_allowed": not reasons,
        "reasons": reasons,
        "mean_difference_pp": mean_difference,
        "per_seed_binary_agreement_pct": {
            seed: per_seed.get(seed, {}).get("binary_correctness_agreement_pct")
            for seed in ("3407", "1", "2")
        },
        "missing_final_answer_by_seed": missing_stats,
        "raw_empty_indicator_discordance": raw_empty_discordance,
        "input_doc_hash_discordance_by_seed": input_discordance,
        "threshold_note": "Operational safeguards only: mean ±0.5%p; each seed ≥85% binary agreement; missing count Δ≤20 and symmetric difference≤100; exact 900 input/doc hashes.",
        "finish_reason_note": "Saved JSONL may lack finish_reason; truncation cannot be inferred automatically.",
    }


def run_step(label: str, command: list[str], state: dict[str, Any], path: Path, env: dict[str, str] | None = None) -> None:
    print(f"=== {label}: {json.dumps(command)}", flush=True)
    state["stage"] = label
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)
    subprocess.run(command, env=env, check=True)


def wait_gpu_idle(max_wait_seconds: int = 7200) -> None:
    """Do not overlap GRPO with another evaluation or training process."""
    deadline = time.monotonic() + max_wait_seconds
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used_mib = int(output.splitlines()[0].strip())
        if used_mib <= 4096:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU remained busy ({used_mib} MiB) after baseline completion")
        print(f"Waiting for idle GPU before GRPO: {used_mib} MiB used", flush=True)
        time.sleep(30)


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.max_wait_hours <= 0:
        raise ValueError("max-steps and max-wait-hours must be positive")
    synthetic_manifest = args.dataset_manifest.resolve(strict=True)
    policy = args.reward_policy.resolve(strict=True)
    dataset_sha = assert_sha(synthetic_manifest, args.dataset_sha256, "synthetic dataset")
    policy_sha = assert_sha(policy, args.reward_policy_sha256, "reward policy")
    if "synthetic" not in str(synthetic_manifest).lower():
        raise ValueError("This insurance queue is only for the explicitly labelled synthetic dataset")
    if "synthetic" not in str(policy).lower():
        raise ValueError("This insurance queue requires the synthetic strict reward policy")
    for required in (args.train_python, args.eval_python, args.train_script,
                     args.ops_dir / "merge_lora.py", args.ops_dir / "candidate_eval.py",
                     args.ops_dir / "ops_common.py", args.ops_dir.parent / "mmdl_b_compare" / "compare_runs.py",
                     args.base_model / "config.json", args.eval_repo / "scripts" / "run_mmmu_eval.sh"):
        if not required.is_file():
            raise FileNotFoundError(required)
    target = paths(args)
    ready, reason = baseline_ready(args.baseline)
    plan = {
        "purpose": "exploratory synthetic base→GRPO transfer screen; not common-parent result",
        "run_name": args.run_name,
        "dataset_manifest": str(synthetic_manifest),
        "dataset_sha256": dataset_sha,
        "reward_policy": str(policy),
        "reward_policy_sha256": policy_sha,
        "max_steps": args.max_steps,
        "outputs": {key: str(value) for key, value in target.items()},
        "baseline_ready": ready,
        "baseline_status": reason,
        "stage": "preflight",
    }
    if args.check_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    target["queue"].mkdir(parents=True, exist_ok=False)
    state_path = target["queue"] / "state.json"
    write_json(state_path, plan)
    deadline = time.monotonic() + args.max_wait_hours * 3600
    while not ready:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Baseline did not finish within {args.max_wait_hours} hours: {reason}")
        print(f"Waiting for baseline: {reason}", flush=True)
        time.sleep(60)
        ready, reason = baseline_ready(args.baseline)
    print("Baseline comparison ready", flush=True)
    gate = proposed_baseline_gate(args.baseline)
    plan["baseline_gate"] = gate
    write_json(state_path, plan)
    wait_gpu_idle()

    train_command = [
        str(args.train_python), str(args.train_script),
        "--dataset-manifest", str(synthetic_manifest),
        "--reward-policy", str(policy),
        "--parent-kind", "base",
        "--parent-model", str(args.base_model),
        "--output-dir", str(target["train"]),
        "--max-steps", str(args.max_steps),
    ]
    run_step("train_synthetic_base_grpo", train_command, plan, state_path)
    train_manifest = target["train"] / "run_manifest.json"
    trained = read_json(train_manifest)
    adapter = Path(trained.get("adapter_dir", ""))
    adapter_sha = trained.get("adapter_sha256")
    if (trained.get("status") != "complete" or trained.get("base_revision") != BASE_REVISION
            or trained.get("parent_kind") != "base" or not trained.get("reload_success")
            or trained.get("mixed_reward_groups", 0) < 1 or trained.get("lora_delta_l1", 0) <= 0
            or trained.get("dataset", {}).get("manifest_sha256") != dataset_sha
            or trained.get("reward_policy_sha256") != policy_sha
            or not isinstance(adapter_sha, str) or len(adapter_sha) != 64
            or not adapter.is_dir()):
        raise RuntimeError(f"GRPO training manifest does not prove an updated, reloadable adapter: {train_manifest}")
    assert_sha(adapter / "adapter_model.safetensors", adapter_sha, "trained GRPO adapter")
    assert_sha(target["train"] / "rollouts.jsonl", trained["rollouts_sha256"], "rollout log")

    merge_command = [
        str(args.train_python), str(args.ops_dir / "merge_lora.py"),
        "--base-model", str(args.base_model),
        "--adapter", str(adapter), "--adapter-role", "grpo", "--adapter-sha256", adapter_sha,
        "--output-dir", str(target["merged"]),
    ]
    merge_env = os.environ.copy()
    merge_env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    run_step("merge_synthetic_grpo_on_cpu", merge_command, plan, state_path, env=merge_env)
    merged_manifest = target["merged"] / "model_manifest.json"
    merged = read_json(merged_manifest)
    if (merged.get("status") != "complete" or merged.get("adapter_sha256") != adapter_sha
            or merged.get("parent_kind") != "merged_base_grpo"):
        raise RuntimeError(f"Merged model lineage is wrong: {merged_manifest}")

    eval_common = [
        str(args.train_python), str(args.ops_dir / "candidate_eval.py"),
        "--model-manifest", str(merged_manifest),
        "--train-run-manifest", str(train_manifest),
        "--repo-root", str(args.eval_repo),
        "--data-root", str(args.eval_data),
        "--eval-python", str(args.eval_python),
        "--output-root", str(args.eval_root),
    ]
    smoke_command = [*eval_common, "--run-name", target["smoke"].name, "--mode", "smoke"]
    run_step("smoke_20", smoke_command, plan, state_path)
    if not (target["smoke"] / "handoff.json").is_file():
        raise RuntimeError("20-item smoke did not write handoff.json")
    if not gate["screen_allowed"]:
        plan["stage"] = "screen_blocked_baseline_mismatch"
        plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, plan)
        print("Baseline failed proposed 6000 reproduction gate; 900-item screen not launched", flush=True)
        return

    screen_command = [
        *eval_common, "--run-name", target["screen"].name, "--mode", "screen",
        "--baseline-run", str(args.baseline),
        "--compare-script", str(args.ops_dir.parent / "mmdl_b_compare" / "compare_runs.py"),
    ]
    run_step("exploratory_screen_900_seed_3407", screen_command, plan, state_path)
    screen_handoff = target["screen"] / "handoff.json"
    if not screen_handoff.is_file():
        raise RuntimeError("900-item screen did not write handoff.json")
    plan["stage"] = "complete_exploratory_one_seed"
    plan["screen_handoff"] = str(screen_handoff)
    plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, plan)
    print(f"Exploratory one-seed screen complete: {screen_handoff}", flush=True)


if __name__ == "__main__":
    main()
