"""Run the unchanged MMDL v1.2 evaluator on a verified merged candidate.

Modes: smoke (20 items, one seed), screen (900 items, one seed), full (900
items, three seeds). A baseline run from the *same GPU* can be passed for paired
comparison; only a full 4090 run is an official final result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ops_common import (
    BASE_REVISION,
    EVAL_REPO_COMMIT,
    read_json,
    safe_name,
    sha256_file,
    write_json,
)


MODE = {
    "smoke": {"seeds": ("3407",), "limit": 20},
    "screen": {"seeds": ("3407",), "limit": None},
    "full": {"seeds": ("3407", "1", "2"), "limit": None},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--train-run-manifest", type=Path, help="GRPO run_manifest.json, if this is a GRPO candidate")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--eval-python", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--screen-seed", choices=("3407", "1", "2"),
                        help="Seed for a one-seed screen (default: 3407)")
    parser.add_argument("--baseline-run", type=Path, help="Own-GPU base or parent run for paired comparison")
    parser.add_argument("--compare-script", type=Path, help="compare_runs.py; required with --baseline-run")
    parser.add_argument("--verify-model-hashes", action="store_true", help="Rehash all saved model files before launch")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Explicitly override >4 GiB GPU-use guard")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print command without launching evaluator")
    return parser.parse_args()


def mode_settings(mode: str, screen_seed: str | None) -> dict[str, Any]:
    if screen_seed is not None and mode != "screen":
        raise ValueError("--screen-seed requires --mode screen")
    if mode == "screen":
        return {**MODE[mode], "seeds": (screen_seed or "3407",)}
    return MODE[mode]


def git_check(repo: Path) -> str:
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != EVAL_REPO_COMMIT:
        raise ValueError(f"Evaluation repo must be pinned at {EVAL_REPO_COMMIT}, got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError(f"Evaluation worktree has uncommitted files: {dirty[:500]}")
    script = repo / "scripts" / "run_mmmu_eval.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    return sha256_file(script)


def validate_model(path: Path, verify_hashes: bool) -> tuple[Path, dict[str, Any]]:
    manifest = read_json(path.resolve(strict=True))
    model = Path(manifest.get("model_path", "")).resolve(strict=True)
    if manifest.get("status") != "complete" or (model / "_INCOMPLETE").exists():
        raise ValueError("Merged model is incomplete")
    if path.parent.resolve(strict=True) != model:
        raise ValueError("model_manifest.json must reside inside its merged model directory")
    if manifest.get("base_revision") != BASE_REVISION:
        raise ValueError("Merged model has wrong base revision")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Merged model manifest lacks weight files")
    for name, record in files.items():
        candidate = model / name
        if not candidate.is_file() or candidate.stat().st_size != record["bytes"]:
            raise ValueError(f"Merged model file missing/changed: {candidate}")
        if verify_hashes and sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"Merged model SHA256 mismatch: {candidate}")
    return model, manifest


def validate_training_lineage(model_manifest: dict[str, Any], train_path: Path | None) -> dict[str, Any] | None:
    if train_path is None:
        return None
    train = read_json(train_path.resolve(strict=True))
    if train.get("base_revision") != BASE_REVISION:
        raise ValueError("Training manifest has wrong base revision")
    if train.get("parent_kind") not in {"base", "merged_sft"}:
        raise ValueError("GRPO parent_kind must be base or merged_sft")
    adapters = model_manifest.get("adapters")
    if not isinstance(adapters, list):
        raise ValueError("Merged model manifest lacks ordered adapters")
    expected_roles = ["grpo"] if train["parent_kind"] == "base" else ["sft", "grpo"]
    if [a.get("role") for a in adapters] != expected_roles:
        raise ValueError(f"Merged adapter order differs from GRPO parent: expected {expected_roles}")
    if adapters[-1].get("weights_sha256") != train.get("adapter_sha256"):
        raise ValueError("Merged GRPO adapter SHA256 differs from training manifest")
    if train["parent_kind"] == "merged_sft":
        if adapters[0].get("weights_sha256") != train.get("parent_adapter_sha256"):
            raise ValueError("Merged SFT parent SHA256 differs from training manifest")
        parent_path = Path(train.get("parent_model_path", ""))
        parent_manifest = parent_path / "model_manifest.json"
        expected_parent_hash = train.get("parent_model_manifest_sha256")
        if parent_manifest.is_file() and expected_parent_hash:
            if sha256_file(parent_manifest) != expected_parent_hash:
                raise ValueError("SFT parent model manifest changed after GRPO training")
    return train


def check_eval_python(python: Path) -> Path:
    resolved = python.resolve(strict=True)
    code = (
        "from importlib.metadata import version; "
        "print(version('vllm')+' '+version('transformers'))"
    )
    result = subprocess.check_output([str(resolved), "-c", code], text=True).strip().split()
    if result != ["0.11.0", "4.57.1"]:
        raise ValueError(f"Expected evaluation vLLM 0.11.0 / transformers 4.57.1, got {result}")
    return resolved


def check_data_root(root: Path) -> Path:
    resolved = root.resolve(strict=True)
    parquets = list(resolved.glob("*/validation-*.parquet"))
    if len(parquets) != 30:
        raise ValueError(f"Expected 30 MMMU validation Parquets, found {len(parquets)}")
    return resolved


def check_gpu_idle() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = int(output.splitlines()[0].strip())
    if used > 4096:
        raise RuntimeError(f"GPU already uses {used} MiB; finish evaluation/training before starting another run")


def check_own_gpu_baseline(baseline: Path, seeds: tuple[str, ...]) -> None:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()[0].strip()
    for seed in seeds:
        seed_dir = baseline / f"seed_{seed}"
        meta_path = seed_dir / "run_meta.txt"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Own-GPU baseline seed missing: {meta_path}")
        meta = meta_path.read_text(encoding="utf-8")
        if "exit_status: 0" not in meta:
            raise ValueError(f"Own-GPU baseline seed {seed} did not complete")
        if f"gpu: {gpu_name}," not in meta:
            raise ValueError(f"Baseline seed {seed} was not measured on the current GPU: {gpu_name}")
        if len(list(seed_dir.glob("*/*_samples_*.jsonl"))) != 1:
            raise ValueError(f"Baseline seed {seed} needs one samples JSONL")


def main() -> None:
    args = parse_args()
    name = safe_name(args.run_name)
    repo = args.repo_root.resolve(strict=True)
    repo_script_sha = git_check(repo)
    model, model_manifest = validate_model(args.model_manifest, args.verify_model_hashes)
    train = validate_training_lineage(model_manifest, args.train_run_manifest)
    data = check_data_root(args.data_root)
    python = check_eval_python(args.eval_python)
    output_root = args.output_root.resolve(strict=True)
    run_dir = output_root / name
    if run_dir.exists():
        raise FileExistsError(f"Use a fresh run name: {run_dir}")
    baseline = args.baseline_run.resolve(strict=True) if args.baseline_run else None
    compare_script = args.compare_script.resolve(strict=True) if args.compare_script else None
    if bool(baseline) != bool(compare_script):
        raise ValueError("--baseline-run and --compare-script must be passed together")
    if args.mode in {"screen", "full"} and baseline is None:
        raise ValueError("Screen/full evaluation requires an own-GPU baseline for paired Δ")
    settings = mode_settings(args.mode, args.screen_seed)
    if baseline is not None:
        check_own_gpu_baseline(baseline, settings["seeds"])
    command = [
        "bash", str(repo / "scripts" / "run_mmmu_eval.sh"),
        "--model_path", str(model),
        "--data_root", str(data),
        "--seeds", " ".join(settings["seeds"]),
        "--output_path", str(output_root),
        "--run_name", name,
    ]
    if settings["limit"]:
        command += ["--limit", str(settings["limit"])]
    if not args.dry_run and not args.allow_busy_gpu:
        check_gpu_idle()
    request = {
        "schema_version": 1,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": name,
        "mode": args.mode,
        "seeds": settings["seeds"],
        "limit": settings["limit"],
        "evaluation_repo_commit": EVAL_REPO_COMMIT,
        "evaluation_script_sha256": repo_script_sha,
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "model_path": str(model),
        "model_lineage_sha256": model_manifest["lineage"]["lineage_sha256"],
        "training_manifest_sha256": sha256_file(args.train_run_manifest) if train else None,
        "data_root": str(data),
        "eval_python": str(python),
        "baseline_run": str(baseline) if baseline else None,
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "candidate_manifest.json", request)
    env = os.environ.copy()
    env["PYTHON"] = str(python)
    subprocess.run(command, env=env, check=True)

    result_files: dict[str, dict[str, Any]] = {}
    for seed in settings["seeds"]:
        seed_dir = run_dir / f"seed_{seed}"
        meta_path = seed_dir / "run_meta.txt"
        if not meta_path.is_file() or "exit_status: 0" not in meta_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Seed {seed} did not finish cleanly")
        samples = list(seed_dir.glob("*/*_samples_*.jsonl"))
        results = list(seed_dir.glob("*/*_results.json"))
        if len(samples) != 1 or len(results) != 1:
            raise RuntimeError(f"Seed {seed} needs one samples and one results file")
        result_files[seed] = {
            "samples": {"path": str(samples[0]), "sha256": sha256_file(samples[0])},
            "results": {"path": str(results[0]), "sha256": sha256_file(results[0])},
            "run_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        }
    comparison = None
    if baseline is not None:
        command = [
            str(python), str(compare_script),
            "--repo-root", str(repo),
            "--baseline-run", str(baseline),
            "--candidate-run", str(run_dir),
            "--output-dir", str(run_dir / "comparison"),
        ]
        if args.mode == "smoke":
            command.append("--allow-partial")
        subprocess.run(command, check=True)
        comparison = str(run_dir / "comparison" / "comparison_summary.json")
    handoff = {
        **request,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_files": result_files,
        "comparison_summary": comparison,
        "note": "6000 screen/full values are screening results; final official result requires 4090 v1.2 3 seeds.",
    }
    write_json(run_dir / "handoff.json", handoff)
    print(json.dumps({"run_dir": str(run_dir), "handoff": str(run_dir / "handoff.json")}, indent=2))


if __name__ == "__main__":
    main()
