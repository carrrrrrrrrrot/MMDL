"""Run a fixed-ID, 300-item v1.2 diagnostic MMMU screen on the 6000.

The frozen official 900-item task is unchanged. This invokes the repository's
existing diagnostic task with only MMDL_DIAG_IDS_FILE set and keeps the same
v1.2 prompt, image processing, generation settings, and scorer. Outputs are
screening artifacts, never an official 900-item result.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))
from candidate_eval import (  # noqa: E402
    check_data_root, check_eval_python, check_gpu_idle, check_own_gpu_baseline,
    git_check, validate_model, validate_training_lineage,
)
from ops_common import fixed_base_path, read_json, safe_name, sha256_file, write_json  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-model", type=Path)
    source.add_argument("--model-manifest", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, help="Completed fixed300 base run; omit for base itself")
    parser.add_argument("--compare-script", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--verify-model-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_ids(path: Path) -> tuple[Path, str]:
    path = path.resolve(strict=True)
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError(f"Fixed diagnostic list needs 300 unique IDs, got {len(ids)}")
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest.is_file() or read_json(manifest).get("ids_sha256") != sha256_file(path):
        raise ValueError("Fixed300 ID selection manifest is missing or does not match")
    return path, sha256_file(path)


def verify_finished(run_dir: Path, ids_sha: str, request: dict) -> bool:
    handoff_file = run_dir / "handoff.json"
    if not handoff_file.is_file():
        return False
    handoff = read_json(handoff_file)
    for key in ("run_name", "model_path", "model_manifest_sha256", "training_manifest_sha256", "ids_sha256"):
        if handoff.get(key) != request.get(key):
            raise ValueError(f"Existing run {run_dir} has a different {key}")
    if handoff.get("status") != "complete" or handoff.get("ids_sha256") != ids_sha:
        raise ValueError(f"Existing run is incomplete: {run_dir}")
    for item in handoff.get("result_files", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Existing result changed: {path}")
    return True


def run_with_timeout(command: list[str], env: dict[str, str], log: Path, timeout: int) -> None:
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        print(f"Evaluation PID={process.pid}; live log: {log}", flush=True)
        while True:
            status = process.poll()
            if status is not None:
                if status != 0:
                    raise RuntimeError(f"Evaluation exited {status}; inspect {log}")
                return
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(f"Diagnostic evaluation exceeded {timeout}s; inspect {log}")
            time.sleep(5)


def main() -> None:
    args = args_parse()
    if args.timeout_seconds < 60:
        raise ValueError("timeout-seconds must be at least 60")
    run_name = safe_name(args.run_name)
    repo = args.repo_root.resolve(strict=True)
    eval_script_sha = git_check(repo)
    data = check_data_root(args.data_root)
    python = check_eval_python(args.eval_python)
    ids_file, ids_sha = validate_ids(args.ids_file)
    output_root = args.output_root.resolve(strict=True)
    run_dir = output_root / run_name

    if args.base_model:
        if args.train_manifest or args.baseline_run or args.compare_script:
            raise ValueError("Base fixed300 run needs only --base-model, not training/comparison flags")
        model = fixed_base_path(args.base_model)
        model_manifest_sha = None
        train_manifest_sha = None
        baseline = None
        compare_script = None
    else:
        if not args.train_manifest or not args.baseline_run or not args.compare_script:
            raise ValueError("Candidate needs train manifest, fixed300 base run, and compare script")
        model, model_manifest = validate_model(args.model_manifest, args.verify_model_hashes)
        validate_training_lineage(model_manifest, args.train_manifest)
        model_manifest_sha = sha256_file(args.model_manifest)
        train_manifest_sha = sha256_file(args.train_manifest)
        baseline = args.baseline_run.resolve(strict=True)
        compare_script = args.compare_script.resolve(strict=True)
        baseline_handoff = read_json(baseline / "handoff.json")
        if baseline_handoff.get("status") != "complete" or baseline_handoff.get("ids_sha256") != ids_sha:
            raise ValueError("Fixed300 base run is absent or uses different IDs")
        if baseline_handoff.get("task") != "mmmu_val_mmdl_diag":
            raise ValueError("Fixed300 base run used a different task")
        check_own_gpu_baseline(baseline, ("3407",))

    command = [
        "bash", str(repo / "scripts" / "run_mmmu_eval.sh"),
        "--model_path", str(model), "--data_root", str(data),
        "--task", "mmmu_val_mmdl_diag", "--seeds", "3407",
        "--output_path", str(output_root), "--run_name", run_name,
    ]
    request = {
        "status": "requested", "requested_at_utc": now(),
        "run_name": run_name, "task": "mmmu_val_mmdl_diag", "seed": "3407",
        "items": 300, "ids_file": str(ids_file), "ids_sha256": ids_sha,
        "model_path": str(model), "model_manifest_sha256": model_manifest_sha,
        "training_manifest_sha256": train_manifest_sha,
        "evaluation_repo_commit": "61025850563d0331e365bc974a63c0d3c4ed4c2c",
        "evaluation_script_sha256": eval_script_sha,
        "baseline_run": str(baseline) if baseline else None,
        "command": command, "timeout_seconds": args.timeout_seconds,
        "note": "Fixed-ID 300-item diagnostic screen only; no official MMMU score",
    }
    if args.dry_run:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    if run_dir.exists():
        if verify_finished(run_dir, ids_sha, request):
            print(f"SKIP completed and verified: {run_dir}")
            return
        raise FileExistsError(f"Incomplete run exists; inspect and use a fresh run name: {run_dir}")
    check_gpu_idle()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "pilot_request.json", request)
    environment = os.environ.copy()
    environment["PYTHON"] = str(python)
    environment["MMDL_DIAG_IDS_FILE"] = str(ids_file)
    environment["MMDL_DIAG_MIN_PIXELS"] = "4096"
    environment.pop("MMDL_DIAG_DROP_IMAGES", None)
    try:
        run_with_timeout(command, environment, run_dir / "launch.log", args.timeout_seconds)
        seed_dir = run_dir / "seed_3407"
        check_own_gpu_baseline(run_dir, ("3407",))
        samples = sorted(seed_dir.glob("*/*_samples_*.jsonl"))
        results = sorted(seed_dir.glob("*/*_results.json"))
        meta = seed_dir / "run_meta.txt"
        if len(samples) != 1 or len(results) != 1:
            raise ValueError("Expected exactly one samples and one results file")
        items = [json.loads(line) for line in samples[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(items) != 300 or len({item["mmmu_acc"]["id"] for item in items}) != 300:
            raise ValueError(f"Expected 300 unique scored items, got {len(items)}")
        compared = None
        if baseline:
            compared = run_dir / "comparison" / "comparison_summary.json"
            subprocess.run([
                str(python), str(compare_script),
                "--repo-root", str(repo), "--baseline-run", str(baseline),
                "--candidate-run", str(run_dir), "--output-dir", str(compared.parent),
                "--expected-items", "300",
            ], check=True)
            per_seed = read_json(compared)["per_seed"]["3407"]
            if per_seed["items_compared"] != 300 or per_seed["input_exact_agreement"] != 300 or per_seed["doc_hash_agreement"] != 300:
                raise ValueError("Pilot candidate and base used different inputs/doc hashes")
        artifacts = {
            "samples": {"path": str(samples[0]), "sha256": sha256_file(samples[0])},
            "results": {"path": str(results[0]), "sha256": sha256_file(results[0])},
            "run_meta": {"path": str(meta), "sha256": sha256_file(meta)},
        }
        if compared:
            artifacts["comparison_summary"] = {"path": str(compared), "sha256": sha256_file(compared)}
            paired = compared.parent / "paired_items.csv"
            artifacts["paired_items"] = {"path": str(paired), "sha256": sha256_file(paired)}
        write_json(run_dir / "handoff.json", {**request, "status": "complete", "completed_at_utc": now(), "result_files": artifacts})
        print(f"COMPLETE fixed300: {run_dir}")
    except Exception as error:
        write_json(run_dir / "failed.json", {**request, "status": "failed", "failed_at_utc": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
