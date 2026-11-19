#!/usr/bin/env python3
"""Wait for ECD798 training, then merge and evaluate only the fixed 300 items.

This deliberately has no full-900 path. It reuses the pinned v1.2 evaluator,
the existing fixed ID list, and the 6000 base run from QC-v2.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace


OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS / "rl_ablation_20261119"))
from run_eval_queue import (  # noqa: E402
    completed_fixed300,
    event,
    merge_model,
    pilot_stats,
    run_candidate_eval,
    run_process,
    training_info,
)
sys.path.insert(0, str(OPS))
from candidate_eval import check_gpu_idle, git_check  # noqa: E402
from ops_common import read_json, sha256_file, write_json  # noqa: E402


DATA_SHA256 = "f1570689440a7c5efb173be201096ec0e770caf9c85a18b2b5efda1ef3f11a21"
IDS_SHA256 = "1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EVAL_REPO_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
EXPECTED_STEPS = 1596
RUNS = ("ecd_grpo", "ecd_dapo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--base-fixed300", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--compare-script", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="ecd798full_20261120")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout", type=int, default=28800)
    return parser.parse_args()


def check_training(args: argparse.Namespace) -> list[dict]:
    plan = read_json(args.train_root / "plan.json")
    if plan.get("run_order") != list(RUNS) or plan.get("parent_kind") != "base":
        raise ValueError("Training plan run order or parent changed")
    if (plan.get("base_revision") != BASE_REVISION or plan.get("max_steps") != EXPECTED_STEPS or
            plan.get("save_steps") != 399):
        raise ValueError("Training plan base or steps changed")
    if plan.get("seed") != 3407 or plan.get("num_generations") != 4:
        raise ValueError("Training seed or rollouts changed")
    if plan.get("learning_rate") != 1e-5 or plan.get("max_completion_length") != 512:
        raise ValueError("Training learning rate or completion cap changed")
    dataset = plan.get("datasets", {}).get("ecd", {})
    if dataset.get("sha256") != DATA_SHA256 or dataset.get("split_counts") != {"train": 798, "heldout": 97}:
        raise ValueError("Training plan dataset changed")
    if Path(dataset.get("manifest", "")).resolve(strict=True) != args.train_manifest:
        raise ValueError("Training plan points to a different ECD manifest")
    if sha256_file(args.train_manifest) != DATA_SHA256:
        raise ValueError("ECD798 data SHA256 changed")
    summary = read_json(args.train_root / "summary.json")
    if summary.get("status") != "complete" or summary.get("blocked_datasets"):
        raise ValueError("Training queue is not a complete two-run success")
    items = []
    for name in RUNS:
        path = args.train_root / "runs" / name / "run_manifest.json"
        item = training_info(path, EXPECTED_STEPS)
        if item["name"] != name or item["info"]["loss_type"] != name.rsplit("_", 1)[1]:
            raise ValueError(f"Wrong training result at {path}")
        dataset_info = item["info"].get("dataset", {})
        if dataset_info.get("manifest_sha256") != DATA_SHA256:
            raise ValueError(f"Wrong dataset lineage at {path}")
        items.append(item)
    if items[0]["info"].get("initial_lora_sha256") != items[1]["info"].get("initial_lora_sha256"):
        raise ValueError("GRPO and DAPO did not share the same initial LoRA")
    if len(summary.get("completed", [])) != 2:
        raise ValueError("Training summary does not list both completed runs")
    return items


def wait_for_training(args: argparse.Namespace) -> list[dict]:
    started = time.monotonic()
    while True:
        summary_path = args.train_root / "summary.json"
        if summary_path.is_file():
            state = read_json(summary_path).get("status")
            if state == "complete":
                return check_training(args)
            if state in {"failed", "complete_with_no_signal", "paused"}:
                raise RuntimeError(f"Training queue ended in {state}; no evaluation launched")
        if time.monotonic() - started > args.wait_timeout:
            raise TimeoutError("Training did not complete before wait timeout")
        time.sleep(args.poll_seconds)


def check_fixed_inputs(args: argparse.Namespace) -> str:
    if sha256_file(args.ids_file) != IDS_SHA256:
        raise ValueError("Fixed300 ID list changed")
    ids_manifest = read_json(args.ids_file.with_suffix(args.ids_file.suffix + ".manifest.json"))
    if ids_manifest.get("ids_sha256") != IDS_SHA256:
        raise ValueError("Fixed300 selection manifest changed")
    if not completed_fixed300(args.base_fixed300, IDS_SHA256, None):
        raise ValueError("6000 base fixed300 is not complete")
    base = read_json(args.base_fixed300 / "handoff.json")
    if (base.get("run_name") != args.base_fixed300.name or base.get("items") != 300 or
            base.get("evaluation_repo_commit") != EVAL_REPO_COMMIT or
            Path(base.get("model_path", "")).resolve(strict=True) != args.base_model):
        raise ValueError("Base fixed300 handoff has wrong run name")
    return IDS_SHA256


def fixed300(args: argparse.Namespace, item: dict, model_manifest: Path, ids_sha: str) -> dict:
    name = item["name"]
    run_name = f"{args.run_prefix}_{name}_fixed300"
    output = args.output_root / run_name
    train_sha = sha256_file(item["manifest"])
    if output.exists():
        if not (output / "handoff.json").is_file():
            raise FileExistsError(f"Incomplete fixed300 result: {output}")
        if completed_fixed300(output, ids_sha, train_sha):
            event(args.plan_dir, "fixed300", name, "verified_existing", run=str(output))
            return pilot_stats(output, ids_sha, args.base_fixed300)
    log = args.plan_dir / "logs" / f"fixed300_{name}.log"
    if log.exists():
        raise FileExistsError(f"Refusing to overwrite existing log: {log}")
    command = [
        str(args.eval_python), str(OPS / "rl_ablation_20261119" / "run_fixed300.py"),
        "--model-manifest", str(model_manifest),
        "--train-manifest", str(item["manifest"]),
        "--repo-root", str(args.repo_root),
        "--data-root", str(args.data_root),
        "--eval-python", str(args.eval_python),
        "--output-root", str(args.output_root),
        "--run-name", run_name,
        "--ids-file", str(args.ids_file),
        "--baseline-run", str(args.base_fixed300),
        "--compare-script", str(args.compare_script),
        "--verify-model-hashes",
    ]
    run_process(command, log, 7200, args.plan_dir, "fixed300", name)
    if not completed_fixed300(output, ids_sha, train_sha):
        raise RuntimeError(f"Fixed300 result not complete: {output}")
    event(args.plan_dir, "fixed300", name, "complete", run=str(output))
    return pilot_stats(output, ids_sha, args.base_fixed300)


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 5 or args.wait_timeout < 60:
        raise ValueError("Invalid poll/timeout")
    for key in ("train_root", "train_manifest", "base_model", "repo_root", "data_root",
                "train_python", "eval_python", "output_root", "merged_root",
                "base_fixed300", "ids_file", "compare_script"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    if BASE_REVISION not in str(args.base_model):
        raise ValueError("Wrong base revision")
    if git_check(args.repo_root) is None:
        raise ValueError("Frozen evaluation code did not validate")
    args.plan_dir = args.plan_dir.resolve()
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "kind": "ecd798_after_train_fixed300_only",
        "train_root": str(args.train_root),
        "train_manifest_sha256": DATA_SHA256,
        "ids_sha256": IDS_SHA256,
        "base_fixed300": str(args.base_fixed300),
        "runs": list(RUNS),
        "run_prefix": args.run_prefix,
        "eval_repo_commit": EVAL_REPO_COMMIT,
        "script_sha256": sha256_file(Path(__file__)),
    }
    plan_path = args.plan_dir / "plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("Existing eval plan differs")
    else:
        write_json(plan_path, plan)
    args.merge_timeout = 3600
    args.smoke_timeout = 3600
    args.baseline_900 = None
    ids_sha = check_fixed_inputs(args)
    event(args.plan_dir, "queue", "all", "waiting_for_training")
    try:
        items = wait_for_training(args)
        event(args.plan_dir, "queue", "all", "training_complete")
        scores = []
        for item in items:
            name = item["name"]
            merge_log = args.plan_dir / "logs" / f"merge_{name}.log"
            merge_output = args.merged_root / f"{args.run_prefix}_{name}"
            if merge_log.exists() and not merge_output.exists():
                raise FileExistsError(f"Merge log exists without output: {merge_log}")
            check_gpu_idle()
            model_manifest = merge_model(item, args, args.plan_dir)
            smoke_log = args.plan_dir / "logs" / f"smoke20_{name}.log"
            smoke_output = args.output_root / f"{args.run_prefix}_{name}_smoke20"
            if smoke_log.exists() and not smoke_output.exists():
                raise FileExistsError(f"Smoke log exists without output: {smoke_log}")
            run_candidate_eval(
                item=item, model_manifest=model_manifest, args=args, plan=args.plan_dir,
                stage="smoke20", mode="smoke",
                run_name=f"{args.run_prefix}_{item['name']}_smoke20", timeout=args.smoke_timeout,
            )
            stats = fixed300(args, item, model_manifest, ids_sha)
            scores.append({"name": item["name"], "run": f"{args.run_prefix}_{item['name']}_fixed300",
                           "correct": stats["candidate_correct"], "base_correct": stats["baseline_correct"],
                           "training_manifest_sha256": sha256_file(item["manifest"])})
        result = {
            "kind": "ECD798_fullpass_fixed300_screening_not_official",
            "data_sha256": DATA_SHA256,
            "ids_sha256": IDS_SHA256,
            "optimizer_steps": EXPECTED_STEPS,
            "scores": scores,
            "status": "complete",
            "next_step": "Compare with prior ECD500 fixed300; user review before any 900-item evaluation",
        }
        write_json(args.plan_dir / "screening_summary.json", result)
        event(args.plan_dir, "queue", "all", "complete", summary=str(args.plan_dir / "screening_summary.json"))
    except Exception as error:
        event(args.plan_dir, "queue", "all", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
