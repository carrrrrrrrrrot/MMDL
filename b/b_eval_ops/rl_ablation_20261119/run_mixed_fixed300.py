"""Screen one additional mixed-data GRPO run without releasing the 900-item gate.

This deliberately has no 900-item evaluation path. Run once for mixed500 and
once for mixed1000, using distinct prefixes and plan directories. GPU work
requires the explicit --execute flag and a completed training manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))

from candidate_eval import git_check  # noqa: E402
from ops_common import read_json, safe_name, sha256_file, write_json  # noqa: E402
from run_eval_queue import (  # noqa: E402
    completed_eval, event, merge_model, pilot_stats,
    run_candidate_eval, run_process, training_info,
)
from run_fixed300 import validate_ids  # noqa: E402


EVAL_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
IDS_SHA = "1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d"
REWARD_SHA = "dbaacb4a111d201fc30e3969434f93961c0a0573deb141e634785e8732b74f25"
BASE_CORRECT = 185


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--baseline-300", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--compare-script", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="Print paths without requiring trained weights")
    mode.add_argument("--preflight", action="store_true", help="Verify completed training and all frozen inputs, no GPU launch")
    mode.add_argument("--execute", action="store_true", help="Merge, smoke20, and fixed300, sequentially; no 900")
    return parser.parse_args()


def names(args: argparse.Namespace) -> dict[str, str]:
    prefix = safe_name(args.run_prefix)
    if not prefix.startswith("rlab19mix") or prefix.startswith("rlab19qcv2") or "900" in prefix:
        raise ValueError("Use a new rlab19mix* prefix; the six-run 900 gate is reserved")
    name = safe_name(args.train_manifest.parent.name)
    return {
        "candidate": name,
        "model": f"{prefix}_{name}",
        "smoke": f"{prefix}_{name}_smoke20",
        "fixed300": f"{prefix}_{name}_fixed300",
    }


def verify_base_and_ids(args: argparse.Namespace) -> tuple[Path, Path, str]:
    repo = args.repo_root.resolve(strict=True)
    git_check(repo)  # also refuses a dirty frozen evaluation worktree
    ids_file, ids_sha = validate_ids(args.ids_file)
    if ids_sha != IDS_SHA:
        raise ValueError(f"Wrong fixed300 ID SHA256: {ids_sha}")
    baseline = args.baseline_300.resolve(strict=True)
    handoff = read_json(baseline / "handoff.json")
    required = {
        "status": "complete", "task": "mmmu_val_mmdl_diag", "seed": "3407",
        "items": 300, "ids_sha256": IDS_SHA, "evaluation_repo_commit": EVAL_COMMIT,
    }
    for key, value in required.items():
        if handoff.get(key) != value:
            raise ValueError(f"Fixed300 baseline {key} differs: {handoff.get(key)!r}")
    result = next((Path(v["path"]) for k, v in handoff["result_files"].items() if k == "results"), None)
    if result is None or not result.is_file():
        raise FileNotFoundError("Fixed300 baseline results file missing")
    if sha256_file(result) != handoff["result_files"]["results"]["sha256"]:
        raise ValueError("Fixed300 baseline result hash changed")
    score = read_json(result)["results"]["mmmu_val_mmdl_diag"]["mmmu_acc,none"]
    if round(score * 300) != BASE_CORRECT:
        raise ValueError(f"Fixed300 baseline changed from {BASE_CORRECT}/300: {score}")
    return ids_file, baseline, ids_sha


def main() -> None:
    args = arguments()
    paths = names(args)
    plan = {
        "purpose": "additional mixed-data GRPO smoke20 and fixed300 only; 900 gate stays closed",
        "training_manifest": str(args.train_manifest),
        "expected_dataset_sha256": args.dataset_sha256,
        "expected_examples": args.expected_examples,
        "expected_steps": args.expected_steps,
        "run_prefix": args.run_prefix,
        "paths": {
            "model": str(args.merged_root / paths["model"]),
            "smoke": str(args.output_root / paths["smoke"]),
            "fixed300": str(args.output_root / paths["fixed300"]),
            "plan": str(args.plan_dir),
        },
        "baseline_300": str(args.baseline_300),
        "ids_sha256": IDS_SHA,
        "evaluation_commit": EVAL_COMMIT,
        "stages": ["verify training", "merge", "smoke20", "fixed300", "paired comparison"],
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    ids_file, baseline, ids_sha = verify_base_and_ids(args)
    item = training_info(args.train_manifest, args.expected_steps)
    train = item["info"]
    data = train.get("dataset", {})
    if data.get("manifest_sha256") != args.dataset_sha256:
        raise ValueError("Training dataset SHA differs from requested mixed manifest")
    data_manifest = Path(data.get("manifest_path", "")).resolve(strict=True)
    if sha256_file(data_manifest) != args.dataset_sha256:
        raise ValueError("Training dataset manifest changed on disk")
    if data.get("selected_examples") != args.expected_examples:
        raise ValueError("Training example count differs from requested mixed set")
    if train.get("global_step") != args.expected_steps:
        raise ValueError("Training stopped before the requested optimizer step")
    if train.get("loss_type") != "grpo" or train.get("reward_policy_sha256") != REWARD_SHA:
        raise ValueError("Training must use GRPO and the pinned v1.2 reward policy")
    if item["name"] != paths["candidate"]:
        raise ValueError("Training candidate name changed")
    plan["training_manifest_sha256"] = sha256_file(item["manifest"])
    plan["adapter_sha256"] = train["adapter_sha256"]
    plan["reward_policy_sha256"] = train["reward_policy_sha256"]
    plan["baseline_results_sha256"] = sha256_file(Path(read_json(baseline / "handoff.json")["result_files"]["results"]["path"]))
    if args.preflight:
        print(json.dumps({**plan, "preflight": "passed", "gpu_jobs_started": 0}, indent=2))
        return

    for name in ("base_model", "repo_root", "data_root", "train_python", "eval_python", "output_root", "merged_root", "compare_script"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.plan_dir = args.plan_dir.resolve(strict=False)
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    plan_file = args.plan_dir / "mixed_fixed300_plan.json"
    if plan_file.exists() and read_json(plan_file) != plan:
        raise ValueError(f"Existing mixed screen plan has different inputs: {plan_file}")
    if not plan_file.exists():
        write_json(plan_file, plan)
    event(args.plan_dir, "preflight", item["name"], "complete", dataset_sha=args.dataset_sha256, ids_sha=ids_sha)

    model_manifest = merge_model(item, args, args.plan_dir)
    smoke_dir = args.output_root / paths["smoke"]
    smoke_complete = completed_eval(smoke_dir, "smoke", plan["training_manifest_sha256"])
    if smoke_complete and read_json(smoke_dir / "handoff.json").get("model_manifest_sha256") != sha256_file(model_manifest):
        raise ValueError("Existing smoke uses a different merged model manifest")
    if not smoke_complete:
        run_candidate_eval(
            item=item, model_manifest=model_manifest, args=args, plan=args.plan_dir,
            stage="smoke20", mode="smoke", run_name=paths["smoke"], timeout=3600,
        )
    else:
        event(args.plan_dir, "smoke20", item["name"], "verified_existing", run=str(smoke_dir))

    fixed_dir = args.output_root / paths["fixed300"]
    command = [
        str(args.train_python), str(Path(__file__).with_name("run_fixed300.py")),
        "--model-manifest", str(model_manifest), "--train-manifest", str(item["manifest"]),
        "--repo-root", str(args.repo_root), "--data-root", str(args.data_root),
        "--eval-python", str(args.eval_python), "--output-root", str(args.output_root),
        "--run-name", paths["fixed300"], "--ids-file", str(ids_file),
        "--baseline-run", str(baseline), "--compare-script", str(args.compare_script),
        "--timeout-seconds", "5400", "--verify-model-hashes",
    ]
    # run_fixed300.py verifies and skips a completed matching run; it refuses an
    # incomplete or changed run without overwriting it.
    run_process(command, args.plan_dir / "logs" / f"fixed300_{item['name']}.log", 5700, args.plan_dir, "fixed300", item["name"])
    stats = pilot_stats(fixed_dir, ids_sha, baseline)
    if stats["baseline_correct"] != BASE_CORRECT:
        raise ValueError("Paired comparison did not use the pinned 185/300 baseline")
    output = {
        "status": "complete", "dataset_sha256": args.dataset_sha256,
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "adapter_sha256": train["adapter_sha256"], "ids_sha256": ids_sha,
        "base_correct": BASE_CORRECT, "candidate_correct": stats["candidate_correct"],
        "correct_difference": stats["correct_difference"],
        "input_exact_agreement": stats["input_exact_agreement"],
        "doc_hash_agreement": stats["doc_hash_agreement"],
        "fixed300_run": str(fixed_dir),
        "comparison_summary": str(fixed_dir / "comparison" / "comparison_summary.json"),
        "paired_items": str(fixed_dir / "comparison" / "paired_items.csv"),
        "note": "Diagnostic 6000 seed-3407 screen only; 900 gate and 4090 official evaluation unchanged.",
    }
    write_json(args.plan_dir / "mixed_fixed300_result.json", output)
    event(args.plan_dir, "fixed300", item["name"], "complete", candidate_correct=stats["candidate_correct"], base_correct=BASE_CORRECT)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
