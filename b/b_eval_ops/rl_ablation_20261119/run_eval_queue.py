"""Merge and screen completed B ablations on the 6000, with durable logs.

Input: repeated --train-manifest paths from completed, same-base B runs.
Runs one GPU job at a time: merge -> 20-item smoke -> fixed 300-item v1.2
diagnostic screen for every candidate. Ranks the paired 300-item screens only
for triage, then runs full 900 seed 3407 for the top two. It runs seed 1 for
the 900-item leader, and for the runner-up when the leader's 900-item margin is
small (<=9 questions) or its delta versus base is nonpositive. No 300-item or
6000 result is labeled official. Official 4090 evaluation is a separate step.

Completed exact runs are skipped. Incomplete/failed run directories are never
overwritten; they require inspection and a new plan/run prefix.
"""

from __future__ import annotations

import argparse
import hashlib
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
from ops_common import BASE_REVISION, read_json, safe_name, sha256_file, write_json  # noqa: E402


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--baseline-900", type=Path, required=True)
    parser.add_argument("--compare-script", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="rlab19")
    parser.add_argument("--expected-steps", type=int, default=1000)
    parser.add_argument("--merge-timeout", type=int, default=3600)
    parser.add_argument("--smoke-timeout", type=int, default=3600)
    parser.add_argument("--pilot-timeout", type=int, default=5400)
    parser.add_argument("--screen-timeout", type=int, default=10800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def event(plan: Path, stage: str, name: str, status: str, **extra) -> None:
    record = {"at_utc": stamp(), "stage": stage, "name": name, "status": status, **extra}
    with (plan / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(plan / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run_process(command: list[str], log: Path, timeout: int, plan: Path, stage: str, name: str) -> None:
    if timeout < 60:
        raise ValueError("Timeout must be at least 60 seconds")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        event(plan, stage, name, "running", pid=process.pid, log=str(log), command=command)
        started = time.monotonic()
        last_heartbeat = started
        while True:
            status = process.poll()
            if status is not None:
                if status:
                    raise RuntimeError(f"{stage} {name} exited {status}; inspect {log}")
                event(plan, stage, name, "process_complete", seconds=round(time.monotonic() - started, 1))
                return
            current = time.monotonic()
            if current - started > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(f"{stage} {name} exceeded {timeout}s; inspect {log}")
            if current - last_heartbeat >= 60:
                event(plan, stage, name, "heartbeat", elapsed_seconds=round(current - started, 1), log=str(log))
                last_heartbeat = current
            time.sleep(5)


def candidate_name(path: Path) -> str:
    name = path.resolve(strict=True).parent.name
    return safe_name(name)


def training_info(path: Path, expected_steps: int) -> dict:
    path = path.resolve(strict=True)
    info = read_json(path)
    required = {
        "status": "complete", "base_revision": BASE_REVISION, "parent_kind": "base",
        "seed": 3407, "max_steps": expected_steps, "num_iterations": 2,
        "num_generations": 4, "max_completion_length": 512,
    }
    for key, expected in required.items():
        if info.get(key) != expected:
            raise ValueError(f"Training {path} has {key}={info.get(key)!r}; expected {expected!r}")
    if info.get("loss_type") not in {"grpo", "dapo", "sapo"}:
        raise ValueError(f"Unsupported loss type: {path}")
    if info.get("learning_rate") != 1e-5:
        raise ValueError(f"Training {path} uses a different learning rate")
    if info.get("finite_loss") is not True or info.get("reload_success") is not True:
        raise ValueError(f"Training did not verify finite loss / adapter reload: {path}")
    if not isinstance(info.get("lora_delta_l1"), (float, int)) or info["lora_delta_l1"] <= 0:
        raise ValueError(f"Training has no LoRA update: {path}")
    if not isinstance(info.get("mixed_reward_groups"), int) or info["mixed_reward_groups"] <= 0:
        raise ValueError(f"Training has no mixed reward group: {path}")
    adapter = Path(info["adapter_dir"]).resolve(strict=True)
    if sha256_file(adapter / "adapter_model.safetensors") != info["adapter_sha256"]:
        raise ValueError(f"Adapter SHA256 differs from training record: {path}")
    return {"name": candidate_name(path), "manifest": path, "info": info, "adapter": adapter}


def baseline_fingerprints(baseline: Path) -> dict[str, str]:
    found = {}
    for seed in ("3407", "1", "2"):
        seed_dir = baseline / f"seed_{seed}"
        matches = sorted(seed_dir.glob("*/*_samples_*.jsonl"))
        meta = seed_dir / "run_meta.txt"
        if len(matches) != 1 or not meta.is_file() or "exit_status: 0" not in meta.read_text(encoding="utf-8"):
            raise ValueError(f"Base 6000 run seed {seed} is missing/incomplete: {seed_dir}")
        found[seed] = sha256_file(matches[0])
    return found


def completed_eval(run_dir: Path, mode: str, training_sha: str) -> bool:
    if not run_dir.exists():
        return False
    handoff_path = run_dir / "handoff.json"
    if not handoff_path.is_file():
        raise FileExistsError(f"Incomplete evaluation run; inspect before retry: {run_dir}")
    handoff = read_json(handoff_path)
    if handoff.get("mode") != mode or handoff.get("training_manifest_sha256") != training_sha:
        raise ValueError(f"Existing evaluation has different mode/lineage: {run_dir}")
    for seed_files in handoff.get("result_files", {}).values():
        for record in seed_files.values():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Existing evaluation result changed: {path}")
    if mode == "screen":
        summary = run_dir / "comparison" / "comparison_summary.json"
        paired = run_dir / "comparison" / "paired_items.csv"
        if not summary.is_file() or not paired.is_file():
            raise ValueError(f"Screen has no paired comparison: {run_dir}")
    return True


def completed_fixed300(run_dir: Path, ids_sha: str, training_sha: str | None) -> bool:
    handoff_path = run_dir / "handoff.json"
    if not handoff_path.is_file():
        return False
    handoff = read_json(handoff_path)
    if handoff.get("status") != "complete" or handoff.get("ids_sha256") != ids_sha:
        raise ValueError(f"Existing fixed300 run has wrong status/IDs: {run_dir}")
    if handoff.get("training_manifest_sha256") != training_sha:
        raise ValueError(f"Existing fixed300 run has wrong training lineage: {run_dir}")
    for record in handoff.get("result_files", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Existing fixed300 artifact changed: {path}")
    return True


def merge_model(item: dict, args: argparse.Namespace, plan: Path) -> Path:
    name = item["name"]
    output = args.merged_root.resolve(strict=True) / f"{args.run_prefix}_{name}"
    manifest = output / "model_manifest.json"
    if output.exists():
        if not manifest.is_file() or (output / "_INCOMPLETE").exists():
            raise FileExistsError(f"Incomplete merged model; inspect: {output}")
        record = read_json(manifest)
        if record.get("status") != "complete" or record.get("adapter_sha256") != item["info"]["adapter_sha256"]:
            raise ValueError(f"Existing merged model does not match training: {output}")
        event(plan, "merge", name, "verified_existing", manifest=str(manifest))
        return manifest
    command = [
        str(args.train_python), str(OPS / "merge_lora.py"),
        "--base-model", str(args.base_model),
        "--adapter", str(item["adapter"]), "--adapter-role", "grpo",
        "--adapter-sha256", item["info"]["adapter_sha256"],
        "--output-dir", str(output),
    ]
    run_process(command, plan / "logs" / f"merge_{name}.log", args.merge_timeout, plan, "merge", name)
    if not manifest.is_file() or read_json(manifest).get("adapter_sha256") != item["info"]["adapter_sha256"]:
        raise RuntimeError(f"Merge manifest missing/wrong: {manifest}")
    event(plan, "merge", name, "complete", manifest=str(manifest), sha256=sha256_file(manifest))
    return manifest


def run_candidate_eval(
    *, item: dict, model_manifest: Path, args: argparse.Namespace, plan: Path,
    stage: str, mode: str, run_name: str, timeout: int, seed: str = "3407",
) -> Path:
    run_dir = args.output_root / run_name
    training_sha = sha256_file(item["manifest"])
    if completed_eval(run_dir, mode, training_sha):
        event(plan, stage, item["name"], "verified_existing", run=str(run_dir))
        return run_dir
    command = [
        str(args.train_python), str(OPS / "candidate_eval.py"),
        "--model-manifest", str(model_manifest),
        "--train-run-manifest", str(item["manifest"]),
        "--repo-root", str(args.repo_root), "--data-root", str(args.data_root),
        "--eval-python", str(args.eval_python), "--output-root", str(args.output_root),
        "--run-name", run_name, "--mode", mode,
    ]
    if mode == "screen":
        command += [
            "--screen-seed", seed,
            "--baseline-run", str(args.baseline_900),
            "--compare-script", str(args.compare_script),
        ]
    if mode == "smoke":
        command.append("--verify-model-hashes")
    run_process(command, plan / "logs" / f"{stage}_{item['name']}.log", timeout, plan, stage, item["name"])
    if not completed_eval(run_dir, mode, training_sha):
        raise RuntimeError(f"Evaluation finished without complete handoff: {run_dir}")
    event(plan, stage, item["name"], "complete", run=str(run_dir))
    return run_dir


def pilot_stats(run_dir: Path, ids_sha: str, baseline: Path) -> dict:
    handoff = read_json(run_dir / "handoff.json")
    if handoff.get("status") != "complete" or handoff.get("ids_sha256") != ids_sha:
        raise ValueError(f"Incomplete/wrong-ID 300 screen: {run_dir}")
    stats = read_json(run_dir / "comparison" / "comparison_summary.json")["per_seed"]["3407"]
    if stats["items_compared"] != 300 or stats["full_900_coverage"]:
        raise ValueError(f"300 screen has wrong coverage: {run_dir}")
    if stats["input_exact_agreement"] != 300 or stats["doc_hash_agreement"] != 300:
        raise ValueError(f"300 screen inputs differ from base: {run_dir}")
    if str(baseline.resolve(strict=True)) != str(Path(read_json(run_dir / "comparison" / "comparison_summary.json")["baseline_run"]).resolve(strict=True)):
        raise ValueError(f"300 screen used different baseline: {run_dir}")
    return stats


def screen_stats(run_dir: Path, seed: str) -> dict:
    stats = read_json(run_dir / "comparison" / "comparison_summary.json")["per_seed"][seed]
    if stats["items_compared"] != 900 or stats["full_900_coverage"] is not True:
        raise ValueError(f"900 screen incomplete: {run_dir}")
    return stats


def main() -> None:
    args = parser_args()
    args.run_prefix = safe_name(args.run_prefix)
    args.base_model = args.base_model.resolve(strict=True)
    args.repo_root = args.repo_root.resolve(strict=True)
    args.data_root = args.data_root.resolve(strict=True)
    args.train_python = args.train_python.resolve(strict=True)
    args.eval_python = args.eval_python.resolve(strict=True)
    args.output_root = args.output_root.resolve(strict=True)
    args.merged_root = args.merged_root.resolve(strict=True)
    args.baseline_900 = args.baseline_900.resolve(strict=True)
    args.compare_script = args.compare_script.resolve(strict=True)
    args.plan_dir = args.plan_dir.resolve(strict=False)
    items = [training_info(path, args.expected_steps) for path in args.train_manifest]
    if len({item["name"] for item in items}) != len(items):
        raise ValueError("Training run names must be unique")
    if len(items) < 2:
        raise ValueError("At least two trained candidates are required for top-two screen")
    reward_hashes = {item["info"].get("reward_policy_sha256") for item in items}
    if len(reward_hashes) != 1 or not next(iter(reward_hashes)):
        raise ValueError("All training runs must share one pinned v1.2 reward policy SHA256")
    ids_file = args.plan_dir / "fixed300_ids_seed3407.txt"
    base300 = args.output_root / f"{args.run_prefix}_base_fixed300"
    plan_description = {
        "purpose": "B 6000 candidate screening only, final 4090 v1.2 3 seeds separate",
        "training_runs": [str(item["manifest"]) for item in items],
        "training_manifest_sha256": {item["name"]: sha256_file(item["manifest"]) for item in items},
        "stages": ["base300", "base300_vs_fullbase_parity", "merge", "smoke20", "fixed300", "top2_900_seed3407", "adaptive_900_seed1"],
        "expected_steps": args.expected_steps,
        "base900": str(args.baseline_900),
        "base900_samples_sha256": baseline_fingerprints(args.baseline_900),
    }
    if args.dry_run:
        print(json.dumps(plan_description, ensure_ascii=False, indent=2))
        return
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan_dir / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan_description:
        raise ValueError(f"Plan directory contains different inputs: {args.plan_dir}")
    if not plan_path.exists():
        write_json(plan_path, plan_description)
    try:
        selection = [
            str(args.eval_python), str(Path(__file__).with_name("make_fixed300_ids.py")),
            "--baseline-run", str(args.baseline_900), "--output", str(ids_file),
        ]
        if not ids_file.is_file():
            run_process(selection, args.plan_dir / "logs" / "make_fixed300.log", 300, args.plan_dir, "select300", "base")
        ids_sha = sha256_file(ids_file)
        base_command = [
            str(args.eval_python), str(Path(__file__).with_name("run_fixed300.py")),
            "--base-model", str(args.base_model), "--repo-root", str(args.repo_root),
            "--data-root", str(args.data_root), "--eval-python", str(args.eval_python),
            "--output-root", str(args.output_root), "--run-name", base300.name,
            "--ids-file", str(ids_file), "--timeout-seconds", str(args.pilot_timeout),
        ]
        if not completed_fixed300(base300, ids_sha, None):
            if base300.exists():
                raise FileExistsError(f"Incomplete base300 run; inspect before retry: {base300}")
            run_process(base_command, args.plan_dir / "logs" / "base_fixed300.log", args.pilot_timeout + 180, args.plan_dir, "base300", "base")
        completed_fixed300(base300, ids_sha, None)
        parity_dir = args.plan_dir / "base300_vs_fullbase_parity"
        parity_summary = parity_dir / "comparison_summary.json"
        if not parity_summary.is_file():
            parity_command = [
                str(args.eval_python), str(args.compare_script),
                "--repo-root", str(args.repo_root),
                "--baseline-run", str(args.baseline_900),
                "--candidate-run", str(base300),
                "--output-dir", str(parity_dir), "--allow-partial",
            ]
            run_process(parity_command, args.plan_dir / "logs" / "base300_parity.log", 300, args.plan_dir, "base300_parity", "base")
        parity = read_json(parity_summary)["per_seed"]["3407"]
        if parity["items_compared"] != 300 or parity["input_exact_agreement"] != 300 or parity["doc_hash_agreement"] != 300:
            raise ValueError("Diagnostic base300 task inputs differ from frozen full900 base")
        event(args.plan_dir, "base300_parity", "base", "complete", paired_items=300, comparison=str(parity_summary))

        pilots = []
        for item in items:
            name = item["name"]
            model_manifest = merge_model(item, args, args.plan_dir)
            run_candidate_eval(
                item=item, model_manifest=model_manifest, args=args, plan=args.plan_dir,
                stage="smoke20", mode="smoke", run_name=f"{args.run_prefix}_{name}_smoke20",
                timeout=args.smoke_timeout,
            )
            pilot_name = f"{args.run_prefix}_{name}_fixed300"
            pilot_dir = args.output_root / pilot_name
            pilot_command = [
                str(args.eval_python), str(Path(__file__).with_name("run_fixed300.py")),
                "--model-manifest", str(model_manifest), "--train-manifest", str(item["manifest"]),
                "--repo-root", str(args.repo_root), "--data-root", str(args.data_root),
                "--eval-python", str(args.eval_python), "--output-root", str(args.output_root),
                "--run-name", pilot_name, "--ids-file", str(ids_file),
                "--baseline-run", str(base300), "--compare-script", str(args.compare_script),
                "--timeout-seconds", str(args.pilot_timeout),
            ]
            if not completed_fixed300(pilot_dir, ids_sha, sha256_file(item["manifest"])):
                if pilot_dir.exists():
                    raise FileExistsError(f"Incomplete pilot300 run; inspect before retry: {pilot_dir}")
                run_process(pilot_command, args.plan_dir / "logs" / f"pilot300_{name}.log", args.pilot_timeout + 180, args.plan_dir, "pilot300", name)
            completed_fixed300(pilot_dir, ids_sha, sha256_file(item["manifest"]))
            stats = pilot_stats(pilot_dir, ids_sha, base300)
            pilots.append({
                "name": name, "model_manifest": str(model_manifest), "train_manifest": str(item["manifest"]),
                "run": str(pilot_dir), "correct": stats["candidate_correct"],
                "base_correct": stats["baseline_correct"],
                "missing_final_answer": stats["candidate_missing_final_answer"],
                "wrong_to_right": stats["wrong_to_right"], "right_to_wrong": stats["right_to_wrong"],
            })
        pilots.sort(key=lambda row: (-row["correct"], row["missing_final_answer"], row["name"]))
        ranking = {
            "kind": "fixed300_screening_rank_only_not_official",
            "created_at_utc": stamp(), "ids_sha256": ids_sha,
            "baseline_fixed300": str(base300), "candidates": pilots,
            "selected_for_900": [row["name"] for row in pilots[:2]],
            "warning": "300-item single-seed ranking is noisy; confirm on full 900 and further seeds",
        }
        ranking_path = args.plan_dir / "ranking.json"
        if ranking_path.exists():
            old = read_json(ranking_path)
            if [(r["name"], r["correct"]) for r in old["candidates"]] != [(r["name"], r["correct"]) for r in pilots]:
                raise ValueError("Existing ranking differs; preserve old plan and use a new plan directory")
        else:
            write_json(ranking_path, ranking)
        event(args.plan_dir, "ranking", "all", "complete", selected=ranking["selected_for_900"], path=str(ranking_path))

        selected = pilots[:2]
        screens3407 = []
        for row in selected:
            item = next(item for item in items if item["name"] == row["name"])
            run_dir = run_candidate_eval(
                item=item, model_manifest=Path(row["model_manifest"]), args=args, plan=args.plan_dir,
                stage="screen900_s3407", mode="screen",
                run_name=f"{args.run_prefix}_{item['name']}_screen900_s3407",
                timeout=args.screen_timeout, seed="3407",
            )
            stats = screen_stats(run_dir, "3407")
            screens3407.append({"name": item["name"], "run": str(run_dir), **{k: stats[k] for k in (
                "baseline_correct", "candidate_correct", "correct_difference", "candidate_missing_final_answer",
            )}})
        screens3407.sort(key=lambda row: (-row["candidate_correct"], row["candidate_missing_final_answer"], row["name"]))
        leader, runner = screens3407
        margin = leader["candidate_correct"] - runner["candidate_correct"]
        seed1 = []
        for name in (leader["name"],):
            item = next(item for item in items if item["name"] == name)
            run_dir = run_candidate_eval(
                item=item, model_manifest=args.merged_root / f"{args.run_prefix}_{name}" / "model_manifest.json",
                args=args, plan=args.plan_dir, stage="screen900_s1", mode="screen",
                run_name=f"{args.run_prefix}_{name}_screen900_s1",
                timeout=args.screen_timeout, seed="1",
            )
            stats = screen_stats(run_dir, "1")
            seed1.append({"name": name, "run": str(run_dir), **{k: stats[k] for k in (
                "baseline_correct", "candidate_correct", "correct_difference", "candidate_missing_final_answer",
            )}})
        # A's previous candidate reversed direction between seeds 3407 and 1.
        # Screen the runner-up on seed 1 if the 900-item margin is small or the
        # leader's seed-1 delta is nonpositive, rather than trusting one seed.
        if margin <= 9 or leader["correct_difference"] <= 0 or seed1[0]["correct_difference"] <= 0:
            name = runner["name"]
            item = next(item for item in items if item["name"] == name)
            run_dir = run_candidate_eval(
                item=item, model_manifest=args.merged_root / f"{args.run_prefix}_{name}" / "model_manifest.json",
                args=args, plan=args.plan_dir, stage="screen900_s1", mode="screen",
                run_name=f"{args.run_prefix}_{name}_screen900_s1",
                timeout=args.screen_timeout, seed="1",
            )
            stats = screen_stats(run_dir, "1")
            seed1.append({"name": name, "run": str(run_dir), **{k: stats[k] for k in (
                "baseline_correct", "candidate_correct", "correct_difference", "candidate_missing_final_answer",
            )}})
        final = {
            "kind": "B_6000_screening_only_not_official",
            "created_at_utc": stamp(), "fixed300_ranking": str(ranking_path),
            "top2_seed3407": screens3407, "additional_seed1": seed1,
            "next_step": "Select candidate for unchanged 4090 v1.2 official 3-seed evaluation",
        }
        write_json(args.plan_dir / "screening_summary.json", final)
        event(args.plan_dir, "screening", "all", "complete", summary=str(args.plan_dir / "screening_summary.json"))
    except Exception as error:
        event(args.plan_dir, "queue", "all", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
