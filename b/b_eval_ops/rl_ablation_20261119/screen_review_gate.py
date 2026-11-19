"""Filesystem review gate for an already-running, SHA-pinned evaluation queue.

The current queue has no pause option. Arm after its supervisor has started the
evaluation process: reserve every possible top-two 900-item output directory.
The queue completes all six fixed-300 screens and writes ranking.json, then
stops before the first 900-item evaluation because the selected destination
exists without a completed handoff. Preserve all artifacts and the gate markers
until a human has reviewed the ranking and approved resumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("arm", "status", "release"))
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--approved-ranking-sha256", help="Required for release after user review")
    args = parser.parse_args()
    plan = json.loads((args.plan_dir / "plan.json").read_text(encoding="utf-8"))
    names = [Path(p).parent.name for p in plan["training_runs"]]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError(f"Expected six distinct trained candidates, got {names}")
    if any("/" in n or "\\" in n or n in {".", ".."} for n in names):
        raise ValueError("Unsafe candidate name")
    targets = [args.output_root / f"{args.run_prefix}_{n}_screen900_s3407" for n in names]
    gate_path = args.plan_dir / "review_gate.json"
    ranking_path = args.plan_dir / "ranking.json"

    if args.action == "arm":
        if gate_path.exists() or ranking_path.exists() or any(p.exists() for p in targets):
            raise FileExistsError("Review gate already armed, ranking already written, or 900 output exists")
        if not (args.plan_dir / "progress.json").is_file():
            raise RuntimeError("Evaluation process has not started; arm only after supervisor destination checks")
        record = {
            "status": "armed_before_900", "armed_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "User review of all six fixed300 results before either top-two full900 evaluation",
            "plan_sha256": digest(args.plan_dir / "plan.json"),
            "candidate_names": names, "blocked_900_directories": [str(p) for p in targets],
        }
        for target in targets:
            target.mkdir(exist_ok=False)
            write_json(target / "REVIEW_GATE.json", record)
        write_json(gate_path, record)
    else:
        record = json.loads(gate_path.read_text(encoding="utf-8"))
        if record["plan_sha256"] != digest(args.plan_dir / "plan.json"):
            raise ValueError("Evaluation plan changed since gate was armed")
        if record["candidate_names"] != names or record["blocked_900_directories"] != [str(p) for p in targets]:
            raise ValueError("Review gate targets differ from the plan")
        for target in targets:
            if not target.is_dir() or {p.name for p in target.iterdir()} != {"REVIEW_GATE.json"}:
                raise ValueError(f"Gate marker modified or evaluation output exists: {target}")
        progress_path = args.plan_dir / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        stopped_at_gate = (
            progress.get("stage") == "queue"
            and progress.get("status") == "failed"
            and "FileExistsError" in progress.get("error", "")
            and "screen900_s3407" in progress.get("error", "")
        )
        if ranking_path.is_file() and stopped_at_gate and record["status"] == "armed_before_900":
            record["status"] = "awaiting_user_review"
            record["ranking_sha256"] = digest(ranking_path)
            write_json(gate_path, record)
        if args.action == "release":
            if not stopped_at_gate or record["status"] != "awaiting_user_review":
                raise RuntimeError("The queue must finish all six fixed300 screens and stop at the review gate")
            if not ranking_path.is_file() or digest(ranking_path) != args.approved_ranking_sha256:
                raise ValueError("Supply the SHA256 of the ranking approved by the user")
            selected = json.loads(ranking_path.read_text(encoding="utf-8"))["selected_for_900"]
            if len(selected) != 2 or any(name not in names for name in selected):
                raise ValueError("Invalid top-two ranking")
            for target in targets:
                (target / "REVIEW_GATE.json").unlink()
                target.rmdir()
            record.update(status="released_after_user_review",
                          released_at_utc=datetime.now(timezone.utc).isoformat(),
                          approved_ranking_sha256=args.approved_ranking_sha256,
                          selected_for_900=selected)
            write_json(gate_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
