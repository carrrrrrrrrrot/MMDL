"""Record exact Stage1 smoke inputs, code hashes, and algorithm overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--method", required=True, choices=("grpo", "dapo_core"))
    parser.add_argument("--cap", required=True, type=int)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.engine), "rev-parse", "HEAD"], text=True).strip()
    patch = subprocess.check_output(["git", "-C", str(args.engine), "diff", "--binary"])
    config = args.stage1_root / "config_smoke.yaml"
    reward = args.stage1_root / "reward_v12.py"
    launcher = args.stage1_root / "run_smoke.sh"
    manifest = {
        "engine_git_head": head,
        "engine_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "parent_model": args.parent_model,
        "method": args.method,
        "method_label": "DAPO core, no soft-overlong penalty" if args.method == "dapo_core" else "GRPO",
        "raw_reward": "MMDL v1.2 binary 0/1",
        "optimized_reward": "MMDL v1.2 binary 0/1",
        "max_response_length": args.cap,
        "questions_per_optimizer_step": 5,
        "rollouts_per_question": 5,
        "max_optimizer_steps": 1,
        "learning_rate": 1e-6,
        "kl_penalty": 0,
        "train_file": str(args.train),
        "train_sha256": sha256(args.train),
        "validation_file": str(args.val),
        "validation_sha256": sha256(args.val),
        "validation_role": "5-row pipeline smoke, not a performance evaluation",
        "config_sha256": sha256(config),
        "reward_sha256": sha256(reward),
        "v12_scorer_sha256": sha256(args.stage1_root / "v12_scorer.py"),
        "launcher_sha256": sha256(launcher),
    }
    path = args.run_dir / "run_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_manifest": str(path), "sha256": sha256(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
