"""Copy the pinned, patched Stage1 engine and data from /tmp to persistent /data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/mmdl_easyr1_stage1_20261120"))
    parser.add_argument("--scripts", type=Path, default=Path("/tmp/mmdl_stage1_scripts_20261120"))
    parser.add_argument("--target", type=Path, default=Path("/data/chaewon/mmdl_stage1_paper_style_20261120"))
    args = parser.parse_args()
    if args.target.exists():
        raise FileExistsError(args.target)
    head = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"Unexpected EasyR1 HEAD {head}")
    args.target.mkdir(parents=True)
    shutil.copytree(args.source, args.target / "easyr1",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(args.scripts, args.target / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "select_mixed_smoke.py",
                                                  "*PREVIEW*", "*preview*"))
    patch_text = subprocess.check_output(["git", "-C", str(args.source), "diff"], text=True)
    if not patch_text:
        raise RuntimeError("Expected pinned EasyR1 Stage1 patches")
    (args.target / "easyr1_stage1.patch").write_text(patch_text, encoding="utf-8")

    packages = {}
    for name in ("torch", "transformers", "peft", "vllm", "ray", "flash-attn", "hydra-core", "omegaconf"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = {}
    for relative in (
        "verl/trainer/main.py", "verl/trainer/ray_trainer.py", "verl/utils/dataset.py",
        "verl/utils/tokenizer.py", "verl/workers/fsdp_workers.py",
        "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    ):
        files[f"easyr1/{relative}"] = sha256(args.target / "easyr1" / relative)
    for path in (args.target / "scripts").rglob("*"):
        if path.is_file():
            files[str(path.relative_to(args.target)).replace("\\", "/")] = sha256(path)
    files["easyr1_stage1.patch"] = sha256(args.target / "easyr1_stage1.patch")
    manifest = {
        "easy_r1_git_head": head,
        "qwen3_vl_4b_base_revision": BASE_REVISION,
        "python": sys.version.split()[0],
        "packages": packages,
        "files_sha256": dict(sorted(files.items())),
        "note": "Stage1 smoke code and data; run_persistent.sh selects this package.",
    }
    (args.target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"target": str(args.target), "files": len(files),
                      "manifest_sha256": sha256(args.target / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
