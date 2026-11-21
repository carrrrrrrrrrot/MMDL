"""Verify every file hash in a frozen Stage1 package without starting GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "-C", str(args.package / "easyr1"),
                                    "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["easy_r1_git_head"]:
        raise ValueError("Pinned EasyR1 HEAD mismatch")
    for relative, expected in manifest["files_sha256"].items():
        path = args.package / relative
        if sha256(path) != expected:
            raise ValueError(f"SHA mismatch: {relative}")
    launcher = (args.package / "scripts/run_smoke.sh").read_text(encoding="utf-8")
    wrapper = (args.package / "scripts/run_persistent.sh").read_text(encoding="utf-8")
    if 'worker.reward.reward_function=$stage1_root/reward_v12.py:compute_score' not in launcher:
        raise ValueError("Persistent reward path override missing")
    if 'MMDL_STAGE1_ROOT="$package_root/scripts"' not in wrapper:
        raise ValueError("Persistent script root missing")
    print(json.dumps({"package": str(args.package), "verified_files": len(manifest["files_sha256"]),
                      "manifest_sha256": sha256(args.package / "manifest.json"),
                      "engine_git_head": head}, sort_keys=True))


if __name__ == "__main__":
    main()
