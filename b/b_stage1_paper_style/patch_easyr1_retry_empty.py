"""Allow bounded DAPO retries when one rollout batch has no mixed rewards."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
SOURCE_SHA = "b966f8a3451fcc1bbbba83dd6c9dfa08eef74fbf64ab717a92d469de8e94ae21"


def patch(root: Path) -> Path:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    path = root / "verl/trainer/ray_trainer.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Expected the pinned EasyR1 source with visit-log patch")
    source = raw.decode("utf-8")
    old = (
        '                if len(kept_sample_idxs) == 0:\n'
        '                    raise RuntimeError("No sample is kept after filtering. Please check your data.")\n'
    )
    new = (
        '                if len(kept_sample_idxs) == 0:\n'
        '                    if num_try_make_batch >= self.config.trainer.max_try_make_batch:\n'
        '                        raise RuntimeError("No mixed reward group after bounded DAPO retries")\n'
        '                    continue\n'
    )
    if source.count(old) != 1:
        raise ValueError("Expected exactly one no-group branch")
    path.write_text(source.replace(old, new), encoding="utf-8", newline="")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    print(patch(parser.parse_args().easyr1_root.resolve(strict=True)))
