"""Pass Stage1 parity and visit-log settings to Ray workers in pinned EasyR1."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
SOURCE_SHA = "f6bdd143a7da3f91634e3b6816171a5abadc02e7a8079cbfd29f59402a7d3ec4"


def patch(root: Path) -> Path:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    path = root / "verl/trainer/main.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("EasyR1 main.py hash differs from pinned source")
    source = raw.decode("utf-8")
    old = "import json\n\nimport ray\n"
    new = "import json\nimport os\n\nimport ray\n"
    if source.count(old) != 1:
        raise ValueError("Expected exactly one import block")
    source = source.replace(old, new)
    old = "        ray.init(runtime_env=runtime_env)\n"
    new = (
        '        for key in ("MMDL_V12_IMAGE_PARITY", "MMDL_VISIT_LOG_PATH"):\n'
        '            if key in os.environ:\n'
        '                runtime_env["env_vars"][key] = os.environ[key]\n'
        '        ray.init(runtime_env=runtime_env)\n'
    )
    if source.count(old) != 1:
        raise ValueError("Expected exactly one ray.init call")
    path.write_text(source.replace(old, new), encoding="utf-8", newline="")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    print(patch(parser.parse_args().easyr1_root.resolve(strict=True)))
