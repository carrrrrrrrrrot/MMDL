"""Apply narrowly scoped MMDL v1.2 prompt/image parity to pinned EasyR1.

Run only against EasyR1 commit a73346d028ba0335aa42da104972c86e6cd0fe69.
The patch is opt-in through MMDL_V12_IMAGE_PARITY=1 for image handling.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPECTED_HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
EXPECTED_SHA = {
    "verl/utils/dataset.py": "1c266af7bb86064dc37c414f24ab64bfb0c107e71ca7e1b47a2def4d0d9b0ebc",
    "verl/utils/tokenizer.py": "0e20ac39d453e7dc7a4498da91806a7e2629eb59e34c47a26edfdf6ffa6401dc",
}


def replace_once(source: str, old: str, new: str, path: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one match in {path}, found {count}: {old[:60]!r}")
    return source.replace(old, new)


def patch(root: Path) -> list[Path]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    originals = {}
    for name, expected in EXPECTED_SHA.items():
        path = root / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(f"Source hash differs for {name}: {digest}")
        originals[name] = raw.decode("utf-8")

    dataset = originals["verl/utils/dataset.py"]
    dataset = replace_once(
        dataset,
        '    image.load()  # avoid "Too many open files" errors\n',
        '    image.load()  # avoid "Too many open files" errors\n'
        '    if os.environ.get("MMDL_V12_IMAGE_PARITY") == "1":\n'
        '        from qwen_vl_utils import fetch_image\n'
        '        return fetch_image({"image": image.convert("RGB")}, image_patch_size=16)\n',
        "verl/utils/dataset.py",
    )
    dataset = replace_once(
        dataset,
        '        prompt_str: str = example[self.prompt_key]\n',
        '        prompt_str: str = example[self.prompt_key]\n'
        '        system_prompt = example.get("system_prompt")\n'
        '        if system_prompt is not None and system_prompt != "You are a helpful assistant.":\n'
        '            raise ValueError("Unexpected MMDL system prompt")\n'
        '        def with_system(messages):\n'
        '            return ([{"role": "system", "content": system_prompt}] + messages\n'
        '                    if system_prompt is not None else messages)\n',
        "verl/utils/dataset.py",
    )
    dataset = dataset.replace(
        'return [{"role": "user", "content": content_list}]',
        'return with_system([{"role": "user", "content": content_list}])',
    )
    dataset = replace_once(
        dataset,
        'return [{"role": "user", "content": prompt_str}]',
        'return with_system([{"role": "user", "content": prompt_str}])',
        "verl/utils/dataset.py",
    )
    if dataset.count('return with_system([{"role": "user", "content": content_list}])') != 2:
        raise ValueError("Expected image and video message paths")

    tokenizer = originals["verl/utils/tokenizer.py"]
    tokenizer = replace_once(tokenizer, 'from typing import Optional\n', 'import os\nfrom typing import Optional\n', "verl/utils/tokenizer.py")
    tokenizer = replace_once(
        tokenizer,
        '    processor = AutoProcessor.from_pretrained(model_path, **kwargs)\n',
        '    processor = AutoProcessor.from_pretrained(model_path, **kwargs)\n'
        '    if os.environ.get("MMDL_V12_IMAGE_PARITY") == "1" and processor is not None:\n'
        '        processor.image_processor.size = {"shortest_edge": 4096, "longest_edge": 16777216}\n',
        "verl/utils/tokenizer.py",
    )
    targets = {
        root / "verl/utils/dataset.py": dataset,
        root / "verl/utils/tokenizer.py": tokenizer,
    }
    for path, content in targets.items():
        path.write_text(content, encoding="utf-8", newline="")
    return list(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    args = parser.parse_args()
    for changed in patch(args.easyr1_root.resolve(strict=True)):
        print(changed)
