"""Recover and verify a standard PEFT LoRA adapter from a one-GPU EasyR1 shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


PATTERN = re.compile(
    r"base_model\.model\.model\.language_model\.layers\.\d+\."
    r"self_attn\.(?:q|k|v|o)_proj\.lora_[AB]\.default\.weight"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert config["r"] == 16 and config["lora_alpha"] == 32
    assert set(config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert config["exclude_modules"] == ".*visual.*"

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    adapter = {
        key.replace(".default.weight", ".weight"): value.detach().cpu().contiguous()
        for key, value in state.items() if PATTERN.fullmatch(key)
    }
    numel = sum(value.numel() for value in adapter.values())
    if len(adapter) != 288 or numel != 11796480:
        raise RuntimeError(f"Invalid language LoRA shard: {len(adapter)} tensors, {numel} elements")

    args.output.mkdir(parents=True)
    shutil.copy2(args.config, args.output / "adapter_config.json")
    path = args.output / "adapter_model.safetensors"
    save_file(adapter, path)
    reloaded = load_file(path, device="cpu")
    if set(reloaded) != set(adapter) or any(
        not torch.equal(reloaded[name], adapter[name]) for name in adapter
    ):
        raise RuntimeError("Safetensors round-trip mismatch")
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "adapter": str(path),
        "adapter_sha256": sha256(path),
        "adapter_config_sha256": sha256(args.output / "adapter_config.json"),
        "tensor_count": len(adapter),
        "parameter_count": numel,
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
