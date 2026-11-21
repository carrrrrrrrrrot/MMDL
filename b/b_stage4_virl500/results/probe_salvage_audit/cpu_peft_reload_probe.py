"""Read-only CPU reload audit for the Stage 4 GRPO probe LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import Qwen3VLForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    config = PeftConfig.from_pretrained(args.adapter, local_files_only=True)
    weights = args.adapter / "adapter_model.safetensors"
    saved = {}
    with safe_open(weights, framework="pt", device="cpu") as stream:
        for key in stream.keys():
            tensor = stream.get_tensor(key)
            saved[key] = tensor
    key_count = len(saved)
    all_finite = all(bool(torch.isfinite(tensor).all()) for tensor in saved.values())
    nonzero_b = sum(int(torch.count_nonzero(tensor)) for key, tensor in saved.items() if ".lora_B." in key)
    b_elements = sum(tensor.numel() for key, tensor in saved.items() if ".lora_B." in key)
    assert all_finite and key_count == 288
    assert all("language_model" in key and "visual" not in key for key in saved)
    assert nonzero_b > 0

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    loaded = {
        name.replace(".default.", "."): parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    assert len(loaded) == key_count, (len(loaded), key_count)
    assert saved.keys() == loaded.keys(), (list(saved.keys())[:3], list(loaded.keys())[:3])
    mismatch = [key for key in saved if not torch.equal(saved[key], loaded[key])]
    assert not mismatch, mismatch[:3]
    result = {
        "status": "passed_cpu_peft_reload",
        "base": str(args.base),
        "adapter": str(args.adapter),
        "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256(weights),
        "peft_type": config.peft_type.value,
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "target_modules": sorted(config.target_modules),
        "exclude_modules": config.exclude_modules,
        "saved_tensor_count": key_count,
        "loaded_tensor_count": len(loaded),
        "exact_tensor_equal_count": key_count,
        "all_finite": all_finite,
        "lora_b_nonzero_elements": nonzero_b,
        "lora_b_total_elements": b_elements,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "cpu_only": not torch.cuda.is_available(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
