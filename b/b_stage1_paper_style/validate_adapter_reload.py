"""Reload Stage1 PEFT adapter on CPU and compare every tensor to safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.adapter / "adapter_model.safetensors"
    expected = load_file(path, device="cpu")
    if len(expected) != 288:
        raise ValueError(f"Expected 288 tensors, got {len(expected)}")

    base = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cpu", attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    actual = {
        name.replace(".default.weight", ".weight"): parameter.detach().cpu()
        for name, parameter in model.named_parameters() if "lora_" in name
    }
    if set(actual) != set(expected):
        raise RuntimeError(f"PEFT key mismatch: missing={len(set(expected)-set(actual))}, "
                           f"unexpected={len(set(actual)-set(expected))}")
    different = [name for name in expected if not torch.equal(actual[name], expected[name])]
    if different:
        raise RuntimeError(f"PEFT tensor mismatch: {different[:8]}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"peft_reload": "exact", "tensor_count": len(actual),
              "parameter_count": sum(p.numel() for p in actual.values()),
              "adapter_sha256": sha, "adapter_path": str(path)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
