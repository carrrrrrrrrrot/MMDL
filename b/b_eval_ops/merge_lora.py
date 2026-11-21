"""Merge ordered PEFT adapters into the fixed Qwen3-VL base for v1.2 evaluation.

Examples:
  base -> GRPO:  --adapter <grpo>/adapter --adapter-role grpo --adapter-sha256 <sha>
  SFT  -> GRPO:  --adapter <sft> --adapter-role sft --adapter-sha256 <sha>
                 --adapter <grpo>/adapter --adapter-role grpo --adapter-sha256 <sha>
  DPO  -> GRPO:  --adapter <dpo> --adapter-role dpo --adapter-sha256 <sha>
                 --adapter <grpo>/adapter --adapter-role grpo --adapter-sha256 <sha>

The order matters. This script is a B-side draft; A's shared merge contract must
be agreed before cross-server official comparisons.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ops_common import (
    BASE_MODEL_ID,
    BASE_REVISION,
    adapter_info,
    fixed_base_path,
    is_language_attention_lora_name,
    ordered_lineage,
    read_json,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True, help="Local snapshot ending in fixed revision")
    parser.add_argument("--adapter", type=Path, action="append", required=True, help="PEFT adapter directory, in merge order")
    parser.add_argument("--adapter-role", action="append", required=True, help="sft, dpo, or grpo, in merge order")
    parser.add_argument("--adapter-sha256", action="append", required=True, help="SHA256 of adapter_model.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for merged model")
    check_group = parser.add_mutually_exclusive_group()
    check_group.add_argument("--preflight-only", action="store_true", help="Verify lineage/files without loading model")
    check_group.add_argument("--load-check-only", action="store_true", help="Load/merge on CPU and validate PEFT targets without writing 9 GiB")
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def validate_roles(roles: list[str]) -> None:
    allowed = {("sft",), ("dpo",), ("grpo",), ("sft", "dpo"),
               ("sft", "grpo"), ("dpo", "grpo")}
    if tuple(roles) not in allowed:
        raise ValueError(f"Unsupported adapter order {roles}; expected base branch or SFT/DPO then GRPO")


def preflight(args: argparse.Namespace) -> tuple[Path, list[dict], dict]:
    if len(args.adapter) != len(args.adapter_role) or len(args.adapter) != len(args.adapter_sha256):
        raise ValueError("--adapter, --adapter-role, and --adapter-sha256 counts must match")
    validate_roles(args.adapter_role)
    base = fixed_base_path(args.base_model)
    adapters = [
        adapter_info(path, expected, role)
        for path, expected, role in zip(args.adapter, args.adapter_sha256, args.adapter_role, strict=True)
    ]
    lineage = ordered_lineage(adapters)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Merged output must be a new directory: {output}")
    if output == base or output in base.parents or base in output.parents:
        raise ValueError("Merged output must be separate from the base snapshot")
    return base, adapters, lineage


def verify_output(output: Path) -> dict[str, dict[str, object]]:
    config = read_json(output / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Merged output config is not qwen3_vl")
    if not (output / "preprocessor_config.json").is_file():
        raise FileNotFoundError("Merged output lacks preprocessor_config.json")
    shards = sorted(output.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("Merged output lacks safetensors weights")
    index = output / "model.safetensors.index.json"
    if index.is_file():
        weight_map = read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or {p.name for p in shards} != set(weight_map.values()):
            raise ValueError("Merged safetensors shards do not match index")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"_INCOMPLETE", "model_manifest.json"}
    }


def main() -> None:
    args = parse_args()
    base, adapters, lineage = preflight(args)
    if args.preflight_only:
        print(json.dumps({"base_model": str(base), "lineage": lineage, "adapters": adapters}, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    required = {"peft": "0.21.0", "transformers": "4.57.1"}
    mismatched = {name: version(name) for name, pin in required.items() if version(name) != pin}
    if mismatched:
        raise RuntimeError(f"Merge environment differs from pinned stack: {mismatched}")
    output = args.output_dir.resolve(strict=False)
    if not args.load_check_only:
        output.mkdir(parents=True, exist_ok=False)
        (output / "_INCOMPLETE").write_text("Do not evaluate this model yet.\n", encoding="utf-8")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        local_files_only=True,
    )
    for adapter in adapters:
        peft_model = PeftModel.from_pretrained(
            model, adapter["local_path"], is_trainable=False, local_files_only=True
        )
        loaded_lora = [
            name for name, _ in peft_model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]
        invalid = [name for name in loaded_lora if not is_language_attention_lora_name(name)]
        if invalid or len(loaded_lora) != adapter["tensor_count"]:
            raise RuntimeError(
                "PEFT created unexpected LoRA modules while reloading; aborting before merge. "
                f"loaded={len(loaded_lora)}, saved={adapter['tensor_count']}, "
                f"non_language={invalid[:5]}"
            )
        model = peft_model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("LoRA parameters remain after merge_and_unload")
    if args.load_check_only:
        print(json.dumps({"load_check": "passed", "lineage_sha256": lineage["lineage_sha256"]}, indent=2))
        return
    model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    processor.save_pretrained(output)
    files = verify_output(output)
    roles = [item["role"] for item in adapters]
    parent_kind = {
        ("sft",): "merged_sft",
        ("dpo",): "merged_base_dpo",
        ("grpo",): "merged_base_grpo",
        ("sft", "dpo"): "merged_sft_dpo",
        ("sft", "grpo"): "merged_sft_grpo",
        ("dpo", "grpo"): "merged_dpo_grpo",
    }[tuple(roles)]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(output),
        "merged_model_path": str(output),
        "parent_kind": parent_kind,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "base_snapshot_path": str(base),
        "lineage": lineage,
        "adapters": adapters,
        "source_adapter_path": adapters[-1]["local_path"],
        "adapter_sha256": adapters[-1]["weights_sha256"],
        "merge_method": "PEFT PeftModel.from_pretrained + merge_and_unload(safe_merge=True), sequential",
        "dtype": "bfloat16",
        "device": "cpu",
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "safetensors")},
        "files": files,
    }
    write_json(output / "model_manifest.json", manifest)
    (output / "_INCOMPLETE").unlink()
    print(json.dumps({"model_path": str(output), "lineage_sha256": lineage["lineage_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
