"""Small, dependency-free checks shared by B's merge and evaluation commands."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EVAL_REPO_COMMIT = "61025850563d0331e365bc974a63c0d3c4ed4c2c"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LORA_KEY = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.lora_[AB](?:\.[^.]+)?\.weight$"
)


def is_language_attention_lora_name(name: str) -> bool:
    return _LORA_KEY.search(name) is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Unsafe run/role name: {value!r}")
    return value


def fixed_base_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.name != BASE_REVISION:
        raise ValueError(
            "Base must be the local Hugging Face snapshot directory ending in "
            f"{BASE_REVISION}: {resolved}"
        )
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(resolved / "config.json")
    config = read_json(resolved / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Expected qwen3_vl model_type, got {config.get('model_type')!r}")
    return resolved


def adapter_info(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    safe_name(role)
    weights = resolved / ADAPTER_WEIGHTS
    config_path = resolved / "adapter_config.json"
    if not weights.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"PEFT save_pretrained files missing in {resolved}")
    actual = sha256_file(weights)
    expected = expected_sha256.lower().strip()
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or actual != expected:
        raise ValueError(f"Adapter SHA256 mismatch: {resolved}, expected={expected}, actual={actual}")
    config = read_json(config_path)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Expected LoRA adapter: {resolved}")
    if config.get("task_type") not in {"CAUSAL_LM", None}:
        raise ValueError(f"Unexpected PEFT task_type: {config.get('task_type')!r}")
    # Checking actual weight names catches accidental vision/connector training,
    # even when adapter_config.json lists only generic target-module suffixes.
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt", device="cpu") as tensors:
        names = list(tensors.keys())
    invalid = [name for name in names if not is_language_attention_lora_name(name)]
    if invalid:
        raise ValueError(f"Non-language-attention LoRA tensors in {resolved}: {invalid[:5]}")
    if not names or len(names) % 2:
        raise ValueError(f"Unexpected adapter tensor count {len(names)} in {resolved}")
    return {
        "role": role,
        "local_path": str(resolved),
        "weights_file": ADAPTER_WEIGHTS,
        "weights_sha256": actual,
        "adapter_config_sha256": sha256_file(config_path),
        "tensor_count": len(names),
        "peft_type": config["peft_type"],
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "declared_base_model": config.get("base_model_name_or_path"),
    }


def ordered_lineage(adapters: list[dict[str, Any]]) -> dict[str, Any]:
    slim = {
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "adapters": [
            {"role": item["role"], "weights_sha256": item["weights_sha256"]}
            for item in adapters
        ],
    }
    return {**slim, "lineage_sha256": sha256_json(slim)}
