"""Build a checked EasyR1 training record for B's existing fixed-300 evaluator.

This runs after training and PEFT adapter export. It does not merge a model or
start evaluation. The resulting JSON supplies the three legacy lineage fields
read by run_fixed300.py while retaining the exact EasyR1 inputs and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EASYR1_COMMIT = "a73346d028ba0335aa42da104972c86e6cd0fe69"
LORA_KEY = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.lora_[AB](?:\.[^.]+)?\.weight$"
)
LORA_SLOT = re.compile(
    r"language_model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$"
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_record(path: Path, expected: str | None = None) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def expected_file(recorded_path: str | None, recorded_sha: str | None,
                  path: Path, label: str) -> dict:
    """Check a run-record path and hash against its physical input."""
    if label in {"train data", "validation data"} and recorded_path is None:
        raise ValueError(f"Run manifest lacks {label} path")
    if recorded_path is not None and Path(recorded_path).resolve(strict=True) != path.resolve(strict=True):
        raise ValueError(f"Run manifest {label} differs from supplied file")
    if not isinstance(recorded_sha, str) or not SHA256.fullmatch(recorded_sha):
        raise ValueError(f"Run manifest lacks {label} SHA256")
    return file_record(path, recorded_sha)


def inspect_safetensors(path: Path) -> tuple[int, int]:
    """Read only the safetensors header to validate the LoRA key/shape scope."""
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if header_size < 2 or header_size > 4 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != 288 or any(not LORA_KEY.search(key) for key in tensors):
        raise ValueError("Expected exactly 288 language q/k/v/o LoRA tensors")
    slots = {LORA_SLOT.search(key).groups() for key in tensors}
    expected_slots = {(str(layer), module, side) for layer in range(36)
                      for module in ("q_proj", "k_proj", "v_proj", "o_proj")
                      for side in ("A", "B")}
    if slots != expected_slots:
        raise ValueError("Language LoRA layer/projection coverage differs")
    numel = 0
    offsets: list[tuple[int, int]] = []
    for key, value in tensors.items():
        if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
            raise ValueError(f"Invalid tensor metadata: {key}")
        count = 1
        for size in value["shape"]:
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"Invalid tensor shape: {key}")
            count *= size
        numel += count
        width = {"F32": 4, "BF16": 2}.get(value.get("dtype"))
        span = value.get("data_offsets")
        if width is None or not isinstance(span, list) or len(span) != 2 or not all(
            isinstance(offset, int) for offset in span
        ) or span[1] - span[0] != count * width:
            raise ValueError(f"Invalid tensor dtype/offsets: {key}")
        offsets.append((span[0], span[1]))
    if numel != 11_796_480:
        raise ValueError(f"Unexpected language LoRA parameter count: {numel}")
    position = 0
    for start, end in sorted(offsets):
        if start != position or end <= start:
            raise ValueError("Safetensors data offsets have gaps or overlap")
        position = end
    if position != path.stat().st_size - 8 - header_size:
        raise ValueError("Safetensors data length differs from header")
    return len(tensors), numel


def build(args: argparse.Namespace) -> dict:
    source_path = args.run_manifest.resolve(strict=True)
    completion_path = args.completion_report.resolve(strict=True)
    source = read_json(source_path)
    completion = read_json(completion_path)
    if source.get("status") != "completed" or source.get("process_exit_code") != 0:
        raise ValueError("EasyR1 process did not exit successfully")
    probe_check = bool(getattr(args, "probe_check", False))
    mode = source.get("mode")
    if mode != ("probe" if probe_check else "full"):
        raise ValueError("Expected a probe for --probe-check, otherwise a full Stage2 run")
    if completion.get("status") != "complete":
        raise ValueError("EasyR1 run is not recorded complete")
    if completion.get("mode") != mode:
        raise ValueError("Run and completion modes differ")
    method = source.get("method")
    if method not in {"grpo", "dapo_core"}:
        raise ValueError(f"Unexpected EasyR1 method: {method!r}")
    if completion.get("method", method) != method:
        raise ValueError("Run and completion methods differ")
    parent = source.get("base_model")
    if not isinstance(parent, str) or not parent.rstrip("/").endswith(BASE_REVISION):
        raise ValueError("EasyR1 run did not start from the pinned base snapshot")
    if source.get("base_revision", BASE_REVISION) != BASE_REVISION:
        raise ValueError("EasyR1 base revision differs")
    if source.get("parent_kind", "base") != "base":
        raise ValueError("Stage2 fixed300 lineage requires a base parent")
    if source.get("engine_git_head") != EASYR1_COMMIT:
        raise ValueError("EasyR1 source commit differs")
    if not isinstance(source.get("stage1_manifest_sha256"), str) or not SHA256.fullmatch(
        source["stage1_manifest_sha256"]
    ):
        raise ValueError("Missing frozen Stage1 package manifest SHA256")

    dataset = source.get("data")
    expected_validation_rows = 5 if probe_check else 97
    if not isinstance(dataset, dict) or dataset.get("train_unique_ids") != 500 or dataset.get("validation_rows") != expected_validation_rows:
        raise ValueError("Stage2 training/heldout data counts differ from ECD500 plan")
    if completion.get("validation_rows") != expected_validation_rows:
        raise ValueError("Run and completion heldout counts differ")
    train = expected_file(source.get("train_file"), dataset.get("train_sha256"),
                          args.train_file, "train data")
    validation = expected_file(source.get("validation_file"), dataset.get("validation_sha256"),
                               args.validation_file, "validation data")
    config = expected_file(source.get("resolved_config_file"), source.get("resolved_config_sha256"),
                           args.config_file, "resolved config")
    reward = expected_file(source.get("reward_file"), source.get("reward_v12_sha256"),
                           args.reward_file, "v1.2 reward code")
    source_config = expected_file(source.get("source_config_file"), source.get("source_config_sha256"),
                                  args.source_config_file, "source config")

    adapter_dir = args.adapter_dir.resolve(strict=True)
    if not adapter_dir.is_dir():
        raise ValueError(f"Missing PEFT adapter directory: {adapter_dir}")
    adapter = file_record(adapter_dir / "adapter_model.safetensors")
    adapter_config = file_record(adapter_dir / "adapter_config.json")
    peft = read_json(adapter_dir / "adapter_config.json")
    if peft.get("peft_type") != "LORA" or peft.get("task_type") not in {None, "CAUSAL_LM"}:
        raise ValueError("Adapter is not a causal-LM PEFT LoRA")
    if peft.get("r") != 16 or peft.get("lora_alpha") != 32:
        raise ValueError("Adapter rank/alpha differs from Stage2 contract")
    if set(peft.get("target_modules") or []) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("Adapter LoRA targets differ from language q/k/v/o")
    if peft.get("exclude_modules") != ".*visual.*":
        raise ValueError("Adapter does not exclude vision LoRA")
    tensor_count, parameter_count = inspect_safetensors(adapter_dir / "adapter_model.safetensors")
    completion_sha = completion.get("adapter_sha256")
    if completion_sha != adapter["sha256"]:
        raise ValueError("Completion report and saved adapter SHA256 differ")
    if completion.get("run_manifest_sha256", sha256(source_path)) != sha256(source_path):
        raise ValueError("Completion report names a different run manifest")
    if completion.get("resolved_config_sha256") != config["sha256"]:
        raise ValueError("Completion report names a different resolved config")
    if completion.get("train_sha256") != train["sha256"]:
        raise ValueError("Completion report names different training data")
    reported_adapter_dir = completion.get("adapter_dir", str(adapter_dir))
    if Path(reported_adapter_dir).resolve(strict=True) != adapter_dir:
        raise ValueError("Completion report names a different adapter directory")
    steps = completion.get("optimizer_steps")
    expected_steps = 1 if probe_check else source.get("arithmetic", {}).get("expected_optimizer_steps_full")
    if not isinstance(steps, int) or steps != expected_steps or steps != (1 if probe_check else 15):
        raise ValueError(f"Optimizer steps differ from full Stage2 plan: {steps} != {expected_steps}")
    positive_steps = completion.get("positive_grad_steps")
    if not isinstance(positive_steps, int) or not 1 <= positive_steps <= steps:
        raise ValueError("No observed positive gradient step")
    visit_audit = expected_file(None, completion.get("visit_audit_sha256"),
                                source_path.parent / "visit_audit.json", "visit audit")

    payload = {
        "schema": "mmdl_b_easyr1_fixed300_lineage_v1",
        "status": "probe_validated" if probe_check else "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "parent_kind": "base",
        "parent_model_path": parent,
        "algorithm": method,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": adapter["sha256"],
        "adapter_config_sha256": adapter_config["sha256"],
        "adapter_tensor_count": tensor_count,
        "adapter_parameter_count": parameter_count,
        "optimizer_steps": steps,
        "provenance": {
            "easy_r1_git_head": EASYR1_COMMIT,
            "stage1_package_manifest_sha256": source.get("stage1_manifest_sha256"),
            "run_manifest": file_record(source_path),
            "completion_report": file_record(completion_path),
            "visit_audit": visit_audit,
            "train_data": train,
            "validation_data": validation,
            "training_config": config,
            "source_config": source_config,
            "reward_code": reward,
            "adapter_weights": adapter,
            "adapter_config": adapter_config,
            "run_settings": source,
            "completion": completion,
        },
    }
    if args.model_manifest is not None:
        if probe_check:
            raise ValueError("Probe cannot be paired with a merged evaluation model")
        merged_path = args.model_manifest.resolve(strict=True)
        merged = read_json(merged_path)
        if merged.get("status") != "complete" or merged.get("base_revision") != BASE_REVISION:
            raise ValueError("Merged model is not a completed pinned-base model")
        adapters = merged.get("adapters")
        if not isinstance(adapters, list) or len(adapters) != 1:
            raise ValueError("Expected one base-parent RL adapter in merged model")
        if adapters[0].get("role") != "grpo" or adapters[0].get("weights_sha256") != adapter["sha256"]:
            raise ValueError("Merged model LoRA lineage differs from EasyR1 output")
        payload["provenance"]["merged_model_manifest"] = file_record(merged_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--completion-report", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--source-config-file", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--model-manifest", type=Path, help="Optional merged model lineage check")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-check", action="store_true",
                        help="Validate a completed one-step probe without creating an eval lineage")
    args = parser.parse_args()
    if args.probe_check and not args.dry_run:
        parser.error("--probe-check requires --dry-run")
    payload = build(args)
    target = args.output.resolve(strict=False)
    if args.dry_run:
        print(json.dumps({"status": payload["status"], "output": str(target),
                          "adapter_sha256": payload["adapter_sha256"],
                          "optimizer_steps": payload["optimizer_steps"]}, indent=2))
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite lineage manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": sha256(target)}, indent=2))


if __name__ == "__main__":
    main()
