"""Run A's exact one-adapter merger and write B-compatible lineage metadata.

Run once for base→SFT, then after B training run once for SFT→GRPO. This
wrapper never changes team_share/merge_lora.py or merges adapters itself.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b_eval_ops"))
from merge_lora import verify_output  # noqa: E402 - output validator only
from ops_common import (  # noqa: E402
    BASE_MODEL_ID,
    BASE_REVISION,
    adapter_info,
    fixed_base_path,
    ordered_lineage,
    read_json,
    sha256_file,
    write_json,
)


SHARED_SCRIPT_SHA256 = "f0289466a0874154062498b6f988dcd6422c5253e6f31d8237a6da861543c245"
SHARED_SCRIPT_COMMIT = "468f7bc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("sft", "grpo"))
    parser.add_argument("--shared-script", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parent-manifest", type=Path, help="Required for stage grpo")
    parser.add_argument("--train-run-manifest", type=Path, help="Required for stage grpo")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_shared_script(path: Path) -> tuple[Path, str]:
    resolved = path.resolve(strict=True)
    if resolved.name != "merge_lora.py" or resolved.parent.name != "team_share":
        raise ValueError("Use A's team_share/merge_lora.py, not B's draft merger")
    digest = sha256_file(resolved)
    if digest != SHARED_SCRIPT_SHA256:
        raise ValueError(f"A shared merge script changed: {digest}; review before use")
    return resolved, digest


def check_parent(base: Path, manifest_path: Path, train_path: Path, grpo_sha: str) -> tuple[dict, dict]:
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.parent != base or manifest_path.name != "model_manifest.json":
        raise ValueError("Parent manifest must be inside the merged SFT model directory")
    parent = read_json(manifest_path)
    if (parent.get("status") != "complete" or parent.get("parent_kind") != "merged_sft"
            or parent.get("base_revision") != BASE_REVISION
            or Path(parent.get("merged_model_path", "")).resolve() != base):
        raise ValueError("Parent manifest is not the fixed-revision merged SFT model")
    adapters = parent.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1 or adapters[0].get("role") != "sft":
        raise ValueError("SFT parent must have exactly one SFT adapter")
    sft_sha = adapters[0].get("weights_sha256")
    if parent.get("adapter_sha256") != sft_sha:
        raise ValueError("SFT parent adapter SHA lineage differs")
    merge_info = read_json(base / "MERGE_INFO.json")
    if merge_info.get("adapter_sha256", {}).get("adapter_model.safetensors") != sft_sha:
        raise ValueError("A's SFT MERGE_INFO disagrees with B parent manifest")
    train_path = train_path.resolve(strict=True)
    train = read_json(train_path)
    if (train.get("status") != "complete" or train.get("parent_kind") != "merged_sft"
            or train.get("base_revision") != BASE_REVISION
            or train.get("parent_adapter_sha256") != sft_sha
            or train.get("adapter_sha256") != grpo_sha
            or Path(train.get("parent_model_path", "")).resolve() != base
            or train.get("parent_model_manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("GRPO training manifest does not match this exact SFT parent and adapter")
    return parent, train


def check_merge_info(info: dict, base: Path, adapter: Path, adapter_sha: str) -> None:
    if (info.get("base_revision") != BASE_REVISION
            or Path(info.get("base_path", "")).resolve() != base
            or Path(info.get("adapter_path", "")).resolve() != adapter
            or info.get("adapter_sha256", {}).get("adapter_model.safetensors") != adapter_sha
            or info.get("dtype") != "bfloat16"):
        raise ValueError("A's MERGE_INFO does not match requested base/adapter/revision")


def preflight(args: argparse.Namespace) -> dict:
    script, script_sha = check_shared_script(args.shared_script)
    output = args.out.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    adapter = adapter_info(args.adapter, args.adapter_sha256, args.stage)
    adapter_path = Path(adapter["local_path"])
    if args.stage == "sft":
        if args.parent_manifest or args.train_run_manifest:
            raise ValueError("SFT parent stage must not have prior-parent or training manifest")
        base = fixed_base_path(args.base_model)
        adapters = [adapter]
    else:
        if not args.parent_manifest or not args.train_run_manifest:
            raise ValueError("GRPO stage needs parent and training manifests")
        base = args.base_model.resolve(strict=True)
        if not base.is_dir():
            raise NotADirectoryError(base)
        parent, _ = check_parent(base, args.parent_manifest, args.train_run_manifest, adapter["weights_sha256"])
        adapters = [*parent["adapters"], adapter]
    if output == base or base in output.parents or output in base.parents:
        raise ValueError("Output must be separate from its base model")
    if output == adapter_path or adapter_path in output.parents or output in adapter_path.parents:
        raise ValueError("Output must be separate from its adapter")
    if importlib.metadata.version("transformers") != "4.57.1" or importlib.metadata.version("peft") != "0.21.0":
        raise ValueError("Shared merge requires pinned transformers 4.57.1 and PEFT 0.21.0")
    return {
        "script": str(script), "script_sha256": script_sha,
        "base": str(base), "adapter": str(adapter_path),
        "adapter_sha256": adapter["weights_sha256"],
        "output": str(output), "stage": args.stage,
        "adapters": adapters, "lineage": ordered_lineage(adapters),
    }


def build_manifest(plan: dict, output: Path) -> dict:
    files = verify_output(output)
    merge_info = read_json(output / "MERGE_INFO.json")
    check_merge_info(merge_info, Path(plan["base"]), Path(plan["adapter"]), plan["adapter_sha256"])
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(output),
        "merged_model_path": str(output),
        "parent_kind": "merged_sft" if plan["stage"] == "sft" else "merged_sft_grpo",
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "base_snapshot_path": plan["base"] if plan["stage"] == "sft" else None,
        "lineage": plan["lineage"],
        "adapters": plan["adapters"],
        "source_adapter_path": plan["adapter"],
        "adapter_sha256": plan["adapter_sha256"],
        "merge_method": f"A team_share/merge_lora.py @ {SHARED_SCRIPT_COMMIT}; one adapter per invocation",
        "shared_merge_script_sha256": plan["script_sha256"],
        "dtype": "bfloat16",
        "device": "cpu",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
        "files": files,
    }


def main() -> None:
    args = parse_args()
    plan = preflight(args)
    if args.check_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output = Path(plan["output"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable, plan["script"],
        "--base", plan["base"],
        "--base_revision", BASE_REVISION,
        "--adapter", plan["adapter"],
        "--out", plan["output"],
    ]
    subprocess.run(command, env=env, check=True)
    manifest = build_manifest(plan, output)
    write_json(output / "model_manifest.json", manifest)
    print(json.dumps({"merged_model_path": str(output), "lineage_sha256": manifest["lineage"]["lineage_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
