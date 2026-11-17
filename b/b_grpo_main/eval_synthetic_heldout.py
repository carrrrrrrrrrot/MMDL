"""Exploratory greedy base-vs-GRPO comparison on unseen synthetic images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data import file_sha256, load_manifest
from reward import RewardPolicy, score_response
from train import BASE_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--reward-policy", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--train-run-manifest", type=Path, help="Completed base→GRPO run_manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def provenance_for_split(path: Path, split: str) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] == split:
                if not item.get("task"):
                    raise ValueError(f"Missing task for {item['sample_id']}")
                result[item["sample_id"]] = item
    return result


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict], dict, RewardPolicy, dict | None]:
    if args.max_new_tokens < 16:
        raise ValueError("max-new-tokens must be at least 16")
    dataset_path = args.dataset_manifest.resolve(strict=True)
    rows, dataset_info = load_manifest(dataset_path, "heldout")
    provenance = provenance_for_split(dataset_path, "heldout")
    if set(provenance) != {row["sample_id"] for row in rows}:
        raise ValueError("Heldout provenance does not match validated sample IDs")
    if any(item["source"] != "synthetic_generated" for item in provenance.values()):
        raise ValueError("This exploratory evaluator accepts only generated synthetic data")
    policy = RewardPolicy.from_file(args.reward_policy.resolve(strict=True))
    base_model = args.base_model.resolve(strict=True)
    if BASE_REVISION not in str(base_model) or not (base_model / "config.json").is_file():
        raise ValueError("Base model is not the fixed Qwen3-VL revision snapshot")

    train_manifest = None
    if args.train_run_manifest:
        train_path = args.train_run_manifest.resolve(strict=True)
        train_manifest = json.loads(train_path.read_text(encoding="utf-8"))
        if train_manifest.get("status") != "complete":
            raise ValueError("Training run is not complete")
        if train_manifest.get("parent_kind") != "base" or train_manifest.get("base_revision") != BASE_REVISION:
            raise ValueError("This evaluator expects a fixed-base GRPO run")
        if Path(train_manifest.get("parent_model_path", "")).resolve() != base_model:
            raise ValueError("Training parent model differs from heldout base model")
        if train_manifest.get("dataset", {}).get("manifest_sha256") != file_sha256(dataset_path):
            raise ValueError("Heldout dataset manifest differs from training dataset manifest")
        if train_manifest.get("reward_policy_sha256") != file_sha256(args.reward_policy):
            raise ValueError("Reward policy differs from training")
        adapter_dir = Path(train_manifest.get("adapter_dir", "")).resolve(strict=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if file_sha256(adapter_file) != train_manifest.get("adapter_sha256"):
            raise ValueError("GRPO adapter SHA256 differs from training manifest")
        train_manifest["_resolved_adapter_dir"] = str(adapter_dir)
        train_manifest["_manifest_sha256"] = file_sha256(train_path)
    elif not args.validate_only:
        raise ValueError("--train-run-manifest is required for actual evaluation")

    info = {
        "purpose": "exploratory synthetic heldout comparison; not an MMMU result",
        "base_revision": BASE_REVISION,
        "base_model_path": str(base_model),
        "dataset": dataset_info,
        "heldout_tasks": {
            task: sum(item["task"] == task for item in provenance.values())
            for task in sorted({item["task"] for item in provenance.values()})
        },
        "reward_policy_version": policy.version,
        "reward_policy_sha256": file_sha256(args.reward_policy),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "train_run_manifest_sha256": train_manifest.get("_manifest_sha256") if train_manifest else None,
        "adapter_sha256": train_manifest.get("adapter_sha256") if train_manifest else None,
    }
    return rows, info, policy, train_manifest


def generate_one(model, processor, row: dict, max_new_tokens: int, torch) -> str:
    from PIL import Image
    from trl.data_utils import prepare_multimodal_messages

    with Image.open(row["image"]) as opened:
        image = opened.convert("RGB")
        messages = prepare_multimodal_messages(row["prompt"], images=[image])
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
    if "pixel_values" not in encoded or encoded["pixel_values"].numel() == 0:
        raise RuntimeError(f"Image processing failed for {row['sample_id']}")
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def run_pass(name: str, model, processor, rows: list[dict], policy: RewardPolicy, args: argparse.Namespace, provenance: dict, torch) -> dict:
    path = args.output_dir / f"{name}_outputs.jsonl"
    counts = defaultdict(lambda: {"total": 0, "correct": 0})
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            raw = generate_one(model, processor, row, args.max_new_tokens, torch)
            scored = score_response(raw, row["answer"], row["answer_type"], policy)
            source = provenance[row["sample_id"]]
            record = {
                "model": name,
                "sample_id": row["sample_id"],
                "original_id": source["original_id"],
                "task": source["task"],
                "gold_answer": row["answer"],
                "raw_output": raw,
                "parsed_answer": scored.parsed_answer,
                "reward": scored.reward,
                "reason": scored.reason,
                "finish_reason": "unavailable",
                "max_new_tokens": args.max_new_tokens,
                "reward_version": policy.version,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            counts[source["task"]]["total"] += 1
            counts[source["task"]]["correct"] += int(scored.reward)
            if index % 16 == 0:
                print(f"{name}: {index}/{len(rows)}", flush=True)
    return {"path": str(path), "sha256": file_sha256(path), "by_task": dict(counts)}


def read_outputs(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate heldout sample ID in {path}")
    return {row["sample_id"]: row for row in rows}


def summarize(args: argparse.Namespace, info: dict, base_info: dict, adapter_info: dict) -> dict:
    base = read_outputs(Path(base_info["path"]))
    adapter = read_outputs(Path(adapter_info["path"]))
    if set(base) != set(adapter):
        raise ValueError("Base and adapter heldout IDs differ")
    paired = []
    by_task = defaultdict(lambda: {"total": 0, "base_correct": 0, "adapter_correct": 0, "up": 0, "down": 0})
    for sample_id in sorted(base):
        left, right = base[sample_id], adapter[sample_id]
        if left["gold_answer"] != right["gold_answer"] or left["task"] != right["task"]:
            raise ValueError(f"Heldout target changed for {sample_id}")
        left_ok, right_ok = bool(left["reward"]), bool(right["reward"])
        task = left["task"]
        count = by_task[task]
        count["total"] += 1
        count["base_correct"] += int(left_ok)
        count["adapter_correct"] += int(right_ok)
        count["up"] += int(not left_ok and right_ok)
        count["down"] += int(left_ok and not right_ok)
        paired.append({
            "sample_id": sample_id,
            "task": task,
            "gold_answer": left["gold_answer"],
            "base_parsed_answer": left["parsed_answer"],
            "adapter_parsed_answer": right["parsed_answer"],
            "base_reward": left["reward"],
            "adapter_reward": right["reward"],
            "base_raw_output": left["raw_output"],
            "adapter_raw_output": right["raw_output"],
        })
    paired_path = args.output_dir / "paired_heldout.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    total = len(paired)
    base_correct = sum(row["base_reward"] for row in paired)
    adapter_correct = sum(row["adapter_reward"] for row in paired)
    summary = {
        **info,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "base_correct": int(base_correct),
        "adapter_correct": int(adapter_correct),
        "base_accuracy_pct": 100 * base_correct / total,
        "adapter_accuracy_pct": 100 * adapter_correct / total,
        "difference_pp": 100 * (adapter_correct - base_correct) / total,
        "by_task": dict(by_task),
        "base_outputs_sha256": base_info["sha256"],
        "adapter_outputs_sha256": adapter_info["sha256"],
        "paired_csv_sha256": file_sha256(paired_path),
        "finish_reason_available": False,
        "note": "Exploratory synthetic heldout only; not MMMU performance evidence.",
    }
    (args.output_dir / "heldout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    rows, info, policy, train_manifest = validate_inputs(args)
    if args.validate_only:
        print(json.dumps({"status": "validated", **info}, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    provenance = provenance_for_split(args.dataset_manifest, "heldout")
    base_info = run_pass("base", model, processor, rows, policy, args, provenance, torch)
    adapter = PeftModel.from_pretrained(model, train_manifest["_resolved_adapter_dir"]).eval()
    adapter_info = run_pass("grpo_adapter", adapter, processor, rows, policy, args, provenance, torch)
    summary = summarize(args, info, base_info, adapter_info)
    print(json.dumps({key: summary[key] for key in (
        "total", "base_correct", "adapter_correct", "difference_pp", "by_task"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
