"""Single-GPU Qwen3-VL GRPO from either fixed base or verified merged SFT parent."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from data import file_sha256, load_manifest
from reward import RewardPolicy, make_grpo_reward
from run_reward_cases import run_cases


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
REQUIRED_VERSIONS = {"trl": "1.14.2", "peft": "0.21.0", "transformers": "4.57.1"}
_LANGUAGE_PROJ = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path, help="Validated JSONL; not MMMU")
    parser.add_argument("--train-split", default="train", help="Only this split is used for training")
    parser.add_argument("--reward-policy", required=True, type=Path, help="Explicit, versioned reward JSON")
    parser.add_argument("--parent-kind", required=True, choices=("base", "merged_sft"))
    parser.add_argument("--parent-model", required=True, type=Path, help="Local base snapshot or merged SFT model")
    parser.add_argument("--parent-manifest", type=Path, help="merge_lora.py model_manifest.json for merged_sft")
    parser.add_argument("--expected-sft-adapter-sha256", help="A's transferred adapter_model.safetensors SHA256")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--max-completion-length", default=256, type=int)
    parser.add_argument("--save-steps", default=25, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--validate-only", action="store_true", help="Check data and lineage without loading torch")
    return parser.parse_args()


def validate_parent(args: argparse.Namespace) -> dict:
    model_path = args.parent_model.resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError(f"Parent model is not a directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Parent model has no config.json: {model_path}")
    parent = {
        "base_revision": BASE_REVISION,
        "parent_kind": args.parent_kind,
        "parent_model_path": str(model_path),
        "parent_adapter_sha256": None,
        "parent_model_manifest_path": None,
        "parent_model_manifest_sha256": None,
    }
    if args.parent_kind == "base":
        if BASE_REVISION not in str(model_path):
            raise ValueError(f"Base path must contain fixed snapshot revision {BASE_REVISION}")
        if args.parent_manifest or args.expected_sft_adapter_sha256:
            raise ValueError("Base parent must not have SFT adapter arguments")
    else:
        if not args.parent_manifest or not args.expected_sft_adapter_sha256:
            raise ValueError("merged_sft requires parent manifest and expected SFT adapter SHA256")
        manifest_path = args.parent_manifest.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = args.expected_sft_adapter_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("Expected SFT adapter SHA256 must be 64 lowercase hex digits")
        if manifest.get("base_revision") != BASE_REVISION:
            raise ValueError("Merged parent base revision differs from fixed revision")
        if manifest.get("parent_kind") != "merged_sft":
            raise ValueError("Merged parent manifest must have parent_kind=merged_sft")
        if str(Path(manifest.get("merged_model_path", "")).resolve()) != str(model_path):
            raise ValueError("Merged parent manifest points to a different model directory")
        if manifest.get("adapter_sha256", "").lower() != expected_sha:
            raise ValueError("Merged parent adapter SHA256 differs from A's handoff")
        parent.update(
            parent_adapter_sha256=expected_sha,
            parent_model_manifest_path=str(manifest_path),
            parent_model_manifest_sha256=file_sha256(manifest_path),
        )
    return parent


def targets(model, torch) -> list[str]:
    names = [
        name
        for name, module in model.named_modules()
        if _LANGUAGE_PROJ.search(name) and isinstance(module, torch.nn.Linear)
    ]
    if len(names) != 144 or len(set(names)) != 144:
        raise RuntimeError(f"Expected 144 unique language attention LoRA targets; got {len(names)}")
    return names


def trainable_snapshot(model, torch) -> tuple[dict, int]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [
        name for name, _ in trainable
        if ".language_model.layers." not in name
        or (".lora_A." not in name and ".lora_B." not in name)
    ]
    if invalid:
        raise RuntimeError(f"Only language attention LoRA may train; found {invalid[:8]}")
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != 11_796_480:
        raise RuntimeError(f"Unexpected trainable parameter count {count}")
    snapshot = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
    return snapshot, count


def weight_delta(model, initial: dict) -> float:
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if current.keys() != initial.keys():
        raise RuntimeError("Trainable parameter names changed during training")
    return sum(
        (current[name].detach().float().cpu() - initial[name]).abs().sum().item()
        for name in initial
    )


def prepare_images_v12(dataset, processor):
    """Apply the frozen v1.2 Qwen image resize before the TRL processor."""
    from qwen_vl_utils import fetch_image

    image_mode = "images" if "images" in dataset.column_names else "image"

    def resize(row):
        images = row["images"] if image_mode == "images" else [row["image"]]
        resized = [
            fetch_image({"image": image.convert("RGB")}, image_patch_size=16)
            for image in images
        ]
        return {image_mode: resized if image_mode == "images" else resized[0]}

    dataset = dataset.map(resize, desc="Apply v1.2 Qwen image resize")
    processor.image_processor.size = {"shortest_edge": 4096, "longest_edge": 16777216}
    return dataset


def run_training(args: argparse.Namespace, rows: list[dict], policy: RewardPolicy) -> dict:
    import torch
    from datasets import Dataset, Image as DatasetImage, Sequence
    from peft import LoraConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import prepare_multimodal_messages

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    wrong = {name: (version(name), expected) for name, expected in REQUIRED_VERSIONS.items() if version(name) != expected}
    if wrong:
        raise RuntimeError(f"Training stack differs from pinned versions: {wrong}")

    model_path = args.parent_model.resolve(strict=True)
    dataset = Dataset.from_list(rows)
    if "images" in rows[0]:
        dataset = dataset.cast_column("images", Sequence(DatasetImage()))
    else:
        dataset = dataset.cast_column("image", DatasetImage())
    processor = AutoProcessor.from_pretrained(model_path)
    dataset = prepare_images_v12(dataset, processor)
    first = dataset[0]
    first_images = first["images"] if "images" in first else [first["image"]]
    messages = prepare_multimodal_messages(first["prompt"], images=first_images)
    encoded = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    if "pixel_values" not in encoded or encoded["pixel_values"].numel() == 0:
        raise RuntimeError("Processor failed to produce an image tensor")
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prompt_preflight = [f"=== {first['sample_id']} / {len(first_images)} image(s) ===\n{rendered}"]
    if "images" in first:
        max_index = max(range(len(rows)), key=lambda index: len(rows[index]["images"]))
        if max_index != 0:
            max_row = dataset[max_index]
            max_messages = prepare_multimodal_messages(max_row["prompt"], images=max_row["images"])
            max_encoded = processor.apply_chat_template(
                max_messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            if "pixel_values" not in max_encoded or max_encoded["pixel_values"].numel() == 0:
                raise RuntimeError("Multi-image processor preflight produced no pixels")
            max_rendered = processor.apply_chat_template(
                max_messages, add_generation_prompt=True, tokenize=False,
            )
            prompt_preflight.append(
                f"=== {max_row['sample_id']} / {len(max_row['images'])} image(s) ===\n{max_rendered}"
            )
    (args.output_dir / "prompt_preflight.txt").write_text("\n\n".join(prompt_preflight), encoding="utf-8")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_targets = targets(model, torch)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=lora_targets, task_type="CAUSAL_LM",
    )
    observed_rewards: list[float] = []
    config = GRPOConfig(
        output_dir=str(args.output_dir / "trainer"),
        seed=args.seed, data_seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        remove_unused_columns=False,
        temperature=0.7, top_p=0.8, top_k=20,
        logging_steps=1, log_completions=True,
        report_to=[],
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
    )
    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=dataset,
        reward_funcs=make_grpo_reward(policy, observed_rewards),
        processing_class=processor, peft_config=lora,
    )
    initial, trainable_count = trainable_snapshot(trainer.model, torch)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    delta = weight_delta(trainer.model, initial)
    finite_loss = math.isfinite(float(result.training_loss))
    groups = [observed_rewards[i:i + 4] for i in range(0, len(observed_rewards), 4)]
    mixed_groups = sum(len(set(group)) > 1 for group in groups)
    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir / "processor")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file() or not (adapter_dir / "adapter_config.json").is_file():
        raise RuntimeError("PEFT adapter save failed")

    metrics = {
        "global_step": trainer.state.global_step,
        "training_loss": float(result.training_loss),
        "finite_loss": finite_loss,
        "trainable_parameters": trainable_count,
        "lora_delta_l1": delta,
        "reward_groups": len(groups),
        "mixed_reward_groups": mixed_groups,
        "mean_reward": sum(observed_rewards) / len(observed_rewards) if observed_rewards else None,
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: version(name) for name in ("torch", "trl", "peft", "transformers", "datasets")},
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": file_sha256(adapter_file),
        "trainer_log_history": trainer.state.log_history,
    }
    if trainer.state.global_step < 1 or not finite_loss or delta <= 0 or mixed_groups == 0:
        raise RuntimeError("GRPO had no verified learning signal; inspect completion rewards and logs")

    del initial, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    fresh = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    reloaded = PeftModel.from_pretrained(fresh, adapter_dir).eval()
    with torch.inference_mode():
        generated = reloaded.generate(**encoded.to("cuda"), max_new_tokens=64, do_sample=False)
    metrics["reload_generated_text"] = processor.decode(
        generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
    )
    metrics["reload_success"] = True
    return metrics


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.max_completion_length < 16 or args.save_steps < 1:
        raise ValueError("max-steps/save-steps must be positive and completion length >=16")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    rows, dataset_info = load_manifest(args.dataset_manifest, args.train_split)
    policy_path = args.reward_policy.resolve(strict=True)
    policy = RewardPolicy.from_file(policy_path)
    parent = validate_parent(args)
    case_rows = run_cases(policy, "v12" if policy.mode == "v1_2" else "proposed")
    if policy.mode == "v1_2" and not all(row["pass"] for row in case_rows):
        raise RuntimeError("v1.2 reward failed the 20-case preflight")
    review_ids = {"R07", "R07-C", "R09", "R11", "R13", "R14"}
    metadata = {
        "status": "validated" if args.validate_only else "running",
        **parent,
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("train.py", "data.py", "reward.py", "v12_scorer.py", "run_reward_cases.py", "export_rollouts.py")
        },
        "dataset": dataset_info,
        "reward_policy_path": str(policy_path),
        "reward_policy_sha256": file_sha256(policy_path),
        "reward_policy_version": policy.version,
        "reward_case_v12_comparison": [
            {
                "case_id": row["case_id"],
                "answer_type": row["answer_type"],
                "actual_reward": row["actual_reward"],
                "v12_reward": row["v12_reward"],
                "differs_from_v12": row["differs_from_v12"],
                "used_by_training_data": row["answer_type"] in dataset_info["answer_types"],
            }
            for row in case_rows if row["case_id"] in review_ids
        ],
        "seed": args.seed,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_completion_length": args.max_completion_length,
        "save_steps": args.save_steps,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "targets": "144 language q/k/v/o attention projections"},
        "image_preprocessing": {
            "method": "qwen_vl_utils.fetch_image",
            "image_patch_size": 16,
            "min_pixels": 4096,
            "max_pixels": 16777216,
            "processor_size": {"shortest_edge": 4096, "longest_edge": 16777216},
        },
        "adapter_dir": None,
        "adapter_sha256": None,
    }
    if args.validate_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        metrics = run_training(args, rows, policy)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("export_rollouts.py")),
                "--run-dir", str(args.output_dir),
                "--policy", str(policy_path),
                "--dataset-manifest", str(args.dataset_manifest.resolve(strict=True)),
                "--max-completion-length", str(args.max_completion_length),
            ],
            check=True,
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    metadata.update(metrics)
    metadata["rollout_summary_path"] = str(args.output_dir / "rollout_summary.json")
    metadata["rollouts_sha256"] = file_sha256(args.output_dir / "rollouts.jsonl")
    metadata["status"] = "complete"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "status", "global_step", "mixed_reward_groups", "lora_delta_l1", "adapter_sha256", "reload_success"
    )}, indent=2))


if __name__ == "__main__":
    main()
