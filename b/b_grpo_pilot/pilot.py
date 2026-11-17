"""Qwen3-VL single-GPU GRPO plumbing pilot on synthetic images.

No MMMU/MMMU-Pro examples are used. This script is a compatibility and update
check, not a claim that GRPO improves benchmark accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset, Image as DatasetImage
from peft import LoraConfig, PeftModel
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import prepare_multimodal_messages

from reward import diagnostic_position_reward, strict_mc_reward


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_LANGUAGE_PROJ = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)


def _make_grid(path: Path, red_count: int, seed: int) -> None:
    image = Image.new("RGB", (320, 320), "white")
    draw = ImageDraw.Draw(image)
    cells = list(range(16))
    random.Random(seed).shuffle(cells)
    red_cells = set(cells[:red_count])
    for cell in range(16):
        row, col = divmod(cell, 4)
        left, top = 12 + col * 76, 12 + row * 76
        fill = (210, 45, 45) if cell in red_cells else (45, 90, 185)
        draw.rectangle((left, top, left + 68, top + 68), fill=fill, outline="black", width=2)
    image.save(path)


def _dataset(image_dir: Path) -> Dataset:
    image_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, red_count in enumerate((2, 5, 8, 11, 4, 7, 10, 13, 3, 6, 9, 12)):
        path = image_dir / f"grid_{index:02d}.png"
        _make_grid(path, red_count, 3407 + index)
        values = [red_count - 1, red_count, red_count + 1, red_count + 2]
        shift = index % 4
        values = values[shift:] + values[:shift]
        answer = "ABCD"[values.index(red_count)]
        options = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", values, strict=True))
        question = (
            "Count the red squares in the image. Choose exactly one option.\n"
            f"{options}\n"
            "Explain briefly, then end with one line in this form: Answer: X"
        )
        rows.append(
            {
                "sample_id": f"synthetic_grid_{index:02d}",
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                "image": str(path),
                "answer": answer,
            }
        )
    dataset = Dataset.from_list(rows).cast_column("image", DatasetImage())
    assert len(dataset) == 12
    assert all(row["image"] is not None for row in dataset)
    return dataset


def _targets(model: torch.nn.Module) -> list[str]:
    targets = [
        name
        for name, module in model.named_modules()
        if _LANGUAGE_PROJ.search(name) and isinstance(module, torch.nn.Linear)
    ]
    if len(targets) != 144:
        raise RuntimeError(f"Expected 144 language attention projections, found {len(targets)}")
    if len(set(targets)) != len(targets):
        raise RuntimeError("Duplicate LoRA targets")
    return targets


def _trainable_snapshot(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], int]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [
        name
        for name, _ in trainable
        if ".language_model.layers." not in name or (".lora_A." not in name and ".lora_B." not in name)
    ]
    if invalid:
        raise RuntimeError(f"Non-language LoRA parameters are trainable: {invalid[:8]}")
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != 11_796_480:
        raise RuntimeError(f"Unexpected trainable parameter count: {count}")
    snapshot = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
    return snapshot, count


def _delta(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if current.keys() != initial.keys():
        raise RuntimeError("Trainable parameter names changed during training")
    return sum(
        (current[name].detach().float().cpu() - initial[name]).abs().sum().item() for name in initial
    )


def _image_preflight(processor: AutoProcessor, row: dict) -> list[dict]:
    messages = prepare_multimodal_messages(row["prompt"], images=[row["image"]])
    encoded = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" not in encoded or encoded["pixel_values"].numel() == 0:
        raise RuntimeError("Processor produced no image tensor")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Cached snapshot of the fixed base revision")
    parser.add_argument("--output-dir", required=True, type=Path, help="New empty run directory, preferably on /data")
    parser.add_argument("--reward-mode", choices=("strict", "diagnostic"), default="strict")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires BF16 support")
    required_versions = {"trl": "1.14.2", "peft": "0.21.0", "transformers": "4.57.1"}
    wrong_versions = {
        name: (version(name), required)
        for name, required in required_versions.items()
        if version(name) != required
    }
    if wrong_versions:
        raise RuntimeError(f"Training stack differs from the agreed pins: {wrong_versions}")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    model_path = Path(args.model_path).resolve(strict=True)
    if BASE_REVISION not in str(model_path):
        raise ValueError(f"Model path must contain fixed revision {BASE_REVISION}")

    dataset = _dataset(args.output_dir / "images")
    processor = AutoProcessor.from_pretrained(model_path)
    reload_messages = _image_preflight(processor, dataset[0])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    targets = _targets(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    observed_rewards: list[float] = []

    def reward_func(completions, **kwargs):
        if args.reward_mode == "strict":
            rewards = strict_mc_reward(completions, **kwargs)
        else:
            rewards = diagnostic_position_reward(completions, **kwargs)
        observed_rewards.extend(rewards)
        return rewards

    config = GRPOConfig(
        output_dir=str(args.output_dir / "trainer"),
        seed=3407,
        data_seed=3407,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=128,
        learning_rate=1e-4,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        remove_unused_columns=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        logging_steps=1,
        log_completions=True,
        report_to=[],
        save_strategy="no",
        dataloader_num_workers=0,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=processor,
        peft_config=lora,
    )
    initial, trainable_count = _trainable_snapshot(trainer.model)
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    delta = _delta(trainer.model, initial)
    finite_loss = math.isfinite(float(train_result.training_loss))

    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir / "processor")
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError("PEFT adapter was not saved")

    group_reward_variance = any(
        len(set(observed_rewards[start : start + 4])) > 1
        for start in range(0, len(observed_rewards), 4)
    )
    summary = {
        "purpose": "GRPO plumbing pilot; not an MMMU performance experiment",
        "reward_mode": args.reward_mode,
        "model_revision": BASE_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "datasets")},
        "global_step": trainer.state.global_step,
        "training_loss": train_result.training_loss,
        "finite_loss": finite_loss,
        "trainable_parameters": trainable_count,
        "lora_delta_l1": delta,
        "observed_rewards": observed_rewards,
        "reward_has_variance_within_group": group_reward_variance,
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "log_history": trainer.state.log_history,
        "adapter_dir": str(adapter_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if trainer.state.global_step < 1 or not finite_loss:
        raise RuntimeError("Training did not finish a finite optimizer step")

    del initial, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    fresh = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to("cuda")
    reloaded = PeftModel.from_pretrained(fresh, adapter_dir).eval()
    encoded = processor.apply_chat_template(
        reload_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = reloaded.generate(**encoded, max_new_tokens=64, do_sample=False)
    generated_text = processor.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    summary["reload_generated_text"] = generated_text
    summary["reload_success"] = True
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("global_step", "finite_loss", "lora_delta_l1", "reward_has_variance_within_group", "reload_success")}, indent=2))
    if delta <= 0 or not group_reward_variance:
        raise RuntimeError("No useful GRPO update signal; inspect reward outputs and retry with --reward-mode diagnostic for plumbing only")


if __name__ == "__main__":
    main()
