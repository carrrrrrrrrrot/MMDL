"""Validate and freeze one ECD500 EasyR1 run without starting Ray or GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.config import PPOConfig


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ENGINE_HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_train(train: Path, validation: Path) -> dict:
    rows = load_rows(train)
    if len(rows) != 500:
        raise ValueError(f"Expected 500 training rows, got {len(rows)}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != 500:
        raise ValueError("Training IDs are not unique")
    image_count = 0
    for row in rows:
        if row.get("system_prompt") != "You are a helpful assistant.":
            raise ValueError(f"v1.2 system prompt differs for {row['sample_id']}")
        paths = row.get("images")
        if not isinstance(paths, list) or not paths or row["prompt"].count("<image>") != len(paths):
            raise ValueError(f"Image markers/paths differ for {row['sample_id']}")
        for image in paths:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            image_count += 1
        gold = json.loads(row["answer"])
        if set(gold) != {"answer", "answer_type", "options", "sample_id"}:
            raise ValueError(f"Malformed reward gold for {row['sample_id']}")
        if str(gold["sample_id"]) != str(row["sample_id"]):
            raise ValueError(f"Reward ID mismatch for {row['sample_id']}")
    val_rows = load_rows(validation)
    if set(ids) & {str(row["sample_id"]) for row in val_rows}:
        raise ValueError("Train/validation ID overlap")
    return {"train_rows": 500, "train_unique_ids": 500, "train_images": image_count,
            "validation_rows": len(val_rows), "train_sha256": sha256(train),
            "validation_sha256": sha256(validation)}


def validate_parent(parent: Path, manifest_path: Path | None,
                    expected_dpo_sha: str | None) -> dict:
    parent = parent.resolve(strict=True)
    if manifest_path is None:
        if expected_dpo_sha is not None:
            raise ValueError("DPO adapter SHA requires a parent model manifest")
        if parent.name != BASE_REVISION or not (parent / "config.json").is_file():
            raise ValueError(f"Base revision differs: {parent}")
        return {"parent_kind": "base", "parent_model_path": str(parent),
                "parent_model_manifest_path": None, "parent_model_manifest_sha256": None,
                "parent_adapter_sha256": None}
    if not isinstance(expected_dpo_sha, str) or not SHA256.fullmatch(expected_dpo_sha):
        raise ValueError("Merged DPO parent requires the full adapter SHA256")
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.parent != parent or (parent / "_INCOMPLETE").exists():
        raise ValueError("DPO parent manifest must reside in a complete merged model")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or
            manifest.get("parent_kind") != "merged_base_dpo" or
            manifest.get("base_revision") != BASE_REVISION or
            Path(manifest.get("model_path", "")).resolve() != parent):
        raise ValueError("DPO parent manifest does not match the pinned merged model")
    adapters = manifest.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1 or (
            adapters[0].get("role"), adapters[0].get("weights_sha256")) != (
            "dpo", expected_dpo_sha):
        raise ValueError("DPO parent adapter lineage differs from the handoff SHA256")
    if not (parent / "config.json").is_file() or not list(parent.glob("*.safetensors")):
        raise ValueError("DPO parent model files are missing")
    return {"parent_kind": "merged_dpo", "parent_model_path": str(parent),
            "parent_model_manifest_path": str(manifest_path),
            "parent_model_manifest_sha256": sha256(manifest_path),
            "parent_adapter_sha256": expected_dpo_sha}


def make_config(source: Path, method: str, mode: str, run_dir: Path,
                stage1: Path, parent: Path):
    config = OmegaConf.load(source)
    config.worker.actor.model.model_path = str(parent)
    if method == "grpo":
        config.algorithm.online_filtering = False
        config.worker.actor.loss_avg_mode = "seq"
        config.worker.actor.clip_ratio_high = 0.2
    else:
        config.algorithm.online_filtering = True
        config.worker.actor.loss_avg_mode = "token"
        config.worker.actor.clip_ratio_high = 0.28
    config.worker.actor.clip_ratio_low = 0.2
    config.algorithm.filter_key = "accuracy"
    config.data.train_files = str(stage1 / "scripts/ecd500_easyr1.jsonl")
    config.data.val_files = str(stage1 / ("scripts/ecd5_holdout_easyr1.jsonl" if mode == "probe" else
                                              "scripts/ecd97_holdout_easyr1.jsonl"))
    config.data.val_batch_size = 5 if mode == "probe" else 8
    config.worker.reward.reward_function = str(stage1 / "scripts/reward_v12.py") + ":compute_score"
    config.trainer.experiment_name = f"ecd500_{method}_{mode}_cap1024"
    config.trainer.save_checkpoint_path = str(run_dir / "checkpoints")
    config.trainer.max_steps = 1 if mode == "probe" else None
    config.trainer.total_epochs = 3
    config.trainer.save_freq = 1 if mode == "probe" else 3
    config.trainer.save_model_only = mode == "probe"
    config.trainer.find_last_checkpoint = mode == "full"
    return config


def verify_config(config, mode: str, method: str, parent: Path) -> dict:
    merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), config)
    parsed: PPOConfig = OmegaConf.to_object(merged)
    parsed.deep_post_init()
    d = parsed.data
    a = parsed.worker.actor
    r = parsed.worker.rollout
    t = parsed.trainer
    if Path(a.model.model_path).resolve(strict=True) != parent:
        raise ValueError("Actor model path differs from the checked parent")
    if not (d.rollout_batch_size == 100 and d.mini_rollout_batch_size == 50 and
            a.global_batch_size == 100 and a.ppo_epochs == 1 and r.n == 5 and
            d.max_response_length == 1024 and d.max_prompt_length == 8192 and
            d.seed == 3407 and d.filter_overlong_prompts is False):
        raise ValueError("Batch/epoch/seed/length config differs from Stage2 contract")
    if not (a.optim.lr == 1e-6 and a.optim.lr_scheduler_type == "constant" and
            a.fsdp.use_orig_params is False and a.model.freeze_vision_tower is False and
            a.model.lora.rank == 16 and a.model.lora.exclude_modules == ".*visual.*"):
        raise ValueError("Optimizer/FSDP/LoRA contract differs")
    if not (parsed.algorithm.disable_kl and not parsed.algorithm.use_kl_loss and
            parsed.algorithm.filter_key == "accuracy" and t.total_epochs == 3 and
            t.max_try_make_batch == 24):
        raise ValueError("Reward/DAPO/epoch contract differs")
    if parsed.algorithm.online_filtering != (method == "dapo_core"):
        raise ValueError("Method filter differs")
    if t.max_steps != (1 if mode == "probe" else None):
        raise ValueError("Probe/full step setting differs")
    if t.find_last_checkpoint != (mode == "full"):
        raise ValueError("Checkpoint resume setting differs")
    if parsed.worker.reward.reward_function_name != "compute_score":
        raise ValueError("Reward callback differs")
    if parsed.worker.reward.reward_function_kwargs["reward_mode"] != "v12_binary":
        raise ValueError("Optimized reward is not raw v1.2 binary")
    loader_batches = 500 // d.mini_rollout_batch_size
    outer_per_epoch = (loader_batches * d.mini_rollout_batch_size) // d.rollout_batch_size
    nominal_outer = outer_per_epoch * t.total_epochs
    if (loader_batches, nominal_outer) != (10, 15):
        raise ValueError("Unexpected drop-last or training-step arithmetic")
    return {"dataloader_batches_per_epoch": loader_batches,
            "rollout_cycles_per_epoch": outer_per_epoch,
            "nominal_rollout_cycles_full": nominal_outer,
            "expected_optimizer_steps_full": nominal_outer,
            "groups_per_draw": 50, "rollouts_per_draw": 250,
            "groups_per_update": 100, "rollouts_per_update": 500,
            "max_dapo_draws_per_update": t.max_try_make_batch,
            "max_dapo_attempted_rollouts_per_update": 50 * r.n * t.max_try_make_batch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("grpo", "dapo_core"))
    parser.add_argument("mode", choices=("probe", "full"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--parent-manifest", type=Path,
                        help="Completed base→DPO merged model manifest")
    parser.add_argument("--expected-dpo-adapter-sha256",
                        help="A's original DPO(e) adapter_model.safetensors SHA256")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}")
    parent = validate_parent(args.base, args.parent_manifest,
                             args.expected_dpo_adapter_sha256)
    head = subprocess.check_output(["git", "-C", str(args.engine), "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_HEAD:
        raise ValueError(f"EasyR1 commit differs: {head}")
    stage1_manifest = args.stage1 / "manifest.json"
    if sha256(stage1_manifest) != "fac0d78cf11f79b7ce63de371c043555a68b8194ee449b1cd1e6ea6bd22f39bd":
        raise ValueError("Frozen Stage1 patch/data manifest differs")
    config = make_config(args.source_config, args.method, args.mode, args.run_dir,
                         args.stage1, args.base.resolve(strict=True))
    arithmetic = verify_config(config, args.mode, args.method,
                               args.base.resolve(strict=True))
    train = Path(config.data.train_files)
    validation = Path(config.data.val_files)
    data = validate_train(train, validation)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    resolved = args.run_dir / "resolved_config.yaml"
    OmegaConf.save(config, resolved)
    manifest = {
        "status": "preflight_passed",
        "method": args.method, "mode": args.mode,
        "algorithm_label": "DAPO core (soft-overlong disabled)" if args.method == "dapo_core" else "GRPO",
        "base_revision": BASE_REVISION, "base_model": parent["parent_model_path"],
        **parent,
        "engine_git_head": head, "stage1_manifest_sha256": sha256(stage1_manifest),
        "source_config_sha256": sha256(args.source_config),
        "resolved_config_sha256": sha256(resolved),
        "train_file": str(train), "validation_file": str(validation),
        "reward_v12_sha256": sha256(args.stage1 / "scripts/reward_v12.py"),
        "v12_scorer_sha256": sha256(args.stage1 / "scripts/v12_scorer.py"),
        "data": data, "arithmetic": arithmetic,
        "quality_note": "ECD500 is exploratory B QC v2; C MMMU leakage audit passed, full gold approval not claimed.",
        "epoch_note": "GRPO should visit each of 500 IDs three times if 15 cycles complete; DAPO may redraw and revisit IDs under online filtering.",
    }
    path = args.run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(args.run_dir), "manifest_sha256": sha256(path),
                      "data": data, "arithmetic": arithmetic}, ensure_ascii=False))


if __name__ == "__main__":
    main()
