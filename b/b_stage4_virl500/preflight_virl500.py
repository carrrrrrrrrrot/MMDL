"""CPU-only ViRL500 run gate: data, images, reward, and Stage2 config parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from verl.trainer.config import PPOConfig


BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ENGINE_HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
STAGE2_CONFIG_SHA256 = "737249c0016253b2b92c7f3d3f2e36691f62db4cf850a4e258ce1aafeec2e4a3"
STAGE3_SELECTION_SHA256 = "047b8d0cbab5d40fd2bfc5bd6f717c8b49cac33c5a1eb8c056c61be9148d45c3"
ANSWER_SUFFIX = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'
MC_ANSWER_SUFFIX = 'Think step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_data(data_root: Path, stage4: Path, stage1: Path, mode: str,
                  *, allow_pending_prompt_scan: bool = False) -> dict:
    """Recheck immutable data, exact selected QIDs, images, and gold self-score."""
    sys.path.insert(0, str(stage1 / "scripts"))
    from v12_scorer import score_v12  # type: ignore

    manifest_path = data_root / "data_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    mirror = data_root / "scripts/data_manifest.json"
    if mirror.is_file() and sha256(manifest_path) != sha256(mirror):
        raise ValueError("Stage4 root/scripts data manifest copies differ")
    train = Path(metadata["train_path"])
    validation = Path(metadata["validation_path"])
    probe = Path(metadata["probe_path"])
    for path in (train, validation, probe):
        if not path.resolve(strict=True).is_relative_to(data_root.resolve(strict=True)):
            raise ValueError(f"Training data path escaped frozen package: {path}")
    expected_files = {"train_sha256": train, "validation_sha256": validation,
                      "probe_sha256": probe}
    for key, path in expected_files.items():
        if metadata.get(key) != sha256(path):
            raise ValueError(f"ViRL data manifest {key} differs from {path}")
    if metadata.get("source_primary_sha256") != STAGE3_SELECTION_SHA256:
        raise ValueError("Corrected selection does not cite the pinned original primary500")
    if sha256(Path(metadata["source_primary_path"])) != STAGE3_SELECTION_SHA256:
        raise ValueError("Original Stage3 primary500 changed")
    selection = Path(metadata["selection_path"])
    if not selection.resolve(strict=True).is_relative_to(data_root.resolve(strict=True)):
        raise ValueError("Corrected selection escaped frozen data package")
    if metadata.get("selection_sha256") != sha256(selection):
        raise ValueError("Corrected selection SHA256 differs from data manifest")
    selected_rows = load_rows(selection)
    selected_qids = [str(row["qid"]) for row in selected_rows]
    if len(selected_qids) != 500 or len(set(selected_qids)) != 500:
        raise ValueError("Corrected selection is not exactly 500 unique QIDs")
    selected_by_qid = {str(row["qid"]): row for row in selected_rows}
    if not metadata.get("image_integrity_ok") or not metadata.get("gold_selfcheck_ok"):
        raise ValueError("ViRL data provenance did not pass image/gold checks")
    qc_status = str(metadata.get("provisional_qc_status", ""))
    if not any(word in qc_status.lower() for word in ("provisional", "exploratory")):
        raise ValueError("The exploratory, pre-C-review QC status is not recorded")
    if (not allow_pending_prompt_scan and
            not str(metadata.get("prompt_length_status", "")).lower().startswith("passed")):
        raise ValueError("ViRL prompt-length CPU scan has not passed the 8192-token gate")

    image_root = (stage4.parent / "mmdl_stage3_virl39k_20261121" / "extracted").resolve(strict=True)
    image_cache: set[Path] = set()
    data: dict[str, list[dict]] = {}
    image_sets: dict[str, set[Path]] = {}
    for name, path, expected in (("train", train, 500), ("validation", validation, 97),
                                 ("probe", probe, 5)):
        rows = load_rows(path)
        ids = [str(row["sample_id"]) for row in rows]
        qids = [str(row["source_qid"]) for row in rows]
        if len(rows) != expected or len(set(ids)) != expected or len(set(qids)) != expected:
            raise ValueError(f"{name} lacks {expected} unique IDs and QIDs")
        if metadata.get(f"{name}_rows") != expected:
            raise ValueError(f"{name} count differs from data manifest")
        if name == "train" and (metadata.get("train_unique_qids") != 500 or
                                set(qids) != set(selected_qids)):
            raise ValueError("Training QIDs differ from the corrected 500-ID selection")
        image_paths: set[Path] = set()
        for row in rows:
            sample_id = str(row["sample_id"])
            prompt = row.get("prompt")
            paths = row.get("images")
            if row.get("system_prompt") != "You are a helpful assistant.":
                raise ValueError(f"v1.2 system prompt differs for {sample_id}")
            if (not isinstance(prompt, str) or not prompt.startswith("Question: ") or
                    not prompt.endswith((ANSWER_SUFFIX, MC_ANSWER_SUFFIX)) or not isinstance(paths, list) or
                    not paths or prompt.count("<image>") != len(paths)):
                raise ValueError(f"Prompt/image markers differ from Stage2 for {sample_id}")
            gold = json.loads(row["answer"])
            if set(gold) != {"answer", "answer_type", "options", "sample_id"}:
                raise ValueError(f"Malformed v1.2 gold for {sample_id}")
            if str(gold["sample_id"]) != sample_id:
                raise ValueError(f"Reward ID mismatch for {sample_id}")
            answer = str(gold["answer"])
            kind = gold["answer_type"]
            options = gold["options"]
            if name == "train":
                source = selected_by_qid[str(row["source_qid"])]
                if (answer != str(source["v12_answer"]) or
                        kind != source["answer_type"] or options != source["options"] or
                        prompt != source["expected_prompt"]):
                    raise ValueError(f"Converted question/gold differs from selection for {sample_id}")
            if kind == "mc":
                if (not isinstance(options, list) or not 2 <= len(options) <= 10 or
                        any(not isinstance(option, str) or not option.strip() for option in options) or
                        len(answer) != 1 or answer not in "ABCDEFGHIJ"[:len(options)]):
                    raise ValueError(f"Malformed MC options/answer for {sample_id}")
            elif kind == "open":
                if options != [] or (len(answer) == 1 and answer in "ABCDEFGHIJ"):
                    raise ValueError(f"Ambiguous open/MC label for {sample_id}")
            else:
                raise ValueError(f"Unknown answer type for {sample_id}: {kind}")
            correct, _, _ = score_v12(f"Answer: {answer}", answer, kind, options)
            if not correct:
                raise ValueError(f"Gold self-score failed for {sample_id}")
            for image in paths:
                resolved = Path(image).resolve(strict=True)
                if image_root not in resolved.parents:
                    raise ValueError(f"Image escaped official extracted root: {resolved}")
                if resolved not in image_cache:
                    with Image.open(resolved) as picture:
                        picture.verify()
                    image_cache.add(resolved)
                image_paths.add(resolved)
        data[name] = rows
        image_sets[name] = image_paths
    train_ids = {row["source_qid"] for row in data["train"]}
    val_ids = {row["source_qid"] for row in data["validation"]}
    probe_ids = {row["source_qid"] for row in data["probe"]}
    if train_ids & val_ids or train_ids & probe_ids or not probe_ids <= val_ids:
        raise ValueError("Train/validation/probe QID split differs")
    if image_sets["train"] & image_sets["validation"]:
        raise ValueError("Train/validation image overlap")
    fingerprint_path = stage4.parent / "mmdl_stage3_virl39k_20261121" / "candidate_image_fingerprints.jsonl"
    image_manifest = metadata.get("image_integrity", {})
    if sha256(fingerprint_path) != image_manifest.get("fingerprints_sha256"):
        raise ValueError("Stage3 image fingerprint manifest changed")
    all_images = image_sets["train"] | image_sets["validation"]
    relative_images = {str(image.relative_to(image_root)): image for image in all_images}
    expected_hashes = {}
    expected_pixel_hashes = {}
    with fingerprint_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["path"] in relative_images:
                expected_hashes[item["path"]] = item["file_sha256"]
                expected_pixel_hashes[item["path"]] = item["pixel_sha256"]
    if set(expected_hashes) != set(relative_images):
        raise ValueError("Selected image absent from Stage3 fingerprint manifest")
    selected_digest = hashlib.sha256()
    for relative in sorted(relative_images):
        image = relative_images[relative]
        actual_hash = sha256(image)
        if actual_hash != expected_hashes[relative]:
            raise ValueError(f"ViRL image changed after selection: {relative}")
        selected_digest.update(f"{relative}\t{actual_hash}\n".encode("utf-8"))
    if selected_digest.hexdigest() != image_manifest.get("selected_fingerprints_sha256"):
        raise ValueError("Selected image fingerprint digest differs")
    train_pixels = {expected_pixel_hashes[str(image.relative_to(image_root))]
                    for image in image_sets["train"]}
    validation_pixels = {expected_pixel_hashes[str(image.relative_to(image_root))]
                         for image in image_sets["validation"]}
    if mode == "full" and len(train_pixels) != len(image_sets["train"]):
        raise ValueError("Pixel-identical image duplicated within ViRL500 training split")
    if mode == "full" and len(validation_pixels) != len(image_sets["validation"]):
        raise ValueError("Pixel-identical image duplicated within ViRL97 validation split")
    if mode == "full" and train_pixels & validation_pixels:
        raise ValueError(f"Train/validation pixel-identical image overlap: {len(train_pixels & validation_pixels)}")
    image_bytes = sum(image.stat().st_size for image in all_images)
    if (len(all_images) != image_manifest.get("unique_images") or
            image_bytes != image_manifest.get("image_file_bytes")):
        raise ValueError("Selected image count/bytes differ from manifest")
    result = {"train_rows": 500, "train_unique_ids": 500,
              "train_images": sum(len(row["images"]) for row in data["train"]),
              "train_unique_images": len(image_sets["train"]),
              "train_unique_pixels": len(train_pixels),
              "validation_unique_pixels": len(validation_pixels),
              "all_unique_images": len(all_images), "image_file_bytes": image_bytes,
              "train_validation_pixel_overlap": len(train_pixels & validation_pixels),
              "validation_rows": 97, "probe_rows": 5,
              "train_sha256": sha256(train), "validation_sha256": sha256(validation),
              "probe_sha256": sha256(probe), "selection_sha256": sha256(selection),
              "data_manifest_sha256": sha256(manifest_path),
              "provisional_qc_status": qc_status}
    return result


def verify_stage2_config_parity(source: Path, stage2_source: Path) -> str:
    """Only dataset and experiment labels may differ from frozen Stage2."""
    if sha256(stage2_source) != STAGE2_CONFIG_SHA256:
        raise ValueError("Frozen Stage2 source config SHA256 differs")
    original = OmegaConf.to_container(OmegaConf.load(stage2_source), resolve=True)
    candidate = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    for config in (original, candidate):
        config["data"]["train_files"] = "<DATA>"
        config["data"]["val_files"] = "<DATA>"
        config["trainer"]["project_name"] = "<PROJECT>"
        config["trainer"]["experiment_name"] = "<EXPERIMENT>"
    if original != candidate:
        raise ValueError("ViRL source config changes a Stage2 setting beyond data/labels")
    return sha256(stage2_source)


def make_config(source: Path, method: str, mode: str, run_dir: Path,
                stage1: Path, data_root: Path):
    config = OmegaConf.load(source)
    data_manifest = json.loads((data_root / "data_manifest.json").read_text(encoding="utf-8"))
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
    config.data.train_files = str(data_manifest["train_path"])
    config.data.val_files = str(data_manifest["probe_path"] if mode == "probe" else
                                data_manifest["validation_path"])
    config.data.val_batch_size = 5 if mode == "probe" else 8
    config.worker.reward.reward_function = str(stage1 / "scripts/reward_v12.py") + ":compute_score"
    config.trainer.experiment_name = f"virl500_{method}_{mode}_cap1024"
    config.trainer.save_checkpoint_path = str(run_dir / "checkpoints")
    config.trainer.max_steps = 1 if mode == "probe" else None
    config.trainer.total_epochs = 3
    config.trainer.save_freq = 1 if mode == "probe" else 3
    config.trainer.save_model_only = mode == "probe"
    config.trainer.find_last_checkpoint = mode == "full"
    return config


def verify_config(config, mode: str, method: str) -> dict:
    merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), config)
    parsed: PPOConfig = OmegaConf.to_object(merged)
    parsed.deep_post_init()
    d = parsed.data
    a = parsed.worker.actor
    r = parsed.worker.rollout
    t = parsed.trainer
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
    if (parsed.algorithm.online_filtering != (method == "dapo_core") or
            a.loss_avg_mode != ("token" if method == "dapo_core" else "seq") or
            a.clip_ratio_low != 0.2 or
            a.clip_ratio_high != (0.28 if method == "dapo_core" else 0.2)):
        raise ValueError("GRPO/DAPO method settings differ from Stage2")
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
    parser.add_argument("--stage2-config", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage4", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        help="Frozen ViRL data package; defaults to Stage4 root for the original probe")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Run CPU gates without creating a run directory")
    parser.add_argument("--diagnostic-ignore-pending-token-scan", action="store_true",
                        help="Check the remaining CPU gates while the external token scan is pending; requires --check-only")
    args = parser.parse_args()
    data_root = args.data_root or args.stage4
    if args.diagnostic_ignore_pending_token_scan and not args.check_only:
        parser.error("--diagnostic-ignore-pending-token-scan requires --check-only")
    if not args.check_only and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}")
    if args.base.name != BASE_REVISION:
        raise ValueError(f"Base revision differs: {args.base}")
    head = subprocess.check_output(["git", "-C", str(args.engine), "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_HEAD:
        raise ValueError(f"EasyR1 commit differs: {head}")
    stage1_manifest = args.stage1 / "manifest.json"
    if sha256(stage1_manifest) != "fac0d78cf11f79b7ce63de371c043555a68b8194ee449b1cd1e6ea6bd22f39bd":
        raise ValueError("Frozen Stage1 patch/data manifest differs")
    stage2_config_sha = verify_stage2_config_parity(args.source_config, args.stage2_config)
    config = make_config(args.source_config, args.method, args.mode, args.run_dir,
                         args.stage1, data_root)
    arithmetic = verify_config(config, args.mode, args.method)
    train = Path(config.data.train_files)
    validation = Path(config.data.val_files)
    data = validate_data(data_root, args.stage4, args.stage1, args.mode,
                         allow_pending_prompt_scan=args.diagnostic_ignore_pending_token_scan)
    if sha256(train) != data["train_sha256"] or sha256(validation) != (
            data["probe_sha256"] if args.mode == "probe" else data["validation_sha256"]):
        raise ValueError("Resolved train/validation paths differ from verified data")
    if args.check_only:
        check_status = ("cpu_diagnostic_pending_token_scan" if args.diagnostic_ignore_pending_token_scan
                        else "cpu_preflight_passed")
        print(json.dumps({"status": check_status, "method": args.method,
                          "mode": args.mode, "data": data, "arithmetic": arithmetic,
                          "stage2_config_sha256": stage2_config_sha}, ensure_ascii=False))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    resolved = args.run_dir / "resolved_config.yaml"
    OmegaConf.save(config, resolved)
    manifest = {
        "status": "preflight_passed",
        "method": args.method, "mode": args.mode,
        "algorithm_label": "DAPO core (soft-overlong disabled)" if args.method == "dapo_core" else "GRPO",
        "base_revision": BASE_REVISION, "base_model": str(args.base),
        "engine_git_head": head, "stage1_manifest_sha256": sha256(stage1_manifest),
        "stage2_source_config_sha256": stage2_config_sha,
        "source_config_sha256": sha256(args.source_config),
        "resolved_config_sha256": sha256(resolved),
        "train_file": str(train), "validation_file": str(validation),
        "data_root": str(data_root),
        "data_manifest_path": str(data_root / "data_manifest.json"),
        "data_manifest_sha256": data["data_manifest_sha256"],
        "selection_sha256": data["selection_sha256"],
        "reward_v12_sha256": sha256(args.stage1 / "scripts/reward_v12.py"),
        "v12_scorer_sha256": sha256(args.stage1 / "scripts/v12_scorer.py"),
        "data": data, "arithmetic": arithmetic,
        "quality_note": "ViRL500 is provisional pre-C-review; preliminary MMMU/MMMU-Pro leakage screens passed, but C manual quality/leakage approval is pending. Exploratory experiment only.",
        "epoch_note": "GRPO should use all 500 IDs exactly 3 times if 15 cycles complete; DAPO may repeatedly redraw and use fewer unique IDs under online filtering, as in Stage2 ECD500 (115/500 used).",
    }
    path = args.run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(args.run_dir), "manifest_sha256": sha256(path),
                      "data": data, "arithmetic": arithmetic}, ensure_ascii=False))


if __name__ == "__main__":
    main()
