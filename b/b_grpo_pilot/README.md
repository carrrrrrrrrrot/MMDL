# B: Qwen3-VL GRPO pipeline pilot

This pilot checks the training path on **synthetic 4×4 colored-grid images**. It does not use MMMU or MMMU-Pro data and its score is not a benchmark result.

## Verified on the 6000 server (2026-11-17)

- Code: `/data/chaewon/mmdl_b_pilot_code`; training Python: `/data/chaewon/mmdl_envs/grpo_pilot/bin/python`.
- `strict_001` took one optimizer step but all four rewards were zero, so there was no GRPO learning signal and its LoRA weights did not change.
- `diagnostic_001` verified the weight-update path with an artificial position reward. Its adapter is **not** a benchmark candidate.
- `strict_012` completed 12 steps with the real multiple-choice correctness reward: 4 of 12 groups had mixed rewards, LoRA weights changed (`L1` delta about 821), and PEFT save/reload with image generation succeeded. The run summary is `/data/chaewon/mmdl_grpo_runs/strict_012/summary.json`. This verifies the training pipeline, not an MMMU improvement.

## Environment

Use a training environment separate from the v1.2 evaluation environment. The intended pins are TRL 1.14.2, PEFT 0.21.0, Transformers 4.57.1, and BF16 PyTorch with a CUDA 12.8 build. The first pilot uses `use_vllm=False`; it does not need training vLLM. Keep the model path at Qwen/Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.

## Run

```bash
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_pilot_code/test_reward.py
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_pilot_code/pilot.py \
  --model-path /home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output-dir /data/chaewon/mmdl_grpo_runs/new_strict_pilot \
  --reward-mode strict --max-steps 12
```

The output directory must be new. The script checks that the processor creates an image tensor, selects exactly 144 text-decoder attention projections, verifies that only language LoRA weights are trainable, takes an optimizer step, saves a PEFT adapter, reloads it into a fresh base model, and generates an image answer. Inspect `summary.json` and completion logs for reward variance, finite loss, clipping, VRAM, and LoRA weight change.

If all four completions for every prompt receive the same strict correctness reward, GRPO has no useful group signal. For a **plumbing-only** weight-update check, rerun in a new output directory with `--reward-mode diagnostic`. That mode assigns a deterministic position-based reward; its adapter is never an accuracy experiment or a candidate for MMMU evaluation.

The strict pilot reward accepts only a final line `Answer: A`, `B`, `C`, or `D`. It rejects lowercase, multiple answers, and an empty last `Answer:` line. This is a proposed **multiple-choice pilot policy**. R10–R18 numerical/unit rules and the R07/R09 differences from v1.2 require team agreement before main GRPO.

The dataset stores raw system/user messages plus an `image` column. TRL inserts the image through the processor; adding a literal `<image 1>` in user text would duplicate the placeholder.
