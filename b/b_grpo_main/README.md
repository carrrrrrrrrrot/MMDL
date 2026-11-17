# B GRPO training path

This directory contains the B-side training implementation for the fixed
`Qwen/Qwen3-VL-4B-Instruct` revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. It uses TRL 1.14.2,
PEFT 0.21.0, Transformers 4.57.1, BF16, and language-only LoRA
`r=16, alpha=32` on 144 q/k/v/o attention projections. `use_vllm=False` keeps
the GRPO environment separate from the evaluation vLLM 0.11 environment.

## Input contract

`--dataset-manifest` is UTF-8 JSONL. Every record needs `sample_id`, `source`,
`dataset_revision`, `split`, `original_id`, `question`, `prompt`, `answer`,
`answer_type` (`mc`, `number`, or `quantity`), and exactly one of `image_path`
or `image_paths` (an ordered list of 1–7 paths). Image paths may be relative to
the manifest directory. `prompt` is the **raw** two-message system/user list;
the system content must be exactly `You are a helpful assistant.` The user
content must contain the question and the instruction `Answer: X`. `answer`
is a single uppercase A–D letter for `mc`, a decimal string for `number`, or
a decimal plus cm/m unit for `quantity`.

The loader selects **only `split=train`** by default and rejects duplicate
IDs, missing images, invalid gold answers, and MMMU/MMMU-Pro provenance. This
source check does not replace C's text/image leakage audit for real data.

**Prompt position:** For the synthetic single-image string prompt, TRL 1.14.2
prepends the `image` column to the first user message. For real data with
explicit image positions, provide `image_paths` and a structured user content
list such as `[{"type":"text","text":"Question..."},{"type":"image"},
{"type":"text","text":"..."}]`. TRL fills the image blocks in list order
through its `images` column. Literal `<image N>` text is rejected; it would
duplicate the inserted image. The number of blocks must match the number of
paths. Mixed one-image and multi-image rows are normalized to one homogeneous
HF Dataset column. `prompt_preflight.txt` saves rendered first and maximum-image
examples. **Before real training, compare the rendered prompts with v1.2**, as
the current checks prove image ordering and processor acceptance, not complete
semantic parity with the evaluation task.

## Reward policy

The policy JSON is mandatory. `reward_policy.synthetic_strict.json` is for the
procedural multiple-choice base→GRPO fallback: uppercase A–D, a single final
answer, no ambiguous `B or C`, and zero reward for lowercase. The alternative
`reward_policy.proposed.json` follows the B handoff's provisional R09/R11/R13/R14
choices. **Neither policy is automatically the team's final real-data policy.**

Run the 18 handoff cases and two extras:

```bash
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_main_code/run_reward_cases.py \
  --policy /data/chaewon/mmdl_b_main_code/reward_policy.synthetic_strict.json \
  --expected-profile synthetic_strict \
  --output-dir /data/chaewon/mmdl_b_main_code/reward_test_strict
```

This passed **20/20 on the 6000**. The CSV/JSON report includes expected and
actual reward, parsed answer, reason, version, and v1.2 result. The run manifest
also records R07/R09/R11/R13/R14 and the gold-C ambiguous variant against v1.2.
With the synthetic strict policy, R07 with gold C differs from v1.2; R13 differs
but numeric questions are absent from the synthetic training set.

## Base→GRPO procedural fallback

The synthetic dataset is at
`/data/chaewon/mmdl_grpo_data/synthetic_v1_1536/manifest.jsonl`, with 1,344
training and 192 heldout examples. It is non-MMMU, but it is a **pipeline
fallback**, not evidence of an MMMU improvement.

CPU-only preflight (passed on the 6000):

```bash
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_main_code/train.py \
  --dataset-manifest /data/chaewon/mmdl_grpo_data/synthetic_v1_1536/manifest.jsonl \
  --reward-policy /data/chaewon/mmdl_b_main_code/reward_policy.synthetic_strict.json \
  --parent-kind base \
  --parent-model /home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output-dir /data/chaewon/mmdl_grpo_runs/base_insurance_001 \
  --max-steps 128 --validate-only
```

Remove only `--validate-only` to train. Schedule training **after the 6000
baseline evaluation**, since both use the same GPU. For the common-parent run,
use `--parent-kind merged_sft`, `--parent-model` set to A's locally merged SFT
model, `--parent-manifest` set to its `model_manifest.json`, and
`--expected-sft-adapter-sha256` set to A's transferred adapter SHA256.

Each training run must have a new `--output-dir`. Output includes:

- `run_manifest.json`: parent/base/dataset/policy hashes, LoRA setup, train
  metrics, GRPO adapter SHA256, rollout SHA256, and status.
- `adapter/`: PEFT `save_pretrained` adapter; the merged model is not sent.
- `trainer/completions/*.parquet`: TRL's original logged completions.
- `rollouts.jsonl` and `rollouts.csv`: per sample and candidate, raw output,
  parsed answer, reward, reason, token-count proxy, and provenance for C
  (source ID, question, choices, image path, optional image SHA256).
- `prompt_preflight.txt`: rendered processor prompt for parity inspection.

The CPU contract test `test_multimodal_contract.py` confirms ordered two-image
blocks, mixed one/two-image Dataset casting, Qwen processor rendering, and
rejection of mismatched placeholders or literal `<image N>` text.

## Optional synthetic heldout comparison

After `base_insurance_001/run_manifest.json` reports `status=complete`, the
optional `eval_synthetic_heldout.py` compares greedy base and adapter outputs
on the **192 synthetic heldout images** (64 each: count, bar, spatial). It
checks dataset/policy/adapter SHA256 against the training manifest and writes
per-item outputs, a paired CSV, and aggregate accuracy by task. This is an
exploratory training signal, **not an MMMU score**. It has not yet run on GPU;
allow roughly 10–30 minutes on the 6000, depending on generation length.

```bash
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_main_code/eval_synthetic_heldout.py \
  --dataset-manifest /data/chaewon/mmdl_grpo_data/synthetic_v1_1536/manifest.jsonl \
  --reward-policy /data/chaewon/mmdl_b_main_code/reward_policy.synthetic_strict.json \
  --base-model /home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --train-run-manifest /data/chaewon/mmdl_grpo_runs/base_insurance_001/run_manifest.json \
  --output-dir /data/chaewon/mmdl_grpo_runs/base_insurance_heldout_001
```

The CPU-only `--validate-only` check passed: train/heldout split, source, three
task counts, model revision, and policy file were accepted. Schedule this GPU
comparison after the baseline and GRPO training when time permits.

The exported rollout file requires exactly four candidates per logged group
and recomputes every reward against the logged value. TRL's Parquet files lack
per-candidate `finish_reason` and exact generated token IDs. The export marks
these **unavailable**; token counts are re-tokenized proxies, and `clipped_proxy`
is only a diagnostic hint.
