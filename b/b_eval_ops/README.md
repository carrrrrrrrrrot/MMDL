# B adapter merge and candidate evaluation

These scripts live outside the clean MMDL v1.2 evaluation worktree. They never
copy a merged model between servers. Transfer only each PEFT `adapter/`
directory, its `adapter_model.safetensors` SHA256, and the fixed base revision.

## Verified on the 6000 server

- Code: `/data/chaewon/mmdl_b_eval_ops`.
- `merge_lora.py --preflight-only` accepted the synthetic GRPO pilot adapter:
  SHA256 matched and all 288 LoRA tensors were language attention weights.
- CPU merge of that **synthetic-only** adapter completed in 16.49 seconds,
  maximum RSS 11.1 GiB, writing an 8.3 GiB model at
  `/data/chaewon/mmdl_merged_models/pilot_grpo_cpu_merge_001` with
  `model_manifest.json` and per-file SHA256. The adapter is **not a benchmark
  candidate** and was not evaluated.
- `candidate_eval.py --mode smoke --dry-run` passed using the fixed v1.2 repo,
  actual 6000 evaluation environment and cached MMMU data. Seven CPU contract
  tests pass. No candidate benchmark evaluation has run through this wrapper.

## Inputs and lineage

`merge_lora.py` requires the cached base snapshot ending in revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17` and one or two ordered PEFT
adapters. It checks `adapter_model.safetensors` against the supplied SHA256,
rejects vision/connector LoRA tensors, then applies
`PeftModel.from_pretrained(...).merge_and_unload(safe_merge=True)` in order.
Supported lineages: `base→SFT`, `base→GRPO`, `base→DPO`, `base→SFT→GRPO`, and
`base→SFT→DPO`. The last two are how final same-parent candidates are rebuilt.

The new output directory contains `model_manifest.json` only after every model
file and processor config was saved and hashed. A `_INCOMPLETE` marker blocks
candidate evaluation if a merge fails. For a common SFT parent, the manifest's
top-level keys include `parent_kind=merged_sft`, `base_revision`,
`adapter_sha256`, `source_adapter_path`, and `merged_model_path`. B's GRPO
`train.py` consumes this parent manifest.

Example after A sends its SFT adapter and SHA256:

```bash
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
TRAIN_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
OPS=/data/chaewon/mmdl_b_eval_ops
SFT_ADAPTER=/data/chaewon/incoming/a_sft_adapter
SFT_SHA=<sha256-from-A>

$TRAIN_PY $OPS/merge_lora.py \
  --base-model "$BASE" \
  --adapter "$SFT_ADAPTER" --adapter-role sft --adapter-sha256 "$SFT_SHA" \
  --output-dir /data/chaewon/mmdl_merged_models/a_sft_parent_001
```

After B trains GRPO on that **same** merged SFT parent, merge the original A
SFT adapter first and B GRPO adapter second into a fresh output directory.
For the insurance `base→GRPO` branch, pass only the B GRPO adapter with role
`grpo`. Record the resulting `lineage_sha256` in the team table.

## Evaluation

`candidate_eval.py` validates the complete merged-model manifest, the clean
evaluation worktree at commit `61025850563d0331e365bc974a63c0d3c4ed4c2c`,
vLLM 0.11.0 / transformers 4.57.1, and all 30 MMMU validation Parquets. For
GRPO, pass `--train-run-manifest` so the evaluated adapter SHA256 and SFT parent
SHA256 must match training. It refuses to launch while GPU memory use exceeds
4 GiB unless explicitly overridden. It also refuses a 4090 baseline when
running on 6000: screening Δ must use a baseline measured on the same GPU.

Modes:

| Mode | Evaluation | Use |
| --- | --- | --- |
| `smoke` | `--limit 20`, seed 3407 | Loading/merge check only |
| `screen` | 900 items, seed 3407 | Candidate filtering on 6000 |
| `full` | 900 items, seeds 3407/1/2 | 6000 insurance/losers or 4090 official final |

`screen` and `full` require `--baseline-run` and `--compare-script`.
The baseline must have completed the same seed(s) on the current GPU. A 6000
screening score is labelled separately from the 4090 official result. The
wrapper writes `candidate_manifest.json` before the run and `handoff.json`
with result-file SHA256 after success. The paired comparison is saved under
`<run>/comparison/`, including question × seed rows for C.

Example 6000 smoke for a **real** merged candidate:

```bash
$TRAIN_PY $OPS/candidate_eval.py \
  --model-manifest /data/chaewon/mmdl_merged_models/<candidate>/model_manifest.json \
  --train-run-manifest /data/chaewon/mmdl_grpo_runs/<run>/run_manifest.json \
  --repo-root /home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2 \
  --data-root /home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \
  --eval-python /data/chaewon/mmdl_envs/vllm011_repro/bin/python \
  --output-root /data/chaewon/mmdl_eval_runs \
  --run-name <new-unique-name> --mode smoke
```

For a one-seed screen, change to `--mode screen`, add
`--baseline-run /data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117`,
and `--compare-script /data/chaewon/mmdl_b_compare/compare_runs.py`. Run this
only after the 6000 baseline completes. Use a fresh run name each time.

This is a tested **B-side draft** of `merge_lora.py`. A and B must agree on one
shared script (or byte-identical method and processor files) before official
cross-server model comparisons.

## Optional queued synthetic fallback

`orchestrate_insurance.py` has been copied to the 6000 server but **has not
been launched**. Its `--check-only` preflight passed with the 1,536-item
synthetic manifest and strict reward-policy SHA256 values below. It currently
reports that 6000 baseline seed 1 is still running. Before launching, check
whether A's SFT parent or C's real, leakage-cleared GRPO data is available;
if so, prioritize the common-parent branch and leave this fallback pending.

The queue waits for all three 6000 base seeds and the paired 4090 comparison,
trains synthetic base→GRPO for 128 optimizer steps, verifies a nonzero LoRA
update and saved/reloaded adapter, CPU-merges it, and runs a 20-item smoke.
It launches a one-seed 900-item screen only if the **provisional operational**
6000 gate passes: mean score difference at most 0.5%p, every same-seed binary
agreement at least 85%, identical 900 input/doc hashes, and per-seed missing
final-answer count difference at most 20 with symmetric difference at most
100. Any failed gate stops before the 900-item screen. No three-seed candidate
evaluation is queued. A successful one-seed screen is exploratory transfer
evidence from synthetic training, not a common-parent improvement claim.

```bash
$TRAIN_PY $OPS/orchestrate_insurance.py \
  --run-name base_insurance_001 \
  --dataset-manifest /data/chaewon/mmdl_grpo_data/synthetic_v1_1536/manifest.jsonl \
  --dataset-sha256 48156c4fafb011c307bd847405062b30529fb396cee28ecc6bc25cf8f04f16be \
  --reward-policy /data/chaewon/mmdl_b_main_code/reward_policy.synthetic_strict.json \
  --reward-policy-sha256 011e22248087017282fabb2802e2b4aedb005e48c5295a263ff37d4a3b157ed9 \
  --max-steps 128 --check-only
```

Remove `--check-only` only when the team has decided to start the fallback;
redirect stdout/stderr to a unique log under `/data/chaewon/mmdl_b_eval_ops`.
The queue writes `orchestrations/base_insurance_001/state.json` as it advances.
