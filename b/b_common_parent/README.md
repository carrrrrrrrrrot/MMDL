# A/B common SFT parent merge bridge

`merge_shared.py` invokes the **unchanged** `a/shared` commit `468f7bc`
`team_share/merge_lora.py` once per adapter. It pins that script's SHA256,
checks the received adapter's PEFT weights SHA256 and language-only LoRA keys,
and adds B's `model_manifest.json` after A's script writes `MERGE_INFO.json`.
The B manifest lets `b_grpo_main/train.py` and `b_eval_ops/candidate_eval.py`
verify the exact SFT parent and the ordered SFT→GRPO lineage. No merged model
is transferred between servers. The official evaluation worktree stays at
`6102585`.

Run in `/data/chaewon/mmdl_envs/grpo_pilot` after A sends the selected SFT
adapter directory and the SHA256 of `adapter_model.safetensors`. Use new output
directories and first add `--check-only` to verify inputs without merging.

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
SCRIPT=/data/chaewon/mmdl_b_impl/team_share/merge_lora.py
WRAP=/data/chaewon/mmdl_b_impl/b/b_common_parent/merge_shared.py
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
SFT=/data/chaewon/incoming/<selected-sft-adapter>
SFT_SHA=<A-provided-sha256>
PARENT=/data/chaewon/mmdl_merged_models/<selected-sft-parent>

$PY $WRAP --stage sft --shared-script "$SCRIPT" --base-model "$BASE" \
  --adapter "$SFT" --adapter-sha256 "$SFT_SHA" --out "$PARENT"
```

Then, when C's leakage-checked data and the team-approved v1.2 reward policy
are present, train B's GRPO from that same `PARENT` with
`train.py --parent-kind merged_sft --parent-model "$PARENT"
--parent-manifest "$PARENT/model_manifest.json"
--expected-sft-adapter-sha256 "$SFT_SHA"`. Run `--validate-only` first.

After B saves its GRPO PEFT adapter, invoke A's script once more through the
wrapper, with the **already merged SFT parent as `--base-model`**:

```bash
GRPO=/data/chaewon/mmdl_grpo_runs/<run>/adapter
GRPO_SHA=<run_manifest.adapter_sha256>
TRAIN=/data/chaewon/mmdl_grpo_runs/<run>/run_manifest.json
FINAL=/data/chaewon/mmdl_merged_models/<selected-sft-grpo>

$PY $WRAP --stage grpo --shared-script "$SCRIPT" --base-model "$PARENT" \
  --parent-manifest "$PARENT/model_manifest.json" --train-run-manifest "$TRAIN" \
  --adapter "$GRPO" --adapter-sha256 "$GRPO_SHA" --out "$FINAL"
```

Pass `$FINAL/model_manifest.json` and `$TRAIN` to `candidate_eval.py` for a
20-item smoke, then a 6000 one-seed screen if the 6000 baseline gate passes.
Rebuild from the same adapters with A's script for the 4090 official evaluation.

The wrapper's `--check-only` validates paths/SHA/lineage without loading model
weights or touching the GPU. Actual merges set `CUDA_VISIBLE_DEVICES` empty and
use CPU. A script's `MERGE_INFO.json` is verified before B's compatibility
manifest is written. This B manifest is metadata; it does not alter the model
weights or A's merger.

On 2026-11-17, the wrapper's two CPU contract tests passed. One actual CPU
merge using the old **synthetic GRPO pilot adapter as an SFT-shaped test input**
produced `/data/chaewon/mmdl_merged_models/a_shared_pilot_cpu_001` and valid
`MERGE_INFO.json`/`model_manifest.json`; B's `train.py --validate-only` accepted
that parent. It is a plumbing test, **not** A's selected SFT model or an MMMU
candidate. A real common-parent run still needs A's selected adapter/SHA and
C's leakage-cleared data.
