# DPO(e) → GRPO continuation on the RTX PRO 6000

The queued program `queue_e_grpo_probe.py` only waits for the parent's
one-seed 900-item v1.2 screen and then runs a one-step GRPO probe. It checks
the parent handoff, same-seed 900-item coverage and input/image parity,
ordered DPO lineage, GPU idle state, adapter reload, nonzero gradient,
reward, `Answer:` presence, and response-cap proxy. The operational probe
gate is reward accuracy 0.05–0.95, `Answer:` presence ≥0.80, and cap
proxy ≤0.20. These thresholds are screening rules, not statistical claims.

Run the queue **only once** in a fresh namespace. It refuses to resume or
overwrite any of its three paths.

```bash
STAGE2=/data/chaewon/mmdl_stage2_egrpo_20261122
TRAIN_PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
$TRAIN_PY -u "$STAGE2/queue_e_grpo_probe.py" \
  2>&1 | tee /data/chaewon/mmdl_stage2_runs/e_grpo_probe_queue_20261122.log
```

## After the probe passes

Inspect `/data/chaewon/mmdl_stage2_runs/e_grpo_probe_queue_20261122/status.json`
for `event=probe_passed`, plus the probe's
`completion_report.json` and `metric_summary.json`. An absent or failed
status is a stop. A probe is a learning-signal check, not an MMMU result.

The following commands use a **new** full run directory. They do not resume
the probe's one-step adapter.

```bash
set -euo pipefail
STAGE1=/data/chaewon/mmdl_stage1_paper_style_20261120
STAGE2=/data/chaewon/mmdl_stage2_egrpo_20261122
OPS=/data/chaewon/mmdl_b_eval_ops_egrpo_20261122
TRAIN_PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
MERGE_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
EVAL_PY=/data/chaewon/mmdl_envs/vllm011_repro/bin/python3.10
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
PARENT=/data/chaewon/mmdl_merged_models/dpo_e_trunc_from_a_20261121
PARENT_EVAL=/data/chaewon/mmdl_eval_runs/a_dpo_e_parent_900_s3407_20261122
A_ADAPTER=/data/chaewon/mmdl_a_e_handoff_20261121/adapter
A_SHA=7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa
FULL=/data/chaewon/mmdl_stage2_runs/e_grpo_full_20261122
MERGED=/data/chaewon/mmdl_merged_models/dpo_e_grpo_ecd500_20261122
EVAL_ROOT=/data/chaewon/mmdl_eval_runs
EVAL_NAME=dpo_e_grpo_ecd500_900_s3407_20261122
REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
DATA=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68

test "$(jq -r .event /data/chaewon/mmdl_stage2_runs/e_grpo_probe_queue_20261122/status.json)" = probe_passed
test ! -e "$FULL"
test ! -e "$MERGED"
test ! -e "$EVAL_ROOT/$EVAL_NAME"
export MMDL_STAGE1_PACKAGE_ROOT="$STAGE1"
export MMDL_STAGE2_ROOT="$STAGE2"
export MMDL_TRAIN_PYTHON="$TRAIN_PY"
export MMDL_BASE_MODEL="$PARENT"
export MMDL_DPO_PARENT_MANIFEST="$PARENT/model_manifest.json"
export MMDL_DPO_ADAPTER_SHA256="$A_SHA"
bash "$STAGE2/run_ecd500.sh" grpo full "$FULL" 2>&1 | tee "$FULL.launch.log"
```

The full-run gate is `completion_report.json.status=complete`,
`optimizer_steps=15`, `positive_grad_steps>0`, exactly 500 training IDs
each used three times, 7,500 used rollouts, and exact PEFT reload.
`finalize_ecd500.py` and `build_fixed300_lineage.py` verify these.
Parse `console.log` to inspect all 15 updates; the training run stops
automatically on the first failed finalizer check.

```bash
$TRAIN_PY "$STAGE2/parse_full_metrics.py" "$FULL/console.log" \
  > "$FULL/metric_summary.json"
test "$(jq -r .status "$FULL/completion_report.json")" = complete
test "$(jq -r .optimizer_steps "$FULL/completion_report.json")" = 15
test "$(jq -r .used_rollouts "$FULL/completion_report.json")" = 7500
GRPO_ADAPTER="$FULL/checkpoints/global_step_15/actor/lora_adapter"
GRPO_SHA="$(jq -r .adapter_sha256 "$FULL/completion_report.json")"

"$MERGE_PY" "$OPS/merge_lora.py" --base-model "$BASE" \
  --adapter "$A_ADAPTER" --adapter-role dpo --adapter-sha256 "$A_SHA" \
  --adapter "$GRPO_ADAPTER" --adapter-role grpo --adapter-sha256 "$GRPO_SHA" \
  --output-dir "$MERGED" --preflight-only
"$MERGE_PY" "$OPS/merge_lora.py" --base-model "$BASE" \
  --adapter "$A_ADAPTER" --adapter-role dpo --adapter-sha256 "$A_SHA" \
  --adapter "$GRPO_ADAPTER" --adapter-role grpo --adapter-sha256 "$GRPO_SHA" \
  --output-dir "$MERGED"
```

The ordered merge manifest must say `parent_kind=merged_dpo_grpo` and
contain exactly `[dpo, grpo]` with the two expected adapter SHA256 values.
Build the GRPO provenance JSON against the completed merged model before
evaluation:

```bash
$TRAIN_PY "$STAGE2/build_fixed300_lineage.py" \
  --run-manifest "$FULL/run_manifest.json" \
  --completion-report "$FULL/completion_report.json" \
  --train-file "$STAGE1/scripts/ecd500_easyr1.jsonl" \
  --validation-file "$STAGE1/scripts/ecd97_holdout_easyr1.jsonl" \
  --config-file "$FULL/resolved_config.yaml" \
  --source-config-file "$STAGE2/config_ecd500.yaml" \
  --reward-file "$STAGE1/scripts/reward_v12.py" \
  --adapter-dir "$GRPO_ADAPTER" \
  --model-manifest "$MERGED/model_manifest.json" \
  --output "$FULL/fixed300_training_manifest.json"

"$MERGE_PY" "$OPS/candidate_eval.py" \
  --model-manifest "$MERGED/model_manifest.json" \
  --train-run-manifest "$FULL/fixed300_training_manifest.json" \
  --repo-root "$REPO" --data-root "$DATA" --eval-python "$EVAL_PY" \
  --output-root "$EVAL_ROOT" --run-name "$EVAL_NAME" --mode screen \
  --baseline-run "$PARENT_EVAL" \
  --compare-script /data/chaewon/mmdl_b_compare/compare_runs.py \
  --verify-model-hashes --dry-run
"$MERGE_PY" "$OPS/candidate_eval.py" \
  --model-manifest "$MERGED/model_manifest.json" \
  --train-run-manifest "$FULL/fixed300_training_manifest.json" \
  --repo-root "$REPO" --data-root "$DATA" --eval-python "$EVAL_PY" \
  --output-root "$EVAL_ROOT" --run-name "$EVAL_NAME" --mode screen \
  --baseline-run "$PARENT_EVAL" \
  --compare-script /data/chaewon/mmdl_b_compare/compare_runs.py \
  --verify-model-hashes
```

The evaluation gate is a completed handoff, 900/900 paired items,
`input_exact_agreement=900`, `doc_hash_agreement=900`, and seed 3407
on the same 6000. Compare `candidate_correct-baseline_correct` in
`$EVAL_ROOT/$EVAL_NAME/comparison/comparison_summary.json`. This is
the incremental GRPO gain over DPO(e), not the gain over the original base.
Do not label this one-seed 6000 result as the official 4090 three-seed result.
