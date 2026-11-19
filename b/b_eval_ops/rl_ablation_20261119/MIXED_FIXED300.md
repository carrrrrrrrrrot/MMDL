# Additional mixed-data GRPO screen (6000)

This path screens mixed500 and mixed1000 **before** any further 900-item run.
It uses the frozen v1.2 evaluator at commit
`61025850563d0331e365bc974a63c0d3c4ed4c2c`, seed 3407, the same
fixed 300 IDs (SHA256
`1492c926d4c947a5d9acad96acd1e43b877ec8d00344f8a240b65ad4d673e04d`),
and the freshly measured 6000 base score **185/300**. The wrapper calls the
existing `merge_lora.py`, `candidate_eval.py --mode smoke`, and
`run_fixed300.py`. It has no 900-item stage and writes only to new
`rlab19mix*` model/run directories. The original six-run `ranking.json`,
`review_gate.json`, and 900 markers are not inputs or outputs.

Run each command only after its `run_manifest.json` is complete, the adapter
SHA matches, reload succeeds, and the GPU is idle. `--plan-only` is safe
before training completes; `--preflight` checks finished training without
starting a GPU job. `--execute` performs merge → smoke20 → fixed300 in order.
If a stage fails, inspect its log and run directory before retrying. Completed
matching stages are verified and reused; incomplete outputs are not replaced.

```bash
OPS=/data/chaewon/mmdl_b_eval_ops
TRAIN_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
EVAL_PY=/data/chaewon/mmdl_envs/vllm011_repro/bin/python
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
EVAL_REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
MMMU=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
OUT=/data/chaewon/mmdl_eval_runs
MERGED=/data/chaewon/mmdl_merged_models
BASE300="$OUT/rlab19qcv2_base_fixed300"
IDS="$OUT/rl_ablation_20261119_qcv2_plan/fixed300_ids_seed3407.txt"
COMPARE=/data/chaewon/mmdl_b_compare/compare_runs.py

COMMON=(--base-model "$BASE" --repo-root "$EVAL_REPO" --data-root "$MMMU" \
  --train-python "$TRAIN_PY" --eval-python "$EVAL_PY" \
  --output-root "$OUT" --merged-root "$MERGED" \
  --baseline-300 "$BASE300" --ids-file "$IDS" --compare-script "$COMPARE")

# mixed500: 250 ECD + 250 MAmmoTH/CLEVR training examples.
"$TRAIN_PY" "$OPS/rl_ablation_20261119/run_mixed_fixed300.py" \
  --train-manifest /data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_mixed500_v2/runs/mixed_grpo/run_manifest.json \
  --dataset-sha256 68deeafa94f9f8fdb0f37107635c212520a0ff24dac86a8581fbfbbccbd1e465 \
  --expected-examples 500 --expected-steps 1000 \
  --run-prefix rlab19mix500v2 \
  --plan-dir "$OUT/rl_ablation_20261119_mixed500_v2_plan" \
  "${COMMON[@]}" --preflight

# Change --preflight to --execute after the checks pass and no training/eval
# process is using the 6000. The same command is safe to resume after a clean
# stage; it never launches 900 evaluation.
```

For mixed1000, the tentative training configuration is **2000 optimizer
steps** (one pass at two iterations per example). Confirm the actual
configuration before replacing `--plan-only` with `--preflight` or `--execute`.
The verified dataset SHA is
`e71df0141dabc9d696cf67e5b47d35b05801633d145d855ca83b7efdbb95637e`.
The training manifest path below is the planned output; check it before use.

```bash
MIXED1000_STEPS=2000  # tentative; plan-only until the training run is finalized
"$TRAIN_PY" "$OPS/rl_ablation_20261119/run_mixed_fixed300.py" \
  --train-manifest /data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_mixed1000_v1/runs/mixed_grpo/run_manifest.json \
  --dataset-sha256 e71df0141dabc9d696cf67e5b47d35b05801633d145d855ca83b7efdbb95637e \
  --expected-examples 1000 --expected-steps "$MIXED1000_STEPS" \
  --run-prefix rlab19mix1000v1 \
  --plan-dir "$OUT/rl_ablation_20261119_mixed1000_v1_plan" \
  "${COMMON[@]}" --plan-only
# After the training manifest exists, use --preflight, then --execute.
```

Expected per-candidate artifacts:

- `/data/chaewon/mmdl_merged_models/<prefix>_mixed_grpo/model_manifest.json`
- `$OUT/<prefix>_mixed_grpo_smoke20/handoff.json`
- `$OUT/<prefix>_mixed_grpo_fixed300/handoff.json`
- `$OUT/<prefix>_mixed_grpo_fixed300/comparison/comparison_summary.json`
- `$OUT/<prefix>_mixed_grpo_fixed300/comparison/paired_items.csv`
- `$OUT/rl_ablation_20261119_<size>_plan/events.jsonl`, `progress.json`,
  `mixed_fixed300_result.json`, and `logs/*.log`

Require `items_compared=300`, `input_exact_agreement=300`,
`doc_hash_agreement=300`, and `baseline_correct=185` before comparing the new
scores with the earlier six screens. These are **diagnostic** 6000 single-seed
results. With the tentative 2000-step mixed1000 configuration, both data
volume and optimizer updates differ from mixed500. If mixed1000 improves,
consider a 2000-step mixed500 control before attributing the gain to more
data. Record optimizer steps and document exposures for every run.
