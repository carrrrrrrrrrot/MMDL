# ECD798 training follow-on: fixed 300 only

`after_train_300.py` waits for the new ECD798 GRPO/DAPO full-pass training queue to finish. It verifies the pinned 798-train/97-heldout manifest, both 1,596-step base-parent runs, adapter hashes, matching initial LoRA, the frozen v1.2 evaluator, and the existing 6000 base fixed-300 input. It then runs, sequentially for each model: LoRA merge, 20-item smoke, and the **same** fixed 300 MMMU items used for the ECD500 screen. It never starts a 900-item evaluation.

The fixed-300 result is a single-seed screening score. Compare each candidate to the 6000 base **185/300**, and compare ECD798 to the prior ECD500 GRPO **195/300** and DAPO **193/300** on this same ID list. The new 298 items are a subset of C's final SFT1 leakage audit, but their ECD gold answers were not exhaustively verified; report these runs as exploratory. The new run uses 1,596 steps versus 1,000 for ECD500 so both use approximately two optimizer steps per training question. The follow-on requires 798 distinct training IDs and 798 reward groups in each completed model. **Data count and total compute both change**; this is a full-pass data-scaling experiment, not a compute-matched causal isolation of data count.

On `chaewon`, after copying this script under `/data/chaewon/mmdl_b_eval_ops/rl_ablation_20261120_ecd798/`, launch with a new log:

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
OPS=/data/chaewon/mmdl_b_eval_ops
TRAIN=/data/chaewon/mmdl_grpo_runs/rl_ablation_20261120_ecd798_fullpass
DATA=/data/chaewon/mmdl_grpo_data/rl_ablation_20261120_ecd798/manifest/ecd_manifest.jsonl
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
REPO=/home/chaewon/projects/qwen_eval_project/MMDL_eval_v1_2
MMMU=/home/chaewon/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
PLAN=/data/chaewon/mmdl_eval_runs/rl_ablation_20261120_ecd798_fullpass_plan_v2
nohup "$PY" -u "$OPS/rl_ablation_20261120_ecd798/after_train_300.py" \
  --train-root "$TRAIN" --train-manifest "$DATA" --base-model "$BASE" \
  --repo-root "$REPO" --data-root "$MMMU" \
  --train-python "$PY" --eval-python /data/chaewon/mmdl_envs/vllm011_repro/bin/python \
  --output-root /data/chaewon/mmdl_eval_runs \
  --merged-root /data/chaewon/mmdl_merged_models \
  --base-fixed300 /data/chaewon/mmdl_eval_runs/rlab19qcv2_base_fixed300 \
  --ids-file /data/chaewon/mmdl_eval_runs/rl_ablation_20261119_qcv2_plan/fixed300_ids_seed3407.txt \
  --compare-script /data/chaewon/mmdl_b_compare/compare_runs.py \
  --plan-dir "$PLAN" --run-prefix ecd798full_20261120 \
  > /data/chaewon/mmdl_eval_runs/rl_ablation_20261120_ecd798_fullpass_followon_v2.launch.log 2>&1 < /dev/null &
```

Monitor `$PLAN/progress.json`, `$PLAN/events.jsonl`, `$PLAN/logs/`, and finally `$PLAN/screening_summary.json`. Every completed fixed-300 run contains `handoff.json`, `comparison/comparison_summary.json`, and `comparison/paired_items.csv`. An interrupted/incomplete output or changed hash stops the follow-on, rather than overwriting it. Existing exact complete stages can be revalidated and skipped on a restart with the same script and arguments.
