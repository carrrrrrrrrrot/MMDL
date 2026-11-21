# EasyR1 candidate lineage for the existing 300-item screen

After a **full** Stage2 run finishes and its PEFT adapter passes exact reload,
build the legacy-compatible training manifest. Use the adapter directory named
in that run's `completion_report.json`. The command performs only CPU reads.

```bash
PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
STAGE1=/data/chaewon/mmdl_stage1_paper_style_20261120
STAGE2=/data/chaewon/mmdl_stage2_ecd500_20261120
RUN=/data/chaewon/mmdl_stage2_runs/REPLACE_WITH_COMPLETED_FULL_RUN
ADAPTER=REPLACE_WITH_COMPLETION_REPORT_ADAPTER_DIR

"$PY" "$STAGE2/build_fixed300_lineage.py" \
  --run-manifest "$RUN/run_manifest.json" \
  --completion-report "$RUN/completion_report.json" \
  --train-file "$STAGE1/scripts/ecd500_easyr1.jsonl" \
  --validation-file "$STAGE1/scripts/ecd97_holdout_easyr1.jsonl" \
  --config-file "$RUN/resolved_config.yaml" \
  --source-config-file "$STAGE2/config_ecd500.yaml" \
  --reward-file "$STAGE1/scripts/reward_v12.py" \
  --adapter-dir "$ADAPTER" \
  --output "$RUN/fixed300_training_manifest.json" --dry-run
```

For a completed one-step probe, point `--run-manifest`, `--completion-report`,
`--config-file`, `--adapter-dir`, and `--output` at that probe, use
`ecd5_holdout_easyr1.jsonl` for `--validation-file`, and append `--probe-check`
to `--dry-run`. This checks the real probe adapter but cannot write a candidate
evaluation manifest.

For a full run, keep `--dry-run` through LoRA merge. Then add
`--model-manifest /path/to/merged/model_manifest.json`, repeat the dry run,
and remove `--dry-run` once to write the verified manifest. The merged model
must record the RL adapter with role `grpo` even if the EasyR1 algorithm was
`dapo_core`; `algorithm` in this training manifest preserves the actual method.
The old 300-item runner reads `base_revision`, `parent_kind`, and
`adapter_sha256` from this file. It still checks the merged model SHA itself.

The CPU-only contract check is:

```bash
"$PY" -B "$STAGE2/test_build_fixed300_lineage.py"
```
