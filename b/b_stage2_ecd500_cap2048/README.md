# Isolated 2,048-token DPO(e) → GRPO fallback

This package is a copy of the ECD500 EasyR1 Stage2 code with only the
response cap, matching reward cap, experiment identity, finalizer log match,
and hardcoded preflight contract changed to 2,048. The active 1,024-token
queue and artifacts remain separate. Use this only if its probe fails
specifically on `Answer:` presence or cap proxy; do not use it to mask
missing data, adapter/lineage errors, or zero gradient.

Server destination:
`/data/chaewon/mmdl_stage2_egrpo_cap2048_20261122`.

## CPU preflight into a throwaway namespace

```bash
set -euo pipefail
STAGE1=/data/chaewon/mmdl_stage1_paper_style_20261120
STAGE2=/data/chaewon/mmdl_stage2_egrpo_cap2048_20261122
TRAIN_PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
PARENT=/data/chaewon/mmdl_merged_models/dpo_e_trunc_from_a_20261121
A_SHA=7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa
DRY=/data/chaewon/mmdl_stage2_runs/e_grpo_cap2048_cpu_preflight_20261122
test ! -e "$DRY"
PYTHONPATH="$STAGE1/easyr1" "$TRAIN_PY" "$STAGE2/preflight_ecd500.py" grpo probe "$DRY" \
  --source-config "$STAGE2/config_ecd500.yaml" \
  --engine "$STAGE1/easyr1" --stage1 "$STAGE1" --base "$PARENT" \
  --parent-manifest "$PARENT/model_manifest.json" \
  --expected-dpo-adapter-sha256 "$A_SHA"
```

Check `$DRY/run_manifest.json` for `parent_kind=merged_dpo`,
`base_model=$PARENT`, the expected A adapter SHA256, and 500 unique
train IDs. Check `$DRY/resolved_config.yaml` for both 2,048-token cap
fields, `worker.actor.model.model_path=$PARENT`, five rollouts, fixed
1e-6 LR, three epochs, raw v1.2 0/1 reward, and no KL.

## Fresh GPU probe, only after the active 1,024-token queue has ended

```bash
PROBE=/data/chaewon/mmdl_stage2_runs/e_grpo_probe_cap2048_20261122
test ! -e "$PROBE"
export MMDL_STAGE1_PACKAGE_ROOT="$STAGE1"
export MMDL_STAGE2_ROOT="$STAGE2"
export MMDL_TRAIN_PYTHON="$TRAIN_PY"
export MMDL_BASE_MODEL="$PARENT"
export MMDL_DPO_PARENT_MANIFEST="$PARENT/model_manifest.json"
export MMDL_DPO_ADAPTER_SHA256="$A_SHA"
bash "$STAGE2/run_ecd500.sh" grpo probe "$PROBE" \
  2>&1 | tee /data/chaewon/mmdl_stage2_runs/e_grpo_probe_cap2048_20261122.launch.log
```

Inspect `$PROBE/completion_report.json`, `visit_audit.json`,
`console.log`, and `gpu_telemetry.csv` before a full run. Require
exit 0, one positive-gradient update, 500 generated/used rollouts,
valid PEFT reload, nondegenerate reward, and improved cap/Answer
metrics relative to the 1,024-token probe. A 2,048 cap is a fallback
test, not proof of MMMU improvement.

If the probe passes, use a **new** full directory
`/data/chaewon/mmdl_stage2_runs/e_grpo_full_cap2048_20261122`.
It should still have 15 optimizer updates, three visits per ECD500 ID,
and 7,500 used rollouts. Merge the resulting adapter with the **same**
original A DPO(e) adapter into a distinct model directory
`/data/chaewon/mmdl_merged_models/dpo_e_grpo_ecd500_cap2048_20261122`.
The ordered merged lineage remains `[dpo, grpo]`, with a new GRPO
adapter SHA256 and the new source/resolved config SHA256 recorded by
`build_fixed300_lineage.py`. Use distinct smoke and 900 run names,
and compare the 900 result against the same 6000 DPO(e) parent run.

Stage1's five-question GRPO smoke took 44.1 seconds at 1,024 and
75.5 seconds at 2,048 (1.71×). The 1,024 ECD500 full run took
62.6 minutes with a sampled 47.4 GiB GPU peak. Plan 8–15 minutes
for a 2,048 probe and 1.5–2 hours for full training, subject to
measured 100-question batch memory. A successful Stage1 small probe
does not guarantee that the larger ECD500 batch fits.
