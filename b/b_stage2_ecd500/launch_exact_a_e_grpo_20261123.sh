#!/usr/bin/env bash
# Train GRPO only from the bit-identical 4090 DPO(e) merge A evaluated at 65.15%.
set -euo pipefail

PARENT=/data/chaewon/mmdl_merged_models/dpo_e_a_exact_rebuild_20261123
RUN=/data/chaewon/mmdl_stage2_runs/e_grpo_a_exact_full_20261123
STAGE1=/data/chaewon/mmdl_stage1_paper_style_20261120
STAGE2=/data/chaewon/mmdl_stage2_egrpo_20261122
TRAIN_PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python

test ! -e "$RUN"
test "$(sha256sum "$PARENT/model-00001-of-00002.safetensors" | cut -d' ' -f1)" = 1d9e148aa1a102407bf3dd89b7f5a8d2bfbe792922bee9db243c4b54d03d1b2b
test "$(sha256sum "$PARENT/model-00002-of-00002.safetensors" | cut -d' ' -f1)" = 0c615cbfe078470dfdeb299976580391f8482fed83ac521ec90b3de331b7f0f5
test "$(sha256sum "$PARENT/model.safetensors.index.json" | cut -d' ' -f1)" = c321e968083cc64dbdc77b9cc17842396cf376399f4ff4bca6b0a001fd22afef

export MMDL_STAGE1_PACKAGE_ROOT="$STAGE1"
export MMDL_STAGE2_ROOT="$STAGE2"
export MMDL_TRAIN_PYTHON="$TRAIN_PY"
export MMDL_BASE_MODEL="$PARENT"
export MMDL_DPO_PARENT_MANIFEST="$PARENT/model_manifest.json"
export MMDL_DPO_ADAPTER_SHA256=7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa

printf 'Exact A DPO(e) parent verified; starting GRPO at %s\n' "$(date -u +%FT%TZ)"
set +e
bash "$STAGE2/run_ecd500.sh" grpo full "$RUN"
result=$?
set -e
printf '%s\n' "$result" > /data/chaewon/mmdl_exact_a_e_rebuild_20261123/grpo_exit_code.txt
printf 'GRPO exit=%s at %s\n' "$result" "$(date -u +%FT%TZ)"
exit "$result"
