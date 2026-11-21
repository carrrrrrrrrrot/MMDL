#!/usr/bin/env bash
# Use after freeze_stage1.py has copied pinned source and scripts to /data.
set -euo pipefail

package_root=${MMDL_STAGE1_PACKAGE_ROOT:-/data/chaewon/mmdl_stage1_paper_style_20261120}
export MMDL_EASYR1_ROOT="$package_root/easyr1"
export MMDL_STAGE1_ROOT="$package_root/scripts"
exec bash "$MMDL_STAGE1_ROOT/run_smoke.sh" "$@"
