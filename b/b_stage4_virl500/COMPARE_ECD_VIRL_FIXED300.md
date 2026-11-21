# Compare ECD500 and ViRL500 fixed300 screens

Run only after the Stage4 queue has completed both full candidate evaluations.
The script is read-only with respect to training, evaluation, and their frozen
artifacts. It uses only Python's standard library and does not use a GPU.

Remote script: `/data/chaewon/mmdl_stage4_compare_20261121/compare_ecd_virl_fixed300.py`

```bash
PY=/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python
SCRIPT=/data/chaewon/mmdl_stage4_compare_20261121/compare_ecd_virl_fixed300.py
EVAL=/data/chaewon/mmdl_eval_runs
OUT=/data/chaewon/mmdl_stage4_compare_20261121/results
"$PY" "$SCRIPT" \
  --ecd-grpo "$EVAL/stage2_ecd500_grpo_20261121_fixed300/comparison" \
  --ecd-dapo "$EVAL/stage2_ecd500_dapo_core_20261121_fixed300/comparison" \
  --virl-grpo "$EVAL/stage4_virl500_grpo_20261121_fixed300/comparison" \
  --virl-dapo "$EVAL/stage4_virl500_dapo_core_20261121_fixed300/comparison" \
  --output-dir "$OUT"
```

This writes `report.json`, `report.md`, `paired_changes.csv`, and
`hash_lock.json` to `OUT`. The two Stage2 pairs are checked against pinned
SHA256 values. All four summary/CSV pairs are hashed before and after parsing,
and the eight hashes are recorded. The Stage4 hashes have no independent pin
on the first run; the saved `hash_lock.json` is a snapshot for later reruns.
For a later strict hash check, repeat the command with
`--expected-hashes "$OUT/hash_lock.json"`.

The script also requires the same seed 3407, exactly 300 unique IDs, base
185/300, unchanged gold/base rows, identical input/document hashes, v1.2
scoring, successful evaluation status, and summary/CSV count parity. It
reports corrections and regressions from ECD to ViRL separately for GRPO and
DAPO core, plus a per-question CSV. This is an exploratory 300-item screen,
not a causal data effect or a 900-item result.

Current CPU-only Stage2 self-test:

```bash
"$PY" "$SCRIPT" \
  --ecd-grpo "$EVAL/stage2_ecd500_grpo_20261121_fixed300/comparison" \
  --ecd-dapo "$EVAL/stage2_ecd500_dapo_core_20261121_fixed300/comparison" \
  --self-test-stage2
```

It passed with base 185, ECD GRPO 191, ECD DAPO core 194, and 300 paired IDs.
