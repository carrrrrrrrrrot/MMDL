# B single-GPU GRPO/DAPO/SAPO queue (11/19)

**Data-quality hold:** the first ECD GRPO run was intentionally stopped at
140/1,000 steps after direct image/gold checks found wrong labels in the ECD
and MAmmoTH manifests. See [QC_ABORT_20261119.md](QC_ABORT_20261119.md).
Do not resume the original queue or reuse the `rl_20261119` manifests. The
instructions below describe the automation only; use new reviewed manifests
and a new output root for any resumed experiments.

`orchestrate.py` uses the pinned base and v1.2 0/1 reward. It does **not**
interpret C's leakage-only audit as quality approval: pass B's separately
reviewed, immutable exploratory manifests explicitly. Each manifest must have
exactly 500 `train` and 100 `heldout` rows. The queue trains on `train` only.

Every run uses seed 3407, BF16 language LoRA r16/alpha32, 4 rollouts, 512
completion tokens, learning rate 1e-5, 2 iterations per rollout batch, 1,000
optimizer steps, and checkpoints at steps 250/500/750/1000 (two retained).
For a new, smaller quality-reviewed pilot, the queue accepts
`--expected-train N --expected-heldout M --max-steps S --save-steps K`; record those values
and use a distinct output root. The original defaults remain 500/100/1000.
The first two runs are ECD GRPO and MAmmoTH GRPO. A zero-signal GRPO failure
blocks the other loss variants for that dataset; other failures stop the queue.
The first invocation pauses after two new runs for a signal check, then the
same command without `--pause-after` resumes the remaining variants. Never
change a manifest, reward file, train.py, or plan between invocations.

Example on the 6000 server after the reviewed manifests are present:

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
CODE=/data/chaewon/mmdl_b_main_code
DATA=/data/chaewon/mmdl_grpo_data/rl_ablation_20261119
BASE=/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
OUT=/data/chaewon/mmdl_grpo_runs/rl_ablation_20261119
COMMON=(
  --ecd-manifest "$DATA/ecd_manifest.jsonl"
  --mammoth-manifest "$DATA/mammoth_manifest.jsonl"
  --reward-policy "$CODE/reward_policy.v12.json"
  --train-script "$CODE/train.py"
  --python "$PY"
  --base-model "$BASE"
  --output-root "$OUT"
)
"$PY" "$CODE/rl_ablation_20261119/orchestrate.py" --preflight "${COMMON[@]}"
"$PY" "$CODE/rl_ablation_20261119/orchestrate.py" --execute --pause-after 2 "${COMMON[@]}"
# Inspect $OUT/status.txt and $OUT/summary.json, then resume if signal passes:
"$PY" "$CODE/rl_ablation_20261119/orchestrate.py" --execute "${COMMON[@]}"
```

For progress, read `$OUT/status.txt` (atomic overwrite every 15 seconds),
`$OUT/events.jsonl` (append-only 30-second heartbeats and transitions), and
`$OUT/logs/<dataset>_<loss>.log`. Every run also contains a `run_manifest.json`,
`initial_lora_sha256.txt`, adapter, rollout log, and their hashes. A resumed
queue skips only exact complete runs with matching dataset/reward/code hashes.
The initial LoRA SHA must match across GRPO/DAPO/SAPO on the same dataset.

For a later mixed 250 ECD + 250 MAmmoTH experiment, create a separate reviewed
500 train + 100 heldout manifest and a **new** output root. Specify only the
selected loss methods, for example `--mixed-manifest ... --run-order
mixed_grpo,mixed_sapo`. Mixed runs do not launch automatically with the six
single-source runs.
