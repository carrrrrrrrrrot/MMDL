# B QC-v2 automatic continuation on the 6000

`continue_queue.py` takes over **after** the separately launched ECD qc-v2 GRPO run. It waits for that run to pause with a complete, hashed adapter and a nonzero learning signal. It then uses one GPU job at a time:

1. MAmmoTH CLEVR v3 GRPO, pause and check learning signal.
2. ECD qc-v2 DAPO and SAPO using the existing ECD plan.
3. CLEVR v3 DAPO and SAPO if its GRPO had learning signal. If CLEVR GRPO had no signal, retain the failure and omit its two variants.
4. With at least two complete candidates, run the existing 6000 evaluation queue: fixed 300 v1.2 questions for all, full 900 seed 3407 for the top two, and adaptive seed 1. The final 4090 official evaluation is outside this script.

Both datasets are exploratory, with known limits in gold validation. All six train runs use the same base revision, reward policy, 512 completion token cap, four rollouts, two iterations, and 1,000 optimizer steps. The supervisor checks pinned input hashes and each completed adapter. Any unexpected failure stops downstream work. No failed or incomplete training/evaluation directory is overwritten.

## Invocation on the 6000

Read-only preflight:

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
SCRIPT=/data/chaewon/mmdl_b_main_code/rl_ablation_20261119/continue_queue.py
"$PY" "$SCRIPT" --check
```

Run **once**, after reviewing `--check` output and confirming the separate ECD GRPO process is active or paused:

```bash
"$PY" -u "$SCRIPT" --execute \
  --output-root /data/chaewon/mmdl_grpo_runs/rl_ablation_20261119_supervisor_qcv2
```

The supervisor output root must not exist before starting. It writes `plan.json`, `status.json`, `events.jsonl`, `logs/<stage>.log`, and finally `summary.json`. `status.json` includes training step or evaluation stage in its heartbeat. The underlying ECD and CLEVR roots retain their own `status.txt`, `events.jsonl`, trainer logs, `run_manifest.json`, rollout files, adapters and hashes. The evaluation plan writes its own `progress.json`, `events.jsonl`, rankings and paired results.

Expected paths and SHA256 are hardcoded in the script to prevent accidental substitution. It checks the ECD first-run `plan.json`, the frozen v1.2 worktree HEAD `61025850563d0331e365bc974a63c0d3c4ed4c2c` and clean status, and refuses changed manifests, train/reward/evaluation code, an occupied GPU at stage boundaries, an existing CLEVR output root, or an existing evaluation prefix. If the supervisor fails, preserve its directory and inspect the last event and stage log before any new attempt. Long-stage timeouts **leave the child job running** and stop the supervisor, so check the reported PID and GPU before intervening.

The read-only `--check` passed on the 6000 at 2026-11-19 02:07 KST. Independently recomputing `orchestrate.py`'s ECD plan with the resume command gave exact JSON equality with the running plan and left its SHA256 unchanged (`d8bfd29c14332b69ac9d02d47de8fe88b1d6ae7de717b85286266d08d6f2b394`). `--pause-after` is not part of the plan, so resuming after the first run preserves that file; only `summary.json` advances from `paused` to `running` and then `complete`.

The ECD first run's 512-token cap may truncate final answers; see the rollout diagnostic report. This continuation preserves 512 for a controlled loss-method comparison. A later 1024-token ablation requires a new run root and a separately labeled comparison.
