# Stage 4 ViRL500 exploratory GRPO / DAPO core

This package uses the frozen EasyR1 engine and the same optimizer, rollout, LoRA, reward, and response-cap settings as Stage 2 ECD500. The base is Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. ViRL500 has **not** received C's full semantic/OCR and manual quality approval; results must be labeled exploratory.

## Frozen data packages

- Original probe package: `/data/chaewon/mmdl_stage4_virl500_20261121`, data manifest SHA256 `c755f15a55e744abac5bb823ebf4e459c7627345b63651a45b398da5e4feff41`. It contains 500 training rows and a five-item probe holdout. Later image audit found pixel-identical duplicates in the broader train/97 split; this probe remains a pipeline check only.
- Clean-v2 full package: `/data/chaewon/mmdl_stage4_virl500_20261121/clean_v2_candidate`, manifest SHA256 `e040e19beee588828c3d0bb602b4b4898a9718152c04d5368ff8e6e05a9c9773`. It contains 500 training rows and 97 validation rows, all with distinct IDs and no within-split or cross-split pixel-identical images. Train SHA256 `f033f9e7c5b17ee351a53e993dc1f1d7bea25a924b109d674a24a97dd940eebe`; validation SHA256 `7f8e8e5ebebc1173af41354422aacfe3c4557813f59085436fbe63119f9f584d`.

The CPU preflight requires exact dataset and selection SHA256, 500 unique selected QIDs, v1.2 gold self-score, system/prompt/image parity, MC options and answer consistency, image decoding and file SHA256, Qwen processor prompt-length scan, and source-config parity with Stage 2. Full mode additionally requires zero pixel-identical image duplicates within train/validation and across their boundary. The read-only full GRPO CPU preflight passed on 2026-11-21 KST.

## One-GPU training and evaluation

The single-run launcher checks GPU idle after its CPU preflight. Use the original package only for the one-step probe. Use clean-v2 for every full run and DAPO probe:

```bash
bash /data/chaewon/mmdl_stage4_virl500_20261121/run_virl500.sh grpo probe /data/chaewon/mmdl_stage4_runs/virl500_grpo_probe_20261121
MMDL_STAGE4_DATA_ROOT=/data/chaewon/mmdl_stage4_virl500_20261121/clean_v2_candidate bash /data/chaewon/mmdl_stage4_virl500_20261121/run_virl500.sh grpo full /data/chaewon/mmdl_stage4_runs/virl500_grpo_full_20261121
```

The original GRPO probe completed one optimizer update, wrote a loadable adapter, and validated five questions, but its launcher exited 127 after the live shell file was rewritten. Its manifest **must remain `failed`**. The separate [probe salvage audit](results/probe_salvage_audit/PROBE_SALVAGE.md) records exactly what can be reused as pipeline evidence. The clean-v2 GRPO full run started separately and is guarded by the new CPU preflight.

The watchdog queue was started once with PID `1778249` on 2026-11-21 KST. It accepts that failed probe only through explicit, checksum-verified salvage evidence; waits for the already running GRPO full run without launching a duplicate; then runs DAPO probe/full and sequentially evaluates both full adapters with v1.2 smoke20 and the same 300 MMMU items/seed 3407/base185. It stops on any failed or timed-out training/evaluation gate. The launcher, preflight, finalizer, queue, and evaluator must remain unchanged while this queue is active.

```bash
cat /data/chaewon/mmdl_stage4_runs/virl500_queue_20261121/status.json
tail -f /data/chaewon/mmdl_stage4_runs/virl500_queue_20261121.launch.log
```

The evaluation gate requires exactly 300 paired rows and unique question IDs, baseline 185/300, identical input and document hashes, and both evaluation processes exiting 0. Training completion requires the expected optimizer steps, positive gradients, visit audit, and an exactly reloaded PEFT adapter. DAPO core may discard many rollouts: Stage 2 ECD500 used only 115 of 500 unique IDs. The ViRL500 DAPO visit audit must be read before describing its effective data exposure.
