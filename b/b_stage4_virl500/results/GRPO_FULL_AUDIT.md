# Stage 4 ViRL500 GRPO full-run audit

Run: `/data/chaewon/mmdl_stage4_runs/virl500_grpo_full_20261121` on the RTX PRO 6000. Finished 2026-11-21 12:12:26 KST. This is the **clean-v2 ViRL500** exploratory data comparison, separate from the earlier failed-launcher probe. It is not an MMMU result.

| Check | Verified result |
| --- | --- |
| Process and updates | Exit 0, `status=complete`; 15/15 optimizer updates and 15/15 positive actor gradients. Per-update gradient norm 0.010–0.016. The independent queue completion gate returned `complete`. |
| Question exposure | All 500 unique clean-v2 QIDs were attempted **and used exactly three times**. Every update used 100 distinct IDs, with five fresh rollouts per question: 1,500 question visits and 7,500 generated/used responses. No online filtering. This is three actual generation passes for GRPO. |
| Frozen data | Clean-v2 manifest SHA256 `e040e19beee588828c3d0bb602b4b4898a9718152c04d5368ff8e6e05a9c9773`; train JSONL SHA256 `f033f9e7c5b17ee351a53e993dc1f1d7bea25a924b109d674a24a97dd940eebe`; ViRL97 validation SHA256 `7f8e8e5ebebc1173af41354422aacfe3c4557813f59085436fbe63119f9f584d`. These were independently checked against the files on the server. Train/validation had zero identical-pixel overlaps in the CPU preflight. |
| Configuration | Pinned Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`; constant LR `1e-6`; 1,024 response-token cap; five rollouts/question; 100 questions/update; KL and overlong penalty zero. Stage2 source-config parity SHA256 `737249c0016253b2b92c7f3d3f2e36691f62db4cf850a4e258ce1aafeec2e4a3`. |
| PEFT adapter | Exact reload recorded: 288 tensors, 11,796,480 trainable parameters, all language attention q/k/v/o LoRA. Final adapter SHA256 `d7ebe542796f93b855792d2edcb99aa2d6bced8362d5ccc7e515b58e32916e02`, independently matched on the server. |
| Runtime and memory | 3,055 s (50 min 55 s) end to end; sampled GPU-memory peak 45,550 MiB. |
| Internal validation | ViRL heldout 97: v1.2 binary accuracy 0.381, `Answer:` presence 0.474, 1,024-token cap proxy 0.526. This is only a pipeline/holdout check; fixed-300 MMMU screening is still required. |

## Training-rollout diagnostics

Across 15 equally sized update batches, the raw v1.2 reward averaged **0.5832** (per-update range 0.534–0.640), `Answer:` presence averaged **0.7241**, and the 1,024-token cap proxy averaged **0.2777**. Mean response length averaged **552.7 tokens**. All steps logged `actor.lr=1e-6`, `actor.ppo_kl=0`, and `reward.overlong=0`. The cap proxy counts responses at the configured limit; the console does not record finish reasons, so a definite truncation rate is unavailable. Per-step rewards use different questions and cannot be interpreted as a fixed-set improvement trend.

The heldout cap proxy of 0.526 and `Answer:` presence of 0.474 flag a response-format/length risk for this data distribution. The ECD97 and ViRL97 internal validation sets differ, so their accuracies and training rewards should not be compared as if they were the same benchmark. The upcoming fixed-300 MMMU evaluation supplies that common comparison.

## Evidence and limits

- Compact copies: [completion report](grpo_full/completion_report.json), [run manifest](grpo_full/run_manifest.json), [visit audit](grpo_full/visit_audit.json), [adapter reload](grpo_full/adapter_reload.json), [trainable parameters](grpo_full/trainable_params.json), [resolved config](grpo_full/resolved_config.yaml), and [15-step metrics](grpo_full/metric_summary.json).
- The copied manifest, visit audit, and resolved config SHA256 values matched the completion report. The final adapter-file hash was independently checked on the server. The exact PEFT reload is the run finalizer's recorded result; this audit did not reload the model or use the GPU.
- The original server console log remains at `/data/chaewon/mmdl_stage4_runs/virl500_grpo_full_20261121/console.log` with SHA256 `b467466b067e022880e804471578ac5ada28ab6f4d5178d406cf47080ae966c6`; the compact metrics JSON was parsed from it using [`parse_full_metrics.py`](../parse_full_metrics.py).
- C's complete MMMU/MMMU-Pro Vision OCR, semantic, and manual quality review is still pending. This training is exploratory and should not be presented as an approved data result. The 6000 fixed-300 screen and later official 4090 evaluation determine transfer performance.
