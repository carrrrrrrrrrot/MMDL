# Stage 2 ECD500 GRPO full-run audit

Run: `/data/chaewon/mmdl_stage2_runs/ecd500_grpo_full_20261121` on the RTX PRO 6000. This is the base-parent GRPO candidate, not an MMMU evaluation. The run finished on 2026-11-21 KST.

| Check | Verified result |
| --- | --- |
| Process and update count | Exit 0; 15/15 optimizer updates; all 15 logged actor gradients positive (0.007–0.017). |
| ECD visits | 500 unique training IDs; each attempted **and used exactly 3 times** with fresh generation. 1,500 question visits and 7,500 generated/used responses. |
| Data | Frozen ECD500 JSONL SHA256 `2403dd73164a64b4239c295032b25b2626d6c569ae7f2b0d9b704c82e1b23c8d`. |
| PEFT adapter | Exact reload: 288 tensors, 11,796,480 trainable parameters. Adapter SHA256 `0caac89fb6c8ee927fdd83d6da4f32c00d92a013a6f959b7b329610b9e186918`; independently matched the saved file. |
| Runtime and memory | 3,755 s (1 h 2 min 35 s) end to end; sampled peak GPU memory 47,418 MiB. |
| Internal validation | ECD heldout 97: v1.2 binary accuracy 0.34, `Answer:` present 0.928, cap proxy 0.072. This is a pipeline check, not MMMU performance. |

## Training-rollout diagnostics

All 15 updates used the same 100-question/500-response accepted batch size, constant LR `1e-6`, and no KL or soft-overlong penalty. Across update logs, the raw v1.2 training reward averaged 0.3036 (range 0.222–0.368), `Answer:` presence averaged 0.8976 (range 0.868–0.930), and the 1,024-token cap proxy averaged 0.1025 (range 0.070–0.134). Mean response length averaged 352.2 tokens across updates. The cap proxy counts responses at the configured length; the log does not establish the generation finish reason. Per-update accuracies use different ECD question batches, so the first-to-last values (0.352 to 0.222) do **not** measure a fixed-set accuracy change.

Logged `ppo_kl` and `reward.overlong` were zero in all 15 steps, matching the no-KL and raw binary-reward configuration. The local metrics extraction preserves all per-step values in [metric_summary.json](grpo_full/metric_summary.json).

## Source records

- [Completion report](grpo_full/completion_report.json), [visit audit](grpo_full/visit_audit.json), [adapter reload](grpo_full/adapter_reload.json), and [run manifest](grpo_full/run_manifest.json) are copies from the completed server run.
- The parser is [`parse_full_metrics.py`](../parse_full_metrics.py). The copied console log is under `grpo_full/console.log`; the server retains the original.
- The run-manifest SHA256 independently matched `c717136a43c14c5274dc7837ad70f324ddffb6bd17b2dd46f48a0fc90482eac9`.

This audit establishes that the intended GRPO training ran and saved a loadable adapter. Its completed MMMU fixed-300 screen and the paired DAPO comparison are reported in [STAGE2_FIXED300_REPORT.md](STAGE2_FIXED300_REPORT.md).
