# Stage 2 ECD500 DAPO core full-run audit

Run: `/data/chaewon/mmdl_stage2_runs/ecd500_dapo_core_full_20261121` on the RTX PRO 6000. Training finished at 2026-11-21 04:02:13 KST. The label **DAPO core** is deliberate: online group filtering, asymmetric clipping, and token-level loss were enabled, while the soft-overlong penalty was disabled to retain the team's v1.2 binary reward. This is not a full original-DAPO replication or an MMMU result.

| Check | Verified result |
| --- | --- |
| Process and updates | Exit 0; 15/15 optimizer updates; every logged actor gradient norm positive, 0.027–0.037. Constant LR `1e-6`, KL 0. |
| Source and model | Same Qwen3-VL-4B-Instruct base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` and same frozen ECD500 train file as the GRPO full run; train SHA256 `2403dd73164a64b4239c295032b25b2626d6c569ae7f2b0d9b704c82e1b23c8d`. Five rollouts per question, 1,024-token response cap, 100 accepted question visits per update. |
| Question exposure | All 500 IDs were attempted 21–22 times, but only **115 unique IDs** were used for updates; **385 IDs were never used**. Across 15 updates: 10,550 attempted visits, 1,500 used visits. Each accepted batch had 68–83 distinct IDs among 100 visits, so it included 17–32 repeated ID visits. |
| Rollout exposure | 52,750 responses generated, 7,500 used, 45,250 discarded; accepted fraction **14.22%**. Each update used 500 rollouts from 100 accepted question visits. |
| PEFT adapter | Recorded exact reload of 288 tensors / 11,796,480 trainable parameters. Saved final adapter SHA256 `55b3e67d4877a9604b1feef88826314331c5ee311af89fb99bc3736ab8a9bed3`, independently matched against the server's `adapter_model.safetensors`. The trainable-parameter audit lists only language-model attention q/k/v/o LoRA tensors. |
| Runtime and memory | 9,933 s (2 h 45 min 33 s) end to end; sampled GPU-memory peak 50,766 MiB. |
| Internal validation | ECD heldout 97: v1.2 binary accuracy 0.34, `Answer:` presence 0.928, cap proxy 0.072. This is a pipeline check, not MMMU transfer performance. |

## GRPO comparison and interpretation

| Measure | GRPO full | DAPO core full |
| --- | ---: | ---: |
| Optimizer updates with positive gradient | 15/15 | 15/15 |
| Unique training IDs used | 500/500 | **115/500** |
| Used question visits | 1,500 (each ID exactly 3) | 1,500 (accepted IDs used 1–22 times) |
| Generated / used rollouts | 7,500 / 7,500 | 52,750 / 7,500 |
| End-to-end time | 3,755 s | 9,933 s (**2.65×**) |
| Sampled GPU peak | 47,418 MiB | 50,766 MiB |
| ECD97 internal validation accuracy | 0.34 | 0.34 |

Both runs completed 15 nominal outer rollout cycles, but **DAPO did not train on each of the 500 IDs for three epochs**. Dynamic filtering repeatedly drew and regenerated the 500-question pool, then admitted only groups with mixed v1.2 outcomes. The accepted subset was strongly concentrated: 54 IDs were used at least 15 times, while 385 had zero updates. Therefore, the fixed-300 MMMU screen must compare these as two complete training *procedures*, not as an isolated loss-function comparison with matched ID exposure. The equal ECD97 validation value does not establish equal MMMU performance.

## Training-rollout diagnostics

Across the 15 accepted update batches, the logged raw binary reward averaged **0.3037** (per-update range 0.289–0.322), `Answer:` presence averaged **0.9005** (0.895–0.908), and the 1,024-token cap proxy averaged **0.0999** (0.092–0.106). Mean response length averaged **388.2 tokens** (358.5–405.1). All 15 logs have zero `ppo_kl` and zero `reward.overlong`, consistent with the configured binary reward and disabled length penalty. The cap proxy counts responses at the configured limit; generation finish reasons were not logged, so it does not by itself prove truncation. These per-update figures use changing accepted questions and cannot be read as a fixed-set accuracy trend.

The GRPO full run logged mean raw reward 0.3036, `Answer:` presence 0.8976, cap proxy 0.1025, and mean response length 352.2. Those are *training-batch diagnostics* from different accepted populations. The DAPO core run was materially slower because it generated about 7.03 times as many rollouts to fill the same accepted update count. Neither run's training reward should be used to rank MMMU candidates.

## Evidence and limits

- [Completion report](dapo_full/completion_report.json), [visit audit](dapo_full/visit_audit.json), [visits](dapo_full/visits.jsonl), [adapter reload](dapo_full/adapter_reload.json), [trainable parameters](dapo_full/trainable_params.json), [resolved config](dapo_full/resolved_config.yaml), and [run manifest](dapo_full/run_manifest.json) are read-only copies from the server. The console log is at `dapo_full/console.log` and [per-step parsed metrics](dapo_full/metric_summary.json) were extracted with [`parse_full_metrics.py`](../parse_full_metrics.py).
- The copied manifest, visit audit, resolved config, and console SHA256 values independently matched the completion report. The adapter-file hash was independently checked on the server. The exact PEFT reload was performed by the completed-run finalizer and recorded in `adapter_reload.json`; this audit did not load the model or use the GPU.
- The ECD500 dataset is exploratory B QC v2. C's MMMU leakage audit passed; full gold-label approval is not claimed. A shared SFT-parent comparison is not established by this base-revision run.
- The completed fixed-300 MMMU screening is in [STAGE2_FIXED300_REPORT.md](STAGE2_FIXED300_REPORT.md). The team's 900-item/4090 official evaluation is required for a final claim.
