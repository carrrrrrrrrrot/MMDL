# A's exact 65.15% DPO(e) merge → B ECD500 GRPO

**Status: training and final validation complete on 2026-11-23.** This is a
new GRPO adapter, separate from the earlier B-style DPO(e) merge → GRPO
adapter SHA `7a28ca5b…`. The new adapter has not yet been evaluated on MMMU.

## Parent identity

B used A's shared BF16 CPU merge script with the original DPO(e) adapter SHA
`7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa`.
Both merged weight shard SHA256 values, the weight index, and all
evaluation-relevant nonweight file SHA256 values matched A's 4090 model
evaluated at 65.15% exactly. See [the parent proof](PARENT_PROVENANCE.md)
and [A's published hashes](https://github.com/carrrrrrrrrrot/MMDL/blob/b08ac1f/dpo_e_trunc_for_b/eval_4090_s3407/merged_dpo_e_trunc_SHA256.txt).

## Frozen training recipe

- Parent: `/data/chaewon/mmdl_merged_models/dpo_e_a_exact_rebuild_20261123`.
- Data: ECD QC-v2 500 unique image questions, training JSONL SHA256
  `2403dd73164a64b4239c295032b25b2626d6c569ae7f2b0d9b704c82e1b23c8d`.
- EasyR1 GRPO, raw MMMU v1.2 0/1 reward, no KL or length penalty.
- Three freshly generated passes, five responses/question, 100 question
  groups/update, 15 updates, constant LR `1e-6`, 1,024 response
  tokens, language-attention LoRA r16/alpha32.
- CPU preflight passed before the GPU run. Run path:
  `/data/chaewon/mmdl_stage2_runs/e_grpo_a_exact_full_20261123`.

## Verified outcome

- Exit code `0`; final report status `complete`.
- 15/15 optimizer updates with positive gradient in all 15.
- 500 unique training questions visited exactly three times each;
  7,500/7,500 generated rollouts were used.
- PEFT reloading recovered all 288 adapter tensors and 11,796,480 trainable
  parameters exactly.
- Wall time 3,819 seconds (63 minutes 39 seconds); peak GPU memory 47,434 MiB.
- Final ECD validation accuracy `0.289` on 97 held-out rows. This is the
  training run's validation metric, **not** MMMU performance.
- New adapter weights SHA256:
  `235fb028d7de3408c7ddb69834debae47edc0bf4dfbb72b13aba4f0db6ccdb2b`.
  Adapter config SHA256:
  `5eb3af3ebfaba316f4db52e2d56b57a6a5d089fc3af1e8c2984a679382da2299`.

The [completion report](completion_report.json), [run manifest](run_manifest.json),
[visit audit](visit_audit.json), [adapter reload audit](adapter_reload.json),
and [resolved configuration](resolved_config.yaml) are committed alongside
this report.

## Evaluation handoff

The new adapter passed the 15-step, 7,500-rollout, 500-ID ×3-visit,
positive-gradient, and PEFT-reload checks. A should merge **its existing
exact original DPO(e) model → the new GRPO adapter**. In A's merge script,
pass the original merged DPO(e) folder explicitly as `--base`; otherwise the
default is the unmodified Qwen base. The original A parent path in its
`MERGE_INFO.json` is `/mnt/SSD-2TB/dekim/merged/dpo_e_trunc`.

```bash
python /home/dekim/workspace/MMDL/local/train/merge_lora.py \
  --base /mnt/SSD-2TB/dekim/merged/dpo_e_trunc \
  --adapter <new_adapter_directory> \
  --out <new_empty_merged_model_directory>
```

Evaluate the resulting model with the unchanged 4090 v1.2 pipeline at seeds
`3407`, `1`, and `2`. Compare same seeds against A's exact original DPO(e)
scores `593/586/580`, retaining the per-item and run metadata. This new
adapter has not yet been evaluated and the earlier `7a28ca5b…` adapter must
not be substituted.
