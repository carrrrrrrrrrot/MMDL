# B mixed1000 v1: ECD 500 + MAmmoTH CLEVR 500

Prepared on 2026-11-19 as a **new exploratory input** for a larger-data mixed-source RL ablation. This folder contains data and CPU validation evidence only. **No mixed1000 model has been trained or evaluated.** It does not replace `rl_20261119_mixed_v2` or change the 900-item review gate.

## Pinned inputs and selection

| Source | Input manifest SHA256 | Train | Heldout |
| --- | --- | ---: | ---: |
| ECD qc-v2 | `c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed` | all 500 | 50 of 97 |
| MAmmoTH CLEVR v3 | `b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d` | all 500 | 50 of 100 |

`build_mixed1000_v1.mjs` checks these source hashes. It includes every source training row unchanged and selects 50 heldout rows per source with the same fixed SHA256 ranking as mixed_v2, so the 100 heldout sample IDs are **identical** between the mixed500 and mixed1000 manifests. It puts all training rows first and heldout rows second, each in a deterministic SHA256 order (`mmdl-b-mixed1000-v1`, seed 3407). The builder refuses to overwrite an existing result.

The [manifest](mixed_manifest.jsonl) contains 1,000 train + 100 heldout and has SHA256 `e71df0141dabc9d696cf67e5b47d35b05801633d145d855ca83b7efdbb95637e`. All 1,100 sample IDs, original IDs, questions, image paths and image hashes are unique. Every row is byte-equivalent to its source JSONL row apart from file order; the independent Python validator checks parsed row equivalence. [Selection counts](selection_report.json) include the source hashes and A overlap counts.

## Validation and limits

On the 6000 server, B `train.py --validate-only` returned `status=validated` with 1,000 train rows and the frozen v1.2 reward policy. This was a CPU-only **input check** using the base model path; the example `max_steps=1000` and default DAPO settings in [that record](train_validate_only.json) do **not** commit to a training configuration. The separate [validation report](validation_report.json) checked all 1,100 source rows, file presence, image SHA256 and decode, and that `Answer: <gold>` scores 1 under the v1.2 reward. The server manifest is `/data/chaewon/mmdl_grpo_data/rl_ablation_20261119_mixed1000_v1/mixed_manifest.jsonl` with the same SHA256.

These checks establish file integrity and reward-format consistency, **not visual answer correctness**. C's audits found no MMMU/MMMU-Pro leakage within their documented scopes; C has not approved every gold answer or this mixed set for training. Every manifest row retains `c_training_approval:false` and `visual_gold_quality_verified:false`. ECD qc-v2 removes known incorrect/ambiguous examples and average/mean/median questions, but other label errors can remain. CLEVR v3 has only a small visual spot check.

The CLEVR training half includes **121** items used in A's SFT1 and **346** in A's DPO pool or pairs. These are overlaps with A's experiments, not proof of evaluation leakage. The 50 CLEVR heldout items were intentionally chosen from `base_k_correct=4` and are an easy, biased regression set; do not rank methods by this heldout score alone. Use paired MMMU v1.2 screening for exploratory ranking and the agreed 4090 official evaluation for final claims.

## Rebuild and revalidate

Run from the B repository root with Node.js and the pinned source manifests present; choose an empty output directory:

```bash
OUT_DIR=$(mktemp -d)
node b/b_grpo_data/rl_20261119_mixed1000_v1/build_mixed1000_v1.mjs \
  --output "$OUT_DIR/mixed_manifest.jsonl"
sha256sum "$OUT_DIR/mixed_manifest.jsonl"
```

The builder also accepts `--ecd` and `--clevr` paths but still enforces their pinned SHA256 values. To repeat full server validation, run `validate_mixed1000_v1.py` with `--manifest`, `--ecd`, `--clevr`, `--reward-policy`, `--code-dir`, and `--output` paths. Its report does not alter training or evaluation outputs.
