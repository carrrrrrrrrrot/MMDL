# A handoff: official 4090 check of DPO(e) → GRPO

B completed the 6000 seed-3407 candidate screen: **574/900**, compared with
the same-6000 DPO(e) parent **562/900** (+12 questions). This is a screening
result and does not replace the 4090 three-seed result.

1. Receive B's **PEFT adapter directory** through the team's weight channel,
   separate from Git. B-side source:
   `/data/chaewon/mmdl_stage2_runs/e_grpo_full_20261122/checkpoints/global_step_15/actor/lora_adapter`.
   Verify `adapter_model.safetensors` SHA256
   `7a28ca5b29f201c33c79800d9f96d0673414590c2b60ac6c940cc00a52f9e69a`.
   The adapter config SHA256 is
   `a66a0709081fc6b52f5431620c499d0a0ddc35cec2521ae6fa44b8c13322180c`.
2. Pin `Qwen/Qwen3-VL-4B-Instruct` revision
   `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Use A's **original**
   DPO(e) adapter SHA256
   `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa`.
   Merge base → DPO(e) → B GRPO in that order with B's
   `b/b_eval_ops/merge_lora.py`; pass role and SHA for each adapter. Check the
   completed model manifest says `parent_kind=merged_dpo_grpo` and has exactly
   those two adapter SHAs in order. [B's merged manifest](merged_model_manifest.json)
   records the expected lineage and file hashes.
3. Run the unchanged 4090 v1.2 `scripts/run_mmmu_eval.sh` with seeds
   `3407 1 2`, 900 items each, and the **merged model path**. Save the run
   metadata, per-item samples, summary, and paired results. Compare with the
   same-seed DPO(e) parent on the **same 4090**. Before reusing A's earlier
   DPO(e) 4090 scores as that parent, compare A's earlier merged-model file
   hashes with [B's DPO(e) parent manifest](../a_dpo_e_parent_20261121/model_manifest.json).
   If the merged weights differ, a fresh same-merge 4090 parent evaluation is
   required for a clean incremental GRPO delta.
4. Please share the earlier 4090 original DPO(e) seed-3407 sample JSONL,
   `run_meta.txt`, and merged-model manifest/file hashes. B's same adapter
   scored **562/900** on the 6000 while A reported **593/900** on the 4090.
   Public evaluation code and adapter SHA agree, so these run artifacts are
   needed to distinguish generation variation from a merge difference.

B's [training report](REPORT.md), [lineage](training_lineage.json),
[900-item paired summary](screen900/comparison_summary.json), and
[item-level CSV](screen900/paired_items.csv) contain the data needed for the
team comparison. Do not upload the adapter weights to this Git branch.
