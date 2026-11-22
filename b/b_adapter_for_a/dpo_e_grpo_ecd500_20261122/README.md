# B GRPO adapter for A's 4090 evaluation

This is the **new ECD500 GRPO PEFT adapter trained on A's original DPO(e)**.
It is not the earlier base-parent ECD500 GRPO or DAPO-core adapter. This
temporary transfer branch holds only the adapter and its verification data;
the training report remains on `codex/b_grpo_impl_20261117`.

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Base revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| A DPO(e) `adapter_model.safetensors` SHA256 | `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa` |
| B GRPO `adapter_model.safetensors` SHA256 | `7a28ca5b29f201c33c79800d9f96d0673414590c2b60ac6c940cc00a52f9e69a` |
| B GRPO `adapter_config.json` SHA256 | `a66a0709081fc6b52f5431620c499d0a0ddc35cec2521ae6fa44b8c13322180c` |

After fetching this branch, run `sha256sum -c SHA256SUMS.txt` from this
directory. Both lines must say `OK`. Do not edit `adapter_config.json` to
replace its B-side `base_model_name_or_path`; the merge script receives an
explicit model object, and changing the file would invalidate its SHA256.

Merge **pinned base -> A original DPO(e) -> this B GRPO adapter**, in that
order. The repository's `b/b_eval_ops/merge_lora.py` accepts the two adapter
directories as repeated `--adapter` arguments, paired with `--adapter-role`
values `dpo` then `grpo` and the two weight hashes above. Pass the local
pinned base snapshot with `--base-model` and a new destination with
`--output-dir`; verify the completed manifest says
`parent_kind=merged_dpo_grpo` and lists the two adapter hashes in order.

Run the unchanged 4090 MMMU v1.2 evaluation on the resulting merged model for
seeds `3407`, `1`, and `2`. Compare against DPO(e) on the same 4090. B's 6000
seed-3407 screen was 574/900 vs. its same-GPU DPO(e) parent 562/900; this is
not the official 4090 result. See the [training and evaluation report](../../b_handoff_resume/e_grpo_ecd500_20261122/REPORT.md)
and [A handoff](../../b_handoff_resume/e_grpo_ecd500_20261122/HANDOFF_FOR_A.md).
