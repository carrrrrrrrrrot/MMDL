# GRPO adapter trained from A's exact original DPO(e) model

This PEFT adapter was trained on the bit-identical A DPO(e) merged model whose
4090 three-seed result was `593/586/580` (65.15% average). Both weight shards,
weight index, and all A-listed inference-side nonweight files matched A's
published SHA256 values before training. The earlier B-style-parent GRPO
adapter SHA `7a28ca5b…` is a different experiment.

| Item | SHA256 |
| --- | --- |
| A's original DPO(e) PEFT weights | `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa` |
| A's original merged model shard 1 | `1d9e148aa1a102407bf3dd89b7f5a8d2bfbe792922bee9db243c4b54d03d1b2b` |
| A's original merged model shard 2 | `0c615cbfe078470dfdeb299976580391f8482fed83ac521ec90b3de331b7f0f5` |
| New B GRPO PEFT weights | `235fb028d7de3408c7ddb69834debae47edc0bf4dfbb72b13aba4f0db6ccdb2b` |
| New B GRPO PEFT config | `5eb3af3ebfaba316f4db52e2d56b57a6a5d089fc3af1e8c2984a679382da2299` |

Base: `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

After fetching this branch, run `sha256sum -c SHA256SUMS.txt` from this
directory. Both files must say `OK`. Leave `adapter_config.json` unchanged.

**A's merge:** use `/home/dekim/workspace/MMDL/local/train/merge_lora.py`
with `--base /mnt/SSD-2TB/dekim/merged/dpo_e_trunc`,
`--adapter <this_directory>`, and `--out <new_empty_directory>`. The `--base`
must be A's original merged DPO(e) model, **not** the bare Qwen model and not
the earlier B-style merge. Then run the unchanged 4090 MMMU v1.2 evaluation
at seeds `3407`, `1`, `2` and compare with A's `593/586/580` by seed.

Training used ECD500 EasyR1 GRPO, three fresh passes, five rollouts/question,
15 updates, LR `1e-6`, v1.2 binary reward and no KL or length penalty. Final
checks passed (15 positive-gradient updates, 7,500 rollouts, exact PEFT reload).
The [full report](../../b_handoff_resume/a_exact_e_grpo_20261123/REPORT.md)
has run provenance and validation. **No new MMMU result is claimed yet.**
