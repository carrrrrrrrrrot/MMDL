# Exact A DPO(e) parent for B's GRPO rerun

On 2026-11-23, B rebuilt the merged DPO(e) model on the 6000 with A's shared
`merge_lora.py`, the pinned Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, and A's **original** DPO(e)
PEFT adapter SHA256
`7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa`.
The environment matched A's recorded torch `2.8.0+cu128`, transformers
`4.57.1`, and PEFT `0.21.0` versions. The new B-side parent path is
`/data/chaewon/mmdl_merged_models/dpo_e_a_exact_rebuild_20261123`.

A published the SHA256 values of the **exact merged model evaluated at
65.15%** on the 4090 in
[`a/data_for_b` commit `b08ac1f`](https://github.com/carrrrrrrrrrot/MMDL/blob/b08ac1f/dpo_e_trunc_for_b/eval_4090_s3407/merged_dpo_e_trunc_SHA256.txt).
B checked the rebuilt model against those values before starting GRPO:

| File | A original SHA256 | B rebuild |
| --- | --- | --- |
| `model-00001-of-00002.safetensors` | `1d9e148aa1a102407bf3dd89b7f5a8d2bfbe792922bee9db243c4b54d03d1b2b` | exact match |
| `model-00002-of-00002.safetensors` | `0c615cbfe078470dfdeb299976580391f8482fed83ac521ec90b3de331b7f0f5` | exact match |
| `model.safetensors.index.json` | `c321e968083cc64dbdc77b9cc17842396cf376399f4ff4bca6b0a001fd22afef` | exact match |

The A-listed SHA256 values for `chat_template.json`, `config.json`,
`generation_config.json`, `preprocessor_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `video_preprocessor_config.json`, and `vocab.json`
also **all matched exactly**. `MERGE_INFO.json` differs because it records
server-specific paths and creation time; it is not an inference weight or
processor input. B's `model_manifest.json` in the rebuilt parent records
the original-shard proof and actual B-side file hashes for the training
preflight. Its SHA256 is
`d2aa565c369763862ac0b4e026ec212c388cd95725280b6f991289d7b795b47c`.

The new GRPO run is separate from the earlier B-style-merge experiment. It
starts from the exact A merged parent above and reuses the frozen ECD500
training file SHA256
`2403dd73164a64b4239c295032b25b2626d6c569ae7f2b0d9b704c82e1b23c8d`.
The CPU preflight verified 500 unique image questions, the original EasyR1
source-config SHA256
`737249c0016253b2b92c7f3d3f2e36691f62db4cf850a4e258ce1aafeec2e4a3`,
three fresh passes, five rollouts/question, 100 questions/update, 15 updates,
LR `1e-6`, 1,024 response tokens, and raw v1.2 binary reward. The new run is
`/data/chaewon/mmdl_stage2_runs/e_grpo_a_exact_full_20261123`.

This file establishes **parent identity**, not the result of the in-progress
GRPO training or its MMMU performance.
