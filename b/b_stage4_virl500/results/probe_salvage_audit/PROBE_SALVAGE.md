# ViRL500 GRPO probe: independent salvage audit

The launcher reported **failed** (`process_exit_code=127`, `wall_seconds=268`).
This audit does not change that status. Its process log nevertheless shows
that the one scheduled optimizer step completed, its checkpoint and LoRA
adapter were written, and the five-item validation finished before the shell
error at the end.

## Evidence

| Check | Observation | Source |
| --- | --- | --- |
| Training step | `Running step: 100% 1/1`; Step 1, `grad_norm=0.014`, `pg_loss=0.002`, `lr=1e-6` | `launch.log`, lines 3100–3107 and 3221 |
| Reward signal | `critic/rewards`: min 0, mean 0.634, max 1; `reward/accuracy=0.634`; `answer_present=0.748` | `launch.log`, lines 3123–3153 |
| Sampling | 100 `attempt` and 100 `used` visit events, each with 100 unique sample IDs at step 1 | `visits.jsonl` |
| Parameter scope | 288 trainable LoRA tensors, 11,796,480 parameters; no `visual` parameter names | `trainable_params.json` |
| Checkpoint | `global_step_1/actor/lora_adapter/adapter_model.safetensors`, 47,228,912 bytes; SHA256 `3558a1fa02f0fa21bea06ecfaec3208ac69ccb4986a3f04c31bed4df25afe9d7` | checkpoint file and `cpu_peft_reload.json` |
| CPU reload | Base model plus adapter loaded with `CUDA_VISIBLE_DEVICES=''`; all 288 saved tensors matched loaded tensors exactly, all finite; 5,013,487 of 5,013,504 LoRA-B elements nonzero | `cpu_peft_reload.json` |
| Validation | Five-item probe finished; accuracy reward 0.2 and `at_cap_proxy=0.6`. This sample is too small for a performance claim. | `launch.log`, lines 3182–3220 |
| Launcher failure | After validation, shell reported `line 82: config=.../resolved_config.yaml: No such file or directory`; manifest remained `failed` with exit 127. | `launch.log`, line 3225; `run_manifest.json` |

The current launcher file contains a valid command on line 81 followed by a
pipeline continuation on line 82, whereas the error treats `config=...` as a
command. That is consistent with the reported script reread race, but these
artifacts alone do not establish the precise shell timing. The checkpoint is
usable as a **probe artifact**, not a completed full-run result.

The run used the original Stage4 probe manifest SHA256
`c755f15a55e744abac5bb823ebf4e459c7627345b63651a45b398da5e4feff41`.
This is distinct from the later clean-v2 full-run data. No existing run
manifest, console, script, checkpoint, or data file was modified by this audit.
