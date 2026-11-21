# Stage 2 ECD500 GRPO / DAPO core

This runs on the 6000 server with the pinned EasyR1 package at
`/data/chaewon/mmdl_stage1_paper_style_20261120` and its isolated training
environment. The base is Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

## Conditions

Both methods use the same 500 ECD IDs and images, seed 3407, Qwen v1.2 image
processing and binary 0/1 answer reward, LoRA on language q/k/v/o only,
five fresh responses per question, LR 1e-6 constant, and no KL penalty. The
response cap is 1024. The ECD500 file is B's exploratory QC v2 selection;
C's MMMU leakage audit passed, but full gold approval is not claimed.

The data loader draws 50 questions at a time and the trainer collects 100
questions for each policy update. GRPO consumes two draws per update. The 500
IDs yield 10 loader batches and five updates per pass, for 15 updates over
three fresh-generation passes. The visit audit must confirm that each ID was
attempted and used exactly three times. `ppo_epochs=1` gives one optimizer
update per collected 100-question, 500-response batch.

DAPO core adds asymmetric clip 0.2/0.28, token-level loss, and online filtering
of groups whose raw v1.2 rewards are all equal. It may draw more than two
50-question batches for one update; the bound is 24 draws, at most 1,200
question attempts and 6,000 responses per update. Fifteen nominal updates do
not imply exactly three visits per ID. The soft-overlong penalty is disabled,
so this is **DAPO core**, not full DAPO.

## Commands

Each command needs a distinct empty run directory. The launcher runs CPU
preflight first, checks that the GPU is idle, then records a console log,
five-second GPU memory samples, ID visits, checkpoints, and a completion
report. It does not start a second GPU job automatically.

```bash
bash /data/chaewon/mmdl_stage2_ecd500_20261120/run_ecd500.sh grpo probe /data/chaewon/mmdl_stage2_runs/ecd500_grpo_probe_RUNID
bash /data/chaewon/mmdl_stage2_ecd500_20261120/run_ecd500.sh dapo_core probe /data/chaewon/mmdl_stage2_runs/ecd500_dapo_probe_RUNID
bash /data/chaewon/mmdl_stage2_ecd500_20261120/run_ecd500.sh grpo full /data/chaewon/mmdl_stage2_runs/ecd500_grpo_full_RUNID
bash /data/chaewon/mmdl_stage2_ecd500_20261120/run_ecd500.sh dapo_core full /data/chaewon/mmdl_stage2_runs/ecd500_dapo_full_RUNID
```

Probe mode performs one update and validates five heldout questions solely to
check the pipeline. Full mode performs 15 nominal updates and validates all
97 heldout questions at the end. Both save a PEFT adapter. Full mode saves
optimizer and dataloader state every three updates, retaining two checkpoints.

`completion_report.json` appears only after the process exits successfully,
ID coverage and finite metrics pass, and the final adapter reloads exactly
with PEFT. A probe result is not an MMMU performance measurement. Later
candidate selection uses the fixed 300-question evaluation, then 900-question
evaluation for shortlisted adapters.

## Compare the two fixed300 results

After the guarded queue finishes both evaluations, use the read-only paired
comparator. It requires the same seed 3407, the same 300 IDs and documents,
the same v1.2 scorer and 6000 base run (185/300), and complete 300-row CSVs.
It reports each candidate's wrong-to-right and right-to-wrong counts as well
as the questions where only one method is correct. It never launches a GPU job.

```bash
/data/chaewon/mmdl_envs/easyr1_stage1_20261120/bin/python \
  /data/chaewon/mmdl_stage2_ecd500_20261120/compare_stage2_fixed300.py \
  --grpo-comparison /data/chaewon/mmdl_eval_runs/stage2_ecd500_grpo_20261121_fixed300/comparison \
  --dapo-comparison /data/chaewon/mmdl_eval_runs/stage2_ecd500_dapo_core_20261121_fixed300/comparison \
  --output-json /data/chaewon/mmdl_stage2_ecd500_20261120/STAGE2_FIXED300_COMPARISON.json \
  --output-md /data/chaewon/mmdl_stage2_ecd500_20261120/STAGE2_FIXED300_COMPARISON.md
```

This is a fixed-ID, single-seed diagnostic. The official 900-item evaluation
and 4090 final 3-seed evaluation remain separate stages.
