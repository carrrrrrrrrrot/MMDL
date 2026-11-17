# B synthetic GRPO insurance data

This generator creates original visual multiple-choice questions for an
independent **base → GRPO** insurance run while the checked external data and
common SFT parent are pending. It does **not** use MMMU or MMMU-Pro examples.
The resulting adapter is not a common-parent GRPO result and any MMMU score
must be reported separately as `base → GRPO (synthetic training data)`.

It generates three tasks: red-square counting, tallest-bar identification,
and yellow-star location on a grid. Defaults: 512 examples per task, 1,536
total; 1,344 train and 192 heldout. Answers A–D are exactly balanced within
each task and split. The heldout set uses distinct image files and SHA256s.

```bash
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_data_code/generate.py \
  --output-dir /data/chaewon/mmdl_grpo_data/synthetic_v1_1536 \
  --per-task 512 --seed 3407
/data/chaewon/mmdl_envs/grpo_pilot/bin/python /data/chaewon/mmdl_b_data_code/qa.py \
  --dataset-dir /data/chaewon/mmdl_grpo_data/synthetic_v1_1536
```

`manifest.jsonl` stores each image's relative path and SHA256, source,
revision, split, original ID, question, A–D choices, gold letter, and raw
system/user messages. The user message omits a literal `<image 1>` because
TRL inserts the image through the processor. `metadata.json` includes the
manifest SHA256 and class counts. The generator refuses an existing output
directory; reruns must use a new directory.

This dataset is intentionally narrow and procedural. A positive heldout score
only checks that the training code learned these tasks; it does not establish
MMMU improvement or generalization. Use C's checked real data for the shared
SFT-parent branch when available.
