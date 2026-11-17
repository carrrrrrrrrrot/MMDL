# C-selected ECD charts for B GRPO

`convert_c_ecd.py` accepts C's reviewed JSONL directly. Each row needs `id`,
`question`, `final_answer` (or `answer`), and `image_relative_path` (or
`image`). `source` may be omitted, `ECD`, or `ECD-reasoning`; `split` may be
omitted, `train`, or `heldout`. The original C file is passed unchanged to
`convert_approved.py --approved-ids`, so its SHA256 remains the approval
provenance. C must separately attest MMMU leakage and visual reasoning review.

Example row, with `--image-root /data/chaewon/mmdl_grpo_data/ecd_10k`:

```json
{"id":"ecd_012345_42","question":"What value is marked on the chart?","final_answer":"10","image_relative_path":"images/012345.png"}
```

The PNG must then exist at
`/data/chaewon/mmdl_grpo_data/ecd_10k/images/012345.png`. A's SFT1 list has
the same `images/<chart>.png` convention. The adapter rejects reused SFT1 IDs
and ECD chart images, a second question for any selected chart, missing images,
absolute or parent-relative paths, and final answers longer than 40 characters.
If any row fails, it writes only a rejection report and exits; no partial
training file is produced.

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
BASE=/data/chaewon/mmdl_grpo_data/ecd_10k
$PY b/b_a_format_adapter/convert_c_ecd.py \
  --selection-jsonl "$BASE/c_selected.jsonl" \
  --sft1-list team_share/c_list_sft1.jsonl \
  --image-root "$BASE" \
  --a-converter team_share/convert_ecd.py \
  --output "$BASE/c_selected_a_style.jsonl"

$PY b/b_a_format_adapter/convert_approved.py \
  --converted-jsonl "$BASE/c_selected_a_style.jsonl" \
  --approved-ids "$BASE/c_selected.jsonl" \
  --image-root "$BASE" \
  --dataset-revision ECD-10k-Images-284489f \
  --output "$BASE/grpo_manifest.jsonl" \
  --processor /home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --render-limit 0
```

The bridge imports A's `PRE_PROMPT` and `OPEN_PROMPT` constants from the shared
converter and uses the same system/user text and image block order. The C file
contains only the final answer, so the generated assistant text is `Answer:
<final_answer>` without invented reasoning. `status=ok` here means a reviewed,
scorable **GRPO gold answer**; it does not claim the row satisfies A's SFT
reasoning-length rule. The conversion report records hashes of C's selection,
A's SFT1 list, both A scripts, and the converted output.

CPU check using A's converter and B's loader:

```bash
$PY b/b_a_format_adapter/test_convert_c_ecd.py \
  --a-converter team_share/convert_ecd.py \
  --sft1-list team_share/c_list_sft1.jsonl
```

Add `--processor <Qwen snapshot>` to also compare rendered chat templates.
