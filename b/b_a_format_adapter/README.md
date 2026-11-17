# A/C data to B GRPO manifest

`convert_approved.py` adapts A's `a/shared:team_share/convert_v12.py` JSONL
output to B's GRPO data contract. It **requires a C-approved ID list** and
includes only `status=ok` records with existing images and a single scorable
final answer. No MMMU/MMMU-Pro rows are introduced by the adapter; C still
must complete and record its leakage audit.

The approval list may be CSV with `source,id,split` columns, JSONL objects
with those fields, or plain IDs (only when unique across sources). `split`
defaults to `train`; `heldout` is also allowed. The adapter rejects missing
approved IDs, duplicate source/ID pairs, missing images, invalid MC answer,
long or ambiguous open answer, and changed image/text block order. It writes
JSONL plus `<output>.report.json` with source/approval/output SHA256 and
rejection reasons.

```bash
TRAIN_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
$TRAIN_PY /data/chaewon/mmdl_b_a_format_adapter/convert_approved.py \
  --converted-jsonl /path/to/a_converted.jsonl \
  --approved-ids /path/to/c_approved.csv \
  --image-root /path/to/downloaded/images \
  --dataset-revision mammoth-vl-shard_1-<revision> \
  --output /data/chaewon/mmdl_grpo_data/c_approved_v1/manifest.jsonl \
  --processor /home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --render-limit 0
```

Every converted row preserves A's system/user text and image block order.
`--processor` compares the Qwen chat-template rendering of A's original
messages and B's placeholder messages; `--render-limit 0` checks all selected
rows. The adapter outputs `answer_type=mc` or `open`; MC option texts are
stored as an ordered `choices` dictionary for B's loader. B's reward and data
loader must use the team's confirmed **v1.2 scoring policy** for actual C data.

CPU verification on the 6000: five unit cases passed. An integration test
using A's actual `convert_v12.py`, a mock C-approved CSV, and the fixed Qwen
processor passed for one MC and one open record, with exact structural and
rendered prompt parity 2/2; B's `load_manifest` also accepted both records
and their MC option texts. No real C data or GPU training was used.

## Image-size parity

Run `pixel_preflight.py` on a sample image. The installed Qwen processor
defaults to `shortest_edge=65536`, while A/evaluation uses `min_pixels=4096`
and `max_pixels=16777216` plus
`qwen_vl_utils.fetch_image(image_patch_size=16)`. On a 64×64 image, the default
path produced a 16×16 grid (256 patches); the aligned path produced a 4×4
grid (16 patches). B's training code must set this policy and apply the same
fetch step before **real** A/C GRPO data is used. Chat-template parity alone
does not prove pixel parity.

The team `a/shared` branch at `468f7bce6dbdfa6f15841eb18662c9cd8cd05434`
contains documentation, conversion code, reward cases and a common merge
script. It does not contain a common SFT parent adapter or C-approved training
rows; those remain separate handoffs.
