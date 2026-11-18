# A MAmmoTH handoff → B GRPO data

`convert_mammoth_handoff.py` reads A's `a/data_for_b` `list.jsonl` and the
verified image archive. This list is **not** the `convert_v12.py` output, so
`b/b_a_format_adapter/convert_approved.py` cannot consume it directly.

The bridge only emits rows explicitly approved by C. A C review file may be
CSV or JSONL; each record needs these fields:

```json
{"source":"mavis_math_rule_geo","id":"mammoth_123","approval":"approved","mmmu_leakage":"pass","quality":"pass","split":"train","reviewer":"C"}
```

`approval` may be `rejected` (then the row is ignored); for `approved` it
requires both pass statuses, a reviewer, and a `train`/`heldout` split.
Unreviewed rows are never converted. In particular, A's 900 records not used
in SFT1 need C's leakage review before training. The A list's two suspect
questions (`mammoth_8768873`, residual `<image2>`; `mammoth_6572627`, embedded
`Answer:` lines) are rejected even if mistakenly marked approved. C should
decide whether to correct them upstream or omit them.

After verifying A's tar and extracted image hashes, run:

```bash
TRAIN_PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
ROOT=/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118/mammoth_grpo_for_b
$TRAIN_PY /data/chaewon/mmdl_b_data_code/convert_mammoth_handoff.py \
  --list-jsonl "$ROOT/list.jsonl" \
  --approved-ids /path/to/c_mammoth_review.jsonl \
  --image-root "$ROOT" \
  --dataset-revision mammoth-vl-shard1-a61adb899d3c \
  --handoff-commit a61adb899d3c2274ba98cba7a360c5b4f333979f \
  --output /data/chaewon/mmdl_grpo_data/mammoth_c_approved_001/manifest.jsonl
```

Use a **new** output path for every conversion. The bridge checks all selected
image file sizes and SHA256 values, writes absolute image paths, preserves
`source`, `used_in`, `base_k_correct`, `text_only_suspect`, `part`, source
answer type, A commit, and C approval SHA256, and produces
`manifest.jsonl.report.json` with source/split counts. This manifest is
accepted by `b/b_grpo_main/data.py`; reward policy must be `mode=v1_2`.
MC has 2–5 options and remains `answer_type=mc`; numeric and short word gold
use `answer_type=open`, matching v1.2's open answer scoring. Image is inserted
immediately after `Question: ` in a structured user message and no literal
`<image N>` token is sent to TRL. The v1.2 system prompt and answer instructions
match A's `convert_v12.py` and the evaluation YAML constants. A's handoff
`question` omits the original `<image>` location, so the exact placement of
the image within A's original question **cannot be reconstructed**. The bridge
uses one documented, consistent placement; rendered text and final-answer
instruction parity do not prove original image interleave parity.

The output is a **data preparation artifact**, not permission to start GRPO.
The first real training run also needs an agreed parent checkpoint, data split,
and run configuration.

## 6000 integration check (11/18)

The B server's older `reward.py` did not accept open answers. Its existing
`data.py` and `reward.py` were backed up with `.pre_mammoth_20261118_*` suffixes,
then the current repository `data.py`, `reward.py`, and `v12_scorer.py` were
copied into `/data/chaewon/mmdl_b_main_code/`. Their SHA256 values are
`ef5e1012aa2ca54c531550bc7c5105457308e4db6a9f46a65ec35218705fc0ad`,
`ebd130b8d758c79dab0ae503c44e788ba97d39772c8e494db38ceaa4c9895f55`,
and `c8aaa9dff3764205e196762acbc6b470ee804cd5dd7579a2eec82a8588fc7c5b`.
`test_convert_mammoth_handoff.py` passed 3/3 tests with actual A MC4, MC5,
numeric, and word examples: it verified image SHA and byte size, loaded all
four via the B training loader, and rejected missing C leakage approval plus
two malformed questions. The approval rows used for this test had reviewer
`mock-test-only`; they are not C approvals. No MAmmoTH training has run.
