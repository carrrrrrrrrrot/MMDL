# User review before the two full 900-item screens

On 2026-11-19 09:26 KST, the already-running QC-v2 evaluation queue was at the
base fixed-300 stage. Its supervisor pins the SHA256 of `run_eval_queue.py`,
and that queue moves directly from six fixed-300 screens to the first 900-item
screen. Editing the running Python process or its pinned script would not give
a reliable pause. We therefore armed `screen_review_gate.py` against the **six
possible** `*_screen900_s3407` destinations while none existed. It did not
touch the ongoing GPU process, frozen v1.2 evaluation code, or any completed
result. The gate manifest is:

`/data/chaewon/mmdl_eval_runs/rl_ablation_20261119_qcv2_plan/review_gate.json`

The six candidate screens and their ranking should finish normally. At the
first 900-item destination, the old queue will raise `FileExistsError` because
its selected destination contains only `REVIEW_GATE.json`. The old supervisor
will consequently say `failed`; for this exact error, that is the **intended
review stop**, not an evaluation failure. Do not treat any 300-item result as
an official score. The 4090 three-seed evaluation remains separate.

## Inspect the stop and prepare the user review

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
SCRIPT=/data/chaewon/mmdl_b_eval_ops/rl_ablation_20261119/screen_review_gate.py
PLAN=/data/chaewon/mmdl_eval_runs/rl_ablation_20261119_qcv2_plan
ROOT=/data/chaewon/mmdl_eval_runs
"$PY" "$SCRIPT" status --plan-dir "$PLAN" --output-root "$ROOT" --run-prefix rlab19qcv2
```

`status` checks all six markers and the pinned plan. Once `ranking.json`
exists and the queue stopped at the marker, it writes
`status: awaiting_user_review` into `review_gate.json`. Review:

- `ranking.json`: all six fixed-300 scores, baseline counts, missing final
  answers, right-to-wrong and wrong-to-right, top-two selection.
- Each `rlab19qcv2_<candidate>_fixed300/comparison/paired_items.csv`: paired
  errors; compare against the 6000 base fixed-300, not the full-900 count.
- `progress.json` and the old supervisor status: confirm the error names a
  blocked `screen900_s3407` destination and that no full-900 process started.

Show the results to the user and **wait for their review** before release.

## Release only after review

Use the SHA256 of the exact ranking that the user reviewed:

```bash
sha256sum "$PLAN/ranking.json"
"$PY" "$SCRIPT" release --plan-dir "$PLAN" --output-root "$ROOT" \
  --run-prefix rlab19qcv2 --approved-ranking-sha256 <reviewed-sha256>
```

`release` refuses to remove markers unless all six 300 screens produced a
ranking, the old queue stopped at this gate, the plan is unchanged, and each
blocked directory contains **only** its gate marker. It records the selected
two names and reviewed ranking SHA256. The original queue is not resumed in
place: relaunch the same `run_eval_queue.py` command from the supervisor's
`eval_queue` event in `events.jsonl`, keeping identical arguments and a new
separate launch log. The queue verifies and skips every complete existing
base/merge/smoke/fixed-300 artifact and continues to the two 900-item screens.
Do not rerun `continue_queue.py --execute`: its supervisor output root already
exists. Confirm the GPU is idle before relaunch. If the recorded command or
training manifests changed, stop and investigate instead of overwriting any
output.

This manual gate is specific to the already-running SHA-pinned queue. Future
queues should have a native `--pause-before-900` option and a clean paused
status so a controlled stop does not appear as a failed supervisor.
