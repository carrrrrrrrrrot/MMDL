# Mixed1000 QC v2: ECD 500 + MAmmoTH CLEVR 500

This is an exploratory GRPO/DAPO/SAPO training input from the common pinned
Qwen3-VL base. It supersedes the untrained mixed1000 v1 input; the interrupted
mixed500 experiment is separate. The manifest has 1,000 train and 100 heldout
rows (50 per source in heldout). SHA256:
`3c46073d60f3b75b5515678b8960290daa8bc55b1c5daf3dabb49ace1e45c636`.
The server path is
`/data/chaewon/mmdl_grpo_data/rl_ablation_20261119_mixed1000_qc_v2/mixed_manifest.jsonl`.

## Change from v1

Three CLEVR train questions with visually ambiguous count golds were excluded:
`mammoth_5933982`, `mammoth_8467823`, `mammoth_7615917`. Three different
CLEVR questions were visually checked and added from A's SFT1 handoff:
`mammoth_9365435`, `mammoth_9456721`, `mammoth_5001946`. Their images and
answers were checked against A's pinned handoff and C's 12,278-row SFT1 item
list by `idx`, source, status and question. The 1,097 other rows, including all
ECD and heldout rows, match v1. This targeted spot check is **not** a full gold
audit or C training approval; manifest fields continue to say false.

The C item list is commit `3abb526`; its Windows CRLF checkout has SHA256
`43159c7b…`, while the same Linux Git blob has SHA256 `c950c507…`. The
builder pins the Linux byte hash; the two versions differ by one CR byte per
line across 12,278 lines.

## Checks and limits

The independent [validation report](validation_report.json) confirms unique
IDs/images, 1,100/1,100 image hashes and decodes, and that each supplied gold
earns v1.2 reward 1 when rendered as `Answer: <gold>`. The exact GRPO 2,000-step
`train.py --validate-only` input check also passed on the 6000 server with
1,000 train rows and the frozen v1.2 reward policy. These checks do not establish
visual correctness of all labels. C's documented MMMU leakage audits cover the
original rows; the three replacements fall in C's SFT1 final audit scope.

The 500 CLEVR training rows include 124 used in A's SFT1 and 343 in A's DPO
pool/pairs. That overlap is not evaluation leakage. The heldout CLEVR half is
biased toward easy base questions and should not rank models. Rank only from
the same fixed MMMU screening IDs and obtain the team's 4090 evaluation before
final claims. The existing 900-item review gate remains closed.

`build_mixed1000_qc_v2.py` reproduces this manifest from the pinned prior
manifest, A handoff and C SFT1 list. `validate_mixed1000_qc_v2.py` performs
the independent image/reward checks.
