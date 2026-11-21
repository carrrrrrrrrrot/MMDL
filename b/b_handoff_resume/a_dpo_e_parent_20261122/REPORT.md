# A DPO(e) parent: RTX PRO 6000 v1.2 screen

This is B's **one-seed screening result**, measured before adding GRPO. The
official team result remains the 4090 three-seed evaluation.

| Model | Seed 3407 | Difference on this 6000 |
| --- | ---: | ---: |
| Pinned Qwen3-VL-4B-Instruct base | 558/900 (62.00%) | reference |
| A DPO(e), merged from the original adapter | 562/900 (62.44%) | +4 questions (+0.44 percentage points) |

The 900 question IDs and document hashes match exactly. Paired changes are
72 wrong-to-right and 68 right-to-wrong; binary correctness agrees on 760/900
and extracted answers on 665/900. Missing final `Answer:` falls from 103 to
43. Neither samples file records generation finish reasons, so this result
does not establish how many responses reached the generation limit.

The parent is the fixed base revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17` merged with A's original
DPO(e) PEFT adapter, whose weights SHA256 is
`7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa`.
The merged model manifest and saved output hashes were verified on the 6000.
Evaluation used repository commit `61025850563d0331e365bc974a63c0d3c4ed4c2c`,
the v1.2 script, seed 3407, no item limit, and exited 0. The original base
run recorded a dirty worktree at evaluation time; the paired comparison still
confirms the same question inputs and document hashes.

A reported 593/900 for the original DPO(e) on the 4090 at seed 3407. The
31-question cross-GPU gap is unresolved. Public A/B evaluation script, task
template and scorer blobs agree, and the transferred adapter SHA agrees; A's
original 4090 per-item output, run metadata and merged-model weights hashes
are needed to distinguish generation variation from model-merging differences.
**B's subsequent `(e) → GRPO` delta must be calculated against 562/900 on
this same 6000**, while the final team comparison uses the 4090 official run.

Evidence: [paired summary](comparison_summary.json),
[item-level comparison](paired_items.csv), [candidate handoff](handoff.json),
[request manifest](candidate_manifest.json), [run metadata](run_meta.txt).
