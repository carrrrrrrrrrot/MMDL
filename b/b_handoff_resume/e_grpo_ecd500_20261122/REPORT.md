# A DPO(e) → B ECD500 GRPO on the RTX PRO 6000

This branch measures whether GRPO improves **A's existing DPO(e) model**.
DPO training was not repeated. B used the merged DPO(e) model as the GRPO
actor parent, trained a new LoRA, then evaluated an ordered
`base → DPO(e) → GRPO` merge. These are B-side 6000 screening results; a
4090 three-seed run is needed for an official team result.

## Training

| Setting | Value |
| --- | --- |
| Parent | A original DPO(e) adapter SHA `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa` on pinned Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Data | B's ECD QC-v2, 500 unique image questions; C's MMMU leakage audit passed, while full gold approval is not claimed. No question ID or image SHA overlap with A DPO(e)'s eight ECD questions. |
| Algorithm | EasyR1 GRPO, raw MMMU v1.2 binary answer reward, no KL or length penalty |
| Schedule | Three newly generated passes over the 500 questions, five rollouts per question, 100 question groups per optimizer update, 15 updates total, constant LR `1e-6`, response cap 1,024 tokens |
| Completed run | 15/15 positive-gradient updates, 7,500 generated/used rollouts, each of 500 IDs used three times, exit 0, exact PEFT adapter reload |
| Training diagnostics | Mean reward 0.3015, `Answer:` present 0.9232, cap proxy 0.0775 across 15 updates; 97-question heldout reward accuracy 0.309 |
| Cost | 3,758 seconds (62 min 38 s), sampled GPU peak 47,416 MiB |

The resulting GRPO adapter SHA256 is
`7a28ca5b29f201c33c79800d9f96d0673414590c2b60ac6c940cc00a52f9e69a`.
The final merged model manifest records exactly `[dpo, grpo]` in that order
with lineage SHA256
`6964fe40211c55012a904306eb55f2115d43fc9cf8cde0f45ac0a79437031acc`.
The completed training report selected the **console** as its metric source;
its Step blocks were forwarded at run exit. A SHA-checked Ray stdout fallback
was tested for partial console forwarding but was not used to complete this
run.

## Evaluation

The v1.2 20-question smoke ended normally with exact input and document hash
agreement: DPO(e) parent 15/20, sequential candidate 14/20. This smoke is a
pipeline check, not a performance estimate.

The matched seed-3407 900-question v1.2 evaluation completed with exit 0:

| Model on this 6000 | Correct / 900 | Score | Difference from DPO(e) parent |
| --- | ---: | ---: | ---: |
| Pinned original base | 558 | 62.00% | reference only |
| A DPO(e) parent | 562 | 62.44% | reference |
| A DPO(e) → B ECD500 GRPO | **574** | **63.78%** | **+12 questions (+1.33 percentage points)** |

The parent and sequential candidate use the same 6000, seed 3407,
evaluation repository commit and 900 question IDs. Input and document-hash
agreement are **900/900**. GRPO turns 71 parent errors into correct answers
and 59 parent correct answers into errors. Binary correctness agrees on
770/900 (85.56%); extracted answers agree on 695/900 (77.22%). Missing final
`Answer:` falls from 43 to 36. Neither saved samples file records a
generation finish reason, so these files cannot establish whether a response
hit the generation limit.

The +12 paired result is promising **one-seed candidate screening**, not a
three-seed performance estimate. The 4090 original DPO(e) result differs
materially from this 6000 control; [the parent report](../a_dpo_e_parent_20261122/REPORT.md)
records the unresolved cross-GPU comparison. A 4090 v1.2 three-seed run of
the ordered candidate is needed before making an official team claim.

Evidence: [probe completion](probe/completion_report.json),
[full completion](full/completion_report.json), [per-update metrics](full/metric_summary.json),
[visit audit](full/visit_audit.json), [adapter reload](full/adapter_reload.json),
[training lineage](training_lineage.json), [merged model manifest](merged_model_manifest.json),
[20-item smoke](smoke20/comparison_summary.json),
[900-item paired summary](screen900/comparison_summary.json),
[900-item question comparison](screen900/paired_items.csv),
[900-item handoff](screen900/handoff.json), and
[900-item run metadata](screen900/run_meta.txt).
