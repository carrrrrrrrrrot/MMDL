# Stage 2 ECD500 completion

Completed 2026-11-21 04:50 KST on the RTX PRO 6000. The automated queue exited with `phase=finished`, `state=complete`. Both candidates began at the pinned Qwen3-VL-4B-Instruct base revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, used the same frozen ECD500 input, v1.2 binary reward, seed 3407, constant LR 1e-6, five fresh responses per question, 1,024-token training response cap, zero KL and zero length penalty. Each saved a reload-verified language-attention LoRA adapter after 15 optimizer updates.

| Method | Used unique ECD IDs | Generated / used rollouts | Training time | Fixed300 correct | Base delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original base | — | — | — | 185/300 (61.67%) | — |
| GRPO | 500/500 | 7,500 / 7,500 | 1 h 2 min 35 s | 191/300 (63.67%) | +6 (+2.00 pp) |
| DAPO core | 115/500 | 52,750 / 7,500 | 2 h 45 min 33 s | 194/300 (64.67%) | +9 (+3.00 pp) |

The fixed300 scores use the same MMMU IDs, seed, inputs, document hashes, v1.2 scorer, and 6000 base run; the paired comparison script verified all 300 rows. GRPO repaired 22 base errors and lost 16 base correct answers. DAPO core repaired 23 and lost 14. GRPO alone answered 18 questions correctly; DAPO alone answered 21. Missing final `Answer:` lines: base 36, GRPO 38, DAPO core 30. Finish-reason metadata is unavailable, so these cannot be called confirmed length truncations.

This is one seed on a fixed 300-item screening subset, not an official MMMU score or a statistically established algorithm win. DAPO core has online filtering, asymmetric clipping, and token-level loss, but its soft-overlong penalty was disabled to preserve the team's binary reward. Dynamic filtering admitted only 115 of the 500 training IDs and made training 2.65 times longer than GRPO; the result compares two *procedures*, not isolated loss functions with equal ID exposure. Both candidates use the base parent; they do not constitute the team's shared-SFT-parent final comparison.

Evidence: [paired report](STAGE2_FIXED300_REPORT.md), [paired JSON](stage2_fixed300_paired.json), [queue completion](queue_status_final.json), [GRPO training audit](GRPO_FULL_AUDIT.md), [DAPO training audit](DAPO_FULL_AUDIT.md), and each candidate's `comparison_summary.json` plus `paired_items.csv` in `grpo_fixed300/` and `dapo_fixed300/`.

Stage 3 ViRL39K metadata/images and preliminary 500-ID plus 1,000-ID reserve packets are under `/data/chaewon/mmdl_stage3_virl39k_20261121` and documented in `../../b_stage3_virl39k/README.md`. Those IDs await C's leakage and answer-quality approval before training.
