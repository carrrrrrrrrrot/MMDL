# ECD500 vs ViRL500: fixed300 diagnostic

RTX PRO 6000, seed 3407, same 300 MMMU IDs and same base 185/300. ViRL data awaits C's OCR/semantic/manual review.

| Training data / algorithm | Correct / 300 | Accuracy | vs base | Corrections vs base | Regressions vs base | Missing final answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 185 | 61.67% | — | — | — | — |
| ECD500 GRPO | 191 | 63.67% | +2.00%p | 22 | 16 | 38 |
| ECD500 DAPO core | 194 | 64.67% | +3.00%p | 23 | 14 | 30 |
| ViRL500 GRPO | 192 | 64.00% | +2.33%p | 20 | 13 | 33 |
| ViRL500 DAPO core | 187 | 62.33% | +0.67%p | 20 | 18 | 34 |

| Same algorithm, ECD → ViRL | New corrections | New regressions | Net |
| --- | ---: | ---: | ---: |
| GRPO | 19 | 18 | +1 / 300 |
| DAPO_CORE | 16 | 23 | -7 / 300 |

All four files passed 300-ID, seed, baseline, scorer, input/document, summary/CSV count, and SHA stability checks. Stage2 hashes were checked against pinned values. Stage4 hashes are listed in the JSON; an independent Stage4 hash lock was not supplied.

These are one-seed, 300-question candidate screens. Differences are paired observations; they do not prove which training data caused them or predict a 900-question result.
