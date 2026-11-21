# ECD500 representative batch probes (2026-11-20/21 KST)

Both probes used the same Qwen3-VL-4B-Instruct revision, ECD500 training file,
v1.2 binary answer reward, seed 3407, five fresh responses per question,
1024-token cap, constant LR 1e-6, no KL penalty, and language-only LoRA. Each
probe made **one** policy update from 100 accepted questions (500 responses).
The five-item heldout set was only a pipeline check; these runs do not measure
MMMU improvement.

| Probe | Questions attempted / used | Generated / used responses | One update time | End-to-end time | Sampled GPU peak | Actor gradient | Answer present | Response cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO | 100 / 100 | 500 / 500 | 258.3 s | 338 s | 46,196 MiB | 0.008 | 0.882 | 0.118 |
| DAPO core | 750 / 100 | 3,750 / 500 | 694.9 s | 780 s | 49,818 MiB | 0.032 | 0.895 | 0.102 |

DAPO core retained groups with mixed raw v1.2 rewards. It needed fifteen
50-question draws to collect 100 accepted question visits (13.3% acceptance).
It used asymmetric clipping, token-level loss, and online filtering. The
soft-overlong penalty stayed disabled to preserve the agreed binary reward,
so this is **DAPO core**, rather than the full DAPO recipe. Its 750 attempted
visits covered all 500 unique training IDs, but only 76 unique IDs entered
the update. The final full-run audit will record this sampling imbalance.

Both probes exited successfully, saved a 288-tensor language-only PEFT adapter,
and passed exact adapter reload. The final adapter SHA256 values are
`e88d6acbc31253d9c6de2a5b5bc81e228411b5df759b7313b193349c22828afb`
(GRPO) and `629845fb2c010091fa4d3c7e1bc439468aaee0a4c3bb6f542296cd6f49129947`
(DAPO core). The sampled GPU memory peak is from `nvidia-smi` approximately
every five seconds, not a continuous maximum.

Source records: [GRPO completion](grpo_probe/completion_report.json),
[GRPO visits](grpo_probe/visit_audit.json),
[DAPO completion](dapo_probe/completion_report.json),
[DAPO visits](dapo_probe/visit_audit.json).
