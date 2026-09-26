# baseline_v0: lmms-eval 기본 설정 run (수정 전 대조군)

평가 파이프라인을 수정하기 전, lmms-eval의 기본 MMMU task로 실행한 결과입니다.
새 파이프라인의 회귀 테스트와 격차 분석에서 대조군으로 사용합니다.

| 파일 | 내용 |
|---|---|
| `20260920_011318_*` | run #2: task `mmmu_val`, sdpa, 900문항, acc 0.51111 |
| `20260921_153124_*` | run #4: task `mmmu_val_local`, flash_attention_2, 900문항, acc 0.51222 |
| `run4_lmms_eval_head.txt` | run #4 당시 lmms-eval HEAD 커밋 (`557ade04…`) |
| `run4_uncommitted_qwen3vl.sh.diff` | run #4는 위 커밋에 이 diff(커밋되지 않은 수정)를 적용한 스크립트로 실행됨 (sdpa→FA2, interleave_visuals False→True, conda env temp→vlm) |
| `env_vlm_pip_freeze_before_M0.txt` | 2026-09-24 기준 conda env `vlm`의 `pip freeze` (editable 설치를 MMDL로 옮기기 직전) |

- 두 run은 공통으로 greedy, `max_new_tokens=128`, 로컬 parquet(`/mnt/SSD-2TB/models/MMMU/*/validation-*.parquet`, HF revision `98e6ac0c…`)로 실행했습니다.
- 원본 위치: `/home/dekim/workspace/lmms-eval/logs/qwen3vl_mmmu/snapshots__ebb281ec70b05090aa6165b016eac8ec08e71b17/`
