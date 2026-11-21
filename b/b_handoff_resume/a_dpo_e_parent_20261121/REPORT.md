# A DPO(e) 어댑터를 B 6000의 GRPO 부모로 병합

2026-11-21 KST. A의 [전달 커밋 `4e6df41`](https://github.com/carrrrrrrrrrot/MMDL/tree/4e6df41da5921d429e01e011e1a6ef368360edf4/dpo_e_trunc_for_b)에서 (e) 원 실행 PEFT 어댑터와 학습 ID를 받았다. 원본 모델 revision은 `ebb281ec70b05090aa6165b016eac8ec08e71b17`이다. A의 DPO를 다시 학습하지 않았다.

## 데이터와 전달 검사

- A DPO(e): MAmmoTH-VL 161문항 + ECD 8문항, 고유 169문항에서 만든 선호쌍 305개. chosen은 원본 모델의 정답 응답, rejected는 4,096토큰에서 잘린 응답이다. A의 DPO 설정은 β 0.1, 학습률 `5e-5`, 2 epochs, 78 optimizer steps다.
- 전달받은 `adapter_model.safetensors`는 47,228,944바이트이고 SHA256은 `7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa`다. Git 파일 자체, 로컬 추출본, 6000 복사본에서 모두 일치했다. `adapter_config.json`, `train_ids_169.jsonl`, `ecd_8.jsonl`도 A의 `SHA256SUMS.txt` 검사를 통과했다.
- B의 이후 GRPO 데이터 후보는 이미 6000에 있는 ECD QC-v2 **500문항**이다. A의 ECD 8문항과 B ECD500/798 원본 목록의 문항 ID 및 이미지 SHA256 중복은 모두 0건이었다.

## 병합 결과

- 6000 원본 전달 폴더: `/data/chaewon/mmdl_a_e_handoff_20261121`
- 고정 base: `/home/chaewon/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17`
- 병합 모델: `/data/chaewon/mmdl_merged_models/dpo_e_trunc_from_a_20261121`
- `b_eval_ops/merge_lora.py`의 SHA·언어 attention LoRA 대상 사전 검사와 CPU 로드·병합 검사 통과. 원본 어댑터 config에 A 로컬 base 경로가 적혀 있지만, base 모델 객체를 명시해 정상 로드되어 config를 수정하지 않았다.
- 병합 모델은 bf16 `merge_and_unload(safe_merge=True)`로 저장했다. [model_manifest.json](model_manifest.json)은 `complete`, 계보 `base→dpo`, base revision·어댑터 SHA 일치, 출력 15개 파일 약 8.89GB의 해시를 기록한다. `_INCOMPLETE` 표시는 없다.
- 6000 v1.2 `--limit 20`, 시드 3407 스모크는 20개 샘플 전부 생성, 종료 상태 0, **14/20**이다. [handoff](smoke_handoff.json)에 실행 메타데이터 경로·해시가 있고 [결과 JSON](smoke_results.json)을 함께 보관했다. 20문항 점수는 성능 판정에 사용하지 않는다. vLLM 종료 시 엔진 프로세스 종료 오류 문구가 출력됐지만 점수·샘플 파일 생성 후였고 실행 상태는 0이었다. 다음 전체 평가에서도 종료 상태와 파일 개수를 확인한다.

## 이후 순서

1. 6000에서 병합된 **(e) 단독**을 v1.2·시드 3407·900문항으로 평가한다. 이는 같은 GPU에서 `(e)→GRPO`를 비교할 부모 기준선이다.
2. (e) 병합 모델에 **새 GRPO LoRA**를 붙인다. 기존 GRPO 어댑터를 재사용하거나 DPO를 다시 학습하지 않는다. ECD QC-v2 500문항의 1-step probe에서 보상 분산, 기울기, `Answer:` 존재 및 1,024토큰 상한 도달을 검사한다.
3. probe가 통과하면 기존 EasyR1 GRPO의 시작 설정(질문당 5회 생성, 3회 새 응답 순회, 학습률 `1e-6`, 100질문당 갱신 1회, 총 15회)을 적용한다. 1,024토큰 상한은 probe의 잘림 양상을 보고 확정한다.
4. `(e)→GRPO`를 같은 6000·시드·900문항으로 평가해 (e) 단독과 문항별 정오 및 최종답 누락을 대조한다. 개선이 확인될 때만 추가 시드와 4090 공식 평가를 고려한다.

현재 B의 EasyR1 사전 검사·병합 후보 평가 코드는 원본 또는 SFT 부모만 검증하므로, GRPO 학습 전에 `base→dpo→grpo` 계보를 별도 지원해야 한다. 이 실험은 **DPO 후 GRPO 순차 학습**이며 base 출발 DPO/GRPO 방법 비교와 구분한다.
