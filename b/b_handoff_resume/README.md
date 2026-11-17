# B 공통 부모 GRPO 재개 게이트

`resume.py`는 A의 선정된 SFT 어댑터와 C의 승인 데이터가 모두 B 서버에 도착할 때까지 **대기**한다. 현재 `a/shared`에는 코드와 예시만 있고 이 입력은 없다. 합성 base→GRPO 파일럿을 공통 부모 실험으로 승격하지 않는다.

설정 파일은 [config.example.json](config.example.json)을 복사해 **B 서버의 실제 경로**로 채운다. 특히 `sft_adapter`는 PEFT 디렉터리, `sft_adapter_sha256`은 `adapter_model.safetensors`의 A 제공 SHA256이다. `a_converted_jsonl`은 A의 `team_share/convert_v12.py` 출력, `c_approved_ids`는 C의 ID 목록, `image_root`는 그 이미지가 실제로 있는 B 서버 경로다. A의 어댑터나 C 데이터가 아직 없으면 `null`로 두면 된다. `max_steps`와 `min_train_rows`는 A·B가 정한 학습량과 최소 승인 행 수를 적는다. 승인 행 수에 못 미치거나 변환 결과의 해시·프롬프트 렌더링 검사가 틀리면 학습을 시작하지 않는다.

자동 실행에는 C가 다음 형식의 **누수 검사 확인 파일**을 함께 전달해야 한다. `approved_ids_sha256`은 정확히 ID 목록 파일의 SHA256과 같아야 한다. 이 파일은 C 검수의 출처 기록이며 코드가 누수 검사를 대신 수행하지 않는다.

```json
{
  "status": "passed",
  "approved_ids_sha256": "<64-hex SHA256>",
  "dataset_revision": "<config의 dataset_revision>",
  "checks": {"MMMU": "passed", "MMMU-Pro": "passed"}
}
```

읽기 전용 확인:

```bash
PY=/data/chaewon/mmdl_envs/grpo_pilot/bin/python
RESUME=/data/chaewon/mmdl_b_impl/b/b_handoff_resume/resume.py
CONFIG=/data/chaewon/mmdl_handoff/common_parent_config.json
$PY $RESUME --config "$CONFIG" --mode check
```

입력이 모두 준비되면 CPU에서 C 승인 행만 변환하고 모든 선택 행의 Qwen 렌더링을 확인한다. 이어서 A의 **해시 고정된 병합 스크립트**로 SFT 부모를 병합하고, B 학습 `--validate-only`를 실행한다.

```bash
$PY $RESUME --config "$CONFIG" --mode prepare
```

검증 후 실제 GRPO를 시작하려면 `--mode train`을 쓴다. 대기 중 자료가 도착하는 즉시 검증·학습까지 자동 연결하려면 `--mode wait --auto-train`을 사용한다. `wait_for_queue_state`와 `wait_for_pilot_manifest`가 설정되어 있으면 이전 평가와 짧은 v1.2 GPU 파일럿이 끝나기 전에는 실제 학습을 시작하지 않는다. 두 작업이 끝나도 GPU 메모리 사용량이 4 GiB를 넘으면 계속 기다린다. 별도의 `--auto-train`이 없으면 대기 모드는 준비 단계까지만 수행한다.

```bash
$PY -u $RESUME --config "$CONFIG" --mode wait --auto-train --poll-seconds 300 \
  >> /data/chaewon/mmdl_handoff/common_parent_resume.log 2>&1
```

학습이 완료되면 A의 **동일한** 병합 스크립트로 이미 병합된 SFT 부모에 GRPO 어댑터를 한 번 더 병합한다. 최종 모델의 어댑터 순서와 SHA를 확인하고 6000에서 v1.2 20문항 스모크를 실행한다. 4090↔6000 기준선 재현 게이트가 통과한 경우에만 시드 3407의 900문항 선별 평가를 실행하고, Δ는 6000 기준선에 대해 기록한다. 6000 결과는 선별용이며 최종 공식 3시드 평가는 4090 담당이다.

동일 입력으로 완료된 실행은 다시 돌리지 않는다. 입력 해시가 바뀌면 새 `work_root`가 필요하다. 병합·학습·평가가 중단되거나 실패했을 때 부분 출력을 덮어쓰지 않고 상태 파일에 오류를 남긴다. `work_root/state.json`에서 `waiting`(표준 출력), `prepared`, `training`, `merging_final`, `evaluating_smoke`, `evaluating_screen`, `complete`, `failed` 상태를 볼 수 있다. 준비 단계는 합격 여부를 확인할 뿐, MMMU 성능을 의미하지 않는다.

이 스크립트가 확인하는 것은 파일 존재, A 어댑터 및 병합 코드 해시, 고정 base revision, v1.2 보상 모드, C 누수 검사 확인서와 ID 파일 해시다. 실데이터 학습 품질·누수 검사의 정확성은 별도 검토가 필요하다. `eval_run_prefix`는 평가 출력 이름에 사용하므로 새로운 공통 부모 실험마다 다르게 정한다.
