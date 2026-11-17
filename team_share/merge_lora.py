#!/usr/bin/env python
"""LoRA 어댑터를 base 모델에 병합해 평가용 모델 폴더를 만든다.

평가 스크립트(scripts/run_mmmu_eval.sh)는 --model_path 만 받으므로, 학습한 LoRA 는 이 스크립트로
병합한 폴더를 넘겨 평가한다. A(4090)와 B(6000)가 같은 스크립트를 쓴다.

  python merge_lora.py --adapter <어댑터 폴더> --out <출력 폴더> [--base <base 모델 폴더>]

- base: 로컬 폴더 또는 HF repo id. 기본값은 환경 변수 MMDL_BASE_MODEL, 없으면 HF repo id + 고정 revision.
- 가중치는 base 와 같은 BF16 safetensors 로 저장한다.
- 가중치가 아닌 파일(config, tokenizer, chat_template, preprocessor, generation_config 등)은 base 의 것을
  그대로 복사한다 (save_pretrained 가 다시 쓴 파일도 base 원본으로 덮어씀 → 평가 설정이 base 와 같게 유지).
- out/MERGE_INFO.json 에 base 경로·revision, 어댑터 SHA256, 생성 시각, 라이브러리 버전을 남긴다.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys

DEFAULT_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
WEIGHT_FILE = re.compile(r"(\.safetensors|\.bin|\.safetensors\.index\.json|\.bin\.index\.json)$")


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def resolve_base(base, revision):
    """로컬 폴더면 그대로, 아니면 HF 캐시에서 해당 revision 스냅샷 경로를 받는다."""
    if os.path.isdir(base):
        m = re.search(r"/snapshots/([0-9a-f]{40})/?$", os.path.realpath(base))
        return os.path.realpath(base), (m.group(1) if m else revision)
    from huggingface_hub import snapshot_download

    path = snapshot_download(base, revision=revision)
    return path, revision


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("MMDL_BASE_MODEL", DEFAULT_REPO),
                    help="base 모델 폴더 또는 HF repo id (기본: $MMDL_BASE_MODEL 또는 %(default)s)")
    ap.add_argument("--base_revision", default=DEFAULT_REVISION,
                    help="base 가 repo id 이거나 경로에서 revision 을 알 수 없을 때 쓰는 revision")
    ap.add_argument("--adapter", required=True, help="PEFT save_pretrained 로 저장한 어댑터 폴더")
    ap.add_argument("--out", required=True, help="병합 모델을 저장할 폴더 (없거나 비어 있어야 함)")
    ap.add_argument("--allow_revision_mismatch", action="store_true",
                    help="어댑터에 기록된 base revision 과 다를 때도 진행")
    args = ap.parse_args()

    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"출력 폴더가 비어 있지 않습니다: {args.out}")
    adapter_weights = os.path.join(args.adapter, "adapter_model.safetensors")
    adapter_cfg_path = os.path.join(args.adapter, "adapter_config.json")
    for p in (adapter_weights, adapter_cfg_path):
        if not os.path.isfile(p):
            sys.exit(f"어댑터 파일이 없습니다: {p}")

    base_path, base_rev = resolve_base(args.base, args.base_revision)
    adapter_cfg = json.load(open(adapter_cfg_path))
    adapter_rev = adapter_cfg.get("revision")
    if adapter_rev and base_rev and adapter_rev != base_rev and not args.allow_revision_mismatch:
        sys.exit(f"어댑터 base revision({adapter_rev}) 과 base revision({base_rev}) 이 다릅니다")

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    print(f"[merge] base={base_path} (revision {base_rev})", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cpu")
    n_base = sum(p.numel() for p in model.parameters())
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"[merge] adapter={args.adapter}", flush=True)
    model = model.merge_and_unload()
    n_merged = sum(p.numel() for p in model.parameters())
    if n_base != n_merged:
        sys.exit(f"병합 후 파라미터 수가 다릅니다: base {n_base} vs merged {n_merged}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")

    # 가중치가 아닌 파일은 base 원본으로 복사 (save_pretrained 가 쓴 config/generation_config 도 덮어씀)
    copied = []
    for name in sorted(os.listdir(base_path)):
        src = os.path.join(base_path, name)
        if os.path.isfile(src) and not WEIGHT_FILE.search(name):
            shutil.copyfile(src, os.path.join(args.out, name))
            copied.append(name)

    info = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_path": base_path,
        "base_revision": base_rev,
        "adapter_path": os.path.realpath(args.adapter),
        "adapter_sha256": {
            "adapter_model.safetensors": sha256(adapter_weights),
            "adapter_config.json": sha256(adapter_cfg_path),
        },
        "adapter_config": {k: adapter_cfg.get(k) for k in
                           ("r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path", "revision")},
        "dtype": "bfloat16",
        "n_params": n_merged,
        "copied_from_base": copied,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(args.out, "MERGE_INFO.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"[merge] 완료: {args.out}", flush=True)


if __name__ == "__main__":
    main()
