"""Fail closed unless Stage1 PEFT trains language q/k/v/o LoRA only."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
MAIN_SHA = "9921d5affab6fb905e682ae44a4b790f1c8c7db4bc33f02def769d1d4b06e707"
WORKER_SHA = "4ce08f3214b9eeaf429c789f76ea2560156cbad003ce465e2eaf48233a99bad6"


def patch(root: Path) -> tuple[Path, Path]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    main = root / "verl/trainer/main.py"
    worker = root / "verl/workers/fsdp_workers.py"
    main_raw = main.read_bytes()
    worker_raw = worker.read_bytes()
    if hashlib.sha256(main_raw).hexdigest() != MAIN_SHA:
        raise ValueError("Expected Stage1 Ray environment patch in main.py")
    if hashlib.sha256(worker_raw).hexdigest() != WORKER_SHA:
        raise ValueError("Expected pinned fsdp_workers.py")

    main_source = main_raw.decode("utf-8")
    old = '        for key in ("MMDL_V12_IMAGE_PARITY", "MMDL_VISIT_LOG_PATH"):\n'
    new = '        for key in ("MMDL_V12_IMAGE_PARITY", "MMDL_VISIT_LOG_PATH", "MMDL_PARAM_AUDIT_PATH"):\n'
    if main_source.count(old) != 1:
        raise ValueError("Expected exactly one Stage1 environment key tuple")
    main_source = main_source.replace(old, new)

    worker_source = worker_raw.decode("utf-8")
    old = "from contextlib import nullcontext\n"
    new = "import json\nimport os\nfrom contextlib import nullcontext\n"
    if worker_source.count(old) != 1:
        raise ValueError("Expected exactly one worker import block")
    worker_source = worker_source.replace(old, new)
    old = "            model = get_peft_model(model, lora_config)\n"
    new = (
        "            model = get_peft_model(model, lora_config)\n"
        '            audit_path = os.environ.get("MMDL_PARAM_AUDIT_PATH")\n'
        '            if audit_path:\n'
        '                trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]\n'
        '                allowed = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")\n'
        '                bad = [name for name, _ in trainable\n'
        '                       if "lora_" not in name or not any(x in name for x in allowed)\n'
        '                       or any(x in name for x in ("visual", "merger", "projector", "connector"))]\n'
        '                if not trainable or bad:\n'
        '                    raise RuntimeError(f"MMDL trainable scope invalid: count={len(trainable)} bad={bad[:8]}")\n'
        '                payload = {"num_trainable_tensors": len(trainable),\n'
        '                           "num_trainable_parameters": sum(n for _, n in trainable),\n'
        '                           "names_and_numel": trainable}\n'
        '                with open(audit_path, "w", encoding="utf-8") as stream:\n'
        '                    json.dump(payload, stream, ensure_ascii=False, indent=2)\n'
        '                self.print_rank0(f"MMDL trainable scope: {len(trainable)} tensors, "\n'
        '                                 f"{payload[\'num_trainable_parameters\']} parameters, no vision/connector")\n'
    )
    if worker_source.count(old) != 1:
        raise ValueError("Expected exactly one PEFT initialization")
    worker_source = worker_source.replace(old, new)

    # Both source hashes are verified before either file is changed.
    main.write_text(main_source, encoding="utf-8", newline="")
    worker.write_text(worker_source, encoding="utf-8", newline="")
    return main, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    print(*patch(parser.parse_args().easyr1_root.resolve(strict=True)), sep="\n")
