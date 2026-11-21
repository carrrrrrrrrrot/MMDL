"""Export nonempty, language-only PEFT LoRA from pinned EasyR1 FSDP state."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
SOURCE_SHA = "525fdef04dcc8aba2973407428dc6f600f064187625a892b7d39f205db199179"


def patch(root: Path) -> Path:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    path = root / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("EasyR1 checkpoint manager differs from pinned source")
    source = raw.decode("utf-8")
    old = "import os\n"
    new = "import os\nimport re\n"
    if source.count(old) != 1:
        raise ValueError("Expected exactly one import")
    source = source.replace(old, new)
    old = (
        "            sharded_lora_weights = get_peft_model_state_dict(\n"
        "                self.model._fsdp_wrapped_module, state_dict=model_state_dict\n"
        "            )\n"
    )
    new = (
        "            # PEFT 0.21 structural extraction can return empty for nested FSDP wrappers.\n"
        "            # This Stage1 model is fixed to Qwen3-VL-4B language q/k/v/o LoRA.\n"
        "            pattern = re.compile(\n"
        "                r\"base_model\\.model\\.model\\.language_model\\.layers\\.\\d+\\.\"\n"
        "                r\"self_attn\\.(?:q|k|v|o)_proj\\.lora_[AB]\\.default\\.weight\"\n"
        "            )\n"
        "            sharded_lora_weights = {\n"
        "                name.replace(\".default.weight\", \".weight\"): tensor\n"
        "                for name, tensor in model_state_dict.items() if pattern.fullmatch(name)\n"
        "            }\n"
        "            if len(sharded_lora_weights) != 288:\n"
        "                raise RuntimeError(f\"Expected 288 language LoRA tensors, got {len(sharded_lora_weights)}\")\n"
    )
    if source.count(old) != 1:
        raise ValueError("Expected exactly one PEFT state extraction block")
    source = source.replace(old, new)
    old = "            torch.cuda.empty_cache()\n            if self.rank == 0:\n                save_file(lora_weights,"
    new = (
        "            torch.cuda.empty_cache()\n"
        "            lora_numel = sum(tensor.numel() for tensor in lora_weights.values())\n"
        "            if len(lora_weights) != 288 or lora_numel != 11796480:\n"
        "                raise RuntimeError(f\"Invalid LoRA export: {len(lora_weights)} tensors, {lora_numel} elements\")\n"
        "            if self.rank == 0:\n"
        "                save_file(lora_weights,"
    )
    if source.count(old) != 1:
        raise ValueError("Expected exactly one LoRA save block")
    source = source.replace(old, new)
    path.write_text(source, encoding="utf-8", newline="")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    print(patch(parser.parse_args().easyr1_root.resolve(strict=True)))
